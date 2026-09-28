"""TRAPs 命令列執行、流式 catalog 與可回溯成果輸出。

命令列只接受呼叫端明確提供的 ``--month-dir`` 或 ``--surface-root``；不把本機、
SERVER 或下載目錄寫死在程式裡。``detect`` 逐時呼叫 OCM 的 memory-map snapshot，
因此不會把整月速度讀進記憶體。每一個 sensitivity case 都有自己的 catalog、站點
表、GeoJSON、圖面和 manifest，manifest 會保存輸入 metadata/config 的 SHA-256、
實際門檻、有效分母與輸出檔雜湊。

逐時檢出完成後，CLI 以 ``TrackingFrame`` 串接 Eulerian 結構，再由
``compute_site_metrics`` 計算站點分子/分母、強度、面積密度及資料缺口；單一時刻
執行仍會形成一筆 right-censored track，並在 manifest 標示追蹤邊界。跨 sensitivity
參數的 robust 判定尚未自動執行，所有 case 固定標示 ``not_evaluated``。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields as dataclass_fields
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .detection import detect_snapshot
from .metrics import SiteMetricsConfig, compute_site_metrics, curve_intersects_mask
from .models import DetectionConfig, TrapDetection
from .ocm import SurfaceMonth, SurfaceSnapshot, open_month
from .sites import DEFAULT_STUDY_DESIGN_PATH, load_study_design, site_masks
from .tracking import TrackingConfig, TrackingFrame, track_structures

CLI_SCHEMA_VERSION = "1.0.0"
"""CLI 產物 schema；修改 JSON/CSV 欄位時需同步提高此版本。"""

DEFAULT_COASTLINE_GEOJSON_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "taiwan_exact_coastline"
    / "taiwan_exact_coastline.geojson"
)
"""本專案預設的 CRS84 海岸線來源；只供成果圖 visual-only 疊線使用。"""

COASTLINE_ATTRIBUTION = "© OpenStreetMap contributors; Open Database License (ODbL)"
"""單張圖與 JSON provenance 都必須保留的 OSM/ODbL 來源授權說明。"""

DEFAULT_SENSITIVITY_THRESHOLDS = (-1.0e-8, -5.0e-8)
"""研究計畫要求的兩組預設核心 s1 負門檻，執行時會完整寫入 manifest。"""

SITE_STAT_FIELDS = (
    "site_id",
    "study_site_name_zh",
    "state_label",
    "coverage_threshold",
    "valid_snapshot_count",
    "occurrence_snapshot_count",
    "occurrence_numerator",
    "occurrence_denominator",
    "occurrence_rate",
    "site_area_m2",
    "site_area_km2",
    "core_count",
    "curve_count",
    "total_curve_length_m",
    "mean_curve_length_m",
    "median_curve_length_m",
    "mean_core_s1_sinv",
    "mean_curve_s1_sinv",
    "mean_abs_curve_s1_sinv",
    "mean_abs_s1_sinv",
    "core_density_per_km2",
    "curve_density_per_km2",
    "curve_length_density_m_per_km2",
    "core_density_per_km2_per_valid_snapshot",
    "curve_density_per_km2_per_valid_snapshot",
    "curve_length_density_m_per_km2_per_valid_snapshot",
    "coverage_mean",
    "coverage_min",
    "coverage_max",
    "data_gap_frame_count",
    "coverage_unavailable_frame_count",
    "coverage_below_threshold_frame_count",
    "no_detection_frame_count",
    "track_count",
    "complete_track_count",
    "censored_track_count",
    "mean_observed_track_span_hours",
    "mean_complete_track_duration_hours",
    "lifetime_status",
    "quality_flags",
)


def _sha256_file(path: Path) -> str:
    """以分塊方式計算檔案 SHA-256，避免大圖或 GeoJSON 一次進入記憶體。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> Path:
    """以 UTF-8、穩定縮排寫出可供人工與程式追溯的 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _require_empty_output(path: Path) -> None:
    """要求新的或空的成果目錄，防止重跑混入上一批檔案與過期雜湊。

    每次執行應使用獨立 run ID；對已有內容的目錄封閉失敗，使 manifest 的檔案
    清單只對應本次輸入與參數。此檢查不會清除使用者既有成果。
    """

    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"輸出目錄已有內容，請改用新的 --output-root: {path}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    """以固定欄位順序寫 CSV；沒有資料時仍保留欄名，避免空檔無法解析。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    return path


def _csv_value(value: Any) -> Any:
    """將 ndarray、集合和缺值轉成 CSV 能穩定表示的純量或 JSON 字串。"""

    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _time_iso(time_utc_ns: int) -> str:
    """把 UTC nanosecond 整數轉成 ISO-8601 字串，保留 UTC 來源語意。"""

    timestamp = _datetime.datetime.fromtimestamp(int(time_utc_ns) / 1.0e9, tz=_datetime.UTC)
    return timestamp.isoformat(timespec="seconds")


def _resolve_month_dir(surface_root: str | Path, domain_id: str | None, month: str | None) -> Path:
    """由顯式 surface root、domain 和月份解析唯一 OCM 月份目錄。

    不自動挑選多個 domain 或月份；若呼叫端沒有提供足夠條件，函式會列出需補的
    參數並停止，避免把不同流場區域或月份混合到同一項研究成果。
    """

    root = Path(surface_root)
    if not root.is_dir():
        raise ValueError(f"surface root 不存在或不是目錄: {root}")
    domain_dirs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "months").is_dir())
    if domain_id is not None:
        domain_dirs = [path for path in domain_dirs if path.name == domain_id]
    if len(domain_dirs) != 1:
        names = [path.name for path in domain_dirs]
        raise ValueError(f"surface root 無法解析唯一 domain，請指定 --domain-id；候選={names}")
    months_dir = domain_dirs[0] / "months"
    month_dirs = sorted(path for path in months_dir.iterdir() if path.is_dir() and path.name.isdigit() and len(path.name) == 6)
    if month is not None:
        month_dirs = [path for path in month_dirs if path.name == month]
    if len(month_dirs) != 1:
        names = [path.name for path in month_dirs]
        raise ValueError(f"surface root 無法解析唯一月份，請指定 --month；候選={names}")
    return month_dirs[0]


def _open_month_from_args(args: argparse.Namespace) -> SurfaceMonth:
    """依 CLI 輸入開啟 OCM 月份，並核對本研究鎖定的 flow-domain bbox。"""

    month_dir = args.month_dir
    if month_dir and args.surface_root:
        raise ValueError("--month-dir 與 --surface-root 只能擇一")
    if month_dir is None and args.surface_root is None:
        raise ValueError("必須提供 --month-dir 或 --surface-root")
    if month_dir is None:
        month_dir = _resolve_month_dir(args.surface_root, args.domain_id, args.month)
    opened = open_month(
        month_dir,
        allow_trial_pilot=bool(args.allow_trial_pilot),
        expected_domain_id=args.domain_id,
        analysis_ocean_mask_path=args.ocean_mask,
    )
    design = load_study_design(Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH))
    _validate_month_against_study_design(opened, design)
    return opened


def _validate_month_against_study_design(month: SurfaceMonth, design: Mapping[str, Any]) -> None:
    """核對月份 domain 與固定研究設計的四區 bbox。

    ``open_month`` 只驗證 OCM 自己的 metadata consistency；本研究還需要確認資料確實
    落在已核定的四個 flow-domain 邊界。此 helper 同時供既有 preflight/detect/render
    與 literature-figures 重開 catalog source 使用，避免不同命令各自實作略有差異的
    domain gate。
    """

    domain = next((item for item in design["flow_domains"] if item["flow_domain_id"] == month.domain_id), None)
    if domain is None:
        raise ValueError(f"月份 domain_id 不在四區研究設計內: {month.domain_id}")
    upstream_domain = month.metadata["domain"]
    for field, expected in zip(("lon_min", "lon_max", "lat_min", "lat_max"), domain["bbox_lon_lat"], strict=True):
        actual = upstream_domain.get(field)
        if actual is None or not np.isfinite(float(actual)) or abs(float(actual) - float(expected)) > 1.0e-6:
            raise ValueError(f"OCM {field} 與既定 flow domain 邊界不符: {actual!r} != {expected!r}")


def _resolve_coastline_geojson(args: argparse.Namespace) -> tuple[Any | None, dict[str, Any]]:
    """解析預設或顯式覆寫的 GeoJSON，回傳 geometry 與可序列化 provenance。

    預設檔不存在時屬於可追蹤的 ``DEFAULT_MISSING`` 狀態：CLI 繼續產生不含
    海岸線的模型／統計成果，但 JSON 明確記錄未疊圖。使用者若以
    ``--coastline-geojson`` 指定檔案，則檔案不存在、CRS 非 CRS84、幾何不完整或
    其他 loader gate 失敗都會停止命令；不能把顯式來源錯誤降級為「無海岸線」。
    這個 helper 只建立圖層物件，不會讀取或改寫 OCM mask、flow domain、catalog
    數值或站點統計。
    """

    requested_value = getattr(args, "coastline_geojson", None)
    explicit = requested_value is not None
    requested_path = Path(requested_value).expanduser() if explicit else DEFAULT_COASTLINE_GEOJSON_PATH
    resolved_path = requested_path.resolve()
    base: dict[str, Any] = {
        "path": str(resolved_path),
        "requested_path": str(resolved_path),
        "source_mode": "explicit_override" if explicit else "default",
        "activation_status": "disabled",
        "enabled": False,
        "sha256": None,
        "crs": None,
        "feature_count": None,
        "ring_count": None,
        "point_count": None,
        "bounds": None,
        "visual_overlay_only": True,
        "source": "OpenStreetMap (OSMData land polygons)",
        "license": "ODbL",
        "attribution": COASTLINE_ATTRIBUTION,
        "source_authorization": "OpenStreetMap data used under ODbL; attribution retained in outputs",
        "source_semantics": "CRS84 vector coastline outline only; no bathymetry, tide, or ocean-mask semantics",
    }
    if not resolved_path.is_file():
        if explicit:
            raise ValueError(f"--coastline-geojson 指定檔案不存在: {resolved_path}")
        base["activation_status"] = "default_missing"
        base["status_note"] = "預設海岸線檔不存在，因此本次成果不疊加 coastline。"
        return None, base

    try:
        from .coastline import load_coastline

        coastline = load_coastline(resolved_path)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        source_label = "--coastline-geojson" if explicit else "預設海岸線"
        raise ValueError(f"{source_label} 無法通過 CRS84/幾何檢查: {resolved_path}: {error}") from error

    metadata = dict(coastline.to_metadata())
    metadata.update(
        {
            "requested_path": str(resolved_path),
            "source_mode": "explicit_override" if explicit else "default",
            "activation_status": "enabled",
            "enabled": True,
            "visual_overlay_only": True,
            "source": "OpenStreetMap (OSMData land polygons)",
            "license": "ODbL",
            "attribution": COASTLINE_ATTRIBUTION,
            "source_authorization": "OpenStreetMap data used under ODbL; attribution retained in outputs",
        }
    )
    return coastline, metadata


def _source_provenance(
    month: SurfaceMonth,
    study_design_path: Path,
    *,
    coastline_geojson: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """收集月份、grid、海岸 sidecar、圖面海岸線與研究設計的來源和 SHA-256。"""

    metadata_path = month.month_dir / "metadata.json"
    grid_metadata_path = month.month_dir.parent.parent / "grid" / "metadata.json"
    source = dict(month.provenance)
    arrays = month.metadata.get("arrays") or {}
    source.update(
        {
            "month_metadata_path": str(metadata_path),
            "month_metadata_sha256": _sha256_file(metadata_path),
            "grid_metadata_path": str(grid_metadata_path),
            "grid_metadata_sha256": _sha256_file(grid_metadata_path),
            "study_design_path": str(study_design_path),
            "study_design_sha256": _sha256_file(study_design_path),
            "velocity_unit_metadata": {
                component: (arrays.get(f"{component}_surface_mps.npy") or {}).get("unit")
                for component in ("u", "v")
            },
            "velocity_unit_status": "supplier_confirmation_not_provided_to_this_project",
        }
    )
    if coastline_geojson is not None:
        # 這個欄位只描述 PNG/地理圖的 vector overlay；analysis_ocean_mask 的
        # provenance 仍保留在 source.coastline_mask，兩者不能互相替代。
        source["coastline_geojson"] = dict(coastline_geojson)
    return source


def _science_status(month: SurfaceMonth) -> str:
    """區分上游 cache ready 與本研究尚未完成的速度單位確認。"""

    return "PILOT_ONLY" if month.pilot_only else "PROVISIONAL_UNITS_UNCONFIRMED"


def _selected_indices(month: SurfaceMonth, args: argparse.Namespace) -> list[int]:
    """將單時刻或半開區間參數轉成有界的 time index 清單。"""

    if args.snapshot_index is not None:
        indices = [int(args.snapshot_index)]
    else:
        start = 0 if args.start_index is None else int(args.start_index)
        stop = month.snapshot_count if args.stop_index is None else int(args.stop_index)
        if start < 0 or stop > month.snapshot_count or start >= stop:
            raise ValueError(f"snapshot index 區間不合法: start={start}, stop={stop}, count={month.snapshot_count}")
        indices = list(range(start, stop))
    if any(index < 0 or index >= month.snapshot_count for index in indices):
        raise ValueError(f"snapshot index 超出月份範圍 0..{month.snapshot_count - 1}: {indices}")
    return indices


def _case_values(args: argparse.Namespace) -> tuple[list[int], list[float]]:
    """解析 sensitivity 的尺度與兩組核心門檻，並拒絕空或重複設定。"""

    if args.smoothing_radius_cells:
        radii = [int(value) for value in args.smoothing_radius_cells]
    elif args.sensitivity:
        radii = [0, 1, 2, 3]
    else:
        radii = [0]
    if any(value not in (0, 1, 2, 3) for value in radii):
        raise ValueError("smoothing radius 只能是 0、1、2 或 3")
    if len(set(radii)) != len(radii):
        raise ValueError("smoothing radius 不可重複")

    if args.core_thresholds_sinv is not None:
        thresholds = [float(value) for value in args.core_thresholds_sinv]
    elif args.min_core_s1_sinv is not None:
        thresholds = [float(args.min_core_s1_sinv)]
    elif args.sensitivity:
        thresholds = list(DEFAULT_SENSITIVITY_THRESHOLDS)
    else:
        thresholds = [DetectionConfig().min_core_s1_sinv]
    if not thresholds or any(not np.isfinite(value) or value >= 0.0 for value in thresholds):
        raise ValueError("核心 s1 門檻必須是有限負值")
    if args.sensitivity and len(thresholds) != 2:
        raise ValueError("--sensitivity 必須有兩組核心 s1 門檻")
    return radii, thresholds


def _make_config(args: argparse.Namespace, radius: int, threshold: float) -> DetectionConfig:
    """將 CLI 門檻組成數值核心設定，確保每個 case 的參數可寫入 manifest。"""

    return DetectionConfig(
        min_core_s1_sinv=threshold,
        min_eigenvalue_gap_sinv=float(args.min_eigenvalue_gap_sinv),
        local_min_radius_cells=int(args.local_min_radius_cells),
        core_min_separation_m=float(args.core_min_separation_m),
        integration_step_m=args.integration_step_m,
        max_curve_length_m=float(args.max_curve_length_m),
        max_integration_steps=int(args.max_integration_steps),
        curve_s1_stop_sinv=float(args.curve_s1_stop_sinv),
        curve_min_relative_strength=float(args.curve_min_relative_strength),
        curve_monotonic_tolerance_sinv=float(args.curve_monotonic_tolerance_sinv),
        max_turn_angle_deg=float(args.max_turn_angle_deg),
        min_curve_points=int(args.min_curve_points),
        min_curve_length_m=float(args.min_curve_length_m),
        smoothing_radius_cells=radius,
    )


def _curve_intersects_site(
    detection: TrapDetection,
    snapshot: SurfaceSnapshot,
    site_mask: np.ndarray,
) -> bool:
    """以曲線採樣點最近格點判定是否交會站點統計遮罩。"""

    # 使用 metrics 的半格距加密規則，避免 CLI 自己只查端點而漏掉曲線中段交會。
    return bool(curve_intersects_mask(detection.curve_points_m, snapshot.x_m, snapshot.y_m, site_mask))


def _detection_record(
    detection: TrapDetection,
    *,
    snapshot: SurfaceSnapshot,
    month: SurfaceMonth,
    site_ids: Sequence[str],
    radius: int,
    threshold: float,
) -> dict[str, Any]:
    """將 TrapDetection 展平為 catalog JSON/CSV 可共同使用的欄位。"""

    return {
        "domain_id": month.domain_id,
        "month": month.month,
        "state_label": month.state_label,
        "snapshot_index": int(snapshot.index),
        "time_utc_ns": int(snapshot.time_utc_ns),
        "time_utc_iso": _time_iso(snapshot.time_utc_ns),
        "core_row": int(detection.core_index_ij[0]),
        "core_col": int(detection.core_index_ij[1]),
        "core_x_m": float(detection.core_xy_m[0]),
        "core_y_m": float(detection.core_xy_m[1]),
        "core_s1_sinv": float(detection.core_s1_sinv),
        "curve_length_m": float(detection.curve_length_m),
        "curve_mean_s1_sinv": float(detection.curve_mean_s1_sinv),
        "curve_mean_abs_s1_sinv": float(detection.curve_mean_abs_s1_sinv),
        "curve_points_m": detection.curve_points_m.tolist(),
        "curve_s1_sinv": detection.curve_s1_sinv.tolist(),
        "forward_stop_reason": detection.forward_stop_reason,
        "backward_stop_reason": detection.backward_stop_reason,
        "is_qualified": bool(detection.is_qualified),
        "qc_flags": list(detection.qc_flags),
        "study_site_ids": list(site_ids),
        "smoothing_radius_cells": int(radius),
        "min_core_s1_sinv": float(threshold),
    }


def _geojson(
    detections: Sequence[Mapping[str, Any]],
    *,
    snapshot: SurfaceSnapshot,
    state_label: str,
    source_label: str,
) -> dict[str, Any]:
    """把上游局部公尺座標轉成 WGS84 經緯度後建立 GeoJSON。

    OCM 的 x/y 與 lon/lat 是同一張規則格網的兩組軸，因此在各軸內插可保留
    檢出曲線的位置與來源格點對應。GeoJSON 的座標依 RFC 7946 必須是經緯度；
    原始公尺曲線仍留在 catalog JSON，不能把公尺數值直接冒充 GeoJSON 座標。
    """

    x_axis = np.asarray(snapshot.x_m)[0, :]
    y_axis = np.asarray(snapshot.y_m)[:, 0]
    lon_axis = np.asarray(snapshot.lon)
    lat_axis = np.asarray(snapshot.lat)
    if not (np.all(np.diff(x_axis) > 0) and np.all(np.diff(y_axis) > 0)):
        raise ValueError("GeoJSON 轉換需要遞增的 OCM 公尺座標軸")

    features: list[dict[str, Any]] = []
    for index, row in enumerate(detections):
        metric_points = np.asarray(row.get("curve_points_m") or [], dtype=np.float64)
        if metric_points.ndim != 2 or metric_points.shape[1] != 2:
            raise ValueError("TRAP 曲線須為 (point, xy_m) 才能輸出 GeoJSON")
        if np.any(metric_points[:, 0] < x_axis[0]) or np.any(metric_points[:, 0] > x_axis[-1]) or np.any(metric_points[:, 1] < y_axis[0]) or np.any(metric_points[:, 1] > y_axis[-1]):
            raise ValueError("TRAP 曲線超出 OCM 網格，拒絕將座標夾到 GeoJSON 邊界")
        points = np.column_stack(
            (
                np.interp(metric_points[:, 0], x_axis, lon_axis),
                np.interp(metric_points[:, 1], y_axis, lat_axis),
            )
        ).tolist()
        if len(points) == 1:
            points = [points[0], points[0]]
        features.append(
            {
                "type": "Feature",
                "id": f"trap-{index:06d}",
                "properties": {
                    "catalog_index": index,
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"curve_points_m", "curve_s1_sinv"}
                    },
                },
                "geometry": {"type": "LineString", "coordinates": points},
            }
        )
    return {
        "type": "FeatureCollection",
        "schema_version": CLI_SCHEMA_VERSION,
        "coordinate_units": "degrees",
        "coordinate_reference": "WGS84 longitude/latitude (RFC 7946); metric originals in catalog.json",
        "state_label": state_label,
        "source_label": source_label,
        "features": features,
    }


def _write_manifest(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    """寫出 case manifest 及其外部自雜湊檔，避免 JSON 內自我雜湊循環。"""

    manifest_path = _write_json(output_dir / "manifest.json", payload)
    digest = _sha256_file(manifest_path)
    checksum_path = output_dir / "manifest.sha256"
    checksum_path.write_text(f"{digest}  manifest.json\n", encoding="utf-8")
    return manifest_path


def _artifact_records(output_dir: Path) -> list[dict[str, Any]]:
    """列出 output directory 內除了 manifest 自身的檔案與 SHA-256。"""

    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "manifest.sha256"}:
            continue
        records.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _run_case(
    month: SurfaceMonth,
    args: argparse.Namespace,
    *,
    output_dir: Path,
    study_design_path: Path,
    radius: int,
    threshold: float,
    case_label: str,
    coastline: Any | None = None,
    coastline_geojson: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """流式處理一個 sensitivity case 並寫出所有可回溯產物。

    ``coastline`` 僅傳入每張 snapshot PNG 的 visual-only 疊圖層；數值檢出、
    tracking、站點分母及 catalog schema 完全沿用既有流程。對應的 GeoJSON
    provenance 另存於 source 與 manifest，使「沒有預設檔」也能被明確追溯。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    config = _make_config(args, radius, threshold)
    indices = _selected_indices(month, args)
    first_snapshot = month.snapshot(indices[0])
    design = json.loads(study_design_path.read_text(encoding="utf-8"))
    site_names = {str(site["study_site_id"]): str(site["study_site_name_zh"]) for site in design["study_sites"]}
    masks = site_masks(
        month.domain_id,
        first_snapshot.lon,
        first_snapshot.lat,
        month.analysis_ocean_mask,
        study_design_path=study_design_path,
    )
    site_ids = ["gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"]
    expected_valid_cells = int(np.count_nonzero(month.analysis_ocean_mask))
    snapshot_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    detection_row_by_object: dict[int, dict[str, Any]] = {}
    tracking_frames: list[TrackingFrame] = []
    metric_frames: list[TrackingFrame] = []
    source = _source_provenance(
        month,
        study_design_path,
        coastline_geojson=coastline_geojson,
    )

    for index in indices:
        snapshot = month.snapshot(index)
        result = detect_snapshot(snapshot.u, snapshot.v, snapshot.valid, snapshot.x_m, snapshot.y_m, config)
        valid_cells = int(np.count_nonzero(snapshot.valid))
        coverage = (valid_cells / expected_valid_cells) if expected_valid_cells else 0.0
        qualified_detections = [detection for detection in result.detections if detection.is_qualified]
        coverage_by_site = {
            site_id: (
                float(np.count_nonzero(snapshot.valid & masks[site_id]) / np.count_nonzero(masks[site_id]))
                if np.any(masks[site_id])
                else 0.0
            )
            for site_id in site_ids
        }
        # 站點統計逐站用各自 coverage 決定分母；全域 coverage 不足時，時序追蹤
        # 則視為資料支援缺口，避免把未檢出錯記成結構已消失。
        metric_frames.append(
            TrackingFrame(
                time_utc_ns=int(snapshot.time_utc_ns),
                detections=tuple(result.detections),
                available=True,
                coverage_by_site=coverage_by_site,
            )
        )
        tracking_frames.append(
            TrackingFrame(
                time_utc_ns=int(snapshot.time_utc_ns),
                detections=tuple(result.detections) if coverage >= args.min_coverage else None,
                available=coverage >= args.min_coverage,
                coverage_by_site=coverage_by_site,
            )
        )
        snapshot_rows.append(
            {
                "snapshot_index": int(snapshot.index),
                "time_utc_ns": int(snapshot.time_utc_ns),
                "time_utc_iso": _time_iso(snapshot.time_utc_ns),
                "state_label": month.state_label,
                "valid_cell_count": valid_cells,
                "expected_valid_cell_count": expected_valid_cells,
                "coverage_fraction": coverage,
                "coverage_pass": bool(coverage >= args.min_coverage),
                "derivative_valid_cell_count": int(np.count_nonzero(result.derivative_valid)),
                "detection_count": len(result.detections),
                "qualified_detection_count": len(qualified_detections),
                "qc_flags": list(result.qc_flags),
            }
        )
        if args.render_snapshots:
            from .figures import render_snapshot_figure

            render_snapshot_figure(
                snapshot,
                result,
                output_dir / "figures" / f"snapshot_{index:06d}.png",
                state_label=_science_status(month),
                source_label=f"{month.domain_id}/{month.month}",
                max_arrows=args.max_arrows,
                coastline=coastline,
            )
        for detection in result.detections:
            hit_sites = [
                site_id
                for site_id in site_ids
                if masks[site_id].any() and _curve_intersects_site(detection, snapshot, masks[site_id])
            ]
            row = _detection_record(
                detection,
                snapshot=snapshot,
                month=month,
                site_ids=hit_sites,
                radius=radius,
                threshold=threshold,
            )
            detection_rows.append(row)
            detection_row_by_object[id(detection)] = row

    tracking_config = TrackingConfig(
        max_core_distance_m=float(args.tracking_max_core_distance_m),
        max_curve_distance_m=float(args.tracking_max_curve_distance_m),
        max_time_gap_seconds=float(args.tracking_max_time_gap_hours) * 3_600.0,
        max_missing_frames=int(args.tracking_max_missing_frames),
        curve_sample_step_m=float(args.tracking_curve_sample_step_m),
        require_qualified=True,
    )
    tracking_result = track_structures(tracking_frames, tracking_config)
    for track in tracking_result.tracks:
        for observation in track.observations:
            row = detection_row_by_object.get(id(observation.detection))
            if row is None:
                continue
            row.update(
                {
                    "track_id": int(track.track_id),
                    "track_duration_hours": float(track.duration_seconds / 3_600.0),
                    "track_termination_reason": track.termination_reason,
                    "track_left_censored": bool(track.left_censored),
                    "track_right_censored": bool(track.right_censored),
                    "track_quality_flags": list(track.quality_flags),
                }
            )

    metrics_result = compute_site_metrics(
        metric_frames,
        masks,
        x_m=first_snapshot.x_m,
        y_m=first_snapshot.y_m,
        cell_area_m2=first_snapshot.cell_area_m2,
        config=SiteMetricsConfig(coverage_threshold=float(args.min_coverage), require_qualified=True),
    )
    track_site_spans: dict[str, list[float]] = {site_id: [] for site_id in site_ids}
    track_site_complete: dict[str, list[float]] = {site_id: [] for site_id in site_ids}
    for track in tracking_result.tracks:
        touched_sites: set[str] = set()
        for observation in track.observations:
            for site_id in site_ids:
                if masks[site_id].any() and _curve_intersects_site(observation.detection, first_snapshot, masks[site_id]):
                    touched_sites.add(site_id)
        for site_id in touched_sites:
            observed_hours = float(track.duration_seconds / 3_600.0)
            track_site_spans[site_id].append(observed_hours)
            # 期初/期末或缺時截尾的觀測跨度只代表壽命下界，不能混入完整壽命平均。
            if not track.left_censored and not track.right_censored and track.termination_reason in {"no_detection", "no_match"}:
                track_site_complete[site_id].append(observed_hours)
    site_statistics: list[dict[str, Any]] = []
    metric_records = {record["site_id"]: record for record in metrics_result.to_records()}
    for site_id in site_ids:
        record = dict(metric_records[site_id])
        spans = track_site_spans[site_id]
        complete = track_site_complete[site_id]
        record.update(
            {
                "study_site_id": site_id,
                "study_site_name_zh": site_names.get(site_id, site_id),
                "occurrence_numerator": record["occurrence_snapshot_count"],
                "occurrence_denominator": record["valid_snapshot_count"],
                "mean_abs_s1_sinv": record["mean_abs_curve_s1_sinv"],
                "track_count": len(spans),
                "complete_track_count": len(complete),
                "censored_track_count": len(spans) - len(complete),
                "mean_observed_track_span_hours": float(np.mean(spans)) if spans else None,
                "mean_complete_track_duration_hours": float(np.mean(complete)) if complete else None,
                "lifetime_status": "complete_tracks_available" if complete else "no_complete_track_duration",
            }
        )
        site_statistics.append(record)
    source_label = f"{month.domain_id}/{month.month}"
    catalog = {
        "schema_version": CLI_SCHEMA_VERSION,
        "operation": "detect",
        "case_label": case_label,
        "state_label": month.state_label,
        "science_status": _science_status(month),
        "source": source,
        "parameters": {
            "detection_config": config.to_manifest_dict(),
            "tracking_config": tracking_config.to_manifest_dict(),
            "site_metrics_config": metrics_result.config.to_manifest_dict(),
            "snapshot_indices": indices,
            "coverage_threshold": float(args.min_coverage),
            "sensitivity_requested": bool(args.sensitivity),
            "robust": False,
            "robust_status": "not_evaluated",
        },
        "snapshots": snapshot_rows,
        "detections": detection_rows,
        "tracks": tracking_result.to_records(),
        "tracking_events": [
            {
                "frame_index": int(event.frame_index),
                "time_utc_ns": int(event.time_utc_ns),
                "event_type": event.event_type,
                "track_id": event.track_id,
                "details": dict(event.details),
            }
            for event in tracking_result.events
        ],
        "site_statistics": site_statistics,
    }
    _write_json(output_dir / "catalog.json", catalog)
    _write_csv(
        output_dir / "catalog.csv",
        detection_rows,
        (
            "domain_id",
            "month",
            "state_label",
            "snapshot_index",
            "time_utc_ns",
            "time_utc_iso",
            "core_row",
            "core_col",
            "core_x_m",
            "core_y_m",
            "core_s1_sinv",
            "curve_length_m",
            "curve_mean_s1_sinv",
            "curve_mean_abs_s1_sinv",
            "forward_stop_reason",
            "backward_stop_reason",
            "is_qualified",
            "study_site_ids",
            "smoothing_radius_cells",
            "min_core_s1_sinv",
            "qc_flags",
            "track_id",
            "track_duration_hours",
            "track_termination_reason",
            "track_left_censored",
            "track_right_censored",
            "track_quality_flags",
        ),
    )
    _write_csv(
        output_dir / "snapshots.csv",
        snapshot_rows,
        (
            "snapshot_index",
            "time_utc_ns",
            "time_utc_iso",
            "state_label",
            "valid_cell_count",
            "expected_valid_cell_count",
            "coverage_fraction",
            "coverage_pass",
            "derivative_valid_cell_count",
            "detection_count",
            "qualified_detection_count",
            "qc_flags",
        ),
    )
    _write_json(
        output_dir / "catalog.geojson",
        _geojson(detection_rows, snapshot=first_snapshot, state_label=month.state_label, source_label=source_label),
    )
    _write_json(
        output_dir / "tracks.json",
        {
            "state_label": month.state_label,
            "config": tracking_config.to_manifest_dict(),
            "tracks": tracking_result.to_records(),
            "events": catalog["tracking_events"],
        },
    )
    _write_json(output_dir / "site_statistics.json", {"state_label": month.state_label, "rows": site_statistics})
    _write_csv(
        output_dir / "site_statistics.csv",
        site_statistics,
        SITE_STAT_FIELDS,
    )
    if args.render_summary:
        from .figures import render_site_summary_figure

        render_site_summary_figure(site_statistics, output_dir / "figures" / "site_summary.png", state_label=_science_status(month))

    manifest = {
        "schema_version": CLI_SCHEMA_VERSION,
        "operation": "detect",
        "case_label": case_label,
        "state_label": month.state_label,
        "science_status": _science_status(month),
        "source": source,
        "coastline_geojson": coastline_geojson,
        "parameters": catalog["parameters"],
        "site_statistics": site_statistics,
        "artifacts": _artifact_records(output_dir),
    }
    _write_manifest(output_dir, manifest)
    return {
        "case_label": case_label,
        "output_dir": str(output_dir),
        "state_label": month.state_label,
        "snapshot_count": len(snapshot_rows),
        "detection_count": len(detection_rows),
        "site_statistics": site_statistics,
    }


def _load_catalog(path_value: str | Path) -> tuple[dict[str, Any], Path]:
    """讀取 catalog JSON，接受檔案路徑或 case output directory。"""

    path = Path(path_value)
    catalog_path = path / "catalog.json" if path.is_dir() else path
    if not catalog_path.is_file():
        raise ValueError(f"找不到 catalog.json: {catalog_path}")
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"catalog JSON 無法讀取: {catalog_path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != CLI_SCHEMA_VERSION:
        raise ValueError("catalog schema_version 不符")
    return payload, catalog_path


def _resolve_catalog_source_path(value: Any, catalog_path: Path, field_name: str) -> Path:
    """解析 catalog provenance path，並拒絕空值或不存在的來源。

    目前 ``detect`` 會把絕對路徑寫入 catalog；保留相對路徑支援是為了讓已封存的
    catalog 可以在同一輸出目錄旁重開。相對路徑只相對於 catalog 所在目錄解讀，
    不使用目前工作目錄，避免從 SERVER 或 Mac 執行時得到不同資料。
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"catalog source 缺少 {field_name}")
    path = Path(value)
    if not path.is_absolute():
        path = catalog_path.parent / path
    path = path.resolve()
    if not path.exists():
        raise ValueError(f"catalog source {field_name} 不存在: {path}")
    return path


def _verify_catalog_source_hash(path: Path, expected: Any, label: str) -> str:
    """核對 provenance 記錄的檔案 SHA-256，回傳實際 digest。"""

    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"catalog source 缺少有效 {label} SHA-256")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"catalog source {label} SHA-256 不符: {path}")
    return actual


def _catalog_snapshot_contract(catalog: Mapping[str, Any], month: SurfaceMonth) -> tuple[int, ...]:
    """核對 catalog snapshot index／UTC 序列與重開月份完全相同。

    literature-figures 必須使用 detect 已處理的同一批 UTC snapshot；若只依 catalog
    的數字欄位而不重新核對月份，可能把另一月份的 detection 曲線套到目前流場上。
    這裡先檢查嚴格遞增，再逐筆比對 ``SurfaceMonth.time_utc_ns``。
    """

    raw_snapshots = catalog.get("snapshots")
    if not isinstance(raw_snapshots, Sequence) or isinstance(raw_snapshots, (str, bytes)) or not raw_snapshots:
        raise ValueError("catalog.snapshots 必須是非空清單")
    indices: list[int] = []
    times: list[int] = []
    for row in raw_snapshots:
        if not isinstance(row, Mapping):
            raise TypeError("catalog.snapshots 每項必須是 object")
        try:
            index = int(row["snapshot_index"])
            time = int(row["time_utc_ns"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("catalog snapshot 缺少有效 snapshot_index/time_utc_ns") from error
        if index < 0 or index >= month.snapshot_count:
            raise ValueError(f"catalog snapshot_index 超出月份: {index}")
        if time != int(month.time_utc_ns[index]):
            raise ValueError(f"catalog snapshot {index} 的 UTC 時間與來源月份不符")
        indices.append(index)
        times.append(time)
    if len(set(indices)) != len(indices):
        raise ValueError("catalog snapshot_index 不可重複")
    if any(first >= second for first, second in pairwise(times)):
        raise ValueError("catalog snapshot UTC 時間必須嚴格遞增")
    return tuple(times)


def _catalog_detection_config(catalog: Mapping[str, Any]) -> DetectionConfig:
    """從 catalog 重建 detection 設定，供首個 snapshot 方法圖重算結果。

    catalog 的 detection 設定是 detect manifest 的唯一參數來源；此函式只取
    ``DetectionConfig`` 宣告的欄位，不讓額外 JSON metadata 靜默成為數值參數。
    """

    parameters = catalog.get("parameters")
    raw_config = parameters.get("detection_config") if isinstance(parameters, Mapping) else None
    if not isinstance(raw_config, Mapping):
        raise TypeError("catalog.parameters.detection_config 必須是 object")
    declared = {field.name for field in dataclass_fields(DetectionConfig)}
    values = {name: raw_config[name] for name in declared if name in raw_config}
    try:
        return DetectionConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"catalog detection_config 無法重建: {error}") from error


def _validate_literature_catalog(
    catalog: Mapping[str, Any],
    catalog_path: Path,
    *,
    study_design_path: Path,
    study_design: Mapping[str, Any],
) -> tuple[SurfaceMonth, tuple[int, ...]]:
    """重開單一 catalog 所引用的 OCM 月份與海岸 sidecar，並核對完整 provenance。

    這是 literature-figures 的輸入 gate。它要求 catalog 仍能找到原始月份、grid
    metadata、帶 metadata 的 coastline sidecar 與 study design；任一 hash、domain、
    month、狀態或 UTC 對應不符就拒絕整次圖組，避免用看似相容的不同來源繪圖。
    """

    source = catalog.get("source")
    if not isinstance(source, Mapping):
        raise TypeError(f"catalog 缺少 source provenance: {catalog_path}")
    domain_id = source.get("domain_id")
    month_value = source.get("month")
    if not isinstance(domain_id, str) or not domain_id:
        raise ValueError(f"catalog source.domain_id 缺失: {catalog_path}")
    if not isinstance(month_value, str) or not month_value:
        raise ValueError(f"catalog source.month 缺失: {catalog_path}")
    state_label = str(catalog.get("state_label") or "")
    science_status = str(catalog.get("science_status") or "")
    if not state_label or not science_status:
        raise ValueError(f"catalog 缺少 state_label/science_status: {catalog_path}")

    month_dir = _resolve_catalog_source_path(source.get("month_dir"), catalog_path, "month_dir")
    coastline = source.get("coastline_mask")
    if not isinstance(coastline, Mapping):
        raise TypeError(f"catalog 缺少 coastline_mask provenance: {catalog_path}")
    sidecar_path = _resolve_catalog_source_path(coastline.get("path"), catalog_path, "coastline_mask.path")
    sidecar_metadata_path = _resolve_catalog_source_path(
        coastline.get("metadata_path"), catalog_path, "coastline_mask.metadata_path"
    )
    month_metadata_path = _resolve_catalog_source_path(source.get("month_metadata_path"), catalog_path, "month_metadata_path")
    grid_metadata_path = _resolve_catalog_source_path(source.get("grid_metadata_path"), catalog_path, "grid_metadata_path")
    source_study_design_path = _resolve_catalog_source_path(source.get("study_design_path"), catalog_path, "study_design_path")

    if month_metadata_path != (month_dir / "metadata.json").resolve():
        raise ValueError("catalog month_metadata_path 與 month_dir 不一致")
    expected_grid_path = (month_dir.parent.parent / "grid" / "metadata.json").resolve()
    if grid_metadata_path != expected_grid_path:
        raise ValueError("catalog grid_metadata_path 與 month_dir 不一致")
    _verify_catalog_source_hash(month_metadata_path, source.get("month_metadata_sha256"), "month metadata")
    _verify_catalog_source_hash(grid_metadata_path, source.get("grid_metadata_sha256"), "grid metadata")
    _verify_catalog_source_hash(sidecar_path, coastline.get("sha256"), "coastline sidecar")
    _verify_catalog_source_hash(
        sidecar_metadata_path,
        coastline.get("metadata_sha256"),
        "coastline metadata",
    )
    _verify_catalog_source_hash(
        source_study_design_path,
        source.get("study_design_sha256"),
        "source study design",
    )
    current_design_sha = _sha256_file(study_design_path)
    if current_design_sha != source.get("study_design_sha256"):
        raise ValueError("catalog study_design SHA-256 與目前指定研究設計不符")

    allow_trial = state_label == "PILOT_ONLY"
    month = open_month(
        month_dir,
        allow_trial_pilot=allow_trial,
        expected_domain_id=domain_id,
        analysis_ocean_mask_path=sidecar_path,
    )
    _validate_month_against_study_design(month, study_design)
    if month.month != month_value:
        raise ValueError(f"catalog month 與來源月份不符: {month_value!r} != {month.month!r}")
    if month.state_label != state_label:
        raise ValueError(f"catalog state_label 與來源月份不符: {state_label!r} != {month.state_label!r}")
    if _science_status(month) != science_status:
        raise ValueError(f"catalog science_status 與來源月份不符: {science_status!r} != {_science_status(month)!r}")
    if month.analysis_mask_status != "provided":
        raise ValueError("literature-figures 必須使用帶 metadata 的 coastline sidecar")
    if source.get("analysis_mask_status") != month.analysis_mask_status:
        raise ValueError("catalog analysis_mask_status 與重開月份不符")
    times = _catalog_snapshot_contract(catalog, month)
    _catalog_detection_config(catalog)
    return month, times


def _literature_qc_blocked_report(report: Any, error: BaseException) -> Any:
    """把 READY 圖面階段的例外轉成可保存的 QC_BLOCKED gate 報告。

    conditional loader 已經把缺檔／格式問題轉成 ``DATA_REQUIRED`` 或 ``QC_BLOCKED``；
    圖函式在零分母、時間不足或空間不相容時仍可能拋出
    ``ConditionalGateError``。CLI 將後者保留為 blocked availability，而不建立半成品
    科學圖，也不讓一項條件圖失敗遮蔽同一批可用的模型圖。
    """

    from .conditional_figures import ConditionalGateReport

    if hasattr(error, "report"):
        return error.report
    return ConditionalGateReport(
        kind=str(getattr(report, "kind", "conditional")),
        status="QC_BLOCKED",
        csv_path=getattr(report, "csv_path", None),
        metadata_path=getattr(report, "metadata_path", None),
        table=None,
        metadata=getattr(report, "metadata", None),
        missing=tuple(getattr(report, "missing", ())),
        issues=(*getattr(report, "issues", ()), str(error)),
    )


def _literature_report_payload(report: Any) -> dict[str, Any]:
    """將 conditional gate 轉成 availability JSON 的精簡狀態。"""

    payload = report.to_dict()
    payload["ready"] = bool(report.ready)
    return payload


def _trajectory_comparison_records(comparisons: Sequence[Any]) -> list[dict[str, Any]]:
    """保存軌跡／TRAP 距離摘要，不把逐點模型座標重複塞入 manifest。"""

    return [
        {
            "trajectory_id": comparison.trajectory_id,
            "source_kind": comparison.source_kind,
            "dwell_hours_within_threshold": comparison.dwell_hours_within_threshold,
            "mean_distance_to_trap_m": comparison.mean_distance_to_trap_m,
            "mean_distance_change_rate_mps": comparison.mean_distance_change_rate_mps,
            "mean_observed_speed_mps": comparison.mean_observed_speed_mps,
            "observed_span_hours": comparison.observed_span_hours,
            "point_count": len(comparison.points),
        }
        for comparison in comparisons
    ]


def _validate_literature_trajectory_parameters(args: argparse.Namespace) -> tuple[float, float]:
    """驗證 literature-figures 的軌跡距離與時間間隔工程參數。

    兩個參數會被寫進 availability 與 manifest，也會送入軌跡摘要計算；即使本次
    沒有提供軌跡 CSV，也必須先拒絕 NaN、無限或非正值，避免成果 metadata 保存一個
    後續無法解讀的門檻。12.5 km 與 6 h 的數值意義仍由呼叫端標成
    ``ENGINEERING_DEFAULT``，不是文獻校準或 LBT receptor 定義。
    """

    distance_threshold_m = float(args.trajectory_distance_threshold_m)
    max_interval_hours = float(args.trajectory_max_interval_hours)
    if not np.isfinite(distance_threshold_m) or distance_threshold_m <= 0.0:
        raise ValueError("--trajectory-distance-threshold-m 必須是正有限值")
    if not np.isfinite(max_interval_hours) or max_interval_hours <= 0.0:
        raise ValueError("--trajectory-max-interval-hours 必須是正有限值")
    return distance_threshold_m, max_interval_hours


def _run_literature_conditional_figures(
    *,
    catalog: Mapping[str, Any],
    month: SurfaceMonth,
    spatial_summary: Any,
    output_dir: Path,
    args: argparse.Namespace,
    coastline: Any | None = None,
) -> dict[str, Any]:
    """依 conditional gate 產出潮相、軌跡與海廢圖，並回傳每項 availability。

    ``None`` 輸入固定進入 ``DATA_REQUIRED``；檔案存在但 gate 或圖面檢查失敗則保留
    ``QC_BLOCKED``。只有 ``READY`` 報告才會呼叫 renderer，因此不會用空表、零分母或
    補值資料寫出看似完整的科學圖。軌跡的距離門檻與最大相鄰時間間隔都寫入報告，
    並標明 12.5 km／6 h 只是工程預設，供後續敏感度分析追蹤。若有 coastline，
    只傳入支援地理輪廓的條件圖 renderer；它不參與資料 gate 或任何統計。
    """

    from .conditional_figures import (
        ConditionalGateError,
        read_marine_debris_table,
        read_tidal_phase_table,
        read_trajectory_table,
        render_debris_trap_overlay_figure,
        render_drifter_trap_comparison_figure,
        render_particle_patch_deformation_figure,
        render_tidal_occurrence_figure,
        render_trajectory_trap_comparison_figure,
        summarize_particle_patch_deformation,
        summarize_trajectory_trap_comparison,
    )

    reports: dict[str, Any] = {}

    def run_gate(kind: str, loader: Any) -> Any:
        """統一 loader 例外為 gate 報告，讓 availability 仍可完整輸出。"""

        try:
            return loader()
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            return _literature_qc_blocked_report(
                type("ReportContext", (), {"kind": kind, "csv_path": None, "metadata_path": None, "metadata": None, "missing": (), "issues": ()})(),
                error,
            )

    tide_report = run_gate("tidal_phase", lambda: read_tidal_phase_table(args.tide_table, catalog))
    reports["tidal_phase"] = _literature_report_payload(tide_report)
    if tide_report.ready:
        try:
            render_tidal_occurrence_figure(tide_report, catalog, output_dir / "tidal_occurrence.png")
            reports["tidal_phase"]["figure"] = "tidal_occurrence.png"
        except ConditionalGateError as error:
            tide_report = error.report
            reports["tidal_phase"] = _literature_report_payload(tide_report)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            tide_report = _literature_qc_blocked_report(tide_report, error)
            reports["tidal_phase"] = _literature_report_payload(tide_report)

    trajectory_report = run_gate(
        "trajectory",
        lambda: read_trajectory_table(args.trajectory_table, catalog=catalog),
    )
    trajectory_payload = _literature_report_payload(trajectory_report)
    trajectory_payload["parameters"] = {
        "distance_threshold_m": float(args.trajectory_distance_threshold_m),
        "distance_threshold_status": "ENGINEERING_DEFAULT"
        if float(args.trajectory_distance_threshold_m) == 12_500.0
        else "CLI_OVERRIDE",
        "max_interval_hours": float(args.trajectory_max_interval_hours),
        "max_interval_status": "ENGINEERING_DEFAULT"
        if float(args.trajectory_max_interval_hours) == 6.0
        else "CLI_OVERRIDE",
        "renderer_dwell_interval_hours": float(args.trajectory_max_interval_hours),
        "renderer_dwell_interval_status": "same_as_cli_parameter; sensitivity_required",
    }
    reports["trajectory"] = trajectory_payload
    if trajectory_report.ready:
        try:
            comparisons = summarize_trajectory_trap_comparison(
                trajectory_report,
                catalog,
                month,
                distance_threshold_m=float(args.trajectory_distance_threshold_m),
                max_interval_hours=float(args.trajectory_max_interval_hours),
            )
            trajectory_table = trajectory_report.require_ready()
            if getattr(trajectory_table, "source_kind", None) == "observed_drifter":
                render_drifter_trap_comparison_figure(
                    trajectory_report,
                    catalog,
                    month,
                    output_dir / "trajectory_trap_comparison.png",
                    distance_threshold_m=float(args.trajectory_distance_threshold_m),
                    max_interval_hours=float(args.trajectory_max_interval_hours),
                    coastline=coastline,
                )
            else:
                render_trajectory_trap_comparison_figure(
                    trajectory_report,
                    catalog,
                    month,
                    output_dir / "trajectory_trap_comparison.png",
                    distance_threshold_m=float(args.trajectory_distance_threshold_m),
                    max_interval_hours=float(args.trajectory_max_interval_hours),
                    coastline=coastline,
                )
            _write_json(
                output_dir / "trajectory_summary.json",
                {
                    "source_kind": trajectory_table.source_kind,
                    "distance_threshold_m": float(args.trajectory_distance_threshold_m),
                    "max_interval_hours": float(args.trajectory_max_interval_hours),
                    "parameter_status": "ENGINEERING_DEFAULT unless explicitly overridden",
                    "records": _trajectory_comparison_records(comparisons),
                },
            )
            reports["trajectory"]["figure"] = "trajectory_trap_comparison.png"
            reports["trajectory"]["summary"] = "trajectory_summary.json"
        except ConditionalGateError as error:
            trajectory_report = error.report
            reports["trajectory"] = _literature_report_payload(trajectory_report)
            reports["trajectory"]["parameters"] = trajectory_payload["parameters"]
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            trajectory_report = _literature_qc_blocked_report(trajectory_report, error)
            reports["trajectory"] = _literature_report_payload(trajectory_report)
            reports["trajectory"]["parameters"] = trajectory_payload["parameters"]

    # Serra/Haller 的短期物質變形版面只能在同一模型粒子釋放群可追蹤時借鏡。
    # 獨立漂流器可以做距離驗證，不能自動視為同一模型粒子 patch；缺資料時
    # availability 明列 gate，避免把瞬時流線圖冒充有限時間變形結果。
    deformation_payload: dict[str, Any] = {
        "status": "DATA_REQUIRED",
        "ready": False,
        "missing": ["同一釋放群 model_particle CSV 與 metadata.patch_id，至少三粒子／兩個共同 UTC 時刻"],
        "issues": [],
    }
    if trajectory_report.status == "QC_BLOCKED":
        deformation_payload["status"] = "QC_BLOCKED"
        deformation_payload["issues"] = list(trajectory_report.issues)
    elif trajectory_report.ready and getattr(trajectory_report.require_ready(), "source_kind", None) == "model_particle":
        try:
            patch_summary = summarize_particle_patch_deformation(trajectory_report, catalog)
            patch_path = render_particle_patch_deformation_figure(
                trajectory_report,
                catalog,
                month,
                output_dir / "particle_patch_deformation.png",
            )
            _write_json(
                output_dir / "particle_patch_summary.json",
                {key: value for key, value in patch_summary.items() if not key.startswith("_")},
            )
            deformation_payload = {
                "status": "READY",
                "ready": True,
                "missing": [],
                "issues": [],
                "figure": patch_path.name,
                "summary": "particle_patch_summary.json",
            }
        except ConditionalGateError as error:
            deformation_payload = _literature_report_payload(error.report)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            deformation_payload = _literature_report_payload(_literature_qc_blocked_report(trajectory_report, error))
    reports["short_term_deformation"] = deformation_payload

    debris_report = run_gate(
        "marine_debris",
        lambda: read_marine_debris_table(args.debris_table, catalog=catalog),
    )
    reports["marine_debris"] = _literature_report_payload(debris_report)
    if debris_report.ready:
        try:
            render_debris_trap_overlay_figure(
                debris_report,
                catalog,
                spatial_summary,
                output_dir / "debris_trap_overlay.png",
                coastline=coastline,
            )
            reports["marine_debris"]["figure"] = "debris_trap_overlay.png"
        except ConditionalGateError as error:
            debris_report = error.report
            reports["marine_debris"] = _literature_report_payload(debris_report)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            debris_report = _literature_qc_blocked_report(debris_report, error)
            reports["marine_debris"] = _literature_report_payload(debris_report)
    return reports


def _run_literature_figures(args: argparse.Namespace) -> int:
    """由一至四個同窗 catalog 產出文獻借鏡圖組與來源 availability manifest。

    輸入 gate 先對每份 catalog 重開其原始月份與 coastline sidecar，再檢查四區／五站
    的 domain、UTC snapshot、detection/tracking 參數及科學狀態。只有模型輸入全部
    通過時才建立圖組；潮相、軌跡、海廢則各自由 conditional gate 決定是否產圖。
    """

    if not 1 <= len(args.catalog) <= 4:
        raise ValueError("literature-figures 的 --catalog 必須提供 1 到 4 份 catalog")
    _validate_literature_trajectory_parameters(args)
    # 同一批四區圖共用一份已驗證的 GeoJSON，避免各 domain 讀取不同檔案或在
    # manifest 中留下不一致的 SHA。預設檔缺失仍可產生模型圖，但會明示未疊圖。
    coastline, coastline_geojson = _resolve_coastline_geojson(args)
    output_root = Path(args.output_root)
    study_design_path = Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH).resolve()
    study_design = load_study_design(study_design_path)
    # 人類可讀的區名只用於圖面；來源驗證、檔名與 SHA 仍採固定上游 domain ID。
    # 因此 C 區呈現「南灣海域」而不改動既有 houwan_nmmba_cache_v3 邊界。
    region_names = {str(row["flow_domain_id"]): str(row["analysis_region_name_zh"]) for row in study_design["flow_domains"]}
    loaded: list[dict[str, Any]] = []
    for catalog_argument in args.catalog:
        catalog, catalog_path = _load_catalog(catalog_argument)
        month, times = _validate_literature_catalog(
            catalog,
            catalog_path,
            study_design_path=study_design_path,
            study_design=study_design,
        )
        loaded.append(
            {
                "catalog": catalog,
                "catalog_path": catalog_path,
                "month": month,
                "times": times,
                "domain_id": month.domain_id,
            }
        )

    domains = {str(item["domain_id"]) for item in loaded}
    if len(domains) != len(loaded):
        raise ValueError("literature-figures 不可重複提供同一 flow domain")
    state_labels = {str(item["catalog"]["state_label"]) for item in loaded}
    science_statuses = {str(item["catalog"]["science_status"]) for item in loaded}
    if len(state_labels) != 1 or len(science_statuses) != 1:
        raise ValueError("literature-figures 不可混用不同 state_label/science_status（含 PILOT_ONLY 與正式資料）")
    first = loaded[0]
    first_times = first["times"]
    first_catalog = first["catalog"]
    first_parameters = first_catalog.get("parameters")
    if not isinstance(first_parameters, Mapping):
        raise TypeError("catalog.parameters 必須是 object")
    first_detection_config = first_parameters.get("detection_config")
    first_tracking_config = first_parameters.get("tracking_config")
    first_month = first["month"].month
    for item in loaded[1:]:
        if item["times"] != first_times:
            raise ValueError("literature-figures 的 catalog 必須使用完全相同 UTC snapshot 時間窗")
        if item["month"].month != first_month:
            raise ValueError("literature-figures 不可混用不同月份")
        parameters = item["catalog"].get("parameters")
        if not isinstance(parameters, Mapping) or parameters.get("detection_config") != first_detection_config:
            raise ValueError("literature-figures 的 catalog detection 參數必須完全相同")
        if not isinstance(parameters, Mapping) or parameters.get("tracking_config") != first_tracking_config:
            raise ValueError("literature-figures 的 catalog tracking 參數必須完全相同")

    _require_empty_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state_label = next(iter(state_labels))
    science_status = next(iter(science_statuses))
    figure_records: list[dict[str, Any]] = []
    availability_domains: dict[str, Any] = {}

    from .figures import _site_font_properties, render_site_summary_figure
    from .literature_figures import (
        render_lifecycle_intensity_figure,
        render_lifetime_distribution_figure,
        render_mechanism_figure,
        render_method_comparison_figure,
        summarize_track_lifetimes,
    )
    from .spatial import build_spatial_summary
    from .spatial_figures import render_spatial_summary_figure

    mechanism_path = render_mechanism_figure(
        output_root / "mechanism.png",
        state_label="CONCEPT_ONLY",
        source_label="TRAP mechanism schematic; no catalog values",
    )
    figure_records.append({"kind": "mechanism", "path": str(mechanism_path.relative_to(output_root)), "status": "CONCEPT_ONLY"})

    active_site_rows: list[dict[str, Any]] = []
    spatial_summaries: dict[str, Any] = {}
    for item in loaded:
        catalog = item["catalog"]
        catalog_path = item["catalog_path"]
        month = item["month"]
        domain_id = item["domain_id"]
        canonical_region_name = region_names[str(domain_id)]
        display_name = canonical_region_name if _site_font_properties() is not None else str(domain_id)
        domain_output = output_root / domain_id
        domain_output.mkdir(parents=True, exist_ok=True)
        snapshot_index = min(int(row["snapshot_index"]) for row in catalog["snapshots"])
        snapshot = month.snapshot(snapshot_index)
        config = _catalog_detection_config(catalog)
        result = detect_snapshot(snapshot.u, snapshot.v, snapshot.valid, snapshot.x_m, snapshot.y_m, config)
        source_label = f"{display_name} | {month.month} | {_time_iso(snapshot.time_utc_ns)} UTC"
        method_path = render_method_comparison_figure(
            snapshot,
            result,
            domain_output / "method_comparison.png",
            state_label=science_status,
            source_label=f"{display_name} | {month.month}",
            coastline=coastline,
        )
        spatial_summary = build_spatial_summary(catalog, month)
        spatial_summaries[domain_id] = spatial_summary
        spatial_path = render_spatial_summary_figure(
            spatial_summary,
            domain_output / "spatial_summary.png",
            display_name=display_name,
            coastline=coastline,
        )
        lifetime_path = render_lifetime_distribution_figure(
            catalog,
            domain_output / "observed_span_distribution.png",
            state_label=science_status,
            source_label=source_label,
        )
        lifecycle_path = render_lifecycle_intensity_figure(
            catalog,
            domain_output / "lifecycle_intensity.png",
            state_label=science_status,
            source_label=source_label,
        )
        _write_json(domain_output / "spatial_summary_metadata.json", spatial_summary.to_metadata())
        _write_json(domain_output / "lifetime_summary.json", summarize_track_lifetimes(catalog))
        conditional = _run_literature_conditional_figures(
            catalog=catalog,
            month=month,
            spatial_summary=spatial_summary,
            output_dir=domain_output,
            args=args,
            coastline=coastline,
        )
        domain_figures = {
            "method_comparison": str(method_path.relative_to(output_root)),
            "spatial_summary": str(spatial_path.relative_to(output_root)),
            "observed_span_distribution": str(lifetime_path.relative_to(output_root)),
            "lifecycle_intensity": str(lifecycle_path.relative_to(output_root)),
        }
        # JSON 是圖的統計契約與數值摘要，不列入 figure 清單；其完整性仍由
        # manifest.artifacts 的 SHA-256 獨立保護。
        domain_summaries = {
            "spatial_summary_metadata": str((domain_output / "spatial_summary_metadata.json").relative_to(output_root)),
            "lifetime_summary": str((domain_output / "lifetime_summary.json").relative_to(output_root)),
        }
        for kind, relative_path in domain_figures.items():
            figure_records.append({"kind": kind, "domain_id": domain_id, "path": relative_path, "status": science_status})
        for conditional_kind, conditional_payload in conditional.items():
            conditional_figure = conditional_payload.get("figure")
            if conditional_figure:
                relative_path = str((domain_output / conditional_figure).relative_to(output_root))
                figure_records.append({"kind": conditional_kind, "domain_id": domain_id, "path": relative_path, "status": science_status})
        availability_domains[domain_id] = {
            "region_name_zh": canonical_region_name,
            "catalog_path": str(catalog_path),
            "catalog_sha256": _sha256_file(catalog_path),
            "month": month.month,
            "snapshot_time_utc_ns": list(item["times"]),
            "state_label": state_label,
            "science_status": science_status,
            "coastline_geojson": coastline_geojson,
            "figures": domain_figures,
            "summaries": domain_summaries,
            "conditional": conditional,
        }
        site_statistics = catalog.get("site_statistics")
        if isinstance(site_statistics, list):
            active_site_rows.extend(
                dict(row)
                for row in site_statistics
                if isinstance(row, Mapping) and float(row.get("site_area_m2") or 0.0) > 0.0
            )

    expected_domains = {str(item["flow_domain_id"]) for item in study_design["flow_domains"]}
    if domains == expected_domains:
        expected_sites = {str(item["study_site_id"]) for item in study_design["study_sites"]}
        site_ids = {str(row.get("study_site_id")) for row in active_site_rows}
        if site_ids != expected_sites:
            raise ValueError("四區 catalog 未恰好覆蓋五個 study site，拒絕產出五站 summary")
        site_summary_path = render_site_summary_figure(
            active_site_rows,
            output_root / "five_site_summary.png",
            state_label=science_status,
        )
        figure_records.append({"kind": "five_site_summary", "path": str(site_summary_path.relative_to(output_root)), "status": science_status})

    availability = {
        "schema_version": "literature_figures_v1",
        "operation": "literature-figures",
        "state_label": state_label,
        "science_status": science_status,
        "coastline_geojson": coastline_geojson,
        "catalog_count": len(loaded),
        "catalogs": [
            {
                "path": str(item["catalog_path"]),
                "sha256": _sha256_file(item["catalog_path"]),
                "domain_id": item["domain_id"],
                "month": item["month"].month,
                "snapshot_time_utc_ns": list(item["times"]),
            }
            for item in loaded
        ],
        "figures": figure_records,
        "domains": availability_domains,
        "parameters": {
            "trajectory_distance_threshold_m": float(args.trajectory_distance_threshold_m),
            "trajectory_distance_threshold_status": "ENGINEERING_DEFAULT"
            if float(args.trajectory_distance_threshold_m) == 12_500.0
            else "CLI_OVERRIDE",
            "trajectory_max_interval_hours": float(args.trajectory_max_interval_hours),
            "trajectory_max_interval_status": "ENGINEERING_DEFAULT"
            if float(args.trajectory_max_interval_hours) == 6.0
            else "CLI_OVERRIDE",
            "conditional_trajectory_renderer_max_interval_hours": float(args.trajectory_max_interval_hours),
            "conditional_trajectory_renderer_note": "圖與摘要使用同一相鄰點間隔門檻；正式推論需做 interval sensitivity",
        },
    }
    availability_path = _write_json(output_root / "figure_availability.json", availability)
    manifest = {
        "schema_version": CLI_SCHEMA_VERSION,
        "operation": "literature-figures",
        "state_label": state_label,
        "science_status": science_status,
        "coastline_geojson": coastline_geojson,
        "study_design_path": str(study_design_path),
        "study_design_sha256": _sha256_file(study_design_path),
        "catalogs": availability["catalogs"],
        "parameters": availability["parameters"],
        "figure_availability": str(availability_path.relative_to(output_root)),
        "artifacts": _artifact_records(output_root),
    }
    _write_manifest(output_root, manifest)
    print(json.dumps(availability, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_preflight(args: argparse.Namespace) -> int:
    """執行單月份輸入 gate，輸出可保存的來源與 PILOT 狀態摘要。"""

    month = _open_month_from_args(args)
    study_design_path = Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH)
    source = _source_provenance(month, study_design_path)
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "operation": "preflight",
        "state_label": month.state_label,
        "science_status": _science_status(month),
        "source": source,
        "snapshot_count": month.snapshot_count,
        "domain_id": month.domain_id,
        "month": month.month,
        "analysis_mask_status": month.analysis_mask_status,
    }
    if args.output_root:
        output_dir = Path(args.output_root)
        _require_empty_output(output_dir)
        _write_json(output_dir / "preflight.json", payload)
        digest = _sha256_file(output_dir / "preflight.json")
        (output_dir / "preflight.sha256").write_text(f"{digest}  preflight.json\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_detect(args: argparse.Namespace) -> int:
    """依 sensitivity 笛卡兒積逐時產生 catalog 與站點成果。"""

    month = _open_month_from_args(args)
    # 海岸線只在繪圖層使用，但仍要在 case 開始前驗證一次並寫入每層
    # provenance；如此 sensitivity case 共用同一份檔案 SHA，也不會因重複載入
    # 造成各 case 的圖面來源不一致。
    coastline, coastline_geojson = _resolve_coastline_geojson(args)
    # `mask_static` 只保證內插支撐，並非海岸。正式 cache 必須有可追溯的 coastline
    # sidecar，否則曲線可能跨陸地，不能進入正式 detect/summary 成果。
    if not month.pilot_only and month.analysis_mask_status != "provided":
        raise ValueError("正式 detect 必須提供有 metadata 的 --ocean-mask 海岸 sidecar")
    output_root = Path(args.output_root)
    _require_empty_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    study_design_path = Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH)
    radii, thresholds = _case_values(args)
    cases: list[dict[str, Any]] = []
    total_cases = len(radii) * len(thresholds)
    case_index = 0
    for radius in radii:
        for threshold in thresholds:
            case_label = f"smoothing_r{radius}_threshold_{case_index:02d}"
            case_dir = output_root if total_cases == 1 else output_root / "sensitivity" / case_label
            cases.append(
                _run_case(
                    month,
                    args,
                    output_dir=case_dir,
                    study_design_path=study_design_path,
                    radius=radius,
                    threshold=threshold,
                    case_label=case_label,
                    coastline=coastline,
                    coastline_geojson=coastline_geojson,
                )
            )
            case_index += 1
    root_manifest = {
        "schema_version": CLI_SCHEMA_VERSION,
        "operation": "detect",
        "state_label": month.state_label,
        "science_status": _science_status(month),
        "source": _source_provenance(
            month,
            study_design_path,
            coastline_geojson=coastline_geojson,
        ),
        "coastline_geojson": coastline_geojson,
        "parameters": {
            "smoothing_radius_cells": radii,
            "min_core_s1_sinv": thresholds,
            "case_count": total_cases,
            "robust": False,
            "robust_status": "not_evaluated",
        },
        "cases": cases,
    }
    if total_cases > 1:
        root_manifest["artifacts"] = _artifact_records(output_root)
        _write_manifest(output_root, root_manifest)
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_summary(args: argparse.Namespace) -> int:
    """合併同一參數與時間窗的四區 catalog，輸出五站摘要而不重讀流場。

    每個站點在本次 summary 只能出現一次；跨月份同站點的分子分母、強度與
    截尾壽命不能由本函式草率相加。若提供多份 catalog，必須有相同 snapshot UTC
    時間清單和 detection 參數，才能把四區站點放在同一張比較圖。
    """

    loaded = [_load_catalog(path) for path in args.catalog]
    first_catalog = loaded[0][0]
    first_times = [row["time_utc_ns"] for row in first_catalog.get("snapshots", [])]
    first_config = first_catalog.get("parameters", {}).get("detection_config")
    state_labels = {catalog.get("state_label") for catalog, _ in loaded}
    if len(state_labels) != 1:
        raise ValueError("summary 不可混合 PILOT_ONLY 與 ready catalog")
    rows: list[dict[str, Any]] = []
    seen_sites: set[str] = set()
    sources: list[dict[str, Any]] = []
    for catalog, catalog_path in loaded:
        if [row["time_utc_ns"] for row in catalog.get("snapshots", [])] != first_times:
            raise ValueError("summary 的各區 catalog 必須使用相同 UTC snapshot 時間清單")
        if catalog.get("parameters", {}).get("detection_config") != first_config:
            raise ValueError("summary 的各區 catalog 必須使用相同 detection 參數")
        station_rows = catalog.get("site_statistics")
        if not isinstance(station_rows, list):
            raise TypeError(f"catalog 缺少 site_statistics: {catalog_path}")
        # 單區 catalog 雖固定帶五站 schema，研究面積為 0 的域外站點不進比較圖。
        active = [dict(row) for row in station_rows if float(row.get("site_area_m2") or 0.0) > 0.0]
        for row in active:
            site_id = str(row["study_site_id"])
            if site_id in seen_sites:
                raise ValueError(f"summary 同一站點出現多次；跨月份需要另行追蹤與聚合: {site_id}")
            seen_sites.add(site_id)
            rows.append(row)
        sources.append({"catalog_path": str(catalog_path), "catalog_sha256": _sha256_file(catalog_path), "source": catalog.get("source")})
    if not rows:
        raise ValueError("summary 沒有任何具統計面積的站點")
    site_order = {site_id: index for index, site_id in enumerate(("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"))}
    rows.sort(key=lambda row: site_order.get(str(row["study_site_id"]), len(site_order)))
    output_dir = Path(args.output_root)
    _require_empty_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "operation": "summary",
        "state_label": first_catalog.get("state_label"),
        "science_status": first_catalog.get("science_status"),
        "catalogs": sources,
        "parameters": first_catalog.get("parameters"),
        "snapshot_time_utc_ns": first_times,
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", payload)
    _write_csv(
        output_dir / "site_statistics.csv",
        rows,
        SITE_STAT_FIELDS,
    )
    if args.render:
        from .figures import render_site_summary_figure

        render_site_summary_figure(rows, output_dir / "site_summary.png", state_label=str(first_catalog.get("science_status") or first_catalog.get("state_label")))
    manifest = {
        "schema_version": CLI_SCHEMA_VERSION,
        "operation": "summary",
        "state_label": first_catalog.get("state_label"),
        "science_status": first_catalog.get("science_status"),
        "catalogs": sources,
        "artifacts": _artifact_records(output_dir),
    }
    _write_manifest(output_dir, manifest)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_render(args: argparse.Namespace) -> int:
    """重新繪製 summary，或由顯式月份和 index 產生單一 snapshot 圖。"""

    coastline = None
    coastline_geojson: dict[str, Any] | None = None
    if args.kind == "snapshot":
        # 先解析顯式／預設來源，再建立 output root；顯式 GeoJSON 錯誤不應留下
        # 看似完成但沒有圖的成果目錄。
        coastline, coastline_geojson = _resolve_coastline_geojson(args)
    output_dir = Path(args.output_root)
    _require_empty_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.kind == "summary":
        if args.catalog is None:
            raise ValueError("render summary 必須提供 --catalog")
        catalog, catalog_path = _load_catalog(args.catalog)
        rows = catalog.get("site_statistics")
        if not isinstance(rows, list):
            raise ValueError("catalog 缺少 site_statistics")
        from .figures import render_site_summary_figure

        figure_path = render_site_summary_figure(
            rows,
            output_dir / "site_summary.png",
            state_label=str(catalog.get("science_status") or catalog.get("state_label")),
        )
        payload = {
            "schema_version": CLI_SCHEMA_VERSION,
            "operation": "render",
            "kind": "summary",
            "catalog_path": str(catalog_path),
            "state_label": catalog.get("state_label"),
            "figure": str(figure_path),
        }
    else:
        if args.snapshot_index is None:
            raise ValueError("render snapshot 必須提供 --snapshot-index")
        month = _open_month_from_args(args)
        if not month.pilot_only and month.analysis_mask_status != "provided":
            raise ValueError("正式 snapshot 圖必須提供有 metadata 的 --ocean-mask 海岸 sidecar")
        config = _make_config(args, args.smoothing_radius_cells[0] if args.smoothing_radius_cells else 0, args.min_core_s1_sinv if args.min_core_s1_sinv is not None else DetectionConfig().min_core_s1_sinv)
        snapshot = month.snapshot(int(args.snapshot_index))
        result = detect_snapshot(snapshot.u, snapshot.v, snapshot.valid, snapshot.x_m, snapshot.y_m, config)
        from .figures import render_snapshot_figure

        figure_path = render_snapshot_figure(
            snapshot,
            result,
            output_dir / f"snapshot_{snapshot.index:06d}.png",
            state_label=_science_status(month),
            source_label=f"{month.domain_id}/{month.month}",
            max_arrows=args.max_arrows,
            coastline=coastline,
        )
        payload = {
            "schema_version": CLI_SCHEMA_VERSION,
            "operation": "render",
            "kind": "snapshot",
            "state_label": month.state_label,
            "snapshot_index": snapshot.index,
            "figure": str(figure_path),
            "parameters": config.to_manifest_dict(),
            "coastline_geojson": coastline_geojson,
        }
    _write_json(output_dir / "render.json", payload)
    manifest = {
        "schema_version": CLI_SCHEMA_VERSION,
        "operation": "render",
        "kind": args.kind,
        "state_label": payload.get("state_label"),
        **({"coastline_geojson": coastline_geojson} if args.kind == "snapshot" else {}),
        "artifacts": _artifact_records(output_dir),
    }
    _write_manifest(output_dir, manifest)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _add_input_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_optional: bool = False,
    source_required: bool = True,
) -> None:
    """加入所有需要顯式指定來源的共用 CLI 參數。"""

    source = parser.add_mutually_exclusive_group(required=source_required)
    source.add_argument("--month-dir", type=Path, help="單一 OCM surface 月份目錄")
    source.add_argument("--surface-root", type=Path, help="OCM surface 根目錄，例如 SERVER 的 preprocessed/ocm_surface")
    parser.add_argument("--domain-id", help="surface-root 下的 OCM flow domain ID")
    parser.add_argument("--month", help="surface-root 下的 YYYYMM 月份")
    parser.add_argument("--ocean-mask", type=Path, help="analysis ocean mask sidecar .npy")
    parser.add_argument(
        "--coastline-geojson",
        type=Path,
        help="成果圖 CRS84 海岸線 GeoJSON；未指定時使用專案 data/taiwan_exact_coastline 預設檔",
    )
    parser.add_argument("--allow-trial-pilot", action="store_true", help="明示允許 trial_partial_month，所有產物標 PILOT_ONLY")
    parser.add_argument("--study-design", type=Path, default=DEFAULT_STUDY_DESIGN_PATH, help="四區域五站點研究設計 JSON")
    if output_optional:
        parser.add_argument("--output-root", type=Path, help="可選的 preflight 輸出目錄")


def _add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    """加入 detection、sensitivity 及曲線品質門檻。"""

    parser.add_argument("--sensitivity", action="store_true", help="執行 0/1/2/3 格 × 兩組核心門檻")
    parser.add_argument("--smoothing-radius-cells", "--smoothing-radius", type=int, action="append", help="可重複指定 0、1、2 或 3")
    parser.add_argument("--core-thresholds-sinv", "--thresholds-sinv", nargs=2, type=float, help="兩組負核心 s1 門檻")
    parser.add_argument("--min-core-s1-sinv", type=float, help="單一負核心 s1 門檻（非 sensitivity）")
    parser.add_argument("--min-eigenvalue-gap-sinv", type=float, default=1.0e-12)
    parser.add_argument("--local-min-radius-cells", type=int, default=1)
    parser.add_argument("--core-min-separation-m", type=float, default=0.0)
    parser.add_argument("--integration-step-m", type=float, default=None)
    parser.add_argument("--max-curve-length-m", type=float, default=20_000.0)
    parser.add_argument("--max-integration-steps", type=int, default=2_000)
    parser.add_argument("--curve-s1-stop-sinv", type=float, default=0.0)
    parser.add_argument("--curve-min-relative-strength", type=float, default=0.30)
    parser.add_argument("--curve-monotonic-tolerance-sinv", type=float, default=1.0e-12)
    parser.add_argument("--max-turn-angle-deg", type=float, default=90.0)
    parser.add_argument("--min-curve-points", type=int, default=3)
    parser.add_argument("--min-curve-length-m", type=float, default=0.0)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--tracking-max-core-distance-m", type=float, default=5_000.0)
    parser.add_argument("--tracking-max-curve-distance-m", type=float, default=10_000.0)
    parser.add_argument("--tracking-max-time-gap-hours", type=float, default=3.0)
    parser.add_argument("--tracking-max-missing-frames", type=int, default=1)
    parser.add_argument("--tracking-curve-sample-step-m", type=float, default=500.0)
    parser.add_argument("--snapshot-index", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--max-arrows", type=int, default=900)


def build_parser() -> argparse.ArgumentParser:
    """建立 `python -m traps` 的完整子命令 parser。"""

    parser = argparse.ArgumentParser(description="OCM 表層 TRAP 短期吸引潛勢分析")
    subparsers = parser.add_subparsers(dest="command")

    preflight = subparsers.add_parser("preflight", help="驗證單月份輸入與來源狀態")
    _add_input_arguments(preflight, output_optional=True)
    preflight.set_defaults(handler=_run_preflight)

    detect = subparsers.add_parser("detect", help="逐時檢出 TRAP 並寫出 catalog/站點表/manifest")
    _add_input_arguments(detect)
    detect.add_argument("--output-root", type=Path, required=True)
    detect.add_argument("--render-snapshots", action="store_true", help="輸出每個處理時刻的 s1/速度/TRAP 圖")
    detect.add_argument("--render-summary", action="store_true", help="輸出五站摘要圖")
    _add_detection_arguments(detect)
    detect.set_defaults(handler=_run_detect)

    summary = subparsers.add_parser("summary", help="由 catalog 產生五站摘要表與圖")
    summary.add_argument("--catalog", type=Path, nargs="+", required=True, help="同一 UTC 時間窗、相同參數的四區 catalog 路徑")
    summary.add_argument("--output-root", type=Path, required=True)
    summary.add_argument("--render", action="store_true")
    summary.set_defaults(handler=_run_summary)

    literature = subparsers.add_parser(
        "literature-figures",
        help="依 catalog 產出文獻借鏡方法、空間、觀測跨度與條件圖",
    )
    literature.add_argument(
        "--catalog",
        type=Path,
        nargs="+",
        required=True,
        help="1 到 4 份同 UTC 時窗、同 detection/tracking 參數的單區 catalog.json",
    )
    literature.add_argument("--output-root", type=Path, required=True)
    literature.add_argument("--study-design", type=Path, default=DEFAULT_STUDY_DESIGN_PATH)
    literature.add_argument(
        "--coastline-geojson",
        type=Path,
        help="成果圖 CRS84 海岸線 GeoJSON；未指定時使用專案 data/taiwan_exact_coastline 預設檔",
    )
    literature.add_argument("--tide-table", type=Path, help="潮相 CSV；同 stem JSON 必須通過 provenance/QC gate")
    literature.add_argument("--trajectory-table", type=Path, help="漂流器或模型粒子 CSV；同 stem JSON 必須通過 gate")
    literature.add_argument("--debris-table", type=Path, help="海廢 CSV；同 stem JSON 必須通過 gate")
    literature.add_argument(
        "--trajectory-distance-threshold-m",
        type=float,
        default=12_500.0,
        help="軌跡至 TRAP 距離門檻（m）；12.5 km 僅為 ENGINEERING_DEFAULT",
    )
    literature.add_argument(
        "--trajectory-max-interval-hours",
        type=float,
        default=6.0,
        help="軌跡相鄰點最大時間間隔（h）；6 h 僅為 ENGINEERING_DEFAULT",
    )
    literature.set_defaults(handler=_run_literature_figures)

    render = subparsers.add_parser("render", help="重新輸出 summary 或單一 snapshot 圖")
    render.add_argument("--kind", choices=("summary", "snapshot"), default="summary")
    render.add_argument("--catalog", type=Path)
    render.add_argument("--output-root", type=Path, required=True)
    _add_input_arguments(render, source_required=False)
    _add_detection_arguments(render)
    render.set_defaults(handler=_run_render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令列主入口；錯誤以非零狀態回傳且不吞掉輸入 gate 原因。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"TRAPs error: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
