"""跨月份 TRAP 檢出／追蹤的第一、二階段：detect-shards 與 track-period。

`detect` 一次只能處理單一 domain-month，且把整月結果攤成一份巨大的 JSON catalog，
無法跨月份接續。全期研究（4 個流場區域 × 2024–2025 共 24 個月、每月約 744 個逐時
snapshot）需要可平行、可續跑、可串流的兩階段設計：

* Stage 1 ``detect-shards``：單一 domain-month 一次呼叫，流式呼叫
  :func:`traps.detection.detect_snapshot`，把逐時 snapshot 與逐 detection 結果寫成
  精簡的 ``.npz`` shard，不在記憶體保留整月結果。
* Stage 2 ``track-period``：讀入一個 domain 的多個月份 shard，先建立「串接月份→按 UTC
  穩定排序→重複 UTC 保留較晚月份」的正式研究期時間軸（與 OCM-SVD-Analysis、
  Lagrangian-Ensemble-Backtracking 相同的 ``sort_and_deduplicate_prefer_last``
  契約），再以 :class:`traps.tracking.StreamingTracker` 逐月串流建立跨月份 track。

兩階段都不插值、不補缺，且輸出目錄必須是新的或空的；``--resume`` 只在既有輸出已
通過 ``_SUCCESS`` + manifest artifact SHA-256 驗證時略過重算，否則清除本命令已知會
寫出的檔案後重跑。stage 3（跨月份／跨區彙整）不在本模組範圍內，這裡的輸出欄位是
刻意為那個階段保留的完整證據鏈。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
import os
import subprocess
import sys
import time as _time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cli import (
    DEFAULT_STUDY_DESIGN_PATH,
    _add_detection_arguments,
    _make_config,
    _open_month_from_args,
    _science_status,
    _sha256_file,
    _source_provenance,
)
from .detection import detect_snapshot
from .metrics import _grid_axes
from .models import DetectionConfig, TrapDetection
from .tracking import (
    StreamingTracker,
    TrackingConfig,
    TrackingFrame,
    densify_polyline,
)

SHARD_SCHEMA_VERSION = "traps.period.shard.v1"
TRACK_SCHEMA_VERSION = "traps.period.track.v1"

DEFAULT_PROGRESS_INTERVAL_SNAPSHOTS = 48
"""stage 1 每處理多少個 snapshot 就印一行 stderr 進度。"""

STOP_REASON_CODES: dict[str, int] = {
    "invalid_mask": 0,
    "invalid_step": 1,
    "outside_flow_domain": 2,
    "curvature_limit": 3,
    "s1_threshold": 4,
    "relative_strength_threshold": 5,
    "s1_not_monotonic": 6,
    "zero_step": 7,
    "max_curve_length": 8,
    "max_integration_steps": 9,
}
"""``TrapDetection`` 兩端停止原因的固定整數編碼；同一份表寫入 shard manifest。"""

STOP_CODE_TO_REASON: dict[int, str] = {code: reason for reason, code in STOP_REASON_CODES.items()}

_SHARD_ARTIFACT_NAMES = ("snapshots.npz", "detections.npz", "grid_reference.npz")
_SHARD_KNOWN_FILES = (*_SHARD_ARTIFACT_NAMES, "manifest.json", "manifest.sha256", "_SUCCESS")

_TRACK_ARTIFACT_NAMES = ("canonical_snapshots.npz", "track_observations.npz", "tracks.csv", "tracking_summary.json")
_TRACK_KNOWN_FILES = (*_TRACK_ARTIFACT_NAMES, "manifest.json", "manifest.sha256", "_SUCCESS")


def _require(condition: bool, message: str) -> None:
    """在跨月份契約不成立時立即停止，避免產生看似完整卻不可信的續跑產物。"""

    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# 小型共用工具：pass-cell 曲線交會、原子寫檔、manifest 驗證、git/版本 provenance
# ---------------------------------------------------------------------------


def _nearest_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """向量化版 ``metrics._nearest_axis_index``；輸入須已在軸範圍內。

    與單點版邏輯逐一對應（含遞減軸的處理），只是以 ``np.searchsorted`` 一次算完整批
    取樣點，避免在曲線很長時逐點呼叫 Python 函式。呼叫端必須先以
    ``axis`` 的 min/max 篩掉超出範圍的點，本函式不再重複邊界檢查。
    """

    increasing = axis[-1] > axis[0]
    ordered_axis = axis if increasing else axis[::-1]
    right = np.searchsorted(ordered_axis, values, side="left")
    right = np.clip(right, 0, ordered_axis.size - 1)
    left = np.clip(right - 1, 0, ordered_axis.size - 1)
    choose_left = np.abs(values - ordered_axis[left]) <= np.abs(ordered_axis[right] - values)
    ordered_index = np.where(choose_left, left, right)
    if increasing:
        return ordered_index.astype(np.int64)
    return (ordered_axis.size - 1 - ordered_index).astype(np.int64)


def curve_pass_cells(
    curve_points_m: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    mask_shape: tuple[int, int],
    *,
    max_step_m: float | None = None,
) -> np.ndarray:
    """回傳一條 TRAP 曲線經過的格點集合（``row * nx + col`` 排序後去重）。

    加密與最近格點規則與 :func:`traps.metrics.curve_intersects_mask` 完全相同
    （預設步長 ``0.5 * min(dx, dy)``，取樣點若落在網格外則忽略），因此對任意
    ``site_mask``，``curve_intersects_mask(curve, x, y, mask)`` 恆等於
    ``np.any(mask.flat[curve_pass_cells(curve, x, y, mask.shape)])``；本函式回傳的是
    未受任何單一站點遮罩限制的完整格點集合，供 stage 3 在不同站點分割下重複使用同一
    份 shard，不必重跑曲線積分。
    """

    x_axis, y_axis, dx, dy = _grid_axes(np.asarray(x_m), np.asarray(y_m), mask_shape)
    step = 0.5 * min(dx, dy) if max_step_m is None else float(max_step_m)
    _require(np.isfinite(step) and 0.0 < step <= 0.5 * min(dx, dy) + 1.0e-9, "曲線取樣步長不可大於半格距")
    dense = densify_polyline(np.asarray(curve_points_m, dtype=np.float64), step)
    x_values = dense[:, 0]
    y_values = dense[:, 1]
    x_low, x_high = (x_axis[0], x_axis[-1]) if x_axis[-1] > x_axis[0] else (x_axis[-1], x_axis[0])
    y_low, y_high = (y_axis[0], y_axis[-1]) if y_axis[-1] > y_axis[0] else (y_axis[-1], y_axis[0])
    in_bounds = (x_values >= x_low) & (x_values <= x_high) & (y_values >= y_low) & (y_values <= y_high)
    if not np.any(in_bounds):
        return np.zeros(0, dtype=np.int64)
    cols = _nearest_indices(x_axis, x_values[in_bounds])
    rows = _nearest_indices(y_axis, y_values[in_bounds])
    nx = int(mask_shape[1])
    flat = rows * np.int64(nx) + cols
    return np.unique(flat)


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> Path:
    """把陣列寫到 ``.tmp-<name>`` 再原子改名，避免半寫檔案被 ``_SUCCESS`` 誤判完成。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{path.name}")
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)
    return path


def _atomic_write_json(path: Path, payload: Any) -> Path:
    """以臨時檔＋原子改名寫出 JSON，供 manifest 與其他關鍵產物共用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{path.name}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def _write_manifest_with_checksum(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    """寫出 manifest.json 及外部 manifest.sha256，與既有 CLI 產物格式一致。"""

    manifest_path = _atomic_write_json(output_dir / "manifest.json", payload)
    digest = _sha256_file(manifest_path)
    (output_dir / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="utf-8")
    return manifest_path


def _artifact_records(output_dir: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    """只對本命令已知會寫出的檔名建立 sha256/bytes 清單，避免掃到其他 stray 檔案。"""

    records: list[dict[str, Any]] = []
    for name in names:
        path = output_dir / name
        if path.is_file():
            records.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return records


def _verify_manifest_artifacts(output_dir: Path, manifest: Mapping[str, Any]) -> bool:
    """核對 manifest 記錄的每個 artifact 是否仍存在且 SHA-256 相符。"""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    for record in artifacts:
        if not isinstance(record, Mapping):
            return False
        path = output_dir / str(record.get("path", ""))
        if not path.is_file():
            return False
        if _sha256_file(path) != record.get("sha256"):
            return False
    return True


def _load_manifest(output_dir: Path) -> dict[str, Any] | None:
    """讀取 ``manifest.json``；缺檔或無法解析都回傳 ``None`` 而非拋出例外。"""

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _success_verified(output_dir: Path) -> dict[str, Any] | None:
    """若 ``_SUCCESS`` 存在且 manifest 完整驗證通過，回傳該 manifest；否則回傳 ``None``。"""

    if not (output_dir / "_SUCCESS").is_file():
        return None
    manifest = _load_manifest(output_dir)
    if manifest is None or not _verify_manifest_artifacts(output_dir, manifest):
        return None
    return manifest


def _clean_known_files(output_dir: Path, known_names: Sequence[str]) -> None:
    """只刪除本命令會寫出的已知檔名與殘留的 ``.tmp-*``，不清除其他使用者檔案。"""

    for name in known_names:
        candidate = output_dir / name
        if candidate.is_file():
            candidate.unlink()
    for tmp_candidate in output_dir.glob(".tmp-*"):
        if tmp_candidate.is_file():
            tmp_candidate.unlink()


def _prepare_output_dir(output_dir: Path, *, resume: bool, known_names: Sequence[str]) -> dict[str, Any] | None:
    """依 --resume 契約準備輸出目錄，回傳可略過重算的驗證後 manifest（若有）。

    沒有 ``--resume`` 時要求目錄新建或空；有 ``--resume`` 時若 ``_SUCCESS`` 已驗證通過
    就回傳該 manifest 供呼叫端直接略過，否則清掉已知檔案重跑。
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return None
    if not output_dir.is_dir():
        raise ValueError(f"輸出路徑不是目錄: {output_dir}")
    if resume:
        verified = _success_verified(output_dir)
        if verified is not None:
            return verified
        _clean_known_files(output_dir, known_names)
        return None
    if any(output_dir.iterdir()):
        raise ValueError(f"輸出目錄已有內容，請改用新的 --output-dir 或加上 --resume: {output_dir}")
    return None


def _git_info(package_root: Path) -> dict[str, Any]:
    """回傳 ``git rev-parse HEAD`` 與 dirty 狀態；非 git repo 或指令不存在都回傳 None。"""

    commit: str | None = None
    dirty: bool | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    if commit is not None:
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=package_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _package_version() -> str | None:
    """回傳已安裝套件版本；未以套件形式安裝（例如純原始碼樹）時回傳 None。"""

    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("traps-analysis")
    except Exception:  # noqa: BLE001 - 版本資訊只供 manifest 參考，任何失敗都不阻斷執行
        return None


def _time_iso(time_utc_ns: int) -> str:
    """把 UTC nanosecond 整數轉成 ISO-8601 字串。"""

    timestamp = _datetime.datetime.fromtimestamp(int(time_utc_ns) / 1.0e9, tz=_datetime.UTC)
    return timestamp.isoformat(timespec="seconds")


def _package_root() -> Path:
    """回傳本套件原始碼樹的根目錄（供 git 指令的 cwd 使用）。"""

    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Stage 1：detect-shards
# ---------------------------------------------------------------------------


def _add_shard_source_arguments(parser: argparse.ArgumentParser) -> None:
    """加入單一 domain-month 來源參數；與既有 detect/render 使用相同旗標名稱。"""

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--month-dir", type=Path, help="單一 OCM surface 月份目錄")
    source.add_argument("--surface-root", type=Path, help="OCM surface 根目錄")
    parser.add_argument("--domain-id", help="surface-root 下的 OCM flow domain ID")
    parser.add_argument("--month", help="surface-root 下的 YYYYMM 月份")
    parser.add_argument("--ocean-mask", type=Path, help="analysis ocean mask sidecar .npy")
    parser.add_argument("--allow-trial-pilot", action="store_true", help="明示允許 trial_partial_month")
    parser.add_argument("--study-design", type=Path, default=DEFAULT_STUDY_DESIGN_PATH, help="四區域五站點研究設計 JSON")


def _resolve_shard_radius_and_threshold(args: argparse.Namespace) -> tuple[int, float]:
    """把共用 detection 參數解析器的輸出收斂成 detect-shards 需要的單一 radius/threshold。

    detect-shards 只處理一個 sensitivity case；共用解析器保留的 ``--sensitivity``／
    多組門檻旗標在這裡明確拒絕，避免使用者誤以為本命令也會展開多個 case。
    """

    if getattr(args, "sensitivity", False):
        raise ValueError("detect-shards 一次只產生一個 shard；不支援 --sensitivity，請個別呼叫")
    if args.core_thresholds_sinv is not None:
        raise ValueError("detect-shards 不支援 --core-thresholds-sinv；請用 --min-core-s1-sinv 指定單一門檻")
    radii = args.smoothing_radius_cells or [0]
    if len(radii) != 1:
        raise ValueError("detect-shards 只接受單一 --smoothing-radius-cells 值")
    radius = int(radii[0])
    if radius not in (0, 1, 2, 3):
        raise ValueError("smoothing radius 只能是 0、1、2 或 3")
    threshold = float(args.min_core_s1_sinv) if args.min_core_s1_sinv is not None else float(DetectionConfig().min_core_s1_sinv)
    return radius, threshold


def run_detect_shards(args: argparse.Namespace) -> int:
    """流式檢出單一 domain-month，寫出可續跑、可驗證的 stage-1 shard。"""

    start = _time.monotonic()
    output_dir = Path(args.output_dir)
    skip_manifest = _prepare_output_dir(output_dir, resume=bool(args.resume), known_names=_SHARD_KNOWN_FILES)
    if skip_manifest is not None:
        summary = {
            "operation": "detect-shards",
            "output_dir": str(output_dir),
            "resumed": True,
            "skipped": True,
            "domain_id": skip_manifest.get("domain_id"),
            "month": skip_manifest.get("month"),
            "snapshot_count": skip_manifest.get("snapshot_count"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    month = _open_month_from_args(args)
    if not month.pilot_only and month.analysis_mask_status != "provided":
        raise ValueError("正式 detect-shards 必須提供有 metadata 的 --ocean-mask 海岸 sidecar")

    radius, threshold = _resolve_shard_radius_and_threshold(args)
    config = _make_config(args, radius, threshold)
    study_design_path = Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH)
    param_id = args.param_id or f"smoothing_r{radius}_threshold_{threshold:.3e}"

    snapshot_count = month.snapshot_count
    snapshot_index_col: list[int] = []
    time_utc_ns_col: list[int] = []
    valid_cell_count_col: list[int] = []
    expected_valid_cell_count_col: list[int] = []
    coverage_fraction_col: list[float] = []
    coverage_pass_col: list[bool] = []
    detection_count_col: list[int] = []
    qualified_count_col: list[int] = []
    valid_mask_packed_rows: list[np.ndarray] = []

    det_snapshot_index: list[int] = []
    det_time_utc_ns: list[int] = []
    det_detection_index: list[int] = []
    det_core_row: list[int] = []
    det_core_col: list[int] = []
    det_core_x_m: list[float] = []
    det_core_y_m: list[float] = []
    det_core_s1_sinv: list[float] = []
    det_curve_length_m: list[float] = []
    det_curve_mean_s1_sinv: list[float] = []
    det_curve_mean_abs_s1_sinv: list[float] = []
    det_is_qualified: list[bool] = []
    det_forward_stop_code: list[int] = []
    det_backward_stop_code: list[int] = []
    curve_offsets: list[int] = [0]
    curve_points_chunks: list[np.ndarray] = []
    curve_strength_chunks: list[np.ndarray] = []
    pass_cell_offsets: list[int] = [0]
    pass_cell_chunks: list[np.ndarray] = []

    expected_valid_cells = int(np.count_nonzero(month.analysis_ocean_mask))
    min_coverage = float(args.min_coverage)
    progress_interval = DEFAULT_PROGRESS_INTERVAL_SNAPSHOTS

    grid_written = False
    grid_arrays: dict[str, np.ndarray] | None = None

    for index in range(snapshot_count):
        snapshot = month.snapshot(index)
        if not grid_written:
            grid_arrays = {
                "lon": np.asarray(snapshot.lon, dtype=np.float64),
                "lat": np.asarray(snapshot.lat, dtype=np.float64),
                "x_m_axis": np.asarray(snapshot.x_m)[0, :].astype(np.float64),
                "y_m_axis": np.asarray(snapshot.y_m)[:, 0].astype(np.float64),
                "cell_area_m2": np.asarray(snapshot.cell_area_m2, dtype=np.float64),
                "analysis_ocean_mask": np.asarray(month.analysis_ocean_mask, dtype=bool),
            }
            grid_written = True

        result = detect_snapshot(snapshot.u, snapshot.v, snapshot.valid, snapshot.x_m, snapshot.y_m, config)
        valid_cells = int(np.count_nonzero(snapshot.valid))
        coverage = (valid_cells / expected_valid_cells) if expected_valid_cells else 0.0
        coverage_pass = bool(coverage >= min_coverage)
        qualified_count = sum(1 for detection in result.detections if detection.is_qualified)

        snapshot_index_col.append(int(snapshot.index))
        time_utc_ns_col.append(int(snapshot.time_utc_ns))
        valid_cell_count_col.append(valid_cells)
        expected_valid_cell_count_col.append(expected_valid_cells)
        coverage_fraction_col.append(coverage)
        coverage_pass_col.append(coverage_pass)
        detection_count_col.append(len(result.detections))
        qualified_count_col.append(qualified_count)
        valid_mask_packed_rows.append(np.packbits(np.asarray(snapshot.valid, dtype=bool).reshape(-1)))

        for detection_index, detection in enumerate(result.detections):
            det_snapshot_index.append(int(snapshot.index))
            det_time_utc_ns.append(int(snapshot.time_utc_ns))
            det_detection_index.append(detection_index)
            det_core_row.append(int(detection.core_index_ij[0]))
            det_core_col.append(int(detection.core_index_ij[1]))
            det_core_x_m.append(float(detection.core_xy_m[0]))
            det_core_y_m.append(float(detection.core_xy_m[1]))
            det_core_s1_sinv.append(float(detection.core_s1_sinv))
            det_curve_length_m.append(float(detection.curve_length_m))
            det_curve_mean_s1_sinv.append(float(detection.curve_mean_s1_sinv))
            det_curve_mean_abs_s1_sinv.append(float(detection.curve_mean_abs_s1_sinv))
            det_is_qualified.append(bool(detection.is_qualified))
            det_forward_stop_code.append(STOP_REASON_CODES[detection.forward_stop_reason])
            det_backward_stop_code.append(STOP_REASON_CODES[detection.backward_stop_reason])

            points = np.asarray(detection.curve_points_m, dtype=np.float32)
            strengths = np.asarray(detection.curve_s1_sinv, dtype=np.float32)
            curve_points_chunks.append(points)
            curve_strength_chunks.append(strengths)
            curve_offsets.append(curve_offsets[-1] + points.shape[0])

            cells = curve_pass_cells(detection.curve_points_m, snapshot.x_m, snapshot.y_m, snapshot.valid.shape).astype(np.int32)
            pass_cell_chunks.append(cells)
            pass_cell_offsets.append(pass_cell_offsets[-1] + cells.shape[0])

        done = index + 1
        if done % progress_interval == 0 or done == snapshot_count:
            elapsed = _time.monotonic() - start
            rate = elapsed / done if done else 0.0
            eta = rate * (snapshot_count - done)
            print(
                json.dumps(
                    {
                        "stage": "detect-shards",
                        "domain_id": month.domain_id,
                        "month": month.month,
                        "done": done,
                        "total": snapshot_count,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )

    valid_mask_packed = (
        np.stack(valid_mask_packed_rows, axis=0)
        if valid_mask_packed_rows
        else np.zeros((0, 0), dtype=np.uint8)
    )

    _atomic_write_npz(
        output_dir / "snapshots.npz",
        snapshot_index=np.asarray(snapshot_index_col, dtype=np.int64),
        time_utc_ns=np.asarray(time_utc_ns_col, dtype=np.int64),
        valid_cell_count=np.asarray(valid_cell_count_col, dtype=np.int64),
        expected_valid_cell_count=np.asarray(expected_valid_cell_count_col, dtype=np.int64),
        coverage_fraction=np.asarray(coverage_fraction_col, dtype=np.float64),
        coverage_pass=np.asarray(coverage_pass_col, dtype=bool),
        detection_count=np.asarray(detection_count_col, dtype=np.int64),
        qualified_count=np.asarray(qualified_count_col, dtype=np.int64),
        valid_mask_packed=valid_mask_packed,
    )

    total_curve_points = curve_offsets[-1]
    curve_points_m = (
        np.concatenate(curve_points_chunks, axis=0)
        if curve_points_chunks
        else np.zeros((0, 2), dtype=np.float32)
    )
    curve_s1_sinv = (
        np.concatenate(curve_strength_chunks, axis=0) if curve_strength_chunks else np.zeros((0,), dtype=np.float32)
    )
    pass_cells = (
        np.concatenate(pass_cell_chunks, axis=0) if pass_cell_chunks else np.zeros((0,), dtype=np.int32)
    )
    assert total_curve_points == curve_points_m.shape[0]

    _atomic_write_npz(
        output_dir / "detections.npz",
        snapshot_index=np.asarray(det_snapshot_index, dtype=np.int64),
        time_utc_ns=np.asarray(det_time_utc_ns, dtype=np.int64),
        detection_index=np.asarray(det_detection_index, dtype=np.int64),
        core_row=np.asarray(det_core_row, dtype=np.int64),
        core_col=np.asarray(det_core_col, dtype=np.int64),
        core_x_m=np.asarray(det_core_x_m, dtype=np.float64),
        core_y_m=np.asarray(det_core_y_m, dtype=np.float64),
        core_s1_sinv=np.asarray(det_core_s1_sinv, dtype=np.float64),
        curve_length_m=np.asarray(det_curve_length_m, dtype=np.float64),
        curve_mean_s1_sinv=np.asarray(det_curve_mean_s1_sinv, dtype=np.float64),
        curve_mean_abs_s1_sinv=np.asarray(det_curve_mean_abs_s1_sinv, dtype=np.float64),
        is_qualified=np.asarray(det_is_qualified, dtype=bool),
        forward_stop_code=np.asarray(det_forward_stop_code, dtype=np.int64),
        backward_stop_code=np.asarray(det_backward_stop_code, dtype=np.int64),
        curve_offsets=np.asarray(curve_offsets, dtype=np.int64),
        curve_points_m=curve_points_m,
        curve_s1_sinv=curve_s1_sinv,
        pass_cell_offsets=np.asarray(pass_cell_offsets, dtype=np.int64),
        pass_cells=pass_cells,
    )

    assert grid_arrays is not None
    _atomic_write_npz(output_dir / "grid_reference.npz", **grid_arrays)

    source = _source_provenance(month, study_design_path)
    runtime_seconds = _time.monotonic() - start
    manifest = {
        "operation": "detect-shards",
        "schema_version": SHARD_SCHEMA_VERSION,
        "param_id": param_id,
        "detection_config": config.to_manifest_dict(),
        "min_coverage": min_coverage,
        "source": source,
        "domain_id": month.domain_id,
        "month": month.month,
        "state_label": month.state_label,
        "science_status": _science_status(month),
        "snapshot_count": snapshot_count,
        "detection_count": len(det_snapshot_index),
        "time_range": {
            "start_time_utc_ns": time_utc_ns_col[0] if time_utc_ns_col else None,
            "end_time_utc_ns": time_utc_ns_col[-1] if time_utc_ns_col else None,
        },
        "stop_code_table": STOP_REASON_CODES,
        "runtime_seconds": runtime_seconds,
        "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds"),
        "package_version": _package_version(),
        **_git_info(_package_root()),
    }
    manifest["artifacts"] = _artifact_records(output_dir, _SHARD_ARTIFACT_NAMES)
    _write_manifest_with_checksum(output_dir, manifest)
    (output_dir / "_SUCCESS").write_text(
        json.dumps({"created_at_utc": manifest["created_at_utc"]}, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = {
        "operation": "detect-shards",
        "output_dir": str(output_dir),
        "resumed": False,
        "skipped": False,
        "domain_id": month.domain_id,
        "month": month.month,
        "state_label": month.state_label,
        "snapshot_count": snapshot_count,
        "detection_count": len(det_snapshot_index),
        "runtime_seconds": runtime_seconds,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# Stage 2：track-period
# ---------------------------------------------------------------------------


def _expand_month_token(token: str) -> list[str]:
    """展開單一月份或 ``YYYYMM-YYYYMM`` 範圍為遞增的 ``YYYYMM`` 清單。"""

    token = token.strip()
    if "-" in token:
        start_token, end_token = token.split("-", 1)
        _require(len(start_token) == 6 and start_token.isdigit(), f"月份範圍格式錯誤: {token!r}")
        _require(len(end_token) == 6 and end_token.isdigit(), f"月份範圍格式錯誤: {token!r}")
        start_year, start_month = int(start_token[:4]), int(start_token[4:6])
        end_year, end_month = int(end_token[:4]), int(end_token[4:6])
        _require(1 <= start_month <= 12 and 1 <= end_month <= 12, f"月份範圍月值不合法: {token!r}")
        _require((start_year, start_month) <= (end_year, end_month), f"月份範圍起訖顛倒: {token!r}")
        months: list[str] = []
        year, month = start_year, start_month
        while (year, month) <= (end_year, end_month):
            months.append(f"{year:04d}{month:02d}")
            month += 1
            if month > 12:
                month = 1
                year += 1
        return months
    _require(len(token) == 6 and token.isdigit(), f"月份必須是 YYYYMM: {token!r}")
    return [token]


def _resolve_requested_months(args: argparse.Namespace) -> list[str]:
    """由 ``--months``（可含範圍）或 ``--month``（逐一列舉）解析遞增去重的月份清單。"""

    tokens = args.months if args.months is not None else args.month
    _require(bool(tokens), "必須提供 --months 或 --month")
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(_expand_month_token(token))
    months = sorted(set(expanded))
    _require(bool(months), "月份清單不可為空")
    return months


def _load_shard(shards_root: Path, month: str) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    """讀取並驗證一個月份 shard，回傳 manifest、snapshots 陣列與 shard 目錄。"""

    shard_dir = shards_root / month
    manifest = _success_verified(shard_dir)
    if manifest is None:
        raise ValueError(f"shard 未完成或驗證失敗（缺少 _SUCCESS 或 SHA-256 不符）: {shard_dir}")
    _require(manifest.get("schema_version") == SHARD_SCHEMA_VERSION, f"shard schema_version 不符: {shard_dir}")
    _require(manifest.get("month") == month, f"shard month 與目錄名稱不符: {shard_dir}")
    snapshots_path = shard_dir / "snapshots.npz"
    with np.load(snapshots_path) as archive:
        snapshots = {name: np.array(archive[name]) for name in archive.files}
    return manifest, snapshots, shard_dir


def _validate_shard_consistency(manifests: Mapping[str, Mapping[str, Any]], *, domain_id: str) -> None:
    """核對所有已載入 shard 的 domain/param/detection config/coverage/state_label 一致。"""

    first_month = next(iter(manifests))
    first = manifests[first_month]
    for month, manifest in manifests.items():
        if manifest.get("domain_id") != domain_id:
            raise ValueError(f"shard domain_id 與 --domain-id 不符: {month}")
        if manifest.get("param_id") != first.get("param_id"):
            raise ValueError(f"shard param_id 不一致（{first_month} 與 {month}）")
        if manifest.get("detection_config") != first.get("detection_config"):
            raise ValueError(f"shard detection_config 不一致（{first_month} 與 {month}）")
        if manifest.get("min_coverage") != first.get("min_coverage"):
            raise ValueError(f"shard min_coverage 不一致（{first_month} 與 {month}）")
        if manifest.get("state_label") != first.get("state_label"):
            raise ValueError("track-period 不可混用 PILOT_ONLY 與正式 shard")


def _build_canonical_axis(
    months: Sequence[str], snapshots_by_month: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    """串接月份、依 UTC 穩定排序，重複 UTC 保留較晚月份的一列。"""

    month_codes: list[int] = []
    snapshot_indices: list[int] = []
    times: list[int] = []
    coverage_fraction: list[float] = []
    coverage_pass: list[bool] = []
    for month in months:
        arrays = snapshots_by_month[month]
        count = arrays["time_utc_ns"].shape[0]
        month_codes.extend([int(month)] * count)
        snapshot_indices.extend(int(value) for value in arrays["snapshot_index"])
        times.extend(int(value) for value in arrays["time_utc_ns"])
        coverage_fraction.extend(float(value) for value in arrays["coverage_fraction"])
        coverage_pass.extend(bool(value) for value in arrays["coverage_pass"])

    month_codes_array = np.asarray(month_codes, dtype=np.int64)
    snapshot_indices_array = np.asarray(snapshot_indices, dtype=np.int64)
    times_array = np.asarray(times, dtype=np.int64)
    coverage_fraction_array = np.asarray(coverage_fraction, dtype=np.float64)
    coverage_pass_array = np.asarray(coverage_pass, dtype=bool)

    total_rows = int(times_array.shape[0])
    order = np.argsort(times_array, kind="stable")
    sorted_times = times_array[order]
    # 相鄰重複只保留最後一次出現：由於串接順序已是月份時序（月內時間本身遞增），
    # 穩定排序後同一 UTC 的多筆之間仍維持原始相對順序，較晚月份必排在較後面。
    keep = np.ones(sorted_times.shape[0], dtype=bool)
    if sorted_times.shape[0] > 1:
        keep[:-1] = sorted_times[:-1] != sorted_times[1:]
    dropped_order = order[~keep]
    canonical_order = order[keep]

    canonical_time = times_array[canonical_order]
    canonical_month = month_codes_array[canonical_order]
    canonical_snapshot_index = snapshot_indices_array[canonical_order]
    canonical_coverage_fraction = coverage_fraction_array[canonical_order]
    canonical_coverage_pass = coverage_pass_array[canonical_order]

    duplicates_removed = [
        {
            "month": int(month_codes_array[index]),
            "snapshot_index": int(snapshot_indices_array[index]),
            "time_utc_ns": int(times_array[index]),
        }
        for index in dropped_order.tolist()
    ]

    per_month_kept: dict[str, int] = {month: 0 for month in months}
    for month_value in canonical_month.tolist():
        per_month_kept[f"{int(month_value):06d}"] += 1

    return {
        "canonical_index": np.arange(canonical_time.shape[0], dtype=np.int64),
        "time_utc_ns": canonical_time,
        "month": canonical_month,
        "snapshot_index": canonical_snapshot_index,
        "coverage_fraction": canonical_coverage_fraction,
        "coverage_pass": canonical_coverage_pass,
        "total_rows": total_rows,
        "unique_times": int(canonical_time.shape[0]),
        "duplicates_removed_count": int(dropped_order.shape[0]),
        "duplicates_removed": duplicates_removed,
        "per_month_kept_count": per_month_kept,
        "canonical_order": canonical_order,
    }


def _compute_gaps(canonical_time: np.ndarray, nominal_step_ns: int) -> dict[str, Any]:
    """回傳連續 canonical 時間差超過名目取樣間隔的缺口統計。"""

    if canonical_time.shape[0] < 2:
        return {"gap_count": 0, "max_gap_hours": 0.0, "gaps_over_1h": []}
    diffs = np.diff(canonical_time)
    gap_mask = diffs > nominal_step_ns
    gap_indices = np.nonzero(gap_mask)[0]
    max_gap_hours = float(np.max(diffs) / 3_600_000_000_000.0) if diffs.size else 0.0
    gaps_over_1h = []
    one_hour_ns = 3_600_000_000_000
    for index in gap_indices.tolist():
        gap_ns = int(diffs[index])
        if gap_ns > one_hour_ns:
            gaps_over_1h.append(
                {
                    "start_time_utc_ns": int(canonical_time[index]),
                    "end_time_utc_ns": int(canonical_time[index + 1]),
                    "gap_hours": gap_ns / 3_600_000_000_000.0,
                }
            )
    return {
        "gap_count": int(gap_indices.shape[0]),
        "max_gap_hours": max_gap_hours,
        "gaps_over_1h": gaps_over_1h,
    }


def _rebuild_qualified_detections(
    detections: Mapping[str, np.ndarray], snapshot_index: int
) -> list[tuple[int, TrapDetection]]:
    """從 detections.npz 陣列重建某個 snapshot 內所有合格 ``TrapDetection``。

    回傳 ``(detection_index, TrapDetection)`` list；只重建 ``is_qualified`` 為真的列，
    因為 :class:`~traps.tracking.StreamingTracker` 在 ``require_qualified=True`` 下只會
    使用合格觀測，重建未合格曲線只是白白複製陣列。
    """

    rows = np.nonzero(detections["snapshot_index"] == snapshot_index)[0]
    result: list[tuple[int, TrapDetection]] = []
    for row in rows.tolist():
        if not bool(detections["is_qualified"][row]):
            continue
        start, end = int(detections["curve_offsets"][row]), int(detections["curve_offsets"][row + 1])
        points = detections["curve_points_m"][start:end]
        strengths = detections["curve_s1_sinv"][start:end]
        detection = TrapDetection(
            core_index_ij=(int(detections["core_row"][row]), int(detections["core_col"][row])),
            core_xy_m=(float(detections["core_x_m"][row]), float(detections["core_y_m"][row])),
            core_s1_sinv=float(detections["core_s1_sinv"][row]),
            curve_points_m=points,
            curve_s1_sinv=strengths,
            curve_length_m=float(detections["curve_length_m"][row]),
            curve_mean_s1_sinv=float(detections["curve_mean_s1_sinv"][row]),
            curve_mean_abs_s1_sinv=float(detections["curve_mean_abs_s1_sinv"][row]),
            forward_stop_reason=STOP_CODE_TO_REASON[int(detections["forward_stop_code"][row])],
            backward_stop_reason=STOP_CODE_TO_REASON[int(detections["backward_stop_code"][row])],
            is_qualified=True,
        )
        result.append((int(detections["detection_index"][row]), detection))
    return result


def run_track_period(args: argparse.Namespace) -> int:
    """讀入一個 domain 的多個月份 shard，建立正式研究期時間軸並跨月份追蹤。"""

    start = _time.monotonic()
    output_dir = Path(args.output_dir)
    skip_manifest = _prepare_output_dir(output_dir, resume=bool(args.resume), known_names=_TRACK_KNOWN_FILES)
    if skip_manifest is not None:
        summary = {
            "operation": "track-period",
            "output_dir": str(output_dir),
            "resumed": True,
            "skipped": True,
            "domain_id": skip_manifest.get("domain_id"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    shards_root = Path(args.shards_root)
    domain_id = str(args.domain_id)
    requested_months = _resolve_requested_months(args)

    available_months: list[str] = []
    missing_months: list[str] = []
    for month in requested_months:
        if (shards_root / month).is_dir():
            available_months.append(month)
        else:
            missing_months.append(month)
    if missing_months and not args.allow_missing_months:
        raise ValueError(f"缺少請求的月份 shard，需加 --allow-missing-months 才可略過: {missing_months}")
    _require(bool(available_months), "沒有任何可用的月份 shard")

    manifests: dict[str, dict[str, Any]] = {}
    snapshots_by_month: dict[str, dict[str, np.ndarray]] = {}
    shard_dirs: dict[str, Path] = {}
    for month in available_months:
        manifest, snapshots, shard_dir = _load_shard(shards_root, month)
        manifests[month] = manifest
        snapshots_by_month[month] = snapshots
        shard_dirs[month] = shard_dir
    _validate_shard_consistency(manifests, domain_id=domain_id)

    first_manifest = manifests[available_months[0]]
    param_id = args.param_id or first_manifest.get("param_id")
    state_label = first_manifest.get("state_label")

    canonical = _build_canonical_axis(available_months, snapshots_by_month)
    nominal_step_ns = round(float(args.nominal_time_step_hours) * 3_600.0 * 1_000_000_000.0)
    gaps = _compute_gaps(canonical["time_utc_ns"], nominal_step_ns)

    _atomic_write_npz(
        output_dir / "canonical_snapshots.npz",
        canonical_index=canonical["canonical_index"],
        time_utc_ns=canonical["time_utc_ns"],
        month=canonical["month"],
        snapshot_index=canonical["snapshot_index"],
        coverage_fraction=canonical["coverage_fraction"],
        coverage_pass=canonical["coverage_pass"],
    )

    tracking_config = TrackingConfig(
        max_core_distance_m=float(args.tracking_max_core_distance_m),
        max_curve_distance_m=float(args.tracking_max_curve_distance_m),
        max_time_gap_seconds=float(args.tracking_max_time_gap_hours) * 3_600.0,
        max_missing_frames=int(args.tracking_max_missing_frames),
        curve_sample_step_m=float(args.tracking_curve_sample_step_m),
        require_qualified=True,
        nominal_time_step_seconds=float(args.nominal_time_step_hours) * 3_600.0,
    )
    tracker = StreamingTracker(tracking_config)

    completed_summaries: list[Any] = []
    current_month: str | None = None
    current_detections: dict[str, np.ndarray] | None = None
    canonical_len = int(canonical["time_utc_ns"].shape[0])

    for row in range(canonical_len):
        month_value = f"{int(canonical['month'][row]):06d}"
        snapshot_index = int(canonical["snapshot_index"][row])
        time_utc_ns = int(canonical["time_utc_ns"][row])
        coverage_pass = bool(canonical["coverage_pass"][row])

        if month_value != current_month:
            if current_month is not None:
                elapsed = _time.monotonic() - start
                print(
                    json.dumps(
                        {
                            "stage": "track-period",
                            "domain_id": domain_id,
                            "finished_month": current_month,
                            "elapsed_seconds": elapsed,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            current_month = month_value
            shard_dir = shard_dirs[month_value]
            with np.load(shard_dir / "detections.npz") as archive:
                current_detections = {name: np.array(archive[name]) for name in archive.files}

        if not coverage_pass:
            completed_summaries.extend(tracker.push(TrackingFrame(time_utc_ns=time_utc_ns, available=False)))
        else:
            assert current_detections is not None
            qualified = _rebuild_qualified_detections(current_detections, snapshot_index)
            frame_detections = tuple(detection for _, detection in qualified)
            detection_keys = tuple((month_value, snapshot_index, detection_index) for detection_index, _ in qualified)
            frame = TrackingFrame(time_utc_ns=time_utc_ns, detections=frame_detections, available=True)
            completed_summaries.extend(tracker.push(frame, detection_keys=detection_keys))

    completed_summaries.extend(tracker.finish())
    if current_month is not None:
        elapsed = _time.monotonic() - start
        print(
            json.dumps(
                {"stage": "track-period", "domain_id": domain_id, "finished_month": current_month, "elapsed_seconds": elapsed},
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    track_id_col: list[int] = []
    obs_time_col: list[int] = []
    obs_month_col: list[int] = []
    obs_snapshot_index_col: list[int] = []
    obs_detection_index_col: list[int] = []
    obs_core_x_col: list[float] = []
    obs_core_y_col: list[float] = []
    obs_core_s1_col: list[float] = []
    track_rows: list[dict[str, Any]] = []
    termination_counts: dict[str, int] = {}
    censored_count = 0
    uncensored_count = 0
    uncensored_lifetimes: list[float] = []

    for summary in sorted(completed_summaries, key=lambda item: item.track_id):
        record = summary.to_record()
        record["start_time_utc_iso"] = _time_iso(record["start_time_utc_ns"])
        record["end_time_utc_iso"] = _time_iso(record["end_time_utc_ns"])
        record["first_core_x_m"] = float(summary.core_xy_m[0][0])
        record["first_core_y_m"] = float(summary.core_xy_m[0][1])
        record["last_core_x_m"] = float(summary.core_xy_m[-1][0])
        record["last_core_y_m"] = float(summary.core_xy_m[-1][1])
        record["min_core_s1_sinv"] = float(min(summary.core_s1_sinv))
        track_rows.append(record)

        termination_counts[summary.termination_reason] = termination_counts.get(summary.termination_reason, 0) + 1
        is_censored = bool(summary.left_censored or summary.right_censored)
        if is_censored:
            censored_count += 1
        else:
            uncensored_count += 1
            uncensored_lifetimes.append(float(summary.lifetime_seconds))

        for key, time_value, core_xy, core_s1 in zip(
            summary.observation_keys, summary.observation_times_utc_ns, summary.core_xy_m, summary.core_s1_sinv
        ):
            obs_month, obs_snapshot_index, obs_detection_index = key
            track_id_col.append(int(summary.track_id))
            obs_time_col.append(int(time_value))
            obs_month_col.append(int(obs_month))
            obs_snapshot_index_col.append(int(obs_snapshot_index))
            obs_detection_index_col.append(int(obs_detection_index))
            obs_core_x_col.append(float(core_xy[0]))
            obs_core_y_col.append(float(core_xy[1]))
            obs_core_s1_col.append(float(core_s1))

    obs_order = np.lexsort((np.asarray(obs_time_col, dtype=np.int64), np.asarray(track_id_col, dtype=np.int64)))
    _atomic_write_npz(
        output_dir / "track_observations.npz",
        track_id=np.asarray(track_id_col, dtype=np.int64)[obs_order],
        time_utc_ns=np.asarray(obs_time_col, dtype=np.int64)[obs_order],
        month=np.asarray(obs_month_col, dtype=np.int64)[obs_order],
        snapshot_index=np.asarray(obs_snapshot_index_col, dtype=np.int64)[obs_order],
        detection_index=np.asarray(obs_detection_index_col, dtype=np.int64)[obs_order],
        core_x_m=np.asarray(obs_core_x_col, dtype=np.float64)[obs_order],
        core_y_m=np.asarray(obs_core_y_col, dtype=np.float64)[obs_order],
        core_s1_sinv=np.asarray(obs_core_s1_col, dtype=np.float64)[obs_order],
    )

    tracks_csv_path = output_dir / "tracks.csv"
    track_fieldnames = (
        "track_id",
        "observation_count",
        "start_time_utc_ns",
        "end_time_utc_ns",
        "start_time_utc_iso",
        "end_time_utc_iso",
        "duration_seconds",
        "lifetime_seconds",
        "termination_reason",
        "left_censored",
        "right_censored",
        "missing_frame_count",
        "quality_flags",
        "first_core_x_m",
        "first_core_y_m",
        "last_core_x_m",
        "last_core_y_m",
        "min_core_s1_sinv",
    )
    with tracks_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(track_fieldnames), extrasaction="ignore")
        writer.writeheader()
        for record in track_rows:
            row = dict(record)
            row["quality_flags"] = json.dumps(row["quality_flags"], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)

    def _quantile_or_none(values: list[float], q: float) -> float | None:
        return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None

    tracking_summary = {
        "track_count": len(track_rows),
        "termination_reason_counts": termination_counts,
        "censored_track_count": censored_count,
        "uncensored_track_count": uncensored_count,
        "uncensored_lifetime_seconds_quantiles": {
            "p10": _quantile_or_none(uncensored_lifetimes, 0.10),
            "p25": _quantile_or_none(uncensored_lifetimes, 0.25),
            "p50": _quantile_or_none(uncensored_lifetimes, 0.50),
            "p75": _quantile_or_none(uncensored_lifetimes, 0.75),
            "p90": _quantile_or_none(uncensored_lifetimes, 0.90),
        },
    }
    _atomic_write_json(output_dir / "tracking_summary.json", tracking_summary)

    shard_manifest_records = [
        {
            "month": month,
            "manifest_path": str(shard_dirs[month] / "manifest.json"),
            "manifest_sha256": _sha256_file(shard_dirs[month] / "manifest.json"),
        }
        for month in available_months
    ]

    runtime_seconds = _time.monotonic() - start
    manifest = {
        "operation": "track-period",
        "schema_version": TRACK_SCHEMA_VERSION,
        "param_id": param_id,
        "domain_id": domain_id,
        "state_label": state_label,
        "tracking_config": tracking_config.to_manifest_dict(),
        "nominal_time_step_hours": float(args.nominal_time_step_hours),
        "requested_months": requested_months,
        "available_months": available_months,
        "missing_months": missing_months,
        "canonical_axis": {
            "total_rows": canonical["total_rows"],
            "unique_times": canonical["unique_times"],
            "duplicates_removed_count": canonical["duplicates_removed_count"],
            "duplicates_removed": canonical["duplicates_removed"],
            "per_month_kept_count": canonical["per_month_kept_count"],
            **gaps,
        },
        "shard_manifests": shard_manifest_records,
        "track_count": len(track_rows),
        "runtime_seconds": runtime_seconds,
        "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds"),
        "package_version": _package_version(),
        **_git_info(_package_root()),
    }
    manifest["artifacts"] = _artifact_records(output_dir, _TRACK_ARTIFACT_NAMES)
    _write_manifest_with_checksum(output_dir, manifest)
    (output_dir / "_SUCCESS").write_text(
        json.dumps({"created_at_utc": manifest["created_at_utc"]}, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary_payload = {
        "operation": "track-period",
        "output_dir": str(output_dir),
        "resumed": False,
        "skipped": False,
        "domain_id": domain_id,
        "state_label": state_label,
        "available_months": available_months,
        "missing_months": missing_months,
        "track_count": len(track_rows),
        "runtime_seconds": runtime_seconds,
    }
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# argparse 註冊
# ---------------------------------------------------------------------------


def register_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """把 ``detect-shards`` 與 ``track-period`` 掛到既有的 `python -m traps` parser。

    只在此處新增子命令；既有 ``preflight``／``detect``／``summary``／
    ``literature-figures``／``render`` 的參數與行為完全不變。
    """

    detect_shards = subparsers.add_parser(
        "detect-shards", help="Stage 1：單一 domain-month 流式檢出，寫出可續跑的 shard"
    )
    _add_shard_source_arguments(detect_shards)
    detect_shards.add_argument("--output-dir", type=Path, required=True)
    detect_shards.add_argument("--param-id", help="shard 參數識別字串；未提供時由 radius/threshold 推導")
    detect_shards.add_argument("--resume", action="store_true", help="若已通過 _SUCCESS 驗證則略過；否則清理已知檔案重跑")
    _add_detection_arguments(detect_shards)
    detect_shards.set_defaults(handler=run_detect_shards)

    track_period = subparsers.add_parser(
        "track-period", help="Stage 2：合併一個 domain 的多月份 shard 並建立跨月份 track"
    )
    track_period.add_argument("--shards-root", type=Path, required=True)
    track_period.add_argument("--domain-id", required=True)
    track_period.add_argument("--output-dir", type=Path, required=True)
    month_group = track_period.add_mutually_exclusive_group(required=True)
    month_group.add_argument("--months", nargs="+", help="YYYYMM 或 YYYYMM-YYYYMM 範圍，可多個")
    month_group.add_argument("--month", nargs="+", help="逐一列舉 YYYYMM")
    track_period.add_argument("--param-id", help="覆寫輸出 manifest 的 param_id；預設沿用 shard 的 param_id")
    track_period.add_argument("--tracking-max-core-distance-m", type=float, default=5_000.0)
    track_period.add_argument("--tracking-max-curve-distance-m", type=float, default=10_000.0)
    track_period.add_argument("--tracking-max-time-gap-hours", type=float, default=3.0)
    track_period.add_argument("--tracking-max-missing-frames", type=int, default=1)
    track_period.add_argument("--tracking-curve-sample-step-m", type=float, default=500.0)
    track_period.add_argument("--nominal-time-step-hours", type=float, default=1.0)
    track_period.add_argument("--allow-missing-months", action="store_true")
    track_period.add_argument("--resume", action="store_true")
    track_period.set_defaults(handler=run_track_period)

    # Stage 3（跨月份／跨區彙整）在獨立模組 aggregate.py 實作；這裡只新增掛載，
    # 不改動 detect-shards／track-period 既有的參數或行為。
    from . import aggregate as _aggregate

    _aggregate.register_subcommands(subparsers)


__all__ = [
    "STOP_REASON_CODES",
    "STOP_CODE_TO_REASON",
    "SHARD_SCHEMA_VERSION",
    "TRACK_SCHEMA_VERSION",
    "curve_pass_cells",
    "register_subcommands",
    "run_detect_shards",
    "run_track_period",
]
