"""跨月份／跨區彙整的第三階段：``aggregate``。

Stage 1（``detect-shards``）與 Stage 2（``track-period``）分別把單一 domain-month
的逐時檢出、以及一個 domain 全期的跨月份 track 寫成可續跑、可驗證的 shard／track
artifact。本模組是第三階段：讀入已完成的 track-root（`traps.period.track.v1`）與
detect-root（`traps.period.shard.v1`）成果，核對四個流場區域共用同一組
`param_id`、`detection_config`、`tracking_config`、`min_coverage`、`state_label`，
再以 :func:`traps.sites.site_masks`（`domain_partition`，沿用四區 flow-domain 舊
定義）與站點 anchor 的 12.5 km receptor（PRIMARY 範圍）計算五站逐時出現率、核心
吸引強度與未截尾 track 壽命統計，並輸出可比較的逐格空間分布。

出現率一律用「有效快照分母」，不以 0 假裝資料缺口；核心與曲線通過只計「該快照
是否至少一次」，不重複計曲線交會次數。全部欄位保留 `PILOT_ONLY` 等
`state_label`／`science_status`，輸出只代表模型內部「表層短期吸引潛勢」，不是
實測海廢濃度或污染來源。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
import time as _time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cli import _resolve_coastline_geojson, _sha256_file
from .period import (
    SHARD_SCHEMA_VERSION,
    TRACK_SCHEMA_VERSION,
    _artifact_records,
    _atomic_write_json,
    _atomic_write_npz,
    _git_info,
    _nearest_indices,
    _package_root,
    _package_version,
    _prepare_output_dir,
    _success_verified,
    _write_manifest_with_checksum,
)
from .sites import DEFAULT_STUDY_DESIGN_PATH, EARTH_RADIUS_M, load_study_design, site_masks

AGGREGATE_SCHEMA_VERSION = "traps.period.aggregate.v1"

SEASON_BY_MONTH: dict[int, str] = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASONS = ("DJF", "MAM", "JJA", "SON")
SITE_IDS = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")
SCOPES = ("receptor", "domain_partition")
LIFETIME_THRESHOLD_HOURS = (6.0, 12.0, 24.0)
SECONDS_PER_DAY = 86_400.0

_KNOWN_TOP_LEVEL_NAMES = ("site_statistics.csv", "site_statistics.json", "manifest.json", "manifest.sha256", "_SUCCESS")
_FIGURE_NAMES = ("five_site_summary.png", "seasonal_occurrence.png", "lifetime_distribution.png")


def _require(condition: bool, message: str) -> None:
    """在跨區彙整契約不成立時立即停止，避免產生看似完整卻不可信的統計。"""

    if not condition:
        raise ValueError(message)


def _season_label(time_utc_ns: int) -> str:
    """回傳 UTC 月份對應的氣象季節標籤。"""

    return SEASON_BY_MONTH[_datetime.datetime.fromtimestamp(int(time_utc_ns) / 1.0e9, tz=_datetime.UTC).month]


def _year_label(time_utc_ns: int) -> str:
    """回傳 UTC 年份字串。"""

    return str(_datetime.datetime.fromtimestamp(int(time_utc_ns) / 1.0e9, tz=_datetime.UTC).year)


def _day_label(time_utc_ns: int) -> str:
    """回傳 UTC 日期字串，供 day-block bootstrap 使用。"""

    return _datetime.datetime.fromtimestamp(int(time_utc_ns) / 1.0e9, tz=_datetime.UTC).date().isoformat()


def _haversine_distance_m(lon: np.ndarray, lat: np.ndarray, anchor_lon: float, anchor_lat: float) -> np.ndarray:
    """與 :mod:`traps.sites` 相同公式的球面距離；只用於 receptor 遮罩。"""

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    anchor_lon_rad = np.deg2rad(float(anchor_lon))
    anchor_lat_rad = np.deg2rad(float(anchor_lat))
    dlon = lon_rad - anchor_lon_rad
    dlat = lat_rad - anchor_lat_rad
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(anchor_lat_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def _ragged_gather(offsets: np.ndarray, selected_rows: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """向量化收集一組 ragged row（例如 pass_cells）中被選取的列，回傳串接值與 row 對應。

    ``offsets`` 長度為 ``n_rows + 1``；``selected_rows`` 是要收集的 row 索引（可不連續，
    可重複）。回傳 ``(values_out, row_out)``，其中 ``row_out[i]`` 是 ``values_out[i]``
    所屬的 ``selected_rows`` 位置（而非原始 row 索引），方便呼叫端用
    ``np.repeat``／``np.unique`` 對應回快照或站點。全程不逐列跑 Python 迴圈。
    """

    if selected_rows.size == 0:
        return values[:0], np.zeros(0, dtype=np.int64)
    starts = offsets[selected_rows]
    ends = offsets[selected_rows + 1]
    counts = (ends - starts).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return values[:0], np.zeros(0, dtype=np.int64)
    row_out = np.repeat(np.arange(selected_rows.size, dtype=np.int64), counts)
    cumulative_start = np.cumsum(counts) - counts
    within_row = np.arange(total, dtype=np.int64) - np.repeat(cumulative_start, counts)
    source_index = np.repeat(starts, counts) + within_row
    return values[source_index], row_out


# ---------------------------------------------------------------------------
# 輸入驗證：track-root／detect-root 的 manifest、artifact SHA-256 與跨區一致性
# ---------------------------------------------------------------------------


def _load_track_domain(track_root: Path, domain_id: str) -> dict[str, Any]:
    """讀取並驗證一個 domain 的 stage-2 `track-period` 輸出。"""

    domain_dir = track_root / domain_id
    manifest = _success_verified(domain_dir)
    _require(manifest is not None, f"track-root 缺少已驗證的 {domain_id}（缺 _SUCCESS 或 SHA-256 不符）: {domain_dir}")
    assert manifest is not None
    _require(manifest.get("schema_version") == TRACK_SCHEMA_VERSION, f"{domain_id} track manifest schema_version 不符")
    _require(manifest.get("domain_id") == domain_id, f"{domain_id} track manifest domain_id 不符")
    with np.load(domain_dir / "canonical_snapshots.npz") as archive:
        canonical = {name: np.array(archive[name]) for name in archive.files}
    with np.load(domain_dir / "track_observations.npz") as archive:
        observations = {name: np.array(archive[name]) for name in archive.files}
    tracks_by_id: dict[int, dict[str, Any]] = {}
    with (domain_dir / "tracks.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            tracks_by_id[int(row["track_id"])] = {
                "left_censored": row["left_censored"] == "True",
                "right_censored": row["right_censored"] == "True",
                "lifetime_hours": float(row["lifetime_seconds"]) / 3_600.0,
                "start_time_utc_ns": int(row["start_time_utc_ns"]),
            }
    return {
        "manifest": manifest,
        "canonical": canonical,
        "observations": observations,
        "tracks_by_id": tracks_by_id,
        "domain_dir": domain_dir,
    }


def _verify_shard_manifests(
    detect_root: Path, domain_id: str, available_months: Sequence[str], shard_manifest_records: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """核對 detect-root 的每個月份 shard 仍通過 `_SUCCESS` 驗證，且 manifest SHA-256 與
    track-period 記錄的一致（偵測 track 完成後 shard 被竄改的情況）。
    """

    recorded_sha256 = {str(record["month"]): str(record["manifest_sha256"]) for record in shard_manifest_records}
    manifests: dict[str, dict[str, Any]] = {}
    for month in available_months:
        shard_dir = detect_root / domain_id / month
        manifest = _success_verified(shard_dir)
        _require(manifest is not None, f"detect-root shard 未完成或驗證失敗: {shard_dir}")
        assert manifest is not None
        _require(manifest.get("schema_version") == SHARD_SCHEMA_VERSION, f"{shard_dir} schema_version 不符")
        expected_sha256 = recorded_sha256.get(month)
        _require(expected_sha256 is not None, f"track-period manifest 沒有記錄 {domain_id}/{month} 的 shard manifest")
        actual_sha256 = _sha256_file(shard_dir / "manifest.json")
        _require(
            actual_sha256 == expected_sha256,
            f"detect-root shard manifest 遭竄改或與 track-period 記錄不符: {shard_dir}/manifest.json",
        )
        manifests[month] = manifest
    return manifests


def _validate_cross_domain_consistency(domain_context: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """核對所有已載入 domain 共用同一組 param_id／detection_config／tracking_config／
    min_coverage／state_label；拒絕 PILOT_ONLY 與正式資料混用。
    """

    first_id = next(iter(domain_context))
    first = domain_context[first_id]
    for domain_id, context in domain_context.items():
        _require(context["param_id"] == first["param_id"], f"{domain_id} 與 {first_id} 的 param_id 不一致，無法合併")
        _require(
            context["detection_config"] == first["detection_config"],
            f"{domain_id} 與 {first_id} 的 detection_config 不一致，無法合併",
        )
        _require(
            context["min_coverage"] == first["min_coverage"], f"{domain_id} 與 {first_id} 的 min_coverage 不一致，無法合併"
        )
        _require(
            context["tracking_config"] == first["tracking_config"],
            f"{domain_id} 與 {first_id} 的 tracking_config 不一致，無法合併",
        )
        _require(
            context["state_label"] == first["state_label"],
            f"aggregate 不可混用 PILOT_ONLY 與正式資料（{first_id}={first['state_label']!r}，{domain_id}={context['state_label']!r}）",
        )
    return {
        "param_id": first["param_id"],
        "detection_config": first["detection_config"],
        "min_coverage": float(first["min_coverage"]),
        "tracking_config": first["tracking_config"],
        "state_label": first["state_label"],
    }


# ---------------------------------------------------------------------------
# 網格與站點遮罩（receptor、domain_partition）
# ---------------------------------------------------------------------------


def _load_domain_grid(detect_root: Path, domain_id: str, first_month: str) -> dict[str, Any]:
    """讀取一個 domain 第一個可用月份 shard 的 `grid_reference.npz`。"""

    with np.load(detect_root / domain_id / first_month / "grid_reference.npz") as archive:
        grid = {name: np.array(archive[name]) for name in archive.files}
    lon = grid["lon"]
    lat = grid["lat"]
    _require(lon.ndim == 1 and lat.ndim == 1, f"{domain_id} grid_reference lon/lat 必須是一維軸")
    shape = (int(lat.size), int(lon.size))
    _require(grid["analysis_ocean_mask"].shape == shape, f"{domain_id} analysis_ocean_mask shape 不符")
    grid["shape"] = shape
    return grid


def _build_domain_masks(
    domain_id: str, grid: Mapping[str, Any], design: Mapping[str, Any], study_design_path: Path
) -> dict[str, dict[str, np.ndarray]]:
    """建立一個 domain 的 receptor（PRIMARY）與 domain_partition（舊定義）站點遮罩。"""

    lon = grid["lon"]
    lat = grid["lat"]
    ocean_mask = np.asarray(grid["analysis_ocean_mask"], dtype=bool)
    partition = site_masks(domain_id, lon, lat, ocean_mask, study_design_path=study_design_path)
    domain_sites = [site for site in design["study_sites"] if site["flow_domain_id"] == domain_id]
    grid_lon, grid_lat = np.meshgrid(lon, lat, indexing="xy")

    receptor: dict[str, np.ndarray] = {}
    for site in domain_sites:
        site_id = str(site["study_site_id"])
        anchor_lon, anchor_lat = site["anchor_lonlat"]
        radius_m = float(site["receptor_core_radius_m"])
        distance = _haversine_distance_m(grid_lon, grid_lat, anchor_lon, anchor_lat)
        receptor[site_id] = ocean_mask & (distance <= radius_m)

    if domain_id == "northeast_taiwan_common_cache_v3":
        _require(
            not np.any(receptor["gongliao"] & receptor["guishan"]),
            "A 區貢寮／龜山島 receptor 遮罩必須互斥（12.5 km 半徑不可重疊）",
        )

    domain_site_ids = [str(site["study_site_id"]) for site in domain_sites]
    return {
        "receptor": {site_id: receptor[site_id] for site_id in domain_site_ids},
        "domain_partition": {site_id: partition[site_id] for site_id in domain_site_ids},
    }


# ---------------------------------------------------------------------------
# Pass 1：跨區 pooled 強核心門檻
# ---------------------------------------------------------------------------


def _canonical_valid_snapshot_indices_by_month(canonical: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """回傳每個月份下，canonical 且 coverage_pass 的 snapshot_index 陣列。"""

    month_codes = canonical["month"]
    snapshot_index = canonical["snapshot_index"]
    coverage_pass = canonical["coverage_pass"]
    result: dict[str, np.ndarray] = {}
    for month_code in np.unique(month_codes):
        month = f"{int(month_code):06d}"
        rows = (month_codes == month_code) & coverage_pass
        result[month] = np.sort(snapshot_index[rows])
    return result


def _pooled_core_s1(detect_root: Path, domain_id: str, month: str, valid_snapshot_indices: np.ndarray) -> np.ndarray:
    """只讀取 detections.npz 的少數欄位，收集該月份合格且 canonical valid 的核心 s1。"""

    if valid_snapshot_indices.size == 0:
        return np.zeros(0, dtype=np.float64)
    with np.load(detect_root / domain_id / month / "detections.npz") as archive:
        snapshot_index = np.array(archive["snapshot_index"])
        is_qualified = np.array(archive["is_qualified"])
        core_s1 = np.array(archive["core_s1_sinv"])
    mask = is_qualified & np.isin(snapshot_index, valid_snapshot_indices)
    return core_s1[mask]


def _compute_strong_threshold(
    detect_root: Path, domain_contexts: Mapping[str, dict[str, Any]], strong_quantile: float
) -> tuple[float | None, float | None, int]:
    """計算跨區 pooled 的強核心 s1 門檻（s⁻¹ 與 day⁻¹），回傳門檻與 pooled 核心數。"""

    pooled: list[np.ndarray] = []
    for domain_id, context in sorted(domain_contexts.items()):
        for month, valid_indices in sorted(context["canonical_valid_by_month"].items()):
            pooled.append(_pooled_core_s1(detect_root, domain_id, month, valid_indices))
    values = np.concatenate(pooled) if pooled else np.zeros(0, dtype=np.float64)
    if values.size == 0:
        return None, None, 0
    threshold_sinv = float(np.quantile(values, strong_quantile))
    return threshold_sinv, threshold_sinv * SECONDS_PER_DAY, int(values.size)


# ---------------------------------------------------------------------------
# 累加器：site x scope x stratum
# ---------------------------------------------------------------------------


def _new_site_accumulator() -> dict[str, Any]:
    return {
        "n_valid": 0,
        "core_occurrence": 0,
        "strong_core_occurrence": 0,
        "curve_occurrence": 0,
        "pass_fraction_sum": 0.0,
        "core_count_total": 0,
        "core_s1_values": [],
        "day_stats": defaultdict(lambda: [0, 0, 0, 0, 0.0]),  # day -> [n, core, strong, curve, pass_frac_sum]
    }


def _new_spatial_accumulator(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "n_valid": np.zeros(shape, dtype=np.int64),
        "core_count": np.zeros(shape, dtype=np.int64),
        "strong_core_count": np.zeros(shape, dtype=np.int64),
        "curve_pass_count": np.zeros(shape, dtype=np.int64),
        "core_s1_sum": np.zeros(shape, dtype=np.float64),
        "lifetime_count": np.zeros(shape, dtype=np.int64),
        "lifetime_sum": np.zeros(shape, dtype=np.float64),
    }


def _process_domain(
    *,
    domain_id: str,
    detect_root: Path,
    context: dict[str, Any],
    masks: Mapping[str, Mapping[str, np.ndarray]],
    grid: Mapping[str, Any],
    strong_threshold_sinv: float | None,
    site_accumulators: dict[tuple[str, str, str], dict[str, Any]],
    spatial_accumulators: dict[str, dict[str, np.ndarray]],
) -> None:
    """流式處理一個 domain 的每個月份 shard，累加 site×scope×stratum 與逐格空間統計。"""

    shape = grid["shape"]
    ny, nx = shape
    min_coverage = float(context["min_coverage"])
    canonical = context["canonical"]
    canonical_by_month = context["canonical_valid_by_month"]

    flat_masks: dict[str, dict[str, np.ndarray]] = {
        scope: {site_id: mask.reshape(-1) for site_id, mask in scope_masks.items()} for scope, scope_masks in masks.items()
    }
    mask_counts: dict[str, dict[str, int]] = {
        scope: {site_id: int(mask.sum()) for site_id, mask in scope_masks.items()} for scope, scope_masks in flat_masks.items()
    }

    for month in sorted(canonical_by_month):
        valid_snapshot_indices = canonical_by_month[month]
        if valid_snapshot_indices.size == 0:
            continue
        month_rows = canonical["month"] == int(month)
        month_time_by_index = dict(
            zip(canonical["snapshot_index"][month_rows].tolist(), canonical["time_utc_ns"][month_rows].tolist(), strict=True)
        )

        with np.load(detect_root / domain_id / month / "snapshots.npz") as archive:
            shard_snapshot_index = np.array(archive["snapshot_index"])
            valid_mask_packed = np.array(archive["valid_mask_packed"])
        position_by_index = {int(value): position for position, value in enumerate(shard_snapshot_index.tolist())}

        with np.load(detect_root / domain_id / month / "detections.npz") as archive:
            det_snapshot_index = np.array(archive["snapshot_index"])
            det_is_qualified = np.array(archive["is_qualified"])
            det_core_row = np.array(archive["core_row"])
            det_core_col = np.array(archive["core_col"])
            det_core_s1 = np.array(archive["core_s1_sinv"])
            det_pass_cell_offsets = np.array(archive["pass_cell_offsets"])
            det_pass_cells = np.array(archive["pass_cells"])

        qualified_rows = np.nonzero(det_is_qualified)[0]
        order = np.argsort(det_snapshot_index[qualified_rows], kind="stable")
        qualified_rows = qualified_rows[order]
        qualified_snapshot_sorted = det_snapshot_index[qualified_rows]

        for snapshot_index in valid_snapshot_indices.tolist():
            time_utc_ns = month_time_by_index[int(snapshot_index)]
            day = _day_label(time_utc_ns)
            season = _season_label(time_utc_ns)
            year = _year_label(time_utc_ns)
            strata = ("all", season, year)

            position = position_by_index[int(snapshot_index)]
            valid_bits = np.unpackbits(valid_mask_packed[position], count=ny * nx).astype(bool)

            lo = int(np.searchsorted(qualified_snapshot_sorted, snapshot_index, side="left"))
            hi = int(np.searchsorted(qualified_snapshot_sorted, snapshot_index, side="right"))
            rows = qualified_rows[lo:hi]

            core_flat = det_core_row[rows].astype(np.int64) * np.int64(nx) + det_core_col[rows].astype(np.int64)
            core_s1 = det_core_s1[rows]
            strong_flags = core_s1 <= strong_threshold_sinv if strong_threshold_sinv is not None else np.zeros(rows.shape, dtype=bool)

            cell_values, _row_out = _ragged_gather(det_pass_cell_offsets, rows, det_pass_cells)
            cells_this_snapshot = np.unique(cell_values.astype(np.int64)) if cell_values.size else np.zeros(0, dtype=np.int64)

            for scope in SCOPES:
                for site_id, mask_flat in flat_masks[scope].items():
                    mask_count = mask_counts[scope][site_id]
                    if mask_count == 0:
                        continue
                    site_valid_count = int(np.count_nonzero(valid_bits & mask_flat))
                    if (site_valid_count / mask_count) < min_coverage:
                        continue

                    core_in_mask = mask_flat[core_flat] if core_flat.size else np.zeros(0, dtype=bool)
                    strong_in_mask = core_in_mask & strong_flags
                    curve_cells_in_mask = cells_this_snapshot[mask_flat[cells_this_snapshot]] if cells_this_snapshot.size else cells_this_snapshot
                    pass_fraction = float(curve_cells_in_mask.size) / float(mask_count)
                    has_core = bool(np.any(core_in_mask))
                    has_strong = bool(np.any(strong_in_mask))
                    has_curve = curve_cells_in_mask.size > 0

                    for stratum in strata:
                        accumulator = site_accumulators[(site_id, scope, stratum)]
                        accumulator["n_valid"] += 1
                        if has_core:
                            accumulator["core_occurrence"] += 1
                            accumulator["core_s1_values"].append(core_s1[core_in_mask])
                            accumulator["core_count_total"] += int(np.count_nonzero(core_in_mask))
                        if has_strong:
                            accumulator["strong_core_occurrence"] += 1
                        if has_curve:
                            accumulator["curve_occurrence"] += 1
                        accumulator["pass_fraction_sum"] += pass_fraction

                        day_row = accumulator["day_stats"][day]
                        day_row[0] += 1
                        day_row[1] += int(has_core)
                        day_row[2] += int(has_strong)
                        day_row[3] += int(has_curve)
                        day_row[4] += pass_fraction

            # --- SPATIAL（domain-wide，不受站點遮罩限制） ---
            for stratum in ("all", season):
                spatial = spatial_accumulators.setdefault(stratum, _new_spatial_accumulator(shape))
                spatial["n_valid"] += valid_bits.reshape(shape).astype(np.int64)
                if core_flat.size:
                    np.add.at(spatial["core_count"].reshape(-1), core_flat, 1)
                    np.add.at(spatial["core_s1_sum"].reshape(-1), core_flat, core_s1)
                    if strong_flags.any():
                        np.add.at(spatial["strong_core_count"].reshape(-1), core_flat[strong_flags], 1)
                if cells_this_snapshot.size:
                    spatial["curve_pass_count"].reshape(-1)[cells_this_snapshot] += 1


# ---------------------------------------------------------------------------
# Track 壽命統計（site x scope x stratum，以及 SPATIAL 逐格未截尾壽命）
# ---------------------------------------------------------------------------


def _track_segments(observations: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """把已依 (track_id, time) 排序的觀測陣列切成逐 track 的連續索引區段。"""

    track_id = observations["track_id"]
    if track_id.size == 0:
        return []
    boundaries = np.nonzero(np.diff(track_id))[0] + 1
    return np.split(np.arange(track_id.size), boundaries)


def _process_track_lifetimes(
    *,
    domain_id: str,
    context: dict[str, Any],
    masks: Mapping[str, Mapping[str, np.ndarray]],
    grid: Mapping[str, Any],
    site_accumulators: dict[tuple[str, str, str], dict[str, Any]],
    spatial_accumulators: dict[str, dict[str, np.ndarray]],
) -> None:
    """以 track_observations／tracks.csv 計算站點壽命統計，並累加逐格未截尾壽命。"""

    shape = grid["shape"]
    ny, nx = shape
    x_axis = grid["x_m_axis"]
    y_axis = grid["y_m_axis"]
    observations = context["observations"]
    tracks_by_id = context["tracks_by_id"]
    if observations["track_id"].size == 0:
        return

    cols = _nearest_indices(x_axis, observations["core_x_m"])
    rows = _nearest_indices(y_axis, observations["core_y_m"])
    flat = rows.astype(np.int64) * np.int64(nx) + cols.astype(np.int64)
    times = observations["time_utc_ns"]

    flat_masks: dict[str, dict[str, np.ndarray]] = {
        scope: {site_id: mask.reshape(-1) for site_id, mask in scope_masks.items()} for scope, scope_masks in masks.items()
    }

    for segment in _track_segments(observations):
        track_id = int(observations["track_id"][segment[0]])
        track = tracks_by_id.get(track_id)
        if track is None:
            continue
        segment_flat = flat[segment]
        segment_times = times[segment]
        is_uncensored = not (track["left_censored"] or track["right_censored"])
        lifetime_hours = float(track["lifetime_hours"])

        for scope in SCOPES:
            for site_id, mask_flat in flat_masks[scope].items():
                in_mask = mask_flat[segment_flat]
                if not np.any(in_mask):
                    continue
                first_local = int(np.nonzero(in_mask)[0][0])
                first_time = int(segment_times[first_local])
                strata = ("all", _season_label(first_time), _year_label(first_time))
                for stratum in strata:
                    accumulator = site_accumulators[(site_id, scope, stratum)]
                    accumulator.setdefault("track_n", 0)
                    accumulator["track_n"] += 1
                    accumulator.setdefault("track_lifetimes_all", [])
                    accumulator["track_lifetimes_all"].append(lifetime_hours)
                    if is_uncensored:
                        accumulator.setdefault("track_lifetimes", [])
                        accumulator["track_lifetimes"].append(lifetime_hours)

        # SPATIAL：未截尾 track 的每個造訪格子計一次（"all" 與該 track 起始季節）。
        if is_uncensored:
            visited_cells = np.unique(segment_flat)
            start_season = _season_label(int(track["start_time_utc_ns"]))
            for stratum in ("all", start_season):
                spatial = spatial_accumulators.setdefault(stratum, _new_spatial_accumulator(shape))
                spatial["lifetime_count"].reshape(-1)[visited_cells] += 1
                spatial["lifetime_sum"].reshape(-1)[visited_cells] += lifetime_hours


# ---------------------------------------------------------------------------
# Bootstrap 與最終記錄組裝
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    day_stats: Mapping[str, list[float]], *, reps: int, rng: np.random.Generator
) -> dict[str, tuple[float, float] | None]:
    """對三個出現率與 mean_pass_cell_fraction 做 day-block bootstrap 95% CI。"""

    days = sorted(day_stats)
    if not days:
        return {"core_occurrence_rate": None, "strong_core_occurrence_rate": None, "curve_occurrence_rate": None, "mean_pass_cell_fraction": None}
    table = np.asarray([day_stats[day] for day in days], dtype=np.float64)
    n_col, core_col, strong_col, curve_col, pass_col = table[:, 0], table[:, 1], table[:, 2], table[:, 3], table[:, 4]
    n_days = len(days)
    sample_idx = rng.integers(0, n_days, size=(reps, n_days))

    def _ci_for(numerator: np.ndarray, *, is_mean: bool = False) -> tuple[float, float] | None:
        resampled_n = n_col[sample_idx].sum(axis=1)
        resampled_value = numerator[sample_idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rates = np.where(resampled_n > 0, resampled_value / resampled_n, np.nan)
        if np.all(np.isnan(rates)):
            return None
        low, high = np.nanpercentile(rates, [2.5, 97.5])
        return float(low), float(high)

    return {
        "core_occurrence_rate": _ci_for(core_col),
        "strong_core_occurrence_rate": _ci_for(strong_col),
        "curve_occurrence_rate": _ci_for(curve_col),
        "mean_pass_cell_fraction": _ci_for(pass_col),
    }


def _quantile_or_none(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None


def _finalize_site_record(
    site_id: str,
    scope: str,
    stratum: str,
    accumulator: Mapping[str, Any],
    *,
    site_area_km2: float | None,
    site_name_zh: str,
    strong_threshold_sinv: float | None,
    bootstrap_reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """把一組 site×scope×stratum 累加器轉成 site_statistics 記錄。"""

    n_valid = int(accumulator["n_valid"])

    def _rate(numerator: int) -> float | None:
        return float(numerator / n_valid) if n_valid else None

    core_s1_values = np.concatenate(accumulator["core_s1_values"]) if accumulator["core_s1_values"] else np.zeros(0)
    lifetimes = accumulator.get("track_lifetimes", [])
    cis = _bootstrap_ci(accumulator["day_stats"], reps=bootstrap_reps, rng=rng) if n_valid else {
        "core_occurrence_rate": None, "strong_core_occurrence_rate": None, "curve_occurrence_rate": None, "mean_pass_cell_fraction": None
    }
    mean_pass_fraction = float(accumulator["pass_fraction_sum"] / n_valid) if n_valid else None

    frac_ge = {}
    for threshold in LIFETIME_THRESHOLD_HOURS:
        n_uncensored = len(lifetimes)
        frac_ge[threshold] = float(sum(1 for value in lifetimes if value >= threshold) / n_uncensored) if n_uncensored else None

    return {
        "site_id": site_id,
        "site_name_zh": site_name_zh,
        "scope": scope,
        "stratum": stratum,
        "n_valid_snapshots": n_valid,
        "core_occurrence_numerator": int(accumulator["core_occurrence"]),
        "core_occurrence_rate": _rate(accumulator["core_occurrence"]),
        "core_occurrence_rate_ci95": cis["core_occurrence_rate"],
        "strong_core_occurrence_numerator": int(accumulator["strong_core_occurrence"]),
        "strong_core_occurrence_rate": _rate(accumulator["strong_core_occurrence"]),
        "strong_core_occurrence_rate_ci95": cis["strong_core_occurrence_rate"],
        "strong_core_threshold_sinv": strong_threshold_sinv,
        "strong_core_threshold_day": strong_threshold_sinv * SECONDS_PER_DAY if strong_threshold_sinv is not None else None,
        "curve_occurrence_numerator": int(accumulator["curve_occurrence"]),
        "curve_occurrence_rate": _rate(accumulator["curve_occurrence"]),
        "curve_occurrence_rate_ci95": cis["curve_occurrence_rate"],
        "mean_pass_cell_fraction": mean_pass_fraction,
        "mean_pass_cell_fraction_ci95": cis["mean_pass_cell_fraction"],
        "site_area_km2": site_area_km2,
        "core_density_per_1000km2_per_snapshot": (
            float(accumulator["core_count_total"] / site_area_km2 / n_valid * 1000.0) if site_area_km2 and n_valid else None
        ),
        "core_s1_count": int(core_s1_values.size),
        "core_s1_mean_sinv": float(np.mean(core_s1_values)) if core_s1_values.size else None,
        "core_s1_median_sinv": float(np.median(core_s1_values)) if core_s1_values.size else None,
        "core_s1_p10_sinv": float(np.quantile(core_s1_values, 0.10)) if core_s1_values.size else None,
        "core_s1_mean_day": float(np.mean(core_s1_values) * SECONDS_PER_DAY) if core_s1_values.size else None,
        "core_s1_median_day": float(np.median(core_s1_values) * SECONDS_PER_DAY) if core_s1_values.size else None,
        "core_s1_p10_day": float(np.quantile(core_s1_values, 0.10) * SECONDS_PER_DAY) if core_s1_values.size else None,
        "track_n_tracks": int(accumulator.get("track_n", 0)),
        "track_n_uncensored": len(lifetimes),
        "track_mean_lifetime_hours": float(np.mean(lifetimes)) if lifetimes else None,
        "track_median_lifetime_hours": _quantile_or_none(lifetimes, 0.5),
        "track_p90_lifetime_hours": _quantile_or_none(lifetimes, 0.9),
        "track_frac_uncensored_ge_6h": frac_ge[6.0],
        "track_frac_uncensored_ge_12h": frac_ge[12.0],
        "track_frac_uncensored_ge_24h": frac_ge[24.0],
    }


SITE_STAT_FIELDNAMES = (
    "site_id", "site_name_zh", "scope", "stratum", "n_valid_snapshots",
    "core_occurrence_numerator", "core_occurrence_rate", "core_occurrence_rate_ci95",
    "strong_core_occurrence_numerator", "strong_core_occurrence_rate", "strong_core_occurrence_rate_ci95",
    "strong_core_threshold_sinv", "strong_core_threshold_day",
    "curve_occurrence_numerator", "curve_occurrence_rate", "curve_occurrence_rate_ci95",
    "mean_pass_cell_fraction", "mean_pass_cell_fraction_ci95",
    "site_area_km2", "core_density_per_1000km2_per_snapshot",
    "core_s1_count", "core_s1_mean_sinv", "core_s1_median_sinv", "core_s1_p10_sinv",
    "core_s1_mean_day", "core_s1_median_day", "core_s1_p10_day",
    "track_n_tracks", "track_n_uncensored", "track_mean_lifetime_hours", "track_median_lifetime_hours",
    "track_p90_lifetime_hours", "track_frac_uncensored_ge_6h", "track_frac_uncensored_ge_12h", "track_frac_uncensored_ge_24h",
)


def _write_site_statistics(output_dir: Path, records: list[dict[str, Any]]) -> None:
    """寫出 long-format ``site_statistics.csv`` 與 ``site_statistics.json``。"""

    csv_path = output_dir / "site_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SITE_STAT_FIELDNAMES))
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in SITE_STAT_FIELDNAMES})
    _atomic_write_json(output_dir / "site_statistics.json", {"schema_version": AGGREGATE_SCHEMA_VERSION, "records": records})


def _finalize_spatial(shape: tuple[int, int], accumulator: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """把逐格累加器轉成 NaN-safe 的頻率／平均欄位。"""

    n_valid = accumulator["n_valid"]
    valid_gt0 = n_valid > 0
    core_frequency = np.full(shape, np.nan, dtype=np.float64)
    np.divide(accumulator["core_count"], n_valid, out=core_frequency, where=valid_gt0)
    strong_core_frequency = np.full(shape, np.nan, dtype=np.float64)
    np.divide(accumulator["strong_core_count"], n_valid, out=strong_core_frequency, where=valid_gt0)
    curve_pass_frequency = np.full(shape, np.nan, dtype=np.float64)
    np.divide(accumulator["curve_pass_count"], n_valid, out=curve_pass_frequency, where=valid_gt0)
    mean_core_s1 = np.full(shape, np.nan, dtype=np.float64)
    np.divide(accumulator["core_s1_sum"], accumulator["core_count"], out=mean_core_s1, where=accumulator["core_count"] > 0)
    mean_lifetime = np.full(shape, np.nan, dtype=np.float64)
    np.divide(accumulator["lifetime_sum"], accumulator["lifetime_count"], out=mean_lifetime, where=accumulator["lifetime_count"] > 0)
    return {
        "n_valid": n_valid,
        "core_frequency": core_frequency,
        "strong_core_frequency": strong_core_frequency,
        "curve_pass_frequency": curve_pass_frequency,
        "mean_core_s1_sinv": mean_core_s1,
        "mean_core_s1_day": mean_core_s1 * SECONDS_PER_DAY,
        "mean_lifetime_hours": mean_lifetime,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_aggregate(args: argparse.Namespace) -> int:
    """讀入已驗證的 track-root／detect-root，計算五站全期統計與逐格空間分布。"""

    start = _time.monotonic()
    design = load_study_design(Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH))
    all_domain_ids = [str(domain["flow_domain_id"]) for domain in design["flow_domains"]]

    track_root = Path(args.track_root)
    detect_root = Path(args.detect_root)
    output_dir = Path(args.output_dir)

    present_domains = [domain_id for domain_id in all_domain_ids if (track_root / domain_id).is_dir()]
    missing_domains = [domain_id for domain_id in all_domain_ids if domain_id not in present_domains]
    if missing_domains and not args.allow_partial_domains:
        raise ValueError(f"缺少 track-root domain，需加 --allow-partial-domains 才可略過: {missing_domains}")
    _require(bool(present_domains), "track-root 沒有任何可用的 domain")

    figures_enabled = not bool(args.no_figures)
    known_names = list(_KNOWN_TOP_LEVEL_NAMES)
    for domain_id in present_domains:
        for stratum in ("all", *SEASONS):
            known_names.append(f"spatial/{domain_id}/spatial_{stratum}.npz")
        if figures_enabled:
            known_names.append(f"spatial_{domain_id}.png")
    if figures_enabled:
        known_names.extend(_FIGURE_NAMES)

    skip_manifest = _prepare_output_dir(output_dir, resume=bool(args.resume), known_names=known_names)
    if skip_manifest is not None:
        summary = {
            "operation": "aggregate",
            "output_dir": str(output_dir),
            "resumed": True,
            "skipped": True,
            "domains": skip_manifest.get("domains_present"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    # --- 讀入並驗證每個 domain 的 track/shard manifest --------------------------------
    domain_context: dict[str, dict[str, Any]] = {}
    for domain_id in present_domains:
        loaded = _load_track_domain(track_root, domain_id)
        manifest = loaded["manifest"]
        available_months = list(manifest.get("available_months") or [])
        shard_manifests = _verify_shard_manifests(detect_root, domain_id, available_months, manifest.get("shard_manifests") or [])
        first_shard = shard_manifests[available_months[0]] if available_months else None
        _require(first_shard is not None, f"{domain_id} 沒有任何可用月份 shard")
        assert first_shard is not None
        for month, shard_manifest in shard_manifests.items():
            _require(shard_manifest["detection_config"] == first_shard["detection_config"], f"{domain_id}/{month} detection_config 與同區其他月份不一致")
            _require(shard_manifest["min_coverage"] == first_shard["min_coverage"], f"{domain_id}/{month} min_coverage 與同區其他月份不一致")

        canonical_valid_by_month = _canonical_valid_snapshot_indices_by_month(loaded["canonical"])
        domain_context[domain_id] = {
            "manifest": manifest,
            "canonical": loaded["canonical"],
            "observations": loaded["observations"],
            "tracks_by_id": loaded["tracks_by_id"],
            "available_months": available_months,
            "canonical_valid_by_month": canonical_valid_by_month,
            "param_id": manifest.get("param_id") or first_shard["param_id"],
            "detection_config": first_shard["detection_config"],
            "min_coverage": float(first_shard["min_coverage"]),
            "tracking_config": manifest.get("tracking_config"),
            "state_label": manifest.get("state_label"),
        }

    shared = _validate_cross_domain_consistency(domain_context)

    # --- 網格與站點遮罩 -----------------------------------------------------------
    grids: dict[str, dict[str, Any]] = {}
    masks: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for domain_id in present_domains:
        first_month = sorted(domain_context[domain_id]["available_months"])[0]
        grid = _load_domain_grid(detect_root, domain_id, first_month)
        grids[domain_id] = grid
        masks[domain_id] = _build_domain_masks(domain_id, grid, design, Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH))

    sites_present = sorted({site_id for domain_masks in masks.values() for site_id in domain_masks["receptor"]})
    sites_missing = sorted(set(SITE_IDS) - set(sites_present))
    site_domain = {}
    site_name_zh = {str(site["study_site_id"]): str(site["study_site_name_zh"]) for site in design["study_sites"]}
    site_anchor = {
        str(site["study_site_id"]): (float(site["anchor_lonlat"][0]), float(site["anchor_lonlat"][1]), float(site["receptor_core_radius_m"]))
        for site in design["study_sites"]
    }
    for domain_id, domain_masks in masks.items():
        for site_id in domain_masks["receptor"]:
            site_domain[site_id] = domain_id

    # --- Pass 1：pooled 強核心門檻 ---------------------------------------------------
    strong_threshold_sinv, strong_threshold_day, pooled_core_count = _compute_strong_threshold(
        detect_root, domain_context, float(args.strong_quantile)
    )

    # --- Pass 2：逐 domain 流式累加 --------------------------------------------------
    site_accumulators: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(_new_site_accumulator)
    spatial_accumulators: dict[str, dict[str, dict[str, np.ndarray]]] = {domain_id: {} for domain_id in present_domains}
    for domain_id in present_domains:
        _process_domain(
            domain_id=domain_id,
            detect_root=detect_root,
            context=domain_context[domain_id],
            masks=masks[domain_id],
            grid=grids[domain_id],
            strong_threshold_sinv=strong_threshold_sinv,
            site_accumulators=site_accumulators,
            spatial_accumulators=spatial_accumulators[domain_id],
        )
        _process_track_lifetimes(
            domain_id=domain_id,
            context=domain_context[domain_id],
            masks=masks[domain_id],
            grid=grids[domain_id],
            site_accumulators=site_accumulators,
            spatial_accumulators=spatial_accumulators[domain_id],
        )

    # --- 站點面積（依 scope）-------------------------------------------------------
    site_area_km2: dict[tuple[str, str], float | None] = {}
    for domain_id, domain_masks in masks.items():
        cell_area_m2 = grids[domain_id]["cell_area_m2"]
        for scope, scope_masks in domain_masks.items():
            for site_id, mask in scope_masks.items():
                area_m2 = float(np.sum(cell_area_m2[mask])) if np.any(mask) else 0.0
                site_area_km2[(site_id, scope)] = (area_m2 / 1.0e6) if area_m2 > 0.0 else None

    rng = np.random.default_rng(int(args.seed))
    records: list[dict[str, Any]] = []
    for site_id in sites_present:
        for scope in SCOPES:
            strata_present = sorted({key[2] for key in site_accumulators if key[0] == site_id and key[1] == scope})
            ordered = ["all"] + [s for s in SEASONS if s in strata_present] + [s for s in strata_present if s not in SEASONS and s != "all"]
            for stratum in ordered:
                accumulator = site_accumulators[(site_id, scope, stratum)]
                records.append(
                    _finalize_site_record(
                        site_id,
                        scope,
                        stratum,
                        accumulator,
                        site_area_km2=site_area_km2.get((site_id, scope)),
                        site_name_zh=site_name_zh.get(site_id, site_id),
                        strong_threshold_sinv=strong_threshold_sinv,
                        bootstrap_reps=int(args.bootstrap_reps),
                        rng=rng,
                    )
                )
    _write_site_statistics(output_dir, records)

    # --- SPATIAL 輸出 ---------------------------------------------------------------
    finalized_spatial: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for domain_id in present_domains:
        shape = grids[domain_id]["shape"]
        finalized_spatial[domain_id] = {}
        for stratum in ("all", *SEASONS):
            accumulator = spatial_accumulators[domain_id].get(stratum, _new_spatial_accumulator(shape))
            finalized = _finalize_spatial(shape, accumulator)
            finalized_spatial[domain_id][stratum] = finalized
            spatial_dir = output_dir / "spatial" / domain_id
            _atomic_write_npz(
                spatial_dir / f"spatial_{stratum}.npz",
                lon=grids[domain_id]["lon"],
                lat=grids[domain_id]["lat"],
                **finalized,
            )

    colour_limits = _compute_colour_limits(finalized_spatial, present_domains)

    # --- 圖面 -------------------------------------------------------------------
    figure_paths: list[str] = []
    if figures_enabled:
        from . import aggregate_figures

        coastline, coastline_meta = _resolve_coastline_geojson(args)
        period_label = _period_label(domain_context)
        common_footer_state = shared["state_label"]
        raw_lifetimes = {
            site_id: {
                "uncensored": list(site_accumulators[(site_id, "receptor", "all")].get("track_lifetimes", [])),
                "all": list(site_accumulators[(site_id, "receptor", "all")].get("track_lifetimes_all", [])),
            }
            for site_id in sites_present
        }
        figure_paths = aggregate_figures.render_all(
            output_dir,
            records=records,
            spatial=finalized_spatial,
            grids=grids,
            masks=masks,
            site_name_zh=site_name_zh,
            colour_limits=colour_limits,
            coastline=coastline,
            state_label=common_footer_state,
            param_id=shared["param_id"],
            period_label=period_label,
            raw_lifetimes=raw_lifetimes,
            site_anchor=site_anchor,
            site_domain=site_domain,
        )

    # --- manifest ----------------------------------------------------------------
    inputs_record = {
        domain_id: {
            "track_manifest_path": str(track_root / domain_id / "manifest.json"),
            "track_manifest_sha256": _sha256_file(track_root / domain_id / "manifest.json"),
            "available_months": domain_context[domain_id]["available_months"],
            "shard_manifest_sha256": {
                month: _sha256_file(detect_root / domain_id / month / "manifest.json")
                for month in domain_context[domain_id]["available_months"]
            },
        }
        for domain_id in present_domains
    }
    runtime_seconds = _time.monotonic() - start
    manifest = {
        "operation": "aggregate",
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "param_id": shared["param_id"],
        "state_label": shared["state_label"],
        "detection_config": shared["detection_config"],
        "tracking_config": shared["tracking_config"],
        "min_coverage": shared["min_coverage"],
        "domains_present": present_domains,
        "domains_missing": missing_domains,
        "sites_present": sites_present,
        "sites_missing": sites_missing,
        "strong_quantile": float(args.strong_quantile),
        "strong_core_threshold_sinv": strong_threshold_sinv,
        "strong_core_threshold_day": strong_threshold_day,
        "pooled_core_count_for_threshold": pooled_core_count,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.seed),
        "colour_limits": colour_limits,
        "inputs": inputs_record,
        "study_design_path": str(Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH)),
        "study_design_sha256": _sha256_file(Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH)),
        "figures_enabled": figures_enabled,
        "runtime_seconds": runtime_seconds,
        "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds"),
        "package_version": _package_version(),
        **_git_info(_package_root()),
    }
    known_names_existing = [name for name in known_names if (output_dir / name).is_file()]
    manifest["artifacts"] = _artifact_records(output_dir, known_names_existing)
    _write_manifest_with_checksum(output_dir, manifest)
    (output_dir / "_SUCCESS").write_text(
        json.dumps({"created_at_utc": manifest["created_at_utc"]}, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = {
        "operation": "aggregate",
        "output_dir": str(output_dir),
        "resumed": False,
        "skipped": False,
        "domains_present": present_domains,
        "domains_missing": missing_domains,
        "sites_present": sites_present,
        "record_count": len(records),
        "figure_count": len(figure_paths),
        "runtime_seconds": runtime_seconds,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _period_label(domain_context: Mapping[str, dict[str, Any]]) -> str:
    """回傳所有已載入 domain available_months 聯集的最早／最晚月份標籤。"""

    months = sorted({month for context in domain_context.values() for month in context["available_months"]})
    if not months:
        return "no data"
    return f"{months[0]}-{months[-1]}"


def _compute_colour_limits(
    finalized_spatial: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], present_domains: Sequence[str]
) -> dict[str, list[float]]:
    """對「all」stratum 的四個逐格欄位計算跨區 2–98 百分位共用色階上下限。"""

    fields = ("curve_pass_frequency", "core_frequency", "mean_core_s1_day", "mean_lifetime_hours")
    limits: dict[str, list[float]] = {}
    for field in fields:
        pooled = [finalized_spatial[domain_id]["all"][field].reshape(-1) for domain_id in present_domains]
        values = np.concatenate(pooled) if pooled else np.zeros(0)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            limits[field] = [0.0, 1.0]
            continue
        low, high = np.nanpercentile(finite, [2.0, 98.0])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            high = low + 1.0
        limits[field] = [float(low), float(high)]
    return limits


# ---------------------------------------------------------------------------
# argparse 註冊
# ---------------------------------------------------------------------------


def register_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """把 ``aggregate`` 掛到既有 `python -m traps` parser；只新增此子命令。"""

    aggregate = subparsers.add_parser("aggregate", help="Stage 3：跨月份／跨區彙整五站出現率、強度與壽命統計")
    aggregate.add_argument("--track-root", type=Path, required=True, help="track-period 輸出根目錄（每個 domain 一個子目錄）")
    aggregate.add_argument("--detect-root", type=Path, required=True, help="detect-shards 輸出根目錄（domain/YYYYMM）")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--param-id", help="覆寫輸出 manifest 的 param_id 顯示值；仍須與輸入一致")
    aggregate.add_argument("--study-design", type=Path, default=DEFAULT_STUDY_DESIGN_PATH)
    aggregate.add_argument("--strong-quantile", type=float, default=0.10, help="pooled 強核心 s1 分位數，預設最強 10%%")
    aggregate.add_argument("--bootstrap-reps", type=int, default=1_000)
    aggregate.add_argument("--seed", type=int, default=20_260_928)
    aggregate.add_argument("--coastline-geojson", type=Path, help="成果圖 CRS84 海岸線 GeoJSON；未指定時使用專案預設檔")
    aggregate.add_argument("--no-figures", action="store_true", help="略過成果圖，只輸出統計表與 manifest")
    aggregate.add_argument("--resume", action="store_true")
    aggregate.add_argument("--allow-partial-domains", action="store_true", help="允許 track-root 只有部分 domain")
    aggregate.set_defaults(handler=run_aggregate)


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "register_subcommands",
    "run_aggregate",
]
