"""跨月份／跨區彙整的第三階段：``aggregate``。

Stage 1（``detect-shards``）與 Stage 2（``track-period``）分別把單一 domain-month
的逐時檢出、以及一個 domain 全期的跨月份 track 寫成可續跑、可驗證的 shard／track
artifact。本模組是第三階段：讀入已完成的 track-root（`traps.period.track.v1`）與
detect-root（`traps.period.shard.v1`），核對已載入 domain 共用同一組 `param_id`、
`detection_config`、`tracking_config`、`min_coverage`、`state_label`，再以
:func:`traps.sites.site_masks`（沿用四區 flow-domain 分割定義）取得每個研究站的
「站點分析區」（``site_area``）：B／C／D 是整個 flow domain 的分析海域，A 區則依
``site_masks`` 的最近 anchor 分割由貢寮與龜山島各取一半。

五個研究站彼此獨立，不做跨站比較，因此任何門檻、色階與熱點定義都只用該站自己的
資料：強核心門檻是「該站分析區內合格核心 s1」的分位數，熱點是該站
``curve_pass_frequency`` 的最高分位格點，圖面色階上下限也逐站計算。本模組不使用任何接收區半徑。

出現率一律用「有效快照分母」，不以 0 假裝資料缺口；核心與曲線通過只計「該快照
是否至少一次」，不重複計曲線交會次數。全部欄位保留 `PILOT_ONLY` 等
`state_label`／`science_status`，輸出只代表模型內部「表層短期吸引潛勢」，不是實測
海廢濃度或污染來源。

成果圖使用 :mod:`traps.report_font` 解析的繁體中文字型（找不到合格字型即 fail closed，
除非 ``--no-figures``），字型選擇紀錄寫入 manifest。
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
from .sites import DEFAULT_STUDY_DESIGN_PATH, load_study_design, site_masks

AGGREGATE_SCHEMA_VERSION = "traps.period.aggregate.v2"

SEASON_BY_MONTH: dict[int, str] = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASONS = ("DJF", "MAM", "JJA", "SON")
SITE_IDS = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")
SITE_SCOPE = "site_area"
"""唯一的空間範圍：站點分析區（flow-domain 分割 × 分析海域遮罩）。保留為常數欄位 ``scope``。"""
LIFETIME_EVOLUTION_MIN_HOURS = 12.0
LIFETIME_EVOLUTION_AGE_POINTS = 21
"""吸引率隨正規化年齡演化曲線的取樣點數（0, 0.05, ..., 1.0）。"""
FIGURE_KINDS = ("spatial", "hotspots", "seasonal", "lifetime", "timeseries")
COLOUR_FIELDS = ("curve_pass_frequency", "core_frequency", "mean_core_s1_day", "mean_lifetime_hours")
"""逐站色階上下限的四個逐格欄位。"""
LIFETIME_THRESHOLD_HOURS = (6.0, 12.0, 24.0)
"""``--lifetime-thresholds-hours`` 的預設值；驅動 ``track_frac_uncensored_ge_<N>h`` 欄位。"""
HOTSPOT_MIN_CELLS = 3
"""``--hotspot-min-cells`` 的預設值：圖面熱點連通塊最少格數（只影響圖面，不影響統計）。"""
SECONDS_PER_DAY = 86_400.0

_KNOWN_TOP_LEVEL_NAMES = ("site_statistics.csv", "site_statistics.json", "manifest.json", "manifest.sha256", "_SUCCESS")


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
    _require(
        manifest.get("null_model") is None,
        f"{domain_id} track manifest 含 null_model（{manifest.get('null_model')}）：null 對照組不可餵入統計",
    )
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
# 網格與站點分析區（site_area）
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
) -> dict[str, np.ndarray]:
    """建立一個 domain 內各研究站的「站點分析區」遮罩（``site_area``）。

    直接沿用 :func:`traps.sites.site_masks`（傳入分析海域遮罩 ``analysis_ocean_mask``）
    的 flow-domain 分割：B／C／D 為整個 flow domain 的分析海域；A 區由貢寮與龜山島依
    最近 anchor 互斥分割。不使用任何接收區半徑。回傳只含該 domain 所屬研究站。
    """

    ocean_mask = np.asarray(grid["analysis_ocean_mask"], dtype=bool)
    partition = site_masks(domain_id, grid["lon"], grid["lat"], ocean_mask, study_design_path=study_design_path)
    domain_site_ids = [str(site["study_site_id"]) for site in design["study_sites"] if site["flow_domain_id"] == domain_id]
    return {site_id: partition[site_id] for site_id in domain_site_ids}


# ---------------------------------------------------------------------------
# Pass 1：逐站強核心門檻（各站只用自己的核心 s1）
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


def _site_core_s1_for_month(
    detect_root: Path,
    domain_id: str,
    month: str,
    valid_snapshot_indices: np.ndarray,
    flat_site_masks: Mapping[str, np.ndarray],
    nx: int,
) -> dict[str, np.ndarray]:
    """只讀 detections.npz 少數欄位，回傳該月份各站分析區內、合格且 canonical valid 的核心 s1。"""

    if valid_snapshot_indices.size == 0:
        return {site_id: np.zeros(0, dtype=np.float64) for site_id in flat_site_masks}
    with np.load(detect_root / domain_id / month / "detections.npz") as archive:
        snapshot_index = np.array(archive["snapshot_index"])
        is_qualified = np.array(archive["is_qualified"])
        core_row = np.array(archive["core_row"])
        core_col = np.array(archive["core_col"])
        core_s1 = np.array(archive["core_s1_sinv"])
    selected = is_qualified & np.isin(snapshot_index, valid_snapshot_indices)
    flat = core_row.astype(np.int64) * np.int64(nx) + core_col.astype(np.int64)
    return {site_id: core_s1[selected & mask_flat[flat]] for site_id, mask_flat in flat_site_masks.items()}


def _thresholds_from_core_values(core_s1_by_site: Mapping[str, np.ndarray], strong_quantile: float) -> dict[str, dict[str, Any]]:
    """各站獨立以自己的核心 s1 計算強核心門檻（s⁻¹ 與 day⁻¹）；站與站之間完全不 pooled。

    ``s1`` 越負代表吸引越強，因此門檻是該站核心 s1 的 ``strong_quantile`` 分位數，
    強核心定義為 ``s1 <= 門檻``。站內沒有合格核心時門檻為 ``None``。
    """

    thresholds: dict[str, dict[str, Any]] = {}
    for site_id, values in core_s1_by_site.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            thresholds[site_id] = {"threshold_sinv": None, "threshold_day": None, "core_count": 0}
            continue
        threshold_sinv = float(np.quantile(finite, strong_quantile))
        thresholds[site_id] = {
            "threshold_sinv": threshold_sinv,
            "threshold_day": threshold_sinv * SECONDS_PER_DAY,
            "core_count": int(finite.size),
        }
    return thresholds


def _compute_site_thresholds(
    detect_root: Path,
    domain_contexts: Mapping[str, dict[str, Any]],
    domain_site_masks: Mapping[str, Mapping[str, np.ndarray]],
    grids: Mapping[str, Mapping[str, Any]],
    strong_quantile: float,
) -> dict[str, dict[str, Any]]:
    """流式讀取每個 domain-month 的 detections，計算各站自己的強核心門檻。"""

    collected: dict[str, list[np.ndarray]] = {}
    for domain_id, context in sorted(domain_contexts.items()):
        nx = int(grids[domain_id]["shape"][1])
        flat_masks = {site_id: mask.reshape(-1) for site_id, mask in domain_site_masks[domain_id].items()}
        for site_id in flat_masks:
            collected.setdefault(site_id, [])
        for month, valid_indices in sorted(context["canonical_valid_by_month"].items()):
            per_site = _site_core_s1_for_month(detect_root, domain_id, month, valid_indices, flat_masks, nx)
            for site_id, values in per_site.items():
                collected[site_id].append(values)
    core_s1_by_site = {
        site_id: (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)) for site_id, chunks in collected.items()
    }
    return _thresholds_from_core_values(core_s1_by_site, strong_quantile)


# ---------------------------------------------------------------------------
# 累加器：site x stratum；空間累加器：site x (season|year)
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
        "track_n": 0,
        "track_lifetimes": [],
        "track_lifetimes_all": [],
        "age_sum_sinv": np.zeros(LIFETIME_EVOLUTION_AGE_POINTS, dtype=np.float64),
        "age_track_count": 0,
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


def _sum_spatial_accumulators(shape: tuple[int, int], accumulators: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """把多個互斥分層（四季）的逐格累加器相加，得到 ``all`` 分層。"""

    total = _new_spatial_accumulator(shape)
    for accumulator in accumulators:
        for key in total:
            total[key] += accumulator[key]
    return total


def _process_domain(
    *,
    domain_id: str,
    detect_root: Path,
    context: dict[str, Any],
    site_masks_2d: Mapping[str, np.ndarray],
    grid: Mapping[str, Any],
    thresholds_sinv: Mapping[str, float | None],
    site_accumulators: dict[tuple[str, str], dict[str, Any]],
    spatial_accumulators: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    """流式處理一個 domain 的每個月份 shard，累加各站 site×stratum 與站內逐格空間統計。

    每個站只看自己分析區內的核心與曲線通過格點；站的有效快照分母要求該站分析區內
    有效格點比例達 ``min_coverage``。強核心用「該站自己的」門檻。空間累加器只記
    季節與年份分層，``all`` 由季節相加取得，避免每個快照重複累加。
    """

    shape = grid["shape"]
    ny, nx = shape
    min_coverage = float(context["min_coverage"])
    canonical = context["canonical"]
    canonical_by_month = context["canonical_valid_by_month"]

    flat_masks = {site_id: mask.reshape(-1) for site_id, mask in site_masks_2d.items()}
    mask_counts = {site_id: int(mask.sum()) for site_id, mask in flat_masks.items()}

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

            cell_values, _row_out = _ragged_gather(det_pass_cell_offsets, rows, det_pass_cells)
            cells_this_snapshot = np.unique(cell_values.astype(np.int64)) if cell_values.size else np.zeros(0, dtype=np.int64)

            for site_id, mask_flat in flat_masks.items():
                mask_count = mask_counts[site_id]
                if mask_count == 0:
                    continue
                site_valid_count = int(np.count_nonzero(valid_bits & mask_flat))
                if (site_valid_count / mask_count) < min_coverage:
                    continue

                threshold = thresholds_sinv.get(site_id)
                core_in_mask = mask_flat[core_flat] if core_flat.size else np.zeros(0, dtype=bool)
                site_core_flat = core_flat[core_in_mask]
                site_core_s1 = core_s1[core_in_mask]
                strong_flags = site_core_s1 <= threshold if threshold is not None else np.zeros(site_core_s1.shape, dtype=bool)
                curve_cells_in_mask = (
                    cells_this_snapshot[mask_flat[cells_this_snapshot]] if cells_this_snapshot.size else cells_this_snapshot
                )
                pass_fraction = float(curve_cells_in_mask.size) / float(mask_count)
                has_core = site_core_flat.size > 0
                has_strong = bool(np.any(strong_flags))
                has_curve = curve_cells_in_mask.size > 0

                for stratum in strata:
                    accumulator = site_accumulators[(site_id, stratum)]
                    accumulator["n_valid"] += 1
                    if has_core:
                        accumulator["core_occurrence"] += 1
                        accumulator["core_s1_values"].append(site_core_s1)
                        accumulator["core_count_total"] += int(site_core_flat.size)
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

                # 站內逐格空間統計（只累加季節／年份；all 由季節相加）
                for stratum in (season, year):
                    spatial = spatial_accumulators[site_id].setdefault(stratum, _new_spatial_accumulator(shape))
                    spatial["n_valid"] += valid_bits.reshape(shape)
                    if has_core:
                        np.add.at(spatial["core_count"].reshape(-1), site_core_flat, 1)
                        np.add.at(spatial["core_s1_sum"].reshape(-1), site_core_flat, site_core_s1)
                        if has_strong:
                            np.add.at(spatial["strong_core_count"].reshape(-1), site_core_flat[strong_flags], 1)
                    if has_curve:
                        spatial["curve_pass_count"].reshape(-1)[curve_cells_in_mask] += 1


# ---------------------------------------------------------------------------
# Track 壽命與吸引強度演化（site x stratum，以及站內逐格未截尾壽命）
# ---------------------------------------------------------------------------


def _track_segments(observations: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """把已依 (track_id, time) 排序的觀測陣列切成逐 track 的連續索引區段。"""

    track_id = observations["track_id"]
    if track_id.size == 0:
        return []
    boundaries = np.nonzero(np.diff(track_id))[0] + 1
    return np.split(np.arange(track_id.size), boundaries)


def _age_profile_sinv(
    times_utc_ns: np.ndarray, core_s1_sinv: np.ndarray, n_points: int = LIFETIME_EVOLUTION_AGE_POINTS
) -> np.ndarray | None:
    """把一條 track 的核心 s1 內插到正規化年齡 0–1 的等距取樣點。

    正規化年齡 = (t - t_first) / (t_last - t_first)，因此 0 是第一個觀測、1 是最後
    一個觀測，不受 track 長短影響（Kunz et al. 2024 Fig. 5b 的作法）。少於兩個觀測或
    時間跨度為零時無法定義年齡，回傳 ``None``。輸入時間必須已遞增排序。
    """

    times = np.asarray(times_utc_ns, dtype=np.float64)
    values = np.asarray(core_s1_sinv, dtype=np.float64)
    if times.size < 2 or values.size != times.size:
        return None
    span = times[-1] - times[0]
    if not span > 0.0:
        return None
    age = (times - times[0]) / span
    return np.interp(np.linspace(0.0, 1.0, n_points), age, values)


def _process_track_lifetimes(
    *,
    context: dict[str, Any],
    site_masks_2d: Mapping[str, np.ndarray],
    grid: Mapping[str, Any],
    site_accumulators: dict[tuple[str, str], dict[str, Any]],
    spatial_accumulators: dict[str, dict[str, dict[str, np.ndarray]]],
    evolution_min_hours: float = LIFETIME_EVOLUTION_MIN_HOURS,
) -> None:
    """以 track_observations／tracks.csv 計算各站壽命統計、演化曲線，並累加站內逐格未截尾壽命。

    track 依「首個落在該站分析區內的觀測」時間指派 stratum；未截尾 track 在該站分析區內
    每個造訪格點計一次逐格壽命（季節與年份分層，all 由季節相加）。壽命 ≥ ``evolution_min_hours``（預設 12）小時的
    未截尾 track 另累加正規化年齡的核心 s1 剖面（只記入 ``all`` 分層）。
    """

    shape = grid["shape"]
    nx = shape[1]
    observations = context["observations"]
    tracks_by_id = context["tracks_by_id"]
    if observations["track_id"].size == 0:
        return

    cols = _nearest_indices(grid["x_m_axis"], observations["core_x_m"])
    rows = _nearest_indices(grid["y_m_axis"], observations["core_y_m"])
    flat = rows.astype(np.int64) * np.int64(nx) + cols.astype(np.int64)
    times = observations["time_utc_ns"]
    obs_s1 = observations["core_s1_sinv"]
    flat_masks = {site_id: mask.reshape(-1) for site_id, mask in site_masks_2d.items()}

    for segment in _track_segments(observations):
        track_id = int(observations["track_id"][segment[0]])
        track = tracks_by_id.get(track_id)
        if track is None:
            continue
        segment_flat = flat[segment]
        segment_times = times[segment]
        is_uncensored = not (track["left_censored"] or track["right_censored"])
        lifetime_hours = float(track["lifetime_hours"])
        profile: np.ndarray | None = None
        profile_ready = False

        for site_id, mask_flat in flat_masks.items():
            in_mask = mask_flat[segment_flat]
            if not np.any(in_mask):
                continue
            first_time = int(segment_times[int(np.nonzero(in_mask)[0][0])])
            season = _season_label(first_time)
            year = _year_label(first_time)
            for stratum in ("all", season, year):
                accumulator = site_accumulators[(site_id, stratum)]
                accumulator["track_n"] += 1
                accumulator["track_lifetimes_all"].append(lifetime_hours)
                if is_uncensored:
                    accumulator["track_lifetimes"].append(lifetime_hours)

            if not is_uncensored:
                continue
            visited_cells = np.unique(segment_flat[in_mask])
            for stratum in (season, year):
                spatial = spatial_accumulators[site_id].setdefault(stratum, _new_spatial_accumulator(shape))
                spatial["lifetime_count"].reshape(-1)[visited_cells] += 1
                spatial["lifetime_sum"].reshape(-1)[visited_cells] += lifetime_hours

            if lifetime_hours >= evolution_min_hours:
                if not profile_ready:
                    profile = _age_profile_sinv(segment_times, obs_s1[segment])
                    profile_ready = True
                if profile is not None:
                    all_accumulator = site_accumulators[(site_id, "all")]
                    all_accumulator["age_sum_sinv"] += profile
                    all_accumulator["age_track_count"] += 1


# ---------------------------------------------------------------------------
# Bootstrap、熱點與最終記錄組裝
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

    def _ci_for(numerator: np.ndarray) -> tuple[float, float] | None:
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


def _top_quantile_mask(field: np.ndarray, site_mask: np.ndarray, quantile: float) -> tuple[np.ndarray, float | None]:
    """回傳站點分析區內 ``field`` 最高分位的格點遮罩與該站自己的分位數門檻。

    只用該站分析區內、非 NaN 的格點計算分位數（不 pooled 其他站）。格點須 ``>= 門檻``
    且 ``> 0``：全域為零的場不會把每個格點都標成熱點；並列於門檻的格點全部納入，因此
    熱點格數可略多於 ``(1 - quantile)`` 比例。站內沒有有效格點時回傳全 False 與 ``None``。
    """

    field = np.asarray(field, dtype=np.float64)
    inside = np.asarray(site_mask, dtype=bool) & np.isfinite(field)
    values = field[inside]
    if values.size == 0:
        return np.zeros(field.shape, dtype=bool), None
    threshold = float(np.quantile(values, quantile))
    return inside & (field >= threshold) & (field > 0.0), threshold


def threshold_label(hours: float) -> str:
    """壽命門檻的欄位標籤：整數小時寫成 ``24``，非整數寫成 ``1.5``。"""

    value = float(hours)
    return f"{int(round(value))}" if value == int(value) else f"{value:g}"


def lifetime_fraction_column(hours: float) -> str:
    """壽命門檻對應的 site_statistics 欄位名 ``track_frac_uncensored_ge_<N>h``。"""

    return f"track_frac_uncensored_ge_{threshold_label(hours)}h"


def remove_small_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    """移除 4-連通格數小於 ``min_cells`` 的連通塊（純 numpy／Python 疊代，不用 scipy）。

    以疊代式 flood fill 標記連通塊；``min_cells <= 1`` 時原樣回傳（複製）。輸入遮罩
    不會被修改。此濾除只供圖面使用，統計欄位與 npz 的原始遮罩保持未濾除定義。
    """

    array = np.asarray(mask, dtype=bool)
    if min_cells <= 1 or not array.any():
        return array.copy()
    n_rows, n_cols = array.shape
    remaining = array.copy()
    kept = np.zeros_like(array)
    for start_row, start_col in zip(*np.nonzero(array), strict=True):
        if not remaining[start_row, start_col]:
            continue
        remaining[start_row, start_col] = False
        stack = [(int(start_row), int(start_col))]
        members: list[tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            members.append((row, col))
            for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= next_row < n_rows and 0 <= next_col < n_cols and remaining[next_row, next_col]:
                    remaining[next_row, next_col] = False
                    stack.append((next_row, next_col))
        if len(members) >= min_cells:
            rows_idx, cols_idx = zip(*members, strict=True)
            kept[list(rows_idx), list(cols_idx)] = True
    return kept


def _site_hotspots(
    accumulator: Mapping[str, np.ndarray],
    finalized: Mapping[str, np.ndarray],
    site_mask: np.ndarray,
    cell_area_m2: np.ndarray,
    hotspot_quantile: float,
    min_cells: int = HOTSPOT_MIN_CELLS,
) -> dict[str, Any]:
    """計算一個站×分層的熱點遮罩與指標。

    熱點 = 該站 ``curve_pass_frequency`` 落在自己分析區內最高 ``1 - hotspot_quantile``
    的格點；另以同一分位數取 ``core_frequency`` 熱點遮罩（供圖面輪廓）。指標：門檻、格數、
    面積（km²）、熱點內平均曲線通過頻率、熱點內核心的平均 s1（day⁻¹，依核心數加權）、
    以及該站強核心落在熱點格點的比例。

    統計與 ``curve_mask``／``core_mask`` 保持未濾除定義；另回傳 ``curve_mask_filtered``／
    ``core_mask_filtered``（移除 4-連通格數 < ``min_cells`` 的小塊），只供圖面與 npz 附加陣列。
    """

    curve_mask, curve_threshold = _top_quantile_mask(finalized["curve_pass_frequency"], site_mask, hotspot_quantile)
    core_mask, core_threshold = _top_quantile_mask(finalized["core_frequency"], site_mask, hotspot_quantile)
    cell_count = int(curve_mask.sum())
    area_km2 = float(np.sum(np.asarray(cell_area_m2, dtype=np.float64)[curve_mask])) / 1.0e6 if cell_count else 0.0
    mean_curve = float(np.mean(finalized["curve_pass_frequency"][curve_mask])) if cell_count else None
    core_count_in_hotspot = int(accumulator["core_count"][curve_mask].sum()) if cell_count else 0
    mean_core_s1_day = (
        float(accumulator["core_s1_sum"][curve_mask].sum() / core_count_in_hotspot * SECONDS_PER_DAY)
        if core_count_in_hotspot
        else None
    )
    strong_total = int(accumulator["strong_core_count"][site_mask].sum())
    strong_in_hotspot = int(accumulator["strong_core_count"][curve_mask].sum()) if cell_count else 0
    return {
        "curve_mask": curve_mask,
        "core_mask": core_mask,
        "curve_mask_filtered": remove_small_components(curve_mask, min_cells),
        "core_mask_filtered": remove_small_components(core_mask, min_cells),
        "core_threshold": core_threshold,
        "metrics": {
            "hotspot_quantile": float(hotspot_quantile),
            "hotspot_threshold_curve_pass_frequency": curve_threshold,
            "hotspot_cell_count": cell_count,
            "hotspot_area_km2": area_km2,
            "hotspot_mean_curve_pass_frequency": mean_curve,
            "hotspot_mean_core_s1_day": mean_core_s1_day,
            "hotspot_strong_core_fraction": (strong_in_hotspot / strong_total) if strong_total else None,
        },
    }


def _finalize_site_record(
    site_id: str,
    stratum: str,
    accumulator: Mapping[str, Any],
    *,
    site_area_km2: float | None,
    site_name_zh: str,
    strong_threshold_sinv: float | None,
    hotspot_metrics: Mapping[str, Any],
    bootstrap_reps: int,
    rng: np.random.Generator,
    lifetime_thresholds_hours: Sequence[float] = LIFETIME_THRESHOLD_HOURS,
) -> dict[str, Any]:
    """把一組 site×stratum 累加器轉成 site_statistics 記錄（``scope`` 固定為 site_area）。"""

    n_valid = int(accumulator["n_valid"])

    def _rate(numerator: int) -> float | None:
        return float(numerator / n_valid) if n_valid else None

    core_s1_values = np.concatenate(accumulator["core_s1_values"]) if accumulator["core_s1_values"] else np.zeros(0)
    lifetimes = accumulator.get("track_lifetimes", [])
    cis = _bootstrap_ci(accumulator["day_stats"], reps=bootstrap_reps, rng=rng) if n_valid else {
        "core_occurrence_rate": None, "strong_core_occurrence_rate": None, "curve_occurrence_rate": None, "mean_pass_cell_fraction": None
    }
    mean_pass_fraction = float(accumulator["pass_fraction_sum"] / n_valid) if n_valid else None

    frac_ge: dict[str, float | None] = {}
    for threshold in lifetime_thresholds_hours:
        n_uncensored = len(lifetimes)
        frac_ge[lifetime_fraction_column(threshold)] = (
            float(sum(1 for value in lifetimes if value >= threshold) / n_uncensored) if n_uncensored else None
        )

    record = {
        "site_id": site_id,
        "site_name_zh": site_name_zh,
        "scope": SITE_SCOPE,
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
        **frac_ge,
    }
    record.update(hotspot_metrics)
    return record


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
    "hotspot_quantile", "hotspot_threshold_curve_pass_frequency", "hotspot_cell_count", "hotspot_area_km2",
    "hotspot_mean_curve_pass_frequency", "hotspot_mean_core_s1_day", "hotspot_strong_core_fraction",
)


def site_stat_fieldnames(lifetime_thresholds_hours: Sequence[float] = LIFETIME_THRESHOLD_HOURS) -> list[str]:
    """依壽命門檻設定產生 CSV 欄位；預設門檻等同 :data:`SITE_STAT_FIELDNAMES`。"""

    default_columns = [lifetime_fraction_column(hours) for hours in LIFETIME_THRESHOLD_HOURS]
    anchor = SITE_STAT_FIELDNAMES.index(default_columns[0])
    columns = [lifetime_fraction_column(hours) for hours in lifetime_thresholds_hours]
    return [*SITE_STAT_FIELDNAMES[:anchor], *columns, *SITE_STAT_FIELDNAMES[anchor + len(default_columns) :]]


def _write_site_statistics(
    output_dir: Path, records: list[dict[str, Any]], lifetime_thresholds_hours: Sequence[float] = LIFETIME_THRESHOLD_HOURS
) -> None:
    """寫出 long-format ``site_statistics.csv`` 與 ``site_statistics.json``。"""

    csv_path = output_dir / "site_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = site_stat_fieldnames(lifetime_thresholds_hours)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})
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


def _mask_to_site(finalized: Mapping[str, np.ndarray], site_mask: np.ndarray) -> dict[str, np.ndarray]:
    """把逐格欄位限制在站點分析區：區外浮點欄位為 NaN，區外有效分母為 0。"""

    result: dict[str, np.ndarray] = {}
    for key, value in finalized.items():
        if np.issubdtype(value.dtype, np.floating):
            result[key] = np.where(site_mask, value, np.nan)
        else:
            result[key] = np.where(site_mask, value, 0)
    return result


def _compute_site_colour_limits(site_all: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    """依「該站自己」分析區內有限格點，計算四個逐格欄位的 2–98 百分位色階上下限（不跨站共用）。"""

    limits: dict[str, list[float]] = {}
    for field in COLOUR_FIELDS:
        values = np.asarray(site_all[field], dtype=np.float64).reshape(-1)
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


def _period_label(domain_context: Mapping[str, dict[str, Any]]) -> str:
    """回傳所有已載入 domain available_months 聯集的最早／最晚月份標籤。"""

    months = sorted({month for context in domain_context.values() for month in context["available_months"]})
    if not months:
        return "no data"
    return f"{months[0]}-{months[-1]}"


def _figure_relative_paths(site_id: str) -> list[str]:
    """一個站的五張成果圖相對路徑（``figures/<site_id>/site_<site_id>_<kind>.png``）。"""

    return [f"figures/{site_id}/site_{site_id}_{kind}.png" for kind in FIGURE_KINDS]


def _resolve_figure_font() -> Any:
    """解析圖面 CJK 字型；找不到合格字型就 fail closed（圖面不可輸出方框字）。"""

    from .report_font import resolve_cjk_font

    return resolve_cjk_font()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_aggregate(args: argparse.Namespace) -> int:
    """讀入已驗證的 track-root／detect-root，逐站獨立計算全期統計、熱點與空間分布。"""

    start = _time.monotonic()
    study_design_path = Path(args.study_design or DEFAULT_STUDY_DESIGN_PATH)
    design = load_study_design(study_design_path)
    all_domain_ids = [str(domain["flow_domain_id"]) for domain in design["flow_domains"]]
    strong_quantile = float(args.strong_quantile)
    hotspot_quantile = float(args.hotspot_quantile)
    _require(0.0 < strong_quantile < 1.0, "--strong-quantile 必須介於 0 與 1 之間（不含端點）")
    _require(0.0 < hotspot_quantile < 1.0, "--hotspot-quantile 必須介於 0 與 1 之間（不含端點）")
    hotspot_min_cells = int(getattr(args, "hotspot_min_cells", HOTSPOT_MIN_CELLS))
    _require(hotspot_min_cells >= 1, "--hotspot-min-cells 必須 >= 1")
    raw_thresholds = getattr(args, "lifetime_thresholds_hours", None)
    lifetime_thresholds = [float(v) for v in (raw_thresholds if raw_thresholds else LIFETIME_THRESHOLD_HOURS)]
    _require(all(v > 0.0 for v in lifetime_thresholds), "--lifetime-thresholds-hours 必須全為正值")
    _require(len({lifetime_fraction_column(v) for v in lifetime_thresholds}) == len(lifetime_thresholds), "--lifetime-thresholds-hours 不可重複")
    evolution_raw = getattr(args, "lifetime_evolution_min_hours", None)
    evolution_min_hours = float(LIFETIME_EVOLUTION_MIN_HOURS if evolution_raw is None else evolution_raw)
    _require(evolution_min_hours > 0.0, "--lifetime-evolution-min-hours 必須為正值")

    track_root = Path(args.track_root)
    detect_root = Path(args.detect_root)
    output_dir = Path(args.output_dir)

    present_domains = [domain_id for domain_id in all_domain_ids if (track_root / domain_id).is_dir()]
    missing_domains = [domain_id for domain_id in all_domain_ids if domain_id not in present_domains]
    if missing_domains and not args.allow_partial_domains:
        raise ValueError(f"缺少 track-root domain，需加 --allow-partial-domains 才可略過: {missing_domains}")
    _require(bool(present_domains), "track-root 沒有任何可用的 domain")

    figures_enabled = not bool(args.no_figures)
    site_domain = {
        str(site["study_site_id"]): str(site["flow_domain_id"])
        for site in design["study_sites"]
        if str(site["flow_domain_id"]) in present_domains
    }
    sites_present = [site_id for site_id in SITE_IDS if site_id in site_domain]
    sites_missing = [site_id for site_id in SITE_IDS if site_id not in site_domain]
    site_name_zh = {str(site["study_site_id"]): str(site["study_site_name_zh"]) for site in design["study_sites"]}
    site_anchor = {str(site["study_site_id"]): (float(site["anchor_lonlat"][0]), float(site["anchor_lonlat"][1])) for site in design["study_sites"]}

    known_names = list(_KNOWN_TOP_LEVEL_NAMES)
    for site_id in sites_present:
        for stratum in ("all", *SEASONS):
            known_names.append(f"spatial/{site_id}/spatial_{stratum}.npz")
        if figures_enabled:
            known_names.extend(_figure_relative_paths(site_id))

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

    # --- 字型 gate：先於任何運算與寫檔，找不到合格 CJK 字型即 fail closed ------------------
    font_selection = _resolve_figure_font() if figures_enabled else None

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

    # --- 網格與站點分析區 ---------------------------------------------------------
    grids: dict[str, dict[str, Any]] = {}
    domain_masks: dict[str, dict[str, np.ndarray]] = {}
    for domain_id in present_domains:
        first_month = sorted(domain_context[domain_id]["available_months"])[0]
        grid = _load_domain_grid(detect_root, domain_id, first_month)
        grids[domain_id] = grid
        domain_masks[domain_id] = _build_domain_masks(domain_id, grid, design, study_design_path)
    site_mask = {site_id: mask for masks in domain_masks.values() for site_id, mask in masks.items()}

    # --- Pass 1：各站自己的強核心門檻 ------------------------------------------------
    site_thresholds = _compute_site_thresholds(detect_root, domain_context, domain_masks, grids, strong_quantile)

    # --- Pass 2：逐 domain 流式累加 --------------------------------------------------
    site_accumulators: dict[tuple[str, str], dict[str, Any]] = defaultdict(_new_site_accumulator)
    spatial_accumulators: dict[str, dict[str, dict[str, np.ndarray]]] = {site_id: {} for site_id in sites_present}
    for domain_id in present_domains:
        thresholds_sinv = {site_id: site_thresholds[site_id]["threshold_sinv"] for site_id in domain_masks[domain_id]}
        _process_domain(
            domain_id=domain_id,
            detect_root=detect_root,
            context=domain_context[domain_id],
            site_masks_2d=domain_masks[domain_id],
            grid=grids[domain_id],
            thresholds_sinv=thresholds_sinv,
            site_accumulators=site_accumulators,
            spatial_accumulators=spatial_accumulators,
        )
        _process_track_lifetimes(
            context=domain_context[domain_id],
            site_masks_2d=domain_masks[domain_id],
            grid=grids[domain_id],
            site_accumulators=site_accumulators,
            spatial_accumulators=spatial_accumulators,
            evolution_min_hours=evolution_min_hours,
        )

    # --- 站點面積 -------------------------------------------------------------------
    site_area_km2: dict[str, float | None] = {}
    for domain_id, masks in domain_masks.items():
        cell_area_m2 = grids[domain_id]["cell_area_m2"]
        for site_id, mask in masks.items():
            area_m2 = float(np.sum(cell_area_m2[mask])) if np.any(mask) else 0.0
            site_area_km2[site_id] = (area_m2 / 1.0e6) if area_m2 > 0.0 else None

    # --- 逐站空間場、熱點與統計記錄 -------------------------------------------------------
    rng = np.random.default_rng(int(args.seed))
    records: list[dict[str, Any]] = []
    site_spatial: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    site_hotspots: dict[str, dict[str, dict[str, Any]]] = {}
    colour_limits: dict[str, dict[str, list[float]]] = {}
    for site_id in sites_present:
        domain_id = site_domain[site_id]
        shape = grids[domain_id]["shape"]
        cell_area_m2 = grids[domain_id]["cell_area_m2"]
        mask = site_mask[site_id]
        strata_acc = spatial_accumulators[site_id]
        spatial_by_stratum = {
            "all": _sum_spatial_accumulators(shape, [strata_acc[s] for s in SEASONS if s in strata_acc]),
        }
        for stratum, accumulator in strata_acc.items():
            spatial_by_stratum[stratum] = accumulator
        for season in SEASONS:
            spatial_by_stratum.setdefault(season, _new_spatial_accumulator(shape))

        strata_present = sorted({key[1] for key in site_accumulators if key[0] == site_id} | {"all"})
        ordered = ["all"] + [s for s in SEASONS if s in strata_present] + [s for s in strata_present if s not in SEASONS and s != "all"]

        site_spatial[site_id] = {}
        site_hotspots[site_id] = {}
        hotspot_metrics_by_stratum: dict[str, dict[str, Any]] = {}
        for stratum in sorted(set(ordered) | {"all", *SEASONS}):
            accumulator = spatial_by_stratum.get(stratum) or _new_spatial_accumulator(shape)
            finalized = _finalize_spatial(shape, accumulator)
            hotspot = _site_hotspots(accumulator, finalized, mask, cell_area_m2, hotspot_quantile, hotspot_min_cells)
            site_spatial[site_id][stratum] = _mask_to_site(finalized, mask)
            site_hotspots[site_id][stratum] = hotspot
            hotspot_metrics_by_stratum[stratum] = hotspot["metrics"]
        colour_limits[site_id] = _compute_site_colour_limits(site_spatial[site_id]["all"])

        for stratum in ordered:
            records.append(
                _finalize_site_record(
                    site_id,
                    stratum,
                    site_accumulators[(site_id, stratum)],
                    site_area_km2=site_area_km2.get(site_id),
                    site_name_zh=site_name_zh.get(site_id, site_id),
                    strong_threshold_sinv=site_thresholds[site_id]["threshold_sinv"],
                    hotspot_metrics=hotspot_metrics_by_stratum[stratum],
                    bootstrap_reps=int(args.bootstrap_reps),
                    rng=rng,
                    lifetime_thresholds_hours=lifetime_thresholds,
                )
            )
    _write_site_statistics(output_dir, records, lifetime_thresholds)

    # --- 逐站 spatial npz ------------------------------------------------------------
    for site_id in sites_present:
        grid = grids[site_domain[site_id]]
        for stratum in ("all", *SEASONS):
            hotspot = site_hotspots[site_id][stratum]
            threshold = site_thresholds[site_id]["threshold_sinv"]
            _atomic_write_npz(
                output_dir / "spatial" / site_id / f"spatial_{stratum}.npz",
                lon=grid["lon"],
                lat=grid["lat"],
                site_mask=site_mask[site_id],
                hotspot_curve_mask=hotspot["curve_mask"],
                hotspot_core_mask=hotspot["core_mask"],
                hotspot_curve_mask_filtered=hotspot["curve_mask_filtered"],
                hotspot_core_mask_filtered=hotspot["core_mask_filtered"],
                strong_core_threshold_sinv=np.float64(np.nan if threshold is None else threshold),
                hotspot_threshold_curve_pass_frequency=np.float64(
                    np.nan if hotspot["metrics"]["hotspot_threshold_curve_pass_frequency"] is None else hotspot["metrics"]["hotspot_threshold_curve_pass_frequency"]
                ),
                hotspot_threshold_core_frequency=np.float64(np.nan if hotspot["core_threshold"] is None else hotspot["core_threshold"]),
                **site_spatial[site_id][stratum],
            )

    # --- 圖面 -------------------------------------------------------------------
    figure_paths: list[str] = []
    if figures_enabled:
        from . import aggregate_figures

        coastline, _coastline_meta = _resolve_coastline_geojson(args)
        period_label = _period_label(domain_context)
        record_index = {(str(r["site_id"]), str(r["stratum"])): r for r in records}
        for site_id in sites_present:
            grid = grids[site_domain[site_id]]
            all_accumulator = site_accumulators[(site_id, "all")]
            age_count = int(all_accumulator["age_track_count"])
            evolution_mean_day = (
                (all_accumulator["age_sum_sinv"] / age_count * SECONDS_PER_DAY).tolist() if age_count else []
            )
            figure_data = aggregate_figures.SiteFigureData(
                site_id=site_id,
                site_name_zh=site_name_zh.get(site_id, site_id),
                lon=grid["lon"],
                lat=grid["lat"],
                site_mask=site_mask[site_id],
                cell_area_m2=grid["cell_area_m2"],
                spatial_all=site_spatial[site_id]["all"],
                spatial_seasons={season: site_spatial[site_id][season] for season in SEASONS},
                hotspot_curve_mask=site_hotspots[site_id]["all"]["curve_mask"],
                hotspot_core_mask=site_hotspots[site_id]["all"]["core_mask"],
                hotspot_metrics=site_hotspots[site_id]["all"]["metrics"],
                hotspot_curve_mask_filtered=site_hotspots[site_id]["all"]["curve_mask_filtered"],
                hotspot_core_mask_filtered=site_hotspots[site_id]["all"]["core_mask_filtered"],
                hotspot_min_cells=hotspot_min_cells,
                lifetime_evolution_min_hours=evolution_min_hours,
                colour_limits=colour_limits[site_id],
                records={stratum: record_index[(site_id, stratum)] for stratum in ("all", *SEASONS) if (site_id, stratum) in record_index},
                lifetimes_uncensored=list(all_accumulator["track_lifetimes"]),
                lifetimes_all=list(all_accumulator["track_lifetimes_all"]),
                evolution_mean_day=evolution_mean_day,
                evolution_track_count=age_count,
                day_stats={day: list(row) for day, row in all_accumulator["day_stats"].items()},
                anchor_lonlat=site_anchor[site_id],
                n_valid_snapshots=int(all_accumulator["n_valid"]),
            )
            figure_paths.extend(
                aggregate_figures.render_site_figures(
                    output_dir,
                    figure_data,
                    font_selection=font_selection,
                    coastline=coastline,
                    state_label=shared["state_label"],
                    param_id=shared["param_id"],
                    period_label=period_label,
                )
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
        "scope": SITE_SCOPE,
        "strong_quantile": strong_quantile,
        "hotspot_quantile": hotspot_quantile,
        "hotspot_min_cells": hotspot_min_cells,
        "hotspot_min_cells_applies_to": "figures_only",
        "lifetime_thresholds_hours": lifetime_thresholds,
        "lifetime_evolution_min_hours": evolution_min_hours,
        "strong_core_thresholds": {
            site_id: {
                "strong_core_threshold_sinv": site_thresholds[site_id]["threshold_sinv"],
                "strong_core_threshold_day": site_thresholds[site_id]["threshold_day"],
                "qualified_core_count": site_thresholds[site_id]["core_count"],
            }
            for site_id in sites_present
        },
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.seed),
        "colour_limits": colour_limits,
        "inputs": inputs_record,
        "study_design_path": str(study_design_path),
        "study_design_sha256": _sha256_file(study_design_path),
        "figures_enabled": figures_enabled,
        "font": font_selection.as_manifest_dict() if font_selection is not None else None,
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


# ---------------------------------------------------------------------------
# argparse 註冊
# ---------------------------------------------------------------------------


def register_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """把 ``aggregate`` 掛到既有 `python -m traps` parser；只新增此子命令。"""

    aggregate = subparsers.add_parser("aggregate", help="Stage 3：跨月份彙整五個獨立研究站的出現率、強度、壽命與熱點統計")
    aggregate.add_argument("--track-root", type=Path, required=True, help="track-period 輸出根目錄（每個 domain 一個子目錄）")
    aggregate.add_argument("--detect-root", type=Path, required=True, help="detect-shards 輸出根目錄（domain/YYYYMM）")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--param-id", help="覆寫輸出 manifest 的 param_id 顯示值；仍須與輸入一致")
    aggregate.add_argument("--study-design", type=Path, default=DEFAULT_STUDY_DESIGN_PATH)
    aggregate.add_argument(
        "--strong-quantile", type=float, default=0.10, help="各站自己的強核心 s1 分位數（不跨站 pooled），預設最強 10%%"
    )
    aggregate.add_argument(
        "--hotspot-quantile", type=float, default=0.90, help="各站 curve_pass_frequency 的熱點分位數，預設最高 10%% 格點"
    )
    aggregate.add_argument(
        "--hotspot-min-cells", type=int, default=HOTSPOT_MIN_CELLS,
        help="成果圖熱點（曲線與核心）移除 4-連通格數小於此值的小塊；只影響圖面，統計與 npz 原始遮罩不變，預設 3",
    )
    aggregate.add_argument(
        "--lifetime-thresholds-hours", type=float, nargs="+", default=list(LIFETIME_THRESHOLD_HOURS), metavar="H",
        help="壽命門檻（小時，可多個），驅動 track_frac_uncensored_ge_<N>h 欄位，預設 6 12 24",
    )
    aggregate.add_argument(
        "--lifetime-evolution-min-hours", type=float, default=LIFETIME_EVOLUTION_MIN_HOURS,
        help="吸引強度演化圖只納入壽命不小於此小時數的未截尾軌跡，預設 12",
    )
    aggregate.add_argument("--bootstrap-reps", type=int, default=1_000)
    aggregate.add_argument("--seed", type=int, default=20_260_928)
    aggregate.add_argument("--coastline-geojson", type=Path, help="成果圖 CRS84 海岸線 GeoJSON；未指定時使用專案預設檔")
    aggregate.add_argument("--no-figures", action="store_true", help="略過成果圖（也不需要 CJK 字型），只輸出統計表與 manifest")
    aggregate.add_argument("--resume", action="store_true")
    aggregate.add_argument("--allow-partial-domains", action="store_true", help="允許 track-root 只有部分 domain")
    aggregate.set_defaults(handler=run_aggregate)


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "register_subcommands",
    "run_aggregate",
]
