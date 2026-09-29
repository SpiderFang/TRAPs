"""跨月份／跨區彙整的第三階段：``aggregate``。

Stage 1（``detect-shards``）與 Stage 2（``track-period``）分別把單一 domain-month
的逐時檢出、以及一個 domain 全期的跨月份 track 寫成可續跑、可驗證的 shard／track
artifact。本模組是第三階段：讀入已完成的 track-root（`traps.period.track.v1`）與
detect-root（`traps.period.shard.v2`，仍可讀 v1：缺少的核心距岸距離由海域遮罩補算），核對已載入 domain 共用同一組 `param_id`、
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

track 壽命的解讀：``track-period --track-strong-quantile q`` 只追蹤各站最強的核心（門檻與本模組的
強核心門檻同一母體定義），此時壽命統計指「強核心結構（各站前 q）」，記錄於 manifest
``track_core_filter`` 與圖表文字。壽命應一律對照 ``--null-track-root`` 的時間置換 null：
逐站在 stratum ``all`` 加上 ``null_``／``excess_``／``ratio_`` 欄位（超額＝真實比例減 null 比例），
壽命圖疊上 null CCDF；null 必須與真實追蹤同設定，僅 seed 不同，否則拒絕。

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
from typing import Any, NamedTuple

import numpy as np

from .cli import _resolve_coastline_geojson, _sha256_file
from .period import (
    ACCEPTED_SHARD_SCHEMA_VERSIONS,
    COAST_DISTANCE_CAP_M,
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
    _canonical_valid_snapshot_indices_by_month,
    _site_core_s1_for_month as _period_site_core_s1_for_month,
    _thresholds_from_core_values,
    compute_coast_distance_field,
    sample_coast_distance,
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
EXAMPLE_FIGURE_KINDS = ("example", "method_comparison")
"""需重開 OCM snapshot 的圖（``--no-example-figures`` 略過）。"""
TRACK_FIGURE_KINDS = ("long_tracks", "track_density", "propagation")
"""只用 track／spatial 產物的圖。"""
COASTAL_FIGURE_KINDS = ("example_coastal", "coastal_diagnostic")
"""WP-I 近岸診斷圖：近岸型代表 track（需重開 OCM，可能不存在）與近岸診斷統計圖。"""
COLOUR_FIELDS = ("curve_pass_frequency", "core_frequency", "mean_core_s1_day", "mean_lifetime_hours")
"""逐站色階上下限的四個逐格欄位。"""
LIFETIME_THRESHOLD_HOURS = (6.0, 12.0, 24.0)
"""``--lifetime-thresholds-hours`` 的預設值；驅動 ``track_frac_uncensored_ge_<N>h`` 欄位。"""
HOTSPOT_MIN_CELLS = 3
"""``--hotspot-min-cells`` 的預設值：圖面熱點連通塊最少格數（只影響圖面，不影響統計）。"""
SECONDS_PER_DAY = 86_400.0
COASTAL_BUFFER_KM = 10.0
"""``--coastal-buffer-km`` 的預設值：近岸緩衝帶寬度（核心距最近非分析海域格點的距離）。

核心的偵測需要完整有效的導數模板與局部極小鄰域，實際上不可能落在距非海域格點約 2.5–3 km 以內（檢出下限），
故 3 km 緩衝帶只剔除核心本來就不存在的區域；強核心與長壽命 track 集中在 3–10 km，預設取 10 km 才真正檢驗近岸影響。"""
PROP_MIN_OBS = 3
"""``--prop-min-obs`` 的下限（及日資料的預設值）：傳播速度線性擬合所需的最少觀測數。"""
PROP_MIN_WINDOW_HOURS = 6.0
"""次日步長資料自動最少觀測數所要求的擬合時間窗（小時）。"""


def auto_prop_min_obs(nominal_step_hours: float) -> int:
    """``--prop-min-obs`` 未指定時的有效值：步長 < 24 h 為 ``max(3, ceil(6 h / step)) + 1``（逐時 7），日資料為 3。"""

    import math

    step = float(nominal_step_hours)
    if step >= 24.0:
        return PROP_MIN_OBS
    return max(PROP_MIN_OBS, math.ceil(PROP_MIN_WINDOW_HOURS / step - 1e-9)) + 1


def resolve_prop_min_obs(raw: int | None, nominal_step_hours: float) -> int:
    """明確指定值（不低於 3）優先；``None`` 則依名目步長自動決定。"""

    return auto_prop_min_obs(nominal_step_hours) if raw is None else max(PROP_MIN_OBS, int(raw))
COAST_HIST_STEP_KM = 2.0
LEGACY_COAST_CAP_M = 30_000.0
"""舊 shard（v1／早期 v2）距離場的飽和上限。"""
COAST_HIST_EDGES_KM = np.arange(0.0, COAST_DISTANCE_CAP_M / 1000.0 + 2.0 * COAST_HIST_STEP_KM, COAST_HIST_STEP_KM)
"""距岸直方圖 bin 邊界：0 至距離場上限每 2 km 一格，最後多一格「≥ 上限」溢位 bin（含飽和值）。"""
"""近岸診斷圖直方圖的固定 bin 邊界（km）；超過距離場飽和上限的值落入最後一格。"""

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


def _load_track_domain(track_root: Path, domain_id: str, *, expect_null: bool = False) -> dict[str, Any]:
    """讀取並驗證一個 domain 的 stage-2 `track-period` 輸出。

    主要 ``--track-root`` 必須是真實追蹤（``null_model`` 為空）；``--null-track-root`` 則以
    ``expect_null=True`` 載入，且必須帶 ``null_model``（時間置換對照組）。
    """

    domain_dir = track_root / domain_id
    manifest = _success_verified(domain_dir)
    _require(manifest is not None, f"track-root 缺少已驗證的 {domain_id}（缺 _SUCCESS 或 SHA-256 不符）: {domain_dir}")
    assert manifest is not None
    _require(manifest.get("schema_version") == TRACK_SCHEMA_VERSION, f"{domain_id} track manifest schema_version 不符")
    _require(manifest.get("domain_id") == domain_id, f"{domain_id} track manifest domain_id 不符")
    if expect_null:
        _require(
            isinstance(manifest.get("null_model"), Mapping) and manifest["null_model"].get("type") == "time_shuffle",
            f"{domain_id} 的 --null-track-root 輸入缺少 null_model（不是 --shuffle-seed 的輸出）: {domain_dir}",
        )
    else:
        _require(
            manifest.get("null_model") is None,
            f"{domain_id} track manifest 含 null_model（{manifest.get('null_model')}）：null 對照組不可餵入統計，請改用 --null-track-root",
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
        _require(
            manifest.get("schema_version") in ACCEPTED_SHARD_SCHEMA_VERSIONS,
            f"{shard_dir} schema_version 不受支援（需要 {ACCEPTED_SHARD_SCHEMA_VERSIONS}）",
        )
        expected_sha256 = recorded_sha256.get(month)
        _require(expected_sha256 is not None, f"track-period manifest 沒有記錄 {domain_id}/{month} 的 shard manifest")
        actual_sha256 = _sha256_file(shard_dir / "manifest.json")
        _require(
            actual_sha256 == expected_sha256,
            f"detect-root shard manifest 遭竄改或與 track-period 記錄不符: {shard_dir}/manifest.json",
        )
        manifests[month] = manifest
    return manifests


def _filter_quantile(track_core_filter: Mapping[str, Any] | None) -> float | None:
    """``track_core_filter`` 記錄的分位數；沒有篩選（追蹤全部合格核心）時為 ``None``。"""

    if not track_core_filter:
        return None
    return float(track_core_filter["quantile"])


def lifetime_scope_text(track_core_filter: Mapping[str, Any] | None) -> str:
    """壽命統計對象的中文說明（圖表與 site_statistics.json 共用同一字串）。"""

    from .report_font import FIGURE_TEXT as T

    quantile = _filter_quantile(track_core_filter)
    if quantile is None:
        return T["all_qualified_structures"]
    return f"{T['strong_core_structures']}（{T['per_site_top']} {quantile * 100.0:g}%）"


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
        _require(
            _filter_quantile(context["track_core_filter"]) == _filter_quantile(first["track_core_filter"]),
            f"{domain_id} 與 {first_id} 的 track_core_filter（--track-strong-quantile）不一致，無法合併",
        )
    merged_thresholds: dict[str, Any] = {}
    for context in domain_context.values():
        track_filter = context["track_core_filter"]
        if track_filter:
            merged_thresholds.update(track_filter.get("thresholds") or {})
    first_filter = first["track_core_filter"]
    return {
        "track_core_filter": (
            None
            if not first_filter
            else {
                "type": first_filter.get("type"),
                "quantile": float(first_filter["quantile"]),
                "population": first_filter.get("population"),
                "thresholds": merged_thresholds,
            }
        ),
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
    if "coast_distance_m" not in grid:
        # v1 shard 沒有近岸距離場：由 analysis_ocean_mask 與座標軸即時計算（與 v2 定義相同）。
        grid["coast_distance_m"] = compute_coast_distance_field(grid["analysis_ocean_mask"], grid["x_m_axis"], grid["y_m_axis"])
        grid["coast_distance_computed"] = True
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


def _site_core_s1_for_month(
    detect_root: Path,
    domain_id: str,
    month: str,
    valid_snapshot_indices: np.ndarray,
    flat_site_masks: Mapping[str, np.ndarray],
    nx: int,
) -> dict[str, np.ndarray]:
    """該月份各站分析區內、合格且 canonical valid 的核心 s1。

    母體定義與 ``track-period --track-strong-quantile`` 共用 :func:`traps.period._site_core_s1_for_month`
    （以及 ``_thresholds_from_core_values``、``_canonical_valid_snapshot_indices_by_month``），
    因此兩處的逐站強核心門檻必然相同。
    """

    return _period_site_core_s1_for_month(detect_root / domain_id / month, valid_snapshot_indices, flat_site_masks, nx)


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
        "prop_tracks": [],  # 每條已擬合 track：(cx_kmday, cy_kmday, resid_rms_m, displacement_m, stationary)
        "born_uncensored": [],
        # --- 近岸診斷（WP-I）---
        "core_near_count": 0,  # 站內合格核心中，距岸小於緩衝帶者
        "strong_core_count_total": 0,
        "strong_core_near_count": 0,
        "offshore_strong_occurrence": 0,  # 含至少一個近岸外強核心的有效快照數
        "coast_hist_all": np.zeros(COAST_HIST_EDGES_KM.size - 1, dtype=np.int64),
        "coast_hist_strong": np.zeros(COAST_HIST_EDGES_KM.size - 1, dtype=np.int64),
        "coast_hist_long": np.zeros(COAST_HIST_EDGES_KM.size - 1, dtype=np.int64),
        "core_coast_min_m": float("inf"),  # 站內合格核心的最小距岸距離（檢出下限）
        "track_lifetimes_offshore": [],  # 未截尾 track，平均核心距岸不小於緩衝帶
        "track_lifetimes_coastal": [],  # 未截尾 track，平均核心距岸小於緩衝帶
        "long_track_n": 0,  # 站內出生、未截尾、壽命 >= 長壽命門檻
        "long_track_coastal_n": 0,
        "coast_info": False,  # 是否有近岸資訊（沒有時上列近岸欄位不輸出）
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
        "track_count": np.zeros(shape, dtype=np.int64),
        "long_track_count": np.zeros(shape, dtype=np.int64),
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
    coast_buffer_m: float | None = None,
) -> None:
    """流式處理一個 domain 的每個月份 shard，累加各站 site×stratum 與站內逐格空間統計。

    每個站只看自己分析區內的核心與曲線通過格點；站的有效快照分母要求該站分析區內
    有效格點比例達 ``min_coverage``。強核心用「該站自己的」門檻。空間累加器只記
    季節與年份分層，``all`` 由季節相加取得，避免每個快照重複累加。

    給定 ``coast_buffer_m`` 時另累加近岸診斷：站內核心／強核心落在近岸緩衝帶內的數量、核心距岸直方圖、
    含近岸外強核心的快照數（強核心門檻不重算，仍是該站全部核心的分位數）。核心距岸距離優先讀 v2 shard 的
    ``core_coast_distance_m``；v1 shard 由 ``grid`` 的距離場在核心位置取樣。
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
            if "core_coast_distance_m" in archive.files:
                det_core_coast = np.array(archive["core_coast_distance_m"])
            else:
                det_core_coast = sample_coast_distance(
                    grid["coast_distance_m"], grid["x_m_axis"], grid["y_m_axis"], np.array(archive["core_x_m"]), np.array(archive["core_y_m"])
                )

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
            core_coast = det_core_coast[rows]

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
                if coast_buffer_m is not None:
                    site_core_coast = core_coast[core_in_mask]
                    near = np.isfinite(site_core_coast) & (site_core_coast < coast_buffer_m)
                    finite_coast = np.isfinite(site_core_coast)
                    coast_km = np.clip(site_core_coast[finite_coast] / 1000.0, 0.0, COAST_HIST_EDGES_KM[-1] - 1.0e-9)
                    hist_all = np.histogram(coast_km, bins=COAST_HIST_EDGES_KM)[0]
                    strong_coast = site_core_coast[strong_flags & finite_coast]
                    hist_strong = np.histogram(
                        np.clip(strong_coast / 1000.0, 0.0, COAST_HIST_EDGES_KM[-1] - 1.0e-9), bins=COAST_HIST_EDGES_KM
                    )[0]
                    min_coast_m = float(site_core_coast[finite_coast].min()) if finite_coast.any() else float("inf")
                    n_near = int(near.sum())
                    n_strong = int(strong_flags.sum())
                    n_strong_near = int((strong_flags & near).sum())
                    has_offshore_strong = bool(np.any(strong_flags & ~near))

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
                    if coast_buffer_m is not None:
                        accumulator["coast_info"] = True
                        accumulator["core_near_count"] += n_near
                        accumulator["strong_core_count_total"] += n_strong
                        accumulator["strong_core_near_count"] += n_strong_near
                        accumulator["offshore_strong_occurrence"] += int(has_offshore_strong)
                        accumulator["coast_hist_all"] += hist_all
                        accumulator["core_coast_min_m"] = min(accumulator["core_coast_min_m"], min_coast_m)
                        accumulator["coast_hist_strong"] += hist_strong

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


M_PER_S_TO_KM_PER_DAY = SECONDS_PER_DAY / 1000.0
"""公尺／秒換算公里／日（86.4）。"""
LONG_TRACK_DRAW_LIMIT = 2000
"""``site_<id>_long_tracks.png`` 最多繪製的 track 數（取壽命最長者）。"""


class PropagationFit(NamedTuple):
    """單一 track 核心位置的線性擬合結果。"""

    cx_kmday: float
    cy_kmday: float
    resid_rms_m: float
    displacement_m: float
    n_obs: int


def fit_track_velocity(times_utc_ns: np.ndarray, core_x_m: np.ndarray, core_y_m: np.ndarray) -> PropagationFit | None:
    """以最小平方線性擬合 ``x(t)``、``y(t)`` 估計整條 track 的核心傳播速度（km/day）。

    核心位置為公尺制 ``core_x_m``（東向）與 ``core_y_m``（北向），時間取實際 UTC 差（跨缺口仍正確）。
    對每個座標個別擬合 ``p(t) = p0 + c (t - t_mean)``，回傳擬合速度 ``(cx, cy)``、位置殘差 RMS
    （兩座標平方殘差平均後開根號，m）與擬合位移 ``hypot(cx, cy) * (t_last - t_first)``（m）。
    少於 2 個觀測或時間跨度非正時回傳 ``None``。這是模型內 TRAP 核心（結構）的位移速度，不是水體或漂浮物速度；
    與單步差分不同，整條 track 的線性擬合對核心位置的次網格抖動不敏感，位移小於 1 格的 track 視為靜止。
    """

    times = np.asarray(times_utc_ns, dtype=np.int64)
    x = np.asarray(core_x_m, dtype=np.float64)
    y = np.asarray(core_y_m, dtype=np.float64)
    if times.size < 2 or x.size != times.size or y.size != times.size:
        return None
    seconds = (times - times[0]).astype(np.float64) / 1.0e9
    centred_t = seconds - float(seconds.mean())
    denominator = float(np.sum(centred_t * centred_t))
    span = float(seconds[-1] - seconds[0])
    if not denominator > 0.0 or not span > 0.0:
        return None
    x_centred = x - float(x.mean())
    y_centred = y - float(y.mean())
    slope_x = float(np.sum(centred_t * x_centred) / denominator)
    slope_y = float(np.sum(centred_t * y_centred) / denominator)
    resid_x = x_centred - slope_x * centred_t
    resid_y = y_centred - slope_y * centred_t
    rms = float(np.sqrt(np.mean(resid_x * resid_x + resid_y * resid_y)))
    return PropagationFit(
        cx_kmday=slope_x * M_PER_S_TO_KM_PER_DAY,
        cy_kmday=slope_y * M_PER_S_TO_KM_PER_DAY,
        resid_rms_m=rms,
        displacement_m=float(np.hypot(slope_x, slope_y) * span),
        n_obs=int(times.size),
    )


def grid_cell_size_m(grid: Mapping[str, Any]) -> float:
    """格點尺寸（m）：x、y 軸中位格距的平均；靜止 track 的位移門檻為一格。"""

    x_axis = np.asarray(grid["x_m_axis"], dtype=np.float64)
    y_axis = np.asarray(grid["y_m_axis"], dtype=np.float64)
    steps = [float(np.median(np.abs(np.diff(axis)))) for axis in (x_axis, y_axis) if axis.size > 1]
    return float(np.mean(steps)) if steps else 1.0


def select_representative_track(
    candidates: Sequence[Mapping[str, Any]],
    min_lifetime_hours: float | None,
    *,
    coastal_buffer_m: float | None = None,
    null_p99_hours: float | None = None,
) -> tuple[Mapping[str, Any] | None, str]:
    """從站內出生的未截尾真實 track 選出代表 track。

    ``candidates`` 每筆含 ``track_id``、``lifetime_hours``、``mean_core_s1_sinv``；有近岸資訊時另含
    ``mean_coast_m``（track 全部觀測的平均核心距岸距離）。排序規則一律為：壽命最長，同壽命取平均核心 s1 最強
    （最負），再同則取較小 ``track_id``（決定性）。

    * 未給 ``coastal_buffer_m``（舊行為）：若有壽命不小於 ``min_lifetime_hours`` 的候選，只在其中選，準則為
      ``ge_threshold``／``longest_available``／``no_candidate``。
    * 給 ``coastal_buffer_m``：優先取「平均距岸不小於緩衝帶」且（給定 ``null_p99_hours`` 時）壽命嚴格大於隨機對照
      p99 的最長者（準則 ``offshore_exceeds_null``；無對照時 ``offshore_longest``）。沒有這樣的候選就退回整體最長者，
      準則明示為 ``fallback_coastal``（該 track 屬近岸型）或 ``fallback_offshore_below_null``（近岸外但壽命未超過對照）。
    """

    if not candidates:
        return None, "no_candidate"
    pool = list(candidates)

    def _best(items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        return min(items, key=lambda item: (-float(item["lifetime_hours"]), float(item["mean_core_s1_sinv"]), int(item["track_id"])))

    if coastal_buffer_m is not None:
        offshore = [
            item for item in pool if item.get("mean_coast_m") is not None and not float(item["mean_coast_m"]) < float(coastal_buffer_m)
        ]
        strict = offshore
        if null_p99_hours is not None:
            strict = [item for item in offshore if float(item["lifetime_hours"]) > float(null_p99_hours)]
        if strict:
            return _best(strict), ("offshore_exceeds_null" if null_p99_hours is not None else "offshore_longest")
        chosen = _best(pool)
        is_coastal = chosen.get("mean_coast_m") is not None and float(chosen["mean_coast_m"]) < float(coastal_buffer_m)
        return chosen, ("fallback_coastal" if is_coastal else "fallback_offshore_below_null")

    criterion = "longest_available"
    if min_lifetime_hours is not None:
        restricted = [item for item in pool if float(item["lifetime_hours"]) >= float(min_lifetime_hours)]
        if restricted:
            pool, criterion = restricted, "ge_threshold"
    return _best(pool), criterion


def select_coastal_track(
    candidates: Sequence[Mapping[str, Any]], coastal_buffer_m: float
) -> Mapping[str, Any] | None:
    """近岸型代表 track：平均核心距岸小於緩衝帶者中壽命最長（同壽命取核心最強，再取較小 track_id）；沒有則 ``None``。"""

    coastal = [
        item for item in candidates if item.get("mean_coast_m") is not None and float(item["mean_coast_m"]) < float(coastal_buffer_m)
    ]
    if not coastal:
        return None
    return min(coastal, key=lambda item: (-float(item["lifetime_hours"]), float(item["mean_core_s1_sinv"]), int(item["track_id"])))


def _observation_coast(observations: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """觀測的核心距岸距離（m）；沒有該欄位（舊 track 輸出且尚未補算）時為 ``None``。"""

    values = observations.get("core_coast_distance_m")
    return None if values is None else np.asarray(values, dtype=np.float64)


def _mean_coast_m(values: np.ndarray) -> float | None:
    """一條 track 的平均核心距岸距離；全部缺值時為 ``None``。"""

    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else None


def ensure_observation_coast(context: dict[str, Any], grid: Mapping[str, Any]) -> None:
    """舊 track 輸出（沒有 ``core_coast_distance_m``）時，以 domain 的距離場在觀測核心位置補算。"""

    observations = context["observations"]
    if "core_coast_distance_m" in observations:
        return
    observations["core_coast_distance_m"] = sample_coast_distance(
        grid["coast_distance_m"], grid["x_m_axis"], grid["y_m_axis"], observations["core_x_m"], observations["core_y_m"]
    )


def _process_track_lifetimes(
    *,
    context: dict[str, Any],
    site_masks_2d: Mapping[str, np.ndarray],
    grid: Mapping[str, Any],
    site_accumulators: dict[tuple[str, str], dict[str, Any]],
    spatial_accumulators: dict[str, dict[str, dict[str, np.ndarray]]],
    evolution_min_hours: float = LIFETIME_EVOLUTION_MIN_HOURS,
    long_track_hours: float | None = None,
    prop_min_obs: int = PROP_MIN_OBS,
    coast_buffer_m: float | None = None,
) -> None:
    """以 track_observations／tracks.csv 計算各站壽命統計、演化曲線、傳播速度，並累加站內逐格分布。

    track 依「首個落在該站分析區內的觀測」時間指派 stratum；未截尾 track 在該站分析區內
    每個造訪格點計一次逐格壽命（季節與年份分層，all 由季節相加）。壽命 ≥ ``evolution_min_hours``（預設 12）小時的
    未截尾 track 另累加正規化年齡的核心 s1 剖面（只記入 ``all`` 分層）。

    另累加：(1) 逐格「通過該格的不同 track 數」``track_count``（全部真實 track，含截尾；每 track 每格計一次，
    Kunz et al. 2024 Fig. 4c）與 ``long_track_count``（未截尾且壽命 ≥ ``long_track_hours``；``None`` 則不計）；
    (2) 首個觀測在該站分析區內的未截尾 track 記錄 ``born_uncensored``（代表 track 與長壽命路徑圖用）；
    (3) 至少 ``max(3, prop_min_obs)`` 個觀測的 track，以全部觀測的線性擬合估計傳播速度（``fit_track_velocity``），
    擬合位移小於 1 格（``grid_cell_size_m``）者標為靜止（僅 ``all`` 分層）。

    給定 ``coast_buffer_m`` 且 track 觀測含 ``core_coast_distance_m`` 時，另以 track 全部觀測的平均距岸距離判定
    近岸型（小於緩衝帶）或近岸外，累加各分層的近岸型／近岸外未截尾壽命、長壽命 track 的近岸型數量，以及長壽命
    track 觀測的距岸直方圖；``born_uncensored`` 記錄附 ``mean_coast_m``。
    """

    shape = grid["shape"]
    nx = shape[1]
    observations = context["observations"]
    tracks_by_id = context["tracks_by_id"]
    if observations["track_id"].size == 0:
        return

    obs_coast = _observation_coast(observations) if coast_buffer_m is not None else None
    min_obs = max(PROP_MIN_OBS, int(prop_min_obs))
    stationary_cell_m = grid_cell_size_m(grid)
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
        is_long = bool(is_uncensored and long_track_hours is not None and lifetime_hours >= float(long_track_hours))
        profile: np.ndarray | None = None
        profile_ready = False
        fit: PropagationFit | None = None
        fit_ready = False
        mean_coast: float | None = None
        is_coastal = False
        if obs_coast is not None:
            mean_coast = _mean_coast_m(obs_coast[segment])
            is_coastal = bool(mean_coast is not None and mean_coast < float(coast_buffer_m))

        for site_id, mask_flat in flat_masks.items():
            in_mask = mask_flat[segment_flat]
            if not np.any(in_mask):
                continue
            first_time = int(segment_times[int(np.nonzero(in_mask)[0][0])])
            season = _season_label(first_time)
            year = _year_label(first_time)
            visited_cells = np.unique(segment_flat[in_mask])
            for stratum in ("all", season, year):
                accumulator = site_accumulators[(site_id, stratum)]
                accumulator["track_n"] += 1
                accumulator["track_lifetimes_all"].append(lifetime_hours)
                if is_uncensored:
                    accumulator["track_lifetimes"].append(lifetime_hours)
                    if obs_coast is not None:
                        accumulator["coast_info"] = True
                        accumulator["track_lifetimes_coastal" if is_coastal else "track_lifetimes_offshore"].append(lifetime_hours)
            for stratum in (season, year):
                spatial = spatial_accumulators[site_id].setdefault(stratum, _new_spatial_accumulator(shape))
                spatial["track_count"].reshape(-1)[visited_cells] += 1
                if is_long:
                    spatial["long_track_count"].reshape(-1)[visited_cells] += 1

            all_accumulator = site_accumulators[(site_id, "all")]
            if segment.size >= min_obs:
                if not fit_ready:
                    fit = fit_track_velocity(segment_times, observations["core_x_m"][segment], observations["core_y_m"][segment])
                    fit_ready = True
                if fit is not None:
                    all_accumulator["prop_tracks"].append(
                        (fit.cx_kmday, fit.cy_kmday, fit.resid_rms_m, fit.displacement_m, bool(fit.displacement_m < stationary_cell_m))
                    )

            if not is_uncensored:
                continue
            if bool(in_mask[0]):
                all_accumulator["born_uncensored"].append(
                    {
                        "track_id": track_id,
                        "lifetime_hours": lifetime_hours,
                        "mean_core_s1_sinv": float(np.mean(obs_s1[segment])),
                        "n_obs": int(segment.size),
                        "start": int(segment[0]),
                        "stop": int(segment[-1]) + 1,
                        "mean_coast_m": mean_coast,
                        "coastal": is_coastal,
                    }
                )
                if is_long and obs_coast is not None:
                    long_coast_km = np.clip(
                        obs_coast[segment][np.isfinite(obs_coast[segment])] / 1000.0, 0.0, COAST_HIST_EDGES_KM[-1] - 1.0e-9
                    )
                    long_hist = np.histogram(long_coast_km, bins=COAST_HIST_EDGES_KM)[0]
                    for stratum in ("all", season, year):
                        accumulator = site_accumulators[(site_id, stratum)]
                        accumulator["coast_info"] = True
                        accumulator["long_track_n"] += 1
                        accumulator["long_track_coastal_n"] += int(is_coastal)
                        accumulator["coast_hist_long"] += long_hist
            for stratum in (season, year):
                spatial = spatial_accumulators[site_id][stratum]
                spatial["lifetime_count"].reshape(-1)[visited_cells] += 1
                spatial["lifetime_sum"].reshape(-1)[visited_cells] += lifetime_hours

            if lifetime_hours >= evolution_min_hours:
                if not profile_ready:
                    profile = _age_profile_sinv(segment_times, obs_s1[segment])
                    profile_ready = True
                if profile is not None:
                    all_accumulator["age_sum_sinv"] += profile
                    all_accumulator["age_track_count"] += 1


# ---------------------------------------------------------------------------
# 時間置換 null 對照（--null-track-root）：同一規則的壽命統計與超額
# ---------------------------------------------------------------------------


def _null_track_records_by_site(
    context: Mapping[str, Any],
    site_masks_2d: Mapping[str, np.ndarray],
    grid: Mapping[str, Any],
    coast_buffer_m: float | None = None,
) -> dict[str, dict[str, list[Any]]]:
    """依與真實 track 完全相同的規則，回傳各站（stratum ``all``）null 未截尾 track 的壽命與近岸型旗標。

    每站回傳四個平行清單：

    * ``lifetimes`` 與 ``coastal``：「首個落在站內」規則之外，只要有任一觀測在站內即屬於該站的未截尾 track
      （:func:`_process_track_lifetimes` 中 ``all`` 分層 ``track_lifetimes`` 的獨立重現）；
    * ``born_lifetimes`` 與 ``born_coastal``：首個觀測落在站點分析區內的未截尾 track（長壽命 track 的 null 對照用）。

    ``coastal`` 為 track 平均核心距岸小於 ``coast_buffer_m``（沒有近岸資訊時一律 ``False``）。
    """

    nx = grid["shape"][1]
    observations = context["observations"]
    tracks_by_id = context["tracks_by_id"]
    result: dict[str, dict[str, list[Any]]] = {
        site_id: {"lifetimes": [], "coastal": [], "born_lifetimes": [], "born_coastal": []} for site_id in site_masks_2d
    }
    if observations["track_id"].size == 0:
        return result
    obs_coast = _observation_coast(observations) if coast_buffer_m is not None else None
    cols = _nearest_indices(grid["x_m_axis"], observations["core_x_m"])
    rows = _nearest_indices(grid["y_m_axis"], observations["core_y_m"])
    flat = rows.astype(np.int64) * np.int64(nx) + cols.astype(np.int64)
    flat_masks = {site_id: mask.reshape(-1) for site_id, mask in site_masks_2d.items()}
    for segment in _track_segments(observations):
        track = tracks_by_id.get(int(observations["track_id"][segment[0]]))
        if track is None or track["left_censored"] or track["right_censored"]:
            continue
        segment_flat = flat[segment]
        is_coastal = False
        if obs_coast is not None:
            mean_coast = _mean_coast_m(obs_coast[segment])
            is_coastal = bool(mean_coast is not None and mean_coast < float(coast_buffer_m))
        first_flat = int(segment_flat[0])
        for site_id, mask_flat in flat_masks.items():
            if np.any(mask_flat[segment_flat]):
                result[site_id]["lifetimes"].append(float(track["lifetime_hours"]))
                result[site_id]["coastal"].append(is_coastal)
            if bool(mask_flat[first_flat]):
                result[site_id]["born_lifetimes"].append(float(track["lifetime_hours"]))
                result[site_id]["born_coastal"].append(is_coastal)
    return result


def _validate_null_against_real(
    real: Mapping[str, Any], null: Mapping[str, Any], *, domain_id: str, null_root: Path
) -> None:
    """null 對照組必須與真實追蹤同 detection／tracking 設定與 track_core_filter，僅 seed 不同。"""

    real_manifest, null_manifest = real["manifest"], null["manifest"]
    where = f"--null-track-root {null_root} / {domain_id}"
    for key in ("param_id", "state_label", "tracking_config", "nominal_time_step_hours", "available_months"):
        _require(real_manifest.get(key) == null_manifest.get(key), f"{where}: {key} 與真實 track-root 不一致，拒絕作為對照組")
    # shard manifest 的 SHA-256 完全相同 => detection_config／min_coverage／偵測輸入相同。
    _require(
        real_manifest.get("shard_manifests") == null_manifest.get("shard_manifests"),
        f"{where}: shard_manifests（偵測輸入與 detection_config）與真實 track-root 不一致，拒絕作為對照組",
    )
    real_filter, null_filter = real_manifest.get("track_core_filter"), null_manifest.get("track_core_filter")
    _require(
        _filter_quantile(real_filter) == _filter_quantile(null_filter),
        f"{where}: track_core_filter（--track-strong-quantile）與真實 track-root 不一致，拒絕作為對照組",
    )
    if real_filter:
        real_thresholds = {site: value.get("threshold_sinv") for site, value in (real_filter.get("thresholds") or {}).items()}
        null_thresholds = {site: value.get("threshold_sinv") for site, value in (null_filter.get("thresholds") or {}).items()}
        _require(real_thresholds == null_thresholds, f"{where}: track_core_filter 各站門檻與真實 track-root 不一致，拒絕作為對照組")


def _lifetime_summary(lifetimes: Sequence[float], thresholds_hours: Sequence[float]) -> dict[str, Any]:
    """一組未截尾壽命的 n、median、p90 與各門檻的 frac(>= h)；n=0 時統計為 ``None``。"""

    n = len(lifetimes)
    values = np.asarray(lifetimes, dtype=np.float64)
    return {
        "n": n,
        "median": _quantile_or_none(lifetimes, 0.5),
        "p90": _quantile_or_none(lifetimes, 0.9),
        "frac": {lifetime_fraction_column(h): (float(np.mean(values >= h)) if n else None) for h in thresholds_hours},
    }


def null_column_names(lifetime_thresholds_hours: Sequence[float]) -> list[str]:
    """``--null-track-root`` 給定時附加在 site_statistics 的欄位（只有 stratum ``all`` 有值）。"""

    fractions = [lifetime_fraction_column(h) for h in lifetime_thresholds_hours]
    return [
        "null_n_seeds",
        "null_n_uncensored",
        "null_median_lifetime_hours",
        "null_p90_lifetime_hours",
        *[f"null_{name}" for name in fractions],
        *[f"excess_{name}" for name in fractions],
        *[f"ratio_{name}" for name in fractions],
        *_null_group_column_names(lifetime_thresholds_hours),
    ]


COAST_GROUPS = ("offshore", "coastal")
"""track 依平均核心距岸距離分成近岸外（不小於緩衝帶）與近岸型（小於緩衝帶）。"""


def group_fraction_column(group: str, hours: float) -> str:
    """近岸外／近岸型未截尾壽命比例欄位名 ``<group>_track_frac_uncensored_ge_<N>h``。"""

    return f"{group}_track_frac_uncensored_ge_{threshold_label(hours)}h"


def _null_group_column_names(thresholds_hours: Sequence[float]) -> list[str]:
    names: list[str] = []
    for group in COAST_GROUPS:
        names.append(f"null_{group}_track_n_uncensored")
        names.extend(f"null_{group_fraction_column(group, h)}" for h in thresholds_hours)
        names.extend(f"excess_{group_fraction_column(group, h)}" for h in thresholds_hours)
    return names


def _null_group_columns(
    real_record: Mapping[str, Any],
    null_sets_by_group: Mapping[str, Sequence[Sequence[float]]],
    thresholds_hours: Sequence[float],
) -> dict[str, Any]:
    """stratum ``all`` 的近岸外／近岸型 null 欄位：各 seed 平均的 n 與 frac，以及超額（真實減 null）。"""

    columns: dict[str, Any] = {}
    for group in COAST_GROUPS:
        summaries = [_lifetime_summary(values, thresholds_hours) for values in null_sets_by_group.get(group, [])]
        columns[f"null_{group}_track_n_uncensored"] = _mean_or_none([float(item["n"]) for item in summaries])
        for hours in thresholds_hours:
            name = lifetime_fraction_column(hours)
            real_name = group_fraction_column(group, hours)
            null_frac = _mean_or_none([item["frac"][name] for item in summaries])
            real_frac = real_record.get(real_name)
            columns[f"null_{real_name}"] = null_frac
            columns[f"excess_{real_name}"] = (
                (real_frac - null_frac) if (real_frac is not None and null_frac is not None) else None
            )
    return columns


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    return float(np.mean(finite)) if finite else None


def _null_columns(
    real_record: Mapping[str, Any], null_lifetimes: Sequence[Sequence[float]], thresholds_hours: Sequence[float]
) -> dict[str, Any]:
    """stratum ``all`` 的 null 欄位：跨 seed 平均的 n／median／p90／frac，以及超額與比值。

    ``excess = 真實 frac − null frac``；``ratio = 真實 frac ÷ null frac``（null frac 為 0 或
    任一邊缺值時為 ``None``）。跨 seed 平均只納入該 seed 有未截尾 track 的統計。
    """

    summaries = [_lifetime_summary(values, thresholds_hours) for values in null_lifetimes]
    columns: dict[str, Any] = {
        "null_n_seeds": len(summaries),
        "null_n_uncensored": _mean_or_none([float(item["n"]) for item in summaries]),
        "null_median_lifetime_hours": _mean_or_none([item["median"] for item in summaries]),
        "null_p90_lifetime_hours": _mean_or_none([item["p90"] for item in summaries]),
    }
    for hours in thresholds_hours:
        name = lifetime_fraction_column(hours)
        null_frac = _mean_or_none([item["frac"][name] for item in summaries])
        real_frac = real_record.get(name)
        columns[f"null_{name}"] = null_frac
        columns[f"excess_{name}"] = (real_frac - null_frac) if (real_frac is not None and null_frac is not None) else None
        columns[f"ratio_{name}"] = (
            (real_frac / null_frac) if (real_frac is not None and null_frac is not None and null_frac > 0.0) else None
        )
    return columns


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
    coast_buffer_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """計算一個站×分層的熱點遮罩與指標。

    熱點 = 該站 ``curve_pass_frequency`` 落在自己分析區內最高 ``1 - hotspot_quantile``
    的格點；另以同一分位數取 ``core_frequency`` 熱點遮罩（供圖面輪廓）。指標：門檻、格數、
    面積（km²）、熱點內平均曲線通過頻率、熱點內核心的平均 s1（day⁻¹，依核心數加權）、
    以及該站強核心落在熱點格點的比例。

    統計與 ``curve_mask``／``core_mask`` 保持未濾除定義；另回傳 ``curve_mask_filtered``／
    ``core_mask_filtered``（移除 4-連通格數 < ``min_cells`` 的小塊），只供圖面與 npz 附加陣列。

    給定 ``coast_buffer_mask``（近岸緩衝帶內的格點）時另算：``hotspot_frac_area_within_buffer``（熱點面積落在緩衝帶內
    的比例），以及只在近岸外格點（站內且不在緩衝帶）重新取分位數的近岸外熱點——格數
    ``offshore_hotspot_cell_count`` 與平均曲線通過頻率 ``offshore_hotspot_mean_curve_pass_frequency``，
    用來檢視「去除海岸帶後熱點仍有多強」。未給遮罩時這些欄位為 ``None``。
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
    frac_area_within_buffer: float | None = None
    offshore_cell_count: int | None = None
    offshore_mean_curve: float | None = None
    if coast_buffer_mask is not None:
        buffer_mask = np.asarray(coast_buffer_mask, dtype=bool)
        area_array = np.asarray(cell_area_m2, dtype=np.float64)
        if cell_count and area_km2 > 0.0:
            frac_area_within_buffer = float(np.sum(area_array[curve_mask & buffer_mask])) / 1.0e6 / area_km2
        offshore_mask, _offshore_threshold = _top_quantile_mask(
            finalized["curve_pass_frequency"], np.asarray(site_mask, dtype=bool) & ~buffer_mask, hotspot_quantile
        )
        offshore_cell_count = int(offshore_mask.sum())
        offshore_mean_curve = (
            float(np.mean(np.asarray(finalized["curve_pass_frequency"])[offshore_mask])) if offshore_cell_count else None
        )
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
            "hotspot_frac_area_within_buffer": frac_area_within_buffer,
            "offshore_hotspot_cell_count": offshore_cell_count,
            "offshore_hotspot_mean_curve_pass_frequency": offshore_mean_curve,
        },
    }


PROPAGATION_COLUMNS = (
    "prop_n_tracks",
    "prop_n_moving",
    "prop_n_stationary",
    "prop_stationary_frac",
    "prop_cx_median_kmday",
    "prop_cy_median_kmday",
    "prop_speed_median_kmday",
    "prop_speed_p90_kmday",
    "prop_resid_rms_median_m",
)


def _propagation_columns(accumulator: Mapping[str, Any] | None) -> dict[str, Any]:
    """stratum ``all`` 的核心傳播速度統計（km/day）；其他分層或無樣本時各欄為 ``None``。

    每條 track（至少 ``max(3, --prop-min-obs)`` 個觀測）一筆線性擬合速度；``prop_n_tracks`` 為已擬合條數，
    ``prop_stationary_frac`` 為擬合位移小於 1 格的靜止 track 比例。速度統計（cx／cy 中位數、速率中位數與 p90）
    只含移動 track（速率為零沒有方向，且會被格點量化左右）；``prop_resid_rms_median_m`` 是全部已擬合 track 位置
    殘差 RMS 的中位數。其他分層固定 ``None``；``all`` 但零條 track 時 ``prop_n_tracks = 0``、其餘為 ``None``。
    """

    columns: dict[str, Any] = dict.fromkeys(PROPAGATION_COLUMNS)
    if accumulator is None:
        return columns
    fitted = np.asarray(accumulator["prop_tracks"], dtype=np.float64).reshape(-1, 5)
    columns["prop_n_tracks"] = int(fitted.shape[0])
    if fitted.shape[0] == 0:
        return columns
    moving = fitted[:, 4] == 0.0
    columns["prop_n_moving"] = int(moving.sum())
    columns["prop_n_stationary"] = int((~moving).sum())
    columns["prop_stationary_frac"] = float((~moving).sum() / fitted.shape[0])
    columns["prop_resid_rms_median_m"] = float(np.median(fitted[:, 2]))
    if moving.any():
        cx, cy = fitted[moving, 0], fitted[moving, 1]
        speed = np.hypot(cx, cy)
        columns["prop_cx_median_kmday"] = float(np.median(cx))
        columns["prop_cy_median_kmday"] = float(np.median(cy))
        columns["prop_speed_median_kmday"] = float(np.median(speed))
        columns["prop_speed_p90_kmday"] = float(np.quantile(speed, 0.9))
    return columns


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
    coast_buffer_km: float | None = None,
) -> dict[str, Any]:
    """把一組 site×stratum 累加器轉成 site_statistics 記錄（``scope`` 固定為 site_area）。

    ``coast_buffer_km`` 給定且累加器帶有近岸資訊時，輸出近岸診斷欄位：站內核心／強核心／長壽命 track 落在
    近岸緩衝帶內的比例，去除近岸帶後的強核心出現率，以及近岸外／近岸型的未截尾壽命比例。
    """

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
    if coast_buffer_km is not None and accumulator.get("coast_info"):
        core_total = int(accumulator.get("core_count_total", 0))
        strong_total = int(accumulator.get("strong_core_count_total", 0))
        long_total = int(accumulator.get("long_track_n", 0))
        record.update(
            {
                "coastal_buffer_km": float(coast_buffer_km),
                "frac_cores_within_buffer": (accumulator["core_near_count"] / core_total) if core_total else None,
                "frac_strong_cores_within_buffer": (accumulator["strong_core_near_count"] / strong_total) if strong_total else None,
                "long_track_n_born_uncensored": long_total,
                "frac_long_tracks_within_buffer": (accumulator["long_track_coastal_n"] / long_total) if long_total else None,
                "offshore_strong_core_occurrence_rate": (
                    float(accumulator["offshore_strong_occurrence"] / n_valid) if n_valid else None
                ),
            }
        )
        for group in COAST_GROUPS:
            group_lifetimes = accumulator.get(f"track_lifetimes_{group}", [])
            record[f"{group}_track_n_uncensored"] = len(group_lifetimes)
            for threshold in lifetime_thresholds_hours:
                record[group_fraction_column(group, threshold)] = (
                    float(sum(1 for value in group_lifetimes if value >= threshold) / len(group_lifetimes)) if group_lifetimes else None
                )
    record.update(_propagation_columns(accumulator if stratum == "all" else None))
    record.update(hotspot_metrics)
    return record


_STAT_BASE_HEAD = (
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
    "track_p90_lifetime_hours",
)
_STAT_BASE_TAIL = (
    "hotspot_quantile", "hotspot_threshold_curve_pass_frequency", "hotspot_cell_count", "hotspot_area_km2",
    "hotspot_mean_curve_pass_frequency", "hotspot_mean_core_s1_day", "hotspot_strong_core_fraction",
    "hotspot_frac_area_within_buffer", "offshore_hotspot_cell_count", "offshore_hotspot_mean_curve_pass_frequency",
    *PROPAGATION_COLUMNS,
    "coastal_buffer_km", "frac_cores_within_buffer", "frac_strong_cores_within_buffer",
    "long_track_n_born_uncensored", "frac_long_tracks_within_buffer", "offshore_strong_core_occurrence_rate",
)


def _stat_fieldnames(thresholds_hours: Sequence[float]) -> list[str]:
    """site_statistics 的基本欄位；壽命門檻決定 ``track_frac_*`` 及近岸外／近岸型壽命比例欄位。"""

    names = list(_STAT_BASE_HEAD)
    names.extend(lifetime_fraction_column(hours) for hours in thresholds_hours)
    names.extend(_STAT_BASE_TAIL)
    for group in COAST_GROUPS:
        names.append(f"{group}_track_n_uncensored")
        names.extend(group_fraction_column(group, hours) for hours in thresholds_hours)
    return names


SITE_STAT_FIELDNAMES = tuple(_stat_fieldnames(LIFETIME_THRESHOLD_HOURS))


def site_stat_fieldnames(
    lifetime_thresholds_hours: Sequence[float] = LIFETIME_THRESHOLD_HOURS,
    *,
    with_null: bool = False,
    with_track_filter: bool = False,
) -> list[str]:
    """依壽命門檻設定產生 CSV 欄位；預設門檻等同 :data:`SITE_STAT_FIELDNAMES`。

    ``with_track_filter`` 附加 ``track_core_filter_quantile``（track 只含各站前 q 強核心時）；
    ``with_null`` 附加 :func:`null_column_names` 的 null／excess／ratio 欄位（含近岸外／近岸型的 null 與超額）。
    兩者未啟用時欄位只含基本欄位與近岸診斷欄位。
    """

    names = _stat_fieldnames(lifetime_thresholds_hours)
    if with_track_filter:
        names.append("track_core_filter_quantile")
    if with_null:
        names.extend(null_column_names(lifetime_thresholds_hours))
    return names


def _write_site_statistics(
    output_dir: Path,
    records: list[dict[str, Any]],
    lifetime_thresholds_hours: Sequence[float] = LIFETIME_THRESHOLD_HOURS,
    *,
    with_null: bool = False,
    track_core_filter: Mapping[str, Any] | None = None,
) -> None:
    """寫出 long-format ``site_statistics.csv`` 與 ``site_statistics.json``。"""

    csv_path = output_dir / "site_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = site_stat_fieldnames(
            lifetime_thresholds_hours, with_null=with_null, with_track_filter=track_core_filter is not None
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})
    payload: dict[str, Any] = {"schema_version": AGGREGATE_SCHEMA_VERSION, "records": records}
    if track_core_filter is not None or with_null:
        payload["track_core_filter"] = track_core_filter
        payload["lifetime_scope_zh"] = lifetime_scope_text(track_core_filter)
    _atomic_write_json(output_dir / "site_statistics.json", payload)


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
        "track_count": np.asarray(accumulator["track_count"], dtype=np.int64),
        "long_track_count": np.asarray(accumulator["long_track_count"], dtype=np.int64),
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
    """一個站的成果圖相對路徑（``figures/<site_id>/site_<site_id>_<kind>.png``）：五張基本圖、WP-H 的五張，以及 WP-I 的
    ``example_coastal``（該站沒有近岸型長壽命 track 時不輸出）與 ``coastal_diagnostic``。"""

    return [
        f"figures/{site_id}/site_{site_id}_{kind}.png"
        for kind in (*FIGURE_KINDS, *EXAMPLE_FIGURE_KINDS, *TRACK_FIGURE_KINDS, *COASTAL_FIGURE_KINDS)
    ]


def _resolve_figure_font() -> Any:
    """解析圖面 CJK 字型；找不到合格字型就 fail closed（圖面不可輸出方框字）。"""

    from .report_font import resolve_cjk_font

    return resolve_cjk_font()


def _null_p99_hours(null_runs: Sequence[Mapping[str, Any]], site_id: str) -> float | None:
    """該站各 seed 對照組未截尾壽命合併後的第 99 百分位（小時）；沒有對照或沒有 null track 時為 ``None``。"""

    pooled = [value for run in null_runs for value in run["lifetimes"].get(site_id, [])]
    return float(np.quantile(np.asarray(pooled, dtype=np.float64), 0.99)) if pooled else None


def _render_site_wp_h_figures(
    *,
    output_dir: Path,
    site_id: str,
    figure_data: Any,
    context: Mapping[str, Any],
    domain_id: str,
    grid: Mapping[str, Any],
    born_records: Sequence[Mapping[str, Any]],
    prop_tracks: Sequence[Sequence[float]],
    null_runs: Sequence[Mapping[str, Any]],
    long_track_hours: float,
    lifetime_thresholds: Sequence[float],
    nominal_step_hours: float,
    prop_min_obs: int = PROP_MIN_OBS,
    example_enabled: bool,
    surface_root: Path | None,
    detect_root: Path,
    font_selection: Any,
    coastline: Any,
    state_label: str,
    param_id: str,
    period_label: str,
    coast_buffer_m: float | None = None,
) -> dict[str, Any]:
    """一個站的 WP-H／WP-I 圖：長壽命 track 路徑、track 通過數、傳播速度，以及（可選）代表 TRAP、近岸型代表與方法比較。

    代表 track 只取「首個觀測在站內、未截尾」的真實 track。給定 ``coast_buffer_m`` 時（預設流程）優先取平均核心
    距岸不小於緩衝帶、且壽命超過該站隨機對照 p99 的最長者（沒有對照時略過 p99 條件）；沒有這樣的 track 就退回整體
    最長者並在圖面明示（「（近岸型）」等）。另輸出 ``site_<id>_example_coastal.png``：最長的近岸型 track（沒有就不輸出）。
    沒有任何候選時仍輸出空白說明圖，不製造資料。回傳 ``{"paths": 相對路徑清單, "manifest": 該站紀錄}``。
    """

    from . import example_figures

    observations = context["observations"]
    long_records = sorted(
        (item for item in born_records if float(item["lifetime_hours"]) >= long_track_hours),
        key=lambda item: (-float(item["lifetime_hours"]), int(item["track_id"])),
    )
    drawn = long_records[:LONG_TRACK_DRAW_LIMIT]
    long_tracks = [
        (
            float(item["lifetime_hours"]),
            observations["core_x_m"][item["start"] : item["stop"]],
            observations["core_y_m"][item["start"] : item["stop"]],
        )
        for item in drawn
    ]
    null_counts = [
        float(sum(1 for value in run["born_lifetimes"].get(site_id, []) if value >= long_track_hours)) for run in null_runs
    ]
    null_long_mean = float(np.mean(null_counts)) if null_counts else None
    has_coast = coast_buffer_m is not None
    n_long_coastal = sum(1 for item in long_records if item.get("coastal")) if has_coast else None
    n_long_offshore = (len(long_records) - n_long_coastal) if has_coast else None
    null_coastal_counts: list[float] = []
    null_offshore_counts: list[float] = []
    for run in null_runs:
        record = run["records"].get(site_id, {"born_lifetimes": [], "born_coastal": []})
        flags = [(value >= long_track_hours, coastal) for value, coastal in zip(record["born_lifetimes"], record["born_coastal"], strict=True)]
        null_coastal_counts.append(float(sum(1 for long, coastal in flags if long and coastal)))
        null_offshore_counts.append(float(sum(1 for long, coastal in flags if long and not coastal)))
    fitted = np.asarray(prop_tracks, dtype=np.float64).reshape(-1, 5)
    track_inputs = example_figures.TrackFigureInputs(
        x_m_axis=grid["x_m_axis"],
        y_m_axis=grid["y_m_axis"],
        long_track_hours=long_track_hours,
        long_tracks=long_tracks,
        n_long_total=len(long_records),
        prop_cx_kmday=fitted[:, 0],
        prop_cy_kmday=fitted[:, 1],
        prop_track_n=int(fitted.shape[0]),
        nominal_step_hours=nominal_step_hours,
        null_long_mean=null_long_mean,
        null_n_seeds=len(null_counts),
        long_coastal=[bool(item.get("coastal")) for item in drawn] if has_coast else (),
        n_long_coastal_total=n_long_coastal,
        n_long_offshore_total=n_long_offshore,
        null_long_coastal_mean=float(np.mean(null_coastal_counts)) if (has_coast and null_coastal_counts) else None,
        null_long_offshore_mean=float(np.mean(null_offshore_counts)) if (has_coast and null_offshore_counts) else None,
        prop_stationary=fitted[:, 4] != 0.0,
        prop_resid_rms_m=fitted[:, 2],
        prop_min_obs=int(prop_min_obs),
    )
    paths = list(
        example_figures.render_track_figures(
            output_dir,
            figure_data,
            track_inputs,
            font_selection=font_selection,
            coastline=coastline,
            state_label=state_label,
            param_id=param_id,
            period_label=period_label,
        )
    )
    n_stationary = int(np.count_nonzero(track_inputs.prop_stationary))
    record: dict[str, Any] = {
        "long_track_hours": long_track_hours,
        "n_long_tracks": len(long_records),
        "n_long_tracks_drawn": len(drawn),
        "n_long_tracks_coastal": n_long_coastal,
        "n_long_tracks_offshore": n_long_offshore,
        "null_long_track_mean": null_long_mean,
        "null_long_track_coastal_mean": track_inputs.null_long_coastal_mean,
        "null_long_track_offshore_mean": track_inputs.null_long_offshore_mean,
        "null_n_seeds": len(null_counts),
        "propagation_n_tracks": int(fitted.shape[0]),
        "propagation_n_moving": int(fitted.shape[0]) - n_stationary,
        "propagation_n_stationary": n_stationary,
        "example": {"enabled": bool(example_enabled)},
    }
    if example_enabled:
        null_p99 = _null_p99_hours(null_runs, site_id) if has_coast else None
        chosen, criterion = select_representative_track(
            born_records, max(lifetime_thresholds), coastal_buffer_m=coast_buffer_m, null_p99_hours=null_p99
        )

        def _prepare(item: Mapping[str, Any], item_criterion: str, variant: str) -> Any:
            return example_figures.prepare_example_snapshot(
                detect_root=detect_root,
                domain_id=domain_id,
                shard_manifests=context["shard_manifests"],
                state_label=state_label,
                surface_root=surface_root,
                observations=observations,
                start=int(item["start"]),
                stop=int(item["stop"]),
                track_id=int(item["track_id"]),
                lifetime_hours=float(item["lifetime_hours"]),
                criterion=item_criterion,
                coast_buffer_m=coast_buffer_m,
                variant=variant,
            )

        snapshot = _prepare(chosen, criterion, "main") if chosen is not None else None
        paths.extend(
            example_figures.render_snapshot_figures(
                output_dir,
                figure_data,
                snapshot,
                font_selection=font_selection,
                coastline=coastline,
                state_label=state_label,
                param_id=param_id,
                period_label=period_label,
                nominal_step_hours=nominal_step_hours,
            )
        )
        record["example"].update(
            {
                "status": "ok" if snapshot is not None else "no_eligible_track",
                "n_candidates": len(born_records),
                "selection_min_lifetime_hours": float(max(lifetime_thresholds)),
                "criterion": criterion,
                "null_p99_lifetime_hours": null_p99,
            }
        )
        if snapshot is not None:
            record["example"].update(
                {
                    "track_id": snapshot.track_id,
                    "lifetime_hours": snapshot.lifetime_hours,
                    "n_observations": snapshot.n_observations,
                    "birth_time_utc_ns": snapshot.time_utc_ns,
                    "birth_time_utc": example_figures.time_label_utc(snapshot.time_utc_ns),
                    "month": snapshot.month,
                    "snapshot_index": snapshot.snapshot_index,
                    "detection_index": snapshot.detection_index,
                    "recomputed_core_offset_cells": snapshot.core_offset_cells,
                    "source_month_dir": snapshot.source_month_dir,
                    "mean_coast_distance_m": snapshot.mean_coast_m,
                    "coastal": snapshot.is_coastal,
                }
            )
        coastal_record: dict[str, Any] = {"status": "none"}
        if has_coast:
            coastal_choice = select_coastal_track(born_records, float(coast_buffer_m))
            if coastal_choice is not None:
                coastal_snapshot = _prepare(coastal_choice, "coastal_longest", "coastal")
                paths.extend(
                    example_figures.render_coastal_example(
                        output_dir,
                        figure_data,
                        coastal_snapshot,
                        font_selection=font_selection,
                        coastline=coastline,
                        state_label=state_label,
                        param_id=param_id,
                        period_label=period_label,
                        nominal_step_hours=nominal_step_hours,
                    )
                )
                coastal_record = {
                    "status": "ok",
                    "track_id": coastal_snapshot.track_id,
                    "lifetime_hours": coastal_snapshot.lifetime_hours,
                    "n_observations": coastal_snapshot.n_observations,
                    "mean_coast_distance_m": coastal_snapshot.mean_coast_m,
                    "recomputed_core_offset_cells": coastal_snapshot.core_offset_cells,
                }
        record["example_coastal"] = coastal_record
    return {"paths": paths, "manifest": record}


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
    long_raw = getattr(args, "long_track_hours", None)
    long_track_hours = float(max(lifetime_thresholds) if long_raw is None else long_raw)
    _require(long_track_hours > 0.0, "--long-track-hours 必須為正值")
    surface_root_raw = getattr(args, "surface_root", None)
    surface_root = Path(surface_root_raw) if surface_root_raw else None
    coastal_buffer_km = float(getattr(args, "coastal_buffer_km", None) or COASTAL_BUFFER_KM)
    _require(coastal_buffer_km > 0.0 and np.isfinite(coastal_buffer_km), "--coastal-buffer-km 必須為正值")
    coast_buffer_m = coastal_buffer_km * 1000.0
    _require(coast_buffer_m < COAST_DISTANCE_CAP_M, f"--coastal-buffer-km 必須小於近岸距離場飽和上限 {COAST_DISTANCE_CAP_M / 1000.0:g} km")
    prop_min_obs_raw = getattr(args, "prop_min_obs", None)

    track_root = Path(args.track_root)
    detect_root = Path(args.detect_root)
    output_dir = Path(args.output_dir)
    null_roots = [Path(item) for item in (getattr(args, "null_track_root", None) or [])]
    _require(len({str(path.resolve()) for path in null_roots}) == len(null_roots), "--null-track-root 不可重複指定同一個目錄")
    _require(
        all(path.resolve() != track_root.resolve() for path in null_roots), "--null-track-root 不可與主要 --track-root 相同"
    )

    present_domains = [domain_id for domain_id in all_domain_ids if (track_root / domain_id).is_dir()]
    missing_domains = [domain_id for domain_id in all_domain_ids if domain_id not in present_domains]
    if missing_domains and not args.allow_partial_domains:
        raise ValueError(f"缺少 track-root domain，需加 --allow-partial-domains 才可略過: {missing_domains}")
    _require(bool(present_domains), "track-root 沒有任何可用的 domain")

    figures_enabled = not bool(args.no_figures)
    example_figures_enabled = figures_enabled and not bool(getattr(args, "no_example_figures", False))
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
    domain_step_hours: dict[str, float] = {}
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

        step_config = manifest.get("tracking_config") or {}
        step_hours = (
            float(step_config["nominal_time_step_seconds"]) / 3600.0
            if step_config.get("nominal_time_step_seconds") is not None
            else float(manifest.get("nominal_time_step_hours") or 1.0)
        )
        domain_step_hours[domain_id] = step_hours
        canonical_valid_by_month = _canonical_valid_snapshot_indices_by_month(loaded["canonical"])
        domain_context[domain_id] = {
            "manifest": manifest,
            "canonical": loaded["canonical"],
            "observations": loaded["observations"],
            "tracks_by_id": loaded["tracks_by_id"],
            "available_months": available_months,
            "shard_manifests": shard_manifests,
            "canonical_valid_by_month": canonical_valid_by_month,
            "param_id": manifest.get("param_id") or first_shard["param_id"],
            "detection_config": first_shard["detection_config"],
            "min_coverage": float(first_shard["min_coverage"]),
            "tracking_config": manifest.get("tracking_config"),
            "state_label": manifest.get("state_label"),
            "track_core_filter": manifest.get("track_core_filter"),
        }

    shared = _validate_cross_domain_consistency(domain_context)

    # --- 網格與站點分析區 ---------------------------------------------------------
    grids: dict[str, dict[str, Any]] = {}
    domain_masks: dict[str, dict[str, np.ndarray]] = {}
    _require(
        len({round(v, 9) for v in domain_step_hours.values()}) == 1,
        f"各 domain 的名目時間步長不一致: {domain_step_hours}",
    )
    prop_min_obs = resolve_prop_min_obs(prop_min_obs_raw, next(iter(domain_step_hours.values())))

    for domain_id in present_domains:
        first_month = sorted(domain_context[domain_id]["available_months"])[0]
        grid = _load_domain_grid(detect_root, domain_id, first_month)
        grids[domain_id] = grid
        domain_masks[domain_id] = _build_domain_masks(domain_id, grid, design, study_design_path)
    site_mask = {site_id: mask for masks in domain_masks.values() for site_id, mask in masks.items()}
    # 有效距離場上限：舊 shard 的距離場飽和於 30 km（最大值恰為該值），新版為 COAST_DISTANCE_CAP_M。
    coast_cap_km = COAST_DISTANCE_CAP_M / 1000.0
    for domain_id in present_domains:
        if np.isclose(float(np.nanmax(np.asarray(grids[domain_id]["coast_distance_m"]))), LEGACY_COAST_CAP_M):
            coast_cap_km = min(coast_cap_km, LEGACY_COAST_CAP_M / 1000.0)
    for domain_id in present_domains:
        ensure_observation_coast(domain_context[domain_id], grids[domain_id])

    # --- 時間置換 null 對照組：驗證設定一致，並以與真實相同的規則取各站未截尾壽命 ---------------
    null_runs: list[dict[str, Any]] = []
    for null_root in null_roots:
        seed_value: int | None = None
        per_site_lifetimes: dict[str, list[float]] = {}
        per_site_born: dict[str, list[float]] = {}
        per_site_records: dict[str, dict[str, list[Any]]] = {}
        domain_records: dict[str, dict[str, str]] = {}
        for domain_id in present_domains:
            _require((null_root / domain_id).is_dir(), f"--null-track-root 缺少 domain {domain_id}: {null_root}")
            null_loaded = _load_track_domain(null_root, domain_id, expect_null=True)
            _validate_null_against_real(
                {"manifest": domain_context[domain_id]["manifest"]}, null_loaded, domain_id=domain_id, null_root=null_root
            )
            domain_seed = int(null_loaded["manifest"]["null_model"]["seed"])
            _require(seed_value is None or seed_value == domain_seed, f"{null_root} 各 domain 的 shuffle seed 不一致")
            seed_value = domain_seed
            ensure_observation_coast(null_loaded, grids[domain_id])
            null_records = _null_track_records_by_site(null_loaded, domain_masks[domain_id], grids[domain_id], coast_buffer_m)
            per_site_records.update(null_records)
            per_site_lifetimes.update({site: rec["lifetimes"] for site, rec in null_records.items()})
            per_site_born.update({site: rec["born_lifetimes"] for site, rec in null_records.items()})
            manifest_path = null_root / domain_id / "manifest.json"
            domain_records[domain_id] = {"manifest_path": str(manifest_path), "manifest_sha256": _sha256_file(manifest_path)}
        assert seed_value is not None
        null_runs.append(
            {
                "root": null_root,
                "seed": seed_value,
                "lifetimes": per_site_lifetimes,
                "born_lifetimes": per_site_born,
                "records": per_site_records,
                "domains": domain_records,
            }
        )
    null_seeds = [run["seed"] for run in null_runs]
    _require(len(set(null_seeds)) == len(null_seeds), f"--null-track-root 的 shuffle seed 重複: {null_seeds}")
    track_core_filter = shared["track_core_filter"]
    scope_text = lifetime_scope_text(track_core_filter)

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
            coast_buffer_m=coast_buffer_m,
        )
        _process_track_lifetimes(
            context=domain_context[domain_id],
            site_masks_2d=domain_masks[domain_id],
            grid=grids[domain_id],
            site_accumulators=site_accumulators,
            spatial_accumulators=spatial_accumulators,
            evolution_min_hours=evolution_min_hours,
            long_track_hours=long_track_hours,
            prop_min_obs=prop_min_obs,
            coast_buffer_m=coast_buffer_m,
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
        buffer_cells = np.asarray(grids[domain_id]["coast_distance_m"]) < coast_buffer_m
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
            hotspot = _site_hotspots(
                accumulator, finalized, mask, cell_area_m2, hotspot_quantile, hotspot_min_cells, coast_buffer_mask=buffer_cells
            )
            site_spatial[site_id][stratum] = _mask_to_site(finalized, mask)
            site_hotspots[site_id][stratum] = hotspot
            hotspot_metrics_by_stratum[stratum] = hotspot["metrics"]
        colour_limits[site_id] = _compute_site_colour_limits(site_spatial[site_id]["all"])

        for stratum in ordered:
            record = (
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
                    coast_buffer_km=coastal_buffer_km,
                )
            )
            if track_core_filter is not None:
                record["track_core_filter_quantile"] = track_core_filter["quantile"]
            if null_runs and stratum == "all":
                record.update(
                    _null_columns(record, [run["lifetimes"].get(site_id, []) for run in null_runs], lifetime_thresholds)
                )
                null_sets_by_group = {
                    group: [
                        [
                            value
                            for value, coastal in zip(
                                run["records"][site_id]["lifetimes"], run["records"][site_id]["coastal"], strict=True
                            )
                            if coastal == (group == "coastal")
                        ]
                        if site_id in run["records"]
                        else []
                        for run in null_runs
                    ]
                    for group in COAST_GROUPS
                }
                record.update(_null_group_columns(record, null_sets_by_group, lifetime_thresholds))
            records.append(record)
    _write_site_statistics(output_dir, records, lifetime_thresholds, with_null=bool(null_runs), track_core_filter=track_core_filter)

    # --- 逐站 spatial npz ------------------------------------------------------------
    for site_id in sites_present:
        grid = grids[site_domain[site_id]]
        coast_field = np.asarray(grid["coast_distance_m"])
        site_coast_distance = np.where(site_mask[site_id], coast_field, np.nan).astype(np.float32)
        site_buffer_mask = site_mask[site_id] & (coast_field < coast_buffer_m)
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
                coast_distance_m=site_coast_distance,
                coast_buffer_mask=site_buffer_mask,
                coast_buffer_m=np.float64(coast_buffer_m),
                **site_spatial[site_id][stratum],
            )

    # --- 圖面 -------------------------------------------------------------------
    figure_paths: list[str] = []
    wp_h_manifest: dict[str, Any] = {}
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
                null_lifetimes=[list(run["lifetimes"].get(site_id, [])) for run in null_runs],
                lifetime_scope_text=scope_text,
                coast_distance_m=grid["coast_distance_m"],
                coast_buffer_km=coastal_buffer_km,
                coastal=aggregate_figures.CoastalDiagnosticData(
                    buffer_km=coastal_buffer_km,
                    hist_edges_km=COAST_HIST_EDGES_KM,
                    cap_km=coast_cap_km,
                    detection_floor_km=(
                        float(all_accumulator["core_coast_min_m"]) / 1000.0 if np.isfinite(all_accumulator["core_coast_min_m"]) else None
                    ),
                    hist_all=all_accumulator["coast_hist_all"],
                    hist_strong=all_accumulator["coast_hist_strong"],
                    hist_long=all_accumulator["coast_hist_long"],
                    lifetimes_offshore=list(all_accumulator["track_lifetimes_offshore"]),
                    lifetimes_coastal=list(all_accumulator["track_lifetimes_coastal"]),
                    null_offshore=[
                        [v for v, c in zip(run["records"][site_id]["lifetimes"], run["records"][site_id]["coastal"], strict=True) if not c]
                        for run in null_runs
                        if site_id in run["records"]
                    ],
                    null_coastal=[
                        [v for v, c in zip(run["records"][site_id]["lifetimes"], run["records"][site_id]["coastal"], strict=True) if c]
                        for run in null_runs
                        if site_id in run["records"]
                    ],
                    nominal_step_hours=domain_step_hours[site_domain[site_id]],
                ),
                lifetime_thresholds_hours=lifetime_thresholds,
                nominal_step_hours=domain_step_hours[site_domain[site_id]],
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
            site_extra = _render_site_wp_h_figures(
                output_dir=output_dir,
                site_id=site_id,
                figure_data=figure_data,
                context=domain_context[site_domain[site_id]],
                domain_id=site_domain[site_id],
                grid=grid,
                born_records=all_accumulator["born_uncensored"],
                prop_tracks=all_accumulator["prop_tracks"],
                null_runs=null_runs,
                long_track_hours=long_track_hours,
                lifetime_thresholds=lifetime_thresholds,
                nominal_step_hours=domain_step_hours[site_domain[site_id]],
                prop_min_obs=prop_min_obs,
                example_enabled=example_figures_enabled,
                surface_root=surface_root,
                detect_root=detect_root,
                font_selection=font_selection,
                coastline=coastline,
                state_label=shared["state_label"],
                param_id=shared["param_id"],
                period_label=period_label,
                coast_buffer_m=coast_buffer_m,
            )
            figure_paths.extend(site_extra["paths"])
            wp_h_manifest[site_id] = site_extra["manifest"]

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
        "long_track_hours": long_track_hours,
        "coastal_buffer_km": coastal_buffer_km,
        "coast_distance": {
            "definition": "distance from point to nearest cell centre that is not analysis ocean (in-grid only)",
            "cap_m": coast_cap_km * 1000.0,
            "coastal_track_rule": "track mean core coast distance < buffer",
            "computed_from_mask_domains": [d for d in present_domains if grids[d].get("coast_distance_computed")],
            "offshore_strong_threshold": "unchanged (site-wide strong-core quantile threshold)",
        },
        "propagation": {
            "method": "per-track least-squares linear fit of refined core x(t), y(t)",
            "min_observations": prop_min_obs,
            "min_observations_source": "explicit" if prop_min_obs_raw is not None else "auto (max(3, ceil(6 h / step)) + 1 for step < 24 h; 3 for daily)",
            "stationary_rule": "fitted displacement over track duration < 1 grid cell",
            "stationary_cell_m": {d: grid_cell_size_m(grids[d]) for d in present_domains},
        },
        "core_subgrid": {
            "method": shared["detection_config"].get("core_subgrid", "none"),
            "refine": bool(shared["detection_config"].get("core_subgrid_refine", False)),
        },
        "shard_schema_versions": {
            d: sorted({m.get("schema_version") for m in domain_context[d]["shard_manifests"].values()}) for d in present_domains
        },
        "example_figures_enabled": example_figures_enabled,
        "surface_root_override": None if surface_root is None else str(surface_root),
        "site_track_figures": wp_h_manifest,
        "track_core_filter": track_core_filter,
        "lifetime_scope_zh": scope_text,
        "null_inputs": [
            {
                "track_root": str(run["root"]),
                "null_model": {"type": "time_shuffle", "seed": run["seed"]},
                "seed": run["seed"],
                "domains": run["domains"],
            }
            for run in null_runs
        ],
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
    aggregate.add_argument(
        "--null-track-root", type=Path, nargs="+", default=None, metavar="DIR",
        help="時間置換對照組的 track 根目錄（可多個，每個含 <domain>/ 子目錄，由 track-period --shuffle-seed 產生）；"
        "必須與主要 track-root 同 detection／tracking 設定與 track_core_filter（僅 seed 不同），否則拒絕。"
        "site_statistics 加上 null_／excess_／ratio_ 欄位，壽命圖疊上 null CCDF",
    )
    aggregate.add_argument(
        "--long-track-hours", type=float, default=None, metavar="H",
        help="site_<id>_long_tracks.png 與 track_density 長壽命面板的壽命門檻（小時；名目步長 24 h 時圖面以天顯示），"
        "預設為 --lifetime-thresholds-hours 的最大值",
    )
    aggregate.add_argument(
        "--no-example-figures", action="store_true",
        help="略過需重開 OCM snapshot 的兩張圖（site_<id>_example.png、site_<id>_method_comparison.png）；其他圖照常輸出",
    )
    aggregate.add_argument(
        "--surface-root", type=Path, default=None,
        help="OCM ocm_surface 根目錄（含 <domain_id>/months、<domain_id>/grid）；未指定時重開 OCM 使用 shard manifest 記錄的"
        "來源路徑，路徑搬移後以此覆寫（仍核對 metadata 與海岸遮罩 SHA-256）",
    )
    aggregate.add_argument(
        "--coastal-buffer-km", type=float, default=COASTAL_BUFFER_KM, metavar="KM",
        help="近岸緩衝帶寬度（km，預設 3）：核心距最近非分析海域格點小於此值視為近岸；驅動近岸比例、近岸外（offshore）"
        "統計、近岸型／近岸外 track 分類與各地圖上的虛線緩衝帶",
    )
    aggregate.add_argument(
        "--prop-min-obs", type=int, default=None, metavar="N",
        help="傳播速度線性擬合所需的最少觀測數（下限 3）；未指定時依步長自動：步長 < 24 h 為 max(3, ceil(6 h / 步長)) + 1"
        "（逐時 7），日資料 3",
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
