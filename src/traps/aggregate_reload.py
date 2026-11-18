"""由已發布的 ``aggregate_report`` 產物重建只需空間欄位的 :class:`SiteFigureData`。

``aggregate`` 重繪壽命 CCDF、逐日時間序列、近岸診斷與 track 診斷圖需要原始 track／detect
資料（只存在 SERVER）；但 ``spatial``、``hotspots``、``seasonal`` 與新增的
``cumulative_days``、``hotspot_overlap`` 只用到 ``spatial_<stratum>.npz``、
``site_statistics.json`` 與 ``manifest.json``，因此可以完全由正式產物重繪，不需重新
detect／track。重建出的壽命、時間序列與近岸診斷欄位刻意留空，呼叫端不得用它們繪製
這些圖。

近岸緩衝帶等值線在 aggregate 內使用整個 flow domain 的距離場；npz 只存站內遮罩後的
距離場。同一 domain 的各站遮罩互斥且合起來等於分析海域（A 區貢寮＋龜山島、B／C／D
單一站），非分析海域的距離定義為 0，故可由同 domain 各站的 npz 還原整個距離場
（:func:`reconstruct_domain_coast_distance`）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .aggregate_figures import SEASONS, SiteFigureData
from .sites import load_study_design

SPATIAL_FIELD_KEYS = (
    "n_valid", "core_frequency", "strong_core_frequency", "curve_pass_frequency", "mean_core_s1_sinv",
    "mean_core_s1_day", "mean_lifetime_hours", "track_count", "long_track_count",
)
"""``site_spatial[site]["all"]`` 中的逐格欄位（與 npz 內同名）。"""

HOTSPOT_METRIC_KEYS = (
    "hotspot_quantile", "hotspot_threshold_curve_pass_frequency", "hotspot_cell_count", "hotspot_area_km2",
    "hotspot_mean_curve_pass_frequency", "hotspot_mean_core_s1_day", "hotspot_strong_core_fraction",
    "hotspot_frac_area_within_buffer", "offshore_hotspot_cell_count", "offshore_hotspot_mean_curve_pass_frequency",
)
"""``site_statistics`` 紀錄中屬於熱點統計的欄位。"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """讀取 npz 全部陣列（不使用 pickle）。"""

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def reconstruct_domain_coast_distance(site_npz_by_site: Mapping[str, Mapping[str, np.ndarray]]) -> np.ndarray:
    """由同一 flow domain 各站的 npz 還原整個 domain 的近岸距離場（float32）。

    各站 ``coast_distance_m`` 只在自己的站內遮罩內有值；站外為 NaN。同 domain 各站遮罩
    必須互斥（否則丟出 ``ValueError``），遮罩聯集外（非分析海域）的距離定義為 0。
    """

    sites = list(site_npz_by_site.values())
    if not sites:
        raise ValueError("至少需要一個站的 npz")
    shape = np.asarray(sites[0]["site_mask"]).shape
    field = np.zeros(shape, dtype=np.float32)
    covered = np.zeros(shape, dtype=bool)
    for archive in sites:
        mask = np.asarray(archive["site_mask"], dtype=bool)
        if mask.shape != shape:
            raise ValueError("同 domain 各站的格網形狀必須相同")
        if np.any(covered & mask):
            raise ValueError("同 domain 各站的站內遮罩必須互斥")
        covered |= mask
        field = np.where(mask, np.asarray(archive["coast_distance_m"], dtype=np.float32), field)
    return field.astype(np.float32)


def site_ids_in_domain(design: Mapping[str, Any], site_id: str) -> list[str]:
    """與 ``site_id`` 同 flow domain 的所有研究站 ID（依設計檔順序）。"""

    domain = next(str(site["flow_domain_id"]) for site in design["study_sites"] if str(site["study_site_id"]) == site_id)
    return [str(site["study_site_id"]) for site in design["study_sites"] if str(site["flow_domain_id"]) == domain]


def load_grid_coast_distance(runs_root: Path, param_id: str, manifest: Mapping[str, Any], domain_id: str) -> np.ndarray:
    """讀正式 detect 輸出的整個 domain 近岸距離場 ``coast_distance_m``（與 aggregate 繪圖使用的同一欄位）。

    由站 npz 還原的距離場在各站遮罩外一律為 0，擴張後的標準地圖視窗會在站外出現不存在的
    緩衝帶等值線；有 detect 輸出（SERVER）時改用真實距離場。網格對一個 domain 的各月份相同，
    取第一個可用月份的 ``grid_reference.npz``。
    """

    months = sorted(manifest["inputs"][domain_id]["available_months"])
    path = Path(runs_root) / param_id / "detect" / domain_id / str(months[0]) / "grid_reference.npz"
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["coast_distance_m"])


def load_site_figure_data(
    aggregate_dir: Path,
    site_id: str,
    *,
    study_design_path: Path | None = None,
    coast_distance_m: np.ndarray | None = None,
) -> SiteFigureData:
    """由 ``aggregate_dir`` 重建一個站的空間／熱點／季節圖資料。

    數值來源：``spatial/<site>/spatial_{all,DJF,MAM,JJA,SON}.npz``（逐格欄位、熱點遮罩）、
    ``site_statistics.json``（季節與整體統計紀錄）、``manifest.json``（色階、熱點最小連通
    格數、近岸緩衝帶、名目步長）與研究設計（站名、anchor）。壽命、時間序列、近岸診斷
    等需要原始 track 的欄位留空。
    """

    root = Path(aggregate_dir)
    manifest = _read_json(root / "manifest.json")
    statistics = _read_json(root / "site_statistics.json")
    design = load_study_design(study_design_path) if study_design_path else load_study_design()
    site = next(item for item in design["study_sites"] if str(item["study_site_id"]) == site_id)
    domain_sites = site_ids_in_domain(design, site_id)

    all_npz = load_npz(root / "spatial" / site_id / "spatial_all.npz")
    season_npz = {season: load_npz(root / "spatial" / site_id / f"spatial_{season}.npz") for season in SEASONS}
    domain_npz = {sid: (all_npz if sid == site_id else load_npz(root / "spatial" / sid / "spatial_all.npz")) for sid in domain_sites}
    coast_field = reconstruct_domain_coast_distance(domain_npz) if coast_distance_m is None else np.asarray(coast_distance_m)

    records = {
        str(item["stratum"]): item
        for item in statistics["records"]
        if str(item["site_id"]) == site_id and str(item["stratum"]) in ("all", *SEASONS)
    }
    record_all = records["all"]
    hotspot_metrics = {key: record_all[key] for key in HOTSPOT_METRIC_KEYS if key in record_all}
    step_seconds = float(manifest["tracking_config"]["nominal_time_step_seconds"])

    def _fields(archive: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.asarray(archive[key]) for key in SPATIAL_FIELD_KEYS if key in archive}

    others = [sid for sid in domain_sites if sid != site_id]
    partner = None
    if len(others) == 1:
        other = others[0]
        other_site = next(item for item in design["study_sites"] if str(item["study_site_id"]) == other)
        partner = {
            "site_id": other,
            "site_name_zh": str(other_site["study_site_name_zh"]),
            "site_mask": np.asarray(domain_npz[other]["site_mask"], dtype=bool),
            "curve_pass_frequency": np.asarray(domain_npz[other]["curve_pass_frequency"], dtype=np.float64),
            "anchor_lonlat": (float(other_site["anchor_lonlat"][0]), float(other_site["anchor_lonlat"][1])),
        }

    return SiteFigureData(
        site_id=site_id,
        site_name_zh=str(site["study_site_name_zh"]),
        lon=np.asarray(all_npz["lon"]),
        lat=np.asarray(all_npz["lat"]),
        site_mask=np.asarray(all_npz["site_mask"], dtype=bool),
        cell_area_m2=np.full(np.asarray(all_npz["site_mask"]).shape, np.nan),
        spatial_all=_fields(all_npz),
        spatial_seasons={season: _fields(season_npz[season]) for season in SEASONS},
        hotspot_curve_mask=np.asarray(all_npz["hotspot_curve_mask"], dtype=bool),
        hotspot_core_mask=np.asarray(all_npz["hotspot_core_mask"], dtype=bool),
        hotspot_metrics=hotspot_metrics,
        colour_limits=manifest["colour_limits"][site_id],
        records=records,
        lifetimes_uncensored=[],
        lifetimes_all=[],
        evolution_mean_day=[],
        evolution_track_count=0,
        day_stats={},
        anchor_lonlat=(float(site["anchor_lonlat"][0]), float(site["anchor_lonlat"][1])),
        n_valid_snapshots=int(record_all["n_valid_snapshots"]),
        hotspot_curve_mask_filtered=np.asarray(all_npz["hotspot_curve_mask_filtered"], dtype=bool),
        hotspot_core_mask_filtered=np.asarray(all_npz["hotspot_core_mask_filtered"], dtype=bool),
        hotspot_min_cells=int(manifest["hotspot_min_cells"]),
        nominal_step_hours=step_seconds / 3600.0,
        coast_distance_m=coast_field,
        coast_buffer_km=float(manifest["coastal_buffer_km"]),
        coastal=None,
        partner=partner,
        valid_sea_mask=np.logical_or.reduce([np.asarray(archive["site_mask"], dtype=bool) for archive in domain_npz.values()]),
    )


__all__ = [
    "load_npz",
    "load_grid_coast_distance",
    "load_site_figure_data",
    "reconstruct_domain_coast_distance",
    "site_ids_in_domain",
]
