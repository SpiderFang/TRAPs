"""``aggregate`` 的成果圖：五站摘要、季節出現率、壽命 CCDF 與逐區空間四分圖。

所有圖面都只描述模型內部 TRAP 結構統計（`state_label`／科學狀態一律標示於圖說），
不代表實測海廢濃度、長期滯留或污染來源。空間圖沿用既有 ``coastline.py``、
``colorbar_layout.py`` 的視覺疊圖與版面對齊規則；receptor 圈以站點 anchor 與
``receptor_core_radius_m`` 換算的近似緯度度數繪製，僅供視覺定位，不是測地精確圓。
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .coastline import CoastlineGeometry, draw_geographic_coastline
from .colorbar_layout import align_colorbars_to_axes
from .figures import _site_font_properties

SEASONS = ("DJF", "MAM", "JJA", "SON")
SITE_ORDER = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")
METERS_PER_DEGREE_LAT = 111_320.0


def _load_matplotlib() -> tuple[Any, Any]:
    """延遲載入無顯示器 matplotlib，與既有成果圖模組共用同一個快取路徑。"""

    try:
        if not os.environ.get("MPLCONFIGDIR"):
            cache_dir = Path(tempfile.gettempdir()) / "traps-matplotlib"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - 由環境依賴 gate 覆蓋
        raise RuntimeError("aggregate 成果圖需要 matplotlib") from error
    return matplotlib, plt


def _footer(period_label: str, state_label: str, param_id: str) -> str:
    """回傳所有 aggregate 圖共用的來源與科學狀態註腳文字。"""

    science_note = "MODEL_ONLY 表層短期吸引潛勢" if state_label != "PILOT_ONLY" else "PILOT_ONLY"
    return f"{period_label}  UTC  param_id={param_id}  {science_note}  © OpenStreetMap contributors / ODbL"


def _ordered_sites(present: Sequence[str]) -> list[str]:
    return [site for site in SITE_ORDER if site in present]


def _records_index(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {(str(r["site_id"]), str(r["scope"]), str(r["stratum"])): r for r in records}


def _error_bar(record: Mapping[str, Any], key: str) -> tuple[float, float]:
    """把 ``<key>_ci95`` 轉成 matplotlib ``yerr`` 需要的 (下誤差, 上誤差)。"""

    rate = record.get(key)
    ci = record.get(f"{key}_ci95")
    if rate is None or ci is None:
        return (0.0, 0.0)
    low, high = ci
    return (max(0.0, float(rate) - float(low)), max(0.0, float(high) - float(rate)))


def render_five_site_summary(
    output_path: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    site_name_zh: Mapping[str, str],
    footer: str,
    dpi: int = 150,
) -> Path:
    """receptor scope、stratum=all 的五站摘要：三個出現率、核心強度、壽命。"""

    _, plt = _load_matplotlib()
    index = _records_index(records)
    sites = _ordered_sites(sorted({r["site_id"] for r in records if r["scope"] == "receptor" and r["stratum"] == "all"}))
    font = _site_font_properties()
    labels = [site_name_zh.get(site, site) for site in sites]
    rows = [index.get((site, "receptor", "all")) for site in sites]

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.24, top=0.86, wspace=0.32)

    x = np.arange(len(sites))
    width = 0.25
    occurrence_keys = ("core_occurrence_rate", "strong_core_occurrence_rate", "curve_occurrence_rate")
    occurrence_labels = ("core", "strong core", "curve")
    axis = axes[0]
    for offset, (key, label) in enumerate(zip(occurrence_keys, occurrence_labels, strict=True)):
        values = [float(row.get(key)) if row and row.get(key) is not None else 0.0 for row in rows]
        errors = np.array([_error_bar(row, key) if row else (0.0, 0.0) for row in rows]).T
        axis.bar(x + (offset - 1) * width, values, width=width, yerr=errors, capsize=3, label=label)
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels, fontproperties=font, rotation=20, ha="right")
    axis.set_ylabel("occurrence rate (receptor, all period)")
    axis.set_ylim(0.0, 1.05)
    axis.legend(fontsize=8)
    for xi, row in zip(x, rows, strict=True):
        n_valid = row.get("n_valid_snapshots") if row else 0
        axis.text(xi, 1.02, f"n={n_valid}", ha="center", va="bottom", fontsize=7)

    axis = axes[1]
    medians = [row.get("core_s1_median_day") if row else None for row in rows]
    p10s = [row.get("core_s1_p10_day") if row else None for row in rows]
    axis.bar(x - width / 2, [v if v is not None else 0.0 for v in medians], width=width, label="median")
    axis.bar(x + width / 2, [v if v is not None else 0.0 for v in p10s], width=width, label="p10 (strongest)")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels, fontproperties=font, rotation=20, ha="right")
    axis.set_ylabel(r"core $s_1$ (day$^{-1}$)")
    axis.legend(fontsize=8)

    axis = axes[2]
    lifetime_median = [row.get("track_median_lifetime_hours") if row else None for row in rows]
    axis.bar(x, [v if v is not None else 0.0 for v in lifetime_median], width=0.5, color="#546e7a")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels, fontproperties=font, rotation=20, ha="right")
    axis.set_ylabel("median uncensored lifetime (h)")

    figure.suptitle("Five-site TRAP summary (receptor scope, whole period)", fontsize=11)
    figure.text(0.5, 0.02, footer, ha="center", fontsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def render_seasonal_occurrence(
    output_path: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    site_name_zh: Mapping[str, str],
    footer: str,
    dpi: int = 150,
) -> Path:
    """receptor scope，逐季節分層的強核心與曲線出現率分組長條圖。"""

    _, plt = _load_matplotlib()
    index = _records_index(records)
    sites = _ordered_sites(sorted({r["site_id"] for r in records if r["scope"] == "receptor"}))
    font = _site_font_properties()
    labels = [site_name_zh.get(site, site) for site in sites]

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.85, wspace=0.25)
    x = np.arange(len(sites))
    width = 0.8 / len(SEASONS)
    for axis, key, title in zip(
        axes, ("strong_core_occurrence_rate", "curve_occurrence_rate"), ("Strong core occurrence", "Curve occurrence"), strict=True
    ):
        for offset, season in enumerate(SEASONS):
            values = []
            for site in sites:
                row = index.get((site, "receptor", season))
                values.append(float(row.get(key)) if row and row.get(key) is not None else 0.0)
            axis.bar(x + (offset - 1.5) * width, values, width=width, label=season)
        axis.set_xticks(list(x))
        axis.set_xticklabels(labels, fontproperties=font, rotation=20, ha="right")
        axis.set_ylabel("occurrence rate")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(title, fontsize=10)
        axis.legend(fontsize=7, ncols=4)

    figure.suptitle("Seasonal occurrence by site (receptor scope)", fontsize=11)
    figure.text(0.5, 0.02, footer, ha="center", fontsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def _ccdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (排序後數值, 互補累積機率)。"""

    if not values:
        return np.zeros(0), np.zeros(0)
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    n = sorted_values.size
    survival = 1.0 - (np.arange(1, n + 1) - 1) / n
    return sorted_values, survival


def render_lifetime_distribution(
    output_path: Path,
    *,
    sites: Sequence[str],
    site_name_zh: Mapping[str, str],
    raw_lifetimes: Mapping[str, Mapping[str, Sequence[float]]],
    footer: str,
    dpi: int = 150,
) -> Path:
    """五站 track 壽命 CCDF：未截尾實線、含截尾虛線（觀測跨度下界）。"""

    _, plt = _load_matplotlib()
    font = _site_font_properties()
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    ordered = _ordered_sites(sites)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])
    for index, site in enumerate(ordered):
        color = colors[index % len(colors)]
        uncensored = raw_lifetimes.get(site, {}).get("uncensored", [])
        all_values = raw_lifetimes.get(site, {}).get("all", [])
        x_u, y_u = _ccdf(uncensored)
        if x_u.size:
            axis.step(x_u, y_u, where="post", color=color, linestyle="-", label=f"{site_name_zh.get(site, site)} (uncensored)")
        x_a, y_a = _ccdf(all_values)
        if x_a.size:
            axis.step(x_a, y_a, where="post", color=color, linestyle="--", alpha=0.6)
    axis.set_yscale("log")
    axis.set_xlabel("lifetime (h)  [uncensored=solid, incl. censored=dashed]")
    axis.set_ylabel("P(lifetime ≥ x)")
    axis.set_title("Track lifetime CCDF by site", fontsize=11)
    axis.legend(fontsize=7, prop=font)
    figure.text(0.5, 0.02, footer, ha="center", fontsize=7)
    figure.subplots_adjust(bottom=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def render_spatial_domain_figure(
    output_path: Path,
    *,
    domain_id: str,
    spatial_all: Mapping[str, np.ndarray],
    grid: Mapping[str, Any],
    masks: Mapping[str, Mapping[str, np.ndarray]],
    site_anchor: Mapping[str, tuple[float, float, float]],
    colour_limits: Mapping[str, Sequence[float]],
    coastline: CoastlineGeometry | None,
    footer: str,
    dpi: int = 150,
) -> Path:
    """一個 domain 的四分空間圖：曲線通過率、核心頻率、平均核心強度、平均壽命。"""

    _, plt = _load_matplotlib()
    lon = np.asarray(grid["lon"], dtype=np.float64)
    lat = np.asarray(grid["lat"], dtype=np.float64)
    fields = (
        ("Curve passage frequency", spatial_all["curve_pass_frequency"], "fraction", "YlOrRd", colour_limits["curve_pass_frequency"]),
        ("Core frequency", spatial_all["core_frequency"], "fraction", "viridis", colour_limits["core_frequency"]),
        ("Mean core attraction", spatial_all["mean_core_s1_day"], r"$s_1$ (day$^{-1}$)", "magma", colour_limits["mean_core_s1_day"]),
        ("Mean uncensored lifetime", spatial_all["mean_lifetime_hours"], "hour", "plasma", colour_limits["mean_lifetime_hours"]),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.12, top=0.87, wspace=0.30, hspace=0.45)
    colorbar_pairs: list[tuple[Any, Any]] = []
    bbox = (float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat)))
    for axis, (title, field, label, cmap, (vmin, vmax)) in zip(axes.flat, fields, strict=True):
        masked = np.ma.masked_invalid(np.asarray(field, dtype=np.float64))
        mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
        colorbar.set_label(label)
        colorbar_pairs.append((axis, colorbar.ax))
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Longitude (°E)")
        axis.set_ylabel("Latitude (°N)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2, linewidth=0.5)
        for site_id in masks.get("receptor", {}):
            if site_id not in site_anchor:
                continue
            anchor_lon, anchor_lat, radius_m = site_anchor[site_id]
            radius_deg = radius_m / METERS_PER_DEGREE_LAT
            axis.add_patch(plt.Circle((anchor_lon, anchor_lat), radius_deg, fill=False, edgecolor="#37474f", linewidth=0.8, zorder=7))
            axis.plot(anchor_lon, anchor_lat, marker="+", color="#37474f", markersize=6, zorder=8)
        if coastline is not None:
            draw_geographic_coastline(axis, coastline, bbox=bbox, color="#263238", linewidth=0.6, zorder=6.0)
    align_colorbars_to_axes(figure, colorbar_pairs)
    figure.suptitle(f"Spatial TRAP structure — {domain_id}", fontsize=12)
    figure.text(0.5, 0.02, footer, ha="center", fontsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def render_all(
    output_dir: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    spatial: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    grids: Mapping[str, Mapping[str, Any]],
    masks: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    site_name_zh: Mapping[str, str],
    colour_limits: Mapping[str, Sequence[float]],
    coastline: CoastlineGeometry | None,
    state_label: str,
    param_id: str,
    period_label: str,
    raw_lifetimes: Mapping[str, Mapping[str, Sequence[float]]],
    site_anchor: Mapping[str, tuple[float, float, float]],
    site_domain: Mapping[str, str],
) -> list[str]:
    """輸出全部 aggregate 圖面，回傳相對 ``output_dir`` 的檔案路徑清單。"""

    footer = _footer(period_label, state_label, param_id)
    sites_present = sorted({str(r["site_id"]) for r in records})
    paths: list[str] = []

    render_five_site_summary(output_dir / "five_site_summary.png", records=records, site_name_zh=site_name_zh, footer=footer)
    paths.append("five_site_summary.png")

    render_seasonal_occurrence(output_dir / "seasonal_occurrence.png", records=records, site_name_zh=site_name_zh, footer=footer)
    paths.append("seasonal_occurrence.png")

    render_lifetime_distribution(
        output_dir / "lifetime_distribution.png",
        sites=sites_present,
        site_name_zh=site_name_zh,
        raw_lifetimes=raw_lifetimes,
        footer=footer,
    )
    paths.append("lifetime_distribution.png")

    for domain_id in sorted(spatial):
        render_spatial_domain_figure(
            output_dir / f"spatial_{domain_id}.png",
            domain_id=domain_id,
            spatial_all=spatial[domain_id]["all"],
            grid=grids[domain_id],
            masks=masks[domain_id],
            site_anchor=site_anchor,
            colour_limits=colour_limits,
            coastline=coastline,
            footer=footer,
        )
        paths.append(f"spatial_{domain_id}.png")

    return paths


__all__ = ["render_all"]
