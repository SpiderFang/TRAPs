"""熱點四季重現圖：每個格點在 DJF／MAM／JJA／SON 四季中有幾季屬於該季曲線通過頻率前 10% 熱點。

* 每季使用該季自己的站內 90 百分位門檻（與 :func:`traps.aggregate._top_quantile_mask` 同一規則：
  ``>= 門檻`` 且 ``> 0``，只用站內非 NaN 格點）；本模組重算遮罩並與 ``spatial_<季>.npz`` 內存的
  ``hotspot_curve_mask`` 逐格核對，不一致即中止。
* 圖面只顯示 1–4 季的格點（4 級離散色條，兩端平整）；0 季的格點留空，疊在淺灰分析海域上。
* 同時輸出每站統計（側車 JSON）：全年熱點格點中在 ≥3 季、4 季皆為熱點的比例，以及
  季熱點格點（聯集）依出現季數的分級比例。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SEASONS = ("DJF", "MAM", "JJA", "SON")
CLASS_COLOURS = ("#fdd07a", "#f98e52", "#d7301f", "#4a0d2e")
"""1–4 季的離散色：淺橙→橙→紅→深酒紅（4 季最深、與淺灰底及藍色海岸線對比最大）。"""
SITE_FILL_COLOUR = "#e1ecf5"
"""分析範圍（站內海域）的淡藍底色：與陸地的 #d9d9d9 淺灰不同色相，0 季的格點留空時仍一眼看出是分析範圍而非陸地。"""
COLOURBAR_LABEL = "熱點出現季數"
SIDECAR_SCHEMA = "traps.seasonal_recurrence.v1"
TITLE_TEXT = "熱點四季重現圖"


def top_quantile_mask(field: np.ndarray, site_mask: np.ndarray, quantile: float) -> tuple[np.ndarray, float | None]:
    """與 ``aggregate._top_quantile_mask`` 相同的規則（獨立實作以便本模組可單獨測試，並由比對確認一致）。"""

    field = np.asarray(field, dtype=np.float64)
    inside = np.asarray(site_mask, dtype=bool) & np.isfinite(field)
    values = field[inside]
    if values.size == 0:
        return np.zeros(field.shape, dtype=bool), None
    threshold = float(np.quantile(values, quantile))
    return inside & (field >= threshold) & (field > 0.0), threshold


def load_site_seasons(spatial_dir: Path, quantile: float) -> dict[str, Any]:
    """讀一個站四季＋全年的 npz，重算季熱點遮罩並與存檔遮罩核對。回傳計數場、遮罩、門檻與核對結果。"""

    masks: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    stored_match: dict[str, bool] = {}
    site_mask = lon = lat = None
    for season in SEASONS:
        with np.load(spatial_dir / f"spatial_{season}.npz") as z:
            data = {key: np.array(z[key]) for key in z.files}
        if site_mask is None:
            site_mask, lon, lat = data["site_mask"].astype(bool), data["lon"], data["lat"]
        elif not np.array_equal(site_mask, data["site_mask"].astype(bool)):
            raise ValueError(f"{spatial_dir}: {season} 的 site_mask 與其他季不同")
        mask, threshold = top_quantile_mask(data["curve_pass_frequency"], site_mask, quantile)
        stored = data["hotspot_curve_mask"].astype(bool)
        stored_match[season] = bool(np.array_equal(mask, stored)) and bool(
            np.isclose(threshold, float(data["hotspot_threshold_curve_pass_frequency"]))
        )
        if not stored_match[season]:
            raise ValueError(f"{spatial_dir}: {season} 重算的熱點遮罩與存檔不一致")
        masks[season] = mask
        thresholds[season] = float(threshold)
    with np.load(spatial_dir / "spatial_all.npz") as z:
        all_mask = z["hotspot_curve_mask"].astype(bool)
        all_threshold = float(z["hotspot_threshold_curve_pass_frequency"])
        all_cpf = np.array(z["curve_pass_frequency"])
    recomputed_all, recomputed_all_threshold = top_quantile_mask(all_cpf, site_mask, quantile)
    stored_match["all"] = bool(np.array_equal(recomputed_all, all_mask)) and bool(np.isclose(recomputed_all_threshold, all_threshold))
    if not stored_match["all"]:
        raise ValueError(f"{spatial_dir}: 全年重算的熱點遮罩與存檔不一致")
    count = np.zeros(site_mask.shape, dtype=np.int64)
    for season in SEASONS:
        count += masks[season].astype(np.int64)
    return {
        "lon": lon, "lat": lat, "site_mask": site_mask, "masks": masks, "thresholds": thresholds, "all_mask": all_mask,
        "all_threshold": all_threshold, "count": count, "stored_match": stored_match,
    }


def recurrence_statistics(loaded: Mapping[str, Any]) -> dict[str, Any]:
    """每站統計：全年熱點格點的重現比例與季熱點聯集的分級比例。"""

    count, all_mask, masks = loaded["count"], loaded["all_mask"], loaded["masks"]
    n_all = int(all_mask.sum())
    union = count >= 1
    n_union = int(union.sum())
    per_class = {str(k): int(np.sum(count == k)) for k in range(1, 5)}
    in_all = count[all_mask]
    return {
        "n_site_cells": int(loaded["site_mask"].sum()),
        "season_hotspot_cells": {season: int(masks[season].sum()) for season in SEASONS},
        "season_thresholds_curve_pass_frequency": {season: float(loaded["thresholds"][season]) for season in SEASONS},
        "all_year_hotspot_cells": n_all,
        "all_year_threshold_curve_pass_frequency": float(loaded["all_threshold"]),
        "all_year_hotspot_share_in_ge3_seasons": float(np.sum(in_all >= 3) / n_all) if n_all else None,
        "all_year_hotspot_share_in_all4_seasons": float(np.sum(in_all == 4) / n_all) if n_all else None,
        "all_year_hotspot_share_in_0_seasons": float(np.sum(in_all == 0) / n_all) if n_all else None,
        "all_year_hotspot_cells_ge3_seasons": int(np.sum(in_all >= 3)),
        "all_year_hotspot_cells_all4_seasons": int(np.sum(in_all == 4)),
        "season_hotspot_union_cells": n_union,
        "season_hotspot_cells_by_class": per_class,
        "season_hotspot_share_by_class": {key: (value / n_union if n_union else None) for key, value in per_class.items()},
        "share_of_site_cells_in_any_season": float(n_union / int(loaded["site_mask"].sum())),
        "stored_mask_recomputation_identical": dict(loaded["stored_match"]),
    }


def render_seasonal_recurrence(
    output_path: Path,
    figure_data: Any,
    count: np.ndarray,
    *,
    coastline: Any,
    percent_text: str,
) -> Path:
    """標準 MAP1 畫布上的四季重現圖（簡要報告樣式；先含標題與副標題，之後由 strip 步驟移除）。"""

    from . import concise_style
    from . import figure_standard as std
    from .aggregate_figures import (
        ANALYSIS_OUTLINE_COLOUR, ANALYSIS_OUTLINE_LABEL, SiteFigureData, _load_matplotlib, _map1_canvas, _prepare_map_axis, _save, _view_extent,
        concise_map_boxes, draw_analysis_outline, draw_flow_domain_boundary, draw_partner_overlay, standard_map_frame, style_colorbar, style_map_axis,
    )
    from .report_font import FIGURE_TEXT as T

    matplotlib, plt = _load_matplotlib()
    data: SiteFigureData = figure_data
    concise = concise_style.concise_enabled()
    if concise:
        data = replace(data, site_name_zh=concise_style.display_site_name(data.site_id, data.site_name_zh))
    rows, cols = _view_extent(data)
    (rows, cols), limits = standard_map_frame(data, (rows, cols), concise_map_boxes()[0], coastline=coastline)
    lon, lat = data.lon[cols], data.lat[rows]
    figure, axis, cax = _map1_canvas(plt, with_colorbar=True)

    partner = getattr(data, "partner", None) if concise else None
    site = np.asarray(data.site_mask, dtype=bool)
    shown_count = np.asarray(count)
    if partner:
        # A 區：搭檔站的統計範圍也畫出該站的四季重現（之後由淡化層弱化），整個 domain 連續呈現
        other_mask = np.asarray(partner["site_mask"], dtype=bool)
        shown_count = np.where(other_mask, np.asarray(partner["count"]), shown_count)
        site = site | other_mask
    site = site[rows, cols]
    axis.pcolormesh(
        lon, lat, np.ma.masked_where(~site, np.ones(site.shape)), shading="auto",
        cmap=matplotlib.colors.ListedColormap([SITE_FILL_COLOUR]), vmin=0.0, vmax=1.0, zorder=1,
    )
    cmap = matplotlib.colors.ListedColormap(list(CLASS_COLOURS))
    norm = matplotlib.colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    shown = np.ma.masked_less(shown_count[rows, cols].astype(float), 1.0)
    mesh = axis.pcolormesh(lon, lat, shown, shading="auto", cmap=cmap, norm=norm, zorder=2)
    colorbar = figure.colorbar(mesh, cax=cax, ticks=[1, 2, 3, 4], extend="neither")
    bar_label = concise_style.term("recurrence_colourbar", COLOURBAR_LABEL)
    style_colorbar(colorbar, bar_label)
    if concise:
        concise_style.style_colorbar_discrete(colorbar, [1, 2, 3, 4], bar_label, name="seasonal_recurrence_seasons")
    buffer_handle = _prepare_map_axis(axis, data, (rows, cols), coastline, max_ticks=4, limits=limits, lbt_style=concise)
    style_map_axis(axis)
    # 分析範圍：淡藍底色（圖例項目「分析範圍」）；外框只畫與站外有效海域相鄰的邊，圖例色塊的深灰框線與實際畫法一致：有外框線段才加框
    outline_handle = draw_analysis_outline(axis, data, limits)
    if outline_handle is not None:
        range_patch = matplotlib.patches.Patch(facecolor=SITE_FILL_COLOUR, edgecolor=ANALYSIS_OUTLINE_COLOUR, linewidth=0.9, label=ANALYSIS_OUTLINE_LABEL)
    else:
        range_patch = matplotlib.patches.Patch(facecolor=SITE_FILL_COLOUR, edgecolor="none", linewidth=0.0, label=ANALYSIS_OUTLINE_LABEL)
    handles = [range_patch]
    if partner:
        partner_handle = draw_partner_overlay(axis, data, (rows, cols), limits)  # A 區：搭檔站範圍淡化、統計歸屬分界
        if partner_handle is not None:
            handles.append(partner_handle)
    if buffer_handle is not None:
        handles.append(buffer_handle)
    if concise:
        handles.append(draw_flow_domain_boundary(axis, data, limits))  # 簡要報告：流場範圍邊界（流場網格外框，深藍色長虛線）；流場範圍外的海域維持白色
    box = concise_map_boxes()[0]
    std.legend_below(
        figure, std.SINGLE_HEIGHT_INCH, handles, [handle.get_label() for handle in handles], framed=concise, center_x_inch=(box.x + 0.5 * box.w) if concise else None
    )
    std.draw_title_block(
        figure,
        std.SINGLE_HEIGHT_INCH,
        f"{data.site_name_zh} — {TITLE_TEXT}",
        f"四季各自的{T['curve_pass_frequency']}{T['hotspot_top']} {percent_text}% 熱點｜逐時產品",
    )
    _save(figure, plt, output_path, 150)
    return output_path


def sidecar_payload(site_id: str, stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SIDECAR_SCHEMA,
        "site_id": site_id,
        "product": "r1_hourly",
        "rule": "各季站內曲線通過頻率 90 百分位門檻；格點 >= 門檻 且 > 0（同 aggregate._top_quantile_mask）；以 spatial_<季>.npz 重算並與存檔遮罩逐格核對",
        "classes": {"1": "1 季", "2": "2 季", "3": "3 季", "4": "4 季"},
        "class_colours": dict(zip(["1", "2", "3", "4"], CLASS_COLOURS, strict=True)),
        "all_year_reference": "spatial_all.npz 的 hotspot_curve_mask（未濾除小連通塊的統計定義）",
        **stats,
    }


def write_sidecar(path: Path, site_id: str, stats: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar_payload(site_id, stats), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "CLASS_COLOURS", "COLOURBAR_LABEL", "SEASONS", "load_site_seasons", "recurrence_statistics",
    "render_seasonal_recurrence", "top_quantile_mask", "write_sidecar",
]
