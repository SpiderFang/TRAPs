"""``aggregate`` 的逐站成果圖：每個研究站各一組五張圖，站與站之間互不比較。

每個站輸出到 ``figures/<site_id>/``：

- ``site_<site_id>_spatial.png``：站點分析區內的 2×2 空間圖；
- ``site_<site_id>_hotspots.png``：曲線通過頻率前段格點（熱點）與核心頻率熱點輪廓；
- ``site_<site_id>_seasonal.png``：四季曲線通過頻率小圖與季節強核心出現率／平均通過格點比例；
- ``site_<site_id>_lifetime.png``：壽命 CCDF 與吸引強度隨正規化年齡演化；
- ``site_<site_id>_timeseries.png``：逐日強核心出現比例與平均通過格點比例（30 日滑動平均）。

色階上下限、熱點與門檻全部只取自該站自己的資料。所有固定文字來自
:data:`traps.report_font.FIGURE_TEXT`，並在 :func:`traps.report_font.cjk_font_context`
內繪製，因此不會出現缺字方框；沒有合格 CJK 字型（``font_selection is None``）時
一律拒絕繪圖。圖面只描述模型內部「表層短期吸引潛勢」，不是實測海廢濃度或污染來源。
空間圖沿用既有 ``coastline.py``、``colorbar_layout.py`` 的海岸線疊圖與色條對齊規則。
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .coastline import CoastlineGeometry, draw_geographic_coastline
from .colorbar_layout import align_colorbars_to_axes
from .report_font import FIGURE_TEXT as T
from .report_font import CJKFontSelection, cjk_font_context

SEASONS = ("DJF", "MAM", "JJA", "SON")
RUNNING_MEAN_DAYS = 30
S1_DAY_LABEL = r"$s_1$ ($\mathrm{day^{-1}}$)"
"""核心 s1 的單位標籤（mathtext，以 DejaVu 排版，不依賴 CJK 字型）。"""


@dataclass
class SiteFigureData:
    """一個研究站繪圖所需的全部資料（已限制在該站分析區、不含其他站）。"""

    site_id: str
    site_name_zh: str
    lon: np.ndarray
    lat: np.ndarray
    site_mask: np.ndarray
    cell_area_m2: np.ndarray
    spatial_all: Mapping[str, np.ndarray]
    spatial_seasons: Mapping[str, Mapping[str, np.ndarray]]
    hotspot_curve_mask: np.ndarray
    hotspot_core_mask: np.ndarray
    hotspot_metrics: Mapping[str, Any]
    colour_limits: Mapping[str, Sequence[float]]
    records: Mapping[str, Mapping[str, Any]]
    lifetimes_uncensored: Sequence[float]
    lifetimes_all: Sequence[float]
    evolution_mean_day: Sequence[float]
    evolution_track_count: int
    day_stats: Mapping[str, Sequence[float]]
    anchor_lonlat: tuple[float, float]
    n_valid_snapshots: int


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


def _footer(period_label: str, state_label: str, param_id: str, n_valid: int, font_family: str, extra: str = "") -> str:
    """所有 aggregate 圖共用的兩行註腳：期間、UTC、param_id、有效快照數、科學狀態、來源與字型。"""

    science = T["science_pilot_only"] if state_label == "PILOT_ONLY" else T["science_model_only"]
    line1 = f"{T['footer_period']} {period_label}  UTC  param_id={param_id}  {T['footer_valid']} n={n_valid}"
    if extra:
        line1 += f"  {extra}"
    line2 = f"{science}  {T['footer_attribution']}  {T['footer_font']}: {font_family}"
    return f"{line1}\n{line2}"


def _error_bar(record: Mapping[str, Any] | None, key: str) -> tuple[float, float]:
    """把 ``<key>_ci95`` 轉成 matplotlib ``yerr`` 需要的 (下誤差, 上誤差)。"""

    if not record:
        return (0.0, 0.0)
    rate = record.get(key)
    ci = record.get(f"{key}_ci95")
    if rate is None or ci is None:
        return (0.0, 0.0)
    low, high = ci
    return (max(0.0, float(rate) - float(low)), max(0.0, float(high) - float(rate)))


def _ccdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (排序後數值, 互補累積機率 P(X >= x))。"""

    if len(values) == 0:
        return np.zeros(0), np.zeros(0)
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    n = sorted_values.size
    survival = 1.0 - (np.arange(1, n + 1) - 1) / n
    return sorted_values, survival


def running_mean_with_gaps(values: np.ndarray, window: int = RUNNING_MEAN_DAYS, min_fraction: float = 0.5) -> np.ndarray:
    """置中的 ``window`` 日滑動平均，資料缺口保留為 NaN（圖上斷線），不內插也不補零。

    只平均視窗內有限值；視窗內有效日數少於 ``min_fraction * window``，或當天本身
    缺資料時輸出 NaN，使缺口呈現為斷點而不是被平滑蓋掉；序列端點的視窗被截短，
    有效日數不足時同樣是 NaN。
    """

    array = np.asarray(values, dtype=np.float64)
    count = array.size
    valid = np.isfinite(array)
    filled = np.where(valid, array, 0.0)
    cumulative_value = np.concatenate([[0.0], np.cumsum(filled)])
    cumulative_count = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
    index = np.arange(count)
    half_low = window // 2
    half_high = window - half_low - 1
    low = np.clip(index - half_low, 0, count)
    high = np.clip(index + half_high + 1, 0, count)
    window_count = cumulative_count[high] - cumulative_count[low]
    window_sum = cumulative_value[high] - cumulative_value[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = window_sum / np.maximum(window_count, 1)
    return np.where(valid & (window_count >= min_fraction * window), mean, np.nan)


def daily_series(day_stats: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 ``day -> [n, core, strong, curve, pass_sum]`` 轉成連續日曆序列。

    回傳 (日期 ``datetime64[D]``, 含強核心有效快照比例, 平均通過格點比例)；沒有任何
    有效快照的日子是 NaN（缺口），不是 0。
    """

    if not day_stats:
        return np.zeros(0, dtype="datetime64[D]"), np.zeros(0), np.zeros(0)
    days = sorted(day_stats)
    first = np.datetime64(days[0], "D")
    last = np.datetime64(days[-1], "D")
    calendar = np.arange(first, last + np.timedelta64(1, "D"), dtype="datetime64[D]")
    strong = np.full(calendar.size, np.nan)
    passed = np.full(calendar.size, np.nan)
    for day in days:
        n_valid, _core, n_strong, _curve, pass_sum = day_stats[day]
        if n_valid > 0:
            position = int((np.datetime64(day, "D") - first).astype(int))
            strong[position] = n_strong / n_valid
            passed[position] = pass_sum / n_valid
    return calendar, strong, passed


def _view_extent(data: SiteFigureData, margin_cells: int = 2) -> tuple[slice, slice]:
    """站點分析區外接框（含少量邊界格）的 (row, col) 切片，讓地圖聚焦在該站分析區。"""

    rows = np.nonzero(np.any(data.site_mask, axis=1))[0]
    cols = np.nonzero(np.any(data.site_mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return slice(0, data.site_mask.shape[0]), slice(0, data.site_mask.shape[1])
    r0 = max(int(rows[0]) - margin_cells, 0)
    r1 = min(int(rows[-1]) + margin_cells + 1, data.site_mask.shape[0])
    c0 = max(int(cols[0]) - margin_cells, 0)
    c1 = min(int(cols[-1]) + margin_cells + 1, data.site_mask.shape[1])
    return slice(r0, r1), slice(c0, c1)


def _prepare_map_axis(
    axis: Any, data: SiteFigureData, view: tuple[slice, slice], coastline: CoastlineGeometry | None, *, max_ticks: int | None = None
) -> None:
    """設定經緯度軸標籤、等比例、海岸線疊圖與站點錨點（只有標記，沒有半徑圓）。"""

    rows, cols = view
    lon = data.lon[cols]
    lat = data.lat[rows]
    half_lon = float(np.median(np.diff(lon))) / 2.0 if lon.size > 1 else 0.01
    half_lat = float(np.median(np.diff(lat))) / 2.0 if lat.size > 1 else 0.01
    axis.set_xlim(float(lon.min()) - half_lon, float(lon.max()) + half_lon)
    axis.set_ylim(float(lat.min()) - half_lat, float(lat.max()) + half_lat)
    axis.set_xlabel(T["lon"])
    axis.set_ylabel(T["lat"])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.ticklabel_format(useOffset=False, style="plain")
    if max_ticks is not None:
        from matplotlib.ticker import MaxNLocator

        axis.xaxis.set_major_locator(MaxNLocator(max_ticks))
        axis.yaxis.set_major_locator(MaxNLocator(max_ticks + 1))
    if coastline is not None:
        bbox = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
        draw_geographic_coastline(axis, coastline, bbox=bbox, color="#263238", linewidth=0.6, zorder=6.0)
    anchor_lon, anchor_lat = data.anchor_lonlat
    if lon.min() <= anchor_lon <= lon.max() and lat.min() <= anchor_lat <= lat.max():
        axis.plot(anchor_lon, anchor_lat, marker="+", color="#37474f", markersize=8, markeredgewidth=1.4, linestyle="none", zorder=8)


def _save(figure: Any, plt: Any, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def render_spatial(output_path: Path, data: SiteFigureData, *, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150) -> Path:
    """站點分析區內的 2×2 空間圖：曲線通過頻率、核心頻率、平均核心 s1、平均未截尾壽命。"""

    _, plt = _load_matplotlib()
    rows, cols = _view_extent(data)
    lon, lat = data.lon[cols], data.lat[rows]
    fields = (
        (T["curve_pass_frequency"], "curve_pass_frequency", T["fraction"], "YlOrRd"),
        (T["core_frequency"], "core_frequency", T["fraction"], "viridis"),
        (f"{T['mean_core_s1']} {S1_DAY_LABEL}", "mean_core_s1_day", S1_DAY_LABEL, "magma"),
        (T["mean_lifetime"], "mean_lifetime_hours", T["hour"], "plasma"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 10.4))
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.13, top=0.89, wspace=0.30, hspace=0.42)
    pairs: list[tuple[Any, Any]] = []
    for axis, (title, key, label, cmap) in zip(axes.flat, fields, strict=True):
        vmin, vmax = data.colour_limits[key]
        masked = np.ma.masked_invalid(np.asarray(data.spatial_all[key], dtype=np.float64)[rows, cols])
        mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
        colorbar.set_label(label)
        pairs.append((axis, colorbar.ax))
        axis.set_title(title, fontsize=10)
        _prepare_map_axis(axis, data, (rows, cols), coastline)
    axes.flat[0].plot([], [], marker="+", color="#37474f", linestyle="none", label=T["anchor"])
    axes.flat[0].legend(loc="lower right", fontsize=7)
    align_colorbars_to_axes(figure, pairs)
    figure.suptitle(f"{data.site_name_zh} — {T['spatial_title']}", fontsize=13)
    figure.text(0.5, 0.02, footer, ha="center", fontsize=7)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_hotspots(
    output_path: Path, data: SiteFigureData, *, hotspot_quantile: float, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150
) -> Path:
    """曲線通過頻率場上標出熱點格點，並疊加核心頻率熱點輪廓；標題註明熱點面積。"""

    matplotlib, plt = _load_matplotlib()
    rows, cols = _view_extent(data)
    lon, lat = data.lon[cols], data.lat[rows]
    vmin, vmax = data.colour_limits["curve_pass_frequency"]
    figure, axis = plt.subplots(figsize=(8.6, 8.2))
    figure.subplots_adjust(left=0.11, right=0.90, bottom=0.17, top=0.88)
    field = np.ma.masked_invalid(np.asarray(data.spatial_all["curve_pass_frequency"], dtype=np.float64)[rows, cols])
    mesh = axis.pcolormesh(lon, lat, field, shading="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label(f"{T['curve_pass_frequency']} ({T['fraction']})")

    hot = np.asarray(data.hotspot_curve_mask, dtype=bool)[rows, cols]
    handles = []
    if hot.any():
        overlay = np.ma.masked_where(~hot, np.ones(hot.shape))
        axis.pcolormesh(
            lon, lat, overlay, shading="auto", cmap=matplotlib.colors.ListedColormap(["#e63946"]), alpha=0.55, zorder=3
        )
        handles.append(matplotlib.patches.Patch(facecolor="#e63946", alpha=0.55, label=T["hotspot_curve_legend"]))
    core_hot = np.asarray(data.hotspot_core_mask, dtype=bool)[rows, cols]
    if core_hot.any() and not core_hot.all():
        axis.contour(lon, lat, core_hot.astype(np.float64), levels=[0.5], colors="#111111", linewidths=1.3, zorder=5)
        handles.append(matplotlib.lines.Line2D([], [], color="#111111", linewidth=1.3, label=T["hotspot_core_legend"]))
    _prepare_map_axis(axis, data, (rows, cols), coastline, max_ticks=5)
    if handles:
        axis.legend(handles=handles, loc="lower right", fontsize=7)

    metrics = data.hotspot_metrics
    percent = f"{100.0 * (1.0 - float(metrics['hotspot_quantile'])):g}"
    area = float(metrics["hotspot_area_km2"])
    axis.set_title(
        f"{data.site_name_zh} — {T['hotspot_title']}（{T['hotspot_top']} {percent}%，{T['hotspot_area']} {area:,.0f} km²）", fontsize=11
    )
    threshold = metrics["hotspot_threshold_curve_pass_frequency"]
    threshold_text = "n/a" if threshold is None else f"{float(threshold):.3g}"
    extra = (
        f"{T['footer_hotspot_area']} {area:,.1f} km²  {T['footer_hotspot_cells']} {int(metrics['hotspot_cell_count'])}"
        f"  {T['footer_threshold']} {threshold_text}"
    )
    figure.text(0.5, 0.02, footer + "\n" + extra, ha="center", fontsize=7)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_seasonal(output_path: Path, data: SiteFigureData, *, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150) -> Path:
    """DJF/MAM/JJA/SON 曲線通過頻率小圖（站內共用色階），加季節強核心出現率與平均通過格點比例（含 CI）。"""

    _, plt = _load_matplotlib()
    rows, cols = _view_extent(data)
    lon, lat = data.lon[cols], data.lat[rows]
    vmin, vmax = data.colour_limits["curve_pass_frequency"]
    figure = plt.figure(figsize=(14.0, 9.4))
    grid = figure.add_gridspec(2, 8, height_ratios=[1.3, 1.0], left=0.06, right=0.97, bottom=0.14, top=0.90, wspace=0.9, hspace=0.42)
    map_axes = []
    mesh = None
    for index, season in enumerate(SEASONS):
        axis = figure.add_subplot(grid[0, 2 * index : 2 * index + 2])
        masked = np.ma.masked_invalid(np.asarray(data.spatial_seasons[season]["curve_pass_frequency"], dtype=np.float64)[rows, cols])
        mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
        axis.set_title(T[f"season_{season}"], fontsize=10)
        _prepare_map_axis(axis, data, (rows, cols), coastline, max_ticks=3)
        if index > 0:
            axis.set_ylabel("")
        map_axes.append(axis)
    if mesh is not None:
        colorbar = figure.colorbar(mesh, ax=map_axes, orientation="horizontal", fraction=0.045, pad=0.22, aspect=45)
        colorbar.set_label(f"{T['curve_pass_frequency']} ({T['fraction']})")

    x = np.arange(len(SEASONS))
    for column, (key, title) in enumerate(
        (("strong_core_occurrence_rate", T["strong_core_occurrence"]), ("mean_pass_cell_fraction", T["mean_pass_fraction"]))
    ):
        axis = figure.add_subplot(grid[1, 4 * column : 4 * column + 4])
        values, errors, counts = [], [], []
        for season in SEASONS:
            record = data.records.get(season)
            value = record.get(key) if record else None
            values.append(float(value) if value is not None else 0.0)
            errors.append(_error_bar(record, key))
            counts.append(int(record["n_valid_snapshots"]) if record else 0)
        axis.bar(x, values, yerr=np.asarray(errors).T, capsize=4, color="#4c78a8", label=T["ci95"])
        axis.set_xticks(list(x))
        axis.set_xticklabels([T[f"season_{season}"] for season in SEASONS], fontsize=8)
        axis.set_ylabel(T["fraction"])
        axis.set_title(f"{title}（{T['ci95']}）", fontsize=10)
        tops = [value + high for value, (_low, high) in zip(values, errors, strict=True)]
        top = max(tops + [1.0e-9])
        for xi, count, level in zip(x, counts, tops, strict=True):
            axis.text(xi, level + 0.02 * top, f"n={count}" if count else T["no_data"], ha="center", va="bottom", fontsize=7)
        axis.set_ylim(0.0, top * 1.2)
    figure.suptitle(f"{data.site_name_zh} — {T['seasonal_title']}", fontsize=13)
    figure.text(0.5, 0.02, footer, ha="center", fontsize=7)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_lifetime(output_path: Path, data: SiteFigureData, *, footer: str, dpi: int = 150) -> Path:
    """壽命 CCDF（未截尾實線、含截尾虛線），以及吸引強度隨正規化年齡的演化（Kunz et al. 2024 Fig. 5b 風格）。"""

    _, plt = _load_matplotlib()
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.6))
    figure.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.86, wspace=0.25)

    x_u, y_u = _ccdf(data.lifetimes_uncensored)
    x_a, y_a = _ccdf(data.lifetimes_all)
    if x_u.size:
        left.step(x_u, y_u, where="post", color="#1b4965", linestyle="-", label=f"{T['uncensored']} (n={x_u.size})")
    if x_a.size:
        left.step(x_a, y_a, where="post", color="#1b4965", linestyle="--", alpha=0.7, label=f"{T['all_tracks']} (n={x_a.size})")
    if x_u.size or x_a.size:
        left.set_yscale("log")
        left.legend(fontsize=8)
    else:
        left.text(0.5, 0.5, T["no_data"], ha="center", va="center", transform=left.transAxes)
    left.set_xlabel(T["lifetime_hours"])
    left.set_ylabel(T["ccdf_ylabel"])
    left.set_title(T["ccdf_title"], fontsize=10)
    left.grid(alpha=0.25)

    age = np.linspace(0.0, 1.0, len(data.evolution_mean_day)) if len(data.evolution_mean_day) else np.zeros(0)
    if age.size:
        right.plot(age, np.asarray(data.evolution_mean_day, dtype=np.float64), color="#c1121f", marker="o", markersize=3.5, linewidth=1.6)
        right.text(
            0.03, 0.05, f"n = {data.evolution_track_count} {T['n_tracks']}", transform=right.transAxes, fontsize=9, ha="left", va="bottom"
        )
    else:
        right.text(0.5, 0.5, T["no_data"], ha="center", va="center", transform=right.transAxes)
    right.set_xlim(0.0, 1.0)
    right.set_xlabel(T["normalized_age"])
    right.set_ylabel(f"{T['mean_core_s1']} {S1_DAY_LABEL}")
    right.set_title(f"{T['evolution_title']}\n{T['evolution_subtitle']}", fontsize=10)
    right.grid(alpha=0.25)

    figure.suptitle(f"{data.site_name_zh} — {T['lifetime_title']}", fontsize=13)
    figure.text(0.5, 0.02, footer, ha="center", fontsize=7)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_timeseries(output_path: Path, data: SiteFigureData, *, footer: str, dpi: int = 150) -> Path:
    """逐日含強核心的有效快照比例與平均通過格點比例；30 日滑動平均，資料缺口以斷線呈現。"""

    _, plt = _load_matplotlib()
    calendar, strong, passed = daily_series(data.day_stats)
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True)
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.90, hspace=0.18)
    for axis, series, label, color in (
        (axes[0], strong, T["strong_core_daily"], "#c1121f"),
        (axes[1], passed, T["pass_fraction_daily"], "#1b4965"),
    ):
        if calendar.size:
            axis.plot(calendar, series, color=color, linewidth=0.6, alpha=0.45, label=T["daily"])
            axis.plot(calendar, running_mean_with_gaps(series), color=color, linewidth=2.0, label=T["running_mean_30d"])
            axis.legend(fontsize=8, loc="upper right")
        else:
            axis.text(0.5, 0.5, T["no_data"], ha="center", va="center", transform=axis.transAxes)
        axis.set_ylabel(label, fontsize=9)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    axes[1].set_xlabel(T["date"])
    figure.suptitle(f"{data.site_name_zh} — {T['timeseries_title']}", fontsize=13)
    figure.text(0.5, 0.02, footer, ha="center", fontsize=7)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_site_figures(
    output_dir: Path,
    data: SiteFigureData,
    *,
    font_selection: CJKFontSelection | None,
    coastline: CoastlineGeometry | None,
    state_label: str,
    param_id: str,
    period_label: str,
    hotspot_quantile: float | None = None,
    dpi: int = 150,
) -> list[str]:
    """輸出一個站的五張圖到 ``figures/<site_id>/``，回傳相對 ``output_dir`` 的路徑清單。

    ``font_selection`` 必須是已通過字元涵蓋檢查的 CJK 字型；為 ``None`` 時拒絕繪圖
    （圖面不可輸出方框字）。繪製期間以 :func:`cjk_font_context` 暫時設定
    ``font.family``，結束後還原全域 rcParams。
    """

    if font_selection is None:
        raise ValueError("沒有合格的 CJK 字型，拒絕輸出成果圖（可改用 --no-figures）")
    quantile = float(hotspot_quantile if hotspot_quantile is not None else data.hotspot_metrics["hotspot_quantile"])
    footer = _footer(period_label, state_label, param_id, data.n_valid_snapshots, font_selection.family)
    site_dir = output_dir / "figures" / data.site_id
    stem = f"site_{data.site_id}"
    with cjk_font_context(font_selection):
        render_spatial(site_dir / f"{stem}_spatial.png", data, coastline=coastline, footer=footer, dpi=dpi)
        render_hotspots(site_dir / f"{stem}_hotspots.png", data, hotspot_quantile=quantile, coastline=coastline, footer=footer, dpi=dpi)
        render_seasonal(site_dir / f"{stem}_seasonal.png", data, coastline=coastline, footer=footer, dpi=dpi)
        render_lifetime(site_dir / f"{stem}_lifetime.png", data, footer=footer, dpi=dpi)
        render_timeseries(site_dir / f"{stem}_timeseries.png", data, footer=footer, dpi=dpi)
    return [f"figures/{data.site_id}/{stem}_{kind}.png" for kind in ("spatial", "hotspots", "seasonal", "lifetime", "timeseries")]


__all__ = [
    "SiteFigureData",
    "daily_series",
    "render_site_figures",
    "running_mean_with_gaps",
]
