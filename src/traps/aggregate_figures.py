"""``aggregate`` 的逐站成果圖：每個研究站各一組五張圖，站與站之間互不比較。

每個站輸出到 ``figures/<site_id>/``：

- ``site_<site_id>_spatial.png``：站點分析區內的 2×2 空間圖；
- ``site_<site_id>_hotspots.png``：曲線通過頻率前段格點（熱點）與核心頻率熱點格點；
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
class CoastalDiagnosticData:
    """近岸診斷圖（``site_<id>_coastal_diagnostic.png``）所需的彙整資料。

    直方圖為固定 bin 的計數（不保留逐核心陣列，避免全期核心數造成記憶體負擔）；壽命列表只含
    未截尾 track，依 track 平均核心距岸距離是否小於緩衝帶分為近岸型與近岸外，並附各 seed 的隨機對照。
    """

    buffer_km: float
    hist_edges_km: np.ndarray
    hist_all: np.ndarray
    hist_strong: np.ndarray
    hist_long: np.ndarray
    lifetimes_offshore: Sequence[float] = ()
    lifetimes_coastal: Sequence[float] = ()
    null_offshore: Sequence[Sequence[float]] = ()
    null_coastal: Sequence[Sequence[float]] = ()
    nominal_step_hours: float = 1.0


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
    hotspot_curve_mask_filtered: np.ndarray | None = None
    """濾除小連通塊後的曲線熱點遮罩（只供繪圖）；``None`` 表示不濾除。"""
    hotspot_core_mask_filtered: np.ndarray | None = None
    """濾除小連通塊後的核心熱點遮罩（只供繪圖）；``None`` 表示不濾除。"""
    hotspot_min_cells: int = 1
    """圖面熱點連通塊最小格數，僅用於註腳文字。"""
    lifetime_thresholds_hours: Sequence[float] = (6.0, 12.0, 24.0)
    """壽命 CCDF 圖上標示的門檻（小時）。"""
    lifetime_evolution_min_hours: float = 12.0
    """吸引強度演化面板納入的未截尾軌跡最短壽命（小時）。"""
    null_lifetimes: Sequence[Sequence[float]] = ()
    """時間置換對照組：每個 seed 一組該站未截尾壽命（小時）；空序列表示沒有對照組。"""
    nominal_step_hours: float = 1.0
    """名目時間步長（小時）；24 或門檻皆為 24 的倍數時，壽命 CCDF 以「天」顯示。"""
    lifetime_scope_text: str = ""
    """壽命統計對象說明（例如「強核心結構（各站前 10%）」）；空字串表示不註明。"""
    coast_distance_m: np.ndarray | None = None
    """整個 domain 的核心距最近非分析海域格點距離場（m，與 lon/lat 同形）；``None`` 時不畫近岸緩衝帶。"""
    coast_buffer_km: float = 3.0
    """近岸緩衝帶寬度（km）。"""
    coastal: CoastalDiagnosticData | None = None
    """近岸診斷圖資料；``None`` 時不輸出 ``coastal_diagnostic``。"""


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


def null_ccdf_band(
    null_lifetimes: Sequence[Sequence[float]], extra_values: Sequence[float] = ()
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """各 seed 的 CCDF P(X >= x) 在共同 x 格點上的（x、跨 seed 平均、最小、最大）。

    共同格點為所有 seed（及 ``extra_values``）壽命值的聯集；沒有未截尾壽命的 seed 不納入。
    沒有任何有效 seed 時回傳四個空陣列。
    """

    arrays = [np.sort(np.asarray(values, dtype=np.float64)) for values in null_lifetimes if len(values)]
    if not arrays:
        empty = np.zeros(0)
        return empty, empty, empty, empty
    grid = np.unique(np.concatenate([*arrays, np.asarray(extra_values, dtype=np.float64)]))
    survival = np.stack([1.0 - np.searchsorted(values, grid, side="left") / values.size for values in arrays], axis=0)
    return grid, survival.mean(axis=0), survival.min(axis=0), survival.max(axis=0)


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


FOOTER_LINE_INCH = 0.135
"""註腳每行（7 pt）預留的高度（英吋）。"""
FOOTER_BASE_INCH = 0.14


def matplotlib_ticker_null() -> Any:
    from matplotlib.ticker import NullLocator

    return NullLocator()


def _hours_label(hours: float) -> str:
    """把小時數寫成不帶多餘小數的文字（``12`` 或 ``1.5``）。"""

    return f"{float(hours):g}"


def _footer_height_inch(footer: str) -> float:
    return FOOTER_BASE_INCH + FOOTER_LINE_INCH * (footer.count("\n") + 1)


def _place_footer(figure: Any, footer: str) -> None:
    """把註腳貼在 figure 最底端（以英吋定位），配合下邊界預留高度，不會壓到座標軸或 xlabel。"""

    figure.text(0.5, FOOTER_BASE_INCH / 2.0 / figure.get_figheight(), footer, ha="center", va="bottom", fontsize=7)


def _view_limits(data: SiteFigureData, view: tuple[slice, slice]) -> tuple[tuple[float, float], tuple[float, float]]:
    """地圖視窗的 (xlim, ylim)：格點中心外擴半格，使 pcolormesh 格緣剛好貼齊座標框。"""

    rows, cols = view
    lon = data.lon[cols]
    lat = data.lat[rows]
    half_lon = float(np.median(np.diff(lon))) / 2.0 if lon.size > 1 else 0.01
    half_lat = float(np.median(np.diff(lat))) / 2.0 if lat.size > 1 else 0.01
    return (
        (float(lon.min()) - half_lon, float(lon.max()) + half_lon),
        (float(lat.min()) - half_lat, float(lat.max()) + half_lat),
    )


def _lat_scale(ylim: tuple[float, float]) -> float:
    """經度→等距長度的縮放 ``cos(平均緯度)``（避免極端值使比例失控）。"""

    return max(float(np.cos(np.radians(0.5 * (ylim[0] + ylim[1])))), 0.05)


def map_data_aspect(data: SiteFigureData, view: tuple[slice, slice]) -> float:
    """地圖資料寬高比 ``(經度跨度 × cos 平均緯度) / 緯度跨度``（>1 為寬扁）。"""

    xlim, ylim = _view_limits(data, view)
    lat_extent = max(ylim[1] - ylim[0], 1.0e-12)
    return (xlim[1] - xlim[0]) * _lat_scale(ylim) / lat_extent


def _fit_panel(aspect: float, max_width: float, max_height: float, min_side: float = 0.5) -> tuple[float, float]:
    """在 ``max_width × max_height`` 英吋內放入寬高比為 ``aspect`` 的最大地圖框。"""

    width = max_width
    height = width / aspect
    if height > max_height:
        height = max_height
        width = height * aspect
    return max(width, min_side), max(height, min_side)


def _add_inch_axes(figure: Any, x: float, y: float, width: float, height: float) -> Any:
    """以英吋（左下角起算）建立 axes。"""

    fig_w, fig_h = figure.get_size_inches()
    return figure.add_axes((x / fig_w, y / fig_h, width / fig_w, height / fig_h))


def coast_buffer_label(data: SiteFigureData) -> str:
    """近岸緩衝帶圖例文字，例如「近岸 3 km 帶」。"""

    return f"{T['coast_buffer_prefix']} {float(data.coast_buffer_km):g} {T['coast_buffer_suffix']}"


COAST_BUFFER_COLOUR = "#6a3d9a"


def _draw_coast_buffer(axis: Any, data: SiteFigureData, view: tuple[slice, slice]) -> Any | None:
    """在地圖上以細虛線畫出近岸緩衝帶外緣（距離場的 buffer 等值線），回傳圖例 handle。

    距離場整個 domain 計算，等值線在視窗內不存在（全部近岸內或全部近岸外）時不畫也不回傳 handle。
    """

    if data.coast_distance_m is None:
        return None
    from matplotlib.lines import Line2D

    rows, cols = view
    level = float(data.coast_buffer_km) * 1000.0
    field = np.asarray(data.coast_distance_m, dtype=np.float64)[rows, cols]
    finite = field[np.isfinite(field)]
    if finite.size == 0 or not (float(finite.min()) < level < float(finite.max())):
        return None
    axis.contour(
        data.lon[cols], data.lat[rows], np.ma.masked_invalid(field), levels=[level], colors=COAST_BUFFER_COLOUR,
        linewidths=0.7, linestyles="--", zorder=6.5,
    )
    return Line2D([], [], color=COAST_BUFFER_COLOUR, linestyle="--", linewidth=1.0, label=coast_buffer_label(data))


def _prepare_map_axis(
    axis: Any,
    data: SiteFigureData,
    view: tuple[slice, slice],
    coastline: CoastlineGeometry | None,
    *,
    max_ticks: int | None = None,
    draw_buffer: bool = True,
) -> Any | None:
    """設定經緯度軸標籤、等比例（考慮 cos 緯度）、海岸線疊圖、近岸緩衝帶與站點錨點（只有標記，沒有半徑圓）。

    回傳近岸緩衝帶的圖例 handle（沒有畫時為 ``None``），由呼叫端併入該圖自己的圖例。
    """

    rows, cols = view
    lon = data.lon[cols]
    lat = data.lat[rows]
    xlim, ylim = _view_limits(data, view)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(T["lon"])
    axis.set_ylabel(T["lat"])
    axis.set_aspect(1.0 / _lat_scale(ylim), adjustable="box")
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
    return _draw_coast_buffer(axis, data, view) if draw_buffer else None


def _save(figure: Any, plt: Any, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


# 地圖面板周邊需要的空間（英吋）：左側 y 軸標籤與刻度、色條間距／寬度／刻度與標籤、上方標題、下方 x 軸標籤。
_LEFT_INCH = 0.95
_CBAR_PAD_INCH = 0.10
_CBAR_WIDTH_INCH = 0.16
_CBAR_RIGHT_INCH = 0.85
_TOP_TITLE_INCH = 0.42
_BOTTOM_LABEL_INCH = 0.65
_SUPTITLE_INCH = 0.62
_MIN_FIGURE_WIDTH = 8.8


def render_spatial(output_path: Path, data: SiteFigureData, *, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150) -> Path:
    """站點分析區內的 2×2 空間圖：曲線通過頻率、核心頻率、平均核心 s1、平均未截尾壽命。

    圖面尺寸由地圖資料寬高比決定，四個面板與其色條各自貼齊主圖框，沒有大片空白。
    """

    _, plt = _load_matplotlib()
    rows, cols = _view_extent(data)
    lon, lat = data.lon[cols], data.lat[rows]
    fields = (
        (T["curve_pass_frequency"], "curve_pass_frequency", T["fraction"], "YlOrRd"),
        (T["core_frequency"], "core_frequency", T["fraction"], "viridis"),
        (f"{T['mean_core_s1']} {S1_DAY_LABEL}", "mean_core_s1_day", S1_DAY_LABEL, "magma"),
        (T["mean_lifetime"], "mean_lifetime_hours", T["hour"], "plasma"),
    )
    panel_w, panel_h = _fit_panel(map_data_aspect(data, (rows, cols)), 4.6, 3.9)
    column_pitch = _LEFT_INCH + panel_w + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH
    row_pitch = _TOP_TITLE_INCH + panel_h + _BOTTOM_LABEL_INCH
    footer_h = _footer_height_inch(footer)
    fig_w = max(2.0 * column_pitch, _MIN_FIGURE_WIDTH)
    fig_h = _SUPTITLE_INCH + 2.0 * row_pitch + footer_h
    x_offset = (fig_w - 2.0 * column_pitch) / 2.0
    figure = plt.figure(figsize=(fig_w, fig_h))
    pairs: list[tuple[Any, Any]] = []
    axes_list: list[Any] = []
    buffer_handles: list[Any] = []
    for index, (title, key, label, cmap) in enumerate(fields):
        row, column = divmod(index, 2)
        x0 = x_offset + column * column_pitch + _LEFT_INCH
        y0 = footer_h + (1 - row) * row_pitch + _BOTTOM_LABEL_INCH
        axis = _add_inch_axes(figure, x0, y0, panel_w, panel_h)
        cax = _add_inch_axes(figure, x0 + panel_w + _CBAR_PAD_INCH, y0, _CBAR_WIDTH_INCH, panel_h)
        vmin, vmax = data.colour_limits[key]
        masked = np.ma.masked_invalid(np.asarray(data.spatial_all[key], dtype=np.float64)[rows, cols])
        mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        colorbar = figure.colorbar(mesh, cax=cax)
        colorbar.set_label(label)
        pairs.append((axis, cax))
        axis.set_title(title, fontsize=10)
        buffer_handle = _prepare_map_axis(axis, data, (rows, cols), coastline)
        if buffer_handle is not None:
            buffer_handles.append(buffer_handle)
        axes_list.append(axis)
    axes_list[0].plot([], [], marker="+", color="#37474f", linestyle="none", label=T["anchor"])
    axes_list[0].legend(handles=[*axes_list[0].get_legend_handles_labels()[0], *buffer_handles[:1]], loc="lower right", fontsize=7)
    align_colorbars_to_axes(figure, pairs)
    figure.suptitle(f"{data.site_name_zh} — {T['spatial_title']}", fontsize=13, y=1.0 - 0.12 / fig_h, va="top")
    _place_footer(figure, footer)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_hotspots(
    output_path: Path, data: SiteFigureData, *, hotspot_quantile: float, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150
) -> Path:
    """曲線通過頻率場上標出熱點格點，並以圓點標出核心頻率熱點格點；標題註明熱點面積。

    圖面只顯示濾除小連通塊後的熱點（``SiteFigureData.hotspot_*_mask_filtered``），
    統計欄位仍是未濾除的定義。圖框尺寸由地圖資料寬高比決定，色條與主圖等高。
    """

    matplotlib, plt = _load_matplotlib()
    rows, cols = _view_extent(data)
    lon, lat = data.lon[cols], data.lat[rows]
    vmin, vmax = data.colour_limits["curve_pass_frequency"]

    metrics = data.hotspot_metrics
    percent = f"{100.0 * (1.0 - float(metrics['hotspot_quantile'])):g}"
    area = float(metrics["hotspot_area_km2"])
    threshold = metrics["hotspot_threshold_curve_pass_frequency"]
    threshold_text = "n/a" if threshold is None else f"{float(threshold):.3g}"
    extra = (
        f"{T['footer_hotspot_area']} {area:,.1f} km²  {T['footer_hotspot_cells']} {int(metrics['hotspot_cell_count'])}"
        f"  {T['footer_threshold']} {threshold_text}  {T['footer_min_cells']} {int(data.hotspot_min_cells)}"
    )
    full_footer = footer + "\n" + extra
    title = (
        f"{data.site_name_zh} — {T['hotspot_title']}\n"
        f"{T['hotspot_top']} {percent}%，{T['hotspot_area']} {area:,.0f} km²"
    )

    panel_w, panel_h = _fit_panel(map_data_aspect(data, (rows, cols)), 6.6, 5.6)
    content_w = _LEFT_INCH + panel_w + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH
    footer_h = _footer_height_inch(full_footer)
    title_h = 0.62
    fig_w = max(content_w, _MIN_FIGURE_WIDTH)
    fig_h = 0.1 + title_h + panel_h + _BOTTOM_LABEL_INCH + footer_h
    x0 = (fig_w - content_w) / 2.0 + _LEFT_INCH
    y0 = footer_h + _BOTTOM_LABEL_INCH
    figure = plt.figure(figsize=(fig_w, fig_h))
    axis = _add_inch_axes(figure, x0, y0, panel_w, panel_h)
    cax = _add_inch_axes(figure, x0 + panel_w + _CBAR_PAD_INCH, y0, _CBAR_WIDTH_INCH, panel_h)
    field = np.ma.masked_invalid(np.asarray(data.spatial_all["curve_pass_frequency"], dtype=np.float64)[rows, cols])
    mesh = axis.pcolormesh(lon, lat, field, shading="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    colorbar = figure.colorbar(mesh, cax=cax)
    colorbar.set_label(f"{T['curve_pass_frequency']} ({T['fraction']})")

    curve_source = data.hotspot_curve_mask if data.hotspot_curve_mask_filtered is None else data.hotspot_curve_mask_filtered
    core_source = data.hotspot_core_mask if data.hotspot_core_mask_filtered is None else data.hotspot_core_mask_filtered
    hot = np.asarray(curve_source, dtype=bool)[rows, cols]
    handles = []
    if hot.any():
        overlay = np.ma.masked_where(~hot, np.ones(hot.shape))
        axis.pcolormesh(
            lon, lat, overlay, shading="auto", cmap=matplotlib.colors.ListedColormap(["#e63946"]), alpha=0.55, zorder=3
        )
        handles.append(matplotlib.patches.Patch(facecolor="#e63946", alpha=0.55, label=T["hotspot_curve_legend"]))
    core_hot = np.asarray(core_source, dtype=bool)[rows, cols]
    if core_hot.any():
        core_rows, core_cols = np.nonzero(core_hot)
        cell_points = panel_w * 72.0 / max(lon.size, 1)
        marker_size = float(np.clip(0.42 * cell_points, 2.0, 9.0))
        axis.plot(
            lon[core_cols], lat[core_rows], linestyle="none", marker="o", markersize=marker_size, markerfacecolor="#111111",
            markeredgecolor="white", markeredgewidth=0.4, zorder=5,
        )
        handles.append(
            matplotlib.lines.Line2D(
                [], [], linestyle="none", marker="o", markersize=5, markerfacecolor="#111111", markeredgecolor="white", label=T["hotspot_core_legend"]
            )
        )
    buffer_handle = _prepare_map_axis(axis, data, (rows, cols), coastline, max_ticks=5)
    if buffer_handle is not None:
        handles.append(buffer_handle)
    if handles:
        axis.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)
    axis.set_title(title, fontsize=11)
    align_colorbars_to_axes(figure, ((axis, cax),))
    _place_footer(figure, full_footer)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_seasonal(output_path: Path, data: SiteFigureData, *, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150) -> Path:
    """DJF/MAM/JJA/SON 曲線通過頻率小圖（站內共用色階），加季節強核心出現率與平均通過格點比例（含 CI）。

    四張地圖依資料寬高比取等大的面板，水平色條橫跨四張地圖且貼在其下方。
    """

    _, plt = _load_matplotlib()
    rows, cols = _view_extent(data)
    lon, lat = data.lon[cols], data.lat[rows]
    vmin, vmax = data.colour_limits["curve_pass_frequency"]
    panel_w, panel_h = _fit_panel(map_data_aspect(data, (rows, cols)), 3.1, 3.7)
    left_first = _LEFT_INCH
    pitch = panel_w + 0.62
    map_span = panel_w + 3.0 * pitch
    content_w = left_first + map_span + 0.35
    bar_title_h, bar_h, bar_label_h = 0.42, 2.5, 0.45
    cbar_block_h = 0.25 + 0.16 + 0.55  # 與 x 軸標籤的間距、色條、刻度＋標籤
    footer_h = _footer_height_inch(footer)
    fig_w = max(content_w, 11.5)
    fig_h = _SUPTITLE_INCH + _TOP_TITLE_INCH + panel_h + _BOTTOM_LABEL_INCH + cbar_block_h + bar_title_h + bar_h + bar_label_h + footer_h
    x_offset = (fig_w - content_w) / 2.0
    figure = plt.figure(figsize=(fig_w, fig_h))

    map_bottom = fig_h - _SUPTITLE_INCH - _TOP_TITLE_INCH - panel_h
    map_axes = []
    mesh = None
    for index, season in enumerate(SEASONS):
        axis = _add_inch_axes(figure, x_offset + left_first + index * pitch, map_bottom, panel_w, panel_h)
        masked = np.ma.masked_invalid(np.asarray(data.spatial_seasons[season]["curve_pass_frequency"], dtype=np.float64)[rows, cols])
        mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
        axis.set_title(T[f"season_{season}"], fontsize=10)
        _prepare_map_axis(axis, data, (rows, cols), coastline, max_ticks=3, draw_buffer=False)
        if index > 0:
            axis.set_ylabel("")
        map_axes.append(axis)
    if mesh is not None:
        cax = _add_inch_axes(figure, x_offset + left_first, map_bottom - _BOTTOM_LABEL_INCH - 0.25 - 0.16, map_span, 0.16)
        colorbar = figure.colorbar(mesh, cax=cax, orientation="horizontal")
        colorbar.set_label(f"{T['curve_pass_frequency']} ({T['fraction']})")

    bars_bottom = footer_h + bar_label_h
    half_w = (content_w - left_first - 0.35 - 0.9) / 2.0
    x = np.arange(len(SEASONS))
    for column, (key, title) in enumerate(
        (("strong_core_occurrence_rate", T["strong_core_occurrence"]), ("mean_pass_cell_fraction", T["mean_pass_fraction"]))
    ):
        axis = _add_inch_axes(figure, x_offset + left_first + column * (half_w + 0.9), bars_bottom, half_w, bar_h)
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
    figure.suptitle(f"{data.site_name_zh} — {T['seasonal_title']}", fontsize=13, y=1.0 - 0.12 / fig_h, va="top")
    _place_footer(figure, footer)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_lifetime(output_path: Path, data: SiteFigureData, *, footer: str, dpi: int = 150) -> Path:
    """壽命 CCDF（未截尾實線、含截尾虛線），以及吸引強度隨正規化年齡的演化（Kunz et al. 2024 Fig. 5b 風格）。"""

    _, plt = _load_matplotlib()
    n_table = sum(1 for t in data.lifetime_thresholds_hours if float(t) > 0)
    table_inch = 0.19 * n_table + 0.1 if n_table else 0.0
    fig_w, fig_h = 13.0, 5.6 + table_inch
    figure, (left, right) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    figure.subplots_adjust(
        left=0.07, right=0.97, bottom=(_footer_height_inch(footer) + 0.75 + table_inch) / fig_h, top=1.0 - 1.05 / fig_h, wspace=0.25
    )

    thresholds = [float(t) for t in data.lifetime_thresholds_hours if float(t) > 0]
    daily = float(data.nominal_step_hours) >= 24.0 or (bool(thresholds) and all(abs(t / 24.0 - round(t / 24.0)) < 1e-9 for t in thresholds))
    unit_h = 24.0 if daily else 1.0
    unit_name = T["day_unit"] if daily else "h"

    def _positive(values: Sequence[float]) -> list[float]:
        return [float(v) for v in values if float(v) > 0]

    x_u, y_u = _ccdf(_positive(data.lifetimes_uncensored))
    x_a, y_a = _ccdf(_positive(data.lifetimes_all))
    x_u, x_a = x_u / unit_h, x_a / unit_h
    null_sets = [_positive(values) for values in data.null_lifetimes]
    grid, null_mean, null_low, null_high = null_ccdf_band(null_sets, x_u * unit_h)
    grid = grid / unit_h
    has_null = grid.size > 0
    if x_u.size:
        left.step(x_u, y_u, where="post", color="#1b4965", linestyle="-", label=f"{T['uncensored']} (n={x_u.size})")
    if x_a.size:
        left.step(x_a, y_a, where="post", color="#1b4965", linestyle="--", alpha=0.7, label=f"{T['all_tracks']} (n={x_a.size})")
    if has_null:
        n_seeds = sum(1 for values in data.null_lifetimes if len(values))
        positive = null_high[null_high > 0]
        floor = 0.5 * float(positive.min()) if positive.size else 1.0e-3
        left.fill_between(
            grid, np.maximum(null_low, floor), np.maximum(null_high, floor), step="post", color="#adb5bd", alpha=0.35, linewidth=0,
            label=T["null_band_legend"],
        )
        left.step(
            grid, np.where(null_mean > 0, null_mean, np.nan), where="post", color="#6c757d", linestyle="--", linewidth=1.6,
            label=f"{T['null_ccdf_legend']}（{n_seeds} seeds）",
        )
    table_lines: list[str] = []
    if x_u.size or x_a.size or has_null:
        left.set_yscale("log")
        left.legend(fontsize=8, loc="upper right")
    else:
        left.text(0.5, 0.5, T["no_data"], ha="center", va="center", transform=left.transAxes)
    if x_u.size or x_a.size:
        all_record = data.records.get("all") or {}
        lifetimes_u = np.asarray(_positive(data.lifetimes_uncensored), dtype=np.float64)
        for threshold in thresholds:
            key = _hours_label(threshold)
            left.axvline(threshold / unit_h, color="#6c757d", linestyle=":", linewidth=1.0, zorder=1)
            actual = all_record.get(f"track_frac_uncensored_ge_{key}h")
            if actual is None and lifetimes_u.size:
                actual = float(np.mean(lifetimes_u >= threshold))
            null_frac = all_record.get(f"null_track_frac_uncensored_ge_{key}h") if has_null else None
            if null_frac is None and has_null:
                fracs = [float(np.mean(np.asarray(v) >= threshold)) for v in null_sets if len(v)]
                null_frac = float(np.mean(fracs)) if fracs else None
            excess = all_record.get(f"excess_track_frac_uncensored_ge_{key}h") if has_null else None
            if excess is None and actual is not None and null_frac is not None:
                excess = float(actual) - float(null_frac)
            sep = T["table_sep"]
            line = f"≥{_hours_label(threshold / unit_h)} {unit_name}"
            if actual is not None:
                line += f"：{T['actual_short']} {float(actual):.2f}"
                if null_frac is not None and excess is not None:
                    line += f"{sep}{T['random_short']} {float(null_frac):.2f}{sep}{T['excess']} {float(excess):+.2f}"
            table_lines.append(line)
        # 對數 x 軸：以門檻與常用時間刻度為刻度，避免小門檻在線性軸上擠在一起。
        lo_candidates = [v for v in (x_u[:1].tolist() + x_a[:1].tolist() + [t / unit_h for t in thresholds]) if v > 0]
        hi_candidates = [v for v in (x_u[-1:].tolist() + x_a[-1:].tolist() + [t / unit_h for t in thresholds]) if v > 0]
        lo, hi = min(lo_candidates), max(hi_candidates)
        if hi <= lo:
            hi = lo * 2.0
        left.set_xscale("log")
        left.set_xlim(lo / 1.25, hi * 1.25)
        base = (1, 2, 3, 5, 7, 10, 14, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000) if daily else (1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1440, 2160, 4320)
        ticks = [t / unit_h for t in thresholds]
        for cand in base:
            if lo / 1.25 <= cand <= hi * 1.25 and all(abs(np.log(cand / t)) > np.log(1.8) for t in ticks):
                ticks.append(float(cand))
        ticks = sorted(set(ticks))
        left.set_xticks(ticks)
        left.set_xticklabels([_hours_label(t) for t in ticks])
        left.xaxis.set_minor_locator(matplotlib_ticker_null())
    left.set_xlabel(T["lifetime_days"] if daily else T["lifetime_hours"])
    left.set_ylabel(T["ccdf_ylabel"])
    left.set_title(
        f"{T['ccdf_title']}\n{T['lifetime_refers_to']}：{data.lifetime_scope_text}" if data.lifetime_scope_text else T["ccdf_title"],
        fontsize=10,
    )
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
    right.set_title(
        f"{T['evolution_title']}\n{T['evolution_subtitle']} ≥ {_hours_label(data.lifetime_evolution_min_hours)} {T['hour']}", fontsize=10
    )
    right.grid(alpha=0.25)

    figure.suptitle(f"{data.site_name_zh} — {T['lifetime_title']}", fontsize=13, y=1.0 - 0.12 / fig_h, va="top")
    if table_lines:
        # 門檻數據表：置於左面板 x 軸標籤下方（figure 座標，預留專用高度），不與曲線、圖例或刻度重疊。
        figure.text(
            0.07, (_footer_height_inch(footer) + 0.1) / fig_h, "\n".join(table_lines), ha="left", va="bottom", fontsize=8,
            color="#343a40", linespacing=1.45,
        )
    _place_footer(figure, footer)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_coastal_diagnostic(output_path: Path, data: SiteFigureData, *, footer: str, dpi: int = 150) -> Path:
    """近岸診斷：(a) 核心距岸距離分布（全部／強核心／長壽命 track 核心）；(b) 近岸型與近岸外的壽命 CCDF 與各自的隨機對照。

    (a) 三組各自正規化為比例（總和 1），因此可比較「分布形狀」而非數量；虛線為近岸緩衝帶外緣。
    (b) 近岸型＝track 平均核心距岸小於緩衝帶，近岸外＝不小於；各配對其時間置換 null（各 seed 平均與最小–最大帶）。
    讀者可由此判斷長壽命結構是否集中在海岸帶、去除海岸帶後是否仍超出隨機對照。
    """

    _, plt = _load_matplotlib()
    diag = data.coastal
    assert diag is not None
    fig_w, fig_h = 13.0, 5.8
    table_lines = [T["coast_type_note"]]
    figure, (left, right) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    figure.subplots_adjust(left=0.07, right=0.97, bottom=(_footer_height_inch(footer) + 0.95) / fig_h, top=1.0 - 1.05 / fig_h, wspace=0.25)

    edges = np.asarray(diag.hist_edges_km, dtype=np.float64)
    groups = (
        (diag.hist_all, T["coast_hist_all"], "#1b4965", "-"),
        (diag.hist_strong, T["coast_hist_strong"], "#c1121f", "-"),
        (diag.hist_long, T["coast_hist_long"], "#2a9d8f", "-"),
    )
    any_hist = False
    for counts, label, colour, style in groups:
        counts = np.asarray(counts, dtype=np.float64)
        total = float(counts.sum())
        if total <= 0.0:
            continue
        any_hist = True
        left.step(edges, np.append(counts / total, counts[-1] / total), where="post", color=colour, linestyle=style, linewidth=1.6, label=f"{label} (n={int(total)})")
    left.axvline(float(diag.buffer_km), color=COAST_BUFFER_COLOUR, linestyle="--", linewidth=1.1, label=coast_buffer_label(data))
    if any_hist:
        left.legend(fontsize=8, loc="upper right")
    else:
        left.text(0.5, 0.5, T["no_data"], ha="center", va="center", transform=left.transAxes)
    left.set_xlim(float(edges[0]), float(edges[-1]))
    left.set_xlabel(T["coast_dist_xlabel"])
    left.set_ylabel(T["coast_hist_ylabel"])
    left.set_title(T["coast_hist_title"], fontsize=10)
    left.grid(alpha=0.25)

    daily = float(diag.nominal_step_hours) >= 24.0
    unit_h = 24.0 if daily else 1.0
    drawn_any = False
    curves = (
        (diag.lifetimes_offshore, diag.null_offshore, "#1b4965", T["coast_offshore_real"], T["coast_offshore_null"], "-"),
        (diag.lifetimes_coastal, diag.null_coastal, "#e76f51", T["coast_coastal_real"], T["coast_coastal_null"], "-"),
    )
    for lifetimes, null_sets, colour, real_label, null_label, style in curves:
        positive = [float(v) for v in lifetimes if float(v) > 0]
        x_values, survival = _ccdf(positive)
        null_positive = [[float(v) for v in values if float(v) > 0] for values in null_sets]
        grid, null_mean, null_low, null_high = null_ccdf_band(null_positive, x_values)
        if x_values.size:
            drawn_any = True
            right.step(x_values / unit_h, survival, where="post", color=colour, linestyle=style, linewidth=1.8, label=f"{real_label} (n={x_values.size})")
        if grid.size:
            drawn_any = True
            floor_candidates = null_high[null_high > 0]
            floor = 0.5 * float(floor_candidates.min()) if floor_candidates.size else 1.0e-3
            right.fill_between(grid / unit_h, np.maximum(null_low, floor), np.maximum(null_high, floor), step="post", color=colour, alpha=0.15, linewidth=0)
            right.step(grid / unit_h, np.where(null_mean > 0, null_mean, np.nan), where="post", color=colour, linestyle="--", linewidth=1.4, label=null_label)
    if drawn_any:
        right.set_yscale("log")
        right.set_xscale("log")
        right.legend(fontsize=8, loc="upper right")
    else:
        right.text(0.5, 0.5, T["coast_no_tracks"], ha="center", va="center", transform=right.transAxes)
    right.set_xlabel(T["lifetime_days"] if daily else T["lifetime_hours"])
    right.set_ylabel(T["ccdf_ylabel"])
    right.set_title(T["coast_ccdf_title"], fontsize=10)
    right.grid(alpha=0.25)

    figure.suptitle(f"{data.site_name_zh} — {T['coastal_diag_title']}", fontsize=13, y=1.0 - 0.12 / fig_h, va="top")
    figure.text(0.07, (_footer_height_inch(footer) + 0.12) / fig_h, "\n".join(table_lines), ha="left", va="bottom", fontsize=8, color="#343a40")
    _place_footer(figure, footer)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_timeseries(output_path: Path, data: SiteFigureData, *, footer: str, dpi: int = 150) -> Path:
    """逐日含強核心的有效快照比例與平均通過格點比例；30 日滑動平均，資料缺口以斷線呈現。"""

    _, plt = _load_matplotlib()
    calendar, strong, passed = daily_series(data.day_stats)
    fig_w, fig_h = 12.5, 7.6
    figure, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)
    figure.subplots_adjust(
        left=0.08, right=0.97, bottom=(_footer_height_inch(footer) + 0.75) / fig_h, top=1.0 - 0.75 / fig_h, hspace=0.18
    )
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
    figure.suptitle(f"{data.site_name_zh} — {T['timeseries_title']}", fontsize=13, y=1.0 - 0.12 / fig_h, va="top")
    _place_footer(figure, footer)
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
        paths = [f"figures/{data.site_id}/{stem}_{kind}.png" for kind in ("spatial", "hotspots", "seasonal", "lifetime", "timeseries")]
        if data.coastal is not None:
            render_coastal_diagnostic(site_dir / f"{stem}_coastal_diagnostic.png", data, footer=footer, dpi=dpi)
            paths.append(f"figures/{data.site_id}/{stem}_coastal_diagnostic.png")
    return paths


__all__ = [
    "CoastalDiagnosticData",
    "SiteFigureData",
    "coast_buffer_label",
    "daily_series",
    "render_site_figures",
    "running_mean_with_gaps",
]
