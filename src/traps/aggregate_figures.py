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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .coastline import CoastlineGeometry, draw_geographic_coastline, draw_land_fill
from .colorbar_layout import align_colorbars_to_axes
from . import concise_style
from . import figure_standard as std
from .report_font import FIGURE_TEXT as T
from .report_font import CJKFontSelection, cjk_font_context
from .vector_figures import VECTOR_REPORT_WIDTH_INCH, save_figure, vector_report_enabled

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
    cap_km: float | None = None  # 距離場飽和上限；None 視為直方圖最後邊界。舊 shard 為 30 km
    detection_floor_km: float | None = None  # 站內合格核心的最小距岸距離（檢出下限）


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
    partner: Mapping[str, Any] | None = None
    """同 flow domain 的另一個研究站（A 區的貢寮／龜山島互為搭檔）：``site_id``、``site_name_zh``、``site_mask``、
    ``curve_pass_frequency``（該站站內場）、``anchor_lonlat``。簡要報告圖組據此在整個 domain 上畫出連續的場，並淡化搭檔的統計範圍；
    ``None`` 表示該 domain 只有一個站。"""
    valid_sea_mask: np.ndarray | None = None
    """整個 flow domain 的有效海域（分析海域：可計算、非陸地非遮罩的格點，與 lon/lat 同形）；分析範圍外框只畫站內與站外有效海域之間的邊。
    ``None`` 時以站內遮罩與搭檔站遮罩的聯集代替（同 domain 各站遮罩互斥且聯集＝分析海域）。"""


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
    """建立既有 aggregate footer metadata 字串，供相容的 renderer 呼叫鏈保留。

    字串仍包含期間、UTC、param_id、有效快照數、科學狀態、來源與字型等 provenance
    欄位，但目前由 :func:`_place_footer` 略過繪製；正式 provenance 以 manifest／報告
    保存，避免成果圖底部再出現描述性文字。
    """

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


FOOTER_LINE_INCH = 0.0
"""保留舊版註腳版面常數名稱；目前成果圖不為註腳預留高度。"""
FOOTER_BASE_INCH = 0.0
"""保留舊版註腳基礎高度介面；來源與分析 provenance 由 manifest／報告保存。"""


def matplotlib_ticker_null() -> Any:
    from matplotlib.ticker import NullLocator

    return NullLocator()


def _hours_label(hours: float) -> str:
    """把小時數寫成不帶多餘小數的文字（``12`` 或 ``1.5``）。"""

    return f"{float(hours):g}"


def _footer_height_inch(footer: str) -> float:
    """回傳共用 footer 介面的保留高度，但目前固定為零。

    ``footer`` 仍由各 renderer 接收，以維持 :mod:`example_figures` 與既有呼叫端的
    相容性；研究期間、參數、有效快照數、字型與資料來源則由 manifest／報告保存，
    不再把描述性文字畫在 r1_hourly、d1_daily25h 等成果圖底部。固定回傳零也讓所有
    基本圖使用同一個介面時，自動移除舊版註腳造成的下方空白。
    """

    del footer  # 保留參數是為了相容既有 renderer 與 example_figures 的呼叫介面。
    return 0.0


def _place_footer(figure: Any, footer: str) -> None:
    """保留 footer 放置介面，但不再繪製底部描述文字。

    呼叫端仍可傳入舊版 footer 字串而不需改動；``figure`` 與 ``footer`` 都刻意不
    使用，因為 provenance 已寫入 manifest／報告，成果圖下方只保留圖例、座標與其他
    圖內資訊。此 no-op 也避免任何隱藏的 figure-level Text 物件改變版面或字型 gate。
    """

    del figure, footer


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
    """近岸緩衝帶圖例文字，例如「近岸 10 km 帶」。"""

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


ANCHOR_MARGIN_DEG = 0.03
"""簡要報告模式：站點錨點（星號）四周要留在範圍內的最小距離（度）。"""


ANALYSIS_OUTLINE_COLOUR = "#4d4d4d"
ANALYSIS_OUTLINE_WIDTH_PT = 0.7
ANALYSIS_OUTLINE_LABEL = "研究地點統計範圍"
ANALYSIS_OUTLINE_FRAME_TOLERANCE_DEG = 1.0e-6
"""分析範圍外框的邊與圖框（軸範圍）重合的判定容差（度）。"""


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    """格點中心 → 格緣座標（與 ``pcolormesh(shading="auto")`` 的 nearest 格緣相同：相鄰中心的中點，兩端依鄰格外推）。"""

    centres = np.asarray(centres, dtype=np.float64)
    if centres.size == 1:
        return np.array([centres[0] - 0.005, centres[0] + 0.005])
    middle = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate([[centres[0] - (middle[0] - centres[0])], middle, [centres[-1] + (centres[-1] - middle[-1])]])


def analysis_outline_paths(
    lon: np.ndarray,
    lat: np.ndarray,
    site_mask: np.ndarray,
    valid_sea_mask: np.ndarray,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    tolerance: float = ANALYSIS_OUTLINE_FRAME_TOLERANCE_DEG,
) -> list[np.ndarray]:
    """分析範圍外框：只取「站內格點」與「站外但仍是有效海域的格點」之間的格緣，串成折線（每條為 N×2 的 lon, lat）。

    以格點邊為單位判斷，不用等值線、不用多邊形：
    * 只保留站內格點（``site_mask``）與有效海域格點（``valid_sea_mask`` 且不在站內）相鄰的邊；
      另一側是陸地、遮罩（fail-closed）或無效格點、或在格網之外的邊一律不畫（貼著海岸與包圍島嶼的邊）。
    * 給定 ``xlim``／``ylim``（圖框）時，與圖框重合的邊（容差 ``tolerance``）與完全落在圖框外的邊不畫。
    連續的邊串成折線，端點相接處沒有缺口。沒有符合的邊時回傳空串列。
    """

    site = np.asarray(site_mask, dtype=bool)
    other = np.asarray(valid_sea_mask, dtype=bool) & ~site
    if site.shape != other.shape or site.shape != (np.asarray(lat).size, np.asarray(lon).size):
        raise ValueError("lon／lat 與遮罩的形狀不一致")
    x_edges, y_edges = _cell_edges(lon), _cell_edges(lat)
    rows, cols = site.shape
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []  # 端點為格角索引 (x 索引, y 索引)
    # 垂直邊：列 i 的格 j 與格 j+1 之間，位於 x 格緣 j+1，y 由格緣 i 到 i+1
    pair = (site[:, :-1] & other[:, 1:]) | (other[:, :-1] & site[:, 1:])
    for i, j in zip(*np.nonzero(pair), strict=True):
        edges.append(((int(j) + 1, int(i)), (int(j) + 1, int(i) + 1)))
    # 水平邊：格 (i, j) 與 (i+1, j) 之間，位於 y 格緣 i+1，x 由格緣 j 到 j+1
    pair = (site[:-1, :] & other[1:, :]) | (other[:-1, :] & site[1:, :])
    for i, j in zip(*np.nonzero(pair), strict=True):
        edges.append(((int(j), int(i) + 1), (int(j) + 1, int(i) + 1)))

    def keep(edge: tuple[tuple[int, int], tuple[int, int]]) -> bool:
        (xa, ya), (xb, yb) = edge
        x0, x1 = sorted((float(x_edges[xa]), float(x_edges[xb])))
        y0, y1 = sorted((float(y_edges[ya]), float(y_edges[yb])))
        if xlim is not None:
            low, high = sorted((float(xlim[0]), float(xlim[1])))
            if x1 < low - tolerance or x0 > high + tolerance:
                return False
            if xa == xb and (abs(x0 - low) <= tolerance or abs(x0 - high) <= tolerance):
                return False
        if ylim is not None:
            low, high = sorted((float(ylim[0]), float(ylim[1])))
            if y1 < low - tolerance or y0 > high + tolerance:
                return False
            if ya == yb and (abs(y0 - low) <= tolerance or abs(y0 - high) <= tolerance):
                return False
        return True

    edges = [edge for edge in edges if keep(edge)]
    if not edges:
        return []
    adjacency: dict[tuple[int, int], list[int]] = {}
    for index, (a, b) in enumerate(edges):
        adjacency.setdefault(a, []).append(index)
        adjacency.setdefault(b, []).append(index)
    used = [False] * len(edges)
    paths: list[np.ndarray] = []

    def walk(start: tuple[int, int]) -> None:
        for first in list(adjacency[start]):
            if used[first]:
                continue
            chain = [start]
            node, index = start, first
            while True:
                used[index] = True
                a, b = edges[index]
                node = b if a == node else a
                chain.append(node)
                following = [k for k in adjacency[node] if not used[k]]
                if len(adjacency[node]) != 2 or not following:
                    if len(adjacency[node]) > 2 and following:
                        index = following[0]
                        continue
                    break
                index = following[0]
            paths.append(np.array([[x_edges[x], y_edges[y]] for x, y in chain], dtype=np.float64))

    ends = [node for node, incident in adjacency.items() if len(incident) != 2]
    for node in sorted(ends):
        walk(node)
    for node in sorted(adjacency):
        walk(node)  # 其餘為封閉環
    return paths


def valid_sea_of(data: "SiteFigureData") -> np.ndarray:
    """分析範圍外框用的「站外有效海域」基準：可計算、非陸地非遮罩，且不屬於同 domain 搭檔站的格點，加上站內格點。

    2026-10-21：搭檔站（A 區的貢寮與龜山島互為搭檔）的統計範圍不算站外有效海域，兩站之間的分界已用橘色「統計歸屬分界」線表示，
    外框不再重複描一次。因此只有與「不屬於任何地點的有效海域」相鄰的邊才會畫。
    ``data.valid_sea_mask`` 給定時取它（server 繪圖為 ``grid_reference.npz`` 的 ``analysis_ocean_mask``，已含各站範圍）；沒有時只有站內遮罩（沒有站外有效海域）。
    """

    mask = np.asarray(data.site_mask, dtype=bool)
    explicit = getattr(data, "valid_sea_mask", None)
    valid = np.asarray(explicit, dtype=bool) | mask if explicit is not None else mask
    partner = getattr(data, "partner", None)
    if partner:
        valid = valid & ~np.asarray(partner["site_mask"], dtype=bool)
    return valid


def draw_analysis_outline(axis: Any, data: SiteFigureData, limits: tuple[tuple[float, float], tuple[float, float]] | None = None) -> Any | None:
    """分析範圍（站內分析區遮罩）的外框：細深灰色線段，只畫「站內與站外有效海域」之間的格緣。

    貼著海岸、包圍島嶼的邊（另一側是陸地、遮罩或無效格點）與和圖框重合的邊都不畫（:func:`analysis_outline_paths`）。
    回傳圖例用的 :class:`Line2D`（標籤 ``分析範圍``）；沒有任何線段時什麼都不畫並回傳 ``None``（圖例不放此項）。
    畫出的邊數記在 ``axis._traps_analysis_outline_segments``（版面紀錄 ``analysis_outline_segments``）。畫完後還原軸範圍，不改任何資料或遮罩。
    """

    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    x_before, y_before = axis.get_xlim(), axis.get_ylim()
    paths = analysis_outline_paths(
        np.asarray(data.lon, dtype=float), np.asarray(data.lat, dtype=float), np.asarray(data.site_mask, dtype=bool), valid_sea_of(data),
        xlim=x_before, ylim=y_before,
    )
    axis._traps_analysis_outline_segments = int(sum(len(path) - 1 for path in paths))  # noqa: SLF001  供 concise_style.concise_checks_of 讀取
    if not paths:
        return None
    axis.add_collection(
        LineCollection(
            paths, colors=[ANALYSIS_OUTLINE_COLOUR], linewidths=ANALYSIS_OUTLINE_WIDTH_PT, linestyles="solid", capstyle="butt", joinstyle="round", zorder=6.5,
        )
    )
    axis.set_xlim(*x_before)
    axis.set_ylim(*y_before)
    return Line2D([], [], color=ANALYSIS_OUTLINE_COLOUR, linewidth=0.9, label=ANALYSIS_OUTLINE_LABEL)


FLOW_BOUNDARY_COLOUR = "#1f3a5f"
"""流場範圍邊界的深藍色（與紫色虛線近岸帶 #6a3d9a、橘色統計歸屬分界 #f4a300、黑色海岸線 #263238 都不同）。"""
FLOW_BOUNDARY_WIDTH_PT = 1.0
FLOW_BOUNDARY_DASHES: tuple[float, float] = (6.5, 2.2)
"""長虛線（線段 6.5 pt、間隔 2.2 pt，線寬 1.0 pt；近岸帶是線寬 0.7 pt 的預設短虛線，線段約 2.6 pt）。"""
FLOW_BOUNDARY_LABEL = "流場範圍邊界"
FLOW_BOUNDARY_ZORDER = 6.6
"""高於海岸線（6.0）、近岸帶與分析範圍外框（6.5），低於統計歸屬分界（7）與地點星號（8）。"""


def flow_domain_extent(data: "SiteFigureData") -> tuple[float, float, float, float]:
    """流場網格的外框 ``(lon_min, lon_max, lat_min, lat_max)``：有效流場格點（規則經緯度網格，整個 flow domain）外緣，即格點中心外推半格（與 ``pcolormesh`` 畫出的場外緣相同）。"""

    lon_edges, lat_edges = _cell_edges(np.asarray(data.lon, dtype=np.float64)), _cell_edges(np.asarray(data.lat, dtype=np.float64))
    return float(lon_edges[0]), float(lon_edges[-1]), float(lat_edges[0]), float(lat_edges[-1])


def draw_flow_domain_boundary(axis: Any, data: "SiteFigureData", limits: Any = None) -> Any:
    """簡要報告：流場範圍邊界——流場網格外框，深藍色長虛線（:data:`FLOW_BOUNDARY_COLOUR`、1.0 pt）；回傳圖例 handle（標籤「流場範圍邊界」）。

    畫法參考 Lagrangian-Ensemble-Backtracking 的「流場範圍開放邊界」（洋紅色虛線）。外框整個落在圖框內時四邊都畫；落在圖框外的部分由軸框裁切。
    流場範圍外的海域維持白色、不填色（只有陸地與海岸線畫滿整張圖框）。不改變軸範圍；畫出的線段數與邊界座標記在 ``axis._traps_flow_boundary``（版面紀錄 ``flow_boundary``）。
    """

    from matplotlib.lines import Line2D

    x_before, y_before = axis.get_xlim(), axis.get_ylim()
    west, east, south, north = flow_domain_extent(data)
    style = {"color": FLOW_BOUNDARY_COLOUR, "linewidth": FLOW_BOUNDARY_WIDTH_PT, "linestyle": (0, FLOW_BOUNDARY_DASHES)}
    axis.plot(
        [west, east, east, west, west], [south, south, north, north, south], zorder=FLOW_BOUNDARY_ZORDER, solid_joinstyle="miter",
        dash_joinstyle="miter", dash_capstyle="butt", scalex=False, scaley=False, **style,
    )
    axis.set_xlim(*x_before)
    axis.set_ylim(*y_before)
    inside = bool(
        min(x_before) <= west and east <= max(x_before) and min(y_before) <= south and north <= max(y_before)
    )
    axis._traps_flow_boundary = {  # noqa: SLF001  供 concise_style.concise_checks_of 讀取
        "label": FLOW_BOUNDARY_LABEL, "colour": FLOW_BOUNDARY_COLOUR, "linewidth_pt": FLOW_BOUNDARY_WIDTH_PT, "dashes_pt": list(FLOW_BOUNDARY_DASHES),
        "segments": 4, "extent": [round(west, 6), round(east, 6), round(south, 6), round(north, 6)], "inside_frame": inside,
    }
    return Line2D([], [], label=FLOW_BOUNDARY_LABEL, **style)


def concise_map_boxes(leg3: bool = False) -> tuple[std.Box, std.Box]:
    """地圖軸框與色條框：簡要報告圖組的大圖用 LBT 長寬比的 :data:`traps.figure_standard.CONCISE_MAP_AXES`／``CONCISE_MAP_CBAR``，其餘用 MAP1。

    ``leg3=True``（v65，聚集熱區與圖1 的三列圖例）回傳下緣上移的 ``CONCISE_LEG3_AXES``／``CONCISE_LEG3_CBAR``（大小與左緣相同）。"""

    if concise_style.concise_enabled():
        return (std.CONCISE_LEG3_AXES, std.CONCISE_LEG3_CBAR) if leg3 else (std.CONCISE_MAP_AXES, std.CONCISE_MAP_CBAR)
    return std.MAP1_AXES, std.MAP1_CBAR


def concise_legend_columns(entries: int) -> int:
    """v65 三列圖例的欄數：項目依欄填入（matplotlib 逐欄分配，前面的欄多一列），5–6 項為 2 欄、7 項為 3 欄；固定三列。"""

    return max(1, -(-int(entries) // std.CONCISE_LEG3_ROWS))


def concise_core_legend_handles(matplotlib: Any, T: Mapping[str, str]) -> list[Any]:
    """聚集熱區圖與圖1 的核心點圖例：黑點（聚集熱區內）與灰點（聚集熱區外）兩項，順序與用詞兩種圖相同。"""

    return [
        matplotlib.lines.Line2D([], [], label=concise_style.term("hotspot_core_legend", T["hotspot_core_legend"]), **concise_style.hotspot_core_legend_kwargs()),
        matplotlib.lines.Line2D([], [], label=concise_style.term("hotspot_core_legend_outside", T["hotspot_core_legend_outside"]), **concise_style.hotspot_core_legend_kwargs(outside=True)),
    ]


PARTNER_VEIL_ALPHA = 0.62
"""同 domain 另一站（A 區的貢寮／龜山島）統計範圍上的白色半透明淡化層的不透明度。"""
PARTNER_VEIL_ZORDER = 4.0
"""淡化層的 zorder：在場與熱點色塊（≤ 3）之上、核心頻率圓點（5）、海岸線（6）與分界線（7）之下。"""


def partner_field(data: SiteFigureData, own: np.ndarray) -> np.ndarray:
    """站內場 ``own`` 在搭檔站的範圍內以搭檔站的場補上（兩站遮罩互斥、聯集＝整個分析海域），得到整個 domain 上連續的場。"""

    partner = getattr(data, "partner", None)
    own = np.asarray(own, dtype=np.float64)
    if not partner:
        return own
    return np.where(np.isfinite(own), own, np.asarray(partner["curve_pass_frequency"], dtype=np.float64))


def draw_partner_overlay(axis: Any, data: SiteFigureData, view: tuple[slice, slice], limits: Any) -> Any | None:
    """簡要報告：A 區站圖上，搭檔站的統計範圍罩一層白色半透明淡化層，畫橘色「統計歸屬分界」並標出搭檔站的 anchor 與站名。

    沒有搭檔站（B、C、D 區）時什麼都不畫並回傳 ``None``；否則回傳分界線的圖例 handle。不改動軸範圍。
    """

    partner = getattr(data, "partner", None)
    if not partner:
        return None
    import matplotlib
    from matplotlib import patheffects
    from matplotlib.lines import Line2D

    from .domain_boundary import BOUNDARY_COLOUR, BOUNDARY_LABEL, boundary_in_analysis_area

    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    other = np.asarray(partner["site_mask"], dtype=bool)[rows, cols]
    x_before, y_before = axis.get_xlim(), axis.get_ylim()
    if other.any():
        axis.pcolormesh(
            lon, lat, np.ma.masked_where(~other, np.ones(other.shape)), shading="auto",
            cmap=matplotlib.colors.ListedColormap(["#ffffff"]), vmin=0.0, vmax=1.0, alpha=PARTNER_VEIL_ALPHA, zorder=PARTNER_VEIL_ZORDER,
        )
    union = np.asarray(data.site_mask, dtype=bool) | np.asarray(partner["site_mask"], dtype=bool)
    b_lon, b_lat = boundary_in_analysis_area(data.anchor_lonlat, partner["anchor_lonlat"], np.asarray(data.lon), np.asarray(data.lat), union)
    stroke = [patheffects.withStroke(linewidth=3.0, foreground="#222222")]
    axis.plot(b_lon, b_lat, color=BOUNDARY_COLOUR, linewidth=1.6, linestyle="-", zorder=7, path_effects=stroke)
    a_lon, a_lat = partner["anchor_lonlat"]
    concise_style.draw_site_star(axis, a_lon, a_lat)  # 搭檔站的地點位置也用同樣的星號（與 LBT 的受體符號一致）
    if concise_style.show_site_names():  # 簡要報告不印地點名稱，只留星號
        axis.annotate(
            concise_style.display_site_name(str(partner["site_id"]), str(partner["site_name_zh"])), (a_lon, a_lat), xytext=(6, 4), textcoords="offset points",
            fontsize=8, color="#55646b", zorder=9, path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")],
        )
    axis.set_xlim(*x_before)
    axis.set_ylim(*y_before)
    return Line2D([], [], color=BOUNDARY_COLOUR, linewidth=1.6, label=BOUNDARY_LABEL, path_effects=stroke)


LAND_FILL_ZORDER = 0.5
"""陸地填色的 zorder：低於所有資料圖層（pcolormesh 預設 1）與分析範圍外框（6.5），海岸線描邊在 6.0。"""


def standard_map_frame(
    data: SiteFigureData, base_view: tuple[slice, slice], box: std.Box, *, extra_points: Sequence[tuple[float, float]] = (), coastline: Any = None
) -> tuple[tuple[slice, slice], tuple[tuple[float, float], tuple[float, float]]]:
    """標準軸框用的地圖視窗：以 ``base_view`` 的中心為準，把經緯度範圍擴張到軸框長寬比。

    回傳 ``(擴張後的 (row, col) 切片, (xlim, ylim))``。只改顯示範圍；資料陣列、遮罩與色階不變，
    擴張出的區域只畫海岸線、陸地與（若有）真實場值，站內遮罩外的海域維持空白。
    """

    xlim0, ylim0 = _view_limits(data, base_view)
    if concise_style.concise_enabled():
        # 簡要報告圖組（2026-10-22 第二輪，LBT run09 axes_unified 規格）：各地點的地圖範圍直接取 concise_style.FRAME_SPECS["map"]["extents"]
        # （經度 1.8°、緯度 1.2°，經緯度刻度間距都是 0.2°，上下限即刻度）；流場範圍在圖框內，其餘部分只畫陸地與海岸線。
        # 軸框為 std.CONCISE_MAP_AXES（以參考緯度 24.1° 計算 LBT 長寬比，五個地點相同），長寬比誤差記錄於 extent_info.aspect_error（上限 MAP_FRAME_ASPECT_TOLERANCE，±2%）。
        xlim, ylim, frame_info = concise_style.map_frame(data.lon, data.lat)
        error = concise_style.frame_aspect_error(xlim, ylim, box.w, box.h)  # 超過上限時由 gate 標示失敗（不在此中止）
        concise_style.record_extent_info(
            xlim, ylim, {
                "kind": "lbt_specified_extent_map", "aspect_error": error, "tolerance": concise_style.MAP_FRAME_ASPECT_TOLERANCE,
                "grid_bbox": [float(np.min(data.lon)), float(np.max(data.lon)), float(np.min(data.lat)), float(np.max(data.lat))],
                "override": None,
                "frame": frame_info,
            },
        )
        # 空白（分析範圍外、也不是陸地的海）比例只作記錄（見 extent_coverage）。
        from .extent_coverage import build_coverage

        mask = np.asarray(data.site_mask, dtype=bool)
        if getattr(data, "partner", None):
            mask = mask | np.asarray(data.partner["site_mask"], dtype=bool)
        coverage = build_coverage(data.lon, data.lat, mask, coastline, (xlim[0], xlim[1], ylim[0], ylim[1]), margin=0.0)
        concise_style.update_extent_info(xlim, ylim, fractions=coverage.fractions(xlim, ylim))
    else:
        xlim, ylim = std.fit_limits_to_box(xlim0, ylim0, box.w, box.h)
    return std.limits_slices(data.lon, data.lat, xlim, ylim), (xlim, ylim)


def style_map_axis(axis: Any) -> None:
    """地圖座標軸的標準字級：刻度與座標軸標籤皆 8 pt。"""

    axis.tick_params(labelsize=std.TEXT_FONT_PT, pad=1)
    axis.xaxis.label.set_size(std.TEXT_FONT_PT)
    axis.yaxis.label.set_size(std.TEXT_FONT_PT)


def style_colorbar(colorbar: Any, label: str | None = None) -> None:
    """色條的標準字級（刻度與標籤 8 pt）。"""

    if label is not None:
        colorbar.set_label(label, fontsize=std.TEXT_FONT_PT, labelpad=2)
    colorbar.ax.tick_params(labelsize=std.TEXT_FONT_PT, pad=1)


def _prepare_map_axis(
    axis: Any,
    data: SiteFigureData,
    view: tuple[slice, slice],
    coastline: CoastlineGeometry | None,
    *,
    max_ticks: int | None = None,
    draw_buffer: bool = True,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    lbt_style: bool = False,
) -> Any | None:
    """設定經緯度軸標籤、等比例（考慮 cos 緯度）、海岸線疊圖、近岸緩衝帶與站點錨點（只有標記，沒有半徑圓）。

    回傳近岸緩衝帶的圖例 handle（沒有畫時為 ``None``），由呼叫端併入該圖自己的圖例。
    給定 ``limits``（標準軸框模式）時，軸範圍、海岸線 bbox 與錨點判斷都採用該範圍，
    軸框大小固定，改以 ``adjustable="datalim"`` 保證長寬比（範圍已事先擴張到與軸框同比例）。
    """

    rows, cols = view
    lon = data.lon[cols]
    lat = data.lat[rows]
    if limits is None:
        xlim, ylim = _view_limits(data, view)
    else:
        xlim, ylim = limits
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(T["lon"])
    axis.set_ylabel(T["lat"])
    if limits is not None and concise_style.concise_enabled():
        # 範圍已貼齊刻度：aspect 取「軸框高寬比對應的精確值」，datalim 不再改動兩端（長寬比誤差由 snap_limits_to_ticks 記錄）。
        position = axis.get_position()
        figure_w, figure_h = axis.figure.get_size_inches()
        axis.set_aspect(
            concise_style.exact_aspect(xlim, ylim, position.width * figure_w, position.height * figure_h), adjustable="datalim"
        )
    else:
        axis.set_aspect(1.0 / _lat_scale(ylim), adjustable="box" if limits is None else "datalim")
    if not lbt_style:  # 簡要報告大圖對齊 LBT 底圖：不畫格線
        axis.grid(alpha=0.2, linewidth=0.5)
    axis.ticklabel_format(useOffset=False, style="plain")
    if max_ticks is not None:
        from matplotlib.ticker import MaxNLocator

        axis.xaxis.set_major_locator(MaxNLocator(max_ticks))
        axis.yaxis.set_major_locator(MaxNLocator(max_ticks + 1))
    if limits is None:
        bbox = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    else:
        bbox = (float(xlim[0]), float(ylim[0]), float(xlim[1]), float(ylim[1]))
    if coastline is not None:
        if concise_style.concise_enabled():
            # 簡要報告圖組：陸地以淺中性色填滿（在資料與分析範圍之下），海岸線描邊畫在其上；分析範圍外的海維持白色。
            draw_land_fill(axis, coastline, bbox=bbox, zorder=LAND_FILL_ZORDER)
        if lbt_style:  # 簡要報告大圖對齊 LBT 底圖：黑色海岸線，線寬為 LBT 0.5 pt 乘印出縮放（std.LBT_COAST_*）
            draw_geographic_coastline(axis, coastline, bbox=bbox, color=std.LBT_COAST_COLOUR, linewidth=std.LBT_COAST_WIDTH_PT, zorder=6.0)
        else:
            draw_geographic_coastline(axis, coastline, bbox=bbox, color="#263238", linewidth=0.6, zorder=6.0)
    anchor_lon, anchor_lat = data.anchor_lonlat
    if bbox[0] <= anchor_lon <= bbox[2] and bbox[1] <= anchor_lat <= bbox[3]:
        if concise_style.concise_enabled():
            concise_style.draw_site_star(axis, anchor_lon, anchor_lat)  # 簡要報告：黃色星號（與 LBT 的受體符號一致，高於熱區色塊、黑點與海岸線）
        else:
            axis.plot(anchor_lon, anchor_lat, marker="+", color="#37474f", markersize=8, markeredgewidth=1.4, linestyle="none", zorder=8)
    if limits is not None:
        # plot／pcolormesh 在 datalim 模式不應再改動範圍；明確還原，並套用標準字級（8 pt）。
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        style_map_axis(axis)
        if concise_style.concise_enabled():
            concise_style.style_map_axes(axis, xlim, ylim)
    return _draw_coast_buffer(axis, data, view) if draw_buffer else None


def _save(figure: Any, plt: Any, output_path: Path, dpi: int) -> None:
    """保存一張成果圖；報告模式另外輸出同名 SVG／PDF 向量檔。

    ``save_figure`` 以同一個已完成的 Matplotlib Figure 直接輸出 PNG、SVG、PDF，
    因而不會重新取樣、平滑、補值或改變任何科學資料。環境變數未啟用時維持舊版
    只輸出 PNG 的行為，讓一般分析與既有測試不受影響。
    """

    save_figure(figure, plt, output_path, dpi)


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
    # 平均未截尾壽命只改顯示單位：逐日產品（名目步長 >= 24 h）以「天」顯示（小時 / 24），
    # 逐時產品維持「小時」；儲存的 npz／CSV 與 colour_limits 仍是小時，不被改寫。
    lifetime_in_days = float(data.nominal_step_hours) >= 24.0
    lifetime_scale = 24.0 if lifetime_in_days else 1.0
    fields = (
        (T["curve_pass_frequency"], "curve_pass_frequency", T["fraction"], "YlOrRd"),
        (T["core_frequency"], "core_frequency", T["fraction"], "viridis"),
        (f"{T['mean_core_s1']} {S1_DAY_LABEL}", "mean_core_s1_day", S1_DAY_LABEL, "magma"),
        (T["mean_lifetime"], "mean_lifetime_hours", T["day_unit"] if lifetime_in_days else T["hour"], "plasma"),
    )
    report_mode = vector_report_enabled()
    if report_mode:
        # 標準 multi 畫布（6.3×7.0）：2×2 面板與各自下方的水平色條，位置與大小固定；
        # 地圖範圍以固定軸框擴張經緯度（figure_standard），色階與資料不變。
        fig_w, fig_h = std.WIDTH_INCH, std.MULTI_HEIGHT_INCH
        panel_box = std.Box(0.0, 0.0, std.SPATIAL_PANEL_W, std.SPATIAL_PANEL_H)
        (rows, cols), limits = standard_map_frame(data, (rows, cols), panel_box)
        lon, lat = data.lon[cols], data.lat[rows]
        figure = plt.figure(figsize=(fig_w, fig_h))
        axes_list: list[Any] = []
        buffer_handles: list[Any] = []
        for index, (title, key, label, cmap) in enumerate(fields):
            row, column = divmod(index, 2)
            x0, map_y = std.SPATIAL_X[column], std.SPATIAL_Y[row]
            axis = _add_inch_axes(figure, x0, map_y, std.SPATIAL_PANEL_W, std.SPATIAL_PANEL_H)
            cax = _add_inch_axes(figure, x0, map_y - std.SPATIAL_CBAR_GAP, std.SPATIAL_PANEL_W, std.SPATIAL_CBAR_H)
            vmin, vmax = data.colour_limits[key]
            masked = np.ma.masked_invalid(np.asarray(data.spatial_all[key], dtype=np.float64)[rows, cols])
            if key == "mean_lifetime_hours":
                vmin, vmax, masked = vmin / lifetime_scale, vmax / lifetime_scale, masked / lifetime_scale
            mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            colorbar = figure.colorbar(mesh, cax=cax, orientation="horizontal")
            style_colorbar(colorbar, label)
            axis.set_title(title, fontsize=std.SUBTITLE_FONT_PT, pad=3)
            buffer_handle = _prepare_map_axis(axis, data, (rows, cols), coastline, max_ticks=3, limits=limits)
            style_map_axis(axis)
            if column == 1:
                # 右欄與左欄共用相同的緯度範圍；隱藏重複的 ylabel／刻度文字，避免文字跑進左圖。
                axis.set_ylabel("")
                axis.tick_params(axis="y", labelleft=False)
            if buffer_handle is not None:
                buffer_handles.append(buffer_handle)
            axes_list.append(axis)
        # 站點錨點與近岸緩衝帶只保留為底部圖例，避免圖例遮蓋空間場。
        axes_list[0].plot([], [], marker="+", color="#37474f", linestyle="none", label=T["anchor"])
        legend_handles, legend_labels = axes_list[0].get_legend_handles_labels()
        if buffer_handles:
            legend_handles.append(buffer_handles[0])
            legend_labels.append(coast_buffer_label(data))
        std.legend_below(figure, fig_h, legend_handles, legend_labels, ncol=len(legend_handles))
    else:
        # 一般分析流程維持既有的並列垂直色條版面，確保未啟用 opt-in 時輸出不變。
        panel_w, panel_h = _fit_panel(map_data_aspect(data, (rows, cols)), 4.6, 3.9)
        column_pitch = _LEFT_INCH + panel_w + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH
        row_pitch = _TOP_TITLE_INCH + panel_h + _BOTTOM_LABEL_INCH
        footer_h = _footer_height_inch(footer)
        fig_w = max(2.0 * column_pitch, _MIN_FIGURE_WIDTH)
        fig_h = _SUPTITLE_INCH + 2.0 * row_pitch + footer_h
        x_offset = (fig_w - 2.0 * column_pitch) / 2.0
        figure = plt.figure(figsize=(fig_w, fig_h))
        pairs: list[tuple[Any, Any]] = []
        axes_list = []
        buffer_handles = []
        for index, (title, key, label, cmap) in enumerate(fields):
            row, column = divmod(index, 2)
            x0 = x_offset + column * column_pitch + _LEFT_INCH
            y0 = (1 - row) * row_pitch + _BOTTOM_LABEL_INCH
            axis = _add_inch_axes(figure, x0, y0, panel_w, panel_h)
            cax = _add_inch_axes(figure, x0 + panel_w + _CBAR_PAD_INCH, y0, _CBAR_WIDTH_INCH, panel_h)
            vmin, vmax = data.colour_limits[key]
            masked = np.ma.masked_invalid(np.asarray(data.spatial_all[key], dtype=np.float64)[rows, cols])
            if key == "mean_lifetime_hours":
                vmin, vmax, masked = vmin / lifetime_scale, vmax / lifetime_scale, masked / lifetime_scale
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
        axes_list[0].legend(
            handles=[*axes_list[0].get_legend_handles_labels()[0], *buffer_handles[:1]],
            loc="lower right",
            fontsize=7,
        )
        align_colorbars_to_axes(figure, pairs)
    if report_mode:
        std.draw_title_block(figure, fig_h, f"{data.site_name_zh} — {T['spatial_title']}")
    else:
        figure.suptitle(f"{data.site_name_zh} — {T['spatial_title']}", fontsize=13, y=1.0 - 0.12 / fig_h, va="top")
    _place_footer(figure, footer)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_hotspots(
    output_path: Path, data: SiteFigureData, *, hotspot_quantile: float, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150
) -> Path:
    """曲線通過頻率場上標出熱點格點，並以圓點標出核心頻率熱點格點；標題註明熱點面積。

    正式圖組只顯示濾除小連通塊後的熱點（``SiteFigureData.hotspot_*_mask_filtered``），
    統計欄位仍是未濾除的定義；簡要報告（concise）改畫未濾除遮罩（``hotspot_curve_mask``／``hotspot_core_mask``），
    圖面與統計定義逐格一致。圖框尺寸由地圖資料寬高比決定，色條與主圖等高。
    """

    matplotlib, plt = _load_matplotlib()
    rows, cols = _view_extent(data)
    report_mode = vector_report_enabled()
    limits = None
    if report_mode:
        # 標準畫布（6.3×4.4）與標準軸框：固定軸框、擴張經緯度範圍（見 figure_standard）。
        (rows, cols), limits = standard_map_frame(data, (rows, cols), concise_map_boxes()[0], coastline=coastline)
    lon, lat = data.lon[cols], data.lat[rows]
    vmin, vmax = data.colour_limits["curve_pass_frequency"]
    concise = report_mode and concise_style.concise_enabled()
    colorbar_extend = "neither"
    if concise:
        # 簡要報告圖組：站名顯示覆寫、跨站共用色階（只改顏色對應上下限）、色條端點刻度。
        data = replace(data, site_name_zh=concise_style.display_site_name(data.site_id, data.site_name_zh))
        concise_style.record_limit("hotspots", data.site_id, "curve_pass_frequency", (vmin, vmax))
        shared = concise_style.shared_limits("hotspots", "curve_pass_frequency")
        if shared is not None:
            vmin, vmax = shared
        colorbar_extend = "neither"  # 簡要報告：色條兩端平整，超出共用色階的值以端點色飽和（比例記錄於 clipped_fraction）

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

    if report_mode:
        panel_w = concise_map_boxes()[0].w
        figure, axis, cax = _map1_canvas(plt, with_colorbar=True, leg3=concise)  # v65：簡要報告三列圖例，畫布加高
    else:
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
    own_field = np.asarray(data.spatial_all["curve_pass_frequency"], dtype=np.float64)
    # 簡要報告：整個 domain 上連續的場（A 區在搭檔站範圍內補上搭檔站的場，再由淡化層弱化）
    field = np.ma.masked_invalid((partner_field(data, own_field) if concise else own_field)[rows, cols])
    mesh = axis.pcolormesh(lon, lat, field, shading="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    colorbar = figure.colorbar(mesh, cax=cax, extend=colorbar_extend)
    cpf_label = f"{concise_style.term('curve_pass_frequency', T['curve_pass_frequency'])} ({T['fraction']})"
    if report_mode:
        style_colorbar(colorbar, cpf_label)
        if concise:
            concise_style.style_colorbar_shared(
                colorbar, vmin, vmax, cpf_label, name="curve_pass_frequency", extend=colorbar_extend,
                clipped=concise_style.clipped_fraction(np.asarray(data.spatial_all["curve_pass_frequency"], dtype=np.float64)[data.site_mask.astype(bool)], vmin, vmax),
            )
    else:
        colorbar.set_label(f"{T['curve_pass_frequency']} ({T['fraction']})", fontsize=8, labelpad=2)
        colorbar.ax.tick_params(labelsize=8, pad=1)

    if concise:
        # 簡要報告（2026-10-27 v45）：所見即所得——圖面畫統計定義本身的未濾除遮罩（各地點前 10% 的所有格點與所有核心黑點，不濾除小連通塊）。
        curve_source, core_source = concise_style.hotspot_masks_for_figure(data.hotspot_curve_mask, data.hotspot_core_mask, data.hotspot_curve_mask_filtered, data.hotspot_core_mask_filtered)
    else:
        curve_source = data.hotspot_curve_mask if data.hotspot_curve_mask_filtered is None else data.hotspot_curve_mask_filtered
        core_source = data.hotspot_core_mask if data.hotspot_core_mask_filtered is None else data.hotspot_core_mask_filtered
    hot = np.asarray(curve_source, dtype=bool)[rows, cols]
    handles = []
    if hot.any():
        overlay = np.ma.masked_where(~hot, np.ones(hot.shape))
        axis.pcolormesh(
            lon, lat, overlay, shading="auto", cmap=matplotlib.colors.ListedColormap(["#e63946"]), alpha=0.55, zorder=3
        )
        handles.append(
            matplotlib.patches.Patch(
                facecolor="#e63946", alpha=0.55, label=concise_style.term("hotspot_curve_legend", T["hotspot_curve_legend"], percent=percent)
            )
        )
    core_black = core_grey = None
    if concise:
        # 簡要報告（v65，2026-11-03）：全圖一致的規則——前 10% 的全部核心格點（hotspot_core_mask）都畫出：位於聚集熱區內（∩ hotspot_curve_mask）者黑色，其餘（聚集熱區外）中灰 #7f7f7f；
        # 沒有任何依地點距離的篩選與圓圈。大小與無邊線兩色相同，灰點的 zorder 略低於黑點。
        core_black, core_grey = (mask[rows, cols] for mask in concise_style.hotspot_core_dot_split(curve_source, core_source))
        core_hot = core_black | core_grey
    else:
        core_hot = np.asarray(core_source, dtype=bool)[rows, cols]
    if core_hot.any():
        if concise:
            # 簡要報告（v46）：黑點縮小並拿掉白邊，不遮住底下的紅色熱區格；大小見 concise_style.HOTSPOT_CORE_DOT_PT。灰點先畫（zorder 4.9），黑點後畫（zorder 5）。
            for mask, outside in ((core_grey, True), (core_black, False)):
                if mask.any():
                    mask_rows, mask_cols = np.nonzero(mask)
                    axis.plot(lon[mask_cols], lat[mask_rows], **concise_style.hotspot_core_dot_kwargs(outside=outside))
            handles.extend(concise_core_legend_handles(matplotlib, T))
        else:
            core_rows, core_cols = np.nonzero(core_hot)
            if limits is None:
                cell_points = panel_w * 72.0 / max(lon.size, 1)
            else:
                cell_points = panel_w * 72.0 * _cell_width_deg(data) / max(limits[0][1] - limits[0][0], 1.0e-12)
            marker_size = float(np.clip(0.42 * cell_points, 2.0, 9.0))
            axis.plot(
                lon[core_cols], lat[core_rows], linestyle="none", marker="o", markersize=marker_size, markerfacecolor="#111111",
                markeredgecolor="white", markeredgewidth=0.4, zorder=5,
            )
            legend_style = {"linestyle": "none", "marker": "o", "markersize": 5, "markerfacecolor": "#111111", "markeredgecolor": "white"}
            handles.append(
                matplotlib.lines.Line2D(
                    [], [], label=concise_style.term("hotspot_core_legend", T["hotspot_core_legend"]), **legend_style,
                )
            )
    if concise:
        axis._traps_hotspot_masks = concise_style.hotspot_mask_record(hot, core_black, core_grey)  # noqa: SLF001  供 concise_style.concise_checks_of 讀取（版面紀錄 hotspot_masks）
    buffer_handle = _prepare_map_axis(axis, data, (rows, cols), coastline, max_ticks=4, limits=limits, lbt_style=concise)
    if report_mode:
        style_map_axis(axis)
    else:
        axis.tick_params(labelsize=8, pad=1)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
    if concise:
        outline_handle = draw_analysis_outline(axis, data, limits)  # 簡要報告：分析範圍外框（只畫與站外有效海域相鄰的邊）；有線段才放圖例「分析範圍」
        if outline_handle is not None:
            handles.append(outline_handle)
        partner_handle = draw_partner_overlay(axis, data, (rows, cols), limits)  # A 區：搭檔站範圍淡化、統計歸屬分界
        if partner_handle is not None:
            handles.append(partner_handle)
    if buffer_handle is not None:
        handles.append(buffer_handle)
    if concise:
        handles.append(draw_flow_domain_boundary(axis, data, limits))  # 簡要報告：流場範圍邊界（流場網格外框，深藍色長虛線）；流場範圍外的海域維持白色
    if report_mode:
        figure_height = float(figure.get_size_inches()[1])
        std.legend_below(
            figure, figure_height, handles, [handle.get_label() for handle in handles], ncol=concise_legend_columns(len(handles)) if concise else 3,
            framed=concise, span_inch=std.legend_span_inch(figure, axis, cax) if concise else None,
        )  # 簡要報告（v65）：固定三列，圖例框置中於主圖框左緣到色條右緣的範圍內且不超出
        std.draw_title_block(
            figure,
            figure_height,
            f"{data.site_name_zh} — {T['hotspot_title']}",
            f"{T['hotspot_top']} {percent}%｜{T['hotspot_area']} {area:,.0f} km²",
        )
    else:
        if handles:
            axis.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)
        axis.set_title(title, fontsize=10, pad=3)
        align_colorbars_to_axes(figure, ((axis, cax),))
        _place_footer(figure, full_footer)
    _save(figure, plt, output_path, dpi)
    return output_path


def _cell_width_deg(data: SiteFigureData) -> float:
    """網格經度間距（度）。"""

    lon = np.asarray(data.lon, dtype=np.float64)
    return float(np.median(np.abs(np.diff(lon)))) if lon.size > 1 else 0.01


def _map1_canvas(plt: Any, *, with_colorbar: bool, leg3: bool = False) -> tuple[Any, Any, Any | None]:
    """標準單一地圖畫布（6.3×4.73，移除標題後 6.3×4.13）：回傳 ``(figure, 地圖 axes, 色條 axes 或 None)``；簡要報告圖組的大圖用 LBT 長寬比的主圖框（:func:`concise_map_boxes`）。

    ``leg3=True``（v65，聚集熱區與圖1）：畫布加高 ``CONCISE_LEG3_EXTRA_INCH``（三列圖例），主圖框與色條框下緣同步上移。"""

    axes_box, cbar_box = concise_map_boxes(leg3)  # 簡要報告：LBT 長寬比的主圖框（std.CONCISE_MAP_AXES）；其餘 MAP1
    height = std.SINGLE_HEIGHT_INCH + (std.CONCISE_LEG3_EXTRA_INCH if leg3 and concise_style.concise_enabled() else 0.0)
    figure = plt.figure(figsize=(std.WIDTH_INCH, height))
    axis = std.add_box_axes(figure, axes_box)
    cax = std.add_box_axes(figure, cbar_box) if with_colorbar else None
    return figure, axis, cax


def _single_map_canvas(
    plt: Any,
    data: SiteFigureData,
    view: tuple[slice, slice],
    *,
    with_colorbar: bool,
    legend_height: float = 0.0,
    tight: bool = False,
) -> tuple[Any, Any, Any | None, float, float]:
    """單一地圖（可含右側色條與底部圖例帶）的 Figure 與 axes。

    報告模式固定 6.3 英吋寬；``panel_w`` 先扣除左右軸與色條空間，因此不同站的地理範圍
    不會產生超寬 Figure。回傳 ``(figure, axis, cax, panel_w, panel_h)``。

    ``tight=False`` 使用與 ``render_hotspots`` 相同的邊界；``tight=True``（累積天數與熱點
    重疊圖）把上方標題帶、下方 x 軸標籤帶縮到剛好容納兩行標題與 8 pt 刻度／標籤，沒有
    色條時右側也只留小邊界，避免大片空白。這些只是 axes 位置，不改任何資料。
    """

    report_mode = vector_report_enabled()
    cbar_block = (_CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH) if with_colorbar else (0.30 if tight else _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH)
    panel_w, panel_h = _fit_panel(
        map_data_aspect(data, view),
        VECTOR_REPORT_WIDTH_INCH - _LEFT_INCH - cbar_block if report_mode else 6.6,
        5.0 if report_mode else 5.6,
    )
    content_w = _LEFT_INCH + panel_w + cbar_block
    top_margin, title_h = (0.06, 0.50) if tight else (0.1, 0.62)
    bottom_label = 0.52 if tight else _BOTTOM_LABEL_INCH
    report_bottom_extra = 0.0 if tight else (0.24 if report_mode else 0.0)
    fig_w = VECTOR_REPORT_WIDTH_INCH if report_mode else max(content_w, _MIN_FIGURE_WIDTH)
    fig_h = top_margin + title_h + panel_h + bottom_label + report_bottom_extra + legend_height
    x0 = (fig_w - content_w) / 2.0 + _LEFT_INCH
    y0 = bottom_label + report_bottom_extra + legend_height
    figure = plt.figure(figsize=(fig_w, fig_h))
    axis = _add_inch_axes(figure, x0, y0, panel_w, panel_h)
    cax = _add_inch_axes(figure, x0 + panel_w + _CBAR_PAD_INCH, y0, _CBAR_WIDTH_INCH, panel_h) if with_colorbar else None
    return figure, axis, cax, panel_w, panel_h


def render_cumulative_days(
    output_path: Path,
    data: SiteFigureData,
    *,
    counts: np.ndarray,
    valid_days: int,
    coastline: CoastlineGeometry | None,
    dpi: int = 150,
) -> Path:
    """累積天數地圖：每格 TRAP 曲線通過的有效日數（見 :mod:`traps.cumulative_days`）。

    ``counts`` 與 ``data.lon/lat`` 同形，站外為 NaN；色階固定從 0 到本站最大日數（站與站
    各自計算，同 hotspots 的逐站色階慣例）。地圖版面、海岸線與近岸緩衝帶等值線沿用
    ``render_hotspots`` 的規則；副標題標明站內有效日數總數。
    """

    _, plt = _load_matplotlib()
    view = _view_extent(data)
    report_mode = vector_report_enabled()
    limits = None
    if report_mode:
        view, limits = standard_map_frame(data, view, concise_map_boxes()[0])  # 簡要報告：與 _map1_canvas 同一個軸框（LBT 長寬比），長寬比誤差才與實際軸框一致
    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    if report_mode:
        figure, axis, cax = _map1_canvas(plt, with_colorbar=True)
    else:
        figure, axis, cax, _panel_w, _panel_h = _single_map_canvas(plt, data, view, with_colorbar=True, tight=True)
    field = np.ma.masked_invalid(np.asarray(counts, dtype=np.float64)[rows, cols])
    finite = np.asarray(counts, dtype=np.float64)
    finite = finite[np.isfinite(finite) & np.asarray(data.site_mask, dtype=bool)]
    vmax = float(finite.max()) if finite.size and float(finite.max()) > 0.0 else 1.0
    mesh = axis.pcolormesh(lon, lat, field, shading="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
    colorbar = figure.colorbar(mesh, cax=cax)
    from matplotlib.ticker import MaxNLocator

    colorbar.locator = MaxNLocator(nbins=6, integer=True)
    colorbar.update_ticks()
    colorbar.set_label(T["cumulative_days_colorbar"], fontsize=8, labelpad=2)
    colorbar.ax.tick_params(labelsize=8, pad=1)
    buffer_handle = _prepare_map_axis(axis, data, view, coastline, max_ticks=4, limits=limits)
    axis.tick_params(labelsize=8, pad=1)
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)
    if not report_mode and buffer_handle is not None:
        axis.legend(handles=[buffer_handle], loc="lower right", fontsize=8, framealpha=0.85)
    if not finite.size:
        axis.text(0.5, 0.5, T["no_data"], transform=axis.transAxes, ha="center", va="center", fontsize=10)
    title = (
        f"{data.site_name_zh} — {T['cumulative_days_title']}\n"
        f"{T['cumulative_days_subtitle_a']} {int(valid_days)} {T['cumulative_days_subtitle_b']}"
    )
    if report_mode:
        handles = [] if buffer_handle is None else [buffer_handle]
        std.legend_below(figure, std.SINGLE_HEIGHT_INCH, handles, [handle.get_label() for handle in handles])
        std.draw_title_block(
            figure,
            std.SINGLE_HEIGHT_INCH,
            f"{data.site_name_zh} — {T['cumulative_days_title']}",
            f"{T['cumulative_days_subtitle_a']} {int(valid_days)} {T['cumulative_days_subtitle_b']}",
        )
    else:
        axis.set_title(title, fontsize=10, pad=3)
        align_colorbars_to_axes(figure, ((axis, cax),))
    _save(figure, plt, output_path, dpi)
    return output_path


OVERLAP_COLOURS = {"site": "#e6e6e6", "hourly": "#0072B2", "daily": "#E69F00", "both": "#009E73"}
"""熱點重疊圖的類別色（色盲友善）：站內其他格點、僅逐時、僅逐日、兩者皆是。"""


def render_hotspot_overlap(
    output_path: Path,
    data: SiteFigureData,
    *,
    hourly_mask: np.ndarray,
    daily_mask: np.ndarray,
    jaccard: float | None,
    coastline: CoastlineGeometry | None,
    dpi: int = 150,
) -> Path:
    """逐時與逐日曲線通過頻率熱點的重疊地圖（僅逐時／僅逐日／兩者皆是）。

    遮罩是 aggregate 定義的熱點（見 :mod:`traps.hotspot_overlap`），不做小連通塊濾除；
    站內其他格點以淺灰顯示，站外留白。標題第二行列出兩產品格數、交集與 Jaccard。
    """

    matplotlib, plt = _load_matplotlib()
    view = _view_extent(data)
    report_mode = vector_report_enabled()
    limits = None
    if report_mode:
        view, limits = standard_map_frame(data, view, concise_map_boxes()[0])
    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    legend_height = 0.46
    if report_mode:
        figure, axis, _cax = _map1_canvas(plt, with_colorbar=False)
    else:
        figure, axis, _cax, _panel_w, _panel_h = _single_map_canvas(plt, data, view, with_colorbar=False, legend_height=legend_height, tight=True)
    site = np.asarray(data.site_mask, dtype=bool)
    hourly = np.asarray(hourly_mask, dtype=bool) & site
    daily = np.asarray(daily_mask, dtype=bool) & site
    category = np.where(site, 0, -1).astype(float)
    category[hourly & ~daily] = 1.0
    category[daily & ~hourly] = 2.0
    category[hourly & daily] = 3.0
    shown = np.ma.masked_less(category[rows, cols], 0.0)
    colours = [OVERLAP_COLOURS[key] for key in ("site", "hourly", "daily", "both")]
    cmap = matplotlib.colors.ListedColormap(colours)
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    axis.pcolormesh(lon, lat, shown, shading="auto", cmap=cmap, norm=norm)
    buffer_handle = _prepare_map_axis(axis, data, view, coastline, max_ticks=4, limits=limits)
    axis.tick_params(labelsize=8, pad=1)
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)
    handles = [
        matplotlib.patches.Patch(facecolor=OVERLAP_COLOURS["hourly"], label=T["hotspot_overlap_hourly_only"]),
        matplotlib.patches.Patch(facecolor=OVERLAP_COLOURS["daily"], label=T["hotspot_overlap_daily_only"]),
        matplotlib.patches.Patch(facecolor=OVERLAP_COLOURS["both"], label=T["hotspot_overlap_both"]),
        matplotlib.patches.Patch(facecolor=OVERLAP_COLOURS["site"], label=T["hotspot_overlap_site_cells"]),
    ]
    if buffer_handle is not None:
        handles.append(buffer_handle)
    cell = T["hotspot_overlap_cells"]
    jaccard_text = "n/a" if jaccard is None else f"{float(jaccard):.2f}"
    title = (
        f"{data.site_name_zh} — {T['hotspot_overlap_title']}\n"
        f"{T['hotspot_overlap_hourly']} {int(hourly.sum())} {cell}｜{T['hotspot_overlap_daily']} {int(daily.sum())} {cell}"
        f"｜{T['hotspot_overlap_intersection']} {int((hourly & daily).sum())} {cell}｜Jaccard {jaccard_text}"
    )
    if report_mode:
        std.legend_below(figure, std.SINGLE_HEIGHT_INCH, handles, [handle.get_label() for handle in handles])
        subtitle = title.split("\n", 1)[1]
        std.draw_title_block(figure, std.SINGLE_HEIGHT_INCH, f"{data.site_name_zh} — {T['hotspot_overlap_title']}", subtitle)
    else:
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.04 / figure.get_figheight()),
            ncol=3,
            fontsize=8,
            frameon=False,
            columnspacing=0.9,
            handlelength=1.6,
        )
        axis.set_title(title, fontsize=10, pad=3)
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
    if vector_report_enabled():
        # 標準 multi 畫布（6.3×7.0）：四季地圖 2×2、橫跨的水平色條、下方兩張統計長條圖，
        # 位置與大小固定；地圖以固定軸框擴張經緯度範圍，逐格數值與統計值不變。
        fig_w, fig_h = std.WIDTH_INCH, std.MULTI_HEIGHT_INCH
        panel_box = std.Box(0.0, 0.0, std.SEASONAL_PANEL_W, std.SEASONAL_PANEL_H)
        (rows, cols), limits = standard_map_frame(data, (rows, cols), panel_box)
        lon, lat = data.lon[cols], data.lat[rows]
        figure = plt.figure(figsize=(fig_w, fig_h))
        mesh = None
        for index, season in enumerate(SEASONS):
            row, column = divmod(index, 2)
            axis = _add_inch_axes(figure, std.SEASONAL_X[column], std.SEASONAL_Y[row], std.SEASONAL_PANEL_W, std.SEASONAL_PANEL_H)
            masked = np.ma.masked_invalid(np.asarray(data.spatial_seasons[season]["curve_pass_frequency"], dtype=np.float64)[rows, cols])
            mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
            axis.set_title(T[f"season_{season}"], fontsize=std.SUBTITLE_FONT_PT, pad=3)
            _prepare_map_axis(axis, data, (rows, cols), coastline, max_ticks=3, draw_buffer=False, limits=limits)
            style_map_axis(axis)
            if column == 1:
                # 右欄仍保留 y 軸 locator、grid 與刻度位置，確保兩欄使用完全相同的地理範圍；
                # 只隱藏重複的 ylabel／左側 tick label。
                axis.set_ylabel("")
                axis.tick_params(axis="y", labelleft=False)
        if mesh is not None:
            cax = std.add_box_axes(figure, std.SEASONAL_CBAR)
            colorbar = figure.colorbar(mesh, cax=cax, orientation="horizontal")
            style_colorbar(colorbar, f"{T['curve_pass_frequency']} ({T['fraction']})")

        # 兩個統計軸共用相同資料紀錄；n、95% CI 與缺值規則保持既有實作。
        x = np.arange(len(SEASONS))
        for column, (key, title) in enumerate(
            (("strong_core_occurrence_rate", T["strong_core_occurrence"]), ("mean_pass_cell_fraction", T["mean_pass_fraction"]))
        ):
            axis = _add_inch_axes(figure, std.SEASONAL_BARS_X[column], std.SEASONAL_BARS_Y, std.SEASONAL_BARS_W, std.SEASONAL_BARS_H)
            values, errors, counts = [], [], []
            for season in SEASONS:
                record = data.records.get(season)
                value = record.get(key) if record else None
                values.append(float(value) if value is not None else 0.0)
                errors.append(_error_bar(record, key))
                counts.append(int(record["n_valid_snapshots"]) if record else 0)
            axis.bar(x, values, yerr=np.asarray(errors).T, capsize=4, color="#4c78a8", label=T["ci95"])
            axis.set_xticks(list(x))
            # 統計柱狀圖使用短中文季名，地圖標題仍保留 DJF/MAM/JJA/SON。
            axis.set_xticklabels(["冬季", "春季", "夏季", "秋季"], fontsize=std.TEXT_FONT_PT)
            axis.set_ylabel(T["fraction"], fontsize=std.TEXT_FONT_PT)
            # 加大 y 上限只改顯示範圍，不改柱高、CI 或任何統計值。
            axis.set_title(f"{title}（{T['ci95']}）", fontsize=std.SUBTITLE_FONT_PT, pad=4)
            tops = [value + high for value, (_low, high) in zip(values, errors, strict=True)]
            top = max(tops + [1.0e-9])
            for xi, count, level in zip(x, counts, tops, strict=True):
                axis.text(xi, level + 0.02 * top, f"n={count}" if count else "n=0", ha="center", va="bottom", fontsize=std.TEXT_FONT_PT)
            axis.set_ylim(0.0, top * 1.45)
            axis.tick_params(labelsize=std.TEXT_FONT_PT, pad=1)
        std.draw_title_block(figure, fig_h, f"{data.site_name_zh} — {T['seasonal_title']}")
        _place_footer(figure, footer)
        _save(figure, plt, output_path, dpi)
        return output_path

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


LIFETIME_LEGEND_BAND_INCH = 0.55
"""壽命圖左面板圖例帶高度（英吋）：兩欄圖例置於面板下方，高度與 8 pt 兩列文字相符。"""


def lifetime_left_title_report(title: str, refers_to: str, scope_text: str) -> str:
    """報告模式 CCDF 標題：在語意邊界換行，避免把「（各站前 10%）」之類括號片段拆開。

    ``scope_text`` 形如「強核心結構（各站前 10%）」時分成三行：標題、「壽命對象：強核心結構」
    與括號補充「（各站前 10%）」；沒有全形左括號時，標題與「壽命對象：…」各一行。
    """

    if not scope_text:
        return title
    head, sep, tail = scope_text.partition("（")
    lines = [title, f"{refers_to}：{head}"]
    if sep:
        lines.append(f"（{tail}")
    return "\n".join(lines)


def render_lifetime(output_path: Path, data: SiteFigureData, *, footer: str, dpi: int = 150) -> Path:
    """繪製壽命 CCDF 與正規化年齡演化；門檻保留垂直線，不再繪製下方數值表。

    壽命門檻的實際比例、隨機對照與超額值仍由彙整結果的 manifest／報告保存；圖中
    只保留門檻垂直線、CCDF 圖例，以及右側面板內的 ``n = ...`` 軌跡數，避免移除
    描述文字後在 x 軸下方留下額外高度。
    """

    _, plt = _load_matplotlib()
    # 報告模式使用固定 6.3 英吋列印寬度；增加左邊界是為了容納 CCDF 的數學
    # y 軸標籤，標題則以換行方式保留完整科學語意，避免縮入報告後被裁切。
    report_mode = vector_report_enabled()
    # CCDF 圖例放在兩個面板下方（figure-level、兩欄），不再壓在左圖曲線上；因此總高度
    # 增加 legend_band_inch，面板與上方標題區的英吋尺寸維持不變。
    legend_band_inch = LIFETIME_LEGEND_BAND_INCH
    if report_mode:
        # 標準 single 畫布（6.3×4.4）：兩個面板的座標軸框固定，三行標題在面板上方，
        # 圖例（兩欄）在面板下方的固定圖例帶。
        fig_w, fig_h = std.WIDTH_INCH, std.SINGLE_HEIGHT_INCH
        figure = plt.figure(figsize=(fig_w, fig_h))
        left = std.add_box_axes(figure, std.LIFETIME_LEFT)
        right = std.add_box_axes(figure, std.LIFETIME_RIGHT)
    else:
        top_inch, axes_inch, bottom_inch = 1.05, 5.6 - 1.05 - (_footer_height_inch(footer) + 0.75), _footer_height_inch(footer) + 0.75
        fig_w = 13.0
        fig_h = top_inch + axes_inch + bottom_inch + legend_band_inch
        figure, (left, right) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        figure.subplots_adjust(
            left=0.07,
            right=0.97,
            bottom=(bottom_inch + legend_band_inch) / fig_h,
            top=1.0 - top_inch / fig_h,
            wspace=0.25,
        )

    thresholds = [float(t) for t in data.lifetime_thresholds_hours if float(t) > 0]
    daily = float(data.nominal_step_hours) >= 24.0 or (bool(thresholds) and all(abs(t / 24.0 - round(t / 24.0)) < 1e-9 for t in thresholds))
    unit_h = 24.0 if daily else 1.0

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
    if x_u.size or x_a.size or has_null:
        left.set_yscale("log")
        legend_handles, legend_labels = left.get_legend_handles_labels()
        if report_mode:
            std.legend_below(figure, fig_h, legend_handles, legend_labels, ncol=2)
        else:
            figure.legend(
                legend_handles,
                legend_labels,
                fontsize=8,
                loc="lower center",
                ncol=2,
                bbox_to_anchor=(0.5, (_footer_height_inch(footer) + 0.04) / fig_h),
                frameon=True,
                columnspacing=1.2,
                handlelength=2.0,
            )
    else:
        left.text(0.5, 0.5, T["no_data"], ha="center", va="center", transform=left.transAxes)
    if x_u.size or x_a.size:
        for threshold in thresholds:
            # 垂直線保留門檻在 CCDF 上的定位；數值比例由 manifest／報告提供，
            # 因此不再在 x 軸下方另外建立表格文字。
            left.axvline(threshold / unit_h, color="#6c757d", linestyle=":", linewidth=1.0, zorder=1)
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
    left.set_xlabel(T["lifetime_days"] if daily else T["lifetime_hours"], fontsize=8 if report_mode else None)
    left.set_ylabel(T["ccdf_ylabel"], fontsize=8 if report_mode else None)
    left_title = f"{T['ccdf_title']}\n{T['lifetime_refers_to']}：{data.lifetime_scope_text}" if data.lifetime_scope_text else T["ccdf_title"]
    if report_mode:
        left_title = lifetime_left_title_report(T["ccdf_title"], T["lifetime_refers_to"], data.lifetime_scope_text)
    left.set_title(left_title, fontsize=10, pad=3)
    left.tick_params(which="both", labelsize=8 if report_mode else None, pad=1)
    left.grid(alpha=0.25)

    age = np.linspace(0.0, 1.0, len(data.evolution_mean_day)) if len(data.evolution_mean_day) else np.zeros(0)
    if age.size:
        right.plot(age, np.asarray(data.evolution_mean_day, dtype=np.float64), color="#c1121f", marker="o", markersize=3.5, linewidth=1.6)
        # 軌跡數放在標題第三行而不是面板內：面板內任何位置都可能壓到曲線。
        n_tracks_line = f"n = {data.evolution_track_count} {T['n_tracks']}"
    else:
        n_tracks_line = ""
        right.text(0.5, 0.5, T["no_data"], ha="center", va="center", transform=right.transAxes)
    right.set_xlim(0.0, 1.0)
    right.set_xlabel(T["normalized_age"], fontsize=8 if report_mode else None)
    right.set_ylabel(f"{T['mean_core_s1']} {S1_DAY_LABEL}", fontsize=8 if report_mode else None)
    right_title = f"{T['evolution_title']}\n{T['evolution_subtitle']} ≥ {_hours_label(data.lifetime_evolution_min_hours)} {T['hour']}"
    # 右標題每行都是完整語意單元（主標題／「未截尾且壽命 ≥ N 小時」／軌跡數），不再以固定字數硬折行。
    if n_tracks_line:
        right_title = f"{right_title}\n{n_tracks_line}"
    right.set_title(right_title, fontsize=10, pad=3)
    right.tick_params(labelsize=8 if report_mode else None, pad=1)
    right.grid(alpha=0.25)

    if report_mode:
        std.draw_title_block(figure, fig_h, f"{data.site_name_zh} — {T['lifetime_title']}")
    else:
        figure.suptitle(f"{data.site_name_zh} — {T['lifetime_title']}", fontsize=13, y=1.0 - 0.12 / fig_h, va="top")
    _place_footer(figure, footer)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_coastal_diagnostic(output_path: Path, data: SiteFigureData, *, footer: str, dpi: int = 150) -> Path:
    """近岸診斷：(a) 核心距岸距離分布（全部／強核心／長壽命 track 核心）；(b) 近岸型與近岸外的壽命 CCDF 與各自的隨機對照。

    (a) 三組各自正規化為比例（總和 1），因此可比較「分布形狀」而非數量；虛線為近岸緩衝帶外緣。
    (b) 近岸型＝track 平均核心距岸小於緩衝帶，近岸外＝不小於；各配對其時間置換 null（各 seed 平均與最小–最大帶）。
    讀者可由此判斷長壽命結構是否集中在海岸帶、去除海岸帶後是否仍超出隨機對照；
    分類規則與 provenance 由 manifest／報告保存，圖面下方不再繪製說明文字。
    """

    _, plt = _load_matplotlib()
    diag = data.coastal
    assert diag is not None
    # 近岸診斷的分類規則與資料 provenance 由 manifest／報告保存；圖下不再預留
    # 原本兩行說明的高度。報告模式固定 6.3 英吋，並加大左右邊界容納距岸與 CCDF
    # 的 y 軸標籤；長標題只換行，不改變分組、bin 或隨機對照資料。
    report_mode = vector_report_enabled()
    if report_mode:
        # 標準 multi 畫布（6.3×7.0）：(a)(b) 上下兩個固定座標軸框，圖例在各軸下方。
        fig_w, fig_h = std.WIDTH_INCH, std.MULTI_HEIGHT_INCH
        figure = plt.figure(figsize=(fig_w, fig_h))
        left = std.add_box_axes(figure, std.COASTAL_A)
        right = std.add_box_axes(figure, std.COASTAL_B)
    else:
        fig_w, fig_h = (13.0, 5.6)
        figure, (left, right) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        figure.subplots_adjust(left=0.07, right=0.97, bottom=(_footer_height_inch(footer) + 0.75) / fig_h, top=1.0 - 1.05 / fig_h, wspace=0.25)

    def _coastal_legend(axis: Any, ncol: int) -> None:
        if report_mode:
            axis.legend(fontsize=std.TEXT_FONT_PT, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=ncol, frameon=False, columnspacing=1.0, handlelength=1.6)
        else:
            axis.legend(fontsize=8, loc="upper right")

    edges = np.asarray(diag.hist_edges_km, dtype=np.float64)
    # 「≥ 上限」溢位 bin：距離場飽和於 cap_km（新版含最後一格溢位 bin；舊 shard 為 30 km），
    # 起自 cap 的所有 bin 併為一根明確標示的長條，不與前面的曲線混在一起（避免最後一格堆出假尖峰）。
    cap_km = float(diag.cap_km) if diag.cap_km is not None else float(edges[-1])
    cap_index = int(np.searchsorted(edges, cap_km - 1.0e-9))
    cap_index = min(max(cap_index, 1), edges.size - 1)
    cap_label = f"{T['coast_cap_label']} {cap_km:g} km"
    groups = (
        (diag.hist_all, T["coast_hist_all"], "#1b4965", "-"),
        (diag.hist_strong, T["coast_hist_strong"], "#c1121f", "-"),
        (diag.hist_long, T["coast_hist_long"], "#2a9d8f", "-"),
    )
    any_hist = False
    x_max = 0.0
    overflow_drawn = False
    for counts, label, colour, style in groups:
        counts = np.asarray(counts, dtype=np.float64)
        total = float(counts.sum())
        if total <= 0.0:
            continue
        any_hist = True
        body = counts[:cap_index] / total
        overflow = float(counts[cap_index:].sum()) / total
        body_edges = edges[: cap_index + 1]
        left.step(body_edges, np.append(body, body[-1]), where="post", color=colour, linestyle=style, linewidth=1.6, label=f"{label} (n={int(total)})")
        nonzero = np.nonzero(counts[:cap_index])[0]
        x_max = max(x_max, float(edges[nonzero[-1] + 1]) if nonzero.size else 0.0)
        if overflow > 0.0:
            overflow_drawn = True
            width = float(edges[cap_index + 1] - edges[cap_index])
            left.bar(float(edges[cap_index]), overflow, width=width, align="edge", color=colour, alpha=0.35, hatch="//", edgecolor=colour, linewidth=0.6)
            x_max = max(x_max, cap_km + width)
    left.axvline(float(diag.buffer_km), color=COAST_BUFFER_COLOUR, linestyle="--", linewidth=1.1, label=coast_buffer_label(data))
    x_max = max(x_max, 3.0 * float(diag.buffer_km)) if not overflow_drawn else x_max
    x_max = min(max(x_max, 1.0), cap_km + float(edges[cap_index + 1] - edges[cap_index]))
    floor_km = diag.detection_floor_km
    if floor_km is not None and any_hist and floor_km > 0.0:
        left.axvspan(
            0.0, float(floor_km), color="#9e9e9e", alpha=0.25, linewidth=0, zorder=0,
            label=f"{T['coast_floor_a']} {float(floor_km):.1f} km{T['coast_floor_b']}",
        )
    if overflow_drawn:
        left.text(
            cap_km + 0.5 * float(edges[cap_index + 1] - edges[cap_index]),
            0.0,
            cap_label,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8 if report_mode else 7.5,
            color="#424242",
        )
    if any_hist:
        _coastal_legend(left, 2)
    else:
        left.text(0.5, 0.5, T["no_data"], ha="center", va="center", transform=left.transAxes)
    left.set_xlim(0.0, x_max)
    left.set_xlabel(T["coast_dist_xlabel"], fontsize=8 if report_mode else None)
    left.set_ylabel(T["coast_hist_ylabel"], fontsize=8 if report_mode else None)
    # 近岸診斷的既有文字測試與報告引用都以完整標題辨識圖面；因此保留單一
    # Text 物件的原始字串，不在此切開標題。固定的 6.3 英吋版面、10 pt 標題
    # 與左右邊界仍可讓標題在圖內置中呈現，且不影響資料、分組或座標軸。
    left_title = T["coast_hist_title"]
    left.set_title(left_title, fontsize=10, pad=3)
    left.tick_params(labelsize=8 if report_mode else None, pad=1)
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
        _coastal_legend(right, 2)
    else:
        right.text(0.5, 0.5, T["coast_no_tracks"], ha="center", va="center", transform=right.transAxes)
    right.set_xlabel(T["lifetime_days"] if daily else T["lifetime_hours"], fontsize=8 if report_mode else None)
    right.set_ylabel(T["ccdf_ylabel"], fontsize=8 if report_mode else None)
    # 同上，保留完整的 CCDF 標題字串，避免既有報告文字擷取測試與下游索引
    # 將換行後的片段視為不同標題；標題仍受報告模式的 6.3 英吋與 10 pt 契約約束。
    right_title = T["coast_ccdf_title"]
    right.set_title(right_title, fontsize=10, pad=3)
    right.tick_params(labelsize=8 if report_mode else None, pad=1)
    right.grid(alpha=0.25)

    if report_mode:
        std.draw_title_block(figure, fig_h, f"{data.site_name_zh} — {T['coastal_diag_title']}")
    else:
        figure.suptitle(f"{data.site_name_zh} — {T['coastal_diag_title']}", fontsize=13, y=1.0 - 0.12 / fig_h, va="top")
    _place_footer(figure, footer)
    _save(figure, plt, output_path, dpi)
    return output_path


def render_timeseries(output_path: Path, data: SiteFigureData, *, footer: str, dpi: int = 150) -> Path:
    """逐日含強核心的有效快照比例與平均通過格點比例；30 日滑動平均，資料缺口以斷線呈現。"""

    _, plt = _load_matplotlib()
    calendar, strong, passed = daily_series(data.day_stats)
    # 移除兩行 footer 後縮短整體高度；0.75 英吋仍足以容納日期標籤與其刻度。
    # 報告模式使用單欄雙面板，PNG 與 SVG/PDF 共用同一版面。
    report_mode = vector_report_enabled()
    if report_mode:
        # 標準 multi 畫布（6.3×7.0）：上下兩個共用 x 軸的固定面板；圖例在各面板上緣（右對齊）。
        fig_w, fig_h = std.WIDTH_INCH, std.MULTI_HEIGHT_INCH
        figure = plt.figure(figsize=(fig_w, fig_h))
        top_axis = std.add_box_axes(figure, std.TIMESERIES_1)
        bottom_axis = std.add_box_axes(figure, std.TIMESERIES_2, sharex=top_axis)
        axes = [top_axis, bottom_axis]
        top_axis.tick_params(axis="x", labelbottom=False)
    else:
        fig_w, fig_h = (12.5, 7.2)
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
            if report_mode:
                axis.legend(fontsize=std.TEXT_FONT_PT, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, borderaxespad=0.2)
            else:
                axis.legend(fontsize=8, loc="upper right")
        else:
            axis.text(0.5, 0.5, T["no_data"], ha="center", va="center", transform=axis.transAxes)
        axis.set_ylabel(label, fontsize=8 if report_mode else 9)
        axis.set_ylim(bottom=0.0)
        axis.tick_params(labelsize=8 if report_mode else None, pad=1)
        axis.grid(alpha=0.25)
    axes[1].set_xlabel(T["date"], fontsize=8 if report_mode else None)
    if report_mode:
        std.draw_title_block(figure, fig_h, f"{data.site_name_zh} — {T['timeseries_title']}")
    else:
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
    kinds: Collection[str] | None = None,
) -> list[str]:
    """輸出一個站的五張圖到 ``figures/<site_id>/``，回傳相對 ``output_dir`` 的路徑清單。

    ``font_selection`` 必須是已通過字元涵蓋檢查的 CJK 字型；為 ``None`` 時拒絕繪圖
    （圖面不可輸出方框字）。繪製期間以 :func:`cjk_font_context` 暫時設定
    ``font.family``，結束後還原全域 rcParams。``kinds`` 不為 ``None`` 時只輸出指定的圖種
    （僅供局部重繪，例如只重畫 ``lifetime``）；預設輸出全部。
    """

    if font_selection is None:
        raise ValueError("沒有合格的 CJK 字型，拒絕輸出成果圖（可改用 --no-figures）")
    quantile = float(hotspot_quantile if hotspot_quantile is not None else data.hotspot_metrics["hotspot_quantile"])
    footer = _footer(period_label, state_label, param_id, data.n_valid_snapshots, font_selection.family)
    site_dir = output_dir / "figures" / data.site_id
    stem = f"site_{data.site_id}"
    wanted = None if kinds is None else frozenset(kinds)

    def _want(kind: str) -> bool:
        return wanted is None or kind in wanted

    paths: list[str] = []
    with cjk_font_context(font_selection):
        if _want("spatial"):
            render_spatial(site_dir / f"{stem}_spatial.png", data, coastline=coastline, footer=footer, dpi=dpi)
        if _want("hotspots"):
            render_hotspots(site_dir / f"{stem}_hotspots.png", data, hotspot_quantile=quantile, coastline=coastline, footer=footer, dpi=dpi)
        if _want("seasonal"):
            render_seasonal(site_dir / f"{stem}_seasonal.png", data, coastline=coastline, footer=footer, dpi=dpi)
        if _want("lifetime"):
            render_lifetime(site_dir / f"{stem}_lifetime.png", data, footer=footer, dpi=dpi)
        if _want("timeseries"):
            render_timeseries(site_dir / f"{stem}_timeseries.png", data, footer=footer, dpi=dpi)
        paths = [
            f"figures/{data.site_id}/{stem}_{kind}.png"
            for kind in ("spatial", "hotspots", "seasonal", "lifetime", "timeseries")
            if _want(kind)
        ]
        if data.coastal is not None and _want("coastal_diagnostic"):
            render_coastal_diagnostic(site_dir / f"{stem}_coastal_diagnostic.png", data, footer=footer, dpi=dpi)
            paths.append(f"figures/{data.site_id}/{stem}_coastal_diagnostic.png")
    return paths


__all__ = [
    "CoastalDiagnosticData",
    "SiteFigureData",
    "coast_buffer_label",
    "daily_series",
    "render_cumulative_days",
    "render_hotspot_overlap",
    "render_site_figures",
    "running_mean_with_gaps",
]
