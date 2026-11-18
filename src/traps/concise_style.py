"""簡要報告（concise）圖組的共用樣式：顯示名稱、跨站共用色階、規則經緯度刻度、色條端點刻度。

只有設定環境變數 ``TRAPS_CONCISE_OPTIONS``（指向 JSON 選項檔）且 ``TRAPS_VECTOR_REPORT=1`` 時，各繪圖函式的
報告模式分支才會呼叫本模組；環境變數未設定時所有函式都不會被呼叫，正式圖組與一般分析完全不受影響。

選項檔格式（``traps.concise_options.v1``）::

    {
      "schema_version": "traps.concise_options.v1",
      "display_names": {"guishan": "龜山島"},
      "limits": {
        "hotspots": {"curve_pass_frequency": [0.0, 0.35]},
        "example": {"s1_day": [-12.0, 12.0]},
        "domainA_overview": {"curve_pass_frequency": [0.0, 0.35]}
      }
    }

``TRAPS_CONCISE_RECORD`` 指向 JSON Lines 檔時，每張圖會把自己「原本」的色階附加記錄（供選共用色階用）。
這些設定只改顏色對應的上下限、刻度與標示文字，不改任何資料、遮罩或統計。
"""

from __future__ import annotations

import json
import math
import re
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ENV_OPTIONS = "TRAPS_CONCISE_OPTIONS"
ENV_RECORD = "TRAPS_CONCISE_RECORD"
OPTIONS_SCHEMA = "traps.concise_options.v1"
CHECKS_SCHEMA = "traps.concise_figure_checks.v1"

MIN_COLORBAR_INTERIOR_TICKS = 2
"""色條端點之間至少要有的標示刻度數。"""

MINUS = "−"
FRAME_COLOUR = "#222222"
FRAME_LINEWIDTH = 1.1
TICK_LENGTH_PT = 3.5

_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def options() -> dict[str, Any] | None:
    """讀取選項檔；環境變數未設定時回傳 ``None``（代表不是 concise 圖組）。"""

    value = os.environ.get(ENV_OPTIONS)
    if not value:
        return None
    path = Path(value)
    mtime = path.stat().st_mtime
    cached = _cache.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != OPTIONS_SCHEMA:
        raise ValueError(f"concise 選項檔 schema 不符: {path}")
    _cache[str(path)] = (mtime, payload)
    return payload


def concise_enabled() -> bool:
    return options() is not None


CONCISE_TERMS: dict[str, str] = {
    "hotspot_curve_legend_domain": "聚集熱區（各研究地點 TRAP 曲線出現頻率前 {percent} %）",
    "recurrence_colourbar": "聚集熱區出現季數",
    "hotspot_core_legend_outside": "TRAP 核心最常出現的位置（聚集熱區外）",
}
"""簡要報告圖組圖內文字中，不在 :data:`traps.report_font.FIGURE_TEXT` 的用詞（A 區總覽的圖例、四季重現色條）。
其餘用詞（TRAP 曲線出現頻率、聚集熱區、TRAP 核心最常出現的位置、地點位置）由 ``FIGURE_TEXT`` 提供（對齊服務建議書：「熱點」指五個地點）。"""

HOTSPOT_MASK_BASIS = "unfiltered"
"""簡要報告（2026-10-27 v45）：聚集熱區與核心黑點畫統計定義本身的未濾除遮罩（``hotspot_curve_mask``／``hotspot_core_mask``，各地點前 10%），
不濾除 4-連通格數小於 ``hotspot_min_cells`` 的小塊，圖面所見即統計母體。正式圖組不受影響（仍畫 ``*_filtered``）。"""
HOTSPOT_MASK_NAMES = {"curve": "hotspot_curve_mask", "core": "hotspot_core_mask"}


def hotspot_masks_for_figure(curve_mask: Any, core_mask: Any, curve_mask_filtered: Any = None, core_mask_filtered: Any = None) -> tuple[Any, Any]:
    """concise 模式圖面使用的 (聚集熱區遮罩, 核心統計遮罩)：永遠是未濾除的統計遮罩；``*_filtered`` 參數只為明示「不使用」（呼叫端可照舊傳入）。

    注意（v65）：核心統計遮罩的格點全部畫出，依是否落在聚集熱區內分成黑點與灰點（:func:`hotspot_core_dot_split`）。"""

    return curve_mask, core_mask


HOTSPOT_CORE_OUTSIDE_COLOUR = "#7f7f7f"
"""v65（2026-11-03）：位於聚集熱區外的核心格點（``hotspot_core_mask`` 減去 ``hotspot_curve_mask``）的中灰填色；聚集熱區內者（兩遮罩的交集）維持黑色 :data:`HOTSPOT_CORE_DOT_FACE`。
全圖一致的規則：前 10% 的全部核心格點都畫出，沒有任何依地點距離的篩選，也沒有圓圈。"""


def hotspot_core_dot_split(curve_mask: Any, core_mask: Any) -> tuple[np.ndarray, np.ndarray]:
    """核心格點分成 (黑點遮罩, 灰點遮罩)：黑點 ＝ ``core ∩ curve``（聚集熱區內），灰點 ＝ ``core − curve``（聚集熱區外）；兩者合起來就是完整的 ``core``。"""

    core = np.asarray(core_mask, dtype=bool)
    curve = np.asarray(curve_mask, dtype=bool)
    return core & curve, core & ~curve


def hotspot_mask_record(curve_drawn: Any, black_drawn: Any, grey_drawn: Any) -> dict[str, Any]:
    """版面紀錄 ``concise.map_axes[].hotspot_masks``：圖面使用的遮罩基準與實際畫出的格數（視窗內）。

    v65：``core_dots`` 記錄雙色規則（黑點 ＝ core ∩ curve、灰點 ＝ core − curve、全部核心格點都畫）與兩色格數；``core_cells_drawn`` ＝ 黑點數 ＋ 灰點數。"""

    black, grey = int(np.count_nonzero(black_drawn)), int(np.count_nonzero(grey_drawn))
    return {
        "basis": HOTSPOT_MASK_BASIS, "curve_mask": HOTSPOT_MASK_NAMES["curve"], "core_mask": HOTSPOT_MASK_NAMES["core"],
        "curve_cells_drawn": int(np.count_nonzero(curve_drawn)), "core_cells_drawn": black + grey,
        "core_dots": {"rule": "core_black_inside_hotspot_grey_outside", "black_colour": HOTSPOT_CORE_DOT_FACE, "grey_colour": HOTSPOT_CORE_OUTSIDE_COLOUR,
                      "black_cells_drawn": black, "grey_cells_drawn": grey},
    }


HOTSPOT_CORE_DOT_PT = 1.2
"""簡要報告聚集熱區圖（5 張）與圖1：核心黑點（``hotspot_core_mask``）的直徑（pt，matplotlib ``markersize``）。
五個地點共用同一個主圖框與格距（約 1.6 pt），所以是固定值。v46（2026-10-27）由 2.0 pt（外加 0.4 pt 白邊，外徑 2.4 pt，比一格還寬）縮小為 1.2 pt、無邊線
（約為一格寬的 3/4、佔格面積約 44%），紅色聚集熱區格不再被黑點蓋住。"""
HOTSPOT_CORE_DOT_EDGE_PT = 0.0
"""核心黑點的邊線寬（pt）；0 ＝ 不畫白邊（舊版 0.4 pt 白邊會多遮住紅格）。"""
HOTSPOT_CORE_DOT_EDGE_COLOUR = "#111111"
HOTSPOT_CORE_DOT_FACE = "#111111"
HOTSPOT_CORE_LEGEND_PT = 3.0
"""圖例的核心黑點符號直徑（pt）：舊版 5.0 pt 依圖上縮小的比例（1.2 / 2.0）同步縮為 3.0 pt，無邊線，與圖上同樣式。"""
HOTSPOT_CORE_LEGEND_EDGE_PT = 0.0


def hotspot_core_dot_kwargs(outside: bool = False) -> dict[str, Any]:
    """地圖上核心點的 ``Axes.plot`` 樣式（簡要報告用；正式圖組不經此函式）。``outside=True`` 為聚集熱區外的灰點（:data:`HOTSPOT_CORE_OUTSIDE_COLOUR`，zorder 略低於黑點）；大小、無邊線相同。"""

    colour = HOTSPOT_CORE_OUTSIDE_COLOUR if outside else HOTSPOT_CORE_DOT_FACE
    return {
        "linestyle": "none", "marker": "o", "markersize": float(HOTSPOT_CORE_DOT_PT), "markerfacecolor": colour,
        "markeredgecolor": colour, "markeredgewidth": float(HOTSPOT_CORE_DOT_EDGE_PT), "zorder": 4.9 if outside else 5,
    }


def hotspot_core_legend_kwargs(outside: bool = False) -> dict[str, Any]:
    """圖例核心點符號的 ``Line2D`` 樣式：填色與圖上相同（``outside=True`` 為灰），大小為 :data:`HOTSPOT_CORE_LEGEND_PT`。"""

    colour = HOTSPOT_CORE_OUTSIDE_COLOUR if outside else HOTSPOT_CORE_DOT_FACE
    return {
        "linestyle": "none", "marker": "o", "markersize": float(HOTSPOT_CORE_LEGEND_PT), "markerfacecolor": colour,
        "markeredgecolor": colour, "markeredgewidth": float(HOTSPOT_CORE_LEGEND_EDGE_PT),
    }


BANNED_FIGURE_TERMS: tuple[str, ...] = ("曲線通過頻率", "熱點", "核心頻率", "研究站")
"""簡要報告 16 張圖的圖內文字不得出現的舊用詞（gate 檢查）。"""

SITE_NAMES_ZH: tuple[str, ...] = ("貢寮", "龜山島", "新竹外海", "南灣", "連江")
"""地點名稱：簡要報告圖內不印地點名稱（由圖說說明地點），只保留錨點星號記號；gate 檢查圖內文字不含這些名稱。"""


def show_site_names() -> bool:
    """圖內是否印地點名稱：簡要報告圖組不印（錨點只畫星號），其餘模式照舊。"""

    return not concise_enabled()


def term(key: str, default: str, **fields: Any) -> str:
    """簡要報告模式回傳 :data:`CONCISE_TERMS` 的用詞（``{percent}`` 等欄位以 ``fields`` 代入）；其餘模式回傳 ``default``。"""

    if not concise_enabled() or key not in CONCISE_TERMS:
        return default
    return CONCISE_TERMS[key].format(**fields)


def display_site_name(site_id: str, default: str) -> str:
    """concise 圖組的站名顯示覆寫（site_id 不變）；未設定時回傳原名稱。"""

    payload = options()
    if payload is None:
        return default
    return str(payload.get("display_names", {}).get(site_id, default))


def shared_limits(kind: str, name: str) -> tuple[float, float] | None:
    """某圖種某色階的跨站共用上下限；未設定時回傳 ``None``。"""

    payload = options()
    if payload is None:
        return None
    value = payload.get("limits", {}).get(kind, {}).get(name)
    if value is None:
        return None
    low, high = (float(item) for item in value)
    if not high > low:
        raise ValueError(f"共用色階上限必須大於下限: {kind}/{name} = {value}")
    return low, high


def shared_ticks(kind: str, name: str) -> list[float] | None:
    """某圖種某色條指定的刻度位置（選項檔 ``ticks``）；未設定時回傳 ``None``，由 :func:`colorbar_tick_values` 自動選。"""

    payload = options()
    if payload is None:
        return None
    value = payload.get("ticks", {}).get(kind, {}).get(name)
    return None if value is None else [float(item) for item in value]


def record_limit(kind: str, site_id: str, name: str, value: Sequence[float]) -> None:
    """把圖「原本」自己的色階附加記錄到 ``TRAPS_CONCISE_RECORD``（JSON Lines）；未設定時不做事。"""

    target = os.environ.get(ENV_RECORD)
    if not target:
        return
    line = json.dumps({"kind": kind, "site_id": site_id, "name": name, "value": [float(item) for item in value]}, ensure_ascii=False)
    with Path(target).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def record_note(kind: str, site_id: str, payload: Mapping[str, Any]) -> None:
    """把圖的選取與視窗說明（純文字／數值）附加記錄到 ``TRAPS_CONCISE_RECORD``；未設定時不做事。"""

    target = os.environ.get(ENV_RECORD)
    if not target:
        return
    line = json.dumps({"kind": kind, "site_id": site_id, "note": dict(payload)}, ensure_ascii=False)
    with Path(target).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


ZOOM_DEFAULTS: dict[str, Any] = {
    "window_km": [78.0, 45.933],
    "min_lifetime_days": 7.0,
    "min_edge_km": [10.0, 5.0],
    "later_fractions": [1.0 / 3.0, 2.0 / 3.0],
    "vector_grid": [15, 10],
    "vector_reference_m_s": 1.0,
    "vector_inch_per_m_s": 0.3,
    "mode": "daily",
}
"""縮放代表圖（``example_zoom``）的預設參數；選項檔的 ``example_zoom`` 物件可逐項覆寫。"""


def example_zoom_options() -> dict[str, Any] | None:
    """縮放代表圖的設定；選項檔沒有 ``example_zoom`` 或其 ``enabled`` 為假時回傳 ``None``（沿用舊版全域代表圖）。"""

    payload = options()
    if payload is None:
        return None
    zoom = payload.get("example_zoom")
    if not zoom or not zoom.get("enabled", True):
        return None
    return {**ZOOM_DEFAULTS, **{key: value for key, value in zoom.items() if key != "enabled"}}


# ---------------------------------------------------------------------------
# 地點錨點符號：黃色星號（與 Lagrangian-Ensemble-Backtracking 的受體符號一致）
# ---------------------------------------------------------------------------


LBT_STAR_CODE_VALUES: dict[str, float] = {"markersize": 13.0, "markeredgewidth": 0.7, "legend_markersize": 12.0}
"""Lagrangian-Ensemble-Backtracking ``native_runner/figures.py`` 的受體符號程式值（``ax.plot(..., marker="*", markersize=13, markerfacecolor="#ffd92f",
markeredgecolor="black", markeredgewidth=0.7, linestyle="none", zorder=8)``；圖例 ``Line2D(marker="*", markersize=12, markerfacecolor="#ffd92f", markeredgecolor="black")``）。"""

LBT_REPORT_CANVAS_WIDTH_IN = 6.27
LBT_FOOTPRINT_OLD_CANVAS_WIDTH_IN = 8.9267
LBT_STAR_PRINT_SCALE = LBT_REPORT_CANVAS_WIDTH_IN / LBT_FOOTPRINT_OLD_CANVAS_WIDTH_IN
"""LBT run09 報告版足跡圖（有星號的地圖）的印出縮放比：報告版把畫布縮為 6.27 吋（Word 文字區寬，插入時不再縮放，``wp:extent`` 6.270 吋 ＝ 畫布寬），
線寬、記號大小、記號邊線寬都乘上 6.27 / 8.9267 ＝ 0.70239（舊足跡圖畫布寬 5356 px / 600 dpi；LBT ``tmp/wp73/new/report_style.py`` 的 ``PRINT_SCALE_FOOTPRINT``，
``scale_print_effect`` 對圖上與圖例的 Line2D 同樣處理；run09 manifest ``print_scale_applied_to_linewidth_and_markers`` ＝ 0.7023872203613877）。
所以 LBT 報告上的星號印出大小 ＝ 程式值 × 此比例（13 → 9.13 pt、邊線 0.7 → 0.49 pt、圖例 12 → 8.43 pt）。
TRAPs 的圖畫布 6.3 吋、以原尺寸放進報告（不縮放），所以直接用印出值作圖內大小，印在報告上就與 LBT 同大。"""

SITE_STAR_STYLE: dict[str, Any] = {
    "marker": "*",
    "markersize": LBT_STAR_CODE_VALUES["markersize"] * LBT_STAR_PRINT_SCALE,
    "markerfacecolor": "#ffd92f",
    "markeredgecolor": "black",
    "markeredgewidth": LBT_STAR_CODE_VALUES["markeredgewidth"] * LBT_STAR_PRINT_SCALE,
    "linestyle": "none",
    "zorder": 8,
}
"""圖上地點錨點的畫法：形狀、填色、邊色、zorder 逐項取自 LBT 受體符號（見 :data:`LBT_STAR_CODE_VALUES`）；markersize 與邊線寬是 LBT 程式值乘
:data:`LBT_STAR_PRINT_SCALE`（LBT 印在報告上的實際大小，約 9.13 pt、0.49 pt）。zorder 8 高於熱區色塊、黑點與海岸線。"""

SITE_STAR_LEGEND_MARKERSIZE = LBT_STAR_CODE_VALUES["legend_markersize"] * LBT_STAR_PRINT_SCALE
"""圖例中的星號大小：LBT 圖例 ``markersize=12`` 乘 :data:`LBT_STAR_PRINT_SCALE`（約 8.43 pt）；填色與邊線同圖上。"""

CROSS_MARKERS: tuple[str, ...] = ("+", "P", "x", "X")
"""十字類符號：簡要報告 16 張圖的地圖與圖例都不得再出現（gate 檢查）。"""


def draw_site_star(axis: Any, lon: float, lat: float) -> Any:
    """在地圖軸畫地點錨點星號（:data:`SITE_STAR_STYLE`）；``clip_on=False`` 讓靠近框線的星號不被軸框裁掉。
    同一位置已有星號時不重複畫（A 區總覽的兩站錨點，其中一個已由 ``_prepare_map_axis`` 畫過）。"""

    for line in axis.get_lines():
        if str(line.get_marker()) == SITE_STAR_STYLE["marker"] and len(line.get_xdata()) == 1 and float(line.get_xdata()[0]) == float(lon) and float(line.get_ydata()[0]) == float(lat):
            return line
    (line,) = axis.plot([lon], [lat], clip_on=False, **SITE_STAR_STYLE)
    return line


def site_star_handle(label: str) -> Any:
    """圖例用的星號 handle（``markersize`` 為 :data:`SITE_STAR_LEGEND_MARKERSIZE`，填色與邊線同 :data:`SITE_STAR_STYLE`）。"""

    from matplotlib.lines import Line2D

    return Line2D(
        [], [], linestyle="none", marker=SITE_STAR_STYLE["marker"], markersize=SITE_STAR_LEGEND_MARKERSIZE,
        markerfacecolor=SITE_STAR_STYLE["markerfacecolor"], markeredgecolor=SITE_STAR_STYLE["markeredgecolor"],
        markeredgewidth=SITE_STAR_STYLE["markeredgewidth"], label=label,
    )


# ---------------------------------------------------------------------------
# 地圖座標軸：規則的經緯度刻度
# ---------------------------------------------------------------------------


GRID_DEG = 0.05
"""軸範圍兩端（也是兩個必標的刻度）的格線：0.05 度。"""

MIN_AXIS_LABELS = 4
MAX_AXIS_LABELS = 7
"""每軸標示刻度數（含兩端）的範圍（4–7）。"""

END_CLEARANCE_FRACTION = 0.4
"""內部刻度與端點標示的最小距離（間距的 40%）；更近的內部刻度不標示，避免與端點標示擠在一起。"""

LABEL_SPACING_INCH = {"lon": 0.46, "lat": 0.17}
"""相鄰標示中心的最小圖面距離（英吋）：經度標示（8 pt、至多 2 位小數，例 ``121.35``）約 0.37 英吋寬；緯度標示是 0.11 英吋高。"""

LON_AXIS_TITLE = "經度 (°E)"
LAT_AXIS_TITLE = "緯度 (°N)"
"""座標軸標題（單位在標題內，刻度標示是純數字）。"""

AXIS_LABEL_DECIMALS = 1
"""全部地圖的經緯度標示一律 1 位小數（例如 121.3、24.6），不依刻度值改變位數。圖框邊界與刻度都落在 0.1° 的整數倍上（見 :data:`FRAME_SPECS`）。"""

AXIS_LABEL_MAX_DECIMALS = AXIS_LABEL_DECIMALS
"""舊名稱（相容）：標示的小數位數。"""

FRAME_GRID_DEG = 0.1
"""標準圖框四邊所在的格線：0.1°（經緯度刻度標示因此一律 1 位小數）。"""

MAP_EXTENTS_SOURCE = (
    "Lagrangian-Ensemble-Backtracking outputs/成果報告_run09/_verify/axes_unified_run09.txt（LBT run09 axes_unified；README「2026-10-22 n 版」）"
)
"""大圖地圖範圍與刻度的來源：LBT run09 的 ``axes_unified_run09.txt``（水平軌跡總覽與條件式來源足跡兩圖種相同）。"""

MAP_REFERENCE_LAT_DEG = 24.1
"""大圖主圖框長寬比的參考緯度（LBT run09 axes_unified 的做法）：一度緯度的長度 ＝ 1／cos(24.1°) 倍一度經度長度；五個地點共用同一個主圖框。"""

FRAME_SPECS: dict[str, dict[str, Any]] = {
    "map": {
        "lon_span": 1.8, "lat_span": 1.2, "lon_step": 0.2, "lat_step": 0.2,
        "source": MAP_EXTENTS_SOURCE,
        "reference_lat_deg": MAP_REFERENCE_LAT_DEG,
        # 各地點（以 flow domain 的分析區代號為鍵）的指定範圍：經度跨度 1.8°、緯度跨度 1.2°，上下限即刻度（引用 LBT run09 axes_unified，不依 flow domain 外框另行計算）。
        "extents": {
            "A": {"sites": ["gongliao", "guishan", "domainA"], "xlim": [121.2, 123.0], "ylim": [24.4, 25.6]},  # 貢寮、龜山島、圖1（海域 A 總覽）
            "B": {"sites": ["hsinchu"], "xlim": [119.6, 121.4], "ylim": [24.2, 25.4]},  # 新竹外海
            "C": {"sites": ["nanwan"], "xlim": [120.0, 121.8], "ylim": [21.4, 22.6]},  # 南灣
            "D": {"sites": ["lienchiang"], "xlim": [119.0, 120.8], "ylim": [25.6, 26.8]},  # 連江
        },
    },
    "zoom": {"lon_span": 0.8, "lat_span": 0.5, "lon_step": 0.2, "lat_step": 0.1},
}
"""標準圖框規格表：同一類圖五個地點的經緯度跨度與刻度間距完全相同。

* ``map``（2026-10-22 第二輪，使用者指定改採 LBT 規格）：flow domain 範圍的大圖（A 區總覽、聚集熱區、四季重現，共 11 張）：經度 1.8°、緯度 1.2°，
  經緯度刻度間距都是 0.2°（經度 10 個、緯度 7 個刻度），上下限就是刻度並標出數值；各地點的範圍直接指定在 ``extents``（來源：:data:`MAP_EXTENTS_SOURCE`），
  不再用上一輪的「向外擴到 0.1° 格線並包住 flow domain 範圍」規則。流場範圍落在圖框內，圖框其餘部分只畫陸地與海岸線。
  主圖框的長寬比以 :data:`MAP_REFERENCE_LAT_DEG` 計算（見 :data:`traps.figure_standard.CONCISE_MAP_AXES`），長寬比誤差上限 :data:`MAP_FRAME_ASPECT_TOLERANCE`（±2%）。
* ``zoom``：代表性 TRAP 放大圖：經度 0.8°（每 0.2° 一格，5 個刻度）、緯度 0.5°（每 0.1° 一格，6 個刻度）；圖框四邊落在 :data:`FRAME_GRID_DEG` 的整數倍上並完整包住原視窗（:func:`standard_frame`）。
  2026-10-22 第三輪：主圖框改為 :data:`traps.figure_standard.CONCISE_ZOOM_AXES`（4.352 × 2.98 英吋，同樣以 :data:`MAP_REFERENCE_LAT_DEG` 計算長寬比），長寬比誤差上限 :data:`ZOOM_FRAME_ASPECT_TOLERANCE`（±2%）。

首末刻度就是框邊，刻度從下／左緣起每隔一個間距一個（:func:`axis_ticks`）。刻度與圖框的數字只來自此表，各圖不自行寫死。"""

FRAME_TIE_TOLERANCE_DEG = 1.0e-3
"""兩種取框方式的最小留白差小於此值（約 110 m，遠小於格距）視為平手，改往東、往北擴。"""

STANDARD_FRAME_ASPECT_TOLERANCE = 0.05
"""不在規格表內的圖框（備援取框、舊版圖組）的長寬比誤差上限（經度 × cos 緯度 ／ 緯度，相對於軸框；見 :func:`frame_aspect_error`）。
規格表內的圖框（``map``、``zoom``）各有自己的上限（:data:`MAP_FRAME_ASPECT_TOLERANCE`、:data:`ZOOM_FRAME_ASPECT_TOLERANCE`），皆為 ±2%。"""

ZOOM_FRAME_ASPECT_TOLERANCE = 0.02
"""放大圖（``zoom``，2026-10-22 第三輪改採與大圖相同的 24.1° 參考緯度主圖框 :data:`traps.figure_standard.CONCISE_ZOOM_AXES`，4.352 × 2.98 英吋）的長寬比誤差上限：±2%
（原軸框 4.5 × 2.98 英吋時誤差 −1.7% 至 −4.9%，上限曾放寬為 5%；改用 LBT 長寬比的主圖框後與大圖同為 2%）。"""

MAP_FRAME_ASPECT_TOLERANCE = 0.02
"""大圖（``map``，LBT 規格）的長寬比誤差上限：±2%（上一輪因軸框 4.5 × 2.98 與圖框跨度不合而放寬到 5%；改採 LBT 的主圖框後改回 2%）。"""


def frame_aspect_tolerance(name: str) -> float:
    """標準圖框規格 ``name`` 的長寬比誤差上限：``"map"``、``"zoom"`` 皆 ±2%；規格表以外（``"other"``）5%。"""

    if name == "map":
        return MAP_FRAME_ASPECT_TOLERANCE
    if name == "zoom":
        return ZOOM_FRAME_ASPECT_TOLERANCE
    return STANDARD_FRAME_ASPECT_TOLERANCE


def frame_spec_name_of(xlim: Sequence[float], ylim: Sequence[float]) -> str | None:
    """依經緯度跨度辨識軸範圍屬於哪個標準圖框規格（``"map"``、``"zoom"``）；不在規格表內時回傳 ``None``。"""

    span_x, span_y = abs(float(xlim[1]) - float(xlim[0])), abs(float(ylim[1]) - float(ylim[0]))
    for name, spec in FRAME_SPECS.items():
        if abs(float(spec["lon_span"]) - span_x) <= 1.0e-6 and abs(float(spec["lat_span"]) - span_y) <= 1.0e-6:
            return name
    return None


def round_to_grid(value: float, grid: float = GRID_DEG) -> float:
    """四捨五入到 ``grid`` 度（預設 0.05 度）的整數倍。"""

    return round(round(float(value) / grid) * grid, 6)


def domain_limits(lon: Sequence[float], lat: Sequence[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """站點圖的地圖範圍：flow domain 網格範圍（經緯度軸的最小與最大值）各自四捨五入到最近的 0.05 度。"""

    lon_arr, lat_arr = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    return apply_domain_limit_override(
        (
            (round_to_grid(float(lon_arr.min())), round_to_grid(float(lon_arr.max()))),
            (round_to_grid(float(lat_arr.min())), round_to_grid(float(lat_arr.max()))),
        )
    )


DOMAIN_LIMIT_OVERRIDES: dict[tuple[tuple[float, float], tuple[float, float]], dict[str, Any]] = {
    ((120.15, 121.6), (21.55, 22.45)): {
        "limits": ((120.15, 121.65), (21.55, 22.45)),
        "reason": "keep the full flow domain and even ticks",
    },
}
"""四捨五入後的 flow domain 範圍 → 改用的範圍與原因。C（南灣）四捨五入後是 120.15–121.60（右緣切掉 2 欄資料、刻度間距不等），
改 120.15–121.65：涵蓋整個 domain，經度刻度每 0.3°（120.15、120.45 … 121.65）；長寬比誤差 +2.3%（仍 ≤ 3%）。"""


def apply_domain_limit_override(limits: tuple[tuple[float, float], tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """四捨五入後的範圍若在 :data:`DOMAIN_LIMIT_OVERRIDES` 內就換成指定範圍（記錄原因於版面紀錄 ``extent_info.override``）。"""

    key = ((float(limits[0][0]), float(limits[0][1])), (float(limits[1][0]), float(limits[1][1])))
    for rounded, item in DOMAIN_LIMIT_OVERRIDES.items():
        if all(abs(a - b) < 1.0e-6 for pair_a, pair_b in zip(key, rounded, strict=True) for a, b in zip(pair_a, pair_b, strict=True)):
            return item["limits"]
    return key


def domain_limit_override_reason(limits: tuple[tuple[float, float], tuple[float, float]]) -> str | None:
    """``limits`` 是某個覆寫後的範圍時回傳原因，否則 ``None``。"""

    for item in DOMAIN_LIMIT_OVERRIDES.values():
        if all(abs(a - b) < 1.0e-6 for pair_a, pair_b in zip(limits, item["limits"], strict=True) for a, b in zip(pair_a, pair_b, strict=True)):
            return str(item["reason"])
    return None


def frame_lows(base_low: float, base_high: float, span: float, grid: float = FRAME_GRID_DEG) -> list[float]:
    """跨度為 ``span``、下限落在 ``grid`` 整數倍上，並完整包住 ``[base_low, base_high]`` 的所有圖框下限（由小到大）。"""

    first = math.ceil((float(base_high) - float(span)) / grid - 1.0e-9)
    last = math.floor(float(base_low) / grid + 1.0e-9)
    return [round(index * grid, 6) for index in range(first, last + 1)]


def standard_frame(
    name: str,
    base_xlim: Sequence[float],
    base_ylim: Sequence[float],
    grid_bbox: Sequence[float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
    """標準圖框（規格表 :data:`FRAME_SPECS` 的 ``name``）：往外擴到規格跨度，四邊落在 :data:`FRAME_GRID_DEG` 的整數倍上並完整包住原圖框。

    ``base_xlim``／``base_ylim`` 是原圖框（flow domain 範圍四捨五入到 0.05°，或原本的代表圖視窗）。可行的取法常有兩種（原圖框的邊已在 0.1° 格線上時，
    往西／東或往南／北多擴一格）：取「流場資料外框（``grid_bbox`` ＝ ``(lon_min, lon_max, lat_min, lat_max)``，有效格點邊界；
    預設為原圖框）距新框四邊的最小留白」較大者；最小留白差在 :data:`FRAME_TIE_TOLERANCE_DEG` 以內視為平手，往東、往北擴。
    回傳 ``(xlim, ylim, 說明)``；說明記錄原圖框、最小留白（負值表示資料外框略超出新框）與各方向的可行下限。
    """

    spec = FRAME_SPECS[name]
    if "extents" in spec:
        raise ValueError(f"標準圖框「{name}」的範圍是直接指定的（FRAME_SPECS[{name!r}]['extents']，見 map_frame），不適用「包住原圖框」規則")
    base_x = (float(min(base_xlim)), float(max(base_xlim)))
    base_y = (float(min(base_ylim)), float(max(base_ylim)))
    valid = tuple(float(v) for v in grid_bbox) if grid_bbox is not None else (base_x[0], base_x[1], base_y[0], base_y[1])
    span_x, span_y = float(spec["lon_span"]), float(spec["lat_span"])
    x_lows = frame_lows(base_x[0], base_x[1], span_x)
    y_lows = frame_lows(base_y[0], base_y[1], span_y)
    if not x_lows or not y_lows:
        raise ValueError(
            f"原圖框 {base_x}×{base_y}（跨度 {base_x[1] - base_x[0]:.3f}°×{base_y[1] - base_y[0]:.3f}°）超過標準圖框「{name}」的跨度 {span_x}°×{span_y}°"
        )
    best: tuple[float, float, float] | None = None
    for low_x in reversed(x_lows):  # 平手時先往東、再往北：由東／北的取法開始，必須明顯較大才換
        for low_y in reversed(y_lows):
            clearance = min(valid[0] - low_x, low_x + span_x - valid[1], valid[2] - low_y, low_y + span_y - valid[3])
            if best is None or clearance > best[0] + FRAME_TIE_TOLERANCE_DEG:
                best = (clearance, low_x, low_y)
    assert best is not None
    clearance, low_x, low_y = best
    xlim = (round(low_x, 6), round(low_x + span_x, 6))
    ylim = (round(low_y, 6), round(low_y + span_y, 6))
    info: dict[str, Any] = {
        "kind": "standard_frame", "spec": name, "base_xlim": list(base_x), "base_ylim": list(base_y), "grid_bbox": [round(v, 6) for v in valid],
        "min_clearance_deg": round(clearance, 6), "x_lows_considered": x_lows, "y_lows_considered": y_lows,
        "contains_base": bool(xlim[0] <= base_x[0] + 1.0e-9 and xlim[1] >= base_x[1] - 1.0e-9 and ylim[0] <= base_y[0] + 1.0e-9 and ylim[1] >= base_y[1] - 1.0e-9),
    }
    return xlim, ylim, info


def centred_frame(name: str, xlim: Sequence[float], ylim: Sequence[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """規格表 ``name`` 跨度的標準圖框，中心取最接近 ``xlim``／``ylim`` 中心的 0.1° 格線。不保證包住 ``xlim``／``ylim``；
    只用於沒有選取視窗（沒有原視窗可包住）的後備路徑，正式圖組的視窗一律走 :func:`standard_frame`。"""

    spec = FRAME_SPECS[name]
    out = []
    for lim, key in ((xlim, "lon_span"), (ylim, "lat_span")):
        span = float(spec[key])
        low = round((0.5 * (float(min(lim)) + float(max(lim))) - 0.5 * span) / FRAME_GRID_DEG) * FRAME_GRID_DEG
        out.append((round(low, 6), round(low + span, 6)))
    return out[0], out[1]


MAP_EXTENT_CONTAIN_TOLERANCE_DEG = 1.0e-6


def map_region_of(lon: Sequence[float], lat: Sequence[float]) -> str:
    """格點範圍（流場網格的經緯度）所屬的大圖範圍代號（``FRAME_SPECS["map"]["extents"]`` 的鍵）：指定範圍完整包住格點範圍者；找不到或不唯一時 ``ValueError``。"""

    lon_arr, lat_arr = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    low_x, high_x, low_y, high_y = float(lon_arr.min()), float(lon_arr.max()), float(lat_arr.min()), float(lat_arr.max())
    tol = MAP_EXTENT_CONTAIN_TOLERANCE_DEG
    hits = [
        key for key, item in FRAME_SPECS["map"]["extents"].items()
        if item["xlim"][0] <= low_x + tol and item["xlim"][1] >= high_x - tol and item["ylim"][0] <= low_y + tol and item["ylim"][1] >= high_y - tol
    ]
    if len(hits) != 1:
        raise ValueError(f"格點範圍 {low_x:.4f}–{high_x:.4f}°E × {low_y:.4f}–{high_y:.4f}°N 對不到唯一的指定大圖範圍（符合者 {hits}）")
    return hits[0]


def map_frame(
    lon: Sequence[float], lat: Sequence[float], region: str | None = None
) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
    """flow domain 大圖的標準圖框（規格 ``map``）：直接取 :data:`FRAME_SPECS` 指定的各地點範圍（LBT run09 axes_unified），不再依格點範圍計算。

    ``region`` 為範圍代號（``"A"``–``"D"``）；省略時由格點範圍辨識（:func:`map_region_of`，指定範圍須完整包住格點範圍）。
    說明記錄來源、代號、格點範圍外框與圖框四邊超出格點範圍的度數（正值＝圖框超出流場範圍，該側只畫陸地與海岸線）。
    """

    lon_arr, lat_arr = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    key = region if region is not None else map_region_of(lon_arr, lat_arr)
    item = FRAME_SPECS["map"]["extents"][key]
    xlim = (float(item["xlim"][0]), float(item["xlim"][1]))
    ylim = (float(item["ylim"][0]), float(item["ylim"][1]))
    grid_bbox = (float(lon_arr.min()), float(lon_arr.max()), float(lat_arr.min()), float(lat_arr.max()))
    info: dict[str, Any] = {
        "kind": "lbt_specified_extent", "spec": "map", "region": key, "sites": list(item["sites"]), "source": MAP_EXTENTS_SOURCE,
        "reference_lat_deg": MAP_REFERENCE_LAT_DEG, "grid_bbox": [round(v, 6) for v in grid_bbox],
        "beyond_grid_deg": {
            "west": round(grid_bbox[0] - xlim[0], 6), "east": round(xlim[1] - grid_bbox[1], 6),
            "south": round(grid_bbox[2] - ylim[0], 6), "north": round(ylim[1] - grid_bbox[3], 6),
        },
        "contains_grid": bool(xlim[0] <= grid_bbox[0] and xlim[1] >= grid_bbox[1] and ylim[0] <= grid_bbox[2] and ylim[1] >= grid_bbox[3]),
    }
    return xlim, ylim, info


def frame_aspect_error(xlim: Sequence[float], ylim: Sequence[float], box_w: float, box_h: float) -> float:
    """軸範圍的長寬比相對軸框的誤差：(經度跨度 × cos 緯度中心 / 緯度跨度) / (軸框寬 / 軸框高) − 1。"""

    scale = max(math.cos(math.radians(0.5 * (float(ylim[0]) + float(ylim[1])))), 0.05)
    ratio = (float(xlim[1]) - float(xlim[0])) * scale / max(float(ylim[1]) - float(ylim[0]), 1.0e-12)
    return ratio / (float(box_w) / float(box_h)) - 1.0


def _on_step(value: float, step: float) -> bool:
    ratio = value / step
    return abs(ratio - round(ratio)) <= 1.0e-6


def tick_step(kind: str, span: float) -> float | None:
    """跨度 ``span`` 的軸（``kind`` 為 ``"lon"`` 或 ``"lat"``）在規格表 :data:`FRAME_SPECS` 內的刻度間距；跨度不在規格表時回傳 ``None``。"""

    key = "lon" if kind == "lon" else "lat"
    for spec in FRAME_SPECS.values():
        if abs(float(spec[f"{key}_span"]) - float(span)) <= 1.0e-6:
            return float(spec[f"{key}_step"])
    return None


def expected_tick_count(kind: str, span: float) -> int | None:
    """跨度 ``span`` 的軸在規格表內應有的標示刻度數（跨度／間距＋1，例如大圖經度 10、緯度 7；放大圖經度 5、緯度 6）；跨度不在規格表時回傳 ``None``。"""

    step = tick_step(kind, span)
    return None if step is None else int(round(float(span) / step)) + 1


def axis_ticks(low: float, high: float, kind: str, length_inch: float | None = None) -> list[float]:
    """地圖軸（``kind`` 為 ``"lon"`` 或 ``"lat"``）的標示刻度：從下限起每隔規格表間距一個，首個刻度＝下限、末個刻度＝上限（框邊都有標值）。

    間距只由軸跨度查規格表 :data:`FRAME_SPECS`（大圖：經度 1.8° → 0.2°、緯度 1.2° → 0.2°；放大圖：經度 0.8° → 0.2°、緯度 0.5° → 0.1°）；
    軸範圍必須是標準圖框（跨度在規格表內、下限落在 :data:`FRAME_GRID_DEG` 的整數倍上），否則 ``ValueError``（不再自行挑間距）。
    ``length_inch`` 只為相容舊呼叫，不影響結果。
    """

    low, high = (float(low), float(high)) if low <= high else (float(high), float(low))
    span = high - low
    step = tick_step(kind, span)
    if step is None:
        raise ValueError(f"{kind} 軸跨度 {span:.4f}° 不在標準圖框規格表 FRAME_SPECS 內，無法決定刻度間距")
    if not _on_step(low, FRAME_GRID_DEG):
        raise ValueError(f"{kind} 軸下限 {low} 不是 {FRAME_GRID_DEG}° 的整數倍（標示一律 1 位小數）")
    count = int(round(span / step)) + 1
    return [round(low + index * step, 6) for index in range(count)]


FRAME_ASPECT_TOLERANCE = 0.03
"""軸範圍的長寬比相對於軸框長寬比（經度 × cos(緯度) 對緯度）的誤差上限（3%）。"""

SNAP_ASPECT_TOLERANCE = STANDARD_FRAME_ASPECT_TOLERANCE

ASPECT_ERROR_WEIGHT = 3.0
"""長寬比誤差的加權：誤差 1% 相當於面積多 3%（面積是主要目標，誤差小者在面積相近時優先）。"""

CENTRING_WEIGHT = 0.5
"""資料中心偏離範圍中心（相對於跨度）的加權：面積相近時讓資料置中。"""

_EXTENT_INFO: dict[tuple[float, float, float, float], dict[str, Any]] = {}
""":func:`fit_limits_to_frame` 的說明（以範圍為鍵），供 :func:`concise_checks_of` 併入版面紀錄（資料外接框、外擴比例）。"""


def fit_limits_to_frame(
    xlim: Sequence[float], ylim: Sequence[float], box_w: float, box_h: float, *, tolerance: float = FRAME_ASPECT_TOLERANCE, coverage: Any = None
) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
    """資料外接框 → 軸範圍：只往「補足固定軸框長寬比所需的那一軸」對稱外擴，再向外取整到 0.05 度；資料置中。

    候選範圍的兩端都在 0.05 度格線上、覆蓋外接框、長寬比誤差 ≤ ``tolerance``（相對於軸框 ``box_w/box_h``，經度已乘 cos 緯度）；
    目標＝面積 ×（1 ＋ ``ASPECT_ERROR_WEIGHT`` × |誤差| ＋ ``CENTRING_WEIGHT`` × 資料中心偏離），即在長寬比正確的前提下面積最小、資料最置中。
    不搜尋刻度間距（刻度見 :func:`axis_ticks`）。回傳 ``(xlim, ylim, 說明)``；說明含 ``area_growth``（相對外接框面積）、``aspect_error``、
    ``data_bbox``；給定 ``coverage``（:class:`traps.extent_coverage.Coverage`）時另含 ``fractions``（資料／陸地／空白比例）。
    """

    x_low, x_high = float(min(xlim)), float(max(xlim))
    y_low, y_high = float(min(ylim)), float(max(ylim))
    target = float(box_w) / float(box_h)
    data_w, data_h = max(x_high - x_low, 1.0e-9), max(y_high - y_low, 1.0e-9)
    centre_x, centre_y = 0.5 * (x_low + x_high), 0.5 * (y_low + y_high)
    base_area = data_w * data_h
    # 理想（精確長寬比）的跨度：只擴張需要的那一軸
    scale = max(math.cos(math.radians(centre_y)), 0.05)
    ideal_w, ideal_h = data_w, data_h
    if data_w * scale / data_h < target:
        ideal_w = target * data_h / scale
    else:
        ideal_h = data_w * scale / target
    min_w, min_h = int(math.ceil(data_w / GRID_DEG - 1.0e-9)), int(math.ceil(data_h / GRID_DEG - 1.0e-9))

    def candidates(low: float, high: float, count: int) -> list[float]:
        """跨度為 ``count`` 格的起點（0.05 度格線上、覆蓋 [low, high]）。"""

        first = math.ceil((high - count * GRID_DEG) / GRID_DEG - 1.0e-9)
        last = math.floor(low / GRID_DEG + 1.0e-9)
        return [round(index * GRID_DEG, 6) for index in range(first, last + 1)]

    best: tuple[float, tuple[float, float, float, float, float]] | None = None
    reach = int(math.ceil(max(ideal_w, ideal_h) / GRID_DEG)) + 4
    for count_w in range(min_w, min_w + reach):
        span_w = count_w * GRID_DEG
        for count_h in range(min_h, min_h + reach):
            span_h = count_h * GRID_DEG
            error = span_w * scale / span_h / target - 1.0
            if abs(error) > tolerance:
                continue
            for origin_x in candidates(x_low, x_high, count_w):
                offset_x = abs(origin_x + 0.5 * span_w - centre_x) / span_w
                for origin_y in candidates(y_low, y_high, count_h):
                    offset_y = abs(origin_y + 0.5 * span_h - centre_y) / span_h
                    cost = span_w * span_h * (1.0 + ASPECT_ERROR_WEIGHT * abs(error) + CENTRING_WEIGHT * (offset_x + offset_y))
                    if best is None or cost < best[0]:
                        best = (cost, (origin_x, origin_x + span_w, origin_y, origin_y + span_h, error))
    if best is None:
        raise ValueError("找不到長寬比誤差在上限內的地圖範圍")
    xl, xh, yl, yh, error = (round(value, 6) for value in best[1])
    info: dict[str, Any] = {
        "area_growth": (xh - xl) * (yh - yl) / base_area - 1.0, "aspect_error": error, "within_tolerance": True,
        "tolerance": tolerance, "data_bbox": [x_low, x_high, y_low, y_high],
        "data_offset_deg": [centre_x - 0.5 * (xl + xh), centre_y - 0.5 * (yl + yh)],
    }
    if coverage is not None:
        info["fractions"] = coverage.fractions((xl, xh), (yl, yh))
    _EXTENT_INFO[(xl, xh, yl, yh)] = dict(info)
    return (xl, xh), (yl, yh), info


def record_extent_info(xlim: Sequence[float], ylim: Sequence[float], info: Mapping[str, Any]) -> None:
    """登記一組軸範圍的說明（以範圍為鍵），供 :func:`concise_checks_of` 併入版面紀錄。"""

    _EXTENT_INFO[(round(float(xlim[0]), 6), round(float(xlim[1]), 6), round(float(ylim[0]), 6), round(float(ylim[1]), 6))] = dict(info)


def update_extent_info(xlim: Sequence[float], ylim: Sequence[float], **fields: Any) -> None:
    """補充 :func:`fit_limits_to_frame` 登記的說明（例如資料／陸地／空白比例）。"""

    key = (round(float(xlim[0]), 6), round(float(xlim[1]), 6), round(float(ylim[0]), 6), round(float(ylim[1]), 6))
    _EXTENT_INFO.setdefault(key, {}).update(fields)


def exact_aspect(xlim: Sequence[float], ylim: Sequence[float], box_w: float, box_h: float) -> float:
    """讓 ``adjustable="datalim"`` 不再改動範圍的座標軸 ``aspect``（資料單位的顯示高度 / 寬度 = (軸框高 / 緯度跨度) / (軸框寬 / 經度跨度)）。"""

    span_x = max(float(xlim[1]) - float(xlim[0]), 1.0e-12)
    span_y = max(float(ylim[1]) - float(ylim[0]), 1.0e-12)
    return (float(box_h) / span_y) / (float(box_w) / span_x)


def decimals_for(*tick_groups: Sequence[float]) -> int:
    """座標軸標示的小數位數：全部地圖一律 :data:`AXIS_LABEL_DECIMALS`（2）位，與刻度值無關。"""

    return AXIS_LABEL_DECIMALS


def degree_label(value: float, kind: str = "lon", decimals: int = AXIS_LABEL_DECIMALS) -> str:
    """座標軸標示：純數字（單位寫在座標軸標題），例如 ``121.30``、``24.75``；一律 2 位小數。"""

    return f"{value:.{decimals}f}"


def style_map_axes(axis: Any, xlim: Sequence[float], ylim: Sequence[float]) -> None:
    """地圖座標軸：兩端標示加規則間距的內部刻度（度數格式，2 位小數）、框線上的刻度記號、清楚的框線；字級 8 pt。"""

    from matplotlib.ticker import FixedLocator, FuncFormatter

    position = axis.get_position()
    fig_w, fig_h = axis.figure.get_size_inches()
    x_ticks = axis_ticks(xlim[0], xlim[1], "lon", float(position.width * fig_w))
    y_ticks = axis_ticks(ylim[0], ylim[1], "lat", float(position.height * fig_h))
    x_decimals, y_decimals = decimals_for(x_ticks), decimals_for(y_ticks)
    axis.xaxis.set_major_locator(FixedLocator(x_ticks))
    axis.yaxis.set_major_locator(FixedLocator(y_ticks))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: degree_label(value, "lon", x_decimals)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: degree_label(value, "lat", y_decimals)))
    axis.minorticks_off()
    axis.tick_params(
        axis="both", which="major", direction="out", length=TICK_LENGTH_PT, width=0.8, color=FRAME_COLOUR,
        labelsize=8, pad=2, bottom=True, left=True, top=False, right=False, labelbottom=True, labelleft=True,
    )
    axis.set_xlabel(LON_AXIS_TITLE, fontsize=8, labelpad=2)
    axis.set_ylabel(LAT_AXIS_TITLE, fontsize=8, labelpad=2)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LINEWIDTH)
        spine.set_edgecolor(FRAME_COLOUR)
        spine.set_zorder(20)
    axis._traps_role = "map"  # noqa: SLF001  供 vector_figures 的版面紀錄讀取


# ---------------------------------------------------------------------------
# 色條：共用上下限、端點刻度、延伸箭頭
# ---------------------------------------------------------------------------


def extend_for(values: np.ndarray, vmin: float, vmax: float) -> str:
    """資料超出共用色階時的延伸箭頭：``neither``／``min``／``max``／``both``。"""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "neither"
    below = float(finite.min()) < vmin - 1.0e-12
    above = float(finite.max()) > vmax + 1.0e-12
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def clipped_fraction(values: np.ndarray, vmin: float, vmax: float) -> float:
    """超出色階的有限格點比例（只作為 manifest 記錄）。"""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.mean((finite < vmin) | (finite > vmax)))


def _decimals_needed(value: float, limit: int = 4) -> int:
    for digits in range(limit + 1):
        if abs(round(value, digits) - value) <= 1.0e-9 * max(1.0, abs(value)):
            return digits
    return limit


def colorbar_tick_values(vmin: float, vmax: float) -> list[float]:
    """色條刻度：兩端點加上規則間距的內部刻度（內部刻度離端點至少 10% 全距）。"""

    span = vmax - vmin
    margin = 0.10 * span
    best: tuple[float, float, list[float]] | None = None
    for exponent in range(-6, 7):
        for mantissa, penalty in ((1.0, 0.0), (2.0, 0.0), (5.0, 0.0), (2.5, 0.5), (4.0, 0.2)):
            step = mantissa * 10.0**exponent
            first = math.ceil((vmin + margin) / step - 1.0e-9)
            last = math.floor((vmax - margin) / step + 1.0e-9)
            if not 2 <= last - first + 1 <= 6:
                continue
            interior = [round(index * step, 9) for index in range(first, last + 1)]
            score = abs(len(interior) - 4) + penalty
            if all(abs(edge / step - round(edge / step)) < 1.0e-9 for edge in (vmin, vmax)):
                score -= 0.5  # 兩端點都落在刻度間距的整數倍上：整條色條刻度等距
            if best is None or score < best[0] - 1.0e-12:
                best = (score, step, interior)
    interior = best[2] if best is not None else []
    return [float(vmin), *interior, float(vmax)]


def colorbar_decimals(ticks: Sequence[float]) -> int:
    """標示小數位數：能同時精確表示全部刻度（含兩端點）的最小位數。"""

    return max(_decimals_needed(float(value)) for value in ticks)


def format_tick(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"
    return text.replace("-", MINUS)


def style_colorbar_shared(
    colorbar: Any,
    vmin: float,
    vmax: float,
    label: str,
    *,
    name: str,
    extend: str = "neither",
    clipped: float = 0.0,
    tick_values: Sequence[float] | None = None,
) -> list[float]:
    """色條：端點與規則內部刻度皆標示、字級 8 pt；登記 vmin／vmax／延伸供 gate 與 manifest 使用。"""

    from matplotlib.ticker import FixedLocator, FuncFormatter

    # matplotlib 預設會在有延伸箭頭時把色條 axes 縮小以容納箭頭；改回固定軸框，讓箭頭畫在軸框外，
    # 這樣有無延伸箭頭的圖，色條本體的軸框完全相同（統一圖面標準的 gate 才能比對）。
    colorbar.ax.set_axes_locator(None)
    if tick_values is None:
        ticks = colorbar_tick_values(vmin, vmax)
    else:
        ticks = [float(value) for value in tick_values]
        if abs(ticks[0] - vmin) > 1.0e-9 or abs(ticks[-1] - vmax) > 1.0e-9 or sorted(ticks) != ticks:
            raise ValueError(f"指定刻度必須由小到大且兩端等於色階端點: {ticks} vs [{vmin}, {vmax}]")
    decimals = colorbar_decimals(ticks)
    colorbar.locator = FixedLocator(ticks)
    colorbar.formatter = FuncFormatter(lambda value, _pos: format_tick(value, decimals))
    colorbar.update_ticks()
    colorbar.minorticks_off()
    colorbar.set_label(label, fontsize=8, labelpad=2)
    colorbar.ax.tick_params(labelsize=8, pad=1, length=3.0, width=0.8)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LINEWIDTH)
        spine.set_edgecolor(FRAME_COLOUR)
    colorbar.ax._traps_role = "colourbar"  # noqa: SLF001
    colorbar.ax._traps_colourbar = {  # noqa: SLF001
        "name": name, "vmin": float(vmin), "vmax": float(vmax), "extend": extend, "label": label, "clipped_fraction": float(clipped),
    }
    return ticks


def style_colorbar_discrete(colorbar: Any, classes: Sequence[int], label: str, *, name: str) -> list[int]:
    """離散色條（整數級別）：色塊邊界在級別 ±0.5，刻度標在每一級中央；兩端平整、字級 8 pt。

    登記 ``discrete=True`` 與 ``classes``，供 gate 檢查刻度恰為各級整數（端點／內部刻度規則不適用）。
    """

    from matplotlib.ticker import FixedLocator, FuncFormatter

    levels = [int(value) for value in classes]
    if levels != list(range(levels[0], levels[0] + len(levels))):
        raise ValueError(f"離散級別必須是連續整數: {levels}")
    colorbar.ax.set_axes_locator(None)
    colorbar.locator = FixedLocator([float(value) for value in levels])
    colorbar.formatter = FuncFormatter(lambda value, _pos: str(int(round(value))))
    colorbar.update_ticks()
    colorbar.minorticks_off()
    colorbar.set_label(label, fontsize=8, labelpad=2)
    colorbar.ax.tick_params(labelsize=8, pad=1, length=3.0, width=0.8)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LINEWIDTH)
        spine.set_edgecolor(FRAME_COLOUR)
    colorbar.ax._traps_role = "colourbar"  # noqa: SLF001
    colorbar.ax._traps_colourbar = {  # noqa: SLF001
        "name": name, "vmin": levels[0] - 0.5, "vmax": levels[-1] + 0.5, "extend": "neither", "label": label, "clipped_fraction": 0.0,
        "discrete": True, "classes": levels,
    }
    return levels


# ---------------------------------------------------------------------------
# 版面紀錄：供 gate 檢查標示刻度與色條端點
# ---------------------------------------------------------------------------


def _labelled_ticks(axis_obj: Any, low: float, high: float) -> list[list[Any]]:
    """範圍內、有可見標示文字的主刻度 ``[位置, 文字]``（左／下或右／上標示皆可）。"""

    low, high = (low, high) if low <= high else (high, low)
    locs = [float(value) for value in axis_obj.get_majorticklocs()]
    ticks = axis_obj.get_major_ticks(len(locs))
    found: list[list[Any]] = []
    for loc, tick in zip(locs, ticks, strict=False):
        label = tick.label1 if tick.label1.get_visible() else tick.label2
        text = label.get_text()
        if low - 1.0e-9 <= loc <= high + 1.0e-9 and label.get_visible() and text.strip():
            found.append([round(loc, 6), text])
    return found


def _marker_entry(handle: Any) -> dict[str, Any]:
    """Line2D（圖上的線或圖例 handle）的符號資訊；顏色一律轉成 ``#rrggbb``。"""

    from matplotlib.colors import to_hex

    def hex_of(colour: Any) -> str | None:
        try:
            return to_hex(colour, keep_alpha=False)
        except (TypeError, ValueError):
            return None

    return {
        "marker": str(handle.get_marker()),
        "markersize": float(handle.get_markersize()),
        "face": hex_of(handle.get_markerfacecolor()),
        "edge": hex_of(handle.get_markeredgecolor()),
        "edge_width": float(handle.get_markeredgewidth()),
        "zorder": float(handle.get_zorder()),
        "clip_on": bool(handle.get_clip_on()),
    }


_NO_MARKER = ("None", "none", "", " ", "nothing")


def _marker_inventory(figure: Any) -> dict[str, Any]:
    """地圖軸上有符號的線與圖例 handle 的清單：``site_stars``（星號）、``cross_markers``（十字類符號：圖上與圖例合計）、``legend_stars``。"""

    stars: list[dict[str, Any]] = []
    crosses = 0
    legend_stars: list[dict[str, Any]] = []
    for axis in figure.axes:
        if getattr(axis, "_traps_role", None) != "map":
            continue
        for line in axis.get_lines():
            if str(line.get_marker()) in _NO_MARKER:
                continue
            entry = _marker_entry(line)
            xs, ys = line.get_xdata(orig=False), line.get_ydata(orig=False)
            if entry["marker"] == SITE_STAR_STYLE["marker"]:
                entry["lonlat"] = [float(xs[0]), float(ys[0])] if len(xs) else None
                stars.append(entry)
            elif entry["marker"] in CROSS_MARKERS:
                crosses += 1
    legends = list(getattr(figure, "legends", [])) + [legend for axis in figure.axes for legend in ([axis.get_legend()] if axis.get_legend() else [])]
    for legend in legends:
        handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
        texts = [text.get_text() for text in legend.get_texts()]
        for handle, text in zip(handles, texts, strict=False):
            if not hasattr(handle, "get_marker") or str(handle.get_marker()) in _NO_MARKER:
                continue
            entry = _marker_entry(handle)
            if entry["marker"] == SITE_STAR_STYLE["marker"]:
                legend_stars.append({**entry, "label": text})
            elif entry["marker"] in CROSS_MARKERS:
                crosses += 1
    return {"site_stars": stars, "cross_markers": crosses, "legend_stars": legend_stars}


def concise_checks_of(figure: Any) -> dict[str, Any] | None:
    """讀出 concise 樣式登記的地圖軸與色條（須在 ``figure.canvas.draw()`` 之後呼叫）；沒有登記時回傳 ``None``。"""

    maps: list[dict[str, Any]] = []
    bars: list[dict[str, Any]] = []
    for axis in figure.axes:
        role = getattr(axis, "_traps_role", None)
        if role == "map":
            x_low, x_high = axis.get_xlim()
            y_low, y_high = axis.get_ylim()
            x_axis_ticks = axis.xaxis.get_major_ticks()
            position = axis.get_position()
            fig_w, fig_h = figure.get_size_inches()
            maps.append(
                {
                    "box_inch": [float(position.width * fig_w), float(position.height * fig_h)],
                    "xticks": _labelled_ticks(axis.xaxis, float(x_low), float(x_high)),
                    "yticks": _labelled_ticks(axis.yaxis, float(y_low), float(y_high)),
                    "xlim": [float(x_low), float(x_high)],
                    "ylim": [float(y_low), float(y_high)],
                    "tick_length_pt": float(x_axis_ticks[0].tick1line.get_markersize()) if x_axis_ticks else 0.0,
                    "frame_linewidth_pt": min(float(spine.get_linewidth()) for spine in axis.spines.values()),
                    "extent_info": _EXTENT_INFO.get(tuple(round(float(v), 6) for v in (x_low, x_high, y_low, y_high))),
                    "analysis_outline_segments": getattr(axis, "_traps_analysis_outline_segments", None),
                    "flow_boundary": getattr(axis, "_traps_flow_boundary", None),
                    "hotspot_masks": getattr(axis, "_traps_hotspot_masks", None),
                }
            )
        elif role == "colourbar":
            meta = dict(getattr(axis, "_traps_colourbar"))
            ticks = _labelled_ticks(axis.yaxis, float(meta["vmin"]), float(meta["vmax"]))
            values = [item[0] for item in ticks]
            span = float(meta["vmax"]) - float(meta["vmin"])
            tolerance = 1.0e-6 * max(span, 1.0e-12)
            meta["ticks"] = ticks
            meta["low_end_labelled"] = any(abs(value - float(meta["vmin"])) <= tolerance for value in values)
            meta["high_end_labelled"] = any(abs(value - float(meta["vmax"])) <= tolerance for value in values)
            meta["interior_labelled_count"] = sum(
                1 for value in values if float(meta["vmin"]) + tolerance < value < float(meta["vmax"]) - tolerance
            )
            bars.append(meta)
    if not maps and not bars:
        return None
    result: dict[str, Any] = {"schema_version": CHECKS_SCHEMA, "map_axes": maps, "colourbars": bars, "markers": _marker_inventory(figure)}
    example_key = getattr(figure, "_traps_example_key", None)
    if example_key is not None:
        result["example_key"] = dict(example_key)
    legend = getattr(figure, "_traps_legend", None)
    if legend is not None:
        result["legend"] = dict(legend)      # v65：圖例框範圍（英吋）、主圖框左緣到色條右緣的允許範圍、欄列數
    return result


# ---------------------------------------------------------------------------
# gate 用的檢查函式（純資料，不依賴 matplotlib）
# ---------------------------------------------------------------------------


def _regular(values: Sequence[float], tolerance: float = 1.0e-6) -> bool:
    if len(values) < 3:
        return True
    steps = np.diff(np.asarray(values, dtype=np.float64))
    return bool(np.all(np.abs(steps - steps[0]) <= tolerance))


def interior_tick_failures(label: str, index: int, axis_name: str, ticks: Sequence[Sequence[Any]], limits: Sequence[float], length_inch: float) -> list[str]:
    """內部刻度（兩端之間）間距規則；內部刻度與端點標示、相鄰標示的圖面距離不小於標示最小距離的 90%（不擠、不重疊）。"""

    errors: list[str] = []
    values = [float(item[0]) for item in ticks]
    if len(values) < 2:
        return errors
    interior = values[1:-1]
    if not _regular(interior):
        errors.append(f"{label}: 地圖軸 {index} {axis_name} 內部刻度間距不規則 {interior}")
    span = max(abs(float(limits[1]) - float(limits[0])), 1.0e-12)
    need = 0.9 * LABEL_SPACING_INCH["lon" if axis_name == "xticks" else "lat"]
    for first, second in zip(values[:-1], values[1:], strict=False):
        gap = abs(second - first) / span * length_inch
        if gap < need:
            errors.append(f"{label}: 地圖軸 {index} {axis_name} 標示 {first:.2f} 與 {second:.2f} 相距 {gap:.2f} 英吋，小於 {need:.2f} 英吋（擁擠）")
    if len(interior) >= 1:
        step = interior[1] - interior[0] if len(interior) >= 2 else None
        if step is not None and min(interior[0] - values[0], values[-1] - interior[-1]) < END_CLEARANCE_FRACTION * step - 1.0e-6:
            errors.append(f"{label}: 地圖軸 {index} {axis_name} 內部刻度距端點標示不足間距的 {END_CLEARANCE_FRACTION:.0%}")
    return errors


AXIS_END_TOLERANCE = 1.0e-4
"""軸兩端與刻度位置的容許差（佔該軸範圍的比例）。"""


EVEN_TICK_TOLERANCE = 1.0e-3
"""標示刻度間距相對於平均間距的容許差（0.1%，涵蓋標示值的四捨五入）。"""


def even_tick_failures(label: str, index: int, axis_name: str, ticks: Sequence[Sequence[Any]]) -> list[str]:
    """每軸的標示刻度必須等間距（相鄰刻度差全相等）。"""

    values = [float(item[0]) for item in ticks]
    if len(values) < 3:
        return []
    gaps = np.diff(values)
    mean = float(np.mean(gaps))
    if float(np.max(np.abs(gaps - mean))) > EVEN_TICK_TOLERANCE * max(abs(mean), 1.0e-12) + 1.0e-6:
        return [f"{label}: 地圖軸 {index} {axis_name} 標示刻度間距不相等 {[round(float(g), 4) for g in gaps]}"]
    return []


def axis_end_failures(label: str, index: int, entry: Mapping[str, Any]) -> list[str]:
    """X、Y 軸兩端都必須是標示刻度：第一個與最後一個標示刻度落在軸範圍的兩端（容許差為軸範圍的 ``AXIS_END_TOLERANCE``）；
    且長寬比誤差不超過該圖框的上限（大圖 `map`、放大圖 `zoom` 各 ±2%，見 :func:`frame_aspect_tolerance`；紀錄沒有 ``box_inch`` 時用該圖框的標準主圖框）。"""

    errors: list[str] = []
    for axis_name, limit_name in (("xticks", "xlim"), ("yticks", "ylim")):
        ticks = entry.get(axis_name, [])
        limits = entry.get(limit_name)
        if not ticks or not limits:
            continue
        low, high = float(min(limits)), float(max(limits))
        tolerance = AXIS_END_TOLERANCE * max(high - low, 1.0e-12)
        want_count = expected_tick_count("lon" if axis_name == "xticks" else "lat", high - low)
        if want_count is not None:
            # 標準圖框規格表內的跨度：標示數必須剛好等於 跨度／間距＋1（大圖經度 10、緯度 7；放大圖經度 5、緯度 6）
            if len(ticks) != want_count:
                errors.append(f"{label}: 地圖軸 {index} {axis_name} 標示 {len(ticks)} 個，規格表要求 {want_count} 個（跨度 {high - low:g}°）")
        elif not (MIN_AXIS_LABELS <= len(ticks) <= MAX_AXIS_LABELS):
            errors.append(f"{label}: 地圖軸 {index} {axis_name} 標示 {len(ticks)} 個，不在 {MIN_AXIS_LABELS}–{MAX_AXIS_LABELS} 個之內")
        box = entry.get("box_inch")
        if box:
            errors.extend(interior_tick_failures(label, index, axis_name, ticks, (low, high), float(box[0] if axis_name == "xticks" else box[1])))
        errors.extend(even_tick_failures(label, index, axis_name, ticks))
        first, last = float(ticks[0][0]), float(ticks[-1][0])
        if abs(first - low) > tolerance:
            errors.append(f"{label}: 地圖軸 {index} {axis_name} 下／左端 {low:.6f} 沒有標示刻度（第一個標示在 {first:.6f}）")
        if abs(last - high) > tolerance:
            errors.append(f"{label}: 地圖軸 {index} {axis_name} 上／右端 {high:.6f} 沒有標示刻度（最後一個標示在 {last:.6f}）")
    xlim, ylim = entry.get("xlim"), entry.get("ylim")
    if xlim and ylim:
        from . import figure_standard as std

        scale = max(math.cos(math.radians(0.5 * (float(ylim[0]) + float(ylim[1])))), 0.05)
        ratio = (float(xlim[1]) - float(xlim[0])) * scale / max(float(ylim[1]) - float(ylim[0]), 1.0e-12)
        spec_name = frame_spec_name_of(xlim, ylim)
        default_box = {"map": std.CONCISE_MAP_AXES, "zoom": std.CONCISE_ZOOM_AXES}.get(spec_name or "", std.MAP1_AXES)
        box = entry.get("box_inch") or [default_box.w, default_box.h]
        error = ratio / (float(box[0]) / float(box[1])) - 1.0
        limit = frame_aspect_tolerance(spec_name or "other")  # 大圖與放大圖各 ±2%、規格表以外 ±5%
        if abs(error) > limit + 1.0e-3:
            errors.append(f"{label}: 地圖軸 {index} 長寬比誤差 {error:+.1%} 超過上限 {limit:.0%}")
    return errors


def concise_record_failures(label: str, checks: Mapping[str, Any] | None) -> list[str]:
    """單張圖的 concise 檢查：地圖軸每軸 4–7 個標示刻度（兩端必標、內部刻度規則且不擁擠）、度數格式 2 位小數；色條兩端點皆標示且有內部刻度。"""

    if checks is None:
        return [f"{label}: 缺少 concise 檢查資料（圖未套用 concise 樣式）"]
    errors: list[str] = []
    if not checks.get("map_axes"):
        errors.append(f"{label}: 沒有登記的地圖軸")
    if not checks.get("colourbars"):
        errors.append(f"{label}: 沒有登記的色條")
    for index, entry in enumerate(checks.get("map_axes", [])):
        for axis_name in ("xticks", "yticks"):
            ticks = entry.get(axis_name, [])
            if not ticks:
                errors.append(f"{label}: 地圖軸 {index} {axis_name} 沒有標示刻度")
                continue
            for _loc, text in ticks:
                if not re.fullmatch(r"\d+\.\d{%d}" % AXIS_LABEL_DECIMALS, str(text)):
                    errors.append(f"{label}: 地圖軸 {index} {axis_name} 標示 {text!r} 不是 {AXIS_LABEL_DECIMALS} 位小數的純數字")
                    break
        errors.extend(axis_end_failures(label, index, entry))
        if float(entry.get("tick_length_pt", 0.0)) <= 0.0:
            errors.append(f"{label}: 地圖軸 {index} 沒有刻度記號")
        if float(entry.get("frame_linewidth_pt", 0.0)) < 0.9:
            errors.append(f"{label}: 地圖軸 {index} 框線過細")
    for index, entry in enumerate(checks.get("colourbars", [])):
        if entry.get("discrete"):
            classes = [int(value) for value in entry.get("classes", [])]
            shown = [(round(float(item[0]), 6), str(item[1])) for item in entry.get("ticks", [])]
            if shown != [(float(value), str(value)) for value in classes]:
                errors.append(f"{label}: 離散色條 {index}（{entry.get('name')}）刻度 {shown} 不是各級整數 {classes}")
            if entry.get("extend", "neither") != "neither":
                errors.append(f"{label}: 離散色條 {index}（{entry.get('name')}）有延伸箭頭 extend={entry.get('extend')!r}（需平整端點）")
            continue
        if not (entry.get("low_end_labelled") and entry.get("high_end_labelled")):
            errors.append(f"{label}: 色條 {index}（{entry.get('name')}）端點 vmin／vmax 沒有同時標示")
        if entry.get("extend", "neither") != "neither":
            errors.append(f"{label}: 色條 {index}（{entry.get('name')}）有延伸箭頭 extend={entry.get('extend')!r}（需平整端點）")
        if int(entry.get("interior_labelled_count", 0)) < MIN_COLORBAR_INTERIOR_TICKS:
            errors.append(f"{label}: 色條 {index}（{entry.get('name')}）內部標示刻度少於 {MIN_COLORBAR_INTERIOR_TICKS} 個")
    key = checks.get("example_key")
    if key is not None:
        if not key.get("inside_axes"):
            errors.append(f"{label}: 1 m/s 參考箭頭不在地圖軸內 {key.get('bbox_axes_inch')}")
        if not key.get("inside_patch"):
            errors.append(f"{label}: 參考箭頭超出固定保留區 {key.get('bbox_axes_inch')} ⊄ {key.get('patch_axes_inch')}")
        if int(key.get("arrows_overlapping_key", 1)) != 0:
            errors.append(f"{label}: {key.get('arrows_overlapping_key')} 支流速箭頭與參考箭頭重疊")
    return errors


def site_marker_failures(label: str, checks: Mapping[str, Any] | None) -> list[str]:
    """地點錨點符號：地圖上至少有一個星號且畫法等於 :data:`SITE_STAR_STYLE`（大小、填色、邊線、zorder，且不被軸框裁掉）；
    圖例中的星號 ``markersize`` 為 :data:`SITE_STAR_LEGEND_MARKERSIZE`、邊線寬同圖上；地圖與圖例都沒有十字類符號（``+`` 等）。
    大小是 LBT 程式值乘 LBT 印出縮放比（:data:`LBT_STAR_PRINT_SCALE`），即 LBT 報告上的實際大小。"""

    markers = (checks or {}).get("markers")
    if markers is None:
        return [f"{label}: 版面紀錄沒有符號清單（concise.markers），無法確認地點錨點是星號（需用新版程式重繪）"]
    errors: list[str] = []
    stars = markers.get("site_stars", [])
    if not stars:
        errors.append(f"{label}: 地圖上沒有地點錨點星號")
    want = {
        "markersize": float(SITE_STAR_STYLE["markersize"]), "face": SITE_STAR_STYLE["markerfacecolor"], "edge": "#000000",
        "edge_width": float(SITE_STAR_STYLE["markeredgewidth"]), "zorder": float(SITE_STAR_STYLE["zorder"]),
    }
    for index, entry in enumerate(stars):
        for key, value in want.items():
            got = entry.get(key)
            if (isinstance(value, float) and (got is None or abs(float(got) - value) > 1.0e-9)) or (isinstance(value, str) and got != value):
                errors.append(f"{label}: 地點錨點星號 {index} 的 {key} 是 {got!r}，應為 {value!r}（與 Lagrangian-Ensemble-Backtracking 一致）")
        if entry.get("clip_on"):
            errors.append(f"{label}: 地點錨點星號 {index} 會被軸框裁切（clip_on 應為 False）")
    for index, entry in enumerate(markers.get("legend_stars", [])):
        if (
            abs(float(entry.get("markersize", 0.0)) - SITE_STAR_LEGEND_MARKERSIZE) > 1.0e-9
            or abs(float(entry.get("edge_width", -1.0)) - want["edge_width"]) > 1.0e-9
            or entry.get("face") != want["face"] or entry.get("edge") != want["edge"]
        ):
            errors.append(
                f"{label}: 圖例星號 {index}（{entry.get('label')}）樣式 {entry} 不符"
                f"（markersize {SITE_STAR_LEGEND_MARKERSIZE:.4f}、邊線寬 {want['edge_width']:.4f}、黃色填色、黑色邊線）"
            )
    if int(markers.get("cross_markers", 0)) != 0:
        errors.append(f"{label}: 圖上或圖例仍有 {markers.get('cross_markers')} 個十字類符號（+ 等），地點錨點須為星號")
    return errors


def shared_limit_failures(records: Sequence[Mapping[str, Any]], kind_of: Any, group_of: Any) -> list[str]:
    """同一產品、同一圖種的各站色條（依序）vmin／vmax 必須完全相同。"""

    groups: dict[tuple[str, str, int, str], set[tuple[float, float]]] = {}
    for record in records:
        checks = record.get("concise")
        if not checks:
            continue
        kind, group = kind_of(record), group_of(record)
        for index, entry in enumerate(checks.get("colourbars", [])):
            groups.setdefault((group, kind, index, str(entry.get("name"))), set()).add((round(float(entry["vmin"]), 9), round(float(entry["vmax"]), 9)))
    errors: list[str] = []
    for (group, kind, index, name), limits in sorted(groups.items()):
        if len(limits) > 1:
            errors.append(f"{group}/{kind}: 色條 {index}（{name}）各站共用色階不一致 {sorted(limits)}")
    return errors


NEGATIVE_WORD_PATTERN = r"不(?!同)|無|未|非|小於|低於|少於"
"""圖面文字須為肯定句：不含否定詞（「不同」是形容詞，不算）。"""
