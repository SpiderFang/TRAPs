"""報告圖的統一畫布與座標軸框（2026-10-14 圖面標準）。

審閱者指出各圖大小不一：地圖圖框高度原本隨各研究區的長寬比改變（例如貢寮熱點圖約 7 cm 高、
連江約 11.4 cm 高），不同圖種的畫布也各自決定。本模組集中定義報告模式（``TRAPS_VECTOR_REPORT=1``）
使用的標準：

* 兩個畫布家族，寬度都是 6.3 英吋（Word 欄寬）：

  - ``single``：6.3 × 4.73 英吋，單一面板的圖（熱點、累積天數、熱點重疊、長壽命核心路徑、A 區總覽）
    與兩面板的壽命圖；
  - ``multi``：6.3 × 7.0 英吋，多面板的圖（空間分布、季節、代表 TRAP、方法比較、傳播速度、
    近岸診斷、逐日時間序列、軌跡通過格點數）。

* 同一個版面類別（``layout class``）的所有圖，主座標軸框（位置與大小）、色條、標題位置與
  圖例帶完全相同；站與站、逐時與逐日之間不因研究區形狀而改變。
* 地圖以「固定軸框、擴張經緯度範圍」配合軸框長寬比（含 cos(緯度) 的地理比例）：視窗中心維持在
  各站原本的分析區中心，短邊方向的範圍向外延伸，延伸出的區域照常顯示海岸線與陸地；站內遮罩外的海域
  維持原本的空白樣式。色階、遮罩、熱點與所有資料值完全不變。
* 字級：標題 10–11 pt（總標題 11 pt、面板標題與副標題 10 pt），其餘文字 8 pt。

本模組只提供版面常數、純幾何函式與版面 gate；不碰任何資料值。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

WIDTH_INCH = 6.3
"""所有家族共同的畫布寬度（英吋）。"""
SINGLE_HEIGHT_INCH = 4.73
"""``single`` 家族畫布高度（英吋）：標準地圖軸框 4.50×2.98（流場範圍的公里長寬比約 1.51）＋上方 0.80 標題帶＋下方 0.95 的軸標與圖例帶。"""
MULTI_HEIGHT_INCH = 7.0
"""``multi`` 家族畫布高度（英吋）。"""
TOLERANCE_INCH = 0.01
"""版面 gate 允許的畫布與座標軸框差異（英吋）。"""

TITLE_FONT_PT = 11.0
"""總標題字級。"""
SUBTITLE_FONT_PT = 10.0
"""副標題與面板標題字級。"""
TEXT_FONT_PT = 8.0
"""刻度、座標軸標籤、色條、圖例的字級。"""

TITLE_TOP_INCH = 0.10
"""總標題上緣距畫布頂端（英吋）。"""
SUBTITLE_TOP_INCH = 0.34
"""副標題第一行上緣距畫布頂端（英吋）。"""
LEGEND_BASE_INCH = 0.04
LEGEND_SPAN_SAFETY_INCH = 0.04
"""圖例框離「主圖框左緣到色條右緣」範圍兩端的最小預留（英吋）：繪圖時以 100 dpi 量框寬，存成 SVG 的實際框寬約有 0.03 英吋的出入。"""
"""圖例帶下緣距畫布底端（英吋）。"""
SUBTITLE_MAX_UNITS = 80
"""副標題每行最大視覺寬度（全形 2、半形 1），約等於 6.3 英吋欄寬內 10 pt 的 40 個全形字。"""


@dataclass(frozen=True)
class Box:
    """以英吋表示、左下角起算的矩形（座標軸框或色條框）。"""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]


# ---------------------------------------------------------------------------
# single 家族（6.3 × 4.73）
# ---------------------------------------------------------------------------

MAP1_AXES = Box(0.72, 0.95, 4.50, 2.98)
"""單一地圖的標準座標軸框：熱點、累積天數、熱點重疊、長壽命核心路徑、A 區總覽共用。"""
MAP1_CBAR = Box(5.32, 0.95, 0.16, 2.98)
"""單一地圖右側垂直色條框（與地圖等高）。"""

CONCISE_MAP_LON_SPAN, CONCISE_MAP_LAT_SPAN = 1.8, 1.2
"""簡要報告大圖（聚集熱區、四季重現、A 區總覽共 11 張）的經緯度跨度（度）：LBT run09 axes_unified 規格（見 :data:`traps.concise_style.FRAME_SPECS`）。"""
CONCISE_MAP_REFERENCE_LAT_DEG = 24.1
"""大圖主圖框長寬比的參考緯度（度）：LBT 的做法，一度緯度的長度 ＝ 1／cos(24.1°) 倍一度經度長度，五個地點共用。"""
CONCISE_MAP_HEIGHT_INCH = MAP1_AXES.h
"""大圖主圖框高度（英吋）：沿用 MAP1 的 2.98 英吋（畫布 6.3 × 4.13 英吋與上下留白不變）。"""
CONCISE_MAP_WIDTH_INCH = round(
    CONCISE_MAP_HEIGHT_INCH * CONCISE_MAP_LON_SPAN * math.cos(math.radians(CONCISE_MAP_REFERENCE_LAT_DEG)) / CONCISE_MAP_LAT_SPAN, 3
)
"""大圖主圖框寬度（英吋）：高度 × 1.8° × cos(24.1°) ／ 1.2°（≈ 4.082）；MAP1 的 4.50 × 2.98 框（長寬比 1.51）放不進 1.8° × 1.2° 的範圍而不變形，所以縮窄寬度（畫布大小不變）。"""
CONCISE_MAP_AXES = Box(0.91, MAP1_AXES.y, CONCISE_MAP_WIDTH_INCH, CONCISE_MAP_HEIGHT_INCH)
"""簡要報告大圖（kind：hotspots、seasonal_recurrence、domainA_overview）的主座標軸框：框高同 MAP1，寬度縮窄為 LBT 長寬比；
左緣右移 0.19 英吋，使「軸框＋色條＋色條刻度與標籤」整體仍置中於 6.3 英吋畫布。代表性 TRAP 放大圖（example）不變，仍用 :data:`MAP1_AXES`。"""
LBT_PRINT_SCALE = 6.27 / (5437.0 / 600.0)
"""LBT run09 條件式來源足跡圖在報告中的縮放比（顯示寬 6.27 英吋 ÷ 圖檔 5437 px／600 dpi ＝ 9.06 英吋 ≈ 0.692）：LBT 的色條寬、間距、線寬等以「圖檔原尺寸」寫在程式裡，
印在報告上時都縮為此比例；TRAPs 的圖以 6.3 英吋原尺寸印出，所以與 LBT 對齊「印出效果」的尺寸＝LBT 程式值 × 此比例。"""
CONCISE_CBAR_WIDTH_INCH = round(0.018 * 9.6 * LBT_PRINT_SCALE, 3)
CONCISE_CBAR_PAD_INCH = round(0.014 * 9.6 * LBT_PRINT_SCALE, 3)
"""色條寬與主圖框到色條的間距（英吋）：LBT ``_add_aligned_colorbar`` 的 width=0.018、pad=0.014（圖寬 9.6 英吋的比例）× 印出縮放 ≈ 0.120、0.093 英吋（TRAPs 原本 0.16、0.10）。"""
CONCISE_MAP_CBAR = Box(CONCISE_MAP_AXES.right + CONCISE_CBAR_PAD_INCH, MAP1_CBAR.y, CONCISE_CBAR_WIDTH_INCH, MAP1_CBAR.h)
"""簡要報告大圖右側垂直色條框（與主座標軸框等高；寬與間距對齊 LBT 的印出尺寸）。"""

CONCISE_ZOOM_LON_SPAN, CONCISE_ZOOM_LAT_SPAN = 0.8, 0.5
"""簡要報告代表性 TRAP 放大圖（5 張）的經緯度跨度（度）：標準圖框 ``zoom``（見 :data:`traps.concise_style.FRAME_SPECS`）。"""
CONCISE_ZOOM_WIDTH_INCH = round(
    CONCISE_MAP_HEIGHT_INCH * CONCISE_ZOOM_LON_SPAN * math.cos(math.radians(CONCISE_MAP_REFERENCE_LAT_DEG)) / CONCISE_ZOOM_LAT_SPAN, 3
)
"""放大圖主圖框寬度（英吋）：高度 × 0.8° × cos(24.1°) ／ 0.5°（≈ 4.352），與大圖相同的 LBT 參考緯度 24.1° 長寬比規則。
放大圖的視窗跨度（0.8° × 0.5°）與大圖（1.8° × 1.2°）不同，所以主圖框寬度不可能和大圖相同（4.08 英吋會使放大圖的長寬比誤差達 +4.9% 至 +8.4%，超過 ±2%）；
但左緣、下緣、框高、色條寬與間距、圖例位置的規則都與大圖相同。"""
CONCISE_ZOOM_AXES = Box(CONCISE_MAP_AXES.x, MAP1_AXES.y, CONCISE_ZOOM_WIDTH_INCH, CONCISE_MAP_HEIGHT_INCH)
"""簡要報告代表性 TRAP 放大圖（kind：example）的主座標軸框：左緣、下緣、框高與大圖 :data:`CONCISE_MAP_AXES` 相同，寬度依 24.1° 長寬比。"""
CONCISE_ZOOM_CBAR = Box(CONCISE_ZOOM_AXES.right + CONCISE_CBAR_PAD_INCH, MAP1_CBAR.y, CONCISE_CBAR_WIDTH_INCH, MAP1_CBAR.h)
"""放大圖右側垂直色條框：寬 0.12 英吋、與主圖框間距 0.093 英吋（同大圖），與主座標軸框等高。"""

CONCISE_LEGEND_ROWS = 2
"""簡要報告其他大圖（四季重現）與代表性 TRAP 放大圖的圖例列數（兩列，圖例帶高約 0.367 英吋）。"""
CONCISE_LEG3_ROWS = 3
"""v65（2026-11-03）聚集熱區圖（5 張）與圖1：圖例為三列（黑點、灰點兩個核心項目並排，圖1 另有統計歸屬分界與地點位置，兩列放不進主圖框到色條右緣的寬度）。"""
CONCISE_LEG3_EXTRA_INCH = 0.1667
"""三列圖例比兩列多出的高度（英吋）＝ 圖例一列的間距（兩列圖例框高 0.3667 英吋、三列 0.5333 英吋，差 0.1667 英吋）。畫布高度加這個值（加在下方）、主圖框與色條框下緣同步上移，
所以「x 軸標籤底緣到圖例框頂緣」與「圖例框底緣到畫布底緣」兩個留白與四季重現圖、代表性 TRAP 圖相同（gate ``legend_spacing`` 檢查），主圖框從畫布頂端算起的位置也與其他圖相同。"""
CONCISE_LEG3_KINDS: tuple[str, ...] = ("hotspots", "domainA_overview")
"""使用三列圖例（較高畫布）的圖種：聚集熱區與 A 區總覽（圖1），共 6 張，彼此畫布、軸框與留白相同。"""
CONCISE_LEG3_CLASS = "map_lbt_leg3"
CONCISE_LEG3_AXES = Box(CONCISE_MAP_AXES.x, round(CONCISE_MAP_AXES.y + CONCISE_LEG3_EXTRA_INCH, 4), CONCISE_MAP_AXES.w, CONCISE_MAP_AXES.h)
CONCISE_LEG3_CBAR = Box(CONCISE_MAP_CBAR.x, round(CONCISE_MAP_CBAR.y + CONCISE_LEG3_EXTRA_INCH, 4), CONCISE_MAP_CBAR.w, CONCISE_MAP_CBAR.h)
"""三列圖例圖種的主圖框與色條框：大小、左緣與 :data:`CONCISE_MAP_AXES`／:data:`CONCISE_MAP_CBAR` 相同，下緣上移 :data:`CONCISE_LEG3_EXTRA_INCH`（從畫布頂端算起的位置不變）。"""

LBT_COAST_COLOUR = "black"
LBT_COAST_WIDTH_PT = round(0.5 * LBT_PRINT_SCALE, 2)
"""海岸線：LBT 陸地多邊形邊線 ``edgecolor="black"``、``linewidth=0.5``（圖檔原尺寸）× 印出縮放 ≈ 0.35 pt（TRAPs 原本 #263238、0.6 pt）。"""

LIFETIME_LEFT = Box(0.85, 0.95, 2.15, 2.50)
LIFETIME_RIGHT = Box(3.85, 0.95, 2.15, 2.50)
"""壽命圖兩個面板（CCDF／吸引強度演化）的座標軸框；圖例在兩面板下方。"""

# ---------------------------------------------------------------------------
# multi 家族（6.3 × 7.0）
# ---------------------------------------------------------------------------

SPATIAL_PANEL_W, SPATIAL_PANEL_H = 2.35, 1.62
SPATIAL_X = (0.66, 3.35)
SPATIAL_Y = (4.65, 1.61)
SPATIAL_CBAR_GAP, SPATIAL_CBAR_H = 0.56, 0.14
"""空間分布 2×2：每個面板下方一條水平色條。"""

SEASONAL_PANEL_W, SEASONAL_PANEL_H = 2.35, 1.30
SEASONAL_X = (0.66, 3.35)
SEASONAL_Y = (4.97, 2.97)
SEASONAL_CBAR = Box(0.66, 2.40, 2.0 * 2.35 + 0.34, 0.14)
SEASONAL_BARS_X, SEASONAL_BARS_W = (0.66, 3.60), 2.10
SEASONAL_BARS_Y, SEASONAL_BARS_H = 0.38, 1.20
"""季節圖：2×2 地圖、橫跨的水平色條、下方兩張長條圖。"""

EXAMPLE_AXES = Box(0.72, 1.12, 4.50, 5.03)
EXAMPLE_CBAR = Box(5.32, 1.12, 0.16, 5.03)
EXAMPLE_LEGEND_ROWS = 3
"""代表 TRAP（含近岸型代表）：高地圖＋垂直 s1 色條；下方圖例帶最多三列。"""

METHOD_A = Box(0.62, 3.62, 2.10, 2.25)
METHOD_A_CBAR = Box(2.80, 3.62, 0.14, 2.25)
METHOD_B = Box(3.45, 3.62, 2.10, 2.25)
METHOD_C = Box(0.62, 0.64, 2.10, 2.25)
METHOD_C_CBAR = Box(2.80, 0.64, 0.14, 2.25)
METHOD_LEGEND_CENTER = (4.50, 1.76)
"""方法比較：(a) s1、(b) 流線在上排，(c) 散度在下排左側；圖例放在下排右側空格。"""

PROP_HIST_X = Box(0.70, 3.70, 2.25, 2.00)
PROP_HIST_Y = Box(3.50, 3.70, 2.25, 2.00)
PROP_ROSE = Box(1.25, 0.55, 2.05, 2.05)
PROP_ROSE_CBAR = Box(3.85, 0.55, 0.16, 2.05)
"""傳播速度：兩張速度分量直方圖在上排，去向玫瑰與色條在下排。"""

COASTAL_A = Box(0.85, 4.62, 5.00, 1.65)
COASTAL_B = Box(0.85, 1.30, 5.00, 1.65)
"""近岸診斷：上下兩個面板，圖例在各面板下方。"""

TIMESERIES_1 = Box(0.85, 3.77, 5.00, 2.45)
TIMESERIES_2 = Box(0.85, 0.96, 5.00, 2.45)
"""逐日時間序列：上下兩個共用 x 軸的面板，圖例在各面板上緣。"""

DENSITY_1 = Box(0.72, 3.97, 4.50, 2.30)
DENSITY_2 = Box(0.72, 0.95, 4.50, 2.30)
DENSITY_CBAR_X, DENSITY_CBAR_W = 5.32, 0.16
"""軌跡通過格點數：上下兩張地圖，各自右側垂直色條，地圖 x 位置與 MAP1 相同。"""

FAMILY_CANVAS: Mapping[str, tuple[float, float]] = {
    "single": (WIDTH_INCH, SINGLE_HEIGHT_INCH),
    "multi": (WIDTH_INCH, MULTI_HEIGHT_INCH),
}

# 圖種 -> (畫布家族, 版面類別)。同一版面類別的圖，主座標軸框必須完全相同。
KIND_LAYOUT: Mapping[str, tuple[str, str]] = {
    "hotspots": ("single", "map1"),
    "cumulative_days": ("single", "map1"),
    "hotspot_overlap": ("single", "map1"),
    "long_tracks": ("single", "map1"),
    "domainA_overview": ("single", "map1"),
    "lifetime": ("single", "lifetime"),
    "spatial": ("multi", "spatial"),
    "seasonal": ("multi", "seasonal"),
    "example": ("multi", "example"),
    "example_coastal": ("multi", "example"),
    "method_comparison": ("multi", "method"),
    "propagation": ("multi", "propagation"),
    "coastal_diagnostic": ("multi", "coastal"),
    "timeseries": ("multi", "timeseries"),
    "track_density": ("multi", "density"),
}


CONCISE_KIND_LAYOUT: Mapping[str, tuple[str, str]] = {
    **KIND_LAYOUT,
    "example": ("single", "map_lbt_zoom"),
    "hotspots": ("single", CONCISE_LEG3_CLASS),
    "seasonal_recurrence": ("single", "map_lbt"),
    "domainA_overview": ("single", CONCISE_LEG3_CLASS),
    "lifetime": ("single", "line_compare"),
}
"""簡要報告圖組的圖種對照：21 張圖（16 張地圖加 5 張壽命分布比較圖，2026-10-28）都用 single 家族（畫布相同）；
壽命分布比較圖（``lifetime``）為折線圖，軸框同 :data:`CONCISE_MAP_AXES`，自成 ``line_compare`` 版面類別；
四季重現（``seasonal_recurrence``）使用 single 家族與 ``map_lbt`` 軸框（:data:`CONCISE_MAP_AXES`，LBT 長寬比；2026-10-22 第二輪）；
聚集熱區（``hotspots``）與 A 區總覽（``domainA_overview``）自 v65 起為 ``map_lbt_leg3``（三列圖例，畫布加高 :data:`CONCISE_LEG3_EXTRA_INCH`，軸框見 :data:`CONCISE_LEG3_AXES`）；代表性 TRAP 放大圖（``example``）使用 ``map_lbt_zoom`` 軸框
（:data:`CONCISE_ZOOM_AXES`，同一個 24.1° 參考緯度規則，2026-10-22 第三輪）；其餘與正式圖組相同。"""


def family_of(kind: str, layout: Mapping[str, tuple[str, str]] | None = None) -> str:
    """圖種所屬的畫布家族名稱（``layout`` 預設為正式圖組的 :data:`KIND_LAYOUT`）。"""

    return (layout or KIND_LAYOUT)[kind][0]


CONCISE_TITLE_CROP_INCH = 0.60
"""簡要報告圖組移除圖內標題與副標題後，從畫布頂端裁掉的高度（英吋）；所有圖相同，軸框相對位置不變。"""

LIFETIME_AXES_Y_INCH = 0.80
"""壽命分布比較圖（``line_compare``）的軸框下緣距畫布底端（英吋；v50, 2026-10-28）。地圖類圖（聚集熱區、代表性 TRAP）為 0.95，其下方有 2 列圖例（高 0.367 英吋）；
壽命圖的圖例只有 1 列（高 0.200 英吋，字級、外框與地圖類圖相同），圖例下緣到畫布底緣的留白維持 0.0956 英吋，所以「x 軸標籤底緣到圖例框頂緣」的間距要與地圖類圖（0.175–0.189 英吋）相同，
軸框下緣就要上移到 0.80 英吋（x 軸標籤底緣 0.480 英吋、圖例框頂緣 0.2956 英吋，間距約 0.184 英吋）。"""
LIFETIME_CANVAS_HEIGHT_INCH = 3.98
"""壽命分布比較圖的畫布高度（英吋）：軸框上緣距畫布頂端維持 0.20 英吋（同地圖類圖）、框高 2.98 英吋不變，故高度 ＝ 0.80 ＋ 2.98 ＋ 0.20；畫布寬度同地圖類圖（6.3 英吋），
Word 內以同一寬度（6.3 英吋）插入，所以比例尺相同、只是圖較矮（不放大字級來填空白）。"""
LIFETIME_LAYOUT_CLASS = "line_compare"


def canvas_of(
    kind: str, layout: Mapping[str, tuple[str, str]] | None = None, *, crop_top_inch: float = 0.0
) -> tuple[float, float]:
    """圖種的標準畫布 ``(寬, 高)`` 英吋；``crop_top_inch`` 為頂端裁掉的高度（移除標題後的簡要報告圖）。"""

    if layout is not None and layout.get(kind, ("", ""))[1] == LIFETIME_LAYOUT_CLASS:
        return WIDTH_INCH, LIFETIME_CANVAS_HEIGHT_INCH      # v50: the line chart has its own canvas height (no title band, one-row legend); same width as the maps
    width, height = FAMILY_CANVAS[family_of(kind, layout)]
    if layout is not None and layout.get(kind, ("", ""))[1] == CONCISE_LEG3_CLASS:
        height += CONCISE_LEG3_EXTRA_INCH      # v65: three-row legend, the extra height is added below the map (top-referenced positions unchanged)
    return width, height - crop_top_inch


# ---------------------------------------------------------------------------
# 地理範圍
# ---------------------------------------------------------------------------


def fit_limits_to_box(
    xlim: tuple[float, float], ylim: tuple[float, float], box_w: float, box_h: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """以視窗中心為固定點，把經緯度範圍擴張到與軸框同長寬比（含 cos 緯度）。

    只向外擴張（短邊方向增加範圍），原本的視窗必定完整包含在新範圍內；中心不動，
    因此站點仍位在圖框中央。地理比例為 ``(經度跨度 × cos 中心緯度) / 緯度跨度 = box_w / box_h``。
    """

    cx, cy = 0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1])
    scale = max(math.cos(math.radians(cy)), 0.05)
    dx = max(float(xlim[1] - xlim[0]), 1.0e-9)
    dy = max(float(ylim[1] - ylim[0]), 1.0e-9)
    target = float(box_w) / float(box_h)
    if dx * scale / dy < target:
        dx = dy * target / scale
    else:
        dy = dx * scale / target
    return (cx - 0.5 * dx, cx + 0.5 * dx), (cy - 0.5 * dy, cy + 0.5 * dy)


def _axis_slice(axis_values: Sequence[float], lo: float, hi: float, margin_cells: int) -> slice:
    values = [float(v) for v in axis_values]
    inside = [index for index, value in enumerate(values) if lo <= value <= hi]
    if not inside:
        return slice(0, len(values))
    first = max(inside[0] - margin_cells, 0)
    last = min(inside[-1] + margin_cells + 1, len(values))
    return slice(first, last)


def limits_slices(
    lon: Sequence[float], lat: Sequence[float], xlim: tuple[float, float], ylim: tuple[float, float], margin_cells: int = 1
) -> tuple[slice, slice]:
    """涵蓋 ``xlim``／``ylim`` 的 (row, col) 切片（再外加 ``margin_cells`` 格，避免邊緣格被切掉）。"""

    rows = _axis_slice(lat, min(ylim), max(ylim), margin_cells)
    cols = _axis_slice(lon, min(xlim), max(xlim), margin_cells)
    return rows, cols


# ---------------------------------------------------------------------------
# 標題與圖例
# ---------------------------------------------------------------------------


def visual_width(text: str) -> int:
    """視覺寬度：全形／寬字元 2，其他 1。"""

    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)


def wrap_subtitle(text: str, max_units: int = SUBTITLE_MAX_UNITS) -> str:
    """把副標題依「｜」語意邊界排成不超過 ``max_units`` 的行；單一片段過長時才在空白處折行。"""

    if not text:
        return text
    lines: list[str] = []
    current = ""
    for segment in text.split("｜"):
        candidate = segment if not current else f"{current}｜{segment}"
        if current and visual_width(candidate) > max_units:
            lines.append(current)
            current = segment
        else:
            current = candidate
    if current:
        lines.append(current)
    wrapped: list[str] = []
    for line in lines:
        if visual_width(line) <= max_units:
            wrapped.append(line)
            continue
        part = ""
        for word in line.split(" "):
            trial = word if not part else f"{part} {word}"
            if part and visual_width(trial) > max_units:
                wrapped.append(part)
                part = word
            else:
                part = trial
        if part:
            wrapped.append(part)
    return "\n".join(wrapped)


def draw_title_block(figure: Any, height: float, title: str, subtitle: str | None = None) -> None:
    """在固定位置畫總標題（11 pt）與可選副標題（10 pt，必要時多行）。"""

    figure.text(0.5, 1.0 - TITLE_TOP_INCH / height, title, ha="center", va="top", fontsize=TITLE_FONT_PT)
    if subtitle:
        figure.text(
            0.5,
            1.0 - SUBTITLE_TOP_INCH / height,
            wrap_subtitle(subtitle),
            ha="center",
            va="top",
            fontsize=SUBTITLE_FONT_PT,
            linespacing=1.2,
        )


def colourbar_right_inch(figure: Any, cax: Any) -> float:
    """色條（含刻度、刻度標示與色條標籤）的右緣（英吋，距畫布左緣）；須在色條樣式與標籤都設定好之後呼叫。"""

    renderer = figure.canvas.get_renderer()
    return float(cax.get_tightbbox(renderer).x1) / float(figure.dpi)


def legend_span_inch(figure: Any, axis: Any, cax: Any) -> tuple[float, float]:
    """圖例允許的水平範圍（英吋）：主圖框左緣到色條（含刻度與標籤）右緣；圖例框不可超出，並置中於此範圍（v65）。"""

    width = float(figure.get_size_inches()[0])
    return float(axis.get_position().x0) * width, colourbar_right_inch(figure, cax)


def legend_below(
    figure: Any, height: float, handles: Sequence[Any], labels: Sequence[str], ncol: int = 3, *, framed: bool = False, center_x_inch: float | None = None,
    span_inch: tuple[float, float] | None = None,
) -> Any | None:
    """在畫布底部（固定位置）畫圖例；沒有項目時回傳 ``None``。

    預設無外框、置中於畫布。``framed=True``（簡要報告大圖，對齊 LBT 圖例）：有外框（matplotlib 預設圓角、邊框 0.8 灰、白底、不透明度 0.8），
    並以 ``center_x_inch`` 置中於主地圖框（LBT 是置中於主圖框下方）。

    v65：``span_inch=(左, 右)`` 優先於 ``center_x_inch``：圖例框置中於該範圍（主圖框左緣到色條右緣），畫出後量測圖例框，
    超出範圍就拋出 ``ValueError``（寧可不出圖，也不出超出主圖框的圖例）；圖例框的位置與列、欄數記在 ``figure._traps_legend``（版面紀錄 ``concise.legend``）。
    """

    if not handles:
        return None
    if span_inch is not None:
        center_x_inch = 0.5 * (float(span_inch[0]) + float(span_inch[1]))
    center = 0.5 if center_x_inch is None else float(center_x_inch) / figure.get_size_inches()[0]
    extra: dict[str, Any] = {"frameon": False}
    if framed:
        extra = {"frameon": True, "fancybox": True, "edgecolor": "0.8", "facecolor": "white", "framealpha": 0.8}
    columns = max(1, min(int(ncol), len(handles)))
    legend = figure.legend(
        list(handles),
        list(labels),
        loc="lower center",
        bbox_to_anchor=(center, LEGEND_BASE_INCH / height),
        ncol=columns,
        fontsize=TEXT_FONT_PT,
        columnspacing=0.9,
        handlelength=1.6,
        **extra,
    )
    if span_inch is not None:
        renderer = figure.canvas.get_renderer()
        box = legend.get_window_extent(renderer)
        left, right = float(box.x0) / float(figure.dpi), float(box.x1) / float(figure.dpi)
        # 量測用 figure.dpi（100）的字形寬度；存成 SVG（72 dpi）時字形寬度略有不同（實測框寬差約 0.05 英吋），所以左右都要預留 :data:`LEGEND_SPAN_SAFETY_INCH`
        if left < float(span_inch[0]) + LEGEND_SPAN_SAFETY_INCH or right > float(span_inch[1]) - LEGEND_SPAN_SAFETY_INCH:
            raise ValueError(f"圖例框 {left:.3f}–{right:.3f} 英吋沒有落在主圖框左緣到色條右緣的範圍 {span_inch[0]:.3f}–{span_inch[1]:.3f} 英吋內（兩側各須預留 {LEGEND_SPAN_SAFETY_INCH} 英吋）")
        figure._traps_legend = {  # noqa: SLF001  供 concise_style.concise_checks_of 寫入版面紀錄
            "frame_x_inch": [round(left, 4), round(right, 4)], "span_inch": [round(float(span_inch[0]), 4), round(float(span_inch[1]), 4)],
            "columns": columns, "rows": -(-len(handles) // columns), "entries": len(handles),
        }
    return legend


def add_box_axes(figure: Any, box: Box, **kwargs: Any) -> Any:
    """依英吋 :class:`Box` 在 Figure 上建立 axes。"""

    width, height = figure.get_size_inches()
    return figure.add_axes((box.x / width, box.y / height, box.w / width, box.h / height), **kwargs)


# ---------------------------------------------------------------------------
# 版面 gate
# ---------------------------------------------------------------------------

_KIND_PATTERN = re.compile(r"(?:^|[\\/])(?:site_[A-Za-z0-9]+_(?P<kind>[A-Za-z0-9_]+)|(?P<overview>domainA_overview))\.png$")


def kind_of_path(path: str) -> str | None:
    """由圖檔路徑推得圖種（``site_<id>_<kind>.png`` 或 ``domainA_overview.png``）。"""

    match = _KIND_PATTERN.search(str(path))
    if match is None:
        return None
    return match.group("overview") or match.group("kind")


def _sorted_boxes(boxes: Iterable[Sequence[float]]) -> list[list[float]]:
    return sorted(([float(v) for v in box] for box in boxes), key=lambda b: (-round(b[1] + b[3], 2), round(b[0], 2), round(b[2], 3)))


def _main_box(boxes: Sequence[Sequence[float]]) -> list[float]:
    return list(max(boxes, key=lambda b: float(b[2]) * float(b[3])))


def _close(first: Sequence[float], second: Sequence[float], tolerance: float) -> bool:
    return len(first) == len(second) and all(abs(float(a) - float(b)) <= tolerance for a, b in zip(first, second, strict=True))


def standard_layout_failures(
    records: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = TOLERANCE_INCH,
    layout: Mapping[str, tuple[str, str]] | None = None,
    crop_top_inch: float = 0.0,
) -> list[str]:
    """檢查圖面統一標準，回傳違規說明（空串列代表全部通過）。

    規則（容差 ``tolerance`` 英吋）：

    1. 每張圖的圖種必須登錄於 :data:`KIND_LAYOUT`；
    2. 畫布必須等於所屬家族的標準畫布（single 6.3×4.73、multi 6.3×7.0）；
    3. 同一圖種的所有圖（五站 × 兩產品）座標軸框個數與位置、大小完全相同；
    4. 同一版面類別（例如 ``map1``）內不同圖種的主座標軸框也完全相同。

    記錄需含 ``path``（或 ``relative_path``）、``figure_size_inch`` 與 ``axes_boxes_inch``
    （各可見 axes 的 ``[x, y, w, h]`` 英吋）；缺欄位視為違規。
    """

    kind_layout = layout or KIND_LAYOUT
    errors: list[str] = []
    by_kind: dict[str, list[tuple[str, list[list[float]]]]] = {}
    for record in records:
        label = str(record.get("relative_path") or record.get("path") or "?")
        kind = kind_of_path(label)
        if kind is None or kind not in kind_layout:
            errors.append(f"{label}: 圖種未登錄於統一標準")
            continue
        size = record.get("figure_size_inch")
        boxes = record.get("axes_boxes_inch")
        if not size or not boxes:
            errors.append(f"{label}: 缺 figure_size_inch 或 axes_boxes_inch")
            continue
        expected = canvas_of(kind, kind_layout, crop_top_inch=crop_top_inch)
        if not _close(size, expected, tolerance):
            errors.append(f"{label}: 畫布 {size[0]:.3f}×{size[1]:.3f} 不是 {family_of(kind, kind_layout)} 家族的 {expected[0]:g}×{expected[1]:g}")
        by_kind.setdefault(kind, []).append((label, _sorted_boxes(boxes)))
    for kind, items in sorted(by_kind.items()):
        reference_label, reference = items[0]
        for label, boxes in items[1:]:
            if len(boxes) != len(reference):
                errors.append(f"{label}: axes 個數 {len(boxes)} 與 {reference_label} 的 {len(reference)} 不同")
                continue
            for index, (got, want) in enumerate(zip(boxes, reference, strict=True)):
                if not _close(got, want, tolerance):
                    errors.append(f"{label}: 第 {index} 個座標軸框 {[round(v, 3) for v in got]} 與 {reference_label} 的 {[round(v, 3) for v in want]} 不同")
                    break
    by_class: dict[str, list[tuple[str, list[float]]]] = {}
    for kind, items in by_kind.items():
        layout_class = kind_layout[kind][1]
        by_class.setdefault(layout_class, []).append((items[0][0], _main_box(items[0][1])))
    for layout_class, items in sorted(by_class.items()):
        reference_label, reference = items[0]
        for label, box in items[1:]:
            if not _close(box, reference, tolerance):
                errors.append(f"{label}: 版面類別 {layout_class} 的主座標軸框 {[round(v, 3) for v in box]} 與 {reference_label} 的 {[round(v, 3) for v in reference]} 不同")
    return errors


def standard_layout_summary(
    records: Sequence[Mapping[str, Any]], layout: Mapping[str, tuple[str, str]] | None = None, *, crop_top_inch: float = 0.0
) -> dict[str, Any]:
    """各圖種的畫布、張數與主座標軸框，供 gate JSON 記錄。"""

    kind_layout = layout or KIND_LAYOUT
    summary: dict[str, dict[str, Any]] = {}
    for record in records:
        label = str(record.get("relative_path") or record.get("path") or "?")
        kind = kind_of_path(label)
        if kind is None or kind not in kind_layout:
            continue
        entry = summary.setdefault(
            kind,
            {
                "family": family_of(kind, kind_layout),
                "layout_class": kind_layout[kind][1],
                "crop_top_inch": crop_top_inch,
                "expected_canvas_inch": [round(v, 3) for v in canvas_of(kind, kind_layout, crop_top_inch=crop_top_inch)],
                "canvas_inch": [round(float(v), 3) for v in record.get("figure_size_inch", [])],
                "main_axes_inch": [round(v, 3) for v in _main_box(record["axes_boxes_inch"])] if record.get("axes_boxes_inch") else None,
                "count": 0,
            },
        )
        entry["count"] += 1
    return summary


__all__ = [
    "Box",
    "CONCISE_KIND_LAYOUT",
    "CONCISE_MAP_AXES",
    "CONCISE_MAP_CBAR",
    "CONCISE_MAP_HEIGHT_INCH",
    "CONCISE_MAP_LAT_SPAN",
    "CONCISE_MAP_LON_SPAN",
    "CONCISE_MAP_REFERENCE_LAT_DEG",
    "CONCISE_MAP_WIDTH_INCH",
    "CONCISE_ZOOM_AXES",
    "CONCISE_ZOOM_CBAR",
    "CONCISE_ZOOM_LAT_SPAN",
    "CONCISE_ZOOM_LON_SPAN",
    "CONCISE_ZOOM_WIDTH_INCH",
    "CONCISE_LEG3_AXES",
    "CONCISE_LEG3_CBAR",
    "CONCISE_LEG3_CLASS",
    "CONCISE_LEG3_EXTRA_INCH",
    "CONCISE_LEG3_KINDS",
    "CONCISE_LEG3_ROWS",
    "CONCISE_LEGEND_ROWS",
    "colourbar_right_inch",
    "legend_span_inch",
    "CONCISE_CBAR_PAD_INCH",
    "CONCISE_CBAR_WIDTH_INCH",
    "LBT_COAST_COLOUR",
    "LBT_COAST_WIDTH_PT",
    "LBT_PRINT_SCALE",
    "CONCISE_TITLE_CROP_INCH",
    "LIFETIME_AXES_Y_INCH",
    "LIFETIME_CANVAS_HEIGHT_INCH",
    "FAMILY_CANVAS",
    "KIND_LAYOUT",
    "MAP1_AXES",
    "MAP1_CBAR",
    "MULTI_HEIGHT_INCH",
    "SINGLE_HEIGHT_INCH",
    "TOLERANCE_INCH",
    "WIDTH_INCH",
    "add_box_axes",
    "canvas_of",
    "draw_title_block",
    "family_of",
    "fit_limits_to_box",
    "kind_of_path",
    "legend_below",
    "limits_slices",
    "standard_layout_failures",
    "standard_layout_summary",
    "visual_width",
    "wrap_subtitle",
]
