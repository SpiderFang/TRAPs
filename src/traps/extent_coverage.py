"""地圖範圍的空白面積：資料格（分析範圍）、陸地、其餘空白（分析範圍外的海）。

簡要報告的地圖範圍要在「覆蓋資料外接框、兩端標示刻度、固定軸框長寬比」的前提下讓空白最小；這裡把「有東西可看」
（分析範圍內的資料格，或陸地）畫在 0.01° 的點陣上，以積分影像在 O(1) 內得到任一矩形範圍內各類別的比例。
陸地由海岸線 Polygon 環以 XOR 填色（外環與內環交錯）取得，只用來判斷範圍內哪些部分不算空白，不改任何圖面內容。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

RASTER_STEP_DEG = 0.01
"""點陣解析度（度）；範圍的起點在 0.05° 格線上，因此範圍邊界與點陣格線對齊。"""

REGION_MARGIN_DEG = 2.5
"""點陣涵蓋資料外接框外擴的距離（度）；範圍候選都落在這個區域內。"""


def _land_raster(coastline: Any, x0: float, y1: float, nx: int, ny: int, step: float) -> np.ndarray:
    """陸地點陣 ``(ny, nx)``（第 0 列在北端）；海岸線為空時全為 ``False``。"""

    from PIL import Image, ImageDraw

    land = np.zeros((ny, nx), dtype=bool)
    if coastline is None:
        return land
    x1, y0 = x0 + nx * step, y1 - ny * step
    for ring in coastline.rings:
        if ring[:, 0].max() < x0 or ring[:, 0].min() > x1 or ring[:, 1].max() < y0 or ring[:, 1].min() > y1:
            continue
        image = Image.new("1", (nx, ny), 0)
        ImageDraw.Draw(image).polygon([((float(lon) - x0) / step, (y1 - float(lat)) / step) for lon, lat in ring], fill=1, outline=1)
        land ^= np.asarray(image, dtype=bool)
    return land


def _grid_raster(lon: np.ndarray, lat: np.ndarray, mask: np.ndarray, x0: float, y1: float, nx: int, ny: int, step: float) -> np.ndarray:
    """資料格遮罩（站內分析區）在點陣上的對應；點陣格心落在哪一格就取該格的值，網格外為 ``False``。"""

    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    d_lon, d_lat = float(np.median(np.diff(lon))), float(np.median(np.diff(lat)))
    centres_x = x0 + (np.arange(nx) + 0.5) * step
    centres_y = y1 - (np.arange(ny) + 0.5) * step
    cols = np.floor((centres_x - (lon[0] - d_lon / 2.0)) / d_lon).astype(int)
    rows = np.floor((centres_y - (lat[0] - d_lat / 2.0)) / d_lat).astype(int)
    ok_c = (cols >= 0) & (cols < lon.size)
    ok_r = (rows >= 0) & (rows < lat.size)
    out = np.zeros((ny, nx), dtype=bool)
    sub = np.asarray(mask, dtype=bool)[np.clip(rows, 0, lat.size - 1)[:, None], np.clip(cols, 0, lon.size - 1)[None, :]]
    out[:] = sub & ok_r[:, None] & ok_c[None, :]
    return out


class Coverage:
    """資料／陸地／空白的點陣與積分影像；``fractions`` 回傳任一矩形範圍內各類別的面積比例。"""

    def __init__(self, data: np.ndarray, land: np.ndarray, x0: float, y1: float, step: float) -> None:
        self.x0, self.y1, self.step = float(x0), float(y1), float(step)
        self.ny, self.nx = data.shape
        self._data = self._integral(data)
        self._useful = self._integral(data | land)
        self._land = self._integral(land)

    @staticmethod
    def _integral(mask: np.ndarray) -> np.ndarray:
        return np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def _count(self, table: np.ndarray, xl: float, xh: float, yl: float, yh: float) -> int:
        c0 = min(max(round((xl - self.x0) / self.step), 0), self.nx)
        c1 = min(max(round((xh - self.x0) / self.step), 0), self.nx)
        r0 = min(max(round((self.y1 - yh) / self.step), 0), self.ny)
        r1 = min(max(round((self.y1 - yl) / self.step), 0), self.ny)
        return int(table[r1, c1] - table[r0, c1] - table[r1, c0] + table[r0, c0])

    def useful_fraction(self, xl: float, xh: float, yl: float, yh: float) -> float:
        total = max(round((xh - xl) / self.step) * round((yh - yl) / self.step), 1)
        return self._count(self._useful, xl, xh, yl, yh) / total

    def fractions(self, xlim: Sequence[float], ylim: Sequence[float]) -> dict[str, float]:
        """``data``（資料格）、``land_only``（陸地且非資料格）、``blank``（兩者以外：分析範圍外的海）各佔範圍面積的比例。"""

        xl, xh, yl, yh = float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1])
        total = max(round((xh - xl) / self.step) * round((yh - yl) / self.step), 1)
        data = self._count(self._data, xl, xh, yl, yh) / total
        useful = self._count(self._useful, xl, xh, yl, yh) / total
        return {"data": round(data, 4), "land_only": round(useful - data, 4), "blank": round(1.0 - useful, 4)}


def build_coverage(
    lon: np.ndarray, lat: np.ndarray, mask: np.ndarray, coastline: Any, bbox: Sequence[float], *, margin: float = REGION_MARGIN_DEG, step: float = RASTER_STEP_DEG
) -> Coverage:
    """以 ``bbox = (x_low, x_high, y_low, y_high)`` 外擴 ``margin`` 度為區域（起點對齊 0.05° 格線）建立 :class:`Coverage`。"""

    x0 = math.floor((bbox[0] - margin) / 0.05) * 0.05
    x1 = math.ceil((bbox[1] + margin) / 0.05) * 0.05
    y0 = math.floor((bbox[2] - margin) / 0.05) * 0.05
    y1 = math.ceil((bbox[3] + margin) / 0.05) * 0.05
    nx, ny = round((x1 - x0) / step), round((y1 - y0) / step)
    data = _grid_raster(lon, lat, mask, x0, y1, nx, ny, step)
    land = _land_raster(coastline, x0, y1, nx, ny, step)
    return Coverage(data, land, x0, y1, step)
