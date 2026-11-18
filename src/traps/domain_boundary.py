"""A 區兩個研究站（貢寮、龜山島）的「統計歸屬分界」：兩個 anchor 的球面中垂大圓。

``plot_domain_a_overview.py`` 的 A 區總覽與簡要報告的貢寮／龜山島熱點圖、四季重現圖共用同一條分界線與顏色。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

BOUNDARY_COLOUR = "#f4a300"
"""統計歸屬分界線的橘色。"""
BOUNDARY_LABEL = "統計歸屬分界"


def _unit(lon: Any, lat: Any) -> np.ndarray:
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def bisector_line(anchor_a: Sequence[float], anchor_b: Sequence[float], half_span_rad: float = 0.019, count: int = 6001) -> tuple[np.ndarray, np.ndarray]:
    """兩 anchor 的球面中垂大圓取樣點（lon, lat 度，東西向由西到東）。與 diagnostics 腳本同一公式。"""

    a, b = _unit(*map(float, anchor_a)), _unit(*map(float, anchor_b))
    normal = (a - b) / np.linalg.norm(a - b)
    mid = (a + b) / np.linalg.norm(a + b)
    tangent = np.cross(normal, mid)
    tangent /= np.linalg.norm(tangent)
    lon_mid = np.arctan2(mid[1], mid[0])
    if float(tangent @ np.array([-np.sin(lon_mid), np.cos(lon_mid), 0.0])) < 0.0:
        tangent = -tangent
    theta = np.linspace(-half_span_rad, half_span_rad, count)
    pts = np.cos(theta)[:, None] * mid[None, :] + np.sin(theta)[:, None] * tangent[None, :]
    return np.rad2deg(np.arctan2(pts[:, 1], pts[:, 0])), np.rad2deg(np.arcsin(np.clip(pts[:, 2], -1.0, 1.0)))


BRIDGE_GAP_SAMPLES = 200
"""遮罩邊緣的單格缺口（海岸線附近）造成的線段中斷：兩段之間不超過此取樣數（約 8 km）時接起來，避免出現孤立的小線段。"""


def _bridge_short_gaps(keep: np.ndarray, max_gap: int) -> np.ndarray:
    """把兩段 ``True`` 之間不長於 ``max_gap`` 個取樣的 ``False`` 缺口補成 ``True``（頭尾的 ``False`` 不補）。"""

    out = np.array(keep, dtype=bool, copy=True)
    idx = np.flatnonzero(out)
    if idx.size < 2:
        return out
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        start, stop = int(idx[g]) + 1, int(idx[g + 1])
        if stop - start <= max_gap:
            out[start:stop] = True
    return out


def boundary_in_analysis_area(
    anchor_a: Sequence[float], anchor_b: Sequence[float], lon: np.ndarray, lat: np.ndarray, analysis_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """分界線只保留落在分析海域（兩站站內遮罩的聯集）內的點，其餘為 NaN（線在陸地與網格外中斷）。"""

    from .period import _nearest_indices

    b_lon, b_lat = bisector_line(anchor_a, anchor_b)
    in_range = (b_lon >= lon.min()) & (b_lon <= lon.max()) & (b_lat >= lat.min()) & (b_lat <= lat.max())
    keep = np.zeros(b_lon.shape, dtype=bool)
    keep[in_range] = np.asarray(analysis_mask, dtype=bool)[_nearest_indices(lat, b_lat[in_range]), _nearest_indices(lon, b_lon[in_range])]
    keep = _bridge_short_gaps(keep, BRIDGE_GAP_SAMPLES)
    return np.where(keep, b_lon, np.nan), np.where(keep, b_lat, np.nan)
