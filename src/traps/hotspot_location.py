"""熱點／長壽命路徑之位置描述量（區域占比、富集度、連通塊幾何、陸地連通塊分類）。

本模組只做格網上的幾何與計數，供報告中「熱點位於某處」類敘述以數字查核；不參與 TRAP
偵測、追蹤或熱點定義。所有函式皆為純 numpy，輸入為 aggregate 的 ``spatial_all.npz`` 欄位
（``lon``、``lat``、``site_mask`` 與各種遮罩／計數）。

名詞
----
* 特徵權重 ``weight``：布林遮罩（熱點格）或格點計數（如 ``long_track_count``）；
  區域占比 = 區域內權重和 / 全站權重和。
* 有效格點 ``valid``：站內分析格（``site_mask``）。區域有效占比 = 區域內有效格數 / 全站有效格數。
* 富集度 ``enrichment`` = 特徵占比 / 有效占比；> 1 表示特徵在該區域比均勻分布更集中。
* 距離以等距圓柱近似換算為 km（1 度緯度 = 110.574 km，1 度經度 = 111.320·cos φ km），
  對 1.5 度寬之 OCM 域誤差小於 1%。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320


def grid_coordinates(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """回傳與 ``(lat, lon)`` 二維陣列同形的經度、緯度網格。"""

    return np.meshgrid(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def to_km(lon2d: np.ndarray, lat2d: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """經緯度轉為相對 ``(lon0, lat0)`` 的東向、北向 km（等距圓柱近似）。"""

    x = (np.asarray(lon2d, dtype=float) - lon0) * KM_PER_DEG_LON_EQ * np.cos(np.deg2rad(lat0))
    y = (np.asarray(lat2d, dtype=float) - lat0) * KM_PER_DEG_LAT
    return x, y


def region_stats(weight: np.ndarray, valid: np.ndarray, region: np.ndarray) -> dict[str, float]:
    """某區域內之特徵占比、有效格占比與富集度。

    ``region`` 先與 ``valid`` 取交集，因此區域可直接以經緯度條件定義。特徵總和為 0 時占比為 NaN。
    """

    w = np.asarray(weight, dtype=float)
    v = np.asarray(valid, dtype=bool)
    r = np.asarray(region, dtype=bool) & v
    w_total = float(w[v].sum())
    n_valid = int(v.sum())
    w_in = float(w[r].sum())
    n_in = int(r.sum())
    share = w_in / w_total if w_total > 0 else float("nan")
    area_share = n_in / n_valid if n_valid > 0 else float("nan")
    enrichment = share / area_share if area_share and area_share > 0 and np.isfinite(share) else float("nan")
    return {
        "n_feature_in": w_in, "n_feature_total": w_total, "n_valid_in": n_in, "n_valid_total": n_valid,
        "share_feature": share, "share_valid": area_share, "enrichment": enrichment,
    }


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """連通塊標記（4 或 8 連通）；回傳 ``(labels, n)``，背景為 0，標記由 1 起算。"""

    if connectivity not in (4, 8):
        raise ValueError("connectivity 必須是 4 或 8")
    m = np.asarray(mask, dtype=bool)
    labels = np.zeros(m.shape, dtype=np.int32)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    n = 0
    rows, cols = m.shape
    for r0, c0 in zip(*np.nonzero(m)):
        if labels[r0, c0]:
            continue
        n += 1
        labels[r0, c0] = n
        queue = deque([(int(r0), int(c0))])
        while queue:
            r, c = queue.popleft()
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and m[rr, cc] and not labels[rr, cc]:
                    labels[rr, cc] = n
                    queue.append((rr, cc))
    return labels, n


def orientation_and_elongation(x_km: np.ndarray, y_km: np.ndarray) -> dict[str, float]:
    """一群格點中心的主軸方向（自正東逆時針，0–180°）、伸長比與主軸長度。

    以座標共變異數矩陣之特徵分解（PCA）求主軸；每格加 1/12 km² 的格內變異（1 km 方格），
    使單列或單點之塊體不產生 0 變異。伸長比 = sqrt(λ1/λ2)；長度 = sqrt(12·λ1)（均勻長條之全長）。
    """

    x = np.asarray(x_km, dtype=float)
    y = np.asarray(y_km, dtype=float)
    if x.size == 1:
        return {"angle_from_east_deg": float("nan"), "elongation": 1.0, "length_km": 1.0, "width_km": 1.0}
    cov = np.cov(np.vstack([x, y]), bias=True) + np.eye(2) / 12.0
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, 1]
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    return {
        "angle_from_east_deg": angle,
        "elongation": float(np.sqrt(vals[1] / vals[0])),
        "length_km": float(np.sqrt(12.0 * vals[1])),
        "width_km": float(np.sqrt(12.0 * vals[0])),
    }


def is_east_west(angle_from_east_deg: float, tolerance_deg: float = 30.0) -> bool:
    """主軸是否與東西向相差不超過 ``tolerance_deg``。"""

    if not np.isfinite(angle_from_east_deg):
        return False
    a = angle_from_east_deg % 180.0
    return a <= tolerance_deg or a >= 180.0 - tolerance_deg


def component_table(mask: np.ndarray, lon: np.ndarray, lat: np.ndarray, connectivity: int = 8,
                    min_cells: int = 1) -> list[dict[str, Any]]:
    """各連通塊之格數、中心經緯度與 PCA 幾何，依格數由大到小排序。"""

    lon2d, lat2d = grid_coordinates(lon, lat)
    labels, n = label_components(mask, connectivity)
    lon0, lat0 = float(np.mean(lon2d)), float(np.mean(lat2d))
    x, y = to_km(lon2d, lat2d, lon0, lat0)
    rows = []
    for k in range(1, n + 1):
        sel = labels == k
        size = int(sel.sum())
        if size < min_cells:
            continue
        geo = orientation_and_elongation(x[sel], y[sel])
        rows.append({
            "n_cells": size, "centroid_lon": float(lon2d[sel].mean()), "centroid_lat": float(lat2d[sel].mean()),
            "lon_min": float(lon2d[sel].min()), "lon_max": float(lon2d[sel].max()),
            "lat_min": float(lat2d[sel].min()), "lat_max": float(lat2d[sel].max()), **geo,
        })
    rows.sort(key=lambda r: -r["n_cells"])
    return rows


def land_components(ocean: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """陸地（或遮罩）格之 8 連通塊，並標示碰到網格邊界者。

    回傳 ``(labels, touches_border, n)``；``touches_border[k]`` 對應標記 ``k``（0 為背景恆為 False）。
    碰到邊界者視為本島或大陸海岸（延伸出分析域外），其餘為域內島嶼。
    """

    land = ~np.asarray(ocean, dtype=bool)
    labels, n = label_components(land, 8)
    touches = np.zeros(n + 1, dtype=bool)
    edge = np.zeros_like(labels, dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    for k in np.unique(labels[edge]):
        if k:
            touches[k] = True
    return labels, touches, n


def nearest_land_label(cells_rc: tuple[np.ndarray, np.ndarray], land_labels: np.ndarray, x_km: np.ndarray,
                       y_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每個格點最近之陸地連通塊標記與距離（km，格點中心至最近陸地格中心）。"""

    lr, lc = np.nonzero(land_labels)
    lx, ly = x_km[lr, lc], y_km[lr, lc]
    lab = land_labels[lr, lc]
    rr, cc = cells_rc
    out_label = np.zeros(rr.size, dtype=np.int32)
    out_dist = np.zeros(rr.size, dtype=float)
    for start in range(0, rr.size, 2000):
        sl = slice(start, start + 2000)
        d2 = (x_km[rr[sl], cc[sl]][:, None] - lx[None, :]) ** 2 + (y_km[rr[sl], cc[sl]][:, None] - ly[None, :]) ** 2
        j = np.argmin(d2, axis=1)
        out_label[sl] = lab[j]
        out_dist[sl] = np.sqrt(d2[np.arange(j.size), j])
    return out_label, out_dist


def within_km_of_point(x_km: np.ndarray, y_km: np.ndarray, px: float, py: float, radius_km: float) -> np.ndarray:
    """距某點 ``radius_km`` 以內之格點遮罩。"""

    return (np.asarray(x_km) - px) ** 2 + (np.asarray(y_km) - py) ** 2 <= radius_km ** 2


def lonlat_box(lon2d: np.ndarray, lat2d: np.ndarray, lon_min: float = -np.inf, lon_max: float = np.inf,
               lat_min: float = -np.inf, lat_max: float = np.inf) -> np.ndarray:
    """經緯度方框遮罩（含邊界）。"""

    return (lon2d >= lon_min) & (lon2d <= lon_max) & (lat2d >= lat_min) & (lat2d <= lat_max)


def verdict_by_rule(share: float, enrichment: float, *, majority: float = 0.50, enrich_hi: float = 1.5,
                    enrich_lo: float = 1.2, meaningful: float = 0.25) -> str:
    """區域敘述之判定規則（見 ``scripts/hotspot_location_descriptors.py`` 頂端說明）。

    * supported：占比 >= majority 且富集度 >= enrich_lo；或富集度 >= enrich_hi 且占比 >= meaningful。
    * partly：占比 >= majority 但富集度 < enrich_lo（多數僅因區域面積大）。
    * unsupported：其餘。
    """

    if not (np.isfinite(share) and np.isfinite(enrichment)):
        return "unsupported"
    if (share >= majority and enrichment >= enrich_lo) or (enrichment >= enrich_hi and share >= meaningful):
        return "supported"
    if share >= majority:
        return "partly"
    return "unsupported"


def present_rule(share: float, enrichment: float, *, clear: float = 0.20, enrich_hi: float = 1.5, floor: float = 0.10) -> str:
    """多區域敘述（「分布於 R1 與 R2」「並存」）中單一區域之判定：

    supported：占比 >= clear（20%），或富集度 >= enrich_hi（1.5）且占比 >= floor（10%）；否則 unsupported。
    """

    if not (np.isfinite(share) and np.isfinite(enrichment)):
        return "unsupported"
    return "supported" if (share >= clear or (enrichment >= enrich_hi and share >= floor)) else "unsupported"
