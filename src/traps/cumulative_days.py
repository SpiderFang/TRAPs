"""逐格「累積天數」：TRAP 曲線通過某格的有效日數。

``aggregate`` 的 ``spatial_<stratum>.npz`` 儲存的是 ``curve_pass_frequency``
（= 通過次數 / 該格有效快照數）與每格有效快照數 ``n_valid``，並未直接存通過次數。
曲線通過次數與有效快照數都是整數，故 ``frequency * n_valid`` 必為（浮點誤差內的）
整數；本模組把它還原成整數日數，並在偏離整數過大時拒絕，避免把不一致的輸入畫成圖。
逐日產品中每個有效快照就是一個有效日，因此還原值即「該格被曲線通過的日數」。

本模組只做顯示用的衍生量，不回寫、不改變任何 aggregate 輸出。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

INTEGER_TOLERANCE = 1.0e-6
"""``frequency * n_valid`` 與最近整數的容許誤差（以計數為單位）。"""


def cumulative_pass_days(curve_pass_frequency: np.ndarray, n_valid: np.ndarray) -> np.ndarray:
    """由通過頻率與每格有效快照數還原每格通過日數（float64 陣列，缺值為 NaN）。

    ``n_valid <= 0`` 或頻率為 NaN 的格點回傳 NaN（站外、無有效快照）。其餘格點的
    ``frequency * n_valid`` 必須落在最近整數的 :data:`INTEGER_TOLERANCE` 內，且不得為負
    或超過該格 ``n_valid``，否則丟出 ``ValueError``。
    """

    frequency = np.asarray(curve_pass_frequency, dtype=np.float64)
    valid_count = np.asarray(n_valid)
    if frequency.shape != valid_count.shape:
        raise ValueError("curve_pass_frequency 與 n_valid 的形狀必須相同")
    usable = np.isfinite(frequency) & (valid_count > 0)
    product = np.where(usable, frequency * valid_count, np.nan)
    counts = np.rint(product)
    if np.any(np.abs(product[usable] - counts[usable]) > INTEGER_TOLERANCE):
        raise ValueError("curve_pass_frequency * n_valid 不是整數，無法還原通過日數")
    if np.any(counts[usable] < 0) or np.any(counts[usable] > valid_count[usable]):
        raise ValueError("還原的通過日數超出 [0, n_valid]")
    return counts


def site_valid_days(n_valid: np.ndarray, site_mask: np.ndarray) -> int:
    """站內有效日數：站點分析區內各格 ``n_valid`` 的最大值。

    同一站的有效快照在所有站內格點通常相同；若各格不同（例如部分格點缺值），取最大值
    作為「該站有效日數」並由 :func:`site_valid_days_summary` 記錄最小值供審查。
    """

    values = np.asarray(n_valid)[np.asarray(site_mask, dtype=bool)]
    return int(values.max()) if values.size else 0


def site_valid_days_summary(n_valid: np.ndarray, site_mask: np.ndarray) -> Mapping[str, Any]:
    """站內各格有效日數的最小／最大值與是否一致，供圖面副標題與 CSV 審查。"""

    values = np.asarray(n_valid)[np.asarray(site_mask, dtype=bool)]
    if values.size == 0:
        return {"min": 0, "max": 0, "uniform": True}
    return {"min": int(values.min()), "max": int(values.max()), "uniform": bool(values.min() == values.max())}


__all__ = ["cumulative_pass_days", "site_valid_days", "site_valid_days_summary"]
