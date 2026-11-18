"""逐時（``r1_hourly``）與逐日（``d1_daily25h``）曲線通過頻率熱點的重疊統計。

熱點定義完全沿用 :func:`traps.aggregate._top_quantile_mask`：只用該站分析區內、非 NaN
的格點，取 ``curve_pass_frequency`` 的 ``hotspot_quantile`` 分位數為門檻，格點須
``>= 門檻`` 且 ``> 0``。這裡不做圖面用的小連通塊濾除（那只影響 aggregate 的繪圖）。

兩個產品必須在同一個格網上才能逐格比較：:func:`check_same_grid` 逐值比對 ``lon``、
``lat`` 與 ``site_mask``。重疊指標包含：

- 交集、聯集、Jaccard；
- 「逐時熱點格點中同時為逐日熱點的比例」與反向比例；
- 容許 k 格（Chebyshev 距離，k = 1、2；約 1、2 km）的版本：逐時熱點格點中，k 格內有
  逐日熱點者的比例，反向亦然；
- 隨機基準：若兩套熱點在有效格點中獨立隨機分布，上述比例的期望值約等於對方熱點
  （或其 k 格膨脹集合）佔有效格點的比例；Jaccard 的期望為 ``fh*fd / (fh + fd - fh*fd)``。

本模組只讀取 aggregate 輸出、不寫回任何科學產物。
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .aggregate import _top_quantile_mask

TOLERANCES_CELLS = (1, 2)
"""容許距離（格；Chebyshev）。1 格約 1 km、2 格約 2 km。"""

GRID_KEYS = ("lon", "lat", "site_mask")
"""兩產品必須逐值相同的格網欄位。"""

CSV_COLUMNS = (
    "site_id", "hotspot_quantile", "n_valid_cells", "n_hourly", "n_daily", "n_intersection", "n_union", "jaccard",
    "pct_hourly_in_daily", "pct_daily_in_hourly",
    "pct_hourly_within_1_of_daily", "pct_daily_within_1_of_hourly",
    "pct_hourly_within_2_of_daily", "pct_daily_within_2_of_hourly",
    "chance_hourly_fraction", "chance_daily_fraction", "chance_jaccard",
    "chance_pct_hourly_in_daily", "chance_pct_daily_in_hourly",
    "chance_pct_hourly_within_1_of_daily", "chance_pct_daily_within_1_of_hourly",
    "chance_pct_hourly_within_2_of_daily", "chance_pct_daily_within_2_of_hourly",
    "hourly_threshold", "daily_threshold", "same_grid", "stored_masks_match",
)
"""CSV 欄位順序；百分比以 0–100 表示，無法計算時留空。"""


def top_quantile_hotspots(curve_pass_frequency: np.ndarray, site_mask: np.ndarray, quantile: float) -> tuple[np.ndarray, float | None]:
    """與 aggregate 完全相同的熱點遮罩與門檻（未濾除小連通塊）。"""

    return _top_quantile_mask(curve_pass_frequency, site_mask, quantile)


def dilate_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """以 ``(2r+1) x (2r+1)`` 方形核（Chebyshev 距離 <= r）膨脹布林遮罩；純 numpy。"""

    base = np.asarray(mask, dtype=bool)
    radius = int(radius_cells)
    if radius < 0:
        raise ValueError("radius_cells 必須 >= 0")
    if radius == 0 or not base.any():
        return base.copy()
    ny, nx = base.shape
    padded = np.zeros((ny + 2 * radius, nx + 2 * radius), dtype=bool)
    padded[radius : radius + ny, radius : radius + nx] = base
    result = np.zeros_like(base)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + ny, dx : dx + nx]
    return result


def check_same_grid(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """逐值比對兩個產品的 ``lon``、``lat``、``site_mask``；回傳是否相同與最大座標差。"""

    result: dict[str, Any] = {"same": True, "max_abs_diff": {}}
    for key in GRID_KEYS:
        a = np.asarray(first[key])
        b = np.asarray(second[key])
        if a.shape != b.shape:
            result["same"] = False
            result["max_abs_diff"][key] = None
            continue
        if key == "site_mask":
            equal = bool(np.array_equal(a, b))
            result["max_abs_diff"][key] = 0.0 if equal else float(np.count_nonzero(a != b))
        else:
            diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
            equal = bool(np.array_equal(a, b))
            result["max_abs_diff"][key] = diff
        result["same"] = bool(result["same"] and equal)
    return result


def _pct(numerator: int, denominator: int) -> float | None:
    """百分比（0–100）；分母為 0 時回傳 ``None``。"""

    return None if denominator == 0 else 100.0 * numerator / denominator


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def overlap_statistics(hourly: np.ndarray, daily: np.ndarray, valid: np.ndarray, *, tolerances: Sequence[int] = TOLERANCES_CELLS) -> dict[str, Any]:
    """兩組熱點遮罩的重疊統計（皆限制在 ``valid`` 格點內）。

    ``valid`` 是計算熱點分位數的母體（站內且頻率有限的格點）。回傳欄位見模組說明與
    :data:`CSV_COLUMNS`；隨機基準中的 k 格版本是「對方熱點膨脹 k 格後，落在有效格點的
    比例」，因為膨脹集合本身佔有效格點的比例才是獨立隨機分布時的命中機率。
    """

    valid_mask = np.asarray(valid, dtype=bool)
    hourly_mask = np.asarray(hourly, dtype=bool) & valid_mask
    daily_mask = np.asarray(daily, dtype=bool) & valid_mask
    n_valid = int(valid_mask.sum())
    n_hourly = int(hourly_mask.sum())
    n_daily = int(daily_mask.sum())
    inter = int((hourly_mask & daily_mask).sum())
    union = int((hourly_mask | daily_mask).sum())
    fraction_hourly = _fraction(n_hourly, n_valid)
    fraction_daily = _fraction(n_daily, n_valid)
    chance_jaccard = None
    if fraction_hourly is not None and fraction_daily is not None:
        denominator = fraction_hourly + fraction_daily - fraction_hourly * fraction_daily
        chance_jaccard = fraction_hourly * fraction_daily / denominator if denominator > 0 else None
    stats: dict[str, Any] = {
        "n_valid_cells": n_valid,
        "n_hourly": n_hourly,
        "n_daily": n_daily,
        "n_intersection": inter,
        "n_union": union,
        "jaccard": _fraction(inter, union),
        "pct_hourly_in_daily": _pct(inter, n_hourly),
        "pct_daily_in_hourly": _pct(inter, n_daily),
        "chance_hourly_fraction": fraction_hourly,
        "chance_daily_fraction": fraction_daily,
        "chance_jaccard": chance_jaccard,
        "chance_pct_hourly_in_daily": None if fraction_daily is None else 100.0 * fraction_daily,
        "chance_pct_daily_in_hourly": None if fraction_hourly is None else 100.0 * fraction_hourly,
    }
    for radius in tolerances:
        daily_near = dilate_mask(daily_mask, radius)
        hourly_near = dilate_mask(hourly_mask, radius)
        stats[f"pct_hourly_within_{radius}_of_daily"] = _pct(int((hourly_mask & daily_near).sum()), n_hourly)
        stats[f"pct_daily_within_{radius}_of_hourly"] = _pct(int((daily_mask & hourly_near).sum()), n_daily)
        stats[f"chance_pct_hourly_within_{radius}_of_daily"] = _pct(int((daily_near & valid_mask).sum()), n_valid)
        stats[f"chance_pct_daily_within_{radius}_of_hourly"] = _pct(int((hourly_near & valid_mask).sum()), n_valid)
    return stats


def load_spatial_all(aggregate_dir: Path, site_id: str) -> dict[str, np.ndarray]:
    """讀取一個站的 ``spatial/<site>/spatial_all.npz`` 全部陣列（不使用 pickle）。"""

    path = Path(aggregate_dir) / "spatial" / site_id / "spatial_all.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def site_overlap(hourly_spatial: Mapping[str, np.ndarray], daily_spatial: Mapping[str, np.ndarray], *, site_id: str, quantile: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """一個站的重疊統計列與兩組遮罩。

    先確認兩產品同格網，再用 aggregate 的熱點定義重算兩組遮罩，並與 npz 內存的
    ``hotspot_curve_mask`` 逐值比對（``stored_masks_match``）。格網不同時直接丟出
    ``ValueError``：不同格網不能逐格比較。
    """

    grid = check_same_grid(hourly_spatial, daily_spatial)
    if not grid["same"]:
        raise ValueError(f"{site_id}: 逐時與逐日產品的格網不同，無法逐格比較: {grid['max_abs_diff']}")
    site_mask = np.asarray(hourly_spatial["site_mask"], dtype=bool)
    hourly_mask, hourly_threshold = top_quantile_hotspots(hourly_spatial["curve_pass_frequency"], site_mask, quantile)
    daily_mask, daily_threshold = top_quantile_hotspots(daily_spatial["curve_pass_frequency"], site_mask, quantile)
    stored_match = bool(
        np.array_equal(hourly_mask, np.asarray(hourly_spatial["hotspot_curve_mask"], dtype=bool))
        and np.array_equal(daily_mask, np.asarray(daily_spatial["hotspot_curve_mask"], dtype=bool))
    )
    hourly_valid = site_mask & np.isfinite(np.asarray(hourly_spatial["curve_pass_frequency"], dtype=np.float64))
    daily_valid = site_mask & np.isfinite(np.asarray(daily_spatial["curve_pass_frequency"], dtype=np.float64))
    # 兩產品的有效格點母體若不同，只在共同有效格點內比較，並在列中誠實記錄。
    valid = hourly_valid & daily_valid
    stats = overlap_statistics(hourly_mask, daily_mask, valid)
    row: dict[str, Any] = {
        "site_id": site_id,
        "hotspot_quantile": float(quantile),
        **stats,
        "hourly_threshold": hourly_threshold,
        "daily_threshold": daily_threshold,
        "same_grid": True,
        "stored_masks_match": stored_match,
        "n_valid_cells_hourly_pool": int(hourly_valid.sum()),
        "n_valid_cells_daily_pool": int(daily_valid.sum()),
        "grid_shape": [int(site_mask.shape[0]), int(site_mask.shape[1])],
        "grid_max_abs_diff": grid["max_abs_diff"],
    }
    return row, {"hourly": hourly_mask & valid, "daily": daily_mask & valid, "valid": valid}


def write_overlap_outputs(rows: Sequence[Mapping[str, Any]], output_dir: Path, *, metadata: Mapping[str, Any] | None = None) -> tuple[Path, Path]:
    """寫入 ``hotspot_overlap.csv`` 與 ``hotspot_overlap.json``（UTF-8、百分比 0–100）。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hotspot_overlap.csv"
    json_path = output_dir / "hotspot_overlap.json"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in CSV_COLUMNS})
    payload = {
        "schema_version": "traps.hotspot_overlap.v1",
        "metadata": dict(metadata or {}),
        "sites": [dict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return csv_path, json_path


def _json_default(value: Any) -> Any:
    """numpy 純量轉原生型別；NaN 不應出現（以 ``None`` 表示無法計算）。"""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"無法序列化 {type(value)}")


__all__ = [
    "CSV_COLUMNS",
    "TOLERANCES_CELLS",
    "check_same_grid",
    "dilate_mask",
    "load_spatial_all",
    "overlap_statistics",
    "site_overlap",
    "top_quantile_hotspots",
    "write_overlap_outputs",
]
