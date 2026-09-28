"""五站 TRAP 出現率與結構統計。

出現率的統計母域是完整 flow domain；A 區兩站使用 ``site_masks`` 已完成的互斥
Voronoi 分割，B/C/D 則使用各自整個 domain。每個站的分母是逐時 coverage 達到明示
門檻的有效 snapshot 數，分子是其中至少一條合格 TRAP 曲線與該站有效格點相交的
snapshot 數。曲線交會會把離散折線加密到不大於半格距後逐點判斷，避免只檢查核心。

統計保留核心強度、曲線長度、平均強度、站點面積和面積標準化密度。零分母輸出
``None``，不以 0 假裝沒有出現；資料缺口、可用時刻沒有檢出與 coverage 不足則分開
計數，讓研究報告能區分物理結果和輸入資料限制。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import TrapDetection
from .tracking import TrackingFrame, densify_polyline


def _require(condition: bool, message: str) -> None:
    """在站點統計的資料契約不成立時立即停止。"""

    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SiteMetricsConfig:
    """站點有效 snapshot 與曲線交會的明示設定。

    ``coverage_threshold`` 是一個 snapshot 能進入該站分母的最小有效格點比例；它不會
    自動補缺、跨越 coastline 或把 data gap 變成零 TRAP。曲線採樣步長預設由輸入
    公尺格距的二分之一推得，亦可用較小值做更保守的交會敏感度。
    """

    coverage_threshold: float = 0.8
    curve_sample_step_m: float | None = None
    require_qualified: bool = True

    def __post_init__(self) -> None:
        """驗證 coverage 與曲線取樣門檻。"""

        if not np.isfinite(self.coverage_threshold) or not (0.0 <= self.coverage_threshold <= 1.0):
            raise ValueError("coverage_threshold 必須落在 0 到 1 之間")
        if self.curve_sample_step_m is not None and (
            not np.isfinite(self.curve_sample_step_m) or self.curve_sample_step_m <= 0.0
        ):
            raise ValueError("curve_sample_step_m 必須是正有限值或 None")

    def to_manifest_dict(self) -> dict[str, Any]:
        """回傳可保存至站點統計 manifest 的設定。"""

        return {
            "coverage_threshold": float(self.coverage_threshold),
            "curve_sample_step_m": None if self.curve_sample_step_m is None else float(self.curve_sample_step_m),
            "require_qualified": bool(self.require_qualified),
        }


@dataclass(frozen=True, slots=True)
class SiteMetric:
    """一個研究站的分子、分母與 TRAP 結構摘要。

    ``occurrence_snapshot_count`` 是分子，``valid_snapshot_count`` 是依 coverage 門檻
    篩選後的分母；``occurrence_rate`` 在分母為零時固定為 ``None``。density 欄位使用
    該站完整有效格點面積（km²）做標準化，不把 receptor core 面積混入 flow-domain
    統計。平均強度欄位單位是上游速度導出的 ``s⁻¹``，在來源單位尚未確認時只能作
    工程統計，不能直接宣稱觀測物理量。
    """

    site_id: str
    coverage_threshold: float
    valid_snapshot_count: int
    occurrence_snapshot_count: int
    occurrence_rate: float | None
    site_area_m2: float
    site_area_km2: float | None
    core_count: int
    curve_count: int
    total_curve_length_m: float
    mean_curve_length_m: float | None
    median_curve_length_m: float | None
    mean_core_s1_sinv: float | None
    mean_curve_s1_sinv: float | None
    mean_abs_curve_s1_sinv: float | None
    core_density_per_km2: float | None
    curve_density_per_km2: float | None
    curve_length_density_m_per_km2: float | None
    core_density_per_km2_per_valid_snapshot: float | None
    curve_density_per_km2_per_valid_snapshot: float | None
    curve_length_density_m_per_km2_per_valid_snapshot: float | None
    coverage_mean: float | None
    coverage_min: float | None
    coverage_max: float | None
    data_gap_frame_count: int
    coverage_unavailable_frame_count: int
    coverage_below_threshold_frame_count: int
    no_detection_frame_count: int
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def numerator(self) -> int:
        """出現率分子別名。"""

        return self.occurrence_snapshot_count

    @property
    def denominator(self) -> int:
        """出現率分母別名。"""

        return self.valid_snapshot_count

    def to_record(self) -> dict[str, Any]:
        """回傳保留 null 分母語意的 JSON 相容記錄。"""

        return {
            "site_id": self.site_id,
            "coverage_threshold": float(self.coverage_threshold),
            "valid_snapshot_count": int(self.valid_snapshot_count),
            "occurrence_snapshot_count": int(self.occurrence_snapshot_count),
            "occurrence_rate": self.occurrence_rate,
            "site_area_m2": float(self.site_area_m2),
            "site_area_km2": self.site_area_km2,
            "core_count": int(self.core_count),
            "curve_count": int(self.curve_count),
            "total_curve_length_m": float(self.total_curve_length_m),
            "mean_curve_length_m": self.mean_curve_length_m,
            "median_curve_length_m": self.median_curve_length_m,
            "mean_core_s1_sinv": self.mean_core_s1_sinv,
            "mean_curve_s1_sinv": self.mean_curve_s1_sinv,
            "mean_abs_curve_s1_sinv": self.mean_abs_curve_s1_sinv,
            "core_density_per_km2": self.core_density_per_km2,
            "curve_density_per_km2": self.curve_density_per_km2,
            "curve_length_density_m_per_km2": self.curve_length_density_m_per_km2,
            "core_density_per_km2_per_valid_snapshot": self.core_density_per_km2_per_valid_snapshot,
            "curve_density_per_km2_per_valid_snapshot": self.curve_density_per_km2_per_valid_snapshot,
            "curve_length_density_m_per_km2_per_valid_snapshot": self.curve_length_density_m_per_km2_per_valid_snapshot,
            "coverage_mean": self.coverage_mean,
            "coverage_min": self.coverage_min,
            "coverage_max": self.coverage_max,
            "data_gap_frame_count": int(self.data_gap_frame_count),
            "coverage_unavailable_frame_count": int(self.coverage_unavailable_frame_count),
            "coverage_below_threshold_frame_count": int(self.coverage_below_threshold_frame_count),
            "no_detection_frame_count": int(self.no_detection_frame_count),
            "quality_flags": list(self.quality_flags),
        }


@dataclass(frozen=True, slots=True)
class SiteMetricsResult:
    """五站 metrics、輸入 frame 摘要與可重現設定。"""

    metrics: Mapping[str, SiteMetric]
    frames: tuple[TrackingFrame, ...]
    config: SiteMetricsConfig
    grid_shape: tuple[int, int]
    grid_spacing_m: tuple[float, float]

    def to_records(self) -> list[dict[str, Any]]:
        """依 site ID 排序回傳 metrics 記錄。"""

        return [self.metrics[site_id].to_record() for site_id in sorted(self.metrics)]


def _extract_axis(coordinate: np.ndarray, shape: tuple[int, int], *, axis: int, name: str) -> np.ndarray:
    """從一維或軸對齊二維公尺座標抽取單調軸。"""

    array = np.asarray(coordinate, dtype=np.float64)
    expected_length = shape[1] if axis == 1 else shape[0]
    if array.ndim == 1:
        _require(array.size == expected_length, f"{name} 一維長度與 site mask 不符")
        axis_values = array
    else:
        _require(array.ndim == 2 and array.shape == shape, f"{name} 必須是一維軸或與 mask 同 shape 的二維座標")
        axis_values = array[0, :] if axis == 1 else array[:, 0]
        expected = axis_values[None, :] if axis == 1 else axis_values[:, None]
        _require(np.allclose(array, expected, rtol=0.0, atol=1.0e-6), f"{name} 必須是軸對齊 rectilinear 座標")
    _require(np.all(np.isfinite(axis_values)), f"{name} 必須全部有限")
    differences = np.diff(axis_values)
    _require(np.all(differences > 0.0) or np.all(differences < 0.0), f"{name} 必須嚴格單調")
    return axis_values


def _grid_axes(x_m: np.ndarray, y_m: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """驗證公尺網格並回傳軸與中位格距。"""

    x_axis = _extract_axis(x_m, shape, axis=1, name="x_m")
    y_axis = _extract_axis(y_m, shape, axis=0, name="y_m")
    dx = float(np.median(np.abs(np.diff(x_axis))))
    dy = float(np.median(np.abs(np.diff(y_axis))))
    _require(np.isfinite(dx) and np.isfinite(dy) and dx > 0.0 and dy > 0.0, "x_m/y_m 必須有正格距")
    return x_axis, y_axis, dx, dy


def _nearest_axis_index(axis: np.ndarray, value: float) -> int | None:
    """回傳公尺座標軸上距離最近的格點索引；軸外點回傳 None。"""

    if not np.isfinite(value) or value < min(axis[0], axis[-1]) or value > max(axis[0], axis[-1]):
        return None
    if axis[-1] < axis[0]:
        return _nearest_axis_index(-axis, -value)
    right = int(np.searchsorted(axis, value, side="left"))
    if right <= 0:
        return 0
    if right >= axis.size:
        return axis.size - 1
    left = right - 1
    return left if abs(value - axis[left]) <= abs(axis[right] - value) else right


def curve_intersects_mask(
    curve_points_m: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    site_mask: np.ndarray,
    *,
    max_step_m: float | None = None,
) -> bool:
    """以不大於半格距的折線取樣檢查 TRAP 曲線是否穿越站點格點。

    函式不只測核心；每一段曲線會先以 ``max_step_m`` 加密，預設使用
    ``0.5 * min(dx, dy)``，再把每個取樣點映射至最近公尺格點並查詢 ``site_mask``。
    因此端點在站外、但中間線段穿過站點的情況仍可被計入。超出 flow domain 的取樣點
    只代表該曲線段沒有與本站相交，不會被當成站點有效格點。
    """

    mask = np.asarray(site_mask, dtype=bool)
    _require(mask.ndim == 2, "site_mask 必須是二維")
    x_axis, y_axis, dx, dy = _grid_axes(np.asarray(x_m), np.asarray(y_m), mask.shape)
    step = 0.5 * min(dx, dy) if max_step_m is None else float(max_step_m)
    _require(np.isfinite(step) and step > 0.0 and step <= 0.5 * min(dx, dy) + 1.0e-9, "曲線取樣步長不可大於半格距")
    dense = densify_polyline(curve_points_m, step)
    for point_x, point_y in dense:
        col = _nearest_axis_index(x_axis, float(point_x))
        row = _nearest_axis_index(y_axis, float(point_y))
        if row is not None and col is not None and mask[row, col]:
            return True
    return False


def _point_in_mask(point_xy_m: tuple[float, float], x_axis: np.ndarray, y_axis: np.ndarray, site_mask: np.ndarray) -> bool:
    """檢查核心座標所在最近格點是否屬於指定站點。"""

    col = _nearest_axis_index(x_axis, float(point_xy_m[0]))
    row = _nearest_axis_index(y_axis, float(point_xy_m[1]))
    return row is not None and col is not None and bool(site_mask[row, col])


def _validate_site_masks(site_masks: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """固定站點 mask 型別、shape，並拒絕重疊以保護 A 區不重複分母。"""

    _require(isinstance(site_masks, Mapping) and bool(site_masks), "site_masks 必須是非空 mapping")
    normalized: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    union = None
    for site_id, value in site_masks.items():
        mask = np.asarray(value)
        _require(mask.dtype == np.dtype(bool) and mask.ndim == 2, f"{site_id} site mask 必須是二維 bool")
        if shape is None:
            shape = tuple(int(v) for v in mask.shape)
            union = np.zeros(shape, dtype=bool)
        _require(mask.shape == shape, "所有 site mask shape 必須一致")
        _require(not np.any(union & mask), "site mask 不可重疊；A 區每個格點只能歸一站")
        normalized[str(site_id)] = mask
        union |= mask
    _require(shape is not None, "site_masks 不可為空")
    return normalized, shape


def _coverage_for_frame(frame: TrackingFrame, site_id: str, site_mask: np.ndarray) -> float | None:
    """由逐時 valid mask 或明示 coverage mapping 取得站點有效率。"""

    if frame.valid_mask is not None:
        _require(frame.valid_mask.shape == site_mask.shape, "TrackingFrame.valid_mask 與 site mask shape 不符")
        site_count = int(site_mask.sum())
        if site_count == 0:
            return None
        return float(np.count_nonzero(frame.valid_mask & site_mask) / site_count)
    if frame.coverage_by_site is not None and site_id in frame.coverage_by_site:
        return float(frame.coverage_by_site[site_id])
    return None


def _optional_mean(values: list[float]) -> float | None:
    """空樣本回傳 null，避免以 0 代表沒有統計資料。"""

    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _optional_median(values: list[float]) -> float | None:
    """空樣本回傳 null，否則計算中位曲線長度。"""

    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def compute_site_metrics(
    frames: Sequence[TrackingFrame],
    site_masks: Mapping[str, np.ndarray],
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    cell_area_m2: np.ndarray | None = None,
    config: SiteMetricsConfig | None = None,
) -> SiteMetricsResult:
    """計算五站逐時出現率、曲線／核心與面積標準化統計。

    每個 frame 的 data gap 永遠排除於分母；可用但沒有 detection 的 frame 在 coverage
    達標時保留於分母並貢獻零分子；coverage 不足的 frame 另行計數。對每個站，若有一條
    ``TrapDetection.qualified`` 曲線與其有效格點相交，該 snapshot 的分子只增加一次，
    即使同一條曲線包含多個取樣點或穿越該站多次。核心、曲線長度與強度則對所有相交
    合格曲線保存，並以完整站點有效面積計算 density。

    此函式不會從 ``TrackingResult`` 自動重建分母或把追蹤壽命代替站點出現率；需要
    Eulerian 時序時可傳入 ``result.frames``。研究期截尾和 data-gap 資訊仍保留在
    ``TrackingResult.tracks``，此處只統計 snapshot level 的站點分母。
    """

    metrics_config = config or SiteMetricsConfig()
    normalized_frames = tuple(frames)
    _require(all(isinstance(frame, TrackingFrame) for frame in normalized_frames), "frames 必須全部是 TrackingFrame")
    _require(
        all(normalized_frames[index].time_utc_ns < normalized_frames[index + 1].time_utc_ns for index in range(len(normalized_frames) - 1)),
        "TrackingFrame 時間軸必須嚴格遞增",
    )
    normalized_masks, shape = _validate_site_masks(site_masks)
    x_axis, y_axis, dx, dy = _grid_axes(np.asarray(x_m), np.asarray(y_m), shape)
    if cell_area_m2 is None:
        area = np.full(shape, dx * dy, dtype=np.float64)
        area_source = "grid_spacing_product"
    else:
        area = np.asarray(cell_area_m2, dtype=np.float64)
        _require(area.shape == shape, "cell_area_m2 與 site mask shape 不符")
        any_mask = np.logical_or.reduce(tuple(normalized_masks.values()))
        if np.any(any_mask):
            _require(np.all(np.isfinite(area[any_mask])) and np.all(area[any_mask] > 0.0), "有效站點 cell_area_m2 必須正且有限")
        area_source = "input_cell_area_m2"

    step = metrics_config.curve_sample_step_m or 0.5 * min(dx, dy)
    _require(step <= 0.5 * min(dx, dy) + 1.0e-9, "curve_sample_step_m 不可大於半格距")

    accumulators: dict[str, dict[str, Any]] = {}
    for site_id, mask in normalized_masks.items():
        site_area = float(np.sum(area[mask])) if np.any(mask) else 0.0
        accumulators[site_id] = {
            "valid": 0,
            "occurrence": 0,
            "core_count": 0,
            "curve_count": 0,
            "lengths": [],
            "core_strengths": [],
            "curve_strengths": [],
            "abs_curve_strengths": [],
            "coverages": [],
            "data_gap": 0,
            "coverage_unavailable": 0,
            "coverage_below": 0,
            "no_detection": 0,
            "site_area_m2": site_area,
        }

    for frame in normalized_frames:
        if not frame.available:
            for accumulator in accumulators.values():
                accumulator["data_gap"] += 1
            continue
        detections = tuple(frame.detections or ())
        for site_id, mask in normalized_masks.items():
            accumulator = accumulators[site_id]
            coverage = _coverage_for_frame(frame, site_id, mask)
            if coverage is None:
                accumulator["coverage_unavailable"] += 1
                continue
            accumulator["coverages"].append(coverage)
            if coverage < metrics_config.coverage_threshold:
                accumulator["coverage_below"] += 1
                continue
            accumulator["valid"] += 1
            intersecting: list[TrapDetection] = []
            for detection in detections:
                if metrics_config.require_qualified and not detection.qualified:
                    continue
                if curve_intersects_mask(detection.curve_points_m, x_axis, y_axis, mask, max_step_m=step):
                    intersecting.append(detection)
                    if _point_in_mask(detection.core_xy_m, x_axis, y_axis, mask):
                        accumulator["core_count"] += 1
                        accumulator["core_strengths"].append(float(detection.core_s1_sinv))
                    accumulator["curve_count"] += 1
                    accumulator["lengths"].append(float(detection.curve_length_m))
                    accumulator["curve_strengths"].append(float(detection.curve_mean_s1_sinv))
                    accumulator["abs_curve_strengths"].append(float(detection.curve_mean_abs_s1_sinv))
            if intersecting:
                accumulator["occurrence"] += 1
            else:
                # 這裡包含「全域有其他站點 TRAP，但本站沒有曲線交會」以及
                # 「此時刻完全沒有合格 detection」兩種情況；對本站而言都屬於
                # 有效分母中的未檢出，不會誤改成資料缺口。
                accumulator["no_detection"] += 1

    result_metrics: dict[str, SiteMetric] = {}
    for site_id, accumulator in accumulators.items():
        valid_count = int(accumulator["valid"])
        occurrence_count = int(accumulator["occurrence"])
        site_area_m2 = float(accumulator["site_area_m2"])
        site_area_km2 = site_area_m2 / 1.0e6 if site_area_m2 > 0.0 else None
        lengths = accumulator["lengths"]
        core_density = accumulator["core_count"] / site_area_km2 if site_area_km2 else None
        curve_density = accumulator["curve_count"] / site_area_km2 if site_area_km2 else None
        length_density = float(np.sum(lengths)) / site_area_km2 if site_area_km2 else None
        # 累計密度保留原始「整段研究期每面積」語意；另除以有效時刻分母，
        # 避免不同站點或月份的缺時數量使單位面積比較產生系統性偏差。
        core_density_per_snapshot = core_density / valid_count if core_density is not None and valid_count else None
        curve_density_per_snapshot = curve_density / valid_count if curve_density is not None and valid_count else None
        length_density_per_snapshot = length_density / valid_count if length_density is not None and valid_count else None
        quality_flags: list[str] = [f"area_source:{area_source}"]
        if valid_count == 0:
            quality_flags.append("zero_denominator")
        if accumulator["data_gap"]:
            quality_flags.append("data_gap_excluded")
        if accumulator["coverage_unavailable"]:
            quality_flags.append("coverage_unavailable_excluded")
        result_metrics[site_id] = SiteMetric(
            site_id=site_id,
            coverage_threshold=float(metrics_config.coverage_threshold),
            valid_snapshot_count=valid_count,
            occurrence_snapshot_count=occurrence_count,
            occurrence_rate=float(occurrence_count / valid_count) if valid_count else None,
            site_area_m2=site_area_m2,
            site_area_km2=site_area_km2,
            core_count=int(accumulator["core_count"]),
            curve_count=int(accumulator["curve_count"]),
            total_curve_length_m=float(np.sum(lengths)) if lengths else 0.0,
            mean_curve_length_m=_optional_mean(lengths),
            median_curve_length_m=_optional_median(lengths),
            mean_core_s1_sinv=_optional_mean(accumulator["core_strengths"]),
            mean_curve_s1_sinv=_optional_mean(accumulator["curve_strengths"]),
            mean_abs_curve_s1_sinv=_optional_mean(accumulator["abs_curve_strengths"]),
            core_density_per_km2=core_density,
            curve_density_per_km2=curve_density,
            curve_length_density_m_per_km2=length_density,
            core_density_per_km2_per_valid_snapshot=core_density_per_snapshot,
            curve_density_per_km2_per_valid_snapshot=curve_density_per_snapshot,
            curve_length_density_m_per_km2_per_valid_snapshot=length_density_per_snapshot,
            coverage_mean=_optional_mean(accumulator["coverages"]),
            coverage_min=float(np.min(accumulator["coverages"])) if accumulator["coverages"] else None,
            coverage_max=float(np.max(accumulator["coverages"])) if accumulator["coverages"] else None,
            data_gap_frame_count=int(accumulator["data_gap"]),
            coverage_unavailable_frame_count=int(accumulator["coverage_unavailable"]),
            coverage_below_threshold_frame_count=int(accumulator["coverage_below"]),
            no_detection_frame_count=int(accumulator["no_detection"]),
            quality_flags=tuple(quality_flags),
        )

    return SiteMetricsResult(
        metrics=result_metrics,
        frames=normalized_frames,
        config=metrics_config,
        grid_shape=shape,
        grid_spacing_m=(dx, dy),
    )


def calculate_site_metrics(
    frames: Sequence[TrackingFrame],
    site_masks: Mapping[str, np.ndarray],
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    cell_area_m2: np.ndarray | None = None,
    config: SiteMetricsConfig | None = None,
) -> SiteMetricsResult:
    """``compute_site_metrics`` 的語意別名，供報告腳本使用。"""

    return compute_site_metrics(frames, site_masks, x_m=x_m, y_m=y_m, cell_area_m2=cell_area_m2, config=config)


__all__ = [
    "SiteMetric",
    "SiteMetricsConfig",
    "SiteMetricsResult",
    "calculate_site_metrics",
    "compute_site_metrics",
    "curve_intersects_mask",
]
