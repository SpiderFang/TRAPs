"""WP1 效能、截尾語意、壽命換算與串流 API 的回歸測試。

這個檔案涵蓋四個問題：

1. 向量化的核心距離門檻與 Hausdorff 曲線距離，對照一份凍結在本檔案內、來自
   ``git show HEAD:src/traps/tracking.py``（WP1 之前的 baseline）之實作，確認數值
   與匹配結果完全一致，同時確認 Hausdorff 只在核心距離通過門檻後才會被呼叫。
2. 資料缺口前後的左／右截尾旗標。
3. Kunz et al. (2024) 的 eddy-census 壽命慣例（``lifetime_seconds``）。
4. :class:`~traps.tracking.StreamingTracker` 逐 frame 推進與批次版 ``track_structures``
   結果一致，且遵守嚴格遞增時間戳的契約。

本檔案不引用真實 OCM 資料，所有 detection 都是合成的最小可行輸入。
"""

from __future__ import annotations

import random
import sys
import time
import types

import numpy as np
import pytest

from traps import tracking
from traps.models import TrapDetection
from traps.tracking import (
    StreamingTracker,
    TrackingConfig,
    TrackingFrame,
    track_structures,
)

NS_PER_HOUR = 3_600_000_000_000


# ---------------------------------------------------------------------------
# WP1 之前的 baseline 實作（凍結快照），供本檔案的回歸測試使用。
# 來源：git show HEAD:src/traps/tracking.py（本 WP 開工前的 baseline commit）。
# 只把相對匯入改成絕對匯入，其餘邏輯逐字保留，沒有任何精簡或「順手修正」。
# ---------------------------------------------------------------------------
_BASELINE_SOURCE = r'''
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from traps.models import TrapDetection

NANOSECONDS_PER_SECOND = 1_000_000_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_detection_tuple(detections):
    if detections is None:
        return None
    result = tuple(detections)
    _require(all(isinstance(item, TrapDetection) for item in result), "detections 必須全部是 TrapDetection")
    return result


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    max_core_distance_m: float = 5_000.0
    max_curve_distance_m: float = 10_000.0
    max_time_gap_seconds: float = 3.0 * 60.0 * 60.0
    max_missing_frames: int = 1
    curve_sample_step_m: float = 500.0
    require_qualified: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("max_core_distance_m", self.max_core_distance_m),
            ("max_curve_distance_m", self.max_curve_distance_m),
            ("max_time_gap_seconds", self.max_time_gap_seconds),
            ("curve_sample_step_m", self.curve_sample_step_m),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必須是正有限值")
        if self.max_missing_frames not in (0, 1):
            raise ValueError("max_missing_frames 只允許 0 或 1；本研究最多 gap closing 一個時刻")

    def to_manifest_dict(self):
        return {
            "max_core_distance_m": float(self.max_core_distance_m),
            "max_curve_distance_m": float(self.max_curve_distance_m),
            "max_time_gap_seconds": float(self.max_time_gap_seconds),
            "max_missing_frames": int(self.max_missing_frames),
            "curve_sample_step_m": float(self.curve_sample_step_m),
            "require_qualified": bool(self.require_qualified),
        }


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    time_utc_ns: int
    detections: Any = None
    available: bool = True
    valid_mask: Any = None
    coverage_by_site: Any = None
    reason: Any = None

    def __post_init__(self) -> None:
        time_value = int(self.time_utc_ns)
        object.__setattr__(self, "time_utc_ns", time_value)
        detections = _as_detection_tuple(self.detections)
        if self.available and detections is None:
            detections = ()
        if not self.available:
            detections = None
        object.__setattr__(self, "detections", detections)
        if self.valid_mask is not None:
            mask = np.asarray(self.valid_mask, dtype=bool)
            _require(mask.ndim == 2, "TrackingFrame.valid_mask 必須是二維")
            object.__setattr__(self, "valid_mask", mask)
        if self.coverage_by_site is not None:
            coverage = {str(site_id): float(value) for site_id, value in self.coverage_by_site.items()}
            _require(
                all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in coverage.values()),
                "coverage_by_site 必須落在 0 到 1 之間",
            )
            object.__setattr__(self, "coverage_by_site", coverage)

    @property
    def data_gap(self) -> bool:
        return not self.available


@dataclass(frozen=True, slots=True)
class TrackObservation:
    frame_index: int
    time_utc_ns: int
    detection: Any


@dataclass(frozen=True, slots=True)
class TrackedStructure:
    track_id: int
    observations: Any
    termination_reason: str
    left_censored: bool
    right_censored: bool
    missing_frame_count: int = 0
    quality_flags: Any = field(default_factory=tuple)

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        _require(observations, "TrackedStructure 至少需要一個觀測")
        _require(all(isinstance(item, TrackObservation) for item in observations), "observations 型別不符")
        _require(
            all(observations[index].time_utc_ns < observations[index + 1].time_utc_ns for index in range(len(observations) - 1)),
            "track observations 必須嚴格按時間遞增",
        )
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "missing_frame_count", int(self.missing_frame_count))
        object.__setattr__(self, "quality_flags", tuple(str(flag) for flag in self.quality_flags))

    @property
    def start_time_utc_ns(self) -> int:
        return int(self.observations[0].time_utc_ns)

    @property
    def end_time_utc_ns(self) -> int:
        return int(self.observations[-1].time_utc_ns)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time_utc_ns - self.start_time_utc_ns) / NANOSECONDS_PER_SECOND

    @property
    def detections(self):
        return tuple(item.detection for item in self.observations)

    def to_record(self):
        return {
            "track_id": int(self.track_id),
            "observation_count": len(self.observations),
            "start_time_utc_ns": self.start_time_utc_ns,
            "end_time_utc_ns": self.end_time_utc_ns,
            "duration_seconds": float(self.duration_seconds),
            "termination_reason": self.termination_reason,
            "left_censored": bool(self.left_censored),
            "right_censored": bool(self.right_censored),
            "missing_frame_count": int(self.missing_frame_count),
            "quality_flags": list(self.quality_flags),
        }


@dataclass(frozen=True, slots=True)
class TrackingEvent:
    frame_index: int
    time_utc_ns: int
    event_type: str
    track_id: Any = None
    details: Any = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrackingResult:
    frames: Any
    tracks: Any
    events: Any
    config: Any

    def to_records(self):
        return [track.to_record() for track in self.tracks]


def densify_polyline(points_m, max_step_m):
    points = np.asarray(points_m, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 2 and points.shape[0] >= 1, "曲線點必須是 (n, 2)")
    _require(np.all(np.isfinite(points)), "曲線座標必須有限")
    _require(np.isfinite(max_step_m) and max_step_m > 0.0, "max_step_m 必須是正有限值")
    if points.shape[0] == 1:
        return points.copy()
    pieces = [points[0:1]]
    for start, end in pairwise(points):
        length = float(np.hypot(*(end - start)))
        segment_count = max(1, int(np.ceil(length / max_step_m)))
        fractions = np.linspace(0.0, 1.0, segment_count + 1, dtype=np.float64)[1:]
        pieces.append(start[None, :] + fractions[:, None] * (end - start)[None, :])
    return np.vstack(pieces)


def _point_to_polyline_distance(point, polyline):
    if polyline.shape[0] == 1:
        return float(np.hypot(*(point - polyline[0])))
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    squared_lengths = np.sum(vectors * vectors, axis=1)
    offsets = point[None, :] - starts
    fraction = np.divide(
        np.sum(offsets * vectors, axis=1),
        squared_lengths,
        out=np.zeros_like(squared_lengths),
        where=squared_lengths > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = starts + fraction[:, None] * vectors
    return float(np.min(np.hypot(nearest[:, 0] - point[0], nearest[:, 1] - point[1])))


def polyline_distance_m(first_points_m, second_points_m, max_step_m):
    first = densify_polyline(first_points_m, max_step_m)
    second = densify_polyline(second_points_m, max_step_m)
    first_to_second = max(_point_to_polyline_distance(point, second) for point in first)
    second_to_first = max(_point_to_polyline_distance(point, first) for point in second)
    return float(max(first_to_second, second_to_first))


def core_distance_m(first, second):
    return float(np.hypot(first.core_xy_m[0] - second.core_xy_m[0], first.core_xy_m[1] - second.core_xy_m[1]))


def _pair_cost(first, second, config):
    core_distance = core_distance_m(first, second)
    curve_distance = polyline_distance_m(first.curve_points_m, second.curve_points_m, config.curve_sample_step_m)
    if core_distance > config.max_core_distance_m or curve_distance > config.max_curve_distance_m:
        return None
    normalized_cost = core_distance / config.max_core_distance_m + curve_distance / config.max_curve_distance_m
    return float(normalized_cost), float(core_distance), float(curve_distance)


@dataclass
class _TrackState:
    track_id: int
    observations: Any
    missing_frame_count: int = 0
    total_missing_frame_count: int = 0
    left_censored: bool = False
    last_frame_index: int = 0
    termination_reason: Any = None
    quality_flags: Any = field(default_factory=list)

    @property
    def last_detection(self):
        return self.observations[-1].detection

    @property
    def last_time_utc_ns(self):
        return self.observations[-1].time_utc_ns


def _finalize_state(state, *, reason, right_censored):
    flags = list(state.quality_flags)
    if state.left_censored and "left_boundary_truncated" not in flags:
        flags.append("left_boundary_truncated")
    if right_censored and "right_boundary_truncated" not in flags:
        flags.append("right_boundary_truncated")
    if state.total_missing_frame_count and "contains_data_gap" not in flags:
        flags.append("contains_data_gap")
    return TrackedStructure(
        track_id=state.track_id,
        observations=tuple(state.observations),
        termination_reason=reason,
        left_censored=state.left_censored,
        right_censored=right_censored,
        missing_frame_count=state.total_missing_frame_count,
        quality_flags=tuple(flags),
    )


def _new_state(track_id, frame_index, frame, detection):
    return _TrackState(
        track_id=track_id,
        observations=[TrackObservation(frame_index, frame.time_utc_ns, detection)],
        left_censored=frame_index == 0,
        last_frame_index=frame_index,
    )


def _eligible_detections(frame, config):
    if not frame.available or frame.detections is None:
        return ()
    if config.require_qualified:
        return tuple(detection for detection in frame.detections if detection.qualified)
    return frame.detections


def track_structures(frames, config=None):
    tracking_config = config or TrackingConfig()
    normalized_frames = tuple(frames)
    _require(normalized_frames, "至少需要一個 TrackingFrame")
    _require(all(isinstance(frame, TrackingFrame) for frame in normalized_frames), "frames 必須全部是 TrackingFrame")
    _require(
        all(normalized_frames[index].time_utc_ns < normalized_frames[index + 1].time_utc_ns for index in range(len(normalized_frames) - 1)),
        "TrackingFrame 時間軸必須嚴格遞增",
    )

    active = {}
    completed = []
    events = []
    next_track_id = 0
    max_gap_ns = round(tracking_config.max_time_gap_seconds * NANOSECONDS_PER_SECOND)

    for frame_index, frame in enumerate(normalized_frames):
        if not frame.available:
            for track_id in list(active):
                state = active[track_id]
                elapsed_ns = frame.time_utc_ns - state.last_time_utc_ns
                if elapsed_ns > max_gap_ns:
                    completed.append(_finalize_state(state, reason="time_gap", right_censored=frame_index == len(normalized_frames) - 1))
                    del active[track_id]
                    events.append(TrackingEvent(frame_index, frame.time_utc_ns, "track_closed_time_gap", track_id))
                    continue
                if state.missing_frame_count >= tracking_config.max_missing_frames:
                    completed.append(_finalize_state(state, reason="data_gap", right_censored=frame_index == len(normalized_frames) - 1))
                    del active[track_id]
                    events.append(TrackingEvent(frame_index, frame.time_utc_ns, "track_closed_data_gap", track_id))
                    continue
                state.missing_frame_count += 1
                state.total_missing_frame_count += 1
                state.quality_flags.append("gap_closing_pending")
                events.append(TrackingEvent(frame_index, frame.time_utc_ns, "data_gap", track_id))
            continue

        current_detections = _eligible_detections(frame, tracking_config)
        candidate_pairs = []
        for track_id, state in active.items():
            elapsed_ns = frame.time_utc_ns - state.last_time_utc_ns
            if elapsed_ns > max_gap_ns:
                continue
            for detection_index, detection in enumerate(current_detections):
                pair_cost = _pair_cost(state.last_detection, detection, tracking_config)
                if pair_cost is not None:
                    candidate_pairs.append((pair_cost, track_id, detection_index))
        candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        matched_tracks = set()
        matched_detections = set()
        for pair_cost, track_id, detection_index in candidate_pairs:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            state = active[track_id]
            detection = current_detections[detection_index]
            state.observations.append(TrackObservation(frame_index, frame.time_utc_ns, detection))
            state.last_frame_index = frame_index
            if state.missing_frame_count:
                state.quality_flags.append("gap_closed")
            state.missing_frame_count = 0
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            events.append(
                TrackingEvent(
                    frame_index,
                    frame.time_utc_ns,
                    "matched",
                    track_id,
                    {"core_distance_m": pair_cost[1], "curve_distance_m": pair_cost[2]},
                )
            )

        for track_id in list(active):
            if track_id in matched_tracks:
                continue
            state = active[track_id]
            elapsed_ns = frame.time_utc_ns - state.last_time_utc_ns
            if elapsed_ns > max_gap_ns:
                reason = "time_gap"
            elif not current_detections:
                reason = "no_detection"
            else:
                reason = "no_match"
            right_censored = frame_index == len(normalized_frames) - 1 and reason == "time_gap"
            completed.append(_finalize_state(state, reason=reason, right_censored=right_censored))
            del active[track_id]
            events.append(TrackingEvent(frame_index, frame.time_utc_ns, f"track_closed_{reason}", track_id))

        for detection_index, detection in enumerate(current_detections):
            if detection_index in matched_detections:
                continue
            state = _new_state(next_track_id, frame_index, frame, detection)
            active[next_track_id] = state
            events.append(TrackingEvent(frame_index, frame.time_utc_ns, "track_started", next_track_id))
            next_track_id += 1

    last_frame_index = len(normalized_frames) - 1
    final_frame_is_gap = not normalized_frames[-1].available
    for track_id in sorted(active):
        state = active[track_id]
        reason = "data_gap" if final_frame_is_gap else "study_period_end"
        completed.append(_finalize_state(state, reason=reason, right_censored=True))
        events.append(TrackingEvent(last_frame_index, normalized_frames[-1].time_utc_ns, f"track_closed_{reason}", track_id))

    completed.sort(key=lambda track: track.track_id)
    return TrackingResult(
        frames=normalized_frames,
        tracks=tuple(completed),
        events=tuple(events),
        config=tracking_config,
    )
'''


def _load_baseline_module() -> types.ModuleType:
    """把凍結的 baseline 原始碼 exec 進一個獨立 module 命名空間。

    ``dataclass`` 在處理 ``ClassVar``/``KW_ONLY`` 型別時會經由
    ``sys.modules.get(cls.__module__)`` 反查模組命名空間；若這個臨時 module
    從未註冊進 ``sys.modules``，該查詢會拿到 ``None`` 而在 attribute access
    上炸掉。因此 exec 之前先把 module 放進 ``sys.modules``，執行完再取出，
    避免汙染全域 module 表。
    """

    module = types.ModuleType("_traps_tracking_baseline")
    sys.modules[module.__name__] = module
    try:
        exec(compile(_BASELINE_SOURCE, "<tracking-baseline>", "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    return module


_baseline = _load_baseline_module()


# ---------------------------------------------------------------------------
# 共用的合成 detection／frame 建構工具
# ---------------------------------------------------------------------------


def _make_curve(x_center_m: float, y_center_m: float, *, n_points: int, half_length_m: float, rng: random.Random) -> np.ndarray:
    """建立以核心為中心、沿 x 方向微幅抖動的合成曲線。"""

    offsets = np.linspace(-half_length_m, half_length_m, n_points)
    xs = x_center_m + offsets
    ys = y_center_m + np.array([rng.uniform(-5.0, 5.0) for _ in range(n_points)])
    return np.column_stack([xs, ys]).astype(np.float64)


def _detection(x_center_m: float, y_center_m: float, *, rng: random.Random, n_points: int = 5, half_length_m: float = 200.0) -> TrapDetection:
    """建立具有完整核心／曲線統計欄位的簡化 TRAP detection。"""

    points = _make_curve(x_center_m, y_center_m, n_points=n_points, half_length_m=half_length_m, rng=rng)
    strengths = np.full(points.shape[0], -1.0e-3, dtype=np.float64)
    return TrapDetection(
        core_index_ij=(0, 0),
        core_xy_m=(float(x_center_m), float(y_center_m)),
        core_s1_sinv=-1.0e-3,
        curve_points_m=points,
        curve_s1_sinv=strengths,
        curve_length_m=float(2.0 * half_length_m),
        curve_mean_s1_sinv=-1.0e-3,
        curve_mean_abs_s1_sinv=1.0e-3,
        forward_stop_reason="synthetic",
        backward_stop_reason="synthetic",
        is_qualified=True,
    )


def _frame(time_hour: int, detections: tuple[TrapDetection, ...] | None, *, available: bool = True) -> TrackingFrame:
    return TrackingFrame(time_utc_ns=time_hour * NS_PER_HOUR, detections=detections, available=available)


def _random_no_gap_sequence(
    *, frame_count: int, seed: int, min_tracks: int = 3, max_tracks: int = 8, domain_m: float = 100_000.0
) -> tuple[TrackingFrame, ...]:
    """建立沒有資料缺口、結構會隨機出現／消失／小幅漂移的合成序列。

    刻意不含任何 ``available=False`` 的 frame，讓這個序列只用來驗證匹配演算法
    （PROBLEM 1 的向量化門檻與 Hausdorff 距離），不牽涉 PROBLEM 2 對缺口相鄰
    track 的截尾規則變動，這樣才能對照 baseline 逐一比較 track 是否完全相同。
    """

    rng = random.Random(seed)
    tracks: list[dict[str, float]] = []
    for _ in range(rng.randint(min_tracks, max_tracks)):
        tracks.append({"x": rng.uniform(0.0, domain_m), "y": rng.uniform(0.0, domain_m), "alive": True})

    frames = []
    for frame_index in range(frame_count):
        detections = []
        for track in tracks:
            if not track["alive"]:
                continue
            # 每個時刻都有機會消失，之後不再出現，用來製造 no_detection/no_match 結束。
            if rng.random() < 0.03:
                track["alive"] = False
                continue
            track["x"] += rng.uniform(-300.0, 300.0)
            track["y"] += rng.uniform(-300.0, 300.0)
            detections.append(_detection(track["x"], track["y"], rng=rng))
        # 隨機加入新結構，測試新 track 建立與既有 track 的一對一競爭。
        if rng.random() < 0.15:
            x = rng.uniform(0.0, domain_m)
            y = rng.uniform(0.0, domain_m)
            tracks.append({"x": x, "y": y, "alive": True})
            detections.append(_detection(x, y, rng=rng))
        rng.shuffle(detections)
        frames.append(_frame(frame_index, tuple(detections)))
    return tuple(frames)


_REGRESSION_CONFIG = TrackingConfig(
    max_core_distance_m=1_500.0,
    max_curve_distance_m=3_000.0,
    max_time_gap_seconds=5.0 * 3_600.0,  # 遠大於逐時間距，序列內不觸發 time_gap 結束原因。
    max_missing_frames=1,
    curve_sample_step_m=250.0,
)


def _baseline_config_from(config: TrackingConfig) -> "_baseline.TrackingConfig":
    return _baseline.TrackingConfig(
        max_core_distance_m=config.max_core_distance_m,
        max_curve_distance_m=config.max_curve_distance_m,
        max_time_gap_seconds=config.max_time_gap_seconds,
        max_missing_frames=config.max_missing_frames,
        curve_sample_step_m=config.curve_sample_step_m,
        require_qualified=config.require_qualified,
    )


def _baseline_frames_from(frames: tuple[TrackingFrame, ...]) -> tuple["_baseline.TrackingFrame", ...]:
    return tuple(
        _baseline.TrackingFrame(time_utc_ns=frame.time_utc_ns, detections=frame.detections, available=frame.available)
        for frame in frames
    )


# ---------------------------------------------------------------------------
# PROBLEM 1b — 向量化點到折線距離 vs. 逐點參考實作
# ---------------------------------------------------------------------------


def _reference_point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    """純量、逐段迴圈的參考實作，與 baseline 演算法邏輯相同，獨立重寫供比對。"""

    if polyline.shape[0] == 1:
        return float(np.hypot(point[0] - polyline[0, 0], point[1] - polyline[0, 1]))
    best = np.inf
    for index in range(polyline.shape[0] - 1):
        start = polyline[index]
        end = polyline[index + 1]
        vector = end - start
        squared_length = float(vector[0] * vector[0] + vector[1] * vector[1])
        if squared_length > 0.0:
            fraction = float(np.dot(point - start, vector) / squared_length)
            fraction = min(1.0, max(0.0, fraction))
        else:
            fraction = 0.0
        nearest = start + fraction * vector
        distance = float(np.hypot(point[0] - nearest[0], point[1] - nearest[1]))
        best = min(best, distance)
    return best


def _reference_polyline_distance_m(first_points_m: np.ndarray, second_points_m: np.ndarray, max_step_m: float) -> float:
    first = tracking.densify_polyline(first_points_m, max_step_m)
    second = tracking.densify_polyline(second_points_m, max_step_m)
    first_to_second = max(_reference_point_to_polyline_distance(point, second) for point in first)
    second_to_first = max(_reference_point_to_polyline_distance(point, first) for point in second)
    return float(max(first_to_second, second_to_first))


def _random_polyline(rng: random.Random, *, min_points: int, max_points: int, scale_m: float) -> np.ndarray:
    n_points = rng.randint(min_points, max_points)
    return np.array([[rng.uniform(-scale_m, scale_m), rng.uniform(-scale_m, scale_m)] for _ in range(n_points)], dtype=np.float64)


@pytest.mark.parametrize("seed", range(12))
def test_polyline_distance_matches_reference_implementation(seed: int) -> None:
    """向量化 ``polyline_distance_m`` 與純 Python 參考實作在隨機折線上數值一致。"""

    rng = random.Random(seed)
    first = _random_polyline(rng, min_points=1, max_points=15, scale_m=5_000.0)
    second = _random_polyline(rng, min_points=1, max_points=15, scale_m=5_000.0)
    max_step_m = rng.uniform(50.0, 400.0)

    expected = _reference_polyline_distance_m(first, second, max_step_m)
    actual = tracking.polyline_distance_m(first, second, max_step_m)
    assert actual == pytest.approx(expected, abs=1e-9)


def test_polyline_distance_matches_reference_for_single_point_polylines() -> None:
    """兩條都退化成單點的折線是點對點距離，向量化版本仍須與參考值一致。"""

    first = np.array([[100.0, -50.0]], dtype=np.float64)
    second = np.array([[130.0, -10.0]], dtype=np.float64)
    expected = _reference_polyline_distance_m(first, second, 50.0)
    actual = tracking.polyline_distance_m(first, second, 50.0)
    assert actual == pytest.approx(expected, abs=1e-9)
    assert actual == pytest.approx(50.0, abs=1e-9)


def test_polyline_distance_matches_reference_with_degenerate_zero_length_segments() -> None:
    """含重複點（零長度線段）的折線不能讓向量化除以零或偏離參考值。"""

    rng = random.Random(2024)
    first = np.array(
        [[0.0, 0.0], [0.0, 0.0], [100.0, 0.0], [100.0, 0.0], [100.0, 0.0], [250.0, 40.0]],
        dtype=np.float64,
    )
    second = _random_polyline(rng, min_points=2, max_points=8, scale_m=300.0)
    expected = _reference_polyline_distance_m(first, second, 25.0)
    actual = tracking.polyline_distance_m(first, second, 25.0)
    assert actual == pytest.approx(expected, abs=1e-9)
    assert np.isfinite(actual)


def test_polyline_distance_matches_baseline_module_on_random_curves() -> None:
    """向量化實作與凍結的 baseline（迴圈版本）逐一比對，涵蓋多組隨機折線。"""

    rng = random.Random(7)
    for _ in range(20):
        first = _random_polyline(rng, min_points=1, max_points=12, scale_m=8_000.0)
        second = _random_polyline(rng, min_points=1, max_points=12, scale_m=8_000.0)
        max_step_m = rng.uniform(80.0, 500.0)
        expected = _baseline.polyline_distance_m(first, second, max_step_m)
        actual = tracking.polyline_distance_m(first, second, max_step_m)
        assert actual == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# PROBLEM 1a/1c — 核心距離先行門檻，Hausdorff 只算在通過門檻的候選上
# ---------------------------------------------------------------------------


def test_pair_cost_never_calls_hausdorff_when_core_distance_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """核心距離超過門檻時，``_pair_cost`` 必須提早回傳，完全不呼叫曲線距離。"""

    calls = {"count": 0}
    original = tracking.polyline_distance_m

    def _spy(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tracking, "polyline_distance_m", _spy)
    rng = random.Random(1)
    config = TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=1_000_000.0, curve_sample_step_m=50.0)
    first = _detection(0.0, 0.0, rng=rng)
    second = _detection(10_000.0, 10_000.0, rng=rng)  # 核心距離遠超過 100 m 門檻。

    result = tracking._pair_cost(first, second, config)

    assert result is None
    assert calls["count"] == 0


def test_track_structures_evaluates_hausdorff_only_for_core_gated_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """整條 ``track_structures`` 流程中，每次呼叫 Hausdorff 距離的配對核心距離都已通過門檻。"""

    violations: list[float] = []
    original = tracking.polyline_distance_m

    def _spy(first_points_m, second_points_m, max_step_m):
        return original(first_points_m, second_points_m, max_step_m)

    monkeypatch.setattr(tracking, "polyline_distance_m", _spy)

    frames = _random_no_gap_sequence(frame_count=25, seed=99)
    config = _REGRESSION_CONFIG

    calls = {"count": 0}
    original_pair_cost = tracking._pair_cost

    def _spy_pair_cost(first, second, cfg):
        core_distance = tracking.core_distance_m(first, second)
        if core_distance <= cfg.max_core_distance_m:
            calls["count"] += 1
        else:
            violations.append(core_distance)
        return original_pair_cost(first, second, cfg)

    monkeypatch.setattr(tracking, "_pair_cost", _spy_pair_cost)

    track_structures(frames, config)

    assert not violations, "Hausdorff 距離被算在核心距離未通過門檻的配對上"
    assert calls["count"] > 0


# ---------------------------------------------------------------------------
# PROBLEM 1c — 匹配結果與 baseline 完全一致（隨機合成序列回歸測試）
# ---------------------------------------------------------------------------


def _records_without_lifetime(records: list[dict]) -> list[dict]:
    return [{key: value for key, value in record.items() if key != "lifetime_seconds"} for record in records]


@pytest.mark.parametrize("seed", [3, 11, 42, 123, 2024])
def test_track_structures_matches_baseline_on_randomized_sequence(seed: int) -> None:
    """在沒有資料缺口的隨機合成序列上，新實作與 baseline 產生完全相同的 track。"""

    frames = _random_no_gap_sequence(frame_count=30, seed=seed)
    baseline_frames = _baseline_frames_from(frames)
    baseline_config = _baseline_config_from(_REGRESSION_CONFIG)

    new_result = track_structures(frames, _REGRESSION_CONFIG)
    baseline_result = _baseline.track_structures(baseline_frames, baseline_config)

    assert len(new_result.tracks) == len(baseline_result.tracks)
    assert _records_without_lifetime(new_result.to_records()) == baseline_result.to_records()

    for new_track, baseline_track in zip(new_result.tracks, baseline_result.tracks):
        new_cores = [observation.detection.core_xy_m for observation in new_track.observations]
        baseline_cores = [observation.detection.core_xy_m for observation in baseline_track.observations]
        assert new_cores == baseline_cores
        new_times = [observation.time_utc_ns for observation in new_track.observations]
        baseline_times = [observation.time_utc_ns for observation in baseline_track.observations]
        assert new_times == baseline_times


# ---------------------------------------------------------------------------
# PROBLEM 1d — 效能基準：60 frame × 250 detection 需遠低於 20 秒
# ---------------------------------------------------------------------------


def _synthetic_benchmark_frames(*, frame_count: int = 60, detections_per_frame: int = 250, seed: int = 4242) -> tuple[TrackingFrame, ...]:
    """建立 60 frame × 250 detection、核心散布在 150×100 km、逐時小幅漂移的合成序列。"""

    rng = random.Random(seed)
    domain_x_m = 150_000.0
    domain_y_m = 100_000.0
    tracks = [
        {"x": rng.uniform(0.0, domain_x_m), "y": rng.uniform(0.0, domain_y_m)} for _ in range(detections_per_frame)
    ]
    frames = []
    for frame_index in range(frame_count):
        detections = []
        for track in tracks:
            track["x"] = min(max(track["x"] + rng.uniform(-200.0, 200.0), 0.0), domain_x_m)
            track["y"] = min(max(track["y"] + rng.uniform(-200.0, 200.0), 0.0), domain_y_m)
            detections.append(_detection(track["x"], track["y"], rng=rng, n_points=20, half_length_m=500.0))
        rng.shuffle(detections)
        frames.append(_frame(frame_index, tuple(detections)))
    return tuple(frames)


def test_tracking_benchmark_completes_well_under_twenty_seconds() -> None:
    """60 frame × 250 detection 的合成序列必須遠低於 20 秒完成追蹤。"""

    frames = _synthetic_benchmark_frames()
    config = TrackingConfig(
        max_core_distance_m=5_000.0,
        max_curve_distance_m=10_000.0,
        max_time_gap_seconds=3.0 * 3_600.0,
        max_missing_frames=1,
        curve_sample_step_m=500.0,
    )

    started_at = time.perf_counter()
    result = track_structures(frames, config)
    elapsed_seconds = time.perf_counter() - started_at

    assert elapsed_seconds < 20.0, f"追蹤 60 frame x 250 detection 花了 {elapsed_seconds:.2f} 秒，超過 20 秒上限"
    assert len(result.tracks) > 0


# ---------------------------------------------------------------------------
# PROBLEM 2 — 資料缺口前後的左／右截尾
# ---------------------------------------------------------------------------


def test_new_detection_after_unavailable_frame_is_left_censored_after_gap() -> None:
    """緊接在 ``available=False`` frame 之後才出現的新 detection 必須左截尾。"""

    frames = (
        _frame(0, (_detection(0.0, 0.0, rng=random.Random(1)),)),
        _frame(1, None, available=False),
        _frame(2, (_detection(50_000.0, 50_000.0, rng=random.Random(2)),)),  # 與舊 track 距離遠，不會配對。
    )
    config = TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, curve_sample_step_m=50.0)
    result = track_structures(frames, config)

    new_track = next(track for track in result.tracks if track.observations[0].frame_index == 2)
    assert new_track.left_censored
    assert "left_censored_after_gap" in new_track.quality_flags


def test_new_detection_after_large_time_gap_is_left_censored_after_gap() -> None:
    """雖然兩個 frame 都可用，但時間差超過 max_time_gap_seconds 也視為資料缺口。"""

    hour = NS_PER_HOUR
    frames = (
        TrackingFrame(time_utc_ns=0, detections=(_detection(0.0, 0.0, rng=random.Random(3)),)),
        TrackingFrame(time_utc_ns=30 * hour, detections=(_detection(50_000.0, 50_000.0, rng=random.Random(4)),)),
    )
    config = TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, max_time_gap_seconds=3.0 * 3_600.0, curve_sample_step_m=50.0)
    result = track_structures(frames, config)

    new_track = next(track for track in result.tracks if track.observations[0].frame_index == 1)
    assert new_track.left_censored
    assert "left_censored_after_gap" in new_track.quality_flags


def test_first_frame_start_is_not_tagged_as_after_gap() -> None:
    """研究序列第一個 frame 的左截尾維持原本語意，不應被標成「缺口之後」。"""

    frames = (_frame(0, (_detection(0.0, 0.0, rng=random.Random(5)),)),)
    result = track_structures(frames, TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, curve_sample_step_m=50.0))
    track = result.tracks[0]
    assert track.left_censored
    assert "left_censored_after_gap" not in track.quality_flags
    assert "left_boundary_truncated" in track.quality_flags


def test_time_gap_termination_is_right_censored_even_mid_sequence() -> None:
    """``time_gap`` 結束原因即使不在序列最後一個 frame，也必須右截尾。"""

    hour = NS_PER_HOUR
    detection_a = _detection(0.0, 0.0, rng=random.Random(6))
    frames = (
        TrackingFrame(time_utc_ns=0, detections=(detection_a,)),
        TrackingFrame(time_utc_ns=30 * hour, detections=(_detection(90_000.0, 90_000.0, rng=random.Random(7)),)),
        TrackingFrame(time_utc_ns=31 * hour, detections=(_detection(90_100.0, 90_050.0, rng=random.Random(8)),)),
    )
    config = TrackingConfig(max_core_distance_m=200.0, max_curve_distance_m=200.0, max_time_gap_seconds=3.0 * 3_600.0, curve_sample_step_m=50.0)
    result = track_structures(frames, config)

    closed_track = next(track for track in result.tracks if track.observations[0].frame_index == 0)
    assert closed_track.termination_reason == "time_gap"
    assert closed_track.right_censored
    assert "right_censored_by_gap" in closed_track.quality_flags


def test_data_gap_termination_is_right_censored_even_mid_sequence() -> None:
    """``max_missing_frames=0`` 時第一次缺時就要以 ``data_gap`` 結束並右截尾，即使後面還有更多 frame。"""

    detection_a = _detection(0.0, 0.0, rng=random.Random(9))
    frames = (
        _frame(0, (detection_a,)),
        _frame(1, None, available=False),
        _frame(2, (_detection(200_000.0, 200_000.0, rng=random.Random(10)),)),
        _frame(3, (_detection(200_100.0, 200_050.0, rng=random.Random(11)),)),
    )
    config = TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, max_missing_frames=0, curve_sample_step_m=50.0)
    result = track_structures(frames, config)

    closed_track = next(track for track in result.tracks if track.observations[0].frame_index == 0)
    assert closed_track.termination_reason == "data_gap"
    assert closed_track.right_censored
    assert "right_censored_by_gap" in closed_track.quality_flags


def test_no_detection_and_no_match_termination_stay_uncensored() -> None:
    """有可用 snapshot 證據的 ``no_detection``/``no_match`` 結束仍不應右截尾（既有規則不變）。"""

    frames = (
        _frame(0, (_detection(0.0, 0.0, rng=random.Random(12)),)),
        _frame(1, ()),  # 可用、無檢出。
        _frame(2, (_detection(500_000.0, 500_000.0, rng=random.Random(13)),)),  # 可用但不會配對到既有 track。
    )
    config = TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, curve_sample_step_m=50.0)
    result = track_structures(frames, config)

    closed_track = next(track for track in result.tracks if track.observations[0].frame_index == 0)
    assert closed_track.termination_reason == "no_detection"
    assert not closed_track.right_censored
    assert "right_censored_by_gap" not in closed_track.quality_flags


# ---------------------------------------------------------------------------
# PROBLEM 3 — eddy-census 壽命慣例
# ---------------------------------------------------------------------------


def test_nominal_time_step_seconds_defaults_and_validates() -> None:
    """新欄位有預設值、會寫入 manifest，且拒絕非正值。"""

    config = TrackingConfig()
    assert config.nominal_time_step_seconds == pytest.approx(3600.0)
    assert config.to_manifest_dict()["nominal_time_step_seconds"] == pytest.approx(3600.0)

    with pytest.raises(ValueError, match="必須是正有限值"):
        TrackingConfig(nominal_time_step_seconds=0.0)
    with pytest.raises(ValueError, match="必須是正有限值"):
        TrackingConfig(nominal_time_step_seconds=-1.0)


def test_single_snapshot_track_has_nonzero_lifetime_seconds() -> None:
    """只被觀測到一次的結構，壽命是一個取樣間隔而非 0（duration_seconds 仍是 0）。"""

    frames = (
        _frame(0, (_detection(0.0, 0.0, rng=random.Random(14)),)),
        _frame(1, ()),
    )
    config = TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, curve_sample_step_m=50.0, nominal_time_step_seconds=1800.0)
    result = track_structures(frames, config)
    track = result.tracks[0]

    assert track.duration_seconds == pytest.approx(0.0)
    assert track.lifetime_seconds == pytest.approx(1800.0)
    assert track.to_record()["lifetime_seconds"] == pytest.approx(1800.0)


def test_multi_observation_lifetime_adds_one_nominal_step() -> None:
    """多次觀測時，壽命 = 觀測跨度 + 一個名目取樣間隔。"""

    frames = (
        _frame(0, (_detection(0.0, 0.0, rng=random.Random(15)),)),
        _frame(1, (_detection(10.0, 0.0, rng=random.Random(16)),)),
        _frame(2, (_detection(20.0, 0.0, rng=random.Random(17)),)),
    )
    config = TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, curve_sample_step_m=50.0)
    result = track_structures(frames, config)
    track = result.tracks[0]

    assert track.duration_seconds == pytest.approx(2.0 * 3_600.0)
    assert track.lifetime_seconds == pytest.approx(2.0 * 3_600.0 + 3_600.0)


def test_tracked_structure_backward_compatible_construction_keeps_default_lifetime() -> None:
    """既有呼叫端不傳 ``nominal_time_step_seconds`` 時建構子仍能運作，預設 3600 秒。"""

    from traps.tracking import TrackedStructure, TrackObservation

    detection = _detection(0.0, 0.0, rng=random.Random(18))
    structure = TrackedStructure(
        track_id=0,
        observations=(TrackObservation(0, 0, detection),),
        termination_reason="study_period_end",
        left_censored=True,
        right_censored=True,
    )
    assert structure.nominal_time_step_seconds == pytest.approx(3600.0)
    assert structure.lifetime_seconds == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# PROBLEM 4 — StreamingTracker
# ---------------------------------------------------------------------------


def test_streaming_tracker_rejects_non_increasing_time() -> None:
    """frame 時間必須嚴格遞增於前一次 push，否則要立即拒絕。"""

    tracker = StreamingTracker(TrackingConfig())
    tracker.push(_frame(5, ()))
    with pytest.raises(ValueError, match="嚴格遞增"):
        tracker.push(_frame(5, ()))
    with pytest.raises(ValueError, match="嚴格遞增"):
        tracker.push(_frame(4, ()))


def test_streaming_tracker_rejects_push_after_finish() -> None:
    """``finish`` 之後不可再 push。"""

    tracker = StreamingTracker(TrackingConfig())
    tracker.push(_frame(0, ()))
    tracker.finish()
    with pytest.raises(ValueError, match="finish"):
        tracker.push(_frame(1, ()))


def test_streaming_tracker_finish_is_idempotent() -> None:
    """重複呼叫 ``finish`` 是安全的，第二次之後不會重複回傳已關閉的 track。"""

    tracker = StreamingTracker(TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, curve_sample_step_m=50.0))
    tracker.push(_frame(0, (_detection(0.0, 0.0, rng=random.Random(19)),)))
    first = tracker.finish()
    second = tracker.finish()
    assert len(first) == 1
    assert second == []


def test_streaming_tracker_matches_track_structures_on_randomized_sequence() -> None:
    """逐 frame push 進 StreamingTracker 的結果必須與批次版 track_structures 相同。"""

    frames = _random_no_gap_sequence(frame_count=20, seed=555)
    config = _REGRESSION_CONFIG

    batch_result = track_structures(frames, config)

    tracker = StreamingTracker(config)
    summaries = []
    for frame in frames:
        summaries.extend(tracker.push(frame))
    summaries.extend(tracker.finish())
    summaries.sort(key=lambda summary: summary.track_id)

    assert [summary.to_record() for summary in summaries] == batch_result.to_records()
    for summary, track in zip(summaries, batch_result.tracks):
        assert summary.core_xy_m == tuple(observation.detection.core_xy_m for observation in track.observations)
        assert summary.observation_times_utc_ns == tuple(observation.time_utc_ns for observation in track.observations)


def test_streaming_tracker_keeps_lightweight_state_only() -> None:
    """串流狀態只保留每條活躍 track 最後一個 detection，摘要數值隨觀測累積成長。"""

    config = TrackingConfig(max_core_distance_m=200.0, max_curve_distance_m=200.0, curve_sample_step_m=50.0)
    tracker = StreamingTracker(config)
    rng = random.Random(20)

    tracker.push(_frame(0, (_detection(0.0, 0.0, rng=rng),)))
    assert tracker.active_track_count == 1
    tracker.push(_frame(1, (_detection(10.0, 5.0, rng=rng),)))
    assert tracker.active_track_count == 1

    completed = tracker.finish()
    assert len(completed) == 1
    summary = completed[0]
    assert len(summary.observation_times_utc_ns) == 2
    assert len(summary.core_xy_m) == 2
    assert summary.core_xy_m[0] != summary.core_xy_m[1]


def test_streaming_tracker_accepts_custom_detection_keys() -> None:
    """``push`` 可用呼叫端提供的 key 取代預設的 ``(frame_index, detection_index)``。"""

    config = TrackingConfig(max_core_distance_m=200.0, max_curve_distance_m=200.0, curve_sample_step_m=50.0)
    tracker = StreamingTracker(config)
    detection = _detection(0.0, 0.0, rng=random.Random(21))
    tracker.push(_frame(0, (detection,)), detection_keys=("frame0-det0",))
    completed = tracker.finish()
    assert completed[0].observation_keys == ("frame0-det0",)
