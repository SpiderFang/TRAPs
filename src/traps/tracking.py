"""逐時 TRAP 結構的 Eulerian 一對一連結。

本模組追蹤的是每個 OCM snapshot 中的幾何結構，不是粒子、水團或廢棄物軌跡。
相鄰可用時刻的候選配對必須同時通過核心公尺距離、整條曲線距離與時間差門檻，
再以確定性的最小配對成本做一對一連結。最多只允許一個缺失時刻進行 gap closing；
資料缺口、可用時刻沒有檢出、沒有通過距離門檻與研究期邊界都保留不同的結束原因，
避免把觀測中斷誤讀為結構消失。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from .models import TrapDetection

NANOSECONDS_PER_SECOND = 1_000_000_000
"""UTC nanosecond 時間軸轉換成秒時使用的固定倍率。"""


def _require(condition: bool, message: str) -> None:
    """在追蹤輸入不符合契約時立即停止，避免產生不可解讀的 track。"""

    if not condition:
        raise ValueError(message)


def _as_detection_tuple(detections: Iterable[TrapDetection] | None) -> tuple[TrapDetection, ...] | None:
    """將候選集合固定成 tuple，保留 ``None`` 對資料缺口的特殊語意。"""

    if detections is None:
        return None
    result = tuple(detections)
    _require(all(isinstance(item, TrapDetection) for item in result), "detections 必須全部是 TrapDetection")
    return result


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """逐時結構匹配的明示門檻。

    ``max_core_distance_m`` 控制核心位置的位移，``max_curve_distance_m`` 控制整條
    e2 曲線的對稱 Hausdorff 距離；兩者都要通過才可建立 link。``max_time_gap_seconds``
    限制最後一次觀測到下一次可用觀測的時間差。``max_missing_frames`` 固定最多為 1，
    因為本研究契約只允許一個缺失時刻的 gap closing；若缺失兩個以上時刻，原 track
    必須截斷並由後續檢出建立新 track。
    """

    max_core_distance_m: float = 5_000.0
    max_curve_distance_m: float = 10_000.0
    max_time_gap_seconds: float = 3.0 * 60.0 * 60.0
    max_missing_frames: int = 1
    curve_sample_step_m: float = 500.0
    require_qualified: bool = True

    def __post_init__(self) -> None:
        """驗證所有距離、時間和 gap 參數，確保結果可寫入 manifest。"""

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

    def to_manifest_dict(self) -> dict[str, Any]:
        """回傳可直接保存到 tracking manifest 的門檻字典。"""

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
    """一個可追蹤時刻的檢出結果與資料可用性。

    ``available=False`` 或 ``detections=None`` 表示資料缺口；這和可用時刻的空 tuple
    （該時刻有資料，但沒有合格 TRAP）有不同語意。``valid_mask`` 可由
    :class:`traps.ocm.SurfaceSnapshot` 提供，供站點 metrics 計算 coverage 分母；若
    尚未保留整張 mask，也可明確保存 ``coverage_by_site``，但不能以缺省值推算有效率。
    """

    time_utc_ns: int
    detections: tuple[TrapDetection, ...] | None = None
    available: bool = True
    valid_mask: np.ndarray | None = None
    coverage_by_site: Mapping[str, float] | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """固定時間、檢出 tuple 與 coverage 欄位的基本型別。"""

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
        """表示此時刻是否因資料缺口而不可作為統計分母。"""

        return not self.available


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """一條 Eulerian track 在單一 snapshot 的觀測。"""

    frame_index: int
    time_utc_ns: int
    detection: TrapDetection


@dataclass(frozen=True, slots=True)
class TrackedStructure:
    """一條結構 track 及其邊界／資料缺口截尾標籤。

    ``left_censored`` 表示第一個觀測落在研究序列第一個時刻，可能在研究期開始前
    已存在；``right_censored`` 表示最後一個觀測因研究期結束、尾端資料缺口或時間
    支援不足而無法觀察是否持續。這些旗標只描述 Eulerian 觀測窗口，不等同物理生命
    週期或粒子 residence time。
    """

    track_id: int
    observations: tuple[TrackObservation, ...]
    termination_reason: str
    left_censored: bool
    right_censored: bool
    missing_frame_count: int = 0
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """檢查 track 觀測按時間遞增且至少含一個 detection。"""

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
        """回傳第一個觀測時間。"""

        return int(self.observations[0].time_utc_ns)

    @property
    def end_time_utc_ns(self) -> int:
        """回傳最後一個觀測時間。"""

        return int(self.observations[-1].time_utc_ns)

    @property
    def duration_seconds(self) -> float:
        """回傳首尾觀測時間差（不把缺失時刻填成持續觀測）。"""

        return (self.end_time_utc_ns - self.start_time_utc_ns) / NANOSECONDS_PER_SECOND

    @property
    def detections(self) -> tuple[TrapDetection, ...]:
        """回傳 track 內按時間排序的 detection tuple。"""

        return tuple(item.detection for item in self.observations)

    def to_record(self) -> dict[str, Any]:
        """回傳小型 track 摘要，供 JSON/CSV manifest 使用。"""

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
    """追蹤過程的非結構性資料事件，供區分缺時與無檢出。"""

    frame_index: int
    time_utc_ns: int
    event_type: str
    track_id: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """逐時連結結果、輸入 frame 與可回溯事件。"""

    frames: tuple[TrackingFrame, ...]
    tracks: tuple[TrackedStructure, ...]
    events: tuple[TrackingEvent, ...]
    config: TrackingConfig

    def to_records(self) -> list[dict[str, Any]]:
        """回傳按 track ID 排序的摘要記錄。"""

        return [track.to_record() for track in self.tracks]


def densify_polyline(points_m: np.ndarray, max_step_m: float) -> np.ndarray:
    """將折線加密，使相鄰取樣距離不大於指定公尺門檻。

    輸入是 ``(n, 2)`` 公尺座標；端點會保留且相鄰段以線性插值補點。這個工具同時
    用於 curve distance 與 metrics 的站點交會檢查，確保離散曲線不會因兩個端點剛好
    落在站外而跳過中間的站點格。
    """

    points = np.asarray(points_m, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 2 and points.shape[0] >= 1, "曲線點必須是 (n, 2)")
    _require(np.all(np.isfinite(points)), "曲線座標必須有限")
    _require(np.isfinite(max_step_m) and max_step_m > 0.0, "max_step_m 必須是正有限值")
    if points.shape[0] == 1:
        return points.copy()
    pieces: list[np.ndarray] = [points[0:1]]
    for start, end in pairwise(points):
        length = float(np.hypot(*(end - start)))
        segment_count = max(1, int(np.ceil(length / max_step_m)))
        fractions = np.linspace(0.0, 1.0, segment_count + 1, dtype=np.float64)[1:]
        pieces.append(start[None, :] + fractions[:, None] * (end - start)[None, :])
    return np.vstack(pieces)


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    """計算一點到折線各線段的最短距離。"""

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


def polyline_distance_m(first_points_m: np.ndarray, second_points_m: np.ndarray, max_step_m: float) -> float:
    """計算兩條 e2 曲線的對稱 Hausdorff 公尺距離。

    兩條輸入折線先以 ``max_step_m`` 加密，再以每個取樣點到另一條折線的最短線段距離
    取最大值。這個距離同時看整條曲線，不會因核心點相近就把已明顯分叉的結構錯連。
    """

    first = densify_polyline(first_points_m, max_step_m)
    second = densify_polyline(second_points_m, max_step_m)
    first_to_second = max(_point_to_polyline_distance(point, second) for point in first)
    second_to_first = max(_point_to_polyline_distance(point, first) for point in second)
    return float(max(first_to_second, second_to_first))


def core_distance_m(first: TrapDetection, second: TrapDetection) -> float:
    """回傳兩個 detection 核心公尺座標的 Euclidean 距離。"""

    return float(np.hypot(first.core_xy_m[0] - second.core_xy_m[0], first.core_xy_m[1] - second.core_xy_m[1]))


def _pair_cost(first: TrapDetection, second: TrapDetection, config: TrackingConfig) -> tuple[float, float, float] | None:
    """回傳通過雙距離門檻的配對成本；不合格配對回傳 ``None``。"""

    core_distance = core_distance_m(first, second)
    curve_distance = polyline_distance_m(first.curve_points_m, second.curve_points_m, config.curve_sample_step_m)
    if core_distance > config.max_core_distance_m or curve_distance > config.max_curve_distance_m:
        return None
    normalized_cost = core_distance / config.max_core_distance_m + curve_distance / config.max_curve_distance_m
    return float(normalized_cost), float(core_distance), float(curve_distance)


@dataclass
class _TrackState:
    """追蹤迴圈內可變的私有 track 狀態。"""

    track_id: int
    observations: list[TrackObservation]
    missing_frame_count: int = 0
    total_missing_frame_count: int = 0
    left_censored: bool = False
    last_frame_index: int = 0
    termination_reason: str | None = None
    quality_flags: list[str] = field(default_factory=list)

    @property
    def last_detection(self) -> TrapDetection:
        """回傳目前 track 的最後一個 detection。"""

        return self.observations[-1].detection

    @property
    def last_time_utc_ns(self) -> int:
        """回傳目前 track 的最後觀測時間。"""

        return self.observations[-1].time_utc_ns


def _finalize_state(state: _TrackState, *, reason: str, right_censored: bool) -> TrackedStructure:
    """把私有狀態轉成不可變 track，並補上截尾品質旗標。"""

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


def _new_state(track_id: int, frame_index: int, frame: TrackingFrame, detection: TrapDetection) -> _TrackState:
    """從一個未配對 detection 建立新 track。"""

    return _TrackState(
        track_id=track_id,
        observations=[TrackObservation(frame_index, frame.time_utc_ns, detection)],
        left_censored=frame_index == 0,
        last_frame_index=frame_index,
    )


def _eligible_detections(frame: TrackingFrame, config: TrackingConfig) -> tuple[TrapDetection, ...]:
    """依 config 過濾尚未達到曲線品質門檻的 detection。"""

    if not frame.available or frame.detections is None:
        return ()
    if config.require_qualified:
        return tuple(detection for detection in frame.detections if detection.qualified)
    return frame.detections


def track_structures(frames: Sequence[TrackingFrame], config: TrackingConfig | None = None) -> TrackingResult:
    """以一對一核心／曲線距離匹配建立逐時 Eulerian structure tracks。

    `frames` 必須按 UTC nanosecond 嚴格遞增。可用 frame 的空 tuple 表示「有資料但沒有
    合格檢出」；``available=False`` 表示資料缺口，最多保留一個 frame 給下一次可用時刻
    gap closing。每個時刻的 detection 最多只會被一條 track 使用，候選配對先依
    normalized core+curve cost、核心距離、曲線距離和 ID 排序，故結果可重現且不需把
    匹配誤當成粒子追蹤。函式最後會把仍活躍的 track 以 ``study_period_end`` 結束並標
    ``right_censored``。
    """

    tracking_config = config or TrackingConfig()
    normalized_frames = tuple(frames)
    _require(normalized_frames, "至少需要一個 TrackingFrame")
    _require(all(isinstance(frame, TrackingFrame) for frame in normalized_frames), "frames 必須全部是 TrackingFrame")
    _require(
        all(normalized_frames[index].time_utc_ns < normalized_frames[index + 1].time_utc_ns for index in range(len(normalized_frames) - 1)),
        "TrackingFrame 時間軸必須嚴格遞增",
    )

    active: dict[int, _TrackState] = {}
    completed: list[TrackedStructure] = []
    events: list[TrackingEvent] = []
    next_track_id = 0
    max_gap_ns = round(tracking_config.max_time_gap_seconds * NANOSECONDS_PER_SECOND)

    for frame_index, frame in enumerate(normalized_frames):
        if not frame.available:
            # 缺時只增加 gap 計數，不製造空 detection；超過一個允許時刻便截斷。
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
        candidate_pairs: list[tuple[tuple[float, float, float], int, int]] = []
        for track_id, state in active.items():
            elapsed_ns = frame.time_utc_ns - state.last_time_utc_ns
            if elapsed_ns > max_gap_ns:
                continue
            for detection_index, detection in enumerate(current_detections):
                pair_cost = _pair_cost(state.last_detection, detection, tracking_config)
                if pair_cost is not None:
                    candidate_pairs.append((pair_cost, track_id, detection_index))
        candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
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
            # 最後一個 frame 若與上一觀測間隔過大，缺的是時間支援而非「明確無結構」；
            # 因此仍視為右截尾。no_detection/no_match 則有可用 snapshot 證據，保留為
            # 非截尾的實際結束。
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


def link_detections(frames: Sequence[TrackingFrame], config: TrackingConfig | None = None) -> TrackingResult:
    """``track_structures`` 的語意別名，便於以 link 名稱呼叫相同契約。"""

    return track_structures(frames, config)


def frames_from_detections(
    time_utc_ns: Sequence[int],
    detections: Sequence[Iterable[TrapDetection] | None],
    *,
    available: Sequence[bool] | None = None,
    valid_masks: Sequence[np.ndarray | None] | None = None,
    coverage_by_site: Sequence[Mapping[str, float] | None] | None = None,
) -> tuple[TrackingFrame, ...]:
    """由平行時間／檢出序列建立 TrackingFrame tuple。

    ``detections[i] is None`` 或 ``available[i] is False`` 都代表第 i 時刻資料缺口；
    若要表達「資料完整但沒有 TRAP」，請傳入空 tuple。可選 valid mask 與站點 coverage
    會原樣附加給 metrics，函式不會填補缺時或重複時間。
    """

    _require(len(time_utc_ns) == len(detections), "time_utc_ns 與 detections 長度必須一致")
    frame_count = len(time_utc_ns)
    available_values = tuple(True for _ in range(frame_count)) if available is None else tuple(bool(value) for value in available)
    _require(len(available_values) == frame_count, "available 長度必須與時間軸一致")
    masks = tuple(None for _ in range(frame_count)) if valid_masks is None else tuple(valid_masks)
    _require(len(masks) == frame_count, "valid_masks 長度必須與時間軸一致")
    coverage = tuple(None for _ in range(frame_count)) if coverage_by_site is None else tuple(coverage_by_site)
    _require(len(coverage) == frame_count, "coverage_by_site 長度必須與時間軸一致")
    return tuple(
        TrackingFrame(
            time_utc_ns=int(time_utc_ns[index]),
            detections=detections[index],
            available=available_values[index],
            valid_mask=masks[index],
            coverage_by_site=coverage[index],
        )
        for index in range(frame_count)
    )


__all__ = [
    "NANOSECONDS_PER_SECOND",
    "TrackObservation",
    "TrackedStructure",
    "TrackingConfig",
    "TrackingEvent",
    "TrackingFrame",
    "TrackingResult",
    "core_distance_m",
    "densify_polyline",
    "frames_from_detections",
    "link_detections",
    "polyline_distance_m",
    "track_structures",
]
