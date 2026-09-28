"""逐時 TRAP 結構的 Eulerian 一對一連結。

本模組追蹤的是每個 OCM snapshot 中的幾何結構，不是粒子、水團或廢棄物軌跡。
相鄰可用時刻的候選配對必須同時通過核心公尺距離、整條曲線距離與時間差門檻，
再以確定性的最小配對成本做一對一連結。最多只允許一個缺失時刻進行 gap closing；
資料缺口、可用時刻沒有檢出、沒有通過距離門檻與研究期邊界都保留不同的結束原因，
避免把觀測中斷誤讀為結構消失。

:class:`StreamingTracker` 是逐 frame 推進的追蹤核心（單一事實來源）；
``track_structures`` 只是把整批 :class:`TrackingFrame` 依序 ``push`` 進
:class:`StreamingTracker` 再收集結果的便利包裝，供一次擁有全部 frame 在記憶體中
的呼叫端使用。多月份研究需要串流處理（見 :class:`StreamingTracker` 說明），因為
逐時 detection 全部保留在記憶體會超出可行範圍。
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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
    必須截斷並由後續檢出建立新 track。``nominal_time_step_seconds`` 是輸入 snapshot
    的名目取樣間隔，只用來把觀測跨度換算成 Kunz et al. (2024) 沿用的 eddy-census
    壽命慣例（見 :attr:`TrackedStructure.lifetime_seconds`），不影響任何匹配門檻。
    """

    max_core_distance_m: float = 5_000.0
    max_curve_distance_m: float = 10_000.0
    max_time_gap_seconds: float = 3.0 * 60.0 * 60.0
    max_missing_frames: int = 1
    curve_sample_step_m: float = 500.0
    require_qualified: bool = True
    nominal_time_step_seconds: float = 3600.0

    def __post_init__(self) -> None:
        """驗證所有距離、時間和 gap 參數，確保結果可寫入 manifest。"""

        for name, value in (
            ("max_core_distance_m", self.max_core_distance_m),
            ("max_curve_distance_m", self.max_curve_distance_m),
            ("max_time_gap_seconds", self.max_time_gap_seconds),
            ("curve_sample_step_m", self.curve_sample_step_m),
            ("nominal_time_step_seconds", self.nominal_time_step_seconds),
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
            "nominal_time_step_seconds": float(self.nominal_time_step_seconds),
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

    ``left_censored`` 表示第一個觀測落在研究序列第一個時刻、或緊接在一次資料缺口
    （前一 frame ``available=False``，或與前一 frame 的時間差超過
    ``max_time_gap_seconds``）之後才第一次被觀測到，兩種情形都代表起點之前的狀態
    未被觀測，故無法排除該結構在序列開始前就已存在。``right_censored`` 表示最後一個
    觀測因研究期結束、尾端資料缺口或時間支援不足而無法觀察是否持續。這些旗標只描述
    Eulerian 觀測窗口，不等同物理生命週期或粒子 residence time。``nominal_time_step_seconds``
    保留建立此 track 時使用的名目取樣間隔，供 :attr:`lifetime_seconds` 換算。
    """

    track_id: int
    observations: tuple[TrackObservation, ...]
    termination_reason: str
    left_censored: bool
    right_censored: bool
    missing_frame_count: int = 0
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    nominal_time_step_seconds: float = 3600.0

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
        step = float(self.nominal_time_step_seconds)
        _require(np.isfinite(step) and step > 0.0, "nominal_time_step_seconds 必須是正有限值")
        object.__setattr__(self, "nominal_time_step_seconds", step)

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
    def lifetime_seconds(self) -> float:
        """回傳 Kunz et al. (2024) 沿用之 eddy-census 壽命慣例。

        壽命 = 首尾觀測時間差 + 一個名目取樣間隔，因此只被觀測到單一時刻的結構
        壽命是一個取樣間隔而非 0；這反映每個 snapshot 代表的是一段取樣區間，不是
        瞬時點。``duration_seconds`` 仍保留原本的首尾觀測跨度，不受此換算影響。
        """

        return self.duration_seconds + self.nominal_time_step_seconds

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
            "lifetime_seconds": float(self.lifetime_seconds),
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


@dataclass(frozen=True, slots=True)
class TrackSummary:
    """:class:`StreamingTracker` 回傳的輕量 track 摘要。

    只保留數值與 key，不含 :class:`~traps.models.TrapDetection` 物件本身，因為
    串流處理的目的就是避免整條研究期間所有 detection 都留在記憶體。
    ``observation_keys`` 預設是 ``(frame_index, detection_index)``，其中
    ``detection_index`` 是該 frame ``detections`` tuple（未經 qualified 篩選前）
    的原始索引，讓呼叫端能把 key 對應回原始 frame 重建完整物件；若 ``push`` 有
    傳入 ``detection_keys``，則改用呼叫端提供的 key。
    """

    track_id: int
    observation_times_utc_ns: tuple[int, ...]
    observation_keys: tuple[Hashable, ...]
    core_xy_m: tuple[tuple[float, float], ...]
    core_s1_sinv: tuple[float, ...]
    termination_reason: str
    left_censored: bool
    right_censored: bool
    missing_frame_count: int
    quality_flags: tuple[str, ...]
    duration_seconds: float
    lifetime_seconds: float

    def to_record(self) -> dict[str, Any]:
        """回傳與 :meth:`TrackedStructure.to_record` 對齊的摘要記錄。"""

        return {
            "track_id": int(self.track_id),
            "observation_count": len(self.observation_times_utc_ns),
            "start_time_utc_ns": int(self.observation_times_utc_ns[0]),
            "end_time_utc_ns": int(self.observation_times_utc_ns[-1]),
            "duration_seconds": float(self.duration_seconds),
            "lifetime_seconds": float(self.lifetime_seconds),
            "termination_reason": self.termination_reason,
            "left_censored": bool(self.left_censored),
            "right_censored": bool(self.right_censored),
            "missing_frame_count": int(self.missing_frame_count),
            "quality_flags": list(self.quality_flags),
        }


def densify_polyline(points_m: np.ndarray, max_step_m: float) -> np.ndarray:
    """將折線加密，使相鄰取樣距離不大於指定公尺門檻。

    輸入是 ``(n, 2)`` 公尺座標；端點會保留且相鄰段以線性插值補點。這個工具同時
    用於 curve distance 與 metrics 的站點交會檢查，確保離散曲線不會因兩個端點剛好
    落在站外而跳過中間的站點格。

    每段的取樣數 ``segment_count = max(1, ceil(length / max_step_m))``、取樣
    分數 ``k / segment_count``（``k = 1..segment_count``）與逐段呼叫
    ``np.linspace`` 的舊版逐字相同；差別是這裡用一次 ``np.repeat``／累加索引
    算出所有段落、所有取樣點的段落編號與段內分數，取代對每個線段個別呼叫
    ``np.linspace`` 的 Python 迴圈——當這個函式在 tracking 的核心配對迴圈中被
    每一對候選呼叫兩次（曲線各自加密）時，這個迴圈與大量微小 ``linspace``
    呼叫的固定開銷才是效能瓶頸所在。
    """

    points = np.asarray(points_m, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 2 and points.shape[0] >= 1, "曲線點必須是 (n, 2)")
    _require(np.all(np.isfinite(points)), "曲線座標必須有限")
    _require(np.isfinite(max_step_m) and max_step_m > 0.0, "max_step_m 必須是正有限值")
    if points.shape[0] == 1:
        return points.copy()
    starts = points[:-1]
    ends = points[1:]
    vectors = ends - starts
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    counts = np.maximum(1, np.ceil(lengths / max_step_m).astype(np.int64))
    total = int(counts.sum())
    segment_ids = np.repeat(np.arange(counts.shape[0]), counts)
    cumulative_exclusive = np.concatenate(([0], np.cumsum(counts)[:-1]))
    position_in_segment = np.arange(total) - np.repeat(cumulative_exclusive, counts)
    fractions = (position_in_segment + 1).astype(np.float64) / counts[segment_ids].astype(np.float64)
    sampled = starts[segment_ids] + fractions[:, None] * vectors[segment_ids]
    return np.vstack([points[0:1], sampled])


def _points_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """向量化計算多個點各自到同一條折線的最短距離。

    與逐點呼叫純量版本語意完全相同（單點折線退化成點對點距離；零長度線段的
    投影分量以 0 取代，等同該線段退化成端點），差別只在於用 numpy broadcasting
    一次算完 ``points`` × ``segments`` 的距離矩陣，取代 Python 迴圈。
    """

    if polyline.shape[0] == 1:
        diff = points - polyline[0][None, :]
        return np.hypot(diff[:, 0], diff[:, 1])
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    squared_lengths = np.sum(vectors * vectors, axis=1)
    offsets = points[:, None, :] - starts[None, :, :]
    dot = np.sum(offsets * vectors[None, :, :], axis=2)
    fraction = np.divide(
        dot,
        squared_lengths[None, :],
        out=np.zeros((points.shape[0], starts.shape[0]), dtype=np.float64),
        where=squared_lengths[None, :] > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
    delta = nearest - points[:, None, :]
    distances = np.hypot(delta[:, :, 0], delta[:, :, 1])
    return np.min(distances, axis=1)


def polyline_distance_m(first_points_m: np.ndarray, second_points_m: np.ndarray, max_step_m: float) -> float:
    """計算兩條 e2 曲線的對稱 Hausdorff 公尺距離。

    兩條輸入折線先以 ``max_step_m`` 加密，再以每個取樣點到另一條折線的最短線段距離
    取最大值。這個距離同時看整條曲線，不會因核心點相近就把已明顯分叉的結構錯連。
    點到折線的距離以 numpy broadcasting（取樣點 × 線段）一次算完，數值上與逐點的
    參考實作相同，只是不再逐點跑 Python 迴圈。
    """

    first = densify_polyline(first_points_m, max_step_m)
    second = densify_polyline(second_points_m, max_step_m)
    first_to_second = float(np.max(_points_to_polyline_distances(first, second)))
    second_to_first = float(np.max(_points_to_polyline_distances(second, first)))
    return float(max(first_to_second, second_to_first))


def core_distance_m(first: TrapDetection, second: TrapDetection) -> float:
    """回傳兩個 detection 核心公尺座標的 Euclidean 距離。"""

    return float(np.hypot(first.core_xy_m[0] - second.core_xy_m[0], first.core_xy_m[1] - second.core_xy_m[1]))


def _pair_cost(first: TrapDetection, second: TrapDetection, config: TrackingConfig) -> tuple[float, float, float] | None:
    """回傳通過雙距離門檻的配對成本；不合格配對回傳 ``None``。

    核心距離的計算是 O(1)，曲線的對稱 Hausdorff 距離則要在加密後的折線上做
    broadcasting，成本高出幾個數量級；因此一定先檢查核心距離門檻，未通過就立刻
    回傳，完全不觸碰 ``polyline_distance_m``，避免對明顯不可能配對的 (track,
    detection) 組合白白計算曲線距離。
    """

    core_distance = core_distance_m(first, second)
    if core_distance > config.max_core_distance_m:
        return None
    curve_distance = polyline_distance_m(first.curve_points_m, second.curve_points_m, config.curve_sample_step_m)
    if curve_distance > config.max_curve_distance_m:
        return None
    normalized_cost = core_distance / config.max_core_distance_m + curve_distance / config.max_curve_distance_m
    return float(normalized_cost), float(core_distance), float(curve_distance)


@dataclass
class _StreamTrackState:
    """:class:`StreamingTracker` 內部可變的私有 track 狀態。

    只保留最後一個 :class:`TrapDetection` 物件（用於下一輪配對的核心／曲線距離
    計算），先前所有觀測只以數值（時間、key、核心座標、核心 s1）摘要保存，藉此
    滿足「每條活躍 track 只留最後一個 detection 物件在記憶體」的串流需求。
    """

    track_id: int
    last_detection: TrapDetection
    last_time_utc_ns: int
    last_frame_index: int
    observation_times_utc_ns: list[int]
    observation_keys: list[Hashable]
    core_xy_m: list[tuple[float, float]]
    core_s1_sinv: list[float]
    missing_frame_count: int = 0
    total_missing_frame_count: int = 0
    left_censored: bool = False
    quality_flags: list[str] = field(default_factory=list)


def _finalize_stream_state(
    state: _StreamTrackState,
    *,
    reason: str,
    right_censored: bool,
    right_censored_by_gap: bool,
    nominal_time_step_seconds: float,
) -> TrackSummary:
    """把串流狀態轉成不可變 :class:`TrackSummary`，並補上截尾品質旗標。"""

    flags = list(state.quality_flags)
    if state.left_censored and "left_boundary_truncated" not in flags:
        flags.append("left_boundary_truncated")
    if right_censored and "right_boundary_truncated" not in flags:
        flags.append("right_boundary_truncated")
    if right_censored_by_gap and "right_censored_by_gap" not in flags:
        flags.append("right_censored_by_gap")
    if state.total_missing_frame_count and "contains_data_gap" not in flags:
        flags.append("contains_data_gap")
    duration_seconds = (state.observation_times_utc_ns[-1] - state.observation_times_utc_ns[0]) / NANOSECONDS_PER_SECOND
    return TrackSummary(
        track_id=state.track_id,
        observation_times_utc_ns=tuple(state.observation_times_utc_ns),
        observation_keys=tuple(state.observation_keys),
        core_xy_m=tuple(state.core_xy_m),
        core_s1_sinv=tuple(state.core_s1_sinv),
        termination_reason=reason,
        left_censored=state.left_censored,
        right_censored=right_censored,
        missing_frame_count=state.total_missing_frame_count,
        quality_flags=tuple(flags),
        duration_seconds=float(duration_seconds),
        lifetime_seconds=float(duration_seconds + nominal_time_step_seconds),
    )


class StreamingTracker:
    """逐 frame 推進的 Eulerian 結構追蹤器，供多月份研究串流使用。

    完整研究期間（約 17,000 個逐時 frame，每個 domain 約 250 個 detection）不可能
    把所有 :class:`~traps.models.TrapDetection` 物件同時留在記憶體，因此本類別
    ``push`` 一個 :class:`TrackingFrame` 就只保留每條活躍 track 最後一個 detection
    物件；已結束的 track 立刻以輕量的 :class:`TrackSummary` 回傳，呼叫端可以邊
    寫出邊丟棄。``track_structures`` 是本類別的批次包裝，兩者共用完全相同的匹配、
    gap closing 與截尾規則，本類別才是唯一的實作來源。

    frame 必須以嚴格遞增的 ``time_utc_ns`` 依序 ``push``；呼叫 :meth:`finish` 之後
    不可再 push，``finish`` 會把仍活躍的 track 以研究期結束或尾端資料缺口關閉並
    標右截尾。
    """

    def __init__(self, config: TrackingConfig | None = None, *, start_track_id: int = 0) -> None:
        self.config = config or TrackingConfig()
        self._max_gap_ns = round(self.config.max_time_gap_seconds * NANOSECONDS_PER_SECOND)
        self._active: dict[int, _StreamTrackState] = {}
        self._events: list[TrackingEvent] = []
        self._next_track_id = int(start_track_id)
        self._frame_index = 0
        self._last_pushed_time_ns: int | None = None
        self._previous_frame: TrackingFrame | None = None
        self._last_frame: TrackingFrame | None = None
        self._finished = False

    @property
    def events(self) -> tuple[TrackingEvent, ...]:
        """回傳目前為止收集到的所有 :class:`TrackingEvent`。"""

        return tuple(self._events)

    @property
    def active_track_count(self) -> int:
        """回傳目前仍活躍（尚未結束）的 track 數量。"""

        return len(self._active)

    def _eligible(self, frame: TrackingFrame) -> tuple[tuple[int, ...], tuple[TrapDetection, ...]]:
        """回傳合格 detection 及其在 ``frame.detections`` 內的原始索引。"""

        if not frame.available or frame.detections is None:
            return (), ()
        if self.config.require_qualified:
            pairs = [(index, detection) for index, detection in enumerate(frame.detections) if detection.qualified]
        else:
            pairs = list(enumerate(frame.detections))
        if not pairs:
            return (), ()
        indices, detections = zip(*pairs)
        return indices, detections

    def _close(self, state: _StreamTrackState, *, reason: str, right_censored: bool, right_censored_by_gap: bool) -> TrackSummary:
        return _finalize_stream_state(
            state,
            reason=reason,
            right_censored=right_censored,
            right_censored_by_gap=right_censored_by_gap,
            nominal_time_step_seconds=self.config.nominal_time_step_seconds,
        )

    def push(self, frame: TrackingFrame, detection_keys: Sequence[Hashable] | None = None) -> list[TrackSummary]:
        """處理一個新 frame，回傳這一步結束的 track（``TrackSummary`` list）。

        ``detection_keys``（若提供）長度必須與 ``frame.detections`` 一致，供呼叫端
        以自訂 key（而非預設的 ``(frame_index, detection_index)``）標記每個觀測，
        方便呼叫端事後把觀測對應回自己的儲存結構。
        """

        _require(isinstance(frame, TrackingFrame), "frame 必須是 TrackingFrame")
        _require(not self._finished, "StreamingTracker 已呼叫 finish()，不可再 push")
        if self._last_pushed_time_ns is not None:
            _require(frame.time_utc_ns > self._last_pushed_time_ns, "frame 時間必須嚴格遞增於前一次 push")
        if detection_keys is not None and frame.detections is not None:
            _require(len(detection_keys) == len(frame.detections), "detection_keys 長度必須與 frame.detections 一致")

        frame_index = self._frame_index
        completed: list[TrackSummary] = []

        if not frame.available:
            # 缺時只增加 gap 計數，不製造空 detection；超過一個允許時刻便截斷。
            for track_id in list(self._active):
                state = self._active[track_id]
                elapsed_ns = frame.time_utc_ns - state.last_time_utc_ns
                if elapsed_ns > self._max_gap_ns:
                    completed.append(self._close(state, reason="time_gap", right_censored=True, right_censored_by_gap=True))
                    del self._active[track_id]
                    self._events.append(TrackingEvent(frame_index, frame.time_utc_ns, "track_closed_time_gap", track_id))
                    continue
                if state.missing_frame_count >= self.config.max_missing_frames:
                    completed.append(self._close(state, reason="data_gap", right_censored=True, right_censored_by_gap=True))
                    del self._active[track_id]
                    self._events.append(TrackingEvent(frame_index, frame.time_utc_ns, "track_closed_data_gap", track_id))
                    continue
                state.missing_frame_count += 1
                state.total_missing_frame_count += 1
                state.quality_flags.append("gap_closing_pending")
                self._events.append(TrackingEvent(frame_index, frame.time_utc_ns, "data_gap", track_id))
        else:
            eligible_indices, current_detections = self._eligible(frame)
            candidate_pairs: list[tuple[tuple[float, float, float], int, int]] = []

            if self._active and current_detections:
                active_ids = list(self._active.keys())
                active_cores = np.array(
                    [self._active[track_id].last_detection.core_xy_m for track_id in active_ids], dtype=np.float64
                )
                active_times = np.array(
                    [self._active[track_id].last_time_utc_ns for track_id in active_ids], dtype=np.int64
                )
                detection_cores = np.array([detection.core_xy_m for detection in current_detections], dtype=np.float64)

                # 先以 numpy broadcasting 算出所有 (active track × detection) 的核心
                # 距離矩陣並用核心距離＋時間差門檻過濾，只有通過的組合才進入逐對的
                # 曲線 Hausdorff 距離計算，避免對每一對都跑一次昂貴的曲線比較。
                diff = active_cores[:, None, :] - detection_cores[None, :, :]
                core_distance_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])
                elapsed_ns = frame.time_utc_ns - active_times
                time_ok = elapsed_ns <= self._max_gap_ns
                gate = time_ok[:, None] & (core_distance_matrix <= self.config.max_core_distance_m)
                track_rows, detection_cols = np.nonzero(gate)

                for row, col in zip(track_rows.tolist(), detection_cols.tolist()):
                    track_id = active_ids[row]
                    state = self._active[track_id]
                    detection = current_detections[col]
                    pair_cost = _pair_cost(state.last_detection, detection, self.config)
                    if pair_cost is not None:
                        candidate_pairs.append((pair_cost, track_id, col))

            candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
            matched_tracks: set[int] = set()
            matched_detections: set[int] = set()
            for pair_cost, track_id, detection_index in candidate_pairs:
                if track_id in matched_tracks or detection_index in matched_detections:
                    continue
                state = self._active[track_id]
                detection = current_detections[detection_index]
                original_index = eligible_indices[detection_index]
                key = detection_keys[original_index] if detection_keys is not None else (frame_index, original_index)
                state.observation_times_utc_ns.append(frame.time_utc_ns)
                state.observation_keys.append(key)
                state.core_xy_m.append((float(detection.core_xy_m[0]), float(detection.core_xy_m[1])))
                state.core_s1_sinv.append(float(detection.core_s1_sinv))
                state.last_detection = detection
                state.last_time_utc_ns = frame.time_utc_ns
                state.last_frame_index = frame_index
                if state.missing_frame_count:
                    state.quality_flags.append("gap_closed")
                state.missing_frame_count = 0
                matched_tracks.add(track_id)
                matched_detections.add(detection_index)
                self._events.append(
                    TrackingEvent(
                        frame_index,
                        frame.time_utc_ns,
                        "matched",
                        track_id,
                        {"core_distance_m": pair_cost[1], "curve_distance_m": pair_cost[2]},
                    )
                )

            for track_id in list(self._active):
                if track_id in matched_tracks:
                    continue
                state = self._active[track_id]
                elapsed_ns = frame.time_utc_ns - state.last_time_utc_ns
                if elapsed_ns > self._max_gap_ns:
                    reason = "time_gap"
                elif not current_detections:
                    reason = "no_detection"
                else:
                    reason = "no_match"
                # time_gap／data_gap 缺的是時間支援，不是「明確無結構」的可用觀測，
                # 因此一律右截尾；no_detection/no_match 則有可用 snapshot 證據，保留
                # 為非截尾的實際結束（即使剛好落在序列最後一個 frame 上）。
                right_censored = reason == "time_gap"
                completed.append(
                    self._close(state, reason=reason, right_censored=right_censored, right_censored_by_gap=right_censored)
                )
                del self._active[track_id]
                self._events.append(TrackingEvent(frame_index, frame.time_utc_ns, f"track_closed_{reason}", track_id))

            # 判斷新 track 的起點是否緊接在一次資料缺口之後：前一個 frame 本身就是
            # 資料缺口，或與前一個 frame 的時間差超過 max_time_gap_seconds（例如
            # 資料源中 24～50 小時的缺口），這兩種情形起點之前的狀態都未被觀測到，
            # 必須視為左截尾並標記 gap 專屬旗標；序列第一個 frame 維持原本的左截尾
            # 語意，但不算「缺口之後」。
            gap_before = frame_index == 0
            left_censored_after_gap = False
            if frame_index > 0 and self._previous_frame is not None:
                previous_frame = self._previous_frame
                if (not previous_frame.available) or (frame.time_utc_ns - previous_frame.time_utc_ns > self._max_gap_ns):
                    gap_before = True
                    left_censored_after_gap = True

            for detection_index, detection in enumerate(current_detections):
                if detection_index in matched_detections:
                    continue
                original_index = eligible_indices[detection_index]
                key = detection_keys[original_index] if detection_keys is not None else (frame_index, original_index)
                track_id = self._next_track_id
                self._next_track_id += 1
                state = _StreamTrackState(
                    track_id=track_id,
                    last_detection=detection,
                    last_time_utc_ns=frame.time_utc_ns,
                    last_frame_index=frame_index,
                    observation_times_utc_ns=[frame.time_utc_ns],
                    observation_keys=[key],
                    core_xy_m=[(float(detection.core_xy_m[0]), float(detection.core_xy_m[1]))],
                    core_s1_sinv=[float(detection.core_s1_sinv)],
                    left_censored=gap_before,
                )
                if left_censored_after_gap:
                    state.quality_flags.append("left_censored_after_gap")
                self._active[track_id] = state
                self._events.append(TrackingEvent(frame_index, frame.time_utc_ns, "track_started", track_id))

        self._previous_frame = frame
        self._last_frame = frame
        self._last_pushed_time_ns = frame.time_utc_ns
        self._frame_index += 1
        return completed

    def finish(self) -> list[TrackSummary]:
        """把仍活躍的 track 以研究期結束或尾端資料缺口關閉並回傳。

        呼叫之後不可再 ``push``；重複呼叫是安全的，第二次之後回傳空 list。
        """

        if self._finished:
            return []
        self._finished = True
        completed: list[TrackSummary] = []
        final_frame_is_gap = self._last_frame is not None and not self._last_frame.available
        last_frame_index = max(self._frame_index - 1, 0)
        last_time_ns = self._last_frame.time_utc_ns if self._last_frame is not None else 0
        reason = "data_gap" if final_frame_is_gap else "study_period_end"
        for track_id in sorted(self._active):
            state = self._active[track_id]
            completed.append(
                self._close(state, reason=reason, right_censored=True, right_censored_by_gap=reason == "data_gap")
            )
            self._events.append(TrackingEvent(last_frame_index, last_time_ns, f"track_closed_{reason}", track_id))
        self._active.clear()
        return completed


def track_structures(frames: Sequence[TrackingFrame], config: TrackingConfig | None = None) -> TrackingResult:
    """以一對一核心／曲線距離匹配建立逐時 Eulerian structure tracks。

    `frames` 必須按 UTC nanosecond 嚴格遞增。可用 frame 的空 tuple 表示「有資料但沒有
    合格檢出」；``available=False`` 表示資料缺口，最多保留一個 frame 給下一次可用時刻
    gap closing。每個時刻的 detection 最多只會被一條 track 使用，候選配對先依
    normalized core+curve cost、核心距離、曲線距離和 ID 排序，故結果可重現且不需把
    匹配誤當成粒子追蹤。函式最後會把仍活躍的 track 以 ``study_period_end`` 結束並標
    ``right_censored``。

    本函式只是 :class:`StreamingTracker` 的批次包裝：把整批 frame 依序 ``push``，再
    用預設 key（``(frame_index, frame.detections 內的原始索引)``）把回傳的
    :class:`TrackSummary` 對應回 ``frames`` 重建完整的 :class:`TrackedStructure`。
    """

    tracking_config = config or TrackingConfig()
    normalized_frames = tuple(frames)
    _require(normalized_frames, "至少需要一個 TrackingFrame")
    _require(all(isinstance(frame, TrackingFrame) for frame in normalized_frames), "frames 必須全部是 TrackingFrame")
    _require(
        all(normalized_frames[index].time_utc_ns < normalized_frames[index + 1].time_utc_ns for index in range(len(normalized_frames) - 1)),
        "TrackingFrame 時間軸必須嚴格遞增",
    )

    tracker = StreamingTracker(tracking_config)
    summaries: list[TrackSummary] = []
    for frame in normalized_frames:
        summaries.extend(tracker.push(frame))
    summaries.extend(tracker.finish())

    tracks: list[TrackedStructure] = []
    for summary in summaries:
        observations: list[TrackObservation] = []
        for time_utc_ns, key in zip(summary.observation_times_utc_ns, summary.observation_keys):
            frame_index, original_index = key
            detection = normalized_frames[frame_index].detections[original_index]
            observations.append(TrackObservation(frame_index, time_utc_ns, detection))
        tracks.append(
            TrackedStructure(
                track_id=summary.track_id,
                observations=tuple(observations),
                termination_reason=summary.termination_reason,
                left_censored=summary.left_censored,
                right_censored=summary.right_censored,
                missing_frame_count=summary.missing_frame_count,
                quality_flags=summary.quality_flags,
                nominal_time_step_seconds=tracking_config.nominal_time_step_seconds,
            )
        )
    tracks.sort(key=lambda track: track.track_id)

    return TrackingResult(
        frames=normalized_frames,
        tracks=tuple(tracks),
        events=tracker.events,
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
    "StreamingTracker",
    "TrackObservation",
    "TrackSummary",
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
