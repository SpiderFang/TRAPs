"""TRAP 時序連結與五站統計的合成契約測試。

測試使用人工建立的 ``TrapDetection`` 與小型公尺格網，只驗證時間缺口、距離門檻、
曲線交會、coverage 分母及截尾標籤；不把合成結果當作 OCM 正式科學結果。
"""

from __future__ import annotations

import numpy as np
import pytest

from traps.metrics import SiteMetricsConfig, compute_site_metrics, curve_intersects_mask
from traps.models import TrapDetection
from traps.tracking import TrackingConfig, TrackingFrame, track_structures

NS_PER_HOUR = 3_600_000_000_000


def _detection(
    x_center_m: float,
    *,
    y_center_m: float = 100.0,
    curve_half_length_m: float = 20.0,
    qualified: bool = True,
) -> TrapDetection:
    """建立具有完整核心／曲線統計欄位的簡化 TRAP detection。"""

    points = np.array(
        [[x_center_m - curve_half_length_m, y_center_m], [x_center_m, y_center_m], [x_center_m + curve_half_length_m, y_center_m]],
        dtype=np.float64,
    )
    strengths = np.full(points.shape[0], -1.0e-3, dtype=np.float64)
    return TrapDetection(
        core_index_ij=(1, round(x_center_m / 100.0)),
        core_xy_m=(x_center_m, y_center_m),
        core_s1_sinv=-1.0e-3,
        curve_points_m=points,
        curve_s1_sinv=strengths,
        curve_length_m=2.0 * curve_half_length_m,
        curve_mean_s1_sinv=-1.0e-3,
        curve_mean_abs_s1_sinv=1.0e-3,
        forward_stop_reason="synthetic",
        backward_stop_reason="synthetic",
        is_qualified=qualified,
    )


def _frame(time_hour: int, detections: tuple[TrapDetection, ...] | None, *, available: bool = True) -> TrackingFrame:
    """建立時間以整小時遞增的 TrackingFrame。"""

    return TrackingFrame(
        time_utc_ns=time_hour * NS_PER_HOUR,
        detections=detections,
        available=available,
    )


def test_tracking_is_one_to_one_and_closes_one_missing_frame() -> None:
    """兩條結構即使輸入順序交換，也只能各自配對且可跨一個缺失時刻。"""

    first_a = _detection(0.0)
    first_b = _detection(1_000.0)
    second_a = _detection(50.0)
    second_b = _detection(1_050.0)
    third_a = _detection(100.0)
    third_b = _detection(1_100.0)
    frames = (
        _frame(0, (first_a, first_b)),
        _frame(1, (second_b, second_a)),
        _frame(2, None, available=False),
        _frame(3, (third_b, third_a)),
    )
    result = track_structures(
        frames,
        TrackingConfig(
            max_core_distance_m=120.0,
            max_curve_distance_m=120.0,
            max_time_gap_seconds=4.0 * 3_600.0,
            max_missing_frames=1,
            curve_sample_step_m=50.0,
        ),
    )

    assert len(result.tracks) == 2
    assert all(len(track.observations) == 3 for track in result.tracks)
    assert all(track.missing_frame_count == 1 for track in result.tracks)
    assert all("gap_closed" in track.quality_flags for track in result.tracks)
    assert all(track.right_censored for track in result.tracks)
    assert len({observation.detection.core_xy_m[0] for track in result.tracks for observation in track.observations if observation.frame_index == 3}) == 2


def test_tracking_distinguishes_data_gap_no_detection_and_boundaries() -> None:
    """缺時、可用但無檢出與研究期邊界要保留不同標籤。"""

    frames = (
        _frame(0, (_detection(0.0),)),
        _frame(1, (), available=True),
        _frame(2, (_detection(1_000.0),)),
    )
    result = track_structures(frames, TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, curve_sample_step_m=50.0))
    first = result.tracks[0]
    second = result.tracks[1]
    assert first.termination_reason == "no_detection"
    assert first.left_censored and not first.right_censored
    assert second.termination_reason == "study_period_end"
    assert not second.left_censored and second.right_censored

    gap_end = track_structures(
        (_frame(0, (_detection(0.0),)), _frame(1, None, available=False), _frame(2, None, available=False)),
        TrackingConfig(max_core_distance_m=100.0, max_curve_distance_m=100.0, curve_sample_step_m=50.0),
    )
    assert gap_end.tracks[0].termination_reason == "data_gap"
    assert gap_end.tracks[0].right_censored


def test_curve_intersection_densifies_segment_and_does_not_use_core_only() -> None:
    """核心在站外、曲線跨過站點時仍應判定相交。"""

    x_m = np.arange(5, dtype=np.float64) * 100.0
    y_m = np.arange(3, dtype=np.float64) * 100.0
    site_mask = np.zeros((3, 5), dtype=bool)
    site_mask[:, 3] = True
    sparse_curve = np.array([[0.0, 100.0], [400.0, 100.0]], dtype=np.float64)
    assert curve_intersects_mask(sparse_curve, x_m, y_m, site_mask, max_step_m=50.0)
    assert not curve_intersects_mask(np.array([[0.0, 100.0], [200.0, 100.0]]), x_m, y_m, site_mask, max_step_m=50.0)

    detection = _detection(100.0, curve_half_length_m=300.0)
    frames = (TrackingFrame(0, (detection,), True, valid_mask=np.ones((3, 5), dtype=bool)),)
    metrics = compute_site_metrics(
        frames,
        {"site": site_mask},
        x_m=x_m,
        y_m=y_m,
        cell_area_m2=np.full((3, 5), 10_000.0),
        config=SiteMetricsConfig(coverage_threshold=1.0, curve_sample_step_m=50.0),
    )
    site = metrics.metrics["site"]
    assert site.valid_snapshot_count == 1
    assert site.occurrence_snapshot_count == 1
    assert site.core_count == 0
    assert site.curve_count == 1


def test_site_metrics_preserves_denominator_and_data_gap_counts() -> None:
    """可用無檢出仍進分母，data gap 則排除且另行計數。"""

    x_m = np.arange(5, dtype=np.float64) * 100.0
    y_m = np.arange(3, dtype=np.float64) * 100.0
    site_a = np.zeros((3, 5), dtype=bool)
    site_b = np.zeros((3, 5), dtype=bool)
    site_a[:, 0] = True
    site_b[:, 4] = True
    valid = np.ones((3, 5), dtype=bool)
    crossing = _detection(200.0, curve_half_length_m=250.0)
    frames = (
        TrackingFrame(0, (crossing,), True, valid_mask=valid),
        TrackingFrame(NS_PER_HOUR, (), True, valid_mask=valid),
        TrackingFrame(2 * NS_PER_HOUR, None, False),
    )
    metrics = compute_site_metrics(
        frames,
        {"gongliao": site_a, "guishan": site_b},
        x_m=x_m,
        y_m=y_m,
        cell_area_m2=np.full((3, 5), 10_000.0),
        config=SiteMetricsConfig(coverage_threshold=1.0, curve_sample_step_m=50.0),
    )
    for site in metrics.metrics.values():
        assert site.valid_snapshot_count == 2
        assert site.occurrence_snapshot_count == 1
        assert site.occurrence_rate == pytest.approx(0.5)
        assert site.data_gap_frame_count == 1
        assert site.no_detection_frame_count == 1

    with pytest.raises(ValueError, match="不可重疊"):
        compute_site_metrics(
            frames[:1],
            {"gongliao": site_a, "guishan": site_a.copy()},
            x_m=x_m,
            y_m=y_m,
            cell_area_m2=np.full((3, 5), 10_000.0),
        )


def test_site_metrics_zero_denominator_is_null() -> None:
    """沒有有效站點格點時，分母與出現率不可用零代替。"""

    x_m = np.arange(3, dtype=np.float64) * 100.0
    y_m = np.arange(3, dtype=np.float64) * 100.0
    empty_site = np.zeros((3, 3), dtype=bool)
    frames = (TrackingFrame(0, (), True, valid_mask=np.ones((3, 3), dtype=bool)),)
    metrics = compute_site_metrics(
        frames,
        {"empty": empty_site},
        x_m=x_m,
        y_m=y_m,
        cell_area_m2=np.full((3, 3), 10_000.0),
    )
    site = metrics.metrics["empty"]
    assert site.valid_snapshot_count == 0
    assert site.occurrence_snapshot_count == 0
    assert site.occurrence_rate is None
    assert "zero_denominator" in site.quality_flags


def test_tracking_rejects_more_than_one_gap_closing_slot() -> None:
    """配置若要求跨兩個缺失時刻，必須先被契約拒絕。"""

    with pytest.raises(ValueError, match="最多 gap closing"):
        TrackingConfig(max_missing_frames=2)
