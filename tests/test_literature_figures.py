"""文獻借鏡圖型的合成資料契約測試。

測試只驗證圖型介面、單位／狀態欄位與完整／截尾 track 的分流；人工速度場與
catalog 不是 OCM 正式結果，也不代表任何觀測或 2024--2025 長期統計。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from traps.literature_figures import (
    _divergence_sinv,
    render_lifecycle_intensity_figure,
    render_lifetime_distribution_figure,
    render_mechanism_figure,
    render_method_comparison_figure,
    summarize_track_lifetimes,
)
from traps.models import DetectionConfig, SnapshotResult, TrapDetection
from traps.ocm import SurfaceSnapshot

NS_PER_HOUR = 3_600_000_000_000


def _surface_and_result() -> tuple[SurfaceSnapshot, SnapshotResult]:
    """建立帶一條合成 TRAP 曲線的 9 x 9 公尺制快照。"""

    axis = (np.arange(9, dtype=np.float64) - 4.0) * 100.0
    x_m, y_m = np.meshgrid(axis, axis, indexing="xy")
    u = (-0.001 * x_m + 0.05).astype(np.float64)
    v = (-0.001 * y_m).astype(np.float64)
    valid = np.ones(u.shape, dtype=bool)
    valid[0, 0] = False
    lon = 121.0 + axis / 100_000.0
    lat = 25.0 + axis / 100_000.0
    snapshot = SurfaceSnapshot(
        index=0,
        time_utc_ns=1_725_096_000_000_000_000,
        u=u,
        v=v,
        valid=valid,
        x_m=x_m,
        y_m=y_m,
        lon=lon,
        lat=lat,
        cell_area_m2=np.full(u.shape, 10_000.0, dtype=np.float64),
    )
    detection = TrapDetection(
        core_index_ij=(4, 4),
        core_xy_m=(0.0, 0.0),
        core_s1_sinv=-1.0e-3,
        curve_points_m=np.asarray([[-200.0, 0.0], [0.0, 0.0], [200.0, 0.0]], dtype=np.float64),
        curve_s1_sinv=np.full(3, -1.0e-3, dtype=np.float64),
        curve_length_m=400.0,
        curve_mean_s1_sinv=-1.0e-3,
        curve_mean_abs_s1_sinv=1.0e-3,
        forward_stop_reason="synthetic",
        backward_stop_reason="synthetic",
        is_qualified=True,
    )
    shape = u.shape
    result = SnapshotResult(
        s1_sinv=np.full(shape, -1.0e-4, dtype=np.float64),
        s2_sinv=np.full(shape, 1.0e-4, dtype=np.float64),
        sxx_sinv=np.zeros(shape, dtype=np.float64),
        sxy_sinv=np.zeros(shape, dtype=np.float64),
        syy_sinv=np.zeros(shape, dtype=np.float64),
        e2_x=np.ones(shape, dtype=np.float64),
        e2_y=np.zeros(shape, dtype=np.float64),
        derivative_valid=valid,
        core_candidate_mask=valid,
        input_valid_mask=valid,
        detections=(detection,),
        config=DetectionConfig(),
        grid_spacing_m=(100.0, 100.0),
    )
    return snapshot, result


def _catalog() -> dict:
    """建立一個含完整、右截尾與左截尾 track 的小型 catalog。"""

    def track(track_id: int, duration_hours: float, *, left: bool, right: bool, reason: str) -> dict:
        start = track_id * 10 * NS_PER_HOUR
        return {
            "track_id": track_id,
            "observation_count": 2,
            "start_time_utc_ns": start,
            "end_time_utc_ns": start + int(duration_hours * NS_PER_HOUR),
            "duration_seconds": duration_hours * 3_600.0,
            "termination_reason": reason,
            "left_censored": left,
            "right_censored": right,
            "missing_frame_count": 0,
            "quality_flags": [],
        }

    tracks = [
        track(1, 2.0, left=False, right=False, reason="no_detection"),
        track(2, 3.0, left=False, right=True, reason="study_period_end"),
        track(3, 1.0, left=True, right=False, reason="no_detection"),
    ]
    detections = []
    for track_id, start_hour in ((1, 10), (2, 20), (3, 30)):
        for age_hour, strength in ((0, 1.0e-3), (1, 8.0e-4), (2, 6.0e-4)):
            detections.append(
                {
                    "track_id": track_id,
                    "time_utc_ns": (start_hour + age_hour) * NS_PER_HOUR,
                    "curve_mean_abs_s1_sinv": strength,
                }
            )
    return {
        "schema_version": "0.1",
        "state_label": "PILOT_ONLY",
        "science_status": "PILOT_ONLY",
        "source": {"domain_id": "synthetic", "month": "202501"},
        "tracks": tracks,
        "detections": detections,
    }


def _assert_png(path: Path) -> None:
    """確認輸出確實是非空 PNG，而不是只建立空檔案。"""

    assert path.is_file()
    assert path.stat().st_size > 100
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_mechanism_and_method_figures_keep_concept_and_snapshot_boundaries(tmp_path: Path) -> None:
    """示意圖與單時刻方法圖都可輸出，且不需要獨立觀測資料。"""

    snapshot, result = _surface_and_result()
    axis = (np.arange(9, dtype=np.float64) - 4.0) * 100.0
    divergence = _divergence_sinv(snapshot.u, snapshot.v, axis, axis, snapshot.valid)
    # 合成場 u=-0.001*x、v=-0.001*y，因此內部有效格點的散度應為 -0.002 s^-1。
    np.testing.assert_allclose(divergence[4, 4], -2.0e-3, atol=1.0e-12)
    mechanism = render_mechanism_figure(tmp_path / "mechanism.png")
    comparison = render_method_comparison_figure(
        snapshot,
        result,
        tmp_path / "method_comparison.png",
        state_label="PILOT_ONLY",
        source_label="synthetic/202501",
    )
    _assert_png(mechanism)
    _assert_png(comparison)


def test_lifetime_summary_excludes_censored_tracks_and_writes_two_figures(tmp_path: Path) -> None:
    """完整壽命只含未截尾 track，截尾樣本另列為觀測跨度下界。"""

    catalog = _catalog()
    summary = summarize_track_lifetimes(catalog)
    assert summary["complete_track_count"] == 1
    assert summary["censored_track_count"] == 2
    assert summary["complete_duration_hours"] == [2.0]
    assert sorted(summary["censored_observed_span_hours"]) == [1.0, 3.0]
    assert summary["duration_semantics"].startswith("first_to_last_detection_observed_span")
    assert summary["lifetime_status"] == "complete_tracks_available"

    lifetime = render_lifetime_distribution_figure(catalog, tmp_path / "lifetime_distribution.png")
    lifecycle = render_lifecycle_intensity_figure(catalog, tmp_path / "lifecycle_intensity.png")
    _assert_png(lifetime)
    _assert_png(lifecycle)

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    loaded_summary = summarize_track_lifetimes(catalog_path)
    assert loaded_summary["complete_duration_hours"] == [2.0]


def test_lifetime_summary_does_not_infer_missing_censor_flags() -> None:
    """缺少邊界旗標的舊 catalog 不得被推測成完整 track。"""

    catalog = {"state_label": "PILOT_ONLY", "tracks": [{"track_id": 1, "duration_seconds": 3_600.0}], "detections": []}
    summary = summarize_track_lifetimes(catalog)
    assert summary["complete_track_count"] == 0
    assert summary["censored_track_count"] == 1
    assert summary["lifetime_status"] == "no_complete_track_duration"
