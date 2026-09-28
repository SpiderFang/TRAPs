"""方法比較與海廢共現圖的色條等高契約測試。

測試以小型合成 SurfaceSnapshot、SnapshotResult 與已通過 READY gate 的海廢表格
建立真實 Matplotlib 圖面。測試攔截實際 ``Figure.savefig`` 時的 figure 物件，直接
量取主圖及色條 axes 的上下界；因此不只檢查 helper 是否被呼叫，也能確認
``aspect='equal'``、多色條與輸出 PNG 共存時仍符合 0.5 point 的版面規則。合成資料
只驗證版面，不代表 OCM 流場、海廢觀測或正式研究結果。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from traps.conditional_figures import (
    ConditionalGateReport,
    MarineDebrisRecord,
    MarineDebrisTable,
    render_debris_trap_overlay_figure,
)
from traps.literature_figures import render_method_comparison_figure
from traps.models import DetectionConfig, SnapshotResult
from traps.ocm import SurfaceSnapshot
from traps.spatial import SpatialSummary

NS_PER_HOUR = 3_600_000_000_000


def _surface_and_result() -> tuple[SurfaceSnapshot, SnapshotResult]:
    """建立含二維局部公尺網格與有限 s1 場的合成方法圖輸入。"""

    axis = (np.arange(9, dtype=np.float64) - 4.0) * 100.0
    x_m, y_m = np.meshgrid(axis, axis, indexing="xy")
    shape = x_m.shape
    snapshot = SurfaceSnapshot(
        index=0,
        time_utc_ns=1_725_096_000_000_000_000,
        u=(-0.001 * x_m + 0.05).astype(np.float64),
        v=(-0.001 * y_m).astype(np.float64),
        valid=np.ones(shape, dtype=bool),
        x_m=x_m,
        y_m=y_m,
        lon=121.0 + axis / 100_000.0,
        lat=25.0 + axis / 100_000.0,
        cell_area_m2=np.full(shape, 10_000.0, dtype=np.float64),
    )
    valid = np.ones(shape, dtype=bool)
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
        detections=(),
        config=DetectionConfig(),
        grid_spacing_m=(100.0, 100.0),
    )
    return snapshot, result


def _spatial_summary() -> SpatialSummary:
    """建立 5×5 地理網格的 TRAP 通過率背景，供海廢圖使用。"""

    shape = (5, 5)
    valid = np.full(shape, 4, dtype=np.int64)
    rate = np.zeros(shape, dtype=np.float64)
    rate[2, 2] = 0.5
    return SpatialSummary(
        domain_id="synthetic",
        month="202501",
        state_label="PILOT_ONLY",
        science_status="PILOT_ONLY",
        snapshot_indices=(0, 1, 2, 3),
        time_utc_ns=tuple(1_725_096_000_000_000_000 + index * NS_PER_HOUR for index in range(4)),
        lon=120.0 + np.arange(5, dtype=np.float64) / 100.0,
        lat=23.0 + np.arange(5, dtype=np.float64) / 100.0,
        valid_snapshot_count=valid,
        core_count=np.zeros(shape, dtype=np.int64),
        mean_abs_core_s1_sinv=np.full(shape, np.nan),
        curve_pass_count=(rate * 4).astype(np.int64),
        curve_pass_rate=rate,
        complete_track_count=np.zeros(shape, dtype=np.int64),
        censored_track_count=np.zeros(shape, dtype=np.int64),
        mean_complete_track_duration_hours=np.full(shape, np.nan),
        mean_censored_track_observed_span_hours=np.full(shape, np.nan),
        metadata={},
    )


def _debris_report() -> tuple[ConditionalGateReport, dict[str, object]]:
    """建立與 catalog 時間窗相交的 READY 海廢 gate 報告。"""

    start = 1_725_096_000_000_000_000
    table = MarineDebrisTable(
        records=(
            MarineDebrisRecord(
                record_id="debris-1",
                start_time_utc_ns=start,
                end_time_utc_ns=start + NS_PER_HOUR,
                lon=120.02,
                lat=23.02,
                count=10.0,
                effort=2.0,
                qc_pass=True,
            ),
        ),
        metadata={
            "units": {"count": "items", "effort": "person_hour"},
        },
        csv_sha256="synthetic",
    )
    report = ConditionalGateReport(
        kind="marine_debris",
        status="READY",
        csv_path="synthetic.csv",
        metadata_path="synthetic.json",
        table=table,
        metadata=table.metadata,
    )
    catalog = {
        "state_label": "PILOT_ONLY",
        "science_status": "PILOT_ONLY",
        "source": {"domain_id": "synthetic", "month": "202501"},
        "snapshots": [
            {"time_utc_ns": start + index * NS_PER_HOUR}
            for index in range(4)
        ],
    }
    return report, catalog


def _capture_saved_figure(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """攔截實際 savefig 的 figure 參考，保留正常 PNG 輸出與 renderer 流程。"""

    # 先載入 pyplot；Matplotlib 初次匯入會以 Figure.savefig 的 docstring 建立
    # pyplot.savefig wrapper，若在匯入前替換方法會讓其無法辨識測試 wrapper 的類別。
    import matplotlib.pyplot as _plt

    del _plt
    from matplotlib.figure import Figure

    saved: dict[str, object] = {}
    original_savefig = Figure.savefig

    def capture(self: Figure, *args: object, **kwargs: object) -> object:
        """記錄 figure 後交回 Matplotlib 原始 savefig，避免測試跳過 PNG 驗證。"""

        saved["figure"] = self
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return saved


def _assert_same_vertical_bounds(main_axis: object, colorbar_axis: object, *, figure: object) -> None:
    """以輸出 figure 的實際英吋/DPI 將 axes fraction 差異換算為 point。"""

    main_position = main_axis.get_position()  # type: ignore[attr-defined]
    colorbar_position = colorbar_axis.get_position()  # type: ignore[attr-defined]
    # axes position 是畫布比例；換算成印刷 point 只需英吋高度乘 72，
    # DPI 決定像素數，不應再乘入而把驗收容差誤縮小。
    figure_height_points = figure.get_figheight() * 72.0  # type: ignore[attr-defined]
    assert abs(main_position.y0 - colorbar_position.y0) * figure_height_points <= 0.5
    assert abs(main_position.y1 - colorbar_position.y1) * figure_height_points <= 0.5


def test_method_comparison_colorbars_match_their_equal_aspect_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """方法三分圖的 s1 與散度色條各自和主圖實際上下界對齊。"""

    snapshot, result = _surface_and_result()
    saved = _capture_saved_figure(monkeypatch)
    output = render_method_comparison_figure(
        snapshot,
        result,
        tmp_path / "method.png",
        state_label="PILOT_ONLY",
        source_label="synthetic",
    )
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    figure = saved["figure"]
    axes = figure.axes  # type: ignore[attr-defined]
    # 方法圖先建立三個主 axes，再建立 s1 與 divergence 色條 axes；以實際順序
    # 取用可避免測試重新實作 colorbar 的搜尋條件，且能偵測未預期的額外 axes。
    assert len(axes) == 5
    _assert_same_vertical_bounds(axes[0], axes[3], figure=figure)
    _assert_same_vertical_bounds(axes[2], axes[4], figure=figure)


def test_debris_overlay_colorbars_both_match_the_map_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """海廢/TRAP 共現圖的兩條色條都與同一張地理主圖等高。"""

    report, catalog = _debris_report()
    saved = _capture_saved_figure(monkeypatch)
    output = render_debris_trap_overlay_figure(
        report,
        catalog,
        _spatial_summary(),
        tmp_path / "debris.png",
    )
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    figure = saved["figure"]
    axes = figure.axes  # type: ignore[attr-defined]
    # 單一地理主圖後面應只有 TRAP passage 與 debris rate 兩個色條 axes。
    assert len(axes) == 3
    _assert_same_vertical_bounds(axes[0], axes[1], figure=figure)
    _assert_same_vertical_bounds(axes[0], axes[2], figure=figure)
