"""局部公尺成果圖的 GeoJSON 海岸線疊圖契約測試。

測試使用小型合成 OCM snapshot，不把實際台灣海岸 GeoJSON 或任何 OCM 掩膜帶入
數值驗證。重點是確認繪圖層拿到同一 snapshot 的 WGS84 經緯度與局部公尺網格、
三分方法圖會在三個地理分圖疊線、岸線 z-order 低於 TRAP 曲線，以及缺軸時 fail
closed。海岸線只是圖面參考，因此測試以 monkeypatch 記錄共用繪圖 API 的參數，
不讓人工 polygon 的形狀影響 TRAP 數值結果。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from traps.figures import render_snapshot_figure
from traps.literature_figures import render_method_comparison_figure
from traps.models import DetectionConfig, SnapshotResult
from traps.ocm import SurfaceSnapshot


def _snapshot_and_result() -> tuple[SurfaceSnapshot, SnapshotResult]:
    """建立 5×5、具二維局部公尺網格與一維 CRS84 軸的合成輸入。

    x/y 的二維形狀模擬上游 OCM snapshot；lon/lat 保持一維，讓測試能確認
    ``draw_metric_coastline`` 收到的是 snapshot 原生座標，而非重新估算的域範圍。
    ``detections`` 留空仍足以測試方法圖的底圖與疊線契約。
    """

    axis = (np.arange(5, dtype=np.float64) - 2.0) * 100.0
    x_m, y_m = np.meshgrid(axis, axis, indexing="xy")
    shape = x_m.shape
    snapshot = SurfaceSnapshot(
        index=0,
        time_utc_ns=1_725_096_000_000_000_000,
        u=np.full(shape, 0.1, dtype=np.float64),
        v=np.full(shape, -0.05, dtype=np.float64),
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


def _assert_png(path: Path) -> None:
    """確認 renderer 產生有效 PNG，避免只測到函式呼叫而未寫出成果。"""

    assert path.is_file()
    assert path.stat().st_size > 100
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_snapshot_overlay_uses_native_axes_and_stays_below_trap_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """單時刻圖把原生座標傳給疊圖 API，且使用低於核心／曲線的 z-order。"""

    from traps import coastline as coastline_module

    snapshot, result = _snapshot_and_result()
    calls: list[dict[str, object]] = []

    def fake_draw(axis: object, coastline: object, **kwargs: object) -> int:
        """記錄共用 API 參數，不在測試中重新實作 GeoJSON 轉換。"""

        calls.append({"axis": axis, "coastline": coastline, **kwargs})
        return 1

    monkeypatch.setattr(coastline_module, "draw_metric_coastline", fake_draw)
    marker = object()
    output = render_snapshot_figure(
        snapshot,
        result,
        tmp_path / "snapshot.png",
        state_label="PILOT_ONLY",
        source_label="synthetic",
        coastline=marker,
    )

    _assert_png(output)
    assert len(calls) == 1
    call = calls[0]
    assert call["coastline"] is marker
    np.testing.assert_array_equal(call["lon_axis"], snapshot.lon)
    np.testing.assert_array_equal(call["lat_axis"], snapshot.lat)
    np.testing.assert_array_equal(call["x_axis"], snapshot.x_m)
    np.testing.assert_array_equal(call["y_axis"], snapshot.y_m)
    assert call["zorder"] == pytest.approx(1.5)
    assert float(call["linewidth"]) > 0.0


def test_method_comparison_overlay_is_applied_to_all_three_metric_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """方法比較圖的 TRAP、流線與散度三個局部公尺分圖都使用同一岸線轉換。"""

    from traps import coastline as coastline_module

    snapshot, result = _snapshot_and_result()
    calls: list[dict[str, object]] = []

    def fake_draw(axis: object, coastline: object, **kwargs: object) -> int:
        """記錄每個分圖的疊線呼叫，並回報一個合成 ring。"""

        calls.append({"axis": axis, "coastline": coastline, **kwargs})
        return 1

    monkeypatch.setattr(coastline_module, "draw_metric_coastline", fake_draw)
    marker = object()
    output = render_method_comparison_figure(
        snapshot,
        result,
        tmp_path / "method.png",
        state_label="PILOT_ONLY",
        source_label="synthetic",
        coastline=marker,
    )

    _assert_png(output)
    assert len(calls) == 3
    assert all(call["coastline"] is marker for call in calls)
    for call in calls:
        np.testing.assert_array_equal(call["lon_axis"], snapshot.lon)
        np.testing.assert_array_equal(call["lat_axis"], snapshot.lat)
        np.testing.assert_array_equal(call["x_axis"], snapshot.x_m)
        np.testing.assert_array_equal(call["y_axis"], snapshot.y_m)
        assert call["zorder"] == pytest.approx(1.5)


def test_metric_overlay_rejects_nonmonotonic_snapshot_axis(tmp_path: Path) -> None:
    """經緯度軸折返時必須明確停止，不能產生位置不可信的圖。"""

    snapshot, result = _snapshot_and_result()
    broken_lon = np.asarray(snapshot.lon).copy()
    broken_lon[2] = broken_lon[1] - 0.0001
    broken = replace(snapshot, lon=broken_lon)

    with pytest.raises(ValueError, match="lon 軸必須嚴格單調"):
        render_snapshot_figure(
            broken,
            result,
            tmp_path / "should_not_exist.png",
            state_label="PILOT_ONLY",
            source_label="synthetic",
            coastline=object(),
        )


def test_no_coastline_preserves_existing_snapshot_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """未提供 coastline 時不匯入或呼叫疊圖 API，維持既有輸出路徑。"""

    from traps import coastline as coastline_module

    snapshot, result = _snapshot_and_result()
    calls: list[object] = []

    def fake_draw(*args: object, **kwargs: object) -> int:
        """若無 coastline，此 sentinel 不應被觸發。"""

        calls.append((args, kwargs))
        return 1

    monkeypatch.setattr(coastline_module, "draw_metric_coastline", fake_draw)
    output = render_snapshot_figure(
        snapshot,
        result,
        tmp_path / "without_coastline.png",
        state_label="PILOT_ONLY",
        source_label="synthetic",
    )

    _assert_png(output)
    assert calls == []
