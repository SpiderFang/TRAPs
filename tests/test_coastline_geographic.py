"""CRS84 海岸線讀取、裁切、座標映射與圖面疊線契約測試。

測試同時涵蓋本專案實際台灣 GeoJSON 的 provenance 摘要與人工合成的錯誤案例。
合成 ring 只用來檢查資料介面、域內／域外裁切及軸範圍保護，不代表海岸線科學
結果，也不會改變任何 TRAP 數值或 ocean mask。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from traps.coastline import (
    CoastlineGeometry,
    CoastlineValidationError,
    draw_geographic_coastline,
    draw_metric_coastline,
    load_coastline,
)

SOURCE = Path(__file__).parents[1] / "data" / "taiwan_exact_coastline" / "taiwan_exact_coastline.geojson"


def _write_geojson(
    path: Path,
    *,
    geometry_type: str = "Polygon",
    coordinates: object | None = None,
    crs_name: str = "urn:ogc:def:crs:OGC:1.3:CRS84",
) -> Path:
    """建立最小 FeatureCollection，讓每個錯誤案例只改一個輸入條件。"""

    if coordinates is None:
        coordinates = [[[120.0, 23.0], [120.2, 23.0], [120.2, 23.2], [120.0, 23.0]]]
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": [{"type": "Feature", "geometry": {"type": geometry_type, "coordinates": coordinates}, "properties": {}}],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _synthetic_coastline(tmp_path: Path) -> CoastlineGeometry:
    """載入一個含域內與域外 ring 的小型 CRS84 海岸線。"""

    inside = [[[120.1, 23.1], [120.3, 23.1], [120.3, 23.3], [120.1, 23.1]]]
    outside = [[[121.0, 24.0], [121.2, 24.0], [121.2, 24.2], [121.0, 24.0]]]
    path = tmp_path / "coastline.geojson"
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": inside}, "properties": {}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": outside}, "properties": {}},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_coastline(path)


def test_real_taiwan_source_has_expected_provenance_and_serializable_metadata() -> None:
    """真實來源的 feature/ring 數與 SHA 必須鎖定，metadata 可直接寫入 manifest。"""

    coastline = load_coastline(SOURCE)
    assert coastline.crs == "urn:ogc:def:crs:OGC:1.3:CRS84"
    assert coastline.feature_count == 1_905
    assert coastline.ring_count == 1_912
    assert coastline.point_count == 151_106
    assert coastline.sha256 == "9e2e0ac9bc527aca87d89332cd428fdcb776eefbf94a85dd70f887f729b95fdd"
    assert coastline.visual_overlay_only is True
    json.dumps(coastline.to_metadata(), ensure_ascii=False)


def test_load_coastline_rejects_wrong_crs_and_unclosed_or_nonfinite_ring(tmp_path: Path) -> None:
    """CRS、閉合性及有限經緯度是進入任何成果圖前的 fail-closed gate。"""

    wrong_crs = _write_geojson(tmp_path / "wrong_crs.geojson", crs_name="EPSG:4326")
    with pytest.raises(CoastlineValidationError, match="只接受 CRS84"):
        load_coastline(wrong_crs)

    unclosed = _write_geojson(
        tmp_path / "unclosed.geojson",
        coordinates=[[[120.0, 23.0], [120.2, 23.0], [120.2, 23.2], [120.0, 23.1]]],
    )
    with pytest.raises(CoastlineValidationError, match="必須閉合"):
        load_coastline(unclosed)

    nonfinite = _write_geojson(
        tmp_path / "nonfinite.geojson",
        coordinates=[[[120.0, 23.0], [float("nan"), 23.0], [120.2, 23.2], [120.0, 23.0]]],
    )
    with pytest.raises(CoastlineValidationError, match="非有限"):
        load_coastline(nonfinite)


def test_load_coastline_accepts_multipolygon_and_counts_all_rings(tmp_path: Path) -> None:
    """MultiPolygon 的外環與內環都保留，因為輪廓圖需顯示洞的邊界。"""

    coordinates = [
        [
            [[120.0, 23.0], [120.4, 23.0], [120.4, 23.4], [120.0, 23.0]],
            [[120.1, 23.1], [120.2, 23.1], [120.2, 23.2], [120.1, 23.1]],
        ],
        [[[120.5, 23.0], [120.6, 23.0], [120.6, 23.1], [120.5, 23.0]]],
    ]
    coastline = load_coastline(_write_geojson(tmp_path / "multi.geojson", geometry_type="MultiPolygon", coordinates=coordinates))
    assert coastline.feature_count == 1
    assert coastline.ring_count == 3
    assert coastline.point_count == 12


def test_geographic_overlay_clips_domain_and_preserves_existing_axis_limits(tmp_path: Path) -> None:
    """域外 ring 不應畫出，且 outline 不可透過 autoscale 改變 flow-domain 軸。"""

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    coastline = _synthetic_coastline(tmp_path)
    figure, axis = plt.subplots()
    axis.set_xlim(120.0, 120.5)
    axis.set_ylim(23.0, 23.5)
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    drawn = draw_geographic_coastline(axis, coastline, bbox=(120.0, 23.0, 120.5, 23.5))
    assert drawn == 1
    assert axis.get_xlim() == x_limits
    assert axis.get_ylim() == y_limits
    assert len(axis.lines) == 1
    plt.close(figure)


def test_metric_overlay_maps_lonlat_to_snapshot_metric_axes_without_domain_expansion(tmp_path: Path) -> None:
    """局部圖使用同一 OCM 軸線性映射，域外 ring 不可被夾到邊界製造假線。"""

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    coastline = _synthetic_coastline(tmp_path)
    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 500.0)
    axis.set_ylim(0.0, 500.0)
    drawn = draw_metric_coastline(
        axis,
        coastline,
        lon_axis=np.array([120.0, 120.5]),
        lat_axis=np.array([23.0, 23.5]),
        x_axis=np.array([0.0, 500.0]),
        y_axis=np.array([0.0, 500.0]),
    )
    assert drawn == 1
    assert len(axis.lines) == 1
    assert float(np.min(axis.lines[0].get_xdata())) >= 0.0
    assert float(np.max(axis.lines[0].get_xdata())) <= 500.0
    assert axis.get_xlim() == (0.0, 500.0)
    # 即使各軸本身都單調，地理／公尺格點數不一致仍會造成錯位；
    # 拒絕此輸入也可保證錯誤情況不在既有圖上留下部分岸線。
    with pytest.raises(CoastlineValidationError, match="lon_axis 與 x_axis 長度"):
        draw_metric_coastline(
            axis,
            coastline,
            lon_axis=np.array([120.0, 120.25, 120.5]),
            lat_axis=np.array([23.0, 23.5]),
            x_axis=np.array([0.0, 500.0]),
            y_axis=np.array([0.0, 500.0]),
        )
    assert len(axis.lines) == 1
    plt.close(figure)
