"""CLI 海岸線來源、疊圖啟用狀態與 provenance 的端到端測試。

測試沿用小型 OCM cache fixture，數值檢出仍由既有 CLI pipeline 負責；本檔只驗證
GeoJSON 來源如何進入 ``detect --render-snapshots``、``render --kind snapshot``
及 ``literature-figures``。海岸線是 visual-only 圖層，故測試同時確認 catalog 的
TRAP 結構欄位仍在、manifest／figure availability 有完整 OSM/ODbL metadata，及
預設檔缺失與顯式錯誤的差異。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from test_cli_pipeline import DOMAIN_ID, _run_cli, _write_month

from traps import cli


def _write_minimal_coastline(path: Path) -> Path:
    """寫入一個 CRS84 小 polygon，作為顯式覆寫檔的可重現 fixture。"""

    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"fixture": True},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[120.762, 21.945], [120.764, 21.945], [120.764, 21.947], [120.762, 21.945]]],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _detect_with_sidecar(tmp_path: Path, *extra: str) -> tuple[Path, Path]:
    """建立帶 analysis ocean mask 的單時刻 catalog，供三條 CLI 路徑重用。"""

    month_dir = _write_month(tmp_path)
    sidecar = month_dir.parent.parent / "grid" / "analysis_ocean_mask_synthetic_v1.npy"
    output_dir = tmp_path / "detect"
    result = _run_cli(
        "detect",
        "--month-dir",
        str(month_dir),
        "--ocean-mask",
        str(sidecar),
        "--output-root",
        str(output_dir),
        "--snapshot-index",
        "0",
        *extra,
    )
    assert result.returncode == 0, result.stderr
    return month_dir, output_dir


def _assert_coastline_metadata(metadata: dict, *, enabled: bool, source_mode: str) -> None:
    """核對所有命令共用的海岸線 provenance 欄位與授權語意。"""

    assert metadata["enabled"] is enabled
    assert metadata["activation_status"] == ("enabled" if enabled else "default_missing")
    assert metadata["source_mode"] == source_mode
    assert metadata["visual_overlay_only"] is True
    assert metadata["license"] == "ODbL"
    assert "OpenStreetMap" in metadata["source"]
    assert "ODbL" in metadata["attribution"]
    assert "no bathymetry" in metadata["source_semantics"]
    if enabled:
        assert len(metadata["sha256"]) == 64
        assert metadata["crs"] in {"CRS84", "OGC:CRS84", "urn:ogc:def:crs:OGC:1.3:CRS84"}
        assert metadata["feature_count"] >= 1
        assert metadata["ring_count"] >= 1
        assert metadata["point_count"] >= 4
    else:
        assert metadata["sha256"] is None


def test_detect_snapshot_and_manifest_record_default_coastline(tmp_path: Path) -> None:
    """detect snapshot 圖使用專案預設海岸線，且不改動 catalog 統計欄位。"""

    _month_dir, output_dir = _detect_with_sidecar(tmp_path, "--render-snapshots")
    catalog = json.loads((output_dir / "catalog.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    source_metadata = catalog["source"]["coastline_geojson"]
    _assert_coastline_metadata(source_metadata, enabled=True, source_mode="default")
    assert source_metadata["path"] == str(cli.DEFAULT_COASTLINE_GEOJSON_PATH.resolve())
    assert source_metadata["sha256"] == hashlib.sha256(cli.DEFAULT_COASTLINE_GEOJSON_PATH.read_bytes()).hexdigest()
    _assert_coastline_metadata(manifest["coastline_geojson"], enabled=True, source_mode="default")
    assert (output_dir / "figures" / "snapshot_000000.png").is_file()
    # 這些欄位是數值 catalog 的既有內容，海岸線疊圖不得移除或重算其統計語意。
    assert len(catalog["detections"]) >= 0
    assert len(catalog["site_statistics"]) == 5


def test_render_snapshot_and_literature_availability_record_coastline(tmp_path: Path) -> None:
    """render snapshot 與 literature-figures 都把同一份來源寫入輸出清冊。"""

    month_dir, detect_dir = _detect_with_sidecar(tmp_path)
    sidecar = month_dir.parent.parent / "grid" / "analysis_ocean_mask_synthetic_v1.npy"
    render_dir = tmp_path / "render"
    rendered = _run_cli(
        "render",
        "--kind",
        "snapshot",
        "--month-dir",
        str(month_dir),
        "--ocean-mask",
        str(sidecar),
        "--snapshot-index",
        "0",
        "--output-root",
        str(render_dir),
    )
    assert rendered.returncode == 0, rendered.stderr
    render_payload = json.loads((render_dir / "render.json").read_text(encoding="utf-8"))
    render_manifest = json.loads((render_dir / "manifest.json").read_text(encoding="utf-8"))
    _assert_coastline_metadata(render_payload["coastline_geojson"], enabled=True, source_mode="default")
    _assert_coastline_metadata(render_manifest["coastline_geojson"], enabled=True, source_mode="default")

    literature_dir = tmp_path / "literature"
    literature = _run_cli(
        "literature-figures",
        "--catalog",
        str(detect_dir / "catalog.json"),
        "--output-root",
        str(literature_dir),
    )
    assert literature.returncode == 0, literature.stderr
    availability = json.loads((literature_dir / "figure_availability.json").read_text(encoding="utf-8"))
    literature_manifest = json.loads((literature_dir / "manifest.json").read_text(encoding="utf-8"))
    _assert_coastline_metadata(availability["coastline_geojson"], enabled=True, source_mode="default")
    _assert_coastline_metadata(literature_manifest["coastline_geojson"], enabled=True, source_mode="default")
    domain = availability["domains"][DOMAIN_ID]
    _assert_coastline_metadata(domain["coastline_geojson"], enabled=True, source_mode="default")
    assert (literature_dir / domain["figures"]["method_comparison"]).is_file()
    assert (literature_dir / domain["figures"]["spatial_summary"]).is_file()


def test_explicit_coastline_override_and_missing_default_are_distinct(tmp_path: Path, monkeypatch) -> None:
    """顯式路徑必須成功載入或拒絕；預設檔缺失則明示停用且不阻斷命令。"""

    override = _write_minimal_coastline(tmp_path / "override.geojson")
    geometry, metadata = cli._resolve_coastline_geojson(argparse.Namespace(coastline_geojson=override))
    assert geometry is not None
    _assert_coastline_metadata(metadata, enabled=True, source_mode="explicit_override")
    assert metadata["path"] == str(override.resolve())

    missing_default = tmp_path / "missing-default.geojson"
    monkeypatch.setattr(cli, "DEFAULT_COASTLINE_GEOJSON_PATH", missing_default)
    geometry, metadata = cli._resolve_coastline_geojson(argparse.Namespace(coastline_geojson=None))
    assert geometry is None
    _assert_coastline_metadata(metadata, enabled=False, source_mode="default")
    assert metadata["path"] == str(missing_default.resolve())

    missing_explicit = tmp_path / "missing-explicit.geojson"
    result = _run_cli(
        "render",
        "--kind",
        "snapshot",
        "--month-dir",
        str(_write_month(tmp_path / "explicit_error")),
        "--snapshot-index",
        "0",
        "--output-root",
        str(tmp_path / "explicit_error_render"),
        "--coastline-geojson",
        str(missing_explicit),
    )
    assert result.returncode != 0
    assert "--coastline-geojson 指定檔案不存在" in result.stderr
