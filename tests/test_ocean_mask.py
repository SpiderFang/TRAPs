"""``traps.ocean_mask`` 匯出 SERVER 海岸線校正 sidecar 的契約測試。

測試只使用小型合成 ``.npy``/JSON fixture，模擬 SERVER 端
``ocm_surface`` 與 ``ocm_surface_coastline_corrected_v2`` 兩種網格 layout；不連線
SERVER、不讀取真實海岸線 GeoJSON。除了匯出函式本身的 fail-closed gate，最後一個
測試同時驗證匯出的 sidecar 可被 :func:`traps.ocm.open_month` 接受，
``analysis_mask_status`` 會是 ``"provided"``。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_ocm_sites import _write_month, _write_json
from traps.ocean_mask import export_ocean_mask_from_corrected_grid
from traps.ocm import open_month

_DOMAIN_ID = "synthetic_domain"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_corrected_grid(
    corrected_dir: Path,
    *,
    surface_grid_dir: Path,
    domain_id: str = _DOMAIN_ID,
    corrected_static: np.ndarray | None = None,
    include_original: bool = True,
    include_land_mask: bool = True,
    corrupt_lon: bool = False,
    wrong_source_sha256: bool = False,
) -> Path:
    """建立一份與 ``surface_grid_dir`` 對齊的合成校正網格目錄。

    預設情況下座標與標準網格逐位元相同，校正遮罩比標準遮罩少一個海洋格點，
    `mask_static_original` 與 `coastline_land_mask` 均齊備，且校正 metadata 記錄的
    `source_mask_static_sha256` 正確指向標準網格 `mask_static.npy` 的檔案雜湊。
    各關鍵字參數讓個別測試案例偏離這個「合格」基準，藉此觸發對應的驗證失敗。
    """

    corrected_dir.mkdir(parents=True, exist_ok=True)
    surface_static = np.load(surface_grid_dir / "mask_static.npy")
    lon = np.load(surface_grid_dir / "lon.npy")
    lat = np.load(surface_grid_dir / "lat.npy")
    x_m = np.load(surface_grid_dir / "x_m.npy")
    y_m = np.load(surface_grid_dir / "y_m.npy")

    if corrupt_lon:
        lon = lon.copy()
        lon[0] = lon[0] - 1.0

    np.save(corrected_dir / "lon.npy", lon, allow_pickle=False)
    np.save(corrected_dir / "lat.npy", lat, allow_pickle=False)
    np.save(corrected_dir / "x_m.npy", x_m, allow_pickle=False)
    np.save(corrected_dir / "y_m.npy", y_m, allow_pickle=False)

    if corrected_static is None:
        corrected_static = surface_static.copy()
        ocean_indices = np.argwhere(corrected_static)
        # 移除一個海洋格點以模擬海岸線校正排除的格點；fixture 保證至少有一格可移除。
        row, col = ocean_indices[0]
        corrected_static[row, col] = False
    np.save(corrected_dir / "mask_static.npy", corrected_static, allow_pickle=False)

    if include_original:
        np.save(corrected_dir / "mask_static_original.npy", surface_static, allow_pickle=False)
    if include_land_mask:
        land_mask = surface_static & ~corrected_static
        np.save(corrected_dir / "coastline_land_mask.npy", land_mask, allow_pickle=False)

    source_sha256 = "0" * 64 if wrong_source_sha256 else _sha256_file(surface_grid_dir / "mask_static.npy")
    _write_json(
        corrected_dir / "metadata.json",
        {
            "cache_schema_version": "3.0.0",
            "domain": {"domain_id": domain_id},
            "grid_shape": list(corrected_static.shape),
            "coastline_correction": {
                "version": "coastline_corrected_v2",
                "source_mask_static_sha256": source_sha256,
                "original_static_ocean_cell_count": int(np.count_nonzero(surface_static)),
                "corrected_static_ocean_cell_count": int(np.count_nonzero(corrected_static)),
            },
        },
    )
    return corrected_dir


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """建立一組合格的月份+標準網格與對齊的校正網格，回傳 (month_dir, corrected_grid_dir)。"""

    month_dir = _write_month(tmp_path / "surface", domain_id=_DOMAIN_ID)
    surface_grid_dir = month_dir.parent.parent / "grid"
    corrected_dir = _write_corrected_grid(tmp_path / "corrected" / _DOMAIN_ID / "grid", surface_grid_dir=surface_grid_dir)
    return month_dir, corrected_dir


def test_export_writes_sidecar_and_is_accepted_by_open_month(tmp_path: Path) -> None:
    """成功匯出的 sidecar 可被 ``open_month`` 接受，且 metadata 內容完整一致。"""

    month_dir, corrected_dir = _build_fixture(tmp_path)
    surface_grid_dir = month_dir.parent.parent / "grid"
    output_dir = tmp_path / "output"

    metadata = export_ocean_mask_from_corrected_grid(
        surface_grid_dir=surface_grid_dir,
        corrected_grid_dir=corrected_dir,
        domain_id=_DOMAIN_ID,
        output_dir=output_dir,
    )

    assert metadata["cache_schema_version"] == "3.0.0"
    assert metadata["cache_kind"] == "coastline_corrected_export"
    assert metadata["domain"]["domain_id"] == _DOMAIN_ID
    surface_static = np.load(surface_grid_dir / "mask_static.npy")
    corrected_static = np.load(corrected_dir / "mask_static.npy")
    assert metadata["arrays"]["static_ocean_cell_count"] == int(np.count_nonzero(surface_static))
    assert metadata["arrays"]["analysis_ocean_cell_count"] == int(np.count_nonzero(corrected_static))
    assert metadata["arrays"]["coastline_land_cells_inside_static_mask"] == int(
        np.count_nonzero(surface_static & ~corrected_static)
    )

    ocean_mask_path = output_dir / "analysis_ocean_mask_coastline_corrected_v2.npy"
    land_mask_path = output_dir / "analysis_land_mask_coastline_corrected_v2.npy"
    metadata_path = output_dir / "analysis_land_mask_coastline_corrected_v2.json"
    assert ocean_mask_path.is_file()
    assert land_mask_path.is_file()
    assert metadata_path.is_file()
    assert np.array_equal(np.load(ocean_mask_path), corrected_static)
    on_disk_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert on_disk_metadata == metadata

    month = open_month(month_dir, expected_domain_id=_DOMAIN_ID, analysis_ocean_mask_path=ocean_mask_path)
    assert month.analysis_mask_status == "provided"
    assert np.array_equal(month.analysis_ocean_mask, corrected_static)


def test_export_rejects_mismatched_coordinates(tmp_path: Path) -> None:
    """校正網格的座標若與標準網格不同，必須 fail closed。"""

    month_dir = _write_month(tmp_path / "surface", domain_id=_DOMAIN_ID)
    surface_grid_dir = month_dir.parent.parent / "grid"
    corrected_dir = _write_corrected_grid(
        tmp_path / "corrected" / _DOMAIN_ID / "grid", surface_grid_dir=surface_grid_dir, corrupt_lon=True
    )

    with pytest.raises(ValueError, match="lon.npy"):
        export_ocean_mask_from_corrected_grid(
            surface_grid_dir=surface_grid_dir,
            corrected_grid_dir=corrected_dir,
            domain_id=_DOMAIN_ID,
            output_dir=tmp_path / "output",
        )


def test_export_rejects_corrected_mask_not_subset(tmp_path: Path) -> None:
    """校正遮罩若把標準網格判定無效的格點標成有效，必須 fail closed。"""

    month_dir = _write_month(tmp_path / "surface", domain_id=_DOMAIN_ID)
    surface_grid_dir = month_dir.parent.parent / "grid"
    surface_static = np.load(surface_grid_dir / "mask_static.npy")
    bad_corrected = surface_static.copy()
    bad_corrected[0, 0] = True  # `_write_month` 保證這個格點在標準網格中是陸地/無效。
    corrected_dir = _write_corrected_grid(
        tmp_path / "corrected" / _DOMAIN_ID / "grid",
        surface_grid_dir=surface_grid_dir,
        corrected_static=bad_corrected,
        include_original=False,
        include_land_mask=False,
    )

    with pytest.raises(ValueError, match="子集"):
        export_ocean_mask_from_corrected_grid(
            surface_grid_dir=surface_grid_dir,
            corrected_grid_dir=corrected_dir,
            domain_id=_DOMAIN_ID,
            output_dir=tmp_path / "output",
        )


def test_export_rejects_source_sha256_mismatch(tmp_path: Path) -> None:
    """校正 metadata 記錄的來源雜湊若與標準網格 mask_static 檔案不符，必須 fail closed。"""

    month_dir = _write_month(tmp_path / "surface", domain_id=_DOMAIN_ID)
    surface_grid_dir = month_dir.parent.parent / "grid"
    corrected_dir = _write_corrected_grid(
        tmp_path / "corrected" / _DOMAIN_ID / "grid", surface_grid_dir=surface_grid_dir, wrong_source_sha256=True
    )

    with pytest.raises(ValueError, match="source_mask_static_sha256"):
        export_ocean_mask_from_corrected_grid(
            surface_grid_dir=surface_grid_dir,
            corrected_grid_dir=corrected_dir,
            domain_id=_DOMAIN_ID,
            output_dir=tmp_path / "output",
        )


def test_export_refuses_nonempty_output_dir(tmp_path: Path) -> None:
    """輸出目錄若非新目錄或非空目錄，必須拒絕覆寫。"""

    month_dir, corrected_dir = _build_fixture(tmp_path)
    surface_grid_dir = month_dir.parent.parent / "grid"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "leftover.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="空目錄"):
        export_ocean_mask_from_corrected_grid(
            surface_grid_dir=surface_grid_dir,
            corrected_grid_dir=corrected_dir,
            domain_id=_DOMAIN_ID,
            output_dir=output_dir,
        )


def test_export_validates_input_manifest_region(tmp_path: Path) -> None:
    """提供 input manifest 時，會核對對應 region 的雜湊與格點計數。"""

    month_dir, corrected_dir = _build_fixture(tmp_path)
    surface_grid_dir = month_dir.parent.parent / "grid"
    surface_static = np.load(surface_grid_dir / "mask_static.npy")
    corrected_static = np.load(corrected_dir / "mask_static.npy")
    manifest_path = tmp_path / "coastline_corrected_surface_input_manifest.json"

    _write_json(
        manifest_path,
        {
            "schema_name": "coastline_corrected_surface_input_manifest",
            "schema_version": "1.0.0",
            "regions": [
                {
                    "region_key": "synthetic",
                    "flow_domain_id": _DOMAIN_ID,
                    "corrected_mask_static_sha256": _sha256_file(corrected_dir / "mask_static.npy"),
                    "original_static_ocean_cell_count": int(np.count_nonzero(surface_static)),
                    "corrected_static_ocean_cell_count": int(np.count_nonzero(corrected_static)),
                }
            ],
        },
    )

    metadata = export_ocean_mask_from_corrected_grid(
        surface_grid_dir=surface_grid_dir,
        corrected_grid_dir=corrected_dir,
        domain_id=_DOMAIN_ID,
        output_dir=tmp_path / "output_ok",
        input_manifest_path=manifest_path,
    )
    assert metadata["sources"]["input_manifest_path"] == str(manifest_path)

    bad_manifest_path = tmp_path / "bad_manifest.json"
    _write_json(
        bad_manifest_path,
        {
            "regions": [
                {
                    "region_key": "synthetic",
                    "flow_domain_id": _DOMAIN_ID,
                    "corrected_mask_static_sha256": "0" * 64,
                    "original_static_ocean_cell_count": int(np.count_nonzero(surface_static)),
                    "corrected_static_ocean_cell_count": int(np.count_nonzero(corrected_static)),
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="corrected_mask_static_sha256"):
        export_ocean_mask_from_corrected_grid(
            surface_grid_dir=surface_grid_dir,
            corrected_grid_dir=corrected_dir,
            domain_id=_DOMAIN_ID,
            output_dir=tmp_path / "output_bad",
            input_manifest_path=bad_manifest_path,
        )
