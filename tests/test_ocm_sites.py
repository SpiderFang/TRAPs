"""OCM 月份讀取與四區域五站點遮罩的契約測試。

測試使用小型合成 `.npy` 快取，不連線 SERVER、不讀取 raw NetCDF，也不把合成資料當成
正式研究結果。每個 case 都保護同一組 fail-closed gate：schema/domain、shape/time、
finite-valid、pilot 旗標、analysis coastline sidecar，以及 A 區兩站的互斥聯集。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from traps.ocm import open_month
from traps.sites import load_study_design, site_masks


def _write_json(path: Path, payload: dict) -> None:
    """以 UTF-8 寫入小型測試 metadata；陣列仍由 ``np.save`` 以純 binary 保存。"""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_month(
    root: Path,
    *,
    domain_id: str = "synthetic_domain",
    cache_kind: str = "standard_month",
    status: str = "ready",
    u_shape: tuple[int, int, int] = (2, 3, 4),
) -> Path:
    """建立符合 schema 3.0 layout 的最小月份 fixture。

    fixture 的 ``mask_static`` 特意排除一格，讓測試能確認 valid 不會穿越靜態無效區；
    ``u/v`` 用有限 float32 常值表示，可在不引入數值方法的情況下專注驗證輸入 gate。
    """

    month_dir = root / "ocm_surface" / domain_id / "months" / "202501"
    grid_dir = month_dir.parent.parent / "grid"
    month_dir.mkdir(parents=True)
    grid_dir.mkdir(parents=True)
    time_count, ny, nx = u_shape
    u = np.ones(u_shape, dtype=np.float32)
    v = np.full(u_shape, 2.0, dtype=np.float32)
    valid = np.ones(u_shape, dtype=bool)
    static = np.ones((ny, nx), dtype=bool)
    static[0, 0] = False
    valid[:, 0, 0] = False
    qc_flags = np.ones(u_shape, dtype=np.uint16)
    time = np.arange(time_count, dtype=np.int64) + np.int64(1_725_096_000_000_000_000)
    lon = np.linspace(120.0, 120.3, nx, dtype=np.float32)
    lat = np.linspace(22.0, 22.2, ny, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x_m = (lon_grid - lon[0]) * 100_000.0
    y_m = (lat_grid - lat[0]) * 100_000.0
    cell_area = np.full((ny, nx), 1_000_000.0, dtype=np.float32)

    arrays = {
        "time_utc_ns.npy": time,
        "u_surface_mps.npy": u,
        "v_surface_mps.npy": v,
        "valid_mask_surface.npy": valid,
        "qc_flags.npy": qc_flags,
    }
    for name, array in arrays.items():
        np.save(month_dir / name, array, allow_pickle=False)
    grid_arrays = {
        "lon.npy": lon,
        "lat.npy": lat,
        "x_m.npy": x_m.astype(np.float32),
        "y_m.npy": y_m.astype(np.float32),
        "cell_area_m2.npy": cell_area,
        "mask_static.npy": static,
    }
    for name, array in grid_arrays.items():
        np.save(grid_dir / name, array, allow_pickle=False)

    metadata = {
        "status": status,
        "cache_kind": cache_kind,
        "cache_schema_version": "3.0.0",
        "month": "202501",
        "domain": {"domain_id": domain_id, "name_zh": "合成海域"},
        "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape)} for name, array in arrays.items()},
    }
    _write_json(month_dir / "metadata.json", metadata)
    _write_json(
        grid_dir / "metadata.json",
        {
            "cache_schema_version": "3.0.0",
            "domain": {"domain_id": domain_id},
            "grid_shape": [ny, nx],
        },
    )
    return month_dir


def test_open_month_returns_readonly_snapshot_and_marks_missing_coastline(tmp_path: Path) -> None:
    """開啟月份時保持 mmap，snapshot 只提供單時刻並明示未提供海岸 sidecar。"""

    month_dir = _write_month(tmp_path)
    month = open_month(month_dir, expected_domain_id="synthetic_domain")

    assert isinstance(month._u, np.memmap)
    assert month.snapshot_count == 2
    assert month.state_label == "READY"
    assert month.analysis_mask_status == "coastline_mask_missing"
    snapshot = month.snapshot(0)
    assert snapshot.time_utc_ns == 1_725_096_000_000_000_000
    assert snapshot.u.shape == (3, 4)
    assert snapshot.valid.shape == (3, 4)
    assert not snapshot.valid[0, 0]
    assert snapshot.valid[1, 1]


def test_open_month_rejects_schema_domain_shape_time_and_finite_invalidity(tmp_path: Path) -> None:
    """所有核心輸入 gate 都應在 TRAP 計算前 fail closed。"""

    cases = ("schema", "domain", "shape", "time", "finite")
    for case in cases:
        with pytest.raises(ValueError):
            month_dir = _write_month(tmp_path / case, domain_id="synthetic_domain")
            if case == "schema":
                metadata_path = month_dir / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["cache_schema_version"] = "2.0.0"
                _write_json(metadata_path, metadata)
                open_month(month_dir)
            elif case == "domain":
                open_month(month_dir, expected_domain_id="other_domain")
            elif case == "shape":
                np.save(month_dir / "v_surface_mps.npy", np.ones((2, 3, 3), dtype=np.float32), allow_pickle=False)
                open_month(month_dir)
            elif case == "time":
                np.save(
                    month_dir / "time_utc_ns.npy",
                    np.array([1_725_096_000_000_000_000, 1_725_096_000_000_000_000], dtype=np.int64),
                    allow_pickle=False,
                )
                open_month(month_dir)
            else:
                u = np.load(month_dir / "u_surface_mps.npy", mmap_mode="r+")
                u[0, 1, 1] = np.nan
                open_month(month_dir)


def test_trial_partial_month_requires_explicit_pilot_and_is_labelled(tmp_path: Path) -> None:
    """trial_partial_month 沒有明示旗標時拒絕，允許後固定帶 PILOT_ONLY。"""

    month_dir = _write_month(tmp_path, cache_kind="trial_partial_month", status="trial_ready")
    with pytest.raises(ValueError, match="allow_trial_pilot"):
        open_month(month_dir)
    month = open_month(month_dir, allow_trial_pilot=True)
    assert month.pilot_only
    assert month.state_label == "PILOT_ONLY"


def test_analysis_ocean_sidecar_is_subset_and_tracked(tmp_path: Path) -> None:
    """明確 sidecar 會套用到逐時 valid，並記錄上游共用 metadata 的 hash。"""

    month_dir = _write_month(tmp_path)
    grid_dir = month_dir.parent.parent / "grid"
    sidecar = grid_dir / "analysis_ocean_mask_osm_coastline_v1.npy"
    mask = np.ones((3, 4), dtype=bool)
    mask[1, 1] = False
    mask[0, 0] = False
    np.save(sidecar, mask, allow_pickle=False)
    sidecar_metadata = grid_dir / "analysis_land_mask_osm_coastline_v1.json"
    _write_json(
        sidecar_metadata,
        {
            "version_tag": "osm_coastline_v1",
            "cache_schema_version": "3.0.0",
            "domain": {"domain_id": "synthetic_domain"},
            "arrays": {"shape_lat_lon": [3, 4]},
        },
    )

    month = open_month(month_dir, analysis_ocean_mask_path=sidecar)
    assert month.analysis_mask_status == "provided"
    assert month.provenance["coastline_mask"]["version_tag"] == "osm_coastline_v1"
    assert not month.snapshot(0).valid[1, 1]
    # 同 shape 的其他 flow domain 海岸遮罩仍不能誤接到本月份。
    _write_json(
        sidecar_metadata,
        {
            "version_tag": "osm_coastline_v1",
            "cache_schema_version": "3.0.0",
            "domain": {"domain_id": "another_domain"},
            "arrays": {"shape_lat_lon": [3, 4]},
        },
    )
    with pytest.raises(ValueError, match="sidecar domain_id"):
        open_month(month_dir, analysis_ocean_mask_path=sidecar)


def test_a_domain_masks_are_disjoint_and_cover_static_grid(tmp_path: Path) -> None:
    """A 區最近 anchor 分割每個有效格點一次，且完整聯集等於 static mask。"""

    design = load_study_design()
    sites = {site["study_site_id"]: site for site in design["study_sites"]}
    gongliao_lon, gongliao_lat = sites["gongliao"]["anchor_lonlat"]
    guishan_lon, guishan_lat = sites["guishan"]["anchor_lonlat"]
    lon = np.array([[gongliao_lon, guishan_lon], [gongliao_lon, guishan_lon]], dtype=np.float64)
    lat = np.array([[gongliao_lat, guishan_lat], [gongliao_lat, guishan_lat]], dtype=np.float64)
    static = np.array([[True, True], [True, False]], dtype=bool)
    masks = site_masks("northeast_taiwan_common_cache_v3", lon, lat, static)
    assert not np.any(masks["gongliao"] & masks["guishan"])
    assert np.array_equal(masks["gongliao"] | masks["guishan"], static)
    assert masks["gongliao"][0, 0]
    assert masks["guishan"][0, 1]


def test_c_site_is_nanwan_with_houwan_flow_domain_and_full_domain_mask() -> None:
    """C 區顯示名稱採南灣，但其 flow product ID 固定沿用 houwan。"""

    design = load_study_design()
    nanwan = next(site for site in design["study_sites"] if site["study_site_id"] == "nanwan")
    assert nanwan["study_site_name_zh"] == "南灣"
    assert nanwan["flow_domain_id"] == "houwan_nmmba_cache_v3"

    lon = np.array([120.0, 120.1], dtype=np.float64)
    lat = np.array([21.9, 22.0], dtype=np.float64)
    static = np.array([[True, False], [True, True]], dtype=bool)
    masks = site_masks("houwan_nmmba_cache_v3", lon, lat, static)
    assert np.array_equal(masks["nanwan"], static)
    assert not np.any(masks["gongliao"])
