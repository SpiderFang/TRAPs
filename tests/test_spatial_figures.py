"""逐格空間統計與 Kunz/Ghosh 借鏡四分圖的合成契約測試。

測試建立最小 OCM schema 3.0 月份，刻意安排同一 UTC 時刻兩條重疊曲線、未通過品質
門檻的曲線、逐格缺值，以及一條完整與一條右截尾 track。這些條件直接對應空間圖的
主要科學風險；合成數值只驗證介面與分母規則，不代表任何真實海域結果。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from traps.ocm import SurfaceMonth, open_month
from traps.spatial import SpatialSummary, build_spatial_summary
from traps.spatial_figures import render_spatial_summary_figure

DOMAIN_ID = "synthetic_spatial_domain"
TIMES = (1_725_096_000_000_000_000, 1_725_099_600_000_000_000, 1_725_103_200_000_000_000)


def _write_json(path: Path, payload: dict) -> None:
    """寫入與上游 cache 相同 UTF-8 JSON 結構，讓測試實際通過 SurfaceMonth gate。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _open_synthetic_month(tmp_path: Path) -> SurfaceMonth:
    """建立五乘五公尺格網的已驗證 SurfaceMonth，第二時刻故意留一個缺值格。"""

    month_dir = tmp_path / "ocm_surface" / DOMAIN_ID / "months" / "202501"
    grid_dir = tmp_path / "ocm_surface" / DOMAIN_ID / "grid"
    month_dir.mkdir(parents=True)
    grid_dir.mkdir(parents=True)
    axis_m = np.arange(5, dtype=np.float64) * 100.0
    x_m, y_m = np.meshgrid(axis_m, axis_m, indexing="xy")
    u = np.zeros((3, 5, 5), dtype=np.float32)
    v = np.zeros_like(u)
    valid = np.ones_like(u, dtype=bool)
    valid[1, 2, 2] = False
    qc_flags = np.ones_like(u, dtype=np.uint16)
    cell_area = np.full((5, 5), 10_000.0, dtype=np.float32)
    lon = 120.0 + axis_m / 100.0
    lat = 23.0 + axis_m / 100.0
    for name, array in {
        "time_utc_ns.npy": np.asarray(TIMES, dtype=np.int64),
        "u_surface_mps.npy": u,
        "v_surface_mps.npy": v,
        "valid_mask_surface.npy": valid,
        "qc_flags.npy": qc_flags,
    }.items():
        np.save(month_dir / name, array, allow_pickle=False)
    for name, array in {
        "lon.npy": lon,
        "lat.npy": lat,
        "x_m.npy": x_m,
        "y_m.npy": y_m,
        "cell_area_m2.npy": cell_area,
        "mask_static.npy": np.ones((5, 5), dtype=bool),
    }.items():
        np.save(grid_dir / name, array, allow_pickle=False)
    np.save(grid_dir / "analysis_ocean_mask_synthetic_v1.npy", np.ones((5, 5), dtype=bool), allow_pickle=False)
    _write_json(
        month_dir / "metadata.json",
        {
            "status": "ready",
            "cache_kind": "standard_month",
            "cache_schema_version": "3.0.0",
            "month": "202501",
            "domain": {"domain_id": DOMAIN_ID},
        },
    )
    _write_json(
        grid_dir / "metadata.json",
        {"cache_schema_version": "3.0.0", "domain": {"domain_id": DOMAIN_ID}, "grid_shape": [5, 5]},
    )
    _write_json(
        grid_dir / "analysis_land_mask_synthetic_v1.json",
        {
            "cache_schema_version": "3.0.0",
            "domain": {"domain_id": DOMAIN_ID},
            "arrays": {"shape_lat_lon": [5, 5]},
        },
    )
    return open_month(
        month_dir,
        analysis_ocean_mask_path=grid_dir / "analysis_ocean_mask_synthetic_v1.npy",
    )


def _catalog() -> dict:
    """建立含重疊曲線、品質不合格曲線及完整/截尾 track 的 catalog。"""

    curve = [[100.0, 200.0], [300.0, 200.0]]
    second_curve = [[100.0, 200.0], [300.0, 200.0]]
    other_curve = [[300.0, 400.0], [400.0, 400.0]]
    snapshots = [
        {"snapshot_index": index, "time_utc_ns": time, "coverage_pass": True}
        for index, time in enumerate(TIMES)
    ]
    detections = [
        {
            "snapshot_index": 0,
            "time_utc_ns": TIMES[0],
            "is_qualified": True,
            "core_x_m": 200.0,
            "core_y_m": 200.0,
            "core_s1_sinv": -2.0e-4,
            "curve_points_m": curve,
            "track_id": 1,
        },
        {
            "snapshot_index": 0,
            "time_utc_ns": TIMES[0],
            "is_qualified": True,
            "core_x_m": 200.0,
            "core_y_m": 300.0,
            "core_s1_sinv": -4.0e-4,
            "curve_points_m": second_curve,
            "track_id": 2,
        },
        {
            "snapshot_index": 0,
            "time_utc_ns": TIMES[0],
            "is_qualified": False,
            "core_x_m": 400.0,
            "core_y_m": 400.0,
            "core_s1_sinv": -9.0e-4,
            "curve_points_m": other_curve,
        },
        {
            "snapshot_index": 1,
            "time_utc_ns": TIMES[1],
            "is_qualified": True,
            "core_x_m": 200.0,
            "core_y_m": 200.0,
            "core_s1_sinv": -2.0e-4,
            "curve_points_m": curve,
            "track_id": 1,
        },
    ]
    return {
        "schema_version": "traps_catalog_v1",
        "state_label": "READY",
        "science_status": "PROVISIONAL_UNITS_UNCONFIRMED",
        "source": {"domain_id": DOMAIN_ID, "month": "202501"},
        "parameters": {"detection_config": {"min_core_s1_sinv": -1.0e-8}},
        "snapshots": snapshots,
        "detections": detections,
        "tracks": [
            {
                "track_id": 1,
                "duration_seconds": 7_200.0,
                "left_censored": False,
                "right_censored": False,
            },
            {
                "track_id": 2,
                "duration_seconds": 3_600.0,
                "left_censored": False,
                "right_censored": True,
            },
        ],
    }


def test_spatial_summary_preserves_cell_denominators_and_deduplicates_curves(tmp_path: Path) -> None:
    """逐格分母、合格篩選與同時刻曲線 union 應符合研究契約。"""

    month = _open_synthetic_month(tmp_path)
    summary = build_spatial_summary(_catalog(), month)

    assert isinstance(summary, SpatialSummary)
    assert summary.valid_snapshot_count[0, 0] == 3
    assert summary.valid_snapshot_count[2, 2] == 2
    assert summary.curve_pass_count[2, 1] == 2
    # 兩條合格曲線在第 0 時刻重疊，故該格最多只算一次；第 1 時刻再算一次。
    assert summary.curve_pass_count[2, 1] <= summary.valid_snapshot_count[2, 1]
    assert summary.curve_pass_rate[2, 1] == pytest.approx(2.0 / 3.0)
    # 不合格曲線落在右下格，不得進入空間通過率。
    assert summary.curve_pass_count[4, 4] == 0


def test_spatial_summary_separates_complete_and_censored_lifetimes(tmp_path: Path) -> None:
    """完整 track 填入完整壽命，右截尾 track 只填入觀測跨度下界。"""

    summary = build_spatial_summary(_catalog(), _open_synthetic_month(tmp_path))
    assert summary.complete_track_count[2, 1] == 1
    assert summary.censored_track_count[2, 1] == 1
    assert summary.mean_complete_track_duration_hours[2, 1] == pytest.approx(2.0)
    assert summary.mean_censored_track_observed_span_hours[2, 1] == pytest.approx(1.0)
    assert np.isnan(summary.mean_complete_track_duration_hours[4, 4])


def test_spatial_figure_has_geographic_axes_and_explicit_empty_lifetime(tmp_path: Path) -> None:
    """四分圖應產生 PNG，且沒有完整樣本時以不足文字區分 0 小時。"""

    month = _open_synthetic_month(tmp_path)
    catalog = _catalog()
    catalog["tracks"] = [
        {
            "track_id": 2,
            "duration_seconds": 3_600.0,
            "left_censored": False,
            "right_censored": True,
        }
    ]
    for row in catalog["detections"]:
        if row.get("track_id") == 1:
            row.pop("track_id")
    summary = build_spatial_summary(catalog, month)
    output = render_spatial_summary_figure(summary, tmp_path / "spatial_summary.png")
    assert output.is_file()
    assert output.stat().st_size > 1_000
    assert summary.to_metadata()["coordinate_units"] == "degrees"
    assert summary.to_metadata()["field_units"]["mean_complete_track_duration_hours"] == "hour"


def test_spatial_summary_rejects_catalog_time_mismatch(tmp_path: Path) -> None:
    """catalog 與 SurfaceMonth 的 UTC 對不上時必須 fail closed。"""

    catalog = _catalog()
    catalog["snapshots"][1]["time_utc_ns"] += 1
    with pytest.raises(ValueError, match="UTC 時間"):
        build_spatial_summary(catalog, _open_synthetic_month(tmp_path))
