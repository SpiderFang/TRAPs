"""條件資料 CSV/metadata gate 與文獻借鏡圖面的合成契約測試。

測試只使用人工的四時刻、單站點 catalog 與小型 rectilinear SurfaceMonth，驗證
缺檔、SHA、UTC、完整覆蓋、獨立性、QC、零分母及圖面輸出規則；合成資料不代表任何
真實潮相、漂流器或海廢觀測。
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from traps.conditional_figures import (
    ConditionalGateError,
    MarineDebrisTable,
    SpatialSummary,
    read_marine_debris_table,
    read_tidal_phase_table,
    read_trajectory_table,
    render_debris_trap_overlay_figure,
    render_drifter_trap_comparison_figure,
    render_particle_patch_deformation_figure,
    render_tidal_occurrence_figure,
    render_trajectory_trap_comparison_figure,
    summarize_particle_patch_deformation,
    summarize_tidal_occurrence,
    summarize_trajectory_trap_comparison,
)
from traps.ocm import SurfaceMonth

TIMES = tuple(1_725_096_000_000_000_000 + index * 3_600_000_000_000 for index in range(4))


def _catalog() -> dict:
    """建立一站四時刻 catalog，讓四種潮相分層各有一個有效分母。"""

    return {
        "state_label": "PILOT_ONLY",
        "science_status": "PILOT_ONLY",
        "source": {"domain_id": "synthetic", "month": "202501"},
        "site_statistics": [{"study_site_id": "nanwan", "site_area_m2": 1.0}],
        "snapshots": [
            {"snapshot_index": index, "time_utc_ns": time, "coverage_pass": True}
            for index, time in enumerate(TIMES)
        ],
        "detections": [
            {
                "snapshot_index": 0,
                "time_utc_ns": TIMES[0],
                "is_qualified": True,
                "study_site_ids": ["nanwan"],
                "curve_points_m": [[100.0, 200.0], [300.0, 200.0]],
            },
            {
                "snapshot_index": 2,
                "time_utc_ns": TIMES[2],
                "is_qualified": True,
                "study_site_ids": ["nanwan"],
                "curve_points_m": [[100.0, 200.0], [300.0, 200.0]],
            },
        ],
    }


def _write_conditional_csv(
    tmp_path: Path,
    stem: str,
    fieldnames: list[str],
    rows: list[dict[str, object]],
    *,
    table_kind: str,
    independence_status: str,
) -> Path:
    """寫入 UTF-8 CSV 與含實際 SHA-256 的同 stem metadata。"""

    csv_path = tmp_path / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": {
            "tidal_phase": "traps.conditional.tidal_phase.v1",
            "trajectory": "traps.conditional.trajectory.v1",
            "marine_debris": "traps.conditional.marine_debris.v1",
        }[table_kind],
        "table_kind": table_kind,
        "timezone": "UTC",
        "source": {"name": "synthetic test", "record_id": stem},
        "independence_status": independence_status,
        "units": {
            "time_utc_ns": "ns since Unix epoch UTC",
            "start_time_utc_ns": "ns since Unix epoch UTC",
            "end_time_utc_ns": "ns since Unix epoch UTC",
            "lon": "degrees_east",
            "lat": "degrees_north",
            "count": "items",
            "effort": "person_hour",
        },
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    csv_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path


def _month() -> SurfaceMonth:
    """建立供軌跡曲線轉經緯度的五乘五小型 SurfaceMonth。"""

    axis = np.arange(5, dtype=np.float64) * 100.0
    x_m, y_m = np.meshgrid(axis, axis, indexing="xy")
    valid = np.ones((4, 5, 5), dtype=bool)
    return SurfaceMonth(
        month_dir=Path("/synthetic/202501"),
        metadata={"month": "202501", "domain": {"domain_id": "synthetic"}},
        grid_metadata={"domain": {"domain_id": "synthetic"}},
        _time_utc_ns=np.asarray(TIMES, dtype=np.int64),
        _u=np.zeros((4, 5, 5), dtype=np.float32),
        _v=np.zeros((4, 5, 5), dtype=np.float32),
        _valid=valid,
        _qc_flags=np.ones((4, 5, 5), dtype=np.uint16),
        x_m=x_m,
        y_m=y_m,
        lon=120.0 + axis / 100.0,
        lat=23.0 + axis / 100.0,
        cell_area_m2=np.full((5, 5), 10_000.0, dtype=np.float32),
        mask_static=np.ones((5, 5), dtype=bool),
        analysis_ocean_mask=np.ones((5, 5), dtype=bool),
        analysis_mask_status="provided",
        coastline_mask_provenance={"path": "synthetic"},
        pilot_only=True,
    )


def _spatial_summary() -> SpatialSummary:
    """建立有 TRAP 通過率背景的合成 SpatialSummary。"""

    shape = (5, 5)
    valid = np.ones(shape, dtype=np.int64) * 4
    rate = np.zeros(shape, dtype=np.float64)
    rate[2, 1] = 0.5
    return SpatialSummary(
        domain_id="synthetic",
        month="202501",
        state_label="PILOT_ONLY",
        science_status="PILOT_ONLY",
        snapshot_indices=(0, 1, 2, 3),
        time_utc_ns=TIMES,
        lon=120.0 + np.arange(5) / 100.0,
        lat=23.0 + np.arange(5) / 100.0,
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


def _tidal_csv(tmp_path: Path) -> Path:
    """建立完整四時刻潮相表。"""

    rows = [
        {
            "site_id": "nanwan",
            "time_utc_ns": time,
            "phase": phase,
            "spring_neap": spring_neap,
            "qc_pass": "true",
        }
        for time, phase, spring_neap in zip(TIMES, ("flood", "ebb", "flood", "ebb"), ("spring", "spring", "neap", "neap"), strict=True)
    ]
    return _write_conditional_csv(tmp_path, "tidal", list(rows[0]), rows, table_kind="tidal_phase", independence_status="not_applicable")


def _trajectory_csv(tmp_path: Path, *, source_kind: str = "observed_drifter", independence_status: str = "confirmed_independent") -> Path:
    """建立沿同一 TRAP 曲線附近移動的短期軌跡。"""

    rows = [
        {
            "trajectory_id": "drifter-1",
            "time_utc_ns": TIMES[index],
            "lon": 121.0 + index / 1_000.0,
            "lat": 25.0,
            "source_kind": source_kind,
            "qc_pass": "true",
        }
        for index in range(2)
    ]
    return _write_conditional_csv(tmp_path, "trajectory", list(rows[0]), rows, table_kind="trajectory", independence_status=independence_status)


def _debris_csv(tmp_path: Path) -> Path:
    """建立與 catalog 時間窗相交的一筆海廢紀錄。"""

    rows = [
        {
            "record_id": "debris-1",
            "start_time_utc_ns": TIMES[0],
            "end_time_utc_ns": TIMES[1],
            "lon": 120.01,
            "lat": 23.02,
            "count": 10,
            "effort": 2,
            "qc_pass": "true",
        }
    ]
    return _write_conditional_csv(tmp_path, "debris", list(rows[0]), rows, table_kind="marine_debris", independence_status="observed_record")


def _assert_png(path: Path) -> None:
    """確認圖函式確實寫出 PNG，而非只回傳尚未建立的路徑。"""

    assert path.is_file()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_tidal_gate_requires_complete_site_time_coverage_and_computes_rates(tmp_path: Path) -> None:
    """潮相表 READY 時需完整覆蓋，並依合格 detection 計算分子／分母。"""

    report = read_tidal_phase_table(_tidal_csv(tmp_path), _catalog())
    assert report.status == "READY"
    occurrence = summarize_tidal_occurrence(report, _catalog())
    assert len(occurrence) == 4
    assert all(row.denominator == 1 for row in occurrence)
    assert sorted(row.numerator for row in occurrence) == [0, 0, 1, 1]

    incomplete = _catalog()
    truncated_csv = _tidal_csv(tmp_path)
    with truncated_csv.open("r", encoding="utf-8", newline="") as stream:
        truncated_rows = list(csv.DictReader(stream))[:3]
    incomplete_csv = _write_conditional_csv(
        tmp_path,
        "tidal_incomplete",
        list(truncated_rows[0]),
        truncated_rows,
        table_kind="tidal_phase",
        independence_status="not_applicable",
    )
    incomplete_report = read_tidal_phase_table(incomplete_csv, incomplete)
    assert incomplete_report.status == "DATA_REQUIRED"
    assert any("未完整涵蓋" in item for item in incomplete_report.missing)


def test_all_conditional_loaders_fail_closed_for_missing_sha_qc_and_independence(tmp_path: Path) -> None:
    """缺 sidecar、SHA、qc 或獨立性證據都不可進入 READY。"""

    missing = read_marine_debris_table(tmp_path / "missing.csv", catalog=_catalog())
    assert missing.status == "DATA_REQUIRED"

    trajectory = _trajectory_csv(tmp_path, independence_status="unconfirmed")
    blocked = read_trajectory_table(trajectory, catalog=_catalog())
    assert blocked.status == "QC_BLOCKED"
    assert any("confirmed_independent" in issue for issue in blocked.issues)

    metadata_path = trajectory.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["csv_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    sha_blocked = read_trajectory_table(trajectory, catalog=_catalog())
    assert sha_blocked.status == "QC_BLOCKED"
    assert any("SHA-256" in issue for issue in sha_blocked.issues)

    debris = _debris_csv(tmp_path)
    raw = list(csv.DictReader(debris.open(encoding="utf-8")))
    raw[0]["qc_pass"] = "false"
    with debris.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    # 修改 CSV 後 metadata SHA 故意不更新，gate 必須仍拒絕。
    debris_blocked = read_marine_debris_table(debris, catalog=_catalog())
    assert debris_blocked.status == "QC_BLOCKED"


def test_trajectory_comparison_distinguishes_observation_and_model_internal_status(tmp_path: Path) -> None:
    """觀測漂流器可做獨立 wrapper；模型粒子只能做 internal comparison。"""

    observed_report = read_trajectory_table(_trajectory_csv(tmp_path), catalog=_catalog())
    comparison = summarize_trajectory_trap_comparison(observed_report, _catalog(), _month())
    assert len(comparison) == 1
    assert comparison[0].mean_distance_to_trap_m is not None
    observed_figure = render_drifter_trap_comparison_figure(observed_report, _catalog(), _month(), tmp_path / "drifter.png")
    _assert_png(observed_figure)

    model_path = _trajectory_csv(tmp_path, source_kind="model_particle", independence_status="model_internal")
    model_report = read_trajectory_table(model_path, catalog=_catalog())
    model_figure = render_trajectory_trap_comparison_figure(model_report, _catalog(), _month(), tmp_path / "model.png")
    _assert_png(model_figure)
    with pytest.raises(ConditionalGateError, match="observed_drifter"):
        render_drifter_trap_comparison_figure(model_report, _catalog(), _month(), tmp_path / "bad.png")


def test_tidal_and_debris_figures_only_write_after_ready_gate(tmp_path: Path) -> None:
    """通過 gate 的潮相與海廢資料可產出圖，圖面保留 pilot 狀態。"""

    tidal_report = read_tidal_phase_table(_tidal_csv(tmp_path), _catalog())
    tidal_figure = render_tidal_occurrence_figure(tidal_report, _catalog(), tmp_path / "tidal.png")
    _assert_png(tidal_figure)

    debris_report = read_marine_debris_table(_debris_csv(tmp_path), catalog=_catalog())
    assert isinstance(debris_report.table, MarineDebrisTable)
    debris_figure = render_debris_trap_overlay_figure(debris_report, _catalog(), _spatial_summary(), tmp_path / "debris.png")
    _assert_png(debris_figure)

    # 即使 UTC 相交，域外紀錄也不能讓地圖自動放大並被誤讀為 TRAP 共現。
    outside_rows = [{
        "record_id": "outside-1",
        "start_time_utc_ns": TIMES[0],
        "end_time_utc_ns": TIMES[1],
        "lon": 121.0,
        "lat": 25.0,
        "count": 2,
        "effort": 1,
        "qc_pass": "true",
    }]
    outside_path = _write_conditional_csv(
        tmp_path, "outside_debris", list(outside_rows[0]), outside_rows,
        table_kind="marine_debris", independence_status="observed_record",
    )
    outside_report = read_marine_debris_table(outside_path, catalog=_catalog())
    with pytest.raises(ConditionalGateError, match="within TRAP flow domain"):
        render_debris_trap_overlay_figure(outside_report, _catalog(), _spatial_summary(), tmp_path / "outside.png")
    assert not (tmp_path / "outside.png").exists()


def test_model_particle_patch_needs_group_provenance_and_computes_area_ratio(tmp_path: Path) -> None:
    """同群三顆粒子兩個 UTC 時刻可算四倍面積；缺群 ID 必須拒絕產圖。"""

    rows = []
    for index, time in enumerate(TIMES[:2]):
        factor = float(index + 1)
        for particle_id, dx, dy in (("p1", 0.0, 0.0), ("p2", 0.001, 0.0), ("p3", 0.0, 0.001)):
            rows.append({
                "trajectory_id": particle_id,
                "time_utc_ns": time,
                "lon": 121.0 + factor * dx,
                "lat": 25.0 + factor * dy,
                "source_kind": "model_particle",
                "qc_pass": "true",
            })
    path = _write_conditional_csv(
        tmp_path, "patch", list(rows[0]), rows,
        table_kind="trajectory", independence_status="model_internal",
    )
    report = read_trajectory_table(path, catalog=_catalog())
    with pytest.raises(ConditionalGateError, match="patch_id"):
        render_particle_patch_deformation_figure(report, _catalog(), _month(), tmp_path / "blocked_patch.png")
    assert not (tmp_path / "blocked_patch.png").exists()

    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["patch_id"] = "synthetic_same_release"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    ready = read_trajectory_table(path, catalog=_catalog())
    summary = summarize_particle_patch_deformation(ready, _catalog())
    assert summary["particle_count"] == 3
    assert summary["final_to_initial_area_ratio"] == pytest.approx(4.0)
    _assert_png(render_particle_patch_deformation_figure(ready, _catalog(), _month(), tmp_path / "patch.png"))


def test_blocked_report_require_ready_never_creates_output(tmp_path: Path) -> None:
    """DATA_REQUIRED 報告經圖函式時要回報 gate，而不是建立空白 PNG。"""

    missing = read_tidal_phase_table(None, _catalog())
    with pytest.raises(ConditionalGateError) as error:
        render_tidal_occurrence_figure(missing, _catalog(), tmp_path / "should_not_exist.png")
    assert error.value.report.status == "DATA_REQUIRED"
    assert not (tmp_path / "should_not_exist.png").exists()
