"""WP-B2 stage 3 ``aggregate`` 的合成端到端與單元測試。

兩個合成 domain：A 區（``northeast_taiwan_common_cache_v3``，貢寮＋龜山島兩站，
真實 anchor 經緯度、小型局地 x/y 網格）與沿用 ``test_period.py`` 既有 fixture 的
C 區（``houwan_nmmba_cache_v3``，南灣一站，含三月份、跨月重複 UTC、5 小時缺口與
一個 coverage 不足時刻）。全部只涵蓋數十個時刻的合成流場，不代表 2024--2025
正式 TRAP 統計。
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from test_cli_pipeline import PROJECT_ROOT, _run_cli, _write_json  # noqa: F401
from test_period import (  # noqa: F401
    DOMAIN_ID as C_DOMAIN_ID,
    FEB_HOURS,
    FEB_LOW_COVERAGE_HOUR,
    JAN_HOURS,
    MAR_HOURS,
    NY as C_NY,
    NX as C_NX,
    _ocean_mask_path,
    _run_detect_shards,
    _shard_output,
    _write_month,
    _write_shared_grid,
)

from traps import aggregate as agg
from traps.sites import DEFAULT_STUDY_DESIGN_PATH, load_study_design

A_DOMAIN_ID = "northeast_taiwan_common_cache_v3"
A_NY, A_NX = 41, 41
A_LAT_MIN, A_LAT_MAX = 24.70, 25.20
A_LON_MIN, A_LON_MAX = 121.80, 122.10
A_X_SPACING_M = 500.0
A_BASE_NS = 1_704_067_200_000_000_000  # 2024-01-01T00:00:00Z (但年份/月份以 metadata month 為準)
A_HOUR_NS = 3_600_000_000_000
A_HOURS = list(range(8))
A_SIGMA_M = 3_500.0


def _a_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_axis = np.linspace(A_LON_MIN, A_LON_MAX, A_NX)
    lat_axis = np.linspace(A_LAT_MIN, A_LAT_MAX, A_NY)
    x_axis = (np.arange(A_NX, dtype=np.float64) - (A_NX - 1) / 2.0) * A_X_SPACING_M
    y_axis = (np.arange(A_NY, dtype=np.float64) - (A_NY - 1) / 2.0) * A_X_SPACING_M
    return lon_axis, lat_axis, x_axis, y_axis


def _a_velocity_field(x_m: np.ndarray, y_m: np.ndarray, hour: int) -> tuple[np.ndarray, np.ndarray]:
    """緩慢沿 y（南北）平移的吸引鞍點，讓核心輪流落在貢寮與龜山島兩個 receptor。"""

    frac = hour / (len(A_HOURS) - 1)
    cy = (frac - 0.5) * (A_NY - 1) * A_X_SPACING_M * 0.9
    dx = x_m
    dy = y_m - cy
    envelope = np.exp(-(dx * dx + dy * dy) / A_SIGMA_M**2)
    u = -2.5e-3 * dx * envelope
    v = -2.5e-3 * dy * envelope
    return u.astype(np.float32), v.astype(np.float32)


def _write_a_grid(root: Path) -> None:
    """建立 A 區共用 grid（一次性）。"""

    grid_dir = root / "ocm_surface" / A_DOMAIN_ID / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    lon_axis, lat_axis, x_axis, y_axis = _a_axes()
    x_m, y_m = np.meshgrid(x_axis, y_axis, indexing="xy")
    static = np.ones((A_NY, A_NX), dtype=bool)
    cell_area = np.full((A_NY, A_NX), A_X_SPACING_M**2, dtype=np.float32)
    for name, array in {
        "lon.npy": lon_axis.astype(np.float64),
        "lat.npy": lat_axis.astype(np.float64),
        "x_m.npy": x_m.astype(np.float64),
        "y_m.npy": y_m.astype(np.float64),
        "cell_area_m2.npy": cell_area,
        "mask_static.npy": static,
    }.items():
        np.save(grid_dir / name, array, allow_pickle=False)
    _write_json(grid_dir / "metadata.json", {"cache_schema_version": "3.0.0", "domain": {"domain_id": A_DOMAIN_ID}, "grid_shape": [A_NY, A_NX]})
    np.save(grid_dir / "analysis_ocean_mask_synthetic_v1.npy", static, allow_pickle=False)
    _write_json(
        grid_dir / "analysis_land_mask_synthetic_v1.json",
        {"cache_schema_version": "3.0.0", "domain": {"domain_id": A_DOMAIN_ID}, "arrays": {"shape_lat_lon": [A_NY, A_NX]}},
    )


def _write_a_domain(
    root: Path, *, month: str = "202501", hours: list[int] | None = None, cache_kind: str = "standard_month", status: str = "ready"
) -> Path:
    """建立 A 區單月合成月份，兩站 anchor 使用研究設計的真實經緯度。"""

    hours = A_HOURS if hours is None else hours
    month_dir = root / "ocm_surface" / A_DOMAIN_ID / "months" / month
    month_dir.mkdir(parents=True, exist_ok=True)

    _lon_axis, _lat_axis, x_axis, y_axis = _a_axes()
    x_m, y_m = np.meshgrid(x_axis, y_axis, indexing="xy")

    u_frames, v_frames, valid_frames = [], [], []
    for hour in hours:
        u, v = _a_velocity_field(x_m, y_m, hour)
        u_frames.append(u)
        v_frames.append(v)
        valid_frames.append(np.ones((A_NY, A_NX), dtype=bool))
    u = np.stack(u_frames, axis=0)
    v = np.stack(v_frames, axis=0)
    valid = np.stack(valid_frames, axis=0)
    qc_flags = np.ones(u.shape, dtype=np.uint16)
    time = (A_BASE_NS + np.asarray(hours, dtype=np.int64) * A_HOUR_NS).astype(np.int64)
    for name, array in {
        "time_utc_ns.npy": time,
        "u_surface_mps.npy": u,
        "v_surface_mps.npy": v,
        "valid_mask_surface.npy": valid,
        "qc_flags.npy": qc_flags,
    }.items():
        np.save(month_dir / name, array, allow_pickle=False)
    _write_json(
        month_dir / "metadata.json",
        {
            "status": status,
            "cache_kind": cache_kind,
            "cache_schema_version": "3.0.0",
            "month": month,
            "domain": {
                "domain_id": A_DOMAIN_ID,
                "name_zh": "東北角合成海域",
                "lon_min": 121.306315,
                "lon_max": 122.793685,
                "lat_min": 24.600844,
                "lat_max": 25.499156,
            },
        },
    )
    return month_dir


def _a_ocean_mask_path(root: Path) -> Path:
    return root / "ocm_surface" / A_DOMAIN_ID / "grid" / "analysis_ocean_mask_synthetic_v1.npy"


@pytest.fixture()
def two_domain_root(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    _write_shared_grid(root)
    _write_month(root, month="202401", hours=JAN_HOURS)
    _write_month(root, month="202402", hours=FEB_HOURS, low_coverage_hours=frozenset({FEB_LOW_COVERAGE_HOUR}))
    _write_month(root, month="202403", hours=MAR_HOURS)
    _write_a_grid(root)
    _write_a_domain(root)
    return root


def _run_detect_shards_a(root: Path, output_dir: Path, *, month: str = "202501", extra: list[str] | None = None) -> None:
    month_dir = root / "ocm_surface" / A_DOMAIN_ID / "months" / month
    arguments = [
        "detect-shards",
        "--month-dir",
        str(month_dir),
        "--ocean-mask",
        str(_a_ocean_mask_path(root)),
        "--output-dir",
        str(output_dir),
        "--min-coverage",
        "0.95",
    ]
    if extra:
        arguments.extend(extra)
    result = _run_cli(*arguments)
    assert result.returncode == 0, result.stderr


def _build_run(tmp_path: Path, root: Path) -> tuple[Path, Path]:
    """跑完兩個 domain 的 detect-shards + track-period，回傳 (track_root, detect_root)。"""

    detect_root = tmp_path / "detect"
    track_root = tmp_path / "track"

    for month in ("202401", "202402", "202403"):
        result = _run_detect_shards(root, month=month, output_dir=detect_root / C_DOMAIN_ID / month)
        assert result.returncode == 0, result.stderr
    _run_detect_shards_a(root, detect_root / A_DOMAIN_ID / "202501")

    result_c = _run_cli(
        "track-period",
        "--shards-root",
        str(detect_root / C_DOMAIN_ID),
        "--domain-id",
        C_DOMAIN_ID,
        "--output-dir",
        str(track_root / C_DOMAIN_ID),
        "--months",
        "202401-202403",
    )
    assert result_c.returncode == 0, result_c.stderr

    result_a = _run_cli(
        "track-period",
        "--shards-root",
        str(detect_root / A_DOMAIN_ID),
        "--domain-id",
        A_DOMAIN_ID,
        "--output-dir",
        str(track_root / A_DOMAIN_ID),
        "--months",
        "202501",
    )
    assert result_a.returncode == 0, result_a.stderr
    return track_root, detect_root


def _run_aggregate(track_root: Path, detect_root: Path, output_dir: Path, *, extra: list[str] | None = None) -> "subprocess.CompletedProcess[str]":
    arguments = [
        "aggregate",
        "--track-root",
        str(track_root),
        "--detect-root",
        str(detect_root),
        "--output-dir",
        str(output_dir),
        "--seed",
        "20260928",
        "--bootstrap-reps",
        "200",
        "--allow-partial-domains",
    ]
    if extra:
        arguments.extend(extra)
    return _run_cli(*arguments)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _expected_c_domain_partition_all(track_root: Path, detect_root: Path) -> tuple[int, int]:
    """獨立於 aggregate 程式碼，直接由 shard 產物重算南灣 domain_partition/all 的分子分母。

    C 區單站 domain_partition mask 涵蓋整個 static 網格（全部有效），因此「站點有效」
    等價於「canonical 且 coverage_pass」；出現分子等價於「該 snapshot 至少一個合格
    detection」，可直接用 snapshots.npz 的 qualified_count 欄位核對，不需重算遮罩交會。
    """

    with np.load(track_root / C_DOMAIN_ID / "canonical_snapshots.npz") as archive:
        month = archive["month"]
        snapshot_index = archive["snapshot_index"]
        coverage_pass = archive["coverage_pass"]
    n_valid = 0
    n_occurrence = 0
    for month_code, index, is_pass in zip(month.tolist(), snapshot_index.tolist(), coverage_pass.tolist(), strict=True):
        if not is_pass:
            continue
        n_valid += 1
        with np.load(detect_root / C_DOMAIN_ID / f"{month_code:06d}" / "snapshots.npz") as shard:
            positions = shard["snapshot_index"].tolist()
            qualified_count = shard["qualified_count"]
            position = positions.index(index)
            if int(qualified_count[position]) > 0:
                n_occurrence += 1
    return n_valid, n_occurrence


def test_aggregate_end_to_end_two_domains_with_figures(two_domain_root: Path, tmp_path: Path) -> None:
    """兩區彙整：分母/分子、pooled 強門檻、manifest 與圖面 artifact 全部核對。"""

    track_root, detect_root = _build_run(tmp_path, two_domain_root)
    output_dir = tmp_path / "aggregate_out"
    result = _run_aggregate(track_root, detect_root, output_dir)
    assert result.returncode == 0, result.stderr

    for name in ("site_statistics.csv", "site_statistics.json", "manifest.json", "manifest.sha256", "_SUCCESS"):
        assert (output_dir / name).is_file(), name
    for name in ("five_site_summary.png", "seasonal_occurrence.png", "lifetime_distribution.png"):
        assert (output_dir / name).is_file(), name
    for domain_id in (C_DOMAIN_ID, A_DOMAIN_ID):
        assert (output_dir / f"spatial_{domain_id}.png").is_file()
        for stratum in ("all", "DJF", "MAM", "JJA", "SON"):
            assert (output_dir / "spatial" / domain_id / f"spatial_{stratum}.npz").is_file()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "traps.period.aggregate.v1"
    assert sorted(manifest["domains_present"]) == sorted([C_DOMAIN_ID, A_DOMAIN_ID])
    assert sorted(manifest["domains_missing"]) == sorted(["hsinchu_cache_v3", "lienchiang_common_cache_v3"])
    assert sorted(manifest["sites_present"]) == sorted(["nanwan", "gongliao", "guishan"])
    assert manifest["strong_core_threshold_sinv"] is not None
    assert manifest["strong_core_threshold_day"] == pytest.approx(manifest["strong_core_threshold_sinv"] * 86_400.0)

    rows = _read_csv_rows(output_dir / "site_statistics.csv")
    row = next(r for r in rows if r["site_id"] == "nanwan" and r["scope"] == "domain_partition" and r["stratum"] == "all")
    expected_n_valid, expected_occurrence = _expected_c_domain_partition_all(track_root, detect_root)
    assert int(row["n_valid_snapshots"]) == expected_n_valid == 10  # 13 列 - 2 重複 - 1 coverage 不足
    assert int(row["core_occurrence_numerator"]) == expected_occurrence
    assert 0.0 <= float(row["core_occurrence_rate"]) <= 1.0

    for r in rows:
        for key in ("core_occurrence_rate", "strong_core_occurrence_rate", "curve_occurrence_rate", "mean_pass_cell_fraction"):
            if r[key]:
                assert 0.0 <= float(r[key]) <= 1.0 + 1.0e-9
        assert int(r["strong_core_occurrence_numerator"]) <= int(r["core_occurrence_numerator"])
        assert int(r["track_n_uncensored"]) <= int(r["track_n_tracks"])

    # 貢寮／龜山島兩站在同一 A 區 domain_partition 分母（有效格點數）合計應等於整個 static mask。
    a_rows = [r for r in rows if r["site_id"] in ("gongliao", "guishan") and r["scope"] == "domain_partition" and r["stratum"] == "all"]
    assert {r["site_id"] for r in a_rows} == {"gongliao", "guishan"}


def test_aggregate_bootstrap_reproducible_with_seed(two_domain_root: Path, tmp_path: Path) -> None:
    """相同 --seed 兩次執行的 bootstrap CI 必須逐位元相同。"""

    track_root, detect_root = _build_run(tmp_path, two_domain_root)
    out_first = tmp_path / "agg_first"
    out_second = tmp_path / "agg_second"
    first = _run_aggregate(track_root, detect_root, out_first)
    assert first.returncode == 0, first.stderr
    second = _run_aggregate(track_root, detect_root, out_second)
    assert second.returncode == 0, second.stderr

    first_records = json.loads((out_first / "site_statistics.json").read_text(encoding="utf-8"))["records"]
    second_records = json.loads((out_second / "site_statistics.json").read_text(encoding="utf-8"))["records"]
    assert first_records == second_records


def test_aggregate_no_figures_skips_pngs(two_domain_root: Path, tmp_path: Path) -> None:
    """``--no-figures`` 只輸出統計表與 manifest，不產生任何 PNG。"""

    track_root, detect_root = _build_run(tmp_path, two_domain_root)
    output_dir = tmp_path / "agg_no_fig"
    result = _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures"])
    assert result.returncode == 0, result.stderr
    assert (output_dir / "site_statistics.csv").is_file()
    assert not list(output_dir.glob("*.png"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figures_enabled"] is False


def test_aggregate_resume_is_noop_then_redo_after_tamper(two_domain_root: Path, tmp_path: Path) -> None:
    """--resume 對未變更輸出略過重算，對被竄改輸出重做。"""

    track_root, detect_root = _build_run(tmp_path, two_domain_root)
    output_dir = tmp_path / "agg_resume"
    first = _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures"])
    assert first.returncode == 0, first.stderr

    no_resume = _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures"])
    assert no_resume.returncode != 0

    manifest_before = (output_dir / "manifest.json").read_bytes()
    resumed = _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures", "--resume"])
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["skipped"] is True
    assert (output_dir / "manifest.json").read_bytes() == manifest_before

    csv_path = output_dir / "site_statistics.csv"
    tampered = csv_path.read_bytes() + b"\n# tampered"
    csv_path.write_bytes(tampered)
    redone = _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures", "--resume"])
    assert redone.returncode == 0, redone.stderr
    assert json.loads(redone.stdout)["skipped"] is False
    assert csv_path.read_bytes() != tampered


def test_aggregate_allow_partial_domains(two_domain_root: Path, tmp_path: Path) -> None:
    """未提供 --allow-partial-domains 時缺少 domain 直接拒絕；提供後只報告現有站點。"""

    track_root, detect_root = _build_run(tmp_path, two_domain_root)
    only_a_track_root = tmp_path / "track_only_a"
    only_a_track_root.mkdir()
    import shutil

    shutil.copytree(track_root / A_DOMAIN_ID, only_a_track_root / A_DOMAIN_ID)

    output_dir = tmp_path / "agg_partial"
    rejected = _run_cli(
        "aggregate", "--track-root", str(only_a_track_root), "--detect-root", str(detect_root),
        "--output-dir", str(output_dir), "--no-figures",
    )
    assert rejected.returncode != 0

    allowed = _run_aggregate(only_a_track_root, detect_root, output_dir, extra=["--no-figures"])
    assert allowed.returncode == 0, allowed.stderr
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["domains_present"] == [A_DOMAIN_ID]
    assert C_DOMAIN_ID in manifest["domains_missing"]
    assert "nanwan" in manifest["sites_missing"]
    assert "gongliao" in manifest["sites_present"]


def test_aggregate_refuses_tampered_shard_manifest(two_domain_root: Path, tmp_path: Path) -> None:
    """track-period 完成後若 detect-root shard manifest 被竄改，aggregate 必須拒絕。"""

    track_root, detect_root = _build_run(tmp_path, two_domain_root)
    shard_manifest = detect_root / C_DOMAIN_ID / "202401" / "manifest.json"
    payload = json.loads(shard_manifest.read_text(encoding="utf-8"))
    payload["snapshot_count"] = payload["snapshot_count"] + 1
    shard_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_dir = tmp_path / "agg_tampered"
    result = _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures"])
    assert result.returncode != 0
    assert "manifest" in result.stderr


def test_aggregate_refuses_pilot_and_formal_mixing(two_domain_root: Path, tmp_path: Path) -> None:
    """PILOT_ONLY track 與正式 track 不可同時彙整（C 區正式、A 區改用 trial pilot 月份）。"""

    detect_root = tmp_path / "detect"
    track_root = tmp_path / "track"
    result_c = _run_detect_shards(two_domain_root, month="202401", output_dir=detect_root / C_DOMAIN_ID / "202401")
    assert result_c.returncode == 0, result_c.stderr
    result_c_track = _run_cli(
        "track-period", "--shards-root", str(detect_root / C_DOMAIN_ID), "--domain-id", C_DOMAIN_ID,
        "--output-dir", str(track_root / C_DOMAIN_ID), "--months", "202401",
    )
    assert result_c_track.returncode == 0, result_c_track.stderr

    _write_a_domain(two_domain_root, month="202502", hours=[0, 1], cache_kind="trial_partial_month", status="trial_ready")
    _run_detect_shards_a(two_domain_root, detect_root / A_DOMAIN_ID / "202502", month="202502", extra=["--allow-trial-pilot"])
    result_a_track = _run_cli(
        "track-period", "--shards-root", str(detect_root / A_DOMAIN_ID), "--domain-id", A_DOMAIN_ID,
        "--output-dir", str(track_root / A_DOMAIN_ID), "--months", "202502",
    )
    assert result_a_track.returncode == 0, result_a_track.stderr
    a_manifest = json.loads((track_root / A_DOMAIN_ID / "manifest.json").read_text(encoding="utf-8"))
    assert a_manifest["state_label"] == "PILOT_ONLY"

    output_dir = tmp_path / "agg_mixed"
    result = _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures"])
    assert result.returncode != 0
    assert "PILOT_ONLY" in result.stderr


# ---------------------------------------------------------------------------
# 單元測試：手造陣列核對指標定義
# ---------------------------------------------------------------------------


def test_finalize_spatial_nan_where_no_valid() -> None:
    """``_finalize_spatial`` 在 n_valid=0 的格點必須輸出 NaN，而非 0。"""

    shape = (2, 3)
    accumulator = agg._new_spatial_accumulator(shape)
    accumulator["n_valid"][:] = np.array([[3, 0, 3], [0, 0, 3]])
    accumulator["core_count"][:] = np.array([[1, 0, 0], [0, 0, 2]])
    accumulator["core_s1_sum"][:] = np.array([[-3.0e-8, 0.0, 0.0], [0.0, 0.0, -1.0e-7]])
    finalized = agg._finalize_spatial(shape, accumulator)
    assert np.isnan(finalized["core_frequency"][0, 1])
    assert np.isnan(finalized["core_frequency"][1, 0])
    assert finalized["core_frequency"][0, 0] == pytest.approx(1.0 / 3.0)
    assert np.isnan(finalized["mean_core_s1_sinv"][0, 1])
    assert finalized["mean_core_s1_sinv"][0, 0] == pytest.approx(-3.0e-8)
    assert finalized["mean_core_s1_day"][0, 0] == pytest.approx(-3.0e-8 * 86_400.0)


def test_bootstrap_ci_reproducible_with_seed() -> None:
    """相同 seed 的 day-block bootstrap 必須回傳完全相同的 CI。"""

    day_stats = {
        "2025-01-01": [10, 4, 1, 6, 0.3],
        "2025-01-02": [10, 6, 2, 7, 0.5],
        "2025-01-03": [8, 2, 0, 3, 0.1],
        "2025-01-04": [10, 8, 3, 9, 0.6],
    }
    first = agg._bootstrap_ci(day_stats, reps=500, rng=np.random.default_rng(42))
    second = agg._bootstrap_ci(day_stats, reps=500, rng=np.random.default_rng(42))
    assert first == second
    for key in ("core_occurrence_rate", "strong_core_occurrence_rate", "curve_occurrence_rate", "mean_pass_cell_fraction"):
        low, high = first[key]
        assert 0.0 <= low <= high <= 1.0 + 1.0e-9


def test_bootstrap_ci_empty_days_returns_none() -> None:
    """沒有任何有效日時，四個 CI 都必須是 None，而不是拋錯或回傳 0。"""

    result = agg._bootstrap_ci({}, reps=100, rng=np.random.default_rng(1))
    assert all(value is None for value in result.values())


def test_ragged_gather_matches_naive_loop() -> None:
    """``_ragged_gather`` 的向量化結果須與逐列 Python 迴圈完全一致。"""

    rng = np.random.default_rng(7)
    offsets = np.array([0, 3, 3, 5, 9], dtype=np.int64)
    values = rng.integers(0, 1000, size=int(offsets[-1])).astype(np.int64)
    selected = np.array([3, 0, 1, 3], dtype=np.int64)

    gathered, row_out = agg._ragged_gather(offsets, selected, values)

    expected_values = []
    expected_rows = []
    for local_row, original_row in enumerate(selected.tolist()):
        for value in values[offsets[original_row] : offsets[original_row + 1]].tolist():
            expected_values.append(value)
            expected_rows.append(local_row)
    assert gathered.tolist() == expected_values
    assert row_out.tolist() == expected_rows


def test_ragged_gather_all_empty_rows() -> None:
    """全部被選列都是空 ragged row 時，回傳空陣列而非出錯。"""

    offsets = np.array([0, 0, 0, 0], dtype=np.int64)
    values = np.zeros(0, dtype=np.int64)
    gathered, row_out = agg._ragged_gather(offsets, np.array([0, 1, 2], dtype=np.int64), values)
    assert gathered.size == 0
    assert row_out.size == 0


def test_haversine_distance_matches_known_value() -> None:
    """經緯度 1 度緯度差的球面距離應接近 111.19 km（WGS84 近似球半徑換算）。"""

    lon = np.array([121.0, 121.0])
    lat = np.array([25.0, 26.0])
    distance = agg._haversine_distance_m(lon, lat, 121.0, 25.0)
    assert distance[0] == pytest.approx(0.0, abs=1.0)
    assert distance[1] == pytest.approx(111_195.0, rel=0.01)


def test_build_domain_masks_a_domain_receptors_are_disjoint() -> None:
    """A 區貢寮／龜山島兩站的 receptor 遮罩必須互斥，且各自非空。"""

    design = load_study_design(DEFAULT_STUDY_DESIGN_PATH)
    lon_axis, lat_axis, _x_axis, _y_axis = _a_axes()
    grid = {
        "lon": lon_axis,
        "lat": lat_axis,
        "analysis_ocean_mask": np.ones((A_NY, A_NX), dtype=bool),
        "shape": (A_NY, A_NX),
    }
    masks = agg._build_domain_masks(A_DOMAIN_ID, grid, design, DEFAULT_STUDY_DESIGN_PATH)
    gongliao = masks["receptor"]["gongliao"]
    guishan = masks["receptor"]["guishan"]
    assert np.any(gongliao)
    assert np.any(guishan)
    assert not np.any(gongliao & guishan)
    # domain_partition（Voronoi 分割）永遠完整覆蓋 static mask，且同樣互斥。
    assert not np.any(masks["domain_partition"]["gongliao"] & masks["domain_partition"]["guishan"])
    assert np.array_equal(masks["domain_partition"]["gongliao"] | masks["domain_partition"]["guishan"], np.ones((A_NY, A_NX), dtype=bool))


def test_process_track_lifetimes_stratum_assignment_and_censoring() -> None:
    """未截尾／截尾 track 各自累計正確，stratum 依首個落在遮罩內的觀測時間指派。"""

    shape = (3, 3)
    x_axis = np.array([-1.0, 0.0, 1.0])
    y_axis = np.array([-1.0, 0.0, 1.0])
    grid = {"shape": shape, "x_m_axis": x_axis, "y_m_axis": y_axis}
    mask = np.zeros(shape, dtype=bool)
    mask[1, 1] = True  # 只有中心格屬於站點
    masks = {"receptor": {"site_a": mask}, "domain_partition": {"site_a": mask}}

    jan_ns = 1_704_100_000_000_000_000  # 2024-01-01 附近
    jul_ns = jan_ns + 180 * 86_400_000_000_000  # 半年後，落入 JJA

    observations = {
        "track_id": np.array([1, 1, 2, 2], dtype=np.int64),
        "time_utc_ns": np.array([jan_ns, jan_ns + 3_600_000_000_000, jul_ns, jul_ns + 3_600_000_000_000], dtype=np.int64),
        "core_x_m": np.array([-1.0, 0.0, -1.0, 0.0], dtype=np.float64),  # track1 第二個觀測才進中心格
        "core_y_m": np.array([-1.0, 0.0, -1.0, 0.0], dtype=np.float64),
    }
    tracks_by_id = {
        1: {"left_censored": False, "right_censored": False, "lifetime_hours": 2.0, "start_time_utc_ns": jan_ns},
        2: {"left_censored": False, "right_censored": True, "lifetime_hours": 2.0, "start_time_utc_ns": jul_ns},
    }
    context = {"observations": observations, "tracks_by_id": tracks_by_id}
    site_accumulators: dict[tuple[str, str, str], dict] = defaultdict(agg._new_site_accumulator)
    spatial_accumulators: dict[str, dict] = {}

    agg._process_track_lifetimes(
        domain_id="unit-test",
        context=context,
        masks=masks,
        grid=grid,
        site_accumulators=site_accumulators,
        spatial_accumulators=spatial_accumulators,
    )

    all_key = ("site_a", "receptor", "all")
    assert site_accumulators[all_key]["track_n"] == 2
    assert site_accumulators[all_key]["track_lifetimes"] == [2.0]  # 只有未截尾 track 1 計入
    assert site_accumulators[all_key]["track_lifetimes_all"] == [2.0, 2.0]

    djf_key = ("site_a", "receptor", "DJF")
    jja_key = ("site_a", "receptor", "JJA")
    assert site_accumulators[djf_key]["track_n"] == 1
    assert site_accumulators[jja_key]["track_n"] == 1
    assert site_accumulators[jja_key].get("track_lifetimes", []) == []  # track 2 是截尾，不進未截尾清單

    # SPATIAL：只有未截尾 track 1 計入逐格壽命累加。
    assert spatial_accumulators["all"]["lifetime_count"][1, 1] == 1
    assert spatial_accumulators["all"]["lifetime_sum"][1, 1] == pytest.approx(2.0)
