"""``build-daily-fields``（潮汐濾波日平均產品）的合成資料測試。

fixture 為三個月（202401–202403）逐時資料，共用同一小型網格：穩態吸引鞍點流場疊加
大振幅 M2（12.42 h）均勻潮流。刻意安排：月份交界重複樣本（後月份值被刻意偏移以驗證
「保留較晚月份」）、1 月某日缺 3 小時（低於 min_hours 應整日省略）、某日缺 2 小時
（仍應保留）、單一小時的單一格點無效（嚴格有效性）。不連線 SERVER，非正式研究結果。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_cli_pipeline import _run_cli, _write_json

from traps.cli import main as cli_main
from traps.ocm import open_month

DOMAIN_ID = "houwan_nmmba_cache_v3"
NY, NX = 21, 31
SPACING_M = 150.0
BASE_NS = 1_704_067_200_000_000_000  # 2024-01-01T00:00:00Z
HOUR_NS = 3_600_000_000_000
TIDE_AMPLITUDE = 0.30
M2_HOURS = 12.42
DUP_OFFSET = 10.0  # 2 月檔重複的 1 月最後一小時，u 被刻意加上此偏移

MONTH_STARTS = {"202401": 0, "202402": 31 * 24, "202403": (31 + 29) * 24}
MONTH_HOURS = {"202401": 31 * 24, "202402": 29 * 24, "202403": 31 * 24}
DROP3_DAY = 9  # 2024-01-10：窗內缺 3 小時 -> 省略
DROP2_DAY = 14  # 2024-01-15：窗內缺 2 小時 -> 保留
BAD_CELL_HOUR = 19 * 24 + 5  # 2024-01-20 05:00：格點 (3, 4) 無效


def _axes() -> tuple[np.ndarray, np.ndarray]:
    axis_x = (np.arange(NX, dtype=np.float64) - (NX - 1) / 2.0) * SPACING_M
    axis_y = (np.arange(NY, dtype=np.float64) - (NY - 1) / 2.0) * SPACING_M
    return axis_x, axis_y


def _steady() -> tuple[np.ndarray, np.ndarray]:
    axis_x, axis_y = _axes()
    x_m, y_m = np.meshgrid(axis_x, axis_y, indexing="xy")
    envelope = np.exp(-(x_m**2 + y_m**2) / 900.0**2)
    return -2.0e-3 * x_m * envelope, -1.0e-3 * y_m * envelope


def _write_grid(root: Path) -> None:
    grid_dir = root / DOMAIN_ID / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    axis_x, axis_y = _axes()
    x_m, y_m = np.meshgrid(axis_x, axis_y, indexing="xy")
    static = np.ones((NY, NX), dtype=bool)
    for name, array in {
        "lon.npy": 120.76 + axis_x / 100_000.0,
        "lat.npy": 21.94 + axis_y / 100_000.0,
        "x_m.npy": x_m,
        "y_m.npy": y_m,
        "cell_area_m2.npy": np.full((NY, NX), SPACING_M**2, dtype=np.float32),
        "mask_static.npy": static,
        "analysis_ocean_mask_synthetic_v1.npy": static,
    }.items():
        np.save(grid_dir / name, array, allow_pickle=False)
    _write_json(grid_dir / "metadata.json", {"cache_schema_version": "3.0.0", "domain": {"domain_id": DOMAIN_ID}, "grid_shape": [NY, NX]})
    _write_json(
        grid_dir / "analysis_land_mask_synthetic_v1.json",
        {"cache_schema_version": "3.0.0", "domain": {"domain_id": DOMAIN_ID}, "arrays": {"shape_lat_lon": [NY, NX]}},
    )


def _hourly_month(root: Path, month: str, hours: list[int], *, dup_offset_hours: frozenset[int] = frozenset()) -> None:
    month_dir = root / DOMAIN_ID / "months" / month
    month_dir.mkdir(parents=True, exist_ok=True)
    u0, v0 = _steady()
    u = np.empty((len(hours), NY, NX), dtype=np.float32)
    v = np.empty_like(u)
    valid = np.ones(u.shape, dtype=bool)
    for k, hour in enumerate(hours):
        tide = TIDE_AMPLITUDE * np.cos(2 * np.pi * hour / M2_HOURS)
        u[k] = u0 + tide + (DUP_OFFSET if hour in dup_offset_hours else 0.0)
        v[k] = v0
        if hour == BAD_CELL_HOUR:
            valid[k, 3, 4] = False
    np.save(month_dir / "time_utc_ns.npy", (BASE_NS + np.asarray(hours, dtype=np.int64) * HOUR_NS).astype(np.int64))
    np.save(month_dir / "u_surface_mps.npy", u)
    np.save(month_dir / "v_surface_mps.npy", v)
    np.save(month_dir / "valid_mask_surface.npy", valid)
    np.save(month_dir / "qc_flags.npy", np.ones(u.shape, dtype=np.uint16))
    _write_json(
        month_dir / "metadata.json",
        {
            "status": "ready",
            "cache_kind": "standard_month",
            "cache_schema_version": "3.0.0",
            "month": month,
            "domain": {
                "domain_id": DOMAIN_ID,
                "name_zh": "南灣合成海域",
                "lon_min": 120.16671,
                "lon_max": 121.62,
                "lat_min": 21.550844,
                "lat_max": 22.449156,
            },
            "units": {"u_surface_mps": "m s-1", "v_surface_mps": "m s-1"},
        },
    )


@pytest.fixture()
def hourly_root(tmp_path: Path) -> Path:
    root = tmp_path / "hourly"
    _write_grid(root)
    jan = [h for h in range(0, 31 * 24)]
    drop3 = {DROP3_DAY * 24 + 3, DROP3_DAY * 24 + 9, DROP3_DAY * 24 + 15}
    drop2 = {DROP2_DAY * 24 + 4, DROP2_DAY * 24 + 20}
    jan = [h for h in jan if h not in drop3 and h not in drop2]
    _hourly_month(root, "202401", jan)
    # 2 月檔多帶 1 月最後一小時（重複 UTC，且 u 被偏移；正式軸須保留此較晚月份的列）。
    feb_start = MONTH_STARTS["202402"]
    _hourly_month(root, "202402", [feb_start - 1] + [feb_start + i for i in range(MONTH_HOURS["202402"])], dup_offset_hours=frozenset({feb_start - 1}))
    mar_start = MONTH_STARTS["202403"]
    _hourly_month(root, "202403", [mar_start + i for i in range(MONTH_HOURS["202403"])])
    return root


def _build(hourly: Path, out: Path, *extra: str) -> int:
    return cli_main(["build-daily-fields", "--surface-root", str(hourly), "--domain-id", DOMAIN_ID, "--output-root", str(out), *extra])


def _tree_hash(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture()
def daily_root(hourly_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    out = tmp_path / "daily"
    assert _build(hourly_root, out) == 0
    capsys.readouterr()
    return out


def _days(root: Path, month: str) -> dict[str, int]:
    times = np.load(root / DOMAIN_ID / "months" / month / "time_utc_ns.npy")
    return {int((t - BASE_NS) // (24 * HOUR_NS)): k for k, t in enumerate(times)}


def test_daily_removes_tide_and_keeps_steady_flow(daily_root: Path) -> None:
    """25 h 平均後 M2 潮流振幅殘留 < 10%，且穩態流場被保留。"""

    u0, v0 = _steady()
    month_dir = daily_root / DOMAIN_ID / "months" / "202402"
    u = np.load(month_dir / "u_surface_mps.npy")
    v = np.load(month_dir / "v_surface_mps.npy")
    assert u.dtype == np.float32 and np.load(month_dir / "valid_mask_surface.npy").all()
    times = np.load(month_dir / "time_utc_ns.npy")
    assert ((times - BASE_NS) % (24 * HOUR_NS) == 12 * HOUR_NS).all()
    residual = np.abs(u - u0[None]).max(axis=(1, 2))
    assert (residual < 0.1 * TIDE_AMPLITUDE).all(), residual.max()
    np.testing.assert_allclose(v, np.broadcast_to(v0, v.shape), atol=1e-5)


def test_day_omitted_below_min_hours_and_kept_at_23(daily_root: Path) -> None:
    jan = _days(daily_root, "202401")
    assert DROP3_DAY not in jan  # 22 < 23 -> 缺口，不補
    assert DROP2_DAY in jan  # 23 >= 23
    metadata = json.loads((daily_root / DOMAIN_ID / "months" / "202401" / "metadata.json").read_text())
    derived = metadata["derived_from"]
    assert {"date": "2024-01-10", "present_hours": 22} in derived["omitted_days"]
    assert derived["present_hours_per_day"]["2024-01-15"] == 23
    assert derived["present_hours_per_day"]["2024-01-05"] == 25
    assert derived["filter"]["method"] == "centered_boxcar_mean"
    assert derived["filter"]["cell_validity"] == "valid_in_all_present_hours"
    assert metadata["cache_kind"] == "derived_daily_mean" and metadata["status"] == "ready"
    assert metadata["arrays"]["u_surface_mps.npy"]["shape"][0] == len(jan)
    assert [r["month"] for r in derived["source_months"]] == ["202401", "202402"]  # 1/31 視窗跨到 2 月


def test_month_crossing_window_and_duplicates_prefer_last(daily_root: Path) -> None:
    """1/31 視窗含 2/1 00:00（來自 2 月檔）與重複的 1/31 23:00（取 2 月檔的偏移值）。"""

    jan = _days(daily_root, "202401")
    assert 30 in jan
    month_dir = daily_root / DOMAIN_ID / "months" / "202401"
    u = np.load(month_dir / "u_surface_mps.npy")[jan[30]]
    u0, _ = _steady()
    # 25 個樣本中恰有 1 個帶 +DUP_OFFSET -> 平均偏移 10/25 = 0.4（潮汐殘留遠小於此）。
    assert float(np.median(u - u0)) == pytest.approx(DUP_OFFSET / 25, abs=0.03)
    metadata = json.loads((month_dir / "metadata.json").read_text())
    assert metadata["derived_from"]["present_hours_per_day"]["2024-01-31"] == 25
    assert [r["month"] for r in metadata["derived_from"]["source_months"]] == ["202401", "202402"]
    # 2 月 1 日不受 1 月 31 日偏移影響（視窗 1/31 12:00 之後 -> 含 1/31 23:00 偏移樣本不在其 00:00–00:00 視窗內）。
    feb = np.load(daily_root / DOMAIN_ID / "months" / "202402" / "u_surface_mps.npy")
    assert float(np.median(feb[0] - u0)) == pytest.approx(0.0, abs=0.03)


def test_strict_cell_validity(daily_root: Path) -> None:
    jan = _days(daily_root, "202401")
    month_dir = daily_root / DOMAIN_ID / "months" / "202401"
    valid = np.load(month_dir / "valid_mask_surface.npy")
    u = np.load(month_dir / "u_surface_mps.npy")
    qc = np.load(month_dir / "qc_flags.npy")
    day = jan[19]
    assert not valid[day, 3, 4] and np.isnan(u[day, 3, 4]) and qc[day, 3, 4] & 2 and not qc[day, 3, 4] & 1
    assert valid[day].sum() == NY * NX - 1
    assert valid[jan[18]].all() and valid[jan[20]].all()
    assert np.isfinite(u[valid]).all()


def test_open_month_accepts_daily_with_ocean_mask(daily_root: Path) -> None:
    month_dir = daily_root / DOMAIN_ID / "months" / "202402"
    mask = daily_root / DOMAIN_ID / "grid" / "analysis_ocean_mask_synthetic_v1.npy"
    surface = open_month(month_dir, expected_domain_id=DOMAIN_ID, analysis_ocean_mask_path=mask)
    assert surface.cache_kind == "derived_daily_mean" and surface.state_label == "READY"
    assert surface.analysis_mask_status == "provided"
    assert surface.snapshot_count == 29
    assert surface.snapshot(0).valid.all()


def test_source_never_modified_and_resume_semantics(hourly_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = _tree_hash(hourly_root)
    out = tmp_path / "daily"
    assert _build(hourly_root, out) == 0
    capsys.readouterr()
    assert _tree_hash(hourly_root) == before

    # 無 --resume 且輸出已存在 -> 拒絕。
    assert cli_main(["build-daily-fields", "--surface-root", str(hourly_root), "--domain-id", DOMAIN_ID, "--output-root", str(out)]) == 2
    capsys.readouterr()

    assert _build(hourly_root, out, "--resume") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["months_skipped_resume"] == ["202401", "202402", "202403"] and summary["months_built"] == []

    # 竄改 2 月 u -> 只重建 2 月。
    target = out / DOMAIN_ID / "months" / "202402" / "u_surface_mps.npy"
    array = np.load(target)
    array[0, 0, 0] += 1.0
    np.save(target, array)
    assert _build(hourly_root, out, "--resume") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["months_built"] == ["202402"]
    assert summary["months_skipped_resume"] == ["202401", "202403"]

    # 不同濾波參數 -> 全部重建。
    assert _build(hourly_root, out, "--resume", "--min-hours", "20") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["months_built"] == ["202401", "202402", "202403"]
    assert _tree_hash(hourly_root) == before


def test_daily_product_runs_detect_and_track_with_24h_step(daily_root: Path, tmp_path: Path) -> None:
    grid = daily_root / DOMAIN_ID / "grid"
    shards = tmp_path / "shards"
    for month in ("202401", "202402", "202403"):
        result = _run_cli(
            "detect-shards", "--surface-root", str(daily_root), "--domain-id", DOMAIN_ID, "--month", month,
            "--ocean-mask", str(grid / "analysis_ocean_mask_synthetic_v1.npy"),
            "--output-dir", str(shards / month), "--min-coverage", "0.95",
        )
        assert result.returncode == 0, result.stderr
    track = tmp_path / "track"
    result = _run_cli(
        "track-period", "--shards-root", str(shards), "--domain-id", DOMAIN_ID, "--months", "202401-202403",
        "--output-dir", str(track), "--nominal-time-step-hours", "24", "--tracking-max-time-gap-hours", "36",
        "--tracking-max-core-distance-m", "10000", "--tracking-max-curve-distance-m", "20000",
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((track / "manifest.json").read_text())
    assert manifest["canonical_axis"]["unique_times"] == 31 + 29 + 31 - 1  # 缺 1 月 10 日
    assert manifest["canonical_axis"]["gap_count"] == 1


def test_rejects_derived_source(daily_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _build(daily_root, tmp_path / "again", "--months", "202402") == 2
    assert "standard" in capsys.readouterr().err
