"""Stage 3 ``aggregate``（逐站獨立、``site_area`` 單一範圍）的合成端到端與單元測試。

兩個合成 domain：A 區（``northeast_taiwan_common_cache_v3``，貢寮＋龜山島兩站，
真實 anchor 經緯度、小型局地 x/y 網格）與沿用 ``test_period.py`` 既有 fixture 的
C 區（``houwan_nmmba_cache_v3``，南灣一站，含三月份、跨月重複 UTC、5 小時缺口與
一個 coverage 不足時刻）。全部只涵蓋數十個時刻的合成流場，不代表 2024--2025
正式 TRAP 統計。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
from traps import aggregate_figures, report_font
from traps.report_font import REQUIRED_FIGURE_GLYPHS, CJKFontSelection
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
    """緩慢沿 y（南北）平移的吸引鞍點，讓核心輪流落在貢寮與龜山島兩個站點分析區。"""

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


@pytest.fixture(scope="module")
def built_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """整個模組只建一次兩個 domain 的 detect-shards + track-period（唯讀使用）。"""

    base = tmp_path_factory.mktemp("aggregate_module")
    root = base / "src"
    _write_shared_grid(root)
    _write_month(root, month="202401", hours=JAN_HOURS)
    _write_month(root, month="202402", hours=FEB_HOURS, low_coverage_hours=frozenset({FEB_LOW_COVERAGE_HOUR}))
    _write_month(root, month="202403", hours=MAR_HOURS)
    _write_a_grid(root)
    _write_a_domain(root)
    track_root, detect_root = _build_run(base, root)
    return root, track_root, detect_root


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
        "track-period", "--shards-root", str(detect_root / C_DOMAIN_ID), "--domain-id", C_DOMAIN_ID,
        "--output-dir", str(track_root / C_DOMAIN_ID), "--months", "202401-202403",
    )
    assert result_c.returncode == 0, result_c.stderr
    result_a = _run_cli(
        "track-period", "--shards-root", str(detect_root / A_DOMAIN_ID), "--domain-id", A_DOMAIN_ID,
        "--output-dir", str(track_root / A_DOMAIN_ID), "--months", "202501",
    )
    assert result_a.returncode == 0, result_a.stderr
    return track_root, detect_root


def _run_aggregate(
    track_root: Path, detect_root: Path, output_dir: Path, *, extra: list[str] | None = None
) -> "subprocess.CompletedProcess[str]":
    """以子行程執行 CLI（只用於 --no-figures 路徑；圖面測試走 in-process 並替換字型）。"""

    arguments = [
        "aggregate", "--track-root", str(track_root), "--detect-root", str(detect_root), "--output-dir", str(output_dir),
        "--seed", "20260928", "--bootstrap-reps", "200", "--allow-partial-domains",
    ]
    if extra:
        arguments.extend(extra)
    return _run_cli(*arguments)


def _namespace(track_root: Path, detect_root: Path, output_dir: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "track_root": track_root, "detect_root": detect_root, "output_dir": output_dir, "param_id": None,
        "study_design": None, "strong_quantile": 0.10, "hotspot_quantile": 0.90, "bootstrap_reps": 200,
        "seed": 20260928, "coastline_geojson": None, "no_figures": False, "resume": False, "allow_partial_domains": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


FAKE_FONT = CJKFontSelection(
    family="Noto Sans CJK TC", filename="NotoSansCJK-Regular.ttc", file_sha256="0" * 64, required_glyph_count=len(REQUIRED_FIGURE_GLYPHS)
)


@pytest.fixture()
def fake_font(monkeypatch: pytest.MonkeyPatch) -> CJKFontSelection:
    """替換字型解析器：測試環境不一定有 CJK 字型，渲染時 Matplotlib 會回退到 DejaVu（僅警告）。"""

    monkeypatch.setattr(report_font, "resolve_cjk_font", lambda: FAKE_FONT)
    return FAKE_FONT


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def _expected_c_all(track_root: Path, detect_root: Path) -> tuple[int, int]:
    """獨立於 aggregate 程式碼，由 shard 產物重算南灣 site_area/all 的有效快照數與核心出現分子。

    C 區單站的站點分析區＝完整 analysis_ocean_mask；合成核心都在分析海域內，因此
    「至少一個合格 detection」等價於 shard 的 ``qualified_count > 0``。
    """

    with np.load(track_root / C_DOMAIN_ID / "canonical_snapshots.npz") as archive:
        month, snapshot_index, coverage_pass = archive["month"], archive["snapshot_index"], archive["coverage_pass"]
    n_valid = 0
    n_occurrence = 0
    for month_code, index, is_pass in zip(month.tolist(), snapshot_index.tolist(), coverage_pass.tolist(), strict=True):
        if not is_pass:
            continue
        n_valid += 1
        with np.load(detect_root / C_DOMAIN_ID / f"{month_code:06d}" / "snapshots.npz") as shard:
            position = shard["snapshot_index"].tolist().index(index)
            if int(shard["qualified_count"][position]) > 0:
                n_occurrence += 1
    return n_valid, n_occurrence


def _expected_c_threshold(track_root: Path, detect_root: Path, quantile: float) -> float:
    """獨立重算南灣：canonical coverage_pass 快照、分析區內合格核心 s1 的分位數。"""

    with np.load(track_root / C_DOMAIN_ID / "canonical_snapshots.npz") as archive:
        month, snapshot_index, coverage_pass = archive["month"], archive["snapshot_index"], archive["coverage_pass"]
    values: list[np.ndarray] = []
    for month_code in np.unique(month):
        valid = snapshot_index[(month == month_code) & coverage_pass]
        month_dir = detect_root / C_DOMAIN_ID / f"{int(month_code):06d}"
        with np.load(month_dir / "detections.npz") as det, np.load(month_dir / "grid_reference.npz") as grid:
            mask = grid["analysis_ocean_mask"][det["core_row"], det["core_col"]]
            keep = det["is_qualified"] & np.isin(det["snapshot_index"], valid) & mask
            values.append(det["core_s1_sinv"][keep])
    return float(np.quantile(np.concatenate(values), quantile))


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
def test_aggregate_end_to_end_per_site_outputs(built_run: tuple[Path, Path, Path], tmp_path: Path, fake_font: CJKFontSelection) -> None:
    """兩區彙整：逐站分母/分子、門檻、per-site 圖組、spatial npz、manifest 與字型紀錄。"""

    _root, track_root, detect_root = built_run
    output_dir = tmp_path / "aggregate_out"
    assert agg.run_aggregate(_namespace(track_root, detect_root, output_dir)) == 0

    for name in ("site_statistics.csv", "site_statistics.json", "manifest.json", "manifest.sha256", "_SUCCESS"):
        assert (output_dir / name).is_file(), name
    sites = ("nanwan", "gongliao", "guishan")
    for site_id in sites:
        for kind in ("spatial", "hotspots", "seasonal", "lifetime", "timeseries"):
            assert (output_dir / "figures" / site_id / f"site_{site_id}_{kind}.png").is_file(), (site_id, kind)
        for stratum in ("all", "DJF", "MAM", "JJA", "SON"):
            assert (output_dir / "spatial" / site_id / f"spatial_{stratum}.npz").is_file()
    # 舊版跨站圖與 domain 圖必須消失
    for legacy in ("five_site_summary.png", "seasonal_occurrence.png", "lifetime_distribution.png", f"spatial_{C_DOMAIN_ID}.png"):
        assert not (output_dir / legacy).exists()
    assert not (output_dir / "spatial" / C_DOMAIN_ID).exists()

    manifest = _manifest(output_dir)
    assert manifest["schema_version"] == "traps.period.aggregate.v2"
    assert manifest["scope"] == "site_area"
    assert sorted(manifest["sites_present"]) == sorted(sites)
    assert manifest["font"] == {
        "family": "Noto Sans CJK TC", "filename": "NotoSansCJK-Regular.ttc", "file_sha256": "0" * 64,
        "required_glyph_count": len(REQUIRED_FIGURE_GLYPHS),
    }
    assert "strong_core_threshold_sinv" not in manifest and "pooled_core_count_for_threshold" not in manifest
    thresholds = manifest["strong_core_thresholds"]
    assert set(thresholds) == set(sites)
    for value in thresholds.values():
        assert value["strong_core_threshold_day"] == pytest.approx(value["strong_core_threshold_sinv"] * 86_400.0)
    assert thresholds["nanwan"]["strong_core_threshold_sinv"] == pytest.approx(_expected_c_threshold(track_root, detect_root, 0.10))
    assert set(manifest["colour_limits"]) == set(sites)

    rows = _read_csv_rows(output_dir / "site_statistics.csv")
    assert {r["scope"] for r in rows} == {"site_area"}
    row = next(r for r in rows if r["site_id"] == "nanwan" and r["stratum"] == "all")
    expected_n_valid, expected_occurrence = _expected_c_all(track_root, detect_root)
    assert int(row["n_valid_snapshots"]) == expected_n_valid == 10  # 13 列 - 2 重複 - 1 coverage 不足
    assert int(row["core_occurrence_numerator"]) == expected_occurrence
    assert float(row["strong_core_threshold_sinv"]) == pytest.approx(thresholds["nanwan"]["strong_core_threshold_sinv"])
    for r in rows:
        for key in ("core_occurrence_rate", "strong_core_occurrence_rate", "curve_occurrence_rate", "mean_pass_cell_fraction"):
            if r[key]:
                assert 0.0 <= float(r[key]) <= 1.0 + 1.0e-9
        assert int(r["strong_core_occurrence_numerator"]) <= int(r["core_occurrence_numerator"])
        assert int(r["track_n_uncensored"]) <= int(r["track_n_tracks"])
        assert int(r["hotspot_cell_count"]) >= 0
        if r["hotspot_strong_core_fraction"]:
            assert 0.0 <= float(r["hotspot_strong_core_fraction"]) <= 1.0

    # 逐站 npz：站點遮罩、熱點遮罩；站外為 NaN；A 區兩站遮罩互斥且合起來涵蓋整個分析海域
    gongliao = _load_npz(output_dir / "spatial" / "gongliao" / "spatial_all.npz")
    guishan = _load_npz(output_dir / "spatial" / "guishan" / "spatial_all.npz")
    assert not np.any(gongliao["site_mask"] & guishan["site_mask"])
    assert np.all(gongliao["site_mask"] | guishan["site_mask"])
    for archive in (gongliao, guishan):
        assert {"site_mask", "hotspot_curve_mask", "hotspot_core_mask", "curve_pass_frequency", "n_valid"} <= set(archive)
        assert np.all(np.isnan(archive["curve_pass_frequency"][~archive["site_mask"]]))
        assert not np.any(archive["hotspot_curve_mask"] & ~archive["site_mask"])
    # 逐站色階＝該站自己 2–98 百分位（不與他站共用）
    for site_id, archive in (("gongliao", gongliao), ("guishan", guishan)):
        finite = archive["curve_pass_frequency"][np.isfinite(archive["curve_pass_frequency"])]
        low, high = np.percentile(finite, [2.0, 98.0])
        got = manifest["colour_limits"][site_id]["curve_pass_frequency"]
        if high > low:
            assert got == pytest.approx([low, high])


def test_aggregate_thresholds_independent_across_sites(built_run: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """只留單一 domain 重新彙整，其站點門檻與統計完全不變＝不受其他站核心影響。"""

    _root, track_root, detect_root = built_run
    both = tmp_path / "both"
    assert _run_aggregate(track_root, detect_root, both, extra=["--no-figures"]).returncode == 0
    both_manifest = _manifest(both)
    for only_domain, site_ids in ((C_DOMAIN_ID, ["nanwan"]), (A_DOMAIN_ID, ["gongliao", "guishan"])):
        partial_track = tmp_path / f"track_only_{only_domain}"
        partial_track.mkdir()
        shutil.copytree(track_root / only_domain, partial_track / only_domain)
        out = tmp_path / f"only_{only_domain}"
        assert _run_aggregate(partial_track, detect_root, out, extra=["--no-figures"]).returncode == 0
        partial_manifest = _manifest(out)
        for site_id in site_ids:
            assert partial_manifest["strong_core_thresholds"][site_id] == both_manifest["strong_core_thresholds"][site_id]
        both_rows = {(r["site_id"], r["stratum"]): r for r in _read_csv_rows(both / "site_statistics.csv")}
        for r in _read_csv_rows(out / "site_statistics.csv"):
            assert r == both_rows[(r["site_id"], r["stratum"])]


def test_thresholds_from_core_values_are_per_site() -> None:
    """改變另一個站的核心 s1，本站門檻位元相同；門檻是本站自己的分位數。"""

    site_a = np.array([-5.0e-5, -4.0e-5, -3.0e-5, -2.0e-5, -1.0e-5])
    first = agg._thresholds_from_core_values({"a": site_a, "b": np.array([-1.0e-4, -2.0e-4])}, 0.25)
    second = agg._thresholds_from_core_values({"a": site_a, "b": np.array([-9.0e-9, 1.0e-3, 5.0e-2])}, 0.25)
    assert first["a"] == second["a"]
    assert first["a"]["threshold_sinv"] == pytest.approx(np.quantile(site_a, 0.25))
    assert first["a"]["threshold_day"] == pytest.approx(first["a"]["threshold_sinv"] * 86_400.0)
    assert first["a"]["core_count"] == 5
    assert first["b"] != second["b"]
    empty = agg._thresholds_from_core_values({"c": np.zeros(0)}, 0.1)
    assert empty["c"] == {"threshold_sinv": None, "threshold_day": None, "core_count": 0}


def test_no_receptor_scope_anywhere(built_run: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """程式碼、CLI、輸出表與 manifest 都不得出現 receptor scope 或 12.5 km 接收區半徑設定。"""

    _root, track_root, detect_root = built_run
    for name in ("aggregate.py", "aggregate_figures.py", "report_font.py"):
        text = (PROJECT_ROOT / "src" / "traps" / name).read_text(encoding="utf-8").lower()
        assert "receptor" not in text, name
        assert "12500" not in text and "radius_m" not in text, name
    help_text = _run_cli("aggregate", "--help").stdout.lower()
    assert "receptor" not in help_text and "radius" not in help_text

    output_dir = tmp_path / "no_receptor"
    assert _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures"]).returncode == 0
    for name in ("site_statistics.csv", "site_statistics.json", "manifest.json"):
        assert "receptor" not in (output_dir / name).read_text(encoding="utf-8").lower(), name
    assert {r["scope"] for r in _read_csv_rows(output_dir / "site_statistics.csv")} == {"site_area"}
    for npz_path in (output_dir / "spatial").rglob("*.npz"):
        assert not any("receptor" in key or "domain_partition" in key for key in _load_npz(npz_path))


def test_aggregate_bootstrap_reproducible_with_seed(built_run: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """相同 --seed 兩次執行的 bootstrap CI 必須逐位元相同。"""

    _root, track_root, detect_root = built_run
    out_first, out_second = tmp_path / "agg_first", tmp_path / "agg_second"
    assert _run_aggregate(track_root, detect_root, out_first, extra=["--no-figures"]).returncode == 0
    assert _run_aggregate(track_root, detect_root, out_second, extra=["--no-figures"]).returncode == 0
    first = json.loads((out_first / "site_statistics.json").read_text(encoding="utf-8"))["records"]
    second = json.loads((out_second / "site_statistics.json").read_text(encoding="utf-8"))["records"]
    assert first == second


def test_aggregate_no_figures_skips_pngs_and_font(built_run: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """``--no-figures`` 只輸出統計表與 manifest，不產生 PNG，也不需要 CJK 字型。"""

    _root, track_root, detect_root = built_run
    output_dir = tmp_path / "agg_no_fig"
    result = _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures"])
    assert result.returncode == 0, result.stderr
    assert not list(output_dir.rglob("*.png"))
    manifest = _manifest(output_dir)
    assert manifest["figures_enabled"] is False
    assert manifest["font"] is None


def test_aggregate_fails_closed_without_cjk_font(
    built_run: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """圖面開啟但找不到合格 CJK 字型：寫檔前就以清楚的 ValueError 停止；--no-figures 不受影響。"""

    _root, track_root, detect_root = built_run

    def _no_font() -> CJKFontSelection:
        raise ValueError("找不到涵蓋 TRAP 圖面必要字元的合格繁體中文字型")

    monkeypatch.setattr(report_font, "resolve_cjk_font", _no_font)
    output_dir = tmp_path / "no_font"
    with pytest.raises(ValueError, match="繁體中文字型"):
        agg.run_aggregate(_namespace(track_root, detect_root, output_dir))
    assert not any(output_dir.rglob("*")) if output_dir.exists() else True

    assert agg.run_aggregate(_namespace(track_root, detect_root, tmp_path / "ok", no_figures=True)) == 0


def test_aggregate_resume_is_noop_then_redo_after_tamper(built_run: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """--resume 對未變更輸出略過重算，對被竄改輸出重做。"""

    _root, track_root, detect_root = built_run
    output_dir = tmp_path / "agg_resume"
    assert _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures"]).returncode == 0
    assert _run_aggregate(track_root, detect_root, output_dir, extra=["--no-figures"]).returncode != 0

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


def test_aggregate_allow_partial_domains(built_run: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """未提供 --allow-partial-domains 時缺少 domain 直接拒絕；提供後只報告現有站點。"""

    _root, track_root, detect_root = built_run
    only_a = tmp_path / "track_only_a"
    only_a.mkdir()
    shutil.copytree(track_root / A_DOMAIN_ID, only_a / A_DOMAIN_ID)
    output_dir = tmp_path / "agg_partial"
    rejected = _run_cli(
        "aggregate", "--track-root", str(only_a), "--detect-root", str(detect_root), "--output-dir", str(output_dir), "--no-figures"
    )
    assert rejected.returncode != 0
    assert _run_aggregate(only_a, detect_root, output_dir, extra=["--no-figures"]).returncode == 0
    manifest = _manifest(output_dir)
    assert manifest["domains_present"] == [A_DOMAIN_ID]
    assert C_DOMAIN_ID in manifest["domains_missing"]
    assert "nanwan" in manifest["sites_missing"]
    assert "gongliao" in manifest["sites_present"]


def test_aggregate_refuses_tampered_shard_manifest(built_run: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """track-period 完成後若 detect-root shard manifest 被竄改，aggregate 必須拒絕。"""

    _root, track_root, detect_root = built_run
    copied_detect = tmp_path / "detect_copy"
    shutil.copytree(detect_root, copied_detect)
    shard_manifest = copied_detect / C_DOMAIN_ID / "202401" / "manifest.json"
    payload = json.loads(shard_manifest.read_text(encoding="utf-8"))
    payload["snapshot_count"] = payload["snapshot_count"] + 1
    shard_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = _run_aggregate(track_root, copied_detect, tmp_path / "agg_tampered", extra=["--no-figures"])
    assert result.returncode != 0
    assert "manifest" in result.stderr


def test_aggregate_refuses_pilot_and_formal_mixing(built_run: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """PILOT_ONLY track 與正式 track 不可同時彙整（C 區正式、A 區改用 trial pilot 月份）。"""

    root, _track_root, _detect_root = built_run
    detect_root = tmp_path / "detect"
    track_root = tmp_path / "track"
    assert _run_detect_shards(root, month="202401", output_dir=detect_root / C_DOMAIN_ID / "202401").returncode == 0
    result_c_track = _run_cli(
        "track-period", "--shards-root", str(detect_root / C_DOMAIN_ID), "--domain-id", C_DOMAIN_ID,
        "--output-dir", str(track_root / C_DOMAIN_ID), "--months", "202401",
    )
    assert result_c_track.returncode == 0, result_c_track.stderr

    _write_a_domain(root, month="202502", hours=[0, 1], cache_kind="trial_partial_month", status="trial_ready")
    _run_detect_shards_a(root, detect_root / A_DOMAIN_ID / "202502", month="202502", extra=["--allow-trial-pilot"])
    result_a_track = _run_cli(
        "track-period", "--shards-root", str(detect_root / A_DOMAIN_ID), "--domain-id", A_DOMAIN_ID,
        "--output-dir", str(track_root / A_DOMAIN_ID), "--months", "202502",
    )
    assert result_a_track.returncode == 0, result_a_track.stderr
    assert json.loads((track_root / A_DOMAIN_ID / "manifest.json").read_text(encoding="utf-8"))["state_label"] == "PILOT_ONLY"
    result = _run_aggregate(track_root, detect_root, tmp_path / "agg_mixed", extra=["--no-figures"])
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

    assert all(value is None for value in agg._bootstrap_ci({}, reps=100, rng=np.random.default_rng(1)).values())


def test_ragged_gather_matches_naive_loop() -> None:
    """``_ragged_gather`` 的向量化結果須與逐列 Python 迴圈完全一致。"""

    rng = np.random.default_rng(7)
    offsets = np.array([0, 3, 3, 5, 9], dtype=np.int64)
    values = rng.integers(0, 1000, size=int(offsets[-1])).astype(np.int64)
    selected = np.array([3, 0, 1, 3], dtype=np.int64)
    gathered, row_out = agg._ragged_gather(offsets, selected, values)
    expected_values, expected_rows = [], []
    for local_row, original_row in enumerate(selected.tolist()):
        for value in values[offsets[original_row] : offsets[original_row + 1]].tolist():
            expected_values.append(value)
            expected_rows.append(local_row)
    assert gathered.tolist() == expected_values
    assert row_out.tolist() == expected_rows


def test_ragged_gather_all_empty_rows() -> None:
    """全部被選列都是空 ragged row 時，回傳空陣列而非出錯。"""

    offsets = np.array([0, 0, 0, 0], dtype=np.int64)
    gathered, row_out = agg._ragged_gather(offsets, np.array([0, 1, 2], dtype=np.int64), np.zeros(0, dtype=np.int64))
    assert gathered.size == 0 and row_out.size == 0


def test_build_domain_masks_follow_site_masks_partition() -> None:
    """A 區兩站依 site_masks 互斥分割整個分析海域；B/C/D 單站取得完整分析海域。"""

    design = load_study_design(DEFAULT_STUDY_DESIGN_PATH)
    lon_axis, lat_axis, _x_axis, _y_axis = _a_axes()
    ocean = np.ones((A_NY, A_NX), dtype=bool)
    ocean[:3, :] = False  # 一列陸地
    grid = {"lon": lon_axis, "lat": lat_axis, "analysis_ocean_mask": ocean, "shape": (A_NY, A_NX)}
    masks = agg._build_domain_masks(A_DOMAIN_ID, grid, design, DEFAULT_STUDY_DESIGN_PATH)
    assert set(masks) == {"gongliao", "guishan"}
    assert np.any(masks["gongliao"]) and np.any(masks["guishan"])
    assert not np.any(masks["gongliao"] & masks["guishan"])
    assert np.array_equal(masks["gongliao"] | masks["guishan"], ocean)

    single = agg._build_domain_masks(C_DOMAIN_ID, grid, design, DEFAULT_STUDY_DESIGN_PATH)
    assert set(single) == {"nanwan"}
    assert np.array_equal(single["nanwan"], ocean)


def test_top_quantile_mask_and_hotspot_metrics_on_hand_made_grid() -> None:
    """4×5 手造格網（站內 16 格）：熱點＝站內 curve_pass_frequency 最高 25%，逐項核對指標。"""

    shape = (4, 5)
    site_mask = np.zeros(shape, dtype=bool)
    site_mask[:, :4] = True  # 最後一欄不屬於站點，故意放最大值驗證不被 pooled
    accumulator = agg._new_spatial_accumulator(shape)
    accumulator["n_valid"][:] = 10
    counts = np.arange(20, dtype=np.int64).reshape(shape)  # curve_pass_count 0..19
    accumulator["curve_pass_count"][:] = counts
    accumulator["curve_pass_count"][:, 4] = 10  # 站外＝頻率 1.0
    accumulator["core_count"][:] = 2
    accumulator["core_s1_sum"][:] = -2.0e-5  # 每格 2 個核心，總 s1 -2e-5 → 平均 -1e-5 s^-1
    accumulator["strong_core_count"][:] = 0
    accumulator["strong_core_count"][3, 2] = 1  # 一個強核心在最高格
    accumulator["strong_core_count"][0, 0] = 1  # 一個不在熱點
    finalized = agg._finalize_spatial(shape, accumulator)
    cell_area = np.full(shape, 4.0e6)  # 每格 4 km²

    hot = agg._site_hotspots(accumulator, finalized, site_mask, cell_area, 0.75)
    freq = finalized["curve_pass_frequency"]
    inside_values = freq[site_mask]
    expected_threshold = float(np.quantile(inside_values, 0.75))
    expected_mask = site_mask & (freq >= expected_threshold)
    assert np.array_equal(hot["curve_mask"], expected_mask)
    assert not np.any(hot["curve_mask"][:, 4])  # 站外不會被標成熱點
    metrics = hot["metrics"]
    assert metrics["hotspot_threshold_curve_pass_frequency"] == pytest.approx(expected_threshold)
    n_hot = int(expected_mask.sum())
    assert metrics["hotspot_cell_count"] == n_hot == 4
    assert metrics["hotspot_area_km2"] == pytest.approx(4 * 4.0)
    assert metrics["hotspot_mean_curve_pass_frequency"] == pytest.approx(float(freq[expected_mask].mean()))
    assert metrics["hotspot_mean_core_s1_day"] == pytest.approx(-1.0e-5 * 86_400.0)
    assert metrics["hotspot_strong_core_fraction"] == pytest.approx(0.5)
    assert metrics["hotspot_quantile"] == 0.75

    # 零場：不把每個格點都標成熱點；站內無資料：None
    zero = agg._new_spatial_accumulator(shape)
    zero["n_valid"][:] = 5
    zero_hot = agg._site_hotspots(zero, agg._finalize_spatial(shape, zero), site_mask, cell_area, 0.9)
    assert zero_hot["metrics"]["hotspot_cell_count"] == 0
    assert zero_hot["metrics"]["hotspot_area_km2"] == 0.0
    assert zero_hot["metrics"]["hotspot_mean_curve_pass_frequency"] is None
    assert zero_hot["metrics"]["hotspot_strong_core_fraction"] is None
    empty = agg._new_spatial_accumulator(shape)
    empty_hot = agg._site_hotspots(empty, agg._finalize_spatial(shape, empty), site_mask, cell_area, 0.9)
    assert empty_hot["metrics"]["hotspot_threshold_curve_pass_frequency"] is None


def test_site_colour_limits_are_computed_per_site() -> None:
    """兩個站各自 2–98 百分位；一站的極端值不會改變另一站的色階。"""

    rng = np.random.default_rng(3)

    def _site_fields(scale: float) -> dict[str, np.ndarray]:
        base = rng.random((6, 7)) * scale
        fields = {key: base.copy() for key in agg.COLOUR_FIELDS}
        fields["mean_core_s1_day"][0, 0] = np.nan
        return fields

    small, large = _site_fields(1.0), _site_fields(100.0)
    limits_small = agg._compute_site_colour_limits(small)
    limits_large = agg._compute_site_colour_limits(large)
    for key in agg.COLOUR_FIELDS:
        finite = small[key][np.isfinite(small[key])]
        assert limits_small[key] == pytest.approx(list(np.percentile(finite, [2.0, 98.0])))
        assert limits_large[key][1] > 10.0 * limits_small[key][1]
    nan_only = agg._compute_site_colour_limits({key: np.full((2, 2), np.nan) for key in agg.COLOUR_FIELDS})
    assert nan_only["curve_pass_frequency"] == [0.0, 1.0]


def test_age_profile_normalises_lifetime_to_unit_interval() -> None:
    """不同長度、不等間距的 track 都映射到 0–1 年齡；端點吻合，中間為線性內插。"""

    hour = 3_600_000_000_000
    times = np.array([0, 2, 3, 10], dtype=np.int64) * hour
    s1 = np.array([-1.0e-5, -3.0e-5, -4.0e-5, -1.1e-4])
    profile = agg._age_profile_sinv(times, s1, n_points=6)
    assert profile is not None and profile.shape == (6,)
    assert profile[0] == pytest.approx(s1[0])
    assert profile[-1] == pytest.approx(s1[-1])
    ages = np.linspace(0.0, 1.0, 6)
    assert profile == pytest.approx(np.interp(ages, (times - times[0]) / (times[-1] - times[0]), s1))
    # 時間軸縮放 3 倍，剖面不變（只取決於正規化年齡）
    scaled = agg._age_profile_sinv(times * 3, s1, n_points=6)
    assert scaled == pytest.approx(profile)
    assert agg._age_profile_sinv(times[:1], s1[:1]) is None
    assert agg._age_profile_sinv(np.array([5, 5], dtype=np.int64), np.array([1.0, 2.0])) is None


def test_process_track_lifetimes_per_site_strata_censoring_and_evolution() -> None:
    """未截尾／截尾 track 各自累計；stratum 依首個落在該站分析區的觀測；≥12 h 未截尾 track 進演化曲線。"""

    shape = (3, 3)
    grid = {"shape": shape, "x_m_axis": np.array([-1.0, 0.0, 1.0]), "y_m_axis": np.array([-1.0, 0.0, 1.0])}
    site_a = np.zeros(shape, dtype=bool)
    site_a[1, 1] = True  # 只有中心格屬於站點 a
    site_b = np.zeros(shape, dtype=bool)
    site_b[0, 0] = True  # 站點 b 只有左下角格

    hour = 3_600_000_000_000
    jan_ns = 1_704_100_000_000_000_000
    jul_ns = jan_ns + 180 * 24 * hour
    observations = {
        "track_id": np.array([1, 1, 1, 2, 2], dtype=np.int64),
        "time_utc_ns": np.array([jan_ns, jan_ns + 6 * hour, jan_ns + 12 * hour, jul_ns, jul_ns + hour], dtype=np.int64),
        "core_x_m": np.array([-1.0, 0.0, 0.0, -1.0, 0.0]),
        "core_y_m": np.array([-1.0, 0.0, 0.0, -1.0, 0.0]),
        "core_s1_sinv": np.array([-1.0e-5, -2.0e-5, -3.0e-5, -9.0e-5, -9.0e-5]),
    }
    tracks_by_id = {
        1: {"left_censored": False, "right_censored": False, "lifetime_hours": 13.0, "start_time_utc_ns": jan_ns},
        2: {"left_censored": False, "right_censored": True, "lifetime_hours": 2.0, "start_time_utc_ns": jul_ns},
    }
    site_accumulators: dict[tuple[str, str], dict] = defaultdict(agg._new_site_accumulator)
    spatial_accumulators: dict[str, dict] = {"site_a": {}, "site_b": {}}
    agg._process_track_lifetimes(
        context={"observations": observations, "tracks_by_id": tracks_by_id},
        site_masks_2d={"site_a": site_a, "site_b": site_b},
        grid=grid,
        site_accumulators=site_accumulators,
        spatial_accumulators=spatial_accumulators,
    )
    a_all = site_accumulators[("site_a", "all")]
    assert a_all["track_n"] == 2
    assert a_all["track_lifetimes"] == [13.0]
    assert a_all["track_lifetimes_all"] == [13.0, 2.0]
    assert site_accumulators[("site_a", "DJF")]["track_n"] == 1
    assert site_accumulators[("site_a", "JJA")]["track_n"] == 1
    assert site_accumulators[("site_a", "JJA")]["track_lifetimes"] == []
    # 演化曲線：只有 track 1（未截尾且 ≥ 12 h）；s1 線性 -1e-5 → -3e-5（第 2 個觀測在 0.5）
    assert a_all["age_track_count"] == 1
    assert a_all["age_sum_sinv"][0] == pytest.approx(-1.0e-5)
    assert a_all["age_sum_sinv"][-1] == pytest.approx(-3.0e-5)
    assert a_all["age_sum_sinv"][agg.LIFETIME_EVOLUTION_AGE_POINTS // 2] == pytest.approx(-2.0e-5)
    # 站點 b 只收到 track 1（第一個觀測就在其分析區內），且不包含站點 a 的 track 2
    b_all = site_accumulators[("site_b", "all")]
    assert b_all["track_n"] == 2 and b_all["track_lifetimes"] == [13.0]
    # 逐格壽命：track 1 只在該站分析區內的造訪格計一次；截尾 track 不計
    assert spatial_accumulators["site_a"]["DJF"]["lifetime_count"][1, 1] == 1
    assert spatial_accumulators["site_a"]["DJF"]["lifetime_sum"][1, 1] == pytest.approx(13.0)
    assert spatial_accumulators["site_a"]["DJF"]["lifetime_count"][0, 0] == 0
    assert "JJA" not in spatial_accumulators["site_a"]
    assert spatial_accumulators["site_b"]["DJF"]["lifetime_count"][0, 0] == 1


def test_sum_spatial_accumulators_gives_all_stratum() -> None:
    """``all`` 分層是季節累加器逐格相加。"""

    shape = (2, 2)
    first, second = agg._new_spatial_accumulator(shape), agg._new_spatial_accumulator(shape)
    first["n_valid"][:] = 3
    second["n_valid"][:] = 4
    first["core_s1_sum"][0, 0] = -1.0
    second["core_s1_sum"][0, 0] = -2.0
    total = agg._sum_spatial_accumulators(shape, [first, second])
    assert total["n_valid"].tolist() == [[7, 7], [7, 7]]
    assert total["core_s1_sum"][0, 0] == -3.0


# ---------------------------------------------------------------------------
# 圖面（合成資料，直接呼叫 renderer）
# ---------------------------------------------------------------------------


def _synthetic_figure_data(*, empty: bool = False) -> aggregate_figures.SiteFigureData:
    ny, nx = 18, 24
    lon = np.linspace(121.9, 122.1, nx)
    lat = np.linspace(24.8, 25.1, ny)
    yy, xx = np.mgrid[0:ny, 0:nx]
    mask = np.ones((ny, nx), dtype=bool)
    mask[:, nx // 2 + 6 :] = False
    rng = np.random.default_rng(11)

    def _fields(scale: float) -> dict[str, np.ndarray]:
        blob = np.exp(-((xx - 8) ** 2 + (yy - 9) ** 2) / 30.0) * scale
        values = {
            "curve_pass_frequency": blob,
            "core_frequency": blob * 0.4,
            "mean_core_s1_day": -blob * 2.0,
            "mean_lifetime_hours": blob * 30.0 + rng.random((ny, nx)),
        }
        return {key: np.where(mask, np.full_like(value, np.nan) if empty else value, np.nan) for key, value in values.items()}

    all_fields = _fields(0.5)
    hot_curve = mask & (all_fields["curve_pass_frequency"] >= 0.3)
    hot_core = mask & (all_fields["core_frequency"] >= 0.12)
    record = {
        "n_valid_snapshots": 0 if empty else 240,
        "strong_core_occurrence_rate": None if empty else 0.2,
        "strong_core_occurrence_rate_ci95": None if empty else (0.1, 0.3),
        "mean_pass_cell_fraction": None if empty else 0.05,
        "mean_pass_cell_fraction_ci95": None if empty else (0.03, 0.08),
    }
    days = {f"2024-01-{d:02d}": [24, 10, 3, 12, 0.6] for d in range(1, 11)}
    return aggregate_figures.SiteFigureData(
        site_id="gongliao",
        site_name_zh="貢寮",
        lon=lon,
        lat=lat,
        site_mask=mask,
        cell_area_m2=np.full((ny, nx), 1.0e6),
        spatial_all=all_fields,
        spatial_seasons={season: _fields(0.3 + 0.1 * i) for i, season in enumerate(aggregate_figures.SEASONS)},
        hotspot_curve_mask=np.zeros_like(mask) if empty else hot_curve,
        hotspot_core_mask=np.zeros_like(mask) if empty else hot_core,
        hotspot_metrics={
            "hotspot_quantile": 0.9, "hotspot_threshold_curve_pass_frequency": None if empty else 0.3,
            "hotspot_cell_count": 0 if empty else int(hot_curve.sum()), "hotspot_area_km2": 0.0 if empty else float(hot_curve.sum()),
        },
        colour_limits={"curve_pass_frequency": [0.0, 0.5], "core_frequency": [0.0, 0.2], "mean_core_s1_day": [-1.0, 0.0], "mean_lifetime_hours": [0.0, 20.0]},
        records={"all": record, **{season: record for season in aggregate_figures.SEASONS}},
        lifetimes_uncensored=[] if empty else [3.0, 5.0, 8.0, 14.0, 30.0],
        lifetimes_all=[] if empty else [3.0, 5.0, 8.0, 14.0, 30.0, 60.0],
        evolution_mean_day=[] if empty else list(np.linspace(-1.0, -3.0, 21)),
        evolution_track_count=0 if empty else 4,
        day_stats={} if empty else days,
        anchor_lonlat=(122.0, 24.95),
        n_valid_snapshots=0 if empty else 240,
    )


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("empty", [False, True])
def test_render_site_figures_produces_five_files_and_restores_rcparams(tmp_path: Path, empty: bool) -> None:
    """五張圖都產生（含全無資料的站）；圖面繪製期間的字型設定在離開後還原。"""

    import matplotlib

    before = dict(family=list(matplotlib.rcParams["font.family"]), minus=matplotlib.rcParams["axes.unicode_minus"])
    paths = aggregate_figures.render_site_figures(
        tmp_path, _synthetic_figure_data(empty=empty), font_selection=FAKE_FONT, coastline=None,
        state_label="MODEL_ONLY", param_id="paramX", period_label="202401-202403", dpi=60,
    )
    assert paths == [f"figures/gongliao/site_gongliao_{kind}.png" for kind in ("spatial", "hotspots", "seasonal", "lifetime", "timeseries")]
    for relative in paths:
        assert (tmp_path / relative).stat().st_size > 1_000
    assert list(matplotlib.rcParams["font.family"]) == before["family"]
    assert matplotlib.rcParams["axes.unicode_minus"] == before["minus"]


def test_render_site_figures_refuses_without_font(tmp_path: Path) -> None:
    """沒有已驗證的 CJK 字型就拒絕繪圖，且不留下任何檔案。"""

    with pytest.raises(ValueError, match="CJK"):
        aggregate_figures.render_site_figures(
            tmp_path, _synthetic_figure_data(), font_selection=None, coastline=None, state_label="MODEL_ONLY",
            param_id="p", period_label="202401",
        )
    assert not list(tmp_path.rglob("*.png"))


@pytest.mark.filterwarnings("ignore")
def test_all_figure_text_is_covered_by_required_glyph_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """實際畫在五張圖上的每個文字（去除 mathtext）都必須落在字型 gate 的必要字元集合內。"""

    import matplotlib.figure
    import matplotlib.text

    collected: list[str] = []
    original_savefig = matplotlib.figure.Figure.savefig

    def _collect(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        for artist in self.findobj(matplotlib.text.Text):
            collected.append(artist.get_text())
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _collect)
    for empty in (False, True):
        aggregate_figures.render_site_figures(
            tmp_path / str(empty), _synthetic_figure_data(empty=empty), font_selection=FAKE_FONT, coastline=None,
            state_label="PILOT_ONLY" if empty else "MODEL_ONLY", param_id="p", period_label="202401-202512", dpi=40,
        )
    assert any("貢寮" in text for text in collected)
    assert any("MODEL_ONLY 表層短期吸引潛勢" in text for text in collected)
    assert any("© OpenStreetMap contributors / ODbL" in text for text in collected)
    assert any(FAKE_FONT.family in text for text in collected)
    import re

    missing = set()
    for text in collected:
        plain = re.sub(r"\$[^$]*\$", "", text)
        missing |= {char for char in plain if char not in REQUIRED_FIGURE_GLYPHS and char not in "\n\t"}
    assert not missing, sorted(missing)


def test_running_mean_with_gaps_keeps_breaks() -> None:
    """滑動平均只平均有限值；缺口日與視窗有效日不足處保持 NaN。"""

    values = np.array([1.0] * 40 + [np.nan] * 5 + [3.0] * 40)
    smooth = aggregate_figures.running_mean_with_gaps(values, window=30)
    assert np.all(np.isnan(smooth[40:45]))  # 缺口日仍是斷點
    assert smooth[20] == pytest.approx(1.0)
    assert smooth[65] == pytest.approx(3.0)
    assert smooth[0] == pytest.approx(1.0)  # 端點視窗恰有一半有效日
    assert np.isnan(aggregate_figures.running_mean_with_gaps(np.ones(45), window=30, min_fraction=0.9)[0])
    short = aggregate_figures.running_mean_with_gaps(np.array([1.0, 2.0, 3.0]), window=30)
    assert np.all(np.isnan(short))


def test_daily_series_marks_missing_days_as_nan() -> None:
    """沒有有效快照的日子是 NaN 缺口，不是 0。"""

    calendar, strong, passed = aggregate_figures.daily_series(
        {"2024-01-01": [10, 5, 2, 4, 1.0], "2024-01-04": [20, 5, 10, 4, 4.0]}
    )
    assert calendar.tolist() == [np.datetime64("2024-01-0%d" % d) for d in (1, 2, 3, 4)]
    assert strong[0] == pytest.approx(0.2) and strong[3] == pytest.approx(0.5)
    assert np.all(np.isnan(strong[1:3])) and np.all(np.isnan(passed[1:3]))
    assert passed[3] == pytest.approx(0.2)
    assert aggregate_figures.daily_series({})[0].size == 0
