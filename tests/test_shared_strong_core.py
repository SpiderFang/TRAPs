"""``aggregate --shared-strong-core-s1-day``（五站共用強核心門檻，opt-in）的合成測試。

沿用 ``test_strong_core_null`` 的合成 shard（A 區 41x41、貢寮／龜山島、每小時一個弱暫態核心、貢寮 25–34 時強核心 -5e-5、
龜山島 5–14 時較弱核心 -3e-5、站外欄位 40–49 時極強核心 -9e-5），但把每條 detection 的 ``pass_cells`` 寫成可預期的格點：

* 一般核心：核心格與其右鄰格（右邊界則取左鄰格）；
* 站外極強核心（s1=-9e-5、位於 analysis_ocean_mask 之外）：除自己兩格外，另外通過一個貢寮站內格點
  （「由站外強核心延伸入站內的曲線」，一般曲線與強核心曲線都應計入貢寮）。

共用門檻 -4e-5 s⁻¹：貢寮核心（-5e-5）與站外核心（-9e-5）為強，龜山島核心（-3e-5）與弱暫態核心（-1e-5）不是。
期望值全部以暴力迴圈由 ``_detections_for_hour`` 獨立重算，不使用 aggregate 內的函式。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from test_aggregate import A_DOMAIN_ID, A_NX, A_NY
from test_cli_pipeline import _run_cli
from test_strong_core_null import (
    GATES,
    MONTH,
    N_HOURS,
    OUT_HOURS,
    G_HOURS,
    H_HOURS,
    S1_G,
    S1_H,
    S1_OUT,
    _detections_for_hour,
    _masks,
    _persistent_cells,
    _write_shard,
)

from traps import aggregate as agg
from traps import period

SHARED_DAY = -3.456  # day^-1，= -4e-5 s^-1
SHARED_SINV = SHARED_DAY / 86400.0
SITES = ("gongliao", "guishan")
LEGACY_NPZ_KEYS = {
    "lon", "lat", "site_mask", "hotspot_curve_mask", "hotspot_core_mask", "hotspot_curve_mask_filtered",
    "hotspot_core_mask_filtered", "strong_core_threshold_sinv", "hotspot_threshold_curve_pass_frequency",
    "hotspot_threshold_core_frequency", "coast_distance_m", "coast_buffer_mask", "coast_buffer_m", "n_valid",
    "core_frequency", "strong_core_frequency", "curve_pass_frequency", "mean_core_s1_sinv", "mean_core_s1_day",
    "mean_lifetime_hours", "track_count", "long_track_count",
}


def _pass_cells_of(row: int, col: int, s1: float) -> list[int]:
    neighbour = col + 1 if col + 1 < A_NX else col - 1
    cells = [row * A_NX + col, row * A_NX + neighbour]
    if s1 == S1_OUT:
        grow, gcol = _persistent_cells()["gongliao"]
        cells.append(grow * A_NX + gcol)
    return cells


def _patch_pass_cells(shard_dir: Path) -> None:
    """把每條 detection 的 pass_cells 改成上面約定的格點，並重寫 shard manifest 的 artifact 校驗。"""

    path = shard_dir / "detections.npz"
    with np.load(path) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    offsets, cells = [0], []
    for row, col, s1 in zip(arrays["core_row"], arrays["core_col"], arrays["core_s1_sinv"], strict=True):
        found = _pass_cells_of(int(row), int(col), float(s1))
        cells.extend(found)
        offsets.append(offsets[-1] + len(found))
    arrays["pass_cell_offsets"] = np.asarray(offsets, dtype=np.int64)
    arrays["pass_cells"] = np.asarray(cells, dtype=np.int32)
    period._atomic_write_npz(path, **arrays)
    manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"] = period._artifact_records(shard_dir, period._SHARD_ARTIFACT_NAMES)
    manifest.pop("manifest_sha256", None)
    period._write_manifest_with_checksum(shard_dir, manifest)


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("shared_strong")
    detect_root = base / "detect"
    shard = detect_root / A_DOMAIN_ID / MONTH
    _write_shard(shard)
    _patch_pass_cells(shard)
    track = base / "track"
    result = _run_cli(
        "track-period", "--shards-root", str(detect_root / A_DOMAIN_ID), "--domain-id", A_DOMAIN_ID,
        "--output-dir", str(track / A_DOMAIN_ID), "--months", MONTH, *GATES,
    )
    assert result.returncode == 0, result.stderr
    return {"base": base, "detect": detect_root, "track": track}


def _run_aggregate(setup: dict[str, Path], out: Path, *extra: str) -> None:
    result = _run_cli(
        "aggregate", "--track-root", str(setup["track"]), "--detect-root", str(setup["detect"]), "--output-dir", str(out),
        "--strong-quantile", "0.1", "--no-figures", "--allow-partial-domains", "--bootstrap-reps", "20", *extra,
    )
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.fixture(scope="module")
def default_run(setup: dict[str, Path]) -> Path:
    out = setup["base"] / "agg_default"
    _run_aggregate(setup, out)
    return out


@pytest.fixture(scope="module")
def shared_run(setup: dict[str, Path]) -> Path:
    out = setup["base"] / "agg_shared"
    _run_aggregate(setup, out, f"--shared-strong-core-s1-day={SHARED_DAY}")
    return out


def _npz(root: Path, site: str, stratum: str = "all") -> dict[str, np.ndarray]:
    with np.load(root / "spatial" / site / f"spatial_{stratum}.npz") as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def _expected() -> dict[str, dict[str, np.ndarray | int]]:
    """暴力重算：逐小時、逐 detection，依共用門檻累計各站的核心數與曲線通過次數。"""

    _ocean, masks = _masks()
    result: dict[str, dict] = {
        site: {
            "strong_curve": np.zeros((A_NY, A_NX), dtype=np.int64),
            "all_curve": np.zeros((A_NY, A_NX), dtype=np.int64),
            "strong_core": np.zeros((A_NY, A_NX), dtype=np.int64),
            "cores": 0, "strong_cores": 0, "strong_core_hours": 0, "strong_curve_hours": 0,
        }
        for site in SITES
    }
    for hour in range(N_HOURS):
        qualified = [(r, c, s) for r, c, s, q in _detections_for_hour(hour) if q]
        strong_cells: set[int] = set()
        all_cells: set[int] = set()
        for row, col, s1 in qualified:
            cells = _pass_cells_of(row, col, s1)
            all_cells.update(cells)
            if s1 <= SHARED_SINV:
                strong_cells.update(cells)
        for site in SITES:
            mask = masks[site]
            rec = result[site]
            in_site = [(r, c, s) for r, c, s in qualified if mask[r, c]]
            rec["cores"] += len(in_site)
            strong_in_site = [(r, c) for r, c, s in in_site if s <= SHARED_SINV]
            rec["strong_cores"] += len(strong_in_site)
            rec["strong_core_hours"] += int(bool(strong_in_site))
            for r, c in strong_in_site:
                rec["strong_core"][r, c] += 1
            site_strong = [cell for cell in strong_cells if mask[cell // A_NX, cell % A_NX]]
            rec["strong_curve_hours"] += int(bool(site_strong))
            for cell in site_strong:
                rec["strong_curve"][cell // A_NX, cell % A_NX] += 1
            for cell in all_cells:
                if mask[cell // A_NX, cell % A_NX]:
                    rec["all_curve"][cell // A_NX, cell % A_NX] += 1
    return result


# ---------------------------------------------------------------------------
# 預設行為不變
# ---------------------------------------------------------------------------


def test_default_outputs_have_no_shared_fields(default_run: Path) -> None:
    for site in SITES:
        assert set(_npz(default_run, site)) == LEGACY_NPZ_KEYS
    with (default_run / "site_statistics.csv").open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert header == agg.site_stat_fieldnames()
    assert not any(name.startswith("shared_") for name in header)
    manifest = json.loads((default_run / "manifest.json").read_text(encoding="utf-8"))
    assert "shared_strong_core" not in manifest
    # 預設仍是各站自己的分位數門檻（貢寮 -5e-5、龜山島 -3e-5），不是共用門檻
    assert manifest["strong_core_thresholds"]["gongliao"]["strong_core_threshold_sinv"] == S1_G
    assert manifest["strong_core_thresholds"]["guishan"]["strong_core_threshold_sinv"] == S1_H


def test_default_curve_and_core_fields_unchanged_by_flag(default_run: Path, shared_run: Path) -> None:
    """開啟旗標只多出欄位；所有舊欄位中與強核心無關者（一般曲線頻率、核心頻率、n_valid、遮罩）逐格相同。"""

    for site in SITES:
        old, new = _npz(default_run, site), _npz(shared_run, site)
        for key in ("n_valid", "core_frequency", "curve_pass_frequency", "site_mask", "lon", "lat", "mean_core_s1_sinv",
                    "hotspot_curve_mask", "hotspot_core_mask", "track_count"):
            np.testing.assert_array_equal(old[key], new[key], err_msg=f"{site}/{key}")


# ---------------------------------------------------------------------------
# 開啟後：只計入 s1 <= 門檻的核心曲線
# ---------------------------------------------------------------------------


def test_strong_curve_frequency_counts_only_strong_core_curves(shared_run: Path) -> None:
    expected = _expected()
    _ocean, masks = _masks()
    for site in SITES:
        data = _npz(shared_run, site)
        assert LEGACY_NPZ_KEYS <= set(data)
        n_valid = data["n_valid"]
        assert int(n_valid[masks[site]].min()) == N_HOURS
        exp = expected[site]
        np.testing.assert_array_equal(data["strong_curve_pass_count"], np.where(masks[site], exp["strong_curve"], 0))
        np.testing.assert_array_equal(data["strong_core_count"], np.where(masks[site], exp["strong_core"], 0))
        # 頻率 ＝ 計數 ／ n_valid（站外為 NaN）
        freq = data["strong_curve_pass_frequency"]
        np.testing.assert_allclose(freq[masks[site]], exp["strong_curve"][masks[site]] / N_HOURS)
        assert np.all(np.isnan(freq[~masks[site]]))
        # 一般曲線（所有核心）仍然逐格等於暴力重算，且強核心曲線 <= 一般曲線
        np.testing.assert_array_equal(
            np.rint(data["curve_pass_frequency"][masks[site]] * N_HOURS).astype(np.int64), exp["all_curve"][masks[site]]
        )
        assert np.all(data["strong_curve_pass_count"] <= np.rint(np.nan_to_num(data["curve_pass_frequency"]) * N_HOURS))


def test_weak_cores_curves_excluded_and_outside_strong_core_curve_included(shared_run: Path) -> None:
    grow, gcol = _persistent_cells()["gongliao"]
    hrow, hcol = _persistent_cells()["guishan"]
    gong, gui = _npz(shared_run, "gongliao"), _npz(shared_run, "guishan")
    # 貢寮強核心格：自己的 10 小時 ＋ 站外強核心曲線延伸入站內的 10 小時
    assert gong["strong_curve_pass_count"][grow, gcol] == len(G_HOURS) + len(OUT_HOURS)
    assert gong["strong_core_count"][grow, gcol] == len(G_HOURS)  # 站外核心不算貢寮的強核心
    # 龜山島核心 -3e-5 弱於門檻：一般曲線有，強核心曲線與強核心計數為零
    assert gui["curve_pass_frequency"][hrow, hcol] * N_HOURS >= len(H_HOURS) - 1e-9
    assert gui["strong_curve_pass_count"][hrow, hcol] == 0
    assert int(gui["strong_core_count"].sum()) == 0 and int(gui["strong_curve_pass_count"].sum()) == 0
    # 共用門檻取代各站分位數門檻
    assert float(gong["strong_core_threshold_sinv"]) == pytest.approx(SHARED_SINV)
    assert float(gui["strong_core_threshold_sinv"]) == pytest.approx(SHARED_SINV)
    assert float(gui["shared_strong_core_s1_day"]) == SHARED_DAY


def test_npz_carries_grid_fields_for_offline_replot(shared_run: Path) -> None:
    for site in SITES:
        data = _npz(shared_run, site)
        for key in ("n_valid", "site_mask", "lon", "lat", "domain_coast_distance_m", "domain_valid_sea_mask",
                    "strong_curve_pass_frequency", "strong_core_frequency", "core_count"):
            assert key in data, key
        assert data["domain_coast_distance_m"].shape == data["site_mask"].shape
        valid_sea = data["domain_valid_sea_mask"]
        _ocean, masks = _masks()
        np.testing.assert_array_equal(valid_sea, masks["gongliao"] | masks["guishan"])


def test_manifest_and_site_statistics_record_shared_counts(shared_run: Path) -> None:
    expected = _expected()
    manifest = json.loads((shared_run / "manifest.json").read_text(encoding="utf-8"))
    block = manifest["shared_strong_core"]
    assert block["shared_strong_core_s1_day"] == SHARED_DAY
    assert block["shared_strong_core_s1_sinv"] == pytest.approx(SHARED_SINV)
    with (shared_run / "site_statistics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (shared_run / "site_statistics.csv").open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert header == agg.site_stat_fieldnames(with_shared_strong=True)
    for site in SITES:
        exp = expected[site]
        info = block["sites"][site]
        assert info["core_count_total"] == exp["cores"] > 0
        assert info["strong_core_count_total"] == exp["strong_cores"]
        assert info["strong_core_snapshots"] == exp["strong_core_hours"]
        assert info["strong_core_hours"] == exp["strong_core_hours"]  # 名目步長 1 h
        assert info["strong_curve_snapshots"] == exp["strong_curve_hours"]
        assert info["strong_core_fraction"] == pytest.approx(exp["strong_cores"] / exp["cores"])
        assert info["reference_site_quantile_threshold_day"] is not None
        row = next(r for r in rows if r["site_id"] == site and r["stratum"] == "all")
        assert float(row["shared_strong_core_s1_day"]) == SHARED_DAY
        assert int(row["shared_core_count_total"]) == exp["cores"]
        assert int(row["shared_strong_core_count_total"]) == exp["strong_cores"]
        assert int(row["shared_strong_core_snapshots"]) == exp["strong_core_hours"]
        assert int(row["shared_strong_curve_snapshots"]) == exp["strong_curve_hours"]
        assert int(row["strong_core_occurrence_numerator"]) == exp["strong_core_hours"]
    assert block["sites"]["gongliao"]["strong_core_count_total"] == len(G_HOURS)
    assert block["sites"]["guishan"]["strong_core_count_total"] == 0
    # 貢寮的強核心曲線時數 ＝ 自己 10 小時 ∪ 站外強核心延伸入站內 10 小時（兩段不重疊）
    assert block["sites"]["gongliao"]["strong_curve_snapshots"] == len(G_HOURS) + len(OUT_HOURS)


@pytest.mark.parametrize("value", ["0", "5", "nan"])
def test_invalid_shared_threshold_rejected(setup: dict[str, Path], tmp_path: Path, value: str) -> None:
    result = _run_cli(
        "aggregate", "--track-root", str(setup["track"]), "--detect-root", str(setup["detect"]), "--output-dir", str(tmp_path / "bad"),
        "--no-figures", "--allow-partial-domains", "--bootstrap-reps", "20", f"--shared-strong-core-s1-day={value}",
    )
    assert result.returncode != 0 and "shared-strong-core-s1-day" in (result.stderr + result.stdout)
