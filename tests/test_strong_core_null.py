"""``track-period --track-strong-quantile``、``aggregate --null-track-root`` 與對應 runner 旗標的合成測試。

合成 shard 直接手寫（不經 detect-shards），使每個核心的位置、強度與合格與否都由測試決定：

* A 區 41x41 網格（500 m 格距）、真實 anchor 經緯度，貢寮／龜山島依最近 anchor 分割；
  最左兩欄不在 analysis_ocean_mask 內（站外）。
* 每小時一個弱「暫態」核心（位置逐時大幅跳動、s1 約 -1e-5）；貢寮在 25–34 時有持續 10 小時的強核心
  （s1=-5e-5）、龜山島在 5–14 時有較弱的持續核心（s1=-3e-5）；站外欄位在 40–49 時有極強核心
  （s1=-9e-5，只在站外）；另有不合格的強核心（不應進入任何母體）。
* 緊追蹤門檻（600 m／1200 m）使暫態核心彼此不會串接，持續核心則串成一條 track。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_aggregate import (  # noqa: F401
    A_BASE_NS,
    A_DOMAIN_ID,
    A_HOUR_NS,
    A_NX,
    A_NY,
    A_X_SPACING_M,
    _a_axes,
    _namespace,
    fake_font,
)
from test_cli_pipeline import PROJECT_ROOT, _run_cli

from traps import aggregate as agg
from traps import period
from traps.models import DetectionConfig
from traps.sites import DEFAULT_STUDY_DESIGN_PATH, site_masks

MONTH = "202401"
N_HOURS = 60
GATES = ["--tracking-max-core-distance-m", "600", "--tracking-max-curve-distance-m", "1200"]
S1_G, S1_H, S1_OUT, S1_UNQ = -5.0e-5, -3.0e-5, -9.0e-5, -8.0e-5
G_HOURS, H_HOURS, OUT_HOURS = range(25, 35), range(5, 15), range(40, 50)


def _masks() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lon_axis, lat_axis, _x, _y = _a_axes()
    ocean = np.ones((A_NY, A_NX), dtype=bool)
    ocean[:, :2] = False
    return ocean, site_masks(A_DOMAIN_ID, lon_axis, lat_axis, ocean)


def _persistent_cells() -> dict[str, tuple[int, int]]:
    _ocean, masks = _masks()
    cells = {}
    for site_id in ("gongliao", "guishan"):
        found = np.argwhere(masks[site_id])
        cells[site_id] = tuple(int(v) for v in found[found.shape[0] // 2])
    return cells


def _detections_for_hour(hour: int) -> list[tuple[int, int, float, bool]]:
    """該小時的 (row, col, s1, qualified) 清單。"""

    persistent = _persistent_cells()
    items = [((3 * hour) % A_NY, 2 + (7 * hour) % (A_NX - 2), -1.0e-5 - 1.0e-8 * hour, True)]
    if hour in G_HOURS:
        items.append((*persistent["gongliao"], S1_G, True))
    if hour in H_HOURS:
        items.append((*persistent["guishan"], S1_H, True))
    if hour in OUT_HOURS:
        items.append((5, 0, S1_OUT, True))
    if hour % 10 == 3:
        items.append((30, 30, S1_UNQ, False))
    return [(int(r), int(c), float(s), bool(q)) for r, c, s, q in items]


def _write_shard(shard_dir: Path) -> None:
    ocean, _masks_ = _masks()
    lon_axis, lat_axis, x_axis, y_axis = _a_axes()
    shard_dir.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {name: [] for name in (
        "snapshot_index", "time_utc_ns", "detection_index", "core_row", "core_col", "core_x_m", "core_y_m", "core_s1_sinv",
        "is_qualified",
    )}
    offsets, points, strengths = [0], [], []
    qualified_count = []
    for hour in range(N_HOURS):
        items = _detections_for_hour(hour)
        qualified_count.append(sum(1 for item in items if item[3]))
        for index, (row, col, s1, qualified) in enumerate(items):
            x, y = float(x_axis[col]), float(y_axis[row])
            cols["snapshot_index"].append(hour)
            cols["time_utc_ns"].append(A_BASE_NS + hour * A_HOUR_NS)
            cols["detection_index"].append(index)
            cols["core_row"].append(row)
            cols["core_col"].append(col)
            cols["core_x_m"].append(x)
            cols["core_y_m"].append(y)
            cols["core_s1_sinv"].append(s1)
            cols["is_qualified"].append(qualified)
            points.append(np.array([[x - 500.0, y], [x + 500.0, y]], dtype=np.float32))
            strengths.append(np.array([s1, s1], dtype=np.float32))
            offsets.append(offsets[-1] + 2)
    n = len(cols["snapshot_index"])
    period._atomic_write_npz(
        shard_dir / "detections.npz",
        **{name: np.asarray(values, dtype=bool if name == "is_qualified" else (np.float64 if "_m" in name or "s1" in name else np.int64))
           for name, values in cols.items()},
        curve_length_m=np.full(n, 1000.0),
        curve_mean_s1_sinv=np.asarray(cols["core_s1_sinv"], dtype=np.float64),
        curve_mean_abs_s1_sinv=np.abs(np.asarray(cols["core_s1_sinv"], dtype=np.float64)),
        forward_stop_code=np.zeros(n, dtype=np.int64),
        backward_stop_code=np.zeros(n, dtype=np.int64),
        curve_offsets=np.asarray(offsets, dtype=np.int64),
        curve_points_m=np.concatenate(points, axis=0),
        curve_s1_sinv=np.concatenate(strengths, axis=0),
        pass_cell_offsets=np.zeros(n + 1, dtype=np.int64),
        pass_cells=np.zeros(0, dtype=np.int32),
    )
    valid_bits = np.packbits(np.ones(A_NY * A_NX, dtype=bool))
    period._atomic_write_npz(
        shard_dir / "snapshots.npz",
        snapshot_index=np.arange(N_HOURS, dtype=np.int64),
        time_utc_ns=A_BASE_NS + np.arange(N_HOURS, dtype=np.int64) * A_HOUR_NS,
        valid_cell_count=np.full(N_HOURS, int(ocean.sum()), dtype=np.int64),
        expected_valid_cell_count=np.full(N_HOURS, int(ocean.sum()), dtype=np.int64),
        coverage_fraction=np.ones(N_HOURS),
        coverage_pass=np.ones(N_HOURS, dtype=bool),
        detection_count=np.asarray(qualified_count, dtype=np.int64),
        qualified_count=np.asarray(qualified_count, dtype=np.int64),
        valid_mask_packed=np.stack([valid_bits] * N_HOURS, axis=0),
    )
    period._atomic_write_npz(
        shard_dir / "grid_reference.npz",
        lon=lon_axis, lat=lat_axis, x_m_axis=x_axis, y_m_axis=y_axis,
        cell_area_m2=np.full((A_NY, A_NX), A_X_SPACING_M**2), analysis_ocean_mask=ocean,
    )
    manifest = {
        "operation": "detect-shards", "schema_version": period.SHARD_SCHEMA_VERSION, "param_id": "synthetic",
        "detection_config": DetectionConfig().to_manifest_dict(), "min_coverage": 0.95, "domain_id": A_DOMAIN_ID,
        "month": MONTH, "state_label": "FORMAL_TEST", "snapshot_count": N_HOURS,
        "artifacts": period._artifact_records(shard_dir, period._SHARD_ARTIFACT_NAMES),
    }
    period._write_manifest_with_checksum(shard_dir, manifest)
    (shard_dir / "_SUCCESS").write_text("{}\n", encoding="utf-8")


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("strong_null")
    detect_root = base / "detect"
    _write_shard(detect_root / A_DOMAIN_ID / MONTH)

    def track(name: str, *extra: str, gates: list[str] = GATES) -> Path:
        out = base / name
        result = _run_cli(
            "track-period", "--shards-root", str(detect_root / A_DOMAIN_ID), "--domain-id", A_DOMAIN_ID,
            "--output-dir", str(out / A_DOMAIN_ID), "--months", MONTH, *gates, *extra,
        )
        assert result.returncode == 0, result.stderr
        return out

    return {
        "detect": detect_root,
        "all": track("track_all"),
        "strong": track("track_strong", "--track-strong-quantile", "0.1"),
        "null1": track("null_s1", "--track-strong-quantile", "0.1", "--shuffle-seed", "1"),
        "null2": track("null_s2", "--track-strong-quantile", "0.1", "--shuffle-seed", "2"),
        "null_nofilter": track("null_nf", "--shuffle-seed", "1"),
        "null_gates": track(
            "null_gates", "--track-strong-quantile", "0.1", "--shuffle-seed", "1",
            gates=["--tracking-max-core-distance-m", "700", "--tracking-max-curve-distance-m", "1200"],
        ),
        "base": base,
    }


def _manifest(root: Path) -> dict:
    return json.loads((root / A_DOMAIN_ID / "manifest.json").read_text(encoding="utf-8"))


def _observations(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / A_DOMAIN_ID / "track_observations.npz") as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def _expected_thresholds(q: float) -> dict[str, float]:
    """獨立於 period／aggregate 程式碼：各站內合格核心 s1 的分位數。"""

    _ocean, masks = _masks()
    values: dict[str, list[float]] = {"gongliao": [], "guishan": []}
    for hour in range(N_HOURS):
        for row, col, s1, qualified in _detections_for_hour(hour):
            if not qualified:
                continue
            for site_id in values:
                if masks[site_id][row, col]:
                    values[site_id].append(s1)
    return {site_id: float(np.quantile(np.asarray(v), q)) for site_id, v in values.items()}


# ---------------------------------------------------------------------------
# track-period --track-strong-quantile
# ---------------------------------------------------------------------------


def test_track_thresholds_equal_aggregate_and_independent_recompute(runs: dict[str, Path], tmp_path: Path) -> None:
    filt = _manifest(runs["strong"])["track_core_filter"]
    assert filt["quantile"] == 0.1 and filt["type"] == "site_strong_core_quantile"
    expected = _expected_thresholds(0.1)
    result = _run_cli(
        "aggregate", "--track-root", str(runs["strong"]), "--detect-root", str(runs["detect"]), "--output-dir", str(tmp_path / "agg"),
        "--strong-quantile", "0.1", "--no-figures", "--allow-partial-domains", "--bootstrap-reps", "20",
    )
    assert result.returncode == 0, result.stderr
    aggregate_manifest = json.loads((tmp_path / "agg" / "manifest.json").read_text(encoding="utf-8"))
    for site_id, value in expected.items():
        track_value = filt["thresholds"][site_id]["threshold_sinv"]
        agg_value = aggregate_manifest["strong_core_thresholds"][site_id]["strong_core_threshold_sinv"]
        assert track_value == agg_value == value  # 位元相同
        assert filt["thresholds"][site_id]["threshold_day"] == aggregate_manifest["strong_core_thresholds"][site_id]["strong_core_threshold_day"]
        assert filt["thresholds"][site_id]["qualified_core_count"] == aggregate_manifest["strong_core_thresholds"][site_id]["qualified_core_count"]
    assert filt["thresholds"]["gongliao"]["threshold_sinv"] == S1_G and filt["thresholds"]["guishan"]["threshold_sinv"] == S1_H
    # 同一份記錄也寫在 tracking_summary.json
    summary = json.loads((runs["strong"] / A_DOMAIN_ID / "tracking_summary.json").read_text(encoding="utf-8"))
    assert summary["track_core_filter"] == filt
    assert aggregate_manifest["track_core_filter"]["quantile"] == 0.1
    assert "強核心結構" in aggregate_manifest["lifetime_scope_zh"] and "10%" in aggregate_manifest["lifetime_scope_zh"]


def test_filter_drops_weak_and_out_of_site_cores(runs: dict[str, Path]) -> None:
    unfiltered = _observations(runs["all"])
    filtered = _observations(runs["strong"])
    filt = _manifest(runs["strong"])["track_core_filter"]
    assert _manifest(runs["all"])["track_core_filter"] is None
    # 未篩選：站外極強核心與弱暫態核心都在 track 內
    assert np.any(unfiltered["core_s1_sinv"] == S1_OUT) and np.any(unfiltered["core_s1_sinv"] > -2.0e-5)
    # 篩選後：只剩兩條持續強核心，站外核心與弱核心全被丟棄
    assert set(np.unique(filtered["core_s1_sinv"]).tolist()) == {S1_G, S1_H}
    assert filt["dropped_outside_sites"] == len(OUT_HOURS)
    assert filt["thresholds"]["gongliao"]["kept_core_count"] == len(G_HOURS)
    assert filt["thresholds"]["guishan"]["kept_core_count"] == len(H_HOURS)
    kept = sum(v["kept_core_count"] for v in filt["thresholds"].values())
    assert filt["dropped_weak"] == filt["qualified_detections_seen"] - kept - filt["dropped_outside_sites"] > 0
    assert len(np.unique(filtered["track_id"])) == 2


def test_shuffled_filtered_uses_real_thresholds(runs: dict[str, Path]) -> None:
    real = _manifest(runs["strong"])["track_core_filter"]
    null = _manifest(runs["null1"])
    assert null["null_model"] == {"type": "time_shuffle", "seed": 1}
    for site_id, info in real["thresholds"].items():
        assert null["track_core_filter"]["thresholds"][site_id]["threshold_sinv"] == info["threshold_sinv"]
        assert null["track_core_filter"]["thresholds"][site_id]["kept_core_count"] == info["kept_core_count"]  # 置換保留多重集合
    obs = _observations(runs["null1"])
    assert set(np.unique(obs["core_s1_sinv"]).tolist()) <= {S1_G, S1_H}
    # 持續核心被打散：置換後最長 track 明顯較短
    longest_real = np.bincount(_observations(runs["strong"])["track_id"]).max()
    assert longest_real == 10 and np.bincount(obs["track_id"]).max() < longest_real


def test_strong_quantile_validation_and_default_unchanged(runs: dict[str, Path], tmp_path: Path) -> None:
    result = _run_cli(
        "track-period", "--shards-root", str(runs["detect"] / A_DOMAIN_ID), "--domain-id", A_DOMAIN_ID,
        "--output-dir", str(tmp_path / "bad"), "--months", MONTH, "--track-strong-quantile", "1.5",
    )
    assert result.returncode != 0 and "track-strong-quantile" in (result.stderr + result.stdout)
    summary = json.loads((runs["all"] / A_DOMAIN_ID / "tracking_summary.json").read_text(encoding="utf-8"))
    assert summary["track_core_filter"] is None


# ---------------------------------------------------------------------------
# aggregate --null-track-root
# ---------------------------------------------------------------------------


def _aggregate(runs: dict[str, Path], out: Path, *extra: str, track: str = "strong") -> subprocess.CompletedProcess[str]:
    return _run_cli(
        "aggregate", "--track-root", str(runs[track]), "--detect-root", str(runs["detect"]), "--output-dir", str(out),
        "--strong-quantile", "0.1", "--no-figures", "--allow-partial-domains", "--bootstrap-reps", "20", *extra,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_aggregate_null_columns_and_positive_excess(runs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "agg"
    result = _aggregate(runs, out, "--null-track-root", str(runs["null1"]), str(runs["null2"]))
    assert result.returncode == 0, result.stderr
    rows = _rows(out / "site_statistics.csv")
    header = rows[0].keys()
    for name in ("null_n_seeds", "null_n_uncensored", "null_median_lifetime_hours", "null_p90_lifetime_hours",
                 "null_track_frac_uncensored_ge_6h", "excess_track_frac_uncensored_ge_6h", "ratio_track_frac_uncensored_ge_6h",
                 "excess_track_frac_uncensored_ge_24h", "track_core_filter_quantile"):
        assert name in header
    gong = next(r for r in rows if r["site_id"] == "gongliao" and r["stratum"] == "all")
    assert float(gong["track_frac_uncensored_ge_6h"]) == 1.0  # 真實：唯一一條 10 h 持續 track
    assert int(gong["null_n_seeds"]) == 2
    null_frac = float(gong["null_track_frac_uncensored_ge_6h"])
    assert null_frac < 1.0
    assert float(gong["excess_track_frac_uncensored_ge_6h"]) == pytest.approx(1.0 - null_frac)
    assert float(gong["excess_track_frac_uncensored_ge_6h"]) > 0.0
    assert float(gong["track_core_filter_quantile"]) == 0.1
    assert float(gong["null_median_lifetime_hours"]) < float(gong["track_median_lifetime_hours"])
    # 只有 stratum all 有 null 欄位
    for row in rows:
        if row["stratum"] != "all":
            assert row["null_n_seeds"] == "" and row["excess_track_frac_uncensored_ge_6h"] == ""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [item["seed"] for item in manifest["null_inputs"]] == [1, 2]
    for item, key in zip(manifest["null_inputs"], ("null1", "null2")):
        record = item["domains"][A_DOMAIN_ID]
        assert record["manifest_sha256"] == hashlib.sha256((runs[key] / A_DOMAIN_ID / "manifest.json").read_bytes()).hexdigest()
        assert item["track_root"] == str(runs[key])
    assert manifest["track_core_filter"]["quantile"] == 0.1
    payload = json.loads((out / "site_statistics.json").read_text(encoding="utf-8"))
    assert "強核心結構" in payload["lifetime_scope_zh"]


def test_aggregate_without_null_or_filter_keeps_legacy_columns(runs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "agg"
    result = _aggregate(runs, out, track="all")
    assert result.returncode == 0, result.stderr
    with (out / "site_statistics.csv").open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert header == agg.site_stat_fieldnames()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["track_core_filter"] is None and manifest["null_inputs"] == []


@pytest.mark.parametrize(
    ("null_key", "message"),
    [("null_gates", "tracking_config"), ("null_nofilter", "track_core_filter"), ("all", "null_model")],
)
def test_aggregate_refuses_mismatched_null_inputs(runs: dict[str, Path], tmp_path: Path, null_key: str, message: str) -> None:
    result = _aggregate(runs, tmp_path / "agg", "--null-track-root", str(runs[null_key]))
    assert result.returncode != 0
    assert message in (result.stderr + result.stdout)


def test_aggregate_refuses_duplicate_seed_and_null_primary(runs: dict[str, Path], tmp_path: Path) -> None:
    dup = _aggregate(runs, tmp_path / "a", "--null-track-root", str(runs["null1"]), str(runs["null1"]))
    assert dup.returncode != 0
    primary = _aggregate(runs, tmp_path / "b", track="null1")
    assert primary.returncode != 0 and "null_model" in (primary.stderr + primary.stdout)


@pytest.mark.filterwarnings("ignore")
def test_lifetime_figure_overlays_null_ccdf(runs: dict[str, Path], tmp_path: Path, fake_font) -> None:  # noqa: F811
    out = tmp_path / "agg"
    args = _namespace(
        runs["strong"], runs["detect"], out, bootstrap_reps=20,
        null_track_root=[runs["null1"], runs["null2"]],
    )
    assert agg.run_aggregate(args) == 0
    assert (out / "figures" / "gongliao" / "site_gongliao_lifetime.png").stat().st_size > 0


def test_null_ccdf_band_and_figure_text_glyphs() -> None:
    from traps import aggregate_figures, report_font

    grid, mean, low, high = aggregate_figures.null_ccdf_band([[1.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
    assert grid.tolist() == [1.0, 2.0, 4.0]
    assert mean.tolist() == pytest.approx([1.0, (2 / 3 + 0) / 2, (1 / 3) / 2])
    assert np.all(low <= mean) and np.all(mean <= high)
    assert aggregate_figures.null_ccdf_band([[], []])[0].size == 0
    text = agg.lifetime_scope_text({"quantile": 0.1})
    assert set(text) <= report_font.REQUIRED_FIGURE_GLYPHS | set("0123456789.")
    for key in ("null_ccdf_legend", "null_band_legend", "excess", "lifetime_refers_to", "strong_core_structures", "per_site_top"):
        assert set(report_font.FIGURE_TEXT[key]) <= report_font.REQUIRED_FIGURE_GLYPHS


# ---------------------------------------------------------------------------
# null_test_tracking.py 標籤
# ---------------------------------------------------------------------------


def test_null_test_script_records_core_filter(runs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "cmp"
    result = subprocess.run(
        ["python3", str(PROJECT_ROOT / "scripts" / "null_test_tracking.py"), "--output-dir", str(out), "--runs",
         f"real={runs['strong']}", f"s1={runs['null1']}", f"s2={runs['null2']}", f"all={runs['all']}"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    rows = {r["run"]: r for r in _rows(out / "comparison.csv")}
    assert float(rows["real"]["core_filter_q"]) == 0.1 and rows["all"]["core_filter_q"] == ""
    assert "strong-core q=0.1" in result.stdout and rows["real"]["coherence_ratio"] != ""
    assert rows["all"]["coherence_ratio"] == ""  # 沒有配對的 null（篩選條件不同）


# ---------------------------------------------------------------------------
# runner：TRACK / TRACKNULL / AGGREGATE 旗標透傳（以 stub 直譯器取代 .venv python）
# ---------------------------------------------------------------------------

RUNNER = PROJECT_ROOT / "scripts" / "run_traps_server.sh"
STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STUB_LOG"
out=""
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--output-dir" ]]; then out="$arg"; fi
  prev="$arg"
done
if [[ -n "$out" && "$1" == "-m" && "$3" == "track-period" ]]; then mkdir -p "$out"; : > "$out/_SUCCESS"; fi
exit 0
"""


def _runner_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    repo = tmp_path / "repo"
    py = repo / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text(STUB, encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRAPS_")}
    env.update(
        TRAPS_WORKSPACE=str(tmp_path), TRAPS_REPO=str(repo), TRAPS_RUNS=str(tmp_path / "runs"), TRAPS_DOMAINS="dA dB",
        TRAPS_MONTHS="202401", TRAPS_PARAM_ID="p", STUB_LOG=str(tmp_path / "calls.log"),
    )
    env.update(extra)
    return env


def _run_mode(mode: str, env: dict[str, str]) -> list[str]:
    log = Path(env["STUB_LOG"])
    before = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    result = subprocess.run(["bash", str(RUNNER), mode], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return log.read_text(encoding="utf-8").splitlines()[len(before):]


def test_runner_bash_syntax() -> None:
    assert subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True, check=False).returncode == 0


def test_runner_track_defaults_unchanged(tmp_path: Path) -> None:
    calls = _run_mode("TRACK", _runner_env(tmp_path))
    assert len(calls) == 2 and all("track-period" in c for c in calls)
    for call in calls:
        assert "--track-strong-quantile" not in call and "--tracking-max-core-distance-m" not in call
        assert "--shuffle-seed" not in call and "/runs/p/track/" in call
    daily = _run_mode("TRACK", _runner_env(tmp_path / "d", TRAPS_FIELD="daily25h"))
    for call in daily:
        assert "--tracking-max-core-distance-m 10000" in call and "--tracking-max-curve-distance-m 20000" in call
        assert "--nominal-time-step-hours 24" in call


def test_runner_track_env_passthrough_and_subdir(tmp_path: Path) -> None:
    env = _runner_env(
        tmp_path, TRAPS_TRACK_STRONG_QUANTILE="0.1", TRAPS_TRACK_CORE_M="8000", TRAPS_TRACK_CURVE_M="16000",
        TRAPS_TRACK_SUBDIR="track_strong",
    )
    calls = _run_mode("TRACK", env)
    assert len(calls) == 2
    for call in calls:
        assert "--track-strong-quantile 0.1" in call
        assert "--tracking-max-core-distance-m 8000" in call and "--tracking-max-curve-distance-m 16000" in call
        assert "/runs/p/track_strong/d" in call
    daily = _run_mode("TRACK", {**_runner_env(tmp_path / "d", TRAPS_FIELD="daily25h", TRAPS_TRACK_CORE_M="7000")})
    assert all("--tracking-max-core-distance-m 7000" in c and "--tracking-max-curve-distance-m 20000" in c for c in daily)


def test_runner_tracknull_and_aggregate_auto_null(tmp_path: Path) -> None:
    env = _runner_env(
        tmp_path, TRAPS_TRACK_STRONG_QUANTILE="0.1", TRAPS_TRACK_SUBDIR="track_strong", TRAPS_NULL_SEEDS="3 4",
        TRAPS_AGGREGATE_SUBDIR="aggregate_strong",
    )
    calls = _run_mode("TRACKNULL", env)
    assert len(calls) == 4
    for seed in ("3", "4"):
        for domain in ("dA", "dB"):
            assert any(
                f"--shuffle-seed {seed}" in c and f"/runs/p/track_strong_null_s{seed}/{domain}" in c and "--track-strong-quantile 0.1" in c
                for c in calls
            ), (seed, domain)
    run_dir = tmp_path / "runs" / "runs" / "p"
    # 一組對照組不完整 -> 略過；完整的 -> 自動帶入
    (run_dir / "track_strong_null_s4" / "dB" / "_SUCCESS").unlink()
    agg_call = [c for c in _run_mode("AGGREGATE", env) if " aggregate " in f" {c} " and "--help" not in c][0]
    assert f"--track-root {run_dir}/track_strong" in agg_call
    assert f"--output-dir {run_dir}/aggregate_strong" in agg_call
    assert f"--null-track-root {run_dir}/track_strong_null_s3" in agg_call and "null_s4" not in agg_call
    # 補齊後兩組都帶入；使用者自行給 --null-track-root 時不重複帶入
    (run_dir / "track_strong_null_s4" / "dB" / "_SUCCESS").write_text("")
    both = [c for c in _run_mode("AGGREGATE", env) if "--output-dir" in c][0]
    assert "null_s3" in both and "null_s4" in both
    manual = [c for c in _run_mode("AGGREGATE", {**env, "TRAPS_AGGREGATE_ARGS": "--null-track-root /x/y"}) if "--output-dir" in c][0]
    assert manual.count("--null-track-root") == 1 and "/x/y" in manual


def test_runner_aggregate_default_has_no_null_args(tmp_path: Path) -> None:
    call = [c for c in _run_mode("AGGREGATE", _runner_env(tmp_path)) if "--output-dir" in c][0]
    assert "--null-track-root" not in call and "/runs/p/aggregate" in call and "--track-root " in call
