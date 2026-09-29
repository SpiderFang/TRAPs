"""track-period ``--shuffle-seed``（時間置換 null model）與 ``scripts/null_test_tracking.py`` 的合成測試。

合成流場為 ``test_period.py`` 的緩慢平移鞍點：真實時序下每小時只移動 15 m（核心吸附到 150 m 格點），
以 100 m／200 m 的緊門檻仍可串成長 track；置換時間後相鄰槽位的鞍點位置相差
平均遠大於門檻，track 應大幅變短。
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_cli_pipeline import PROJECT_ROOT, _run_cli
from test_period import DOMAIN_ID, _run_detect_shards, _shard_output, _write_month, _write_shared_grid

from traps import aggregate as agg
from traps.period import _time_shuffle_source_rows

HOURS = list(range(1000, 1040))
GATES = ["--tracking-max-core-distance-m", "100", "--tracking-max-curve-distance-m", "200"]


@pytest.fixture(scope="module")
def shards_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("null")
    root = tmp_path / "src"
    _write_shared_grid(root)
    # 一個月份、含一個 coverage 不足的槽位（1020），檢驗 availability 保留。
    _write_month(root, month="202401", hours=HOURS, low_coverage_hours=frozenset({1020}))
    result = _run_detect_shards(root, month="202401", output_dir=_shard_output(tmp_path, "202401"))
    assert result.returncode == 0, result.stderr
    return tmp_path / "shards"


def _track(shards_root: Path, out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run_cli(
        "track-period", "--shards-root", str(shards_root), "--domain-id", DOMAIN_ID,
        "--output-dir", str(out), "--months", "202401", *GATES, *extra,
    )


def _hist(out: Path) -> dict:
    return json.loads((out / "lifetime_hist.json").read_text(encoding="utf-8"))


def test_permutation_keeps_unavailable_slots() -> None:
    coverage = np.array([True, True, False, True, True, False, True])
    source = _time_shuffle_source_rows(coverage, 7)
    assert np.array_equal(source[~coverage], np.nonzero(~coverage)[0])
    assert sorted(source[coverage].tolist()) == np.nonzero(coverage)[0].tolist()
    assert np.array_equal(source, _time_shuffle_source_rows(coverage, 7))
    assert not np.array_equal(source, np.arange(coverage.size))


def test_real_long_track_shuffled_short(shards_root: Path, tmp_path: Path) -> None:
    real = _track(shards_root, tmp_path / "real")
    assert real.returncode == 0, real.stderr
    shuf = _track(shards_root, tmp_path / "shuf", "--shuffle-seed", "3")
    assert shuf.returncode == 0, shuf.stderr

    real_manifest = json.loads((tmp_path / "real" / "manifest.json").read_text(encoding="utf-8"))
    shuf_manifest = json.loads((tmp_path / "shuf" / "manifest.json").read_text(encoding="utf-8"))
    assert real_manifest["null_model"] is None
    assert shuf_manifest["null_model"] == {"type": "time_shuffle", "seed": 3}
    assert shuf_manifest["schema_version"] == "traps.period.track.v1"
    summary = json.loads((tmp_path / "shuf" / "tracking_summary.json").read_text(encoding="utf-8"))
    assert summary["null_model"] == {"type": "time_shuffle", "seed": 3}

    # 真實：長 track（跨過 1020 的單一缺失 frame）；置換：明顯較短。
    shuf_hist = _hist(tmp_path / "shuf")
    with np.load(tmp_path / "real" / "track_observations.npz") as archive:
        real_longest = int(np.bincount(archive["track_id"]).max())
    with np.load(tmp_path / "shuf" / "track_observations.npz") as archive:
        shuf_longest = int(np.bincount(archive["track_id"]).max())
    assert real_longest >= 10, real_longest
    assert shuf_longest < real_longest / 2

    # 可用性與時間軸完全相同；detection_keys 指向真實來源 snapshot（不必等於槽位）。
    with np.load(tmp_path / "real" / "canonical_snapshots.npz") as a, np.load(tmp_path / "shuf" / "canonical_snapshots.npz") as b:
        for name in a.files:
            assert np.array_equal(a[name], b[name])
        assert not bool(a["coverage_pass"][HOURS.index(1020)])
    with np.load(tmp_path / "shuf" / "track_observations.npz") as archive:
        slot = (archive["time_utc_ns"] - 1_704_067_200_000_000_000) // 3_600_000_000_000 - 1000
        assert not np.array_equal(archive["snapshot_index"], slot)  # 來源 snapshot != 槽位
        assert 1020 - 1000 not in set(archive["snapshot_index"].tolist())

    # 不同 seed 給不同結果、相同 seed 可重現。
    assert _track(shards_root, tmp_path / "shuf2", "--shuffle-seed", "3").returncode == 0
    assert _hist(tmp_path / "shuf2")["counts"] == shuf_hist["counts"]


def test_aggregate_refuses_null_input(shards_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "root" / DOMAIN_ID
    assert _track(shards_root, out, "--shuffle-seed", "1").returncode == 0
    with pytest.raises(ValueError, match="null_model"):
        agg._load_track_domain(tmp_path / "root", DOMAIN_ID)
    real = tmp_path / "root2" / DOMAIN_ID
    assert _track(shards_root, real).returncode == 0
    agg._load_track_domain(tmp_path / "root2", DOMAIN_ID)  # 真實輸出不受影響


def test_null_test_script_outputs(shards_root: Path, tmp_path: Path) -> None:
    for label, extra in {"real": [], "s1": ["--shuffle-seed", "1"], "s2": ["--shuffle-seed", "2"]}.items():
        assert _track(shards_root, tmp_path / label / DOMAIN_ID, *extra).returncode == 0
    out = tmp_path / "cmp"
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "null_test_tracking.py"), "--output-dir", str(out), "--runs",
         f"real={tmp_path / 'real'}", f"shuf1={tmp_path / 's1'}", f"shuf2={tmp_path / 's2'}"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "comparison.png").stat().st_size > 0
    rows = list(csv.DictReader((out / "comparison.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    by_run = {r["run"]: r for r in rows}
    assert by_run["real"]["shuffled"] == "False" and by_run["shuf1"]["shuffled"] == "True"
    assert by_run["shuf2"]["seed"] == "2"
    assert float(by_run["real"]["frac_ge3"]) >= float(by_run["shuf1"]["frac_ge3"])
    assert "coherence_ratio" in rows[0]
    assert "coherence" in result.stdout.lower()
