"""WP-B1 跨月份 ``detect-shards``／``track-period`` 的合成端到端測試。

fixture 建立三個連續月份（202401–202403）共用同一個小型（21x31）合成流場網格，
速度場是沿 x 方向緩慢平移的吸引鞍點，讓同一個 TRAP 結構理論上可以跨月份延續。
時間軸刻意安排：(a) 202401/202402 邊界重複一個 UTC 時刻、202402/202403 邊界也重複
一個，(b) 202402 內部有一個 5 小時缺口，(c) 202402 有一個 coverage 不足的時刻。
另建一個 ``trial_partial_month`` 月份，只用來驗證 PILOT_ONLY 與正式 shard 不可混用。

不連線 SERVER、不代表 2024–2025 正式 TRAP 統計；grid 與時刻數刻意壓到最小，讓整份
測試在數十秒內完成。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from test_cli_pipeline import PROJECT_ROOT, _run_cli, _write_json  # noqa: F401 - 重用既有 CLI subprocess helper

from traps.metrics import curve_intersects_mask
from traps.period import curve_pass_cells

DOMAIN_ID = "houwan_nmmba_cache_v3"
NY, NX = 21, 31
SPACING_M = 150.0
BASE_NS = 1_704_067_200_000_000_000  # 2024-01-01T00:00:00Z
HOUR_NS = 3_600_000_000_000
STEP_M_PER_HOUR = 15.0
SIGMA_M = 900.0

# 每個月份的絕對小時清單（相對 BASE_NS + 1000h 起算，避免與 0 混淆）；
# 202401/202402 邊界與 202402/202403 邊界各重複一個小時，202402 內部有 5 小時缺口，
# 202402 第 4 個時刻（絕對小時 1011）刻意 coverage 不足。
JAN_HOURS = [1000, 1001, 1002, 1003, 1004]
FEB_HOURS = [1004, 1005, 1010, 1011, 1012]
MAR_HOURS = [1012, 1013, 1014]
FEB_LOW_COVERAGE_HOUR = 1011


def _axes() -> tuple[np.ndarray, np.ndarray]:
    axis_x = (np.arange(NX, dtype=np.float64) - (NX - 1) / 2.0) * SPACING_M
    axis_y = (np.arange(NY, dtype=np.float64) - (NY - 1) / 2.0) * SPACING_M
    return axis_x, axis_y


def _velocity_field(x_m: np.ndarray, y_m: np.ndarray, hour: int) -> tuple[np.ndarray, np.ndarray]:
    """緩慢沿 x 平移的吸引鞍點；位置只由絕對小時決定，跨月份物理上連續一致。"""

    position_index = hour - 1000
    cx = position_index * STEP_M_PER_HOUR
    dx = x_m - cx
    dy = y_m
    envelope = np.exp(-(dx * dx + dy * dy) / SIGMA_M**2)
    u = -2.0e-3 * dx * envelope
    v = -1.0e-3 * dy * envelope
    return u.astype(np.float32), v.astype(np.float32)


def _write_shared_grid(root: Path) -> Path:
    """建立一次性共用 grid（所有月份與 ocean mask sidecar 都指向同一份）。"""

    grid_dir = root / "ocm_surface" / DOMAIN_ID / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    axis_x, axis_y = _axes()
    x_m, y_m = np.meshgrid(axis_x, axis_y, indexing="xy")
    static = np.ones((NY, NX), dtype=bool)
    lon_axis = 120.763161 + axis_x / 100_000.0
    lat_axis = 21.946577 + axis_y / 100_000.0
    cell_area = np.full((NY, NX), SPACING_M**2, dtype=np.float32)

    for name, array in {
        "lon.npy": lon_axis.astype(np.float64),
        "lat.npy": lat_axis.astype(np.float64),
        "x_m.npy": x_m.astype(np.float64),
        "y_m.npy": y_m.astype(np.float64),
        "cell_area_m2.npy": cell_area,
        "mask_static.npy": static,
    }.items():
        np.save(grid_dir / name, array, allow_pickle=False)
    _write_json(
        grid_dir / "metadata.json",
        {"cache_schema_version": "3.0.0", "domain": {"domain_id": DOMAIN_ID}, "grid_shape": [NY, NX]},
    )
    np.save(grid_dir / "analysis_ocean_mask_synthetic_v1.npy", static, allow_pickle=False)
    _write_json(
        grid_dir / "analysis_land_mask_synthetic_v1.json",
        {
            "cache_schema_version": "3.0.0",
            "domain": {"domain_id": DOMAIN_ID},
            "arrays": {"shape_lat_lon": [NY, NX]},
        },
    )
    return grid_dir


def _write_month(
    root: Path,
    *,
    month: str,
    hours: list[int],
    cache_kind: str = "standard_month",
    status: str = "ready",
    low_coverage_hours: frozenset[int] = frozenset(),
) -> Path:
    """寫出一個共用 grid 的月份目錄；``low_coverage_hours`` 內的小時故意壓低有效格點。"""

    month_dir = root / "ocm_surface" / DOMAIN_ID / "months" / month
    month_dir.mkdir(parents=True, exist_ok=True)
    axis_x, axis_y = _axes()
    x_m, y_m = np.meshgrid(axis_x, axis_y, indexing="xy")

    u_frames = []
    v_frames = []
    valid_frames = []
    for hour in hours:
        u, v = _velocity_field(x_m, y_m, hour)
        u_frames.append(u)
        v_frames.append(v)
        valid = np.ones((NY, NX), dtype=bool)
        if hour in low_coverage_hours:
            valid[:, :] = False
            valid[0, 0] = True  # 保留極少數有效格點，coverage 遠低於 0.95 門檻
        valid_frames.append(valid)

    u = np.stack(u_frames, axis=0)
    v = np.stack(v_frames, axis=0)
    valid = np.stack(valid_frames, axis=0)
    qc_flags = np.ones(u.shape, dtype=np.uint16)
    time = (BASE_NS + np.asarray(hours, dtype=np.int64) * HOUR_NS).astype(np.int64)

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
                "domain_id": DOMAIN_ID,
                "name_zh": "南灣合成海域",
                "lon_min": 120.16671,
                "lon_max": 121.62,
                "lat_min": 21.550844,
                "lat_max": 22.449156,
            },
        },
    )
    return month_dir


def _ocean_mask_path(root: Path) -> Path:
    return root / "ocm_surface" / DOMAIN_ID / "grid" / "analysis_ocean_mask_synthetic_v1.npy"


def _run_detect_shards(root: Path, *, month: str, output_dir: Path, extra: list[str] | None = None) -> "subprocess.CompletedProcess[str]":
    month_dir = root / "ocm_surface" / DOMAIN_ID / "months" / month
    arguments = [
        "detect-shards",
        "--month-dir",
        str(month_dir),
        "--ocean-mask",
        str(_ocean_mask_path(root)),
        "--output-dir",
        str(output_dir),
        "--min-coverage",
        "0.95",
    ]
    if extra:
        arguments.extend(extra)
    return _run_cli(*arguments)


@pytest.fixture()
def three_month_root(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    _write_shared_grid(root)
    _write_month(root, month="202401", hours=JAN_HOURS)
    _write_month(root, month="202402", hours=FEB_HOURS, low_coverage_hours=frozenset({FEB_LOW_COVERAGE_HOUR}))
    _write_month(root, month="202403", hours=MAR_HOURS)
    return root


def _shard_output(tmp_path: Path, month: str) -> Path:
    return tmp_path / "shards" / month


def _run_all_shards(root: Path, tmp_path: Path) -> None:
    for month in ("202401", "202402", "202403"):
        result = _run_detect_shards(root, month=month, output_dir=_shard_output(tmp_path, month))
        assert result.returncode == 0, result.stderr


def test_detect_shards_writes_traceable_and_resumable_artifacts(three_month_root: Path, tmp_path: Path) -> None:
    """單月 shard 產生完整 artifact，且 --resume 對未變更輸出是 no-op、對被竄改輸出會重做。"""

    output_dir = _shard_output(tmp_path, "202401")
    result = _run_detect_shards(three_month_root, month="202401", output_dir=output_dir)
    assert result.returncode == 0, result.stderr
    for name in ("snapshots.npz", "detections.npz", "grid_reference.npz", "manifest.json", "manifest.sha256", "_SUCCESS"):
        assert (output_dir / name).is_file(), name

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "traps.period.shard.v2"
    assert manifest["month"] == "202401"
    assert manifest["domain_id"] == DOMAIN_ID
    assert manifest["snapshot_count"] == len(JAN_HOURS)
    assert manifest["state_label"] == "READY"
    for artifact in manifest["artifacts"]:
        assert (output_dir / artifact["path"]).is_file()

    with np.load(output_dir / "snapshots.npz") as archive:
        assert not np.any(np.isnan(archive["coverage_fraction"]))
        assert archive["coverage_pass"].all()  # 202401 全部時刻 coverage 完整

    # 第二次執行不帶 --resume：既有內容視為衝突，直接拒絕。
    conflict = _run_detect_shards(three_month_root, month="202401", output_dir=output_dir)
    assert conflict.returncode != 0

    # --resume 對未變更輸出是 no-op：略過重算並回報 skipped。
    manifest_bytes_before = (output_dir / "manifest.json").read_bytes()
    resumed = _run_detect_shards(three_month_root, month="202401", output_dir=output_dir, extra=["--resume"])
    assert resumed.returncode == 0, resumed.stderr
    summary = json.loads(resumed.stdout)
    assert summary["skipped"] is True
    assert summary["resumed"] is True
    assert (output_dir / "manifest.json").read_bytes() == manifest_bytes_before

    # 竄改 detections.npz 後 --resume 必須偵測 SHA-256 不符並重做。
    detections_path = output_dir / "detections.npz"
    original_bytes = detections_path.read_bytes()
    tampered = bytearray(original_bytes)
    tampered[-1] ^= 0xFF
    detections_path.write_bytes(bytes(tampered))
    redone = _run_detect_shards(three_month_root, month="202401", output_dir=output_dir, extra=["--resume"])
    assert redone.returncode == 0, redone.stderr
    redone_summary = json.loads(redone.stdout)
    assert redone_summary["skipped"] is False
    rewritten_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    rewritten_record = next(item for item in rewritten_manifest["artifacts"] if item["path"] == "detections.npz")
    assert rewritten_record["sha256"] != _sha256_bytes(bytes(tampered))
    import hashlib

    assert rewritten_record["sha256"] == hashlib.sha256(detections_path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def test_low_coverage_snapshot_is_flagged_not_dropped(three_month_root: Path, tmp_path: Path) -> None:
    """coverage 不足的時刻仍保留一列，且 coverage_pass 明確為 False。"""

    output_dir = _shard_output(tmp_path, "202402")
    result = _run_detect_shards(three_month_root, month="202402", output_dir=output_dir)
    assert result.returncode == 0, result.stderr
    with np.load(output_dir / "snapshots.npz") as archive:
        hours = (archive["time_utc_ns"] - BASE_NS) // HOUR_NS
        index = int(np.nonzero(hours == FEB_LOW_COVERAGE_HOUR)[0][0])
        assert archive["coverage_pass"][index] == False  # noqa: E712
        assert archive["coverage_fraction"][index] < 0.95
        assert not np.any(np.isnan(archive["coverage_fraction"]))


def test_track_period_canonical_axis_and_cross_month_tracking(three_month_root: Path, tmp_path: Path) -> None:
    """canonical 去重保留較晚月份、跨月份 track 延續、時間缺口右截尾。"""

    _run_all_shards(three_month_root, tmp_path)
    shards_root = tmp_path / "shards"
    track_output = tmp_path / "track"
    result = _run_cli(
        "track-period",
        "--shards-root",
        str(shards_root),
        "--domain-id",
        DOMAIN_ID,
        "--output-dir",
        str(track_output),
        "--months",
        "202401-202403",
    )
    assert result.returncode == 0, result.stderr

    manifest = json.loads((track_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "traps.period.track.v1"
    axis = manifest["canonical_axis"]
    assert axis["total_rows"] == len(JAN_HOURS) + len(FEB_HOURS) + len(MAR_HOURS)
    assert axis["unique_times"] == axis["total_rows"] - 2
    assert axis["duplicates_removed_count"] == 2
    assert axis["gap_count"] == 1
    assert axis["max_gap_hours"] == pytest.approx(5.0)

    with np.load(track_output / "canonical_snapshots.npz") as archive:
        assert not np.any(np.isnan(archive["time_utc_ns"].astype(np.float64)))
        month_by_hour = {
            int((time_ns - BASE_NS) // HOUR_NS): int(month)
            for time_ns, month in zip(archive["time_utc_ns"], archive["month"])
        }
    # 重複時刻必須保留「較晚月份」那一列。
    assert month_by_hour[1004] == 202402
    assert month_by_hour[1012] == 202403
    assert set(month_by_hour) == {1000, 1001, 1002, 1003, 1004, 1005, 1010, 1011, 1012, 1013, 1014}

    tracks = _read_tracks_csv(track_output / "tracks.csv")
    with np.load(track_output / "track_observations.npz") as archive:
        obs_track_id = archive["track_id"]
        obs_time_ns = archive["time_utc_ns"]
        assert not np.any(np.isnan(archive["core_x_m"]))
        assert not np.any(np.isnan(archive["core_s1_sinv"]))

    def _hours_for_track(track_id: int) -> set[int]:
        mask = obs_track_id == track_id
        return {int((value - BASE_NS) // HOUR_NS) for value in obs_time_ns[mask]}

    # 找出橫跨 202401→202402 邊界（同時含小時 1000 與 1005）的 track：必須是同一條，
    # 且因 5 小時缺口而以 time_gap 右截尾（左邊界也應標左截尾，因為它是序列起點）。
    boundary_track = next(row for row in tracks if 1000 in _hours_for_track(int(row["track_id"])) and 1005 in _hours_for_track(int(row["track_id"])))
    assert _hours_for_track(int(boundary_track["track_id"])) == {1000, 1001, 1002, 1003, 1004, 1005}
    assert boundary_track["termination_reason"] == "time_gap"
    assert boundary_track["right_censored"] == "True"
    assert boundary_track["left_censored"] == "True"

    # 缺口之後另一條 track 應延續穿越 202402→202403 邊界，跨過 coverage 不足的 1011
    # （單一缺失 frame 允許 gap-closing），直到研究期結束。
    post_gap_track = next(row for row in tracks if 1010 in _hours_for_track(int(row["track_id"])))
    post_gap_hours = _hours_for_track(int(post_gap_track["track_id"]))
    assert post_gap_hours == {1010, 1012, 1013, 1014}
    assert post_gap_track["termination_reason"] == "study_period_end"
    assert post_gap_track["right_censored"] == "True"
    assert "left_censored_after_gap" in post_gap_track["quality_flags"]

    tracking_summary = json.loads((track_output / "tracking_summary.json").read_text(encoding="utf-8"))
    assert tracking_summary["termination_reason_counts"].get("time_gap", 0) >= 1
    assert tracking_summary["termination_reason_counts"].get("study_period_end", 0) >= 1
    assert tracking_summary["censored_track_count"] >= 2


def _read_tracks_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_track_period_resume_is_noop_then_redo_after_tamper(three_month_root: Path, tmp_path: Path) -> None:
    """track-period 的 --resume 同樣遵守「已驗證則略過、被竄改則重做」契約。"""

    _run_all_shards(three_month_root, tmp_path)
    shards_root = tmp_path / "shards"
    track_output = tmp_path / "track"
    base_arguments = [
        "track-period",
        "--shards-root",
        str(shards_root),
        "--domain-id",
        DOMAIN_ID,
        "--output-dir",
        str(track_output),
        "--months",
        "202401-202403",
    ]
    first = _run_cli(*base_arguments)
    assert first.returncode == 0, first.stderr

    no_resume = _run_cli(*base_arguments)
    assert no_resume.returncode != 0

    manifest_before = (track_output / "manifest.json").read_bytes()
    resumed = _run_cli(*base_arguments, "--resume")
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["skipped"] is True
    assert (track_output / "manifest.json").read_bytes() == manifest_before

    tracks_path = track_output / "tracks.csv"
    tampered = tracks_path.read_bytes() + b"\n# tampered"
    tracks_path.write_bytes(tampered)
    redone = _run_cli(*base_arguments, "--resume")
    assert redone.returncode == 0, redone.stderr
    assert json.loads(redone.stdout)["skipped"] is False
    assert tracks_path.read_bytes() != tampered


def test_track_period_refuses_missing_month_without_flag(three_month_root: Path, tmp_path: Path) -> None:
    """未提供 --allow-missing-months 時，缺少的請求月份必須直接拒絕整次執行。"""

    output_dir = _shard_output(tmp_path, "202401")
    result = _run_detect_shards(three_month_root, month="202401", output_dir=output_dir)
    assert result.returncode == 0, result.stderr

    shards_root = tmp_path / "shards"
    track_output = tmp_path / "track_missing"
    rejected = _run_cli(
        "track-period",
        "--shards-root",
        str(shards_root),
        "--domain-id",
        DOMAIN_ID,
        "--output-dir",
        str(track_output),
        "--months",
        "202401-202403",
    )
    assert rejected.returncode != 0
    assert "202402" in rejected.stderr

    allowed = _run_cli(
        "track-period",
        "--shards-root",
        str(shards_root),
        "--domain-id",
        DOMAIN_ID,
        "--output-dir",
        str(track_output),
        "--months",
        "202401-202403",
        "--allow-missing-months",
    )
    assert allowed.returncode == 0, allowed.stderr
    manifest = json.loads((track_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_months"] == ["202402", "202403"]
    assert manifest["available_months"] == ["202401"]


def test_track_period_refuses_mixing_pilot_and_formal_shards(three_month_root: Path, tmp_path: Path) -> None:
    """PILOT_ONLY 與正式 shard 不可在同一次 track-period 混用。"""

    _write_month(
        three_month_root,
        month="202404",
        hours=[2000, 2001],
        cache_kind="trial_partial_month",
        status="trial_ready",
    )
    formal_output = _shard_output(tmp_path, "202401")
    result = _run_detect_shards(three_month_root, month="202401", output_dir=formal_output)
    assert result.returncode == 0, result.stderr

    pilot_output = _shard_output(tmp_path, "202404")
    pilot_result = _run_detect_shards(
        three_month_root, month="202404", output_dir=pilot_output, extra=["--allow-trial-pilot"]
    )
    assert pilot_result.returncode == 0, pilot_result.stderr
    pilot_manifest = json.loads((pilot_output / "manifest.json").read_text(encoding="utf-8"))
    assert pilot_manifest["state_label"] == "PILOT_ONLY"

    mixed = _run_cli(
        "track-period",
        "--shards-root",
        str(tmp_path / "shards"),
        "--domain-id",
        DOMAIN_ID,
        "--output-dir",
        str(tmp_path / "track_mixed"),
        "--month",
        "202401",
        "202404",
    )
    assert mixed.returncode != 0
    assert "PILOT_ONLY" in mixed.stderr


def test_curve_pass_cells_matches_curve_intersects_mask() -> None:
    """``curve_pass_cells`` 的格點集合須與 ``curve_intersects_mask`` 的判定完全等價。

    對隨機曲線與隨機遮罩，``curve_intersects_mask(curve, x, y, mask)`` 恆等於
    ``np.any(mask.flat[curve_pass_cells(curve, x, y, mask.shape)])``；同時涵蓋遞增與
    遞減 y 軸兩種常見網格慣例。
    """

    rng = np.random.default_rng(20240101)
    ny, nx = 9, 13
    x_axis = np.linspace(0.0, 1200.0, nx)
    for y_increasing in (True, False):
        y_axis = np.linspace(0.0, 800.0, ny) if y_increasing else np.linspace(800.0, 0.0, ny)
        x_m, y_m = np.meshgrid(x_axis, y_axis, indexing="xy")
        for _ in range(30):
            mask = rng.random((ny, nx)) < 0.2
            point_count = int(rng.integers(2, 6))
            curve = np.column_stack(
                (
                    rng.uniform(-100.0, 1300.0, point_count),
                    rng.uniform(-100.0, 900.0, point_count),
                )
            )
            expected = curve_intersects_mask(curve, x_m, y_m, mask)
            cells = curve_pass_cells(curve, x_m, y_m, mask.shape)
            actual = bool(np.any(mask.flat[cells])) if cells.size else False
            assert actual == expected


# ---------------------------------------------------------------------------
# WP-I：近岸距離場與 shard schema v2
# ---------------------------------------------------------------------------


def _brute_force_coast_distance(mask: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, cap: float) -> np.ndarray:
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    land = np.argwhere(~mask)
    result = np.full(mask.shape, cap)
    for row, col in np.argwhere(mask):
        if land.size:
            distance = np.hypot(xx[land[:, 0], land[:, 1]] - xx[row, col], yy[land[:, 0], land[:, 1]] - yy[row, col]).min()
            result[row, col] = min(distance, cap)
    result[~mask] = 0.0
    return result


def test_coast_distance_field_on_synthetic_mask_matches_brute_force() -> None:
    from traps.period import compute_coast_distance_field

    ny, nx = 14, 17
    x_axis = np.arange(nx) * 1000.0
    y_axis = np.arange(ny) * 1000.0
    mask = np.ones((ny, nx), dtype=bool)
    mask[:, 0] = False  # 西岸一整欄
    mask[6, 9] = False  # 孤島一格
    field = compute_coast_distance_field(mask, x_axis, y_axis)
    assert field.dtype == np.float32 and field.shape == mask.shape
    assert field[6, 9] == 0.0 and np.all(field[:, 0] == 0.0)
    assert field[6, 8] == pytest.approx(1000.0) and field[6, 10] == pytest.approx(1000.0)
    assert field[7, 10] == pytest.approx(np.hypot(1000.0, 1000.0), rel=1e-6)  # 對角格：最近非海域格點為孤島
    assert field[0, 3] == pytest.approx(3000.0)  # 只受西岸影響
    assert np.allclose(field, _brute_force_coast_distance(mask, x_axis, y_axis, 30_000.0), atol=1e-2)

    # 隨機遮罩、非等距且遞減的 y 軸；小 cap 會使遠處飽和（不是陸地）
    rng = np.random.default_rng(3)
    random_mask = rng.random((12, 15)) > 0.15
    x_random = np.cumsum(rng.uniform(800.0, 1200.0, 15))
    y_random = -np.cumsum(rng.uniform(800.0, 1200.0, 12))
    for cap in (30_000.0, 2_500.0):
        got = compute_coast_distance_field(random_mask, x_random, y_random, cap_m=cap)
        assert np.allclose(got, _brute_force_coast_distance(random_mask, x_random, y_random, cap), atol=1e-2)


def test_coast_distance_ignores_cells_outside_grid_and_handles_no_land() -> None:
    from traps.period import compute_coast_distance_field

    axis = np.arange(6) * 500.0
    all_ocean = compute_coast_distance_field(np.ones((6, 6), dtype=bool), axis, axis, cap_m=20_000.0)
    assert np.all(all_ocean == 20_000.0)  # 格網外不算陸地，也就沒有海岸
    edge = np.ones((6, 6), dtype=bool)
    edge[:, -1] = False
    assert compute_coast_distance_field(edge, axis, axis)[3, 0] == pytest.approx(2500.0)


def test_sample_coast_distance_bilinear_decreasing_axis_and_outside() -> None:
    from traps.period import sample_coast_distance

    x_axis = np.array([0.0, 1000.0, 2000.0])
    y_axis = np.array([2000.0, 1000.0, 0.0])  # 遞減
    field = np.array([[0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0], [2000.0, 2000.0, 2000.0]])
    values = sample_coast_distance(field, x_axis, y_axis, np.array([500.0, 1500.0, 2500.0, 0.0]), np.array([1500.0, 500.0, 500.0, 0.0]))
    assert values[0] == pytest.approx(500.0) and values[1] == pytest.approx(1500.0)
    assert np.isnan(values[2])  # 格網外
    assert values[3] == pytest.approx(2000.0)


def _downgrade_shard_to_v1(shard_dir: Path) -> None:
    """把 v2 shard 改寫成 v1 布局（移除新欄位與距離場、去掉 detection_config 新鍵），重寫 manifest。"""

    from traps.period import _SHARD_ARTIFACT_NAMES, _artifact_records, _write_manifest_with_checksum

    detections_path = shard_dir / "detections.npz"
    with np.load(detections_path) as archive:
        kept = {name: np.array(archive[name]) for name in archive.files if name not in ("core_grid_x_m", "core_grid_y_m", "core_refined", "core_coast_distance_m")}
    np.savez(detections_path, **kept)
    grid_path = shard_dir / "grid_reference.npz"
    with np.load(grid_path) as archive:
        grid = {name: np.array(archive[name]) for name in archive.files if name != "coast_distance_m"}
    np.savez(grid_path, **grid)
    manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "traps.period.shard.v1"
    for key in ("core_subgrid", "coast_distance"):
        manifest.pop(key, None)
    for key in ("core_subgrid", "core_subgrid_refine"):
        manifest["detection_config"].pop(key, None)
    manifest["artifacts"] = _artifact_records(shard_dir, _SHARD_ARTIFACT_NAMES)
    _write_manifest_with_checksum(shard_dir, manifest)


def test_detect_shards_v2_columns_and_track_period_coast_distance(three_month_root: Path, tmp_path: Path) -> None:
    from traps.period import sample_coast_distance

    _run_all_shards(three_month_root, tmp_path)
    shard_dir = _shard_output(tmp_path, "202401")
    manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["detection_config"]["core_subgrid"] == "quadratic_3x3"
    assert manifest["detection_config"]["core_subgrid_refine"] is True
    assert manifest["core_subgrid"] == "quadratic_3x3"
    with np.load(shard_dir / "detections.npz") as archive, np.load(shard_dir / "grid_reference.npz") as grid:
        assert archive["core_x_m"].size > 0
        for name in ("core_grid_x_m", "core_grid_y_m", "core_refined", "core_coast_distance_m"):
            assert archive[name].shape == archive["core_x_m"].shape, name
        assert archive["core_refined"].dtype == bool and archive["core_refined"].any()
        coast = grid["coast_distance_m"]
        assert coast.shape == grid["analysis_ocean_mask"].shape
        expected = sample_coast_distance(coast, grid["x_m_axis"], grid["y_m_axis"], archive["core_x_m"], archive["core_y_m"])
        assert np.allclose(archive["core_coast_distance_m"], expected, equal_nan=False)
        # 整格網都是海域（合成遮罩）：沒有非海域格點 → 飽和上限
        assert np.all(coast == 30_000.0) == bool(np.all(grid["analysis_ocean_mask"]))
        # 細化位置相對格點節點的偏移在 ±0.5 格內；未細化者等於節點
        spacing = float(np.median(np.diff(grid["x_m_axis"])))
        offset = np.hypot(archive["core_x_m"] - archive["core_grid_x_m"], archive["core_y_m"] - archive["core_grid_y_m"])
        assert np.all(offset <= 0.5 * spacing * np.sqrt(2.0) + 1e-6)
        assert np.all(offset[~archive["core_refined"]] == 0.0)
        assert np.any(offset[archive["core_refined"]] > 0.0)  # 不再全落在 1 格節點上

    track_output = tmp_path / "track"
    result = _run_cli("track-period", "--shards-root", str(tmp_path / "shards"), "--domain-id", DOMAIN_ID, "--output-dir", str(track_output), "--months", "202401-202403")
    assert result.returncode == 0, result.stderr
    with np.load(track_output / "track_observations.npz") as observations:
        assert observations["core_coast_distance_m"].shape == observations["core_x_m"].shape
        assert np.all(np.isfinite(observations["core_coast_distance_m"]))
        # 追蹤用細化核心：至少有一個核心座標不落在格點節點
        with np.load(shard_dir / "grid_reference.npz") as grid:
            x_axis = grid["x_m_axis"]
        assert np.any(np.min(np.abs(observations["core_x_m"][:, None] - x_axis[None, :]), axis=1) > 1e-6)


def test_track_period_and_aggregate_accept_v1_shards_and_refuse_mixed(three_month_root: Path, tmp_path: Path) -> None:
    _run_all_shards(three_month_root, tmp_path)
    for month in ("202401", "202402", "202403"):
        _downgrade_shard_to_v1(_shard_output(tmp_path, month))
    track_output = tmp_path / "track_v1"
    result = _run_cli("track-period", "--shards-root", str(tmp_path / "shards"), "--domain-id", DOMAIN_ID, "--output-dir", str(track_output), "--months", "202401-202403")
    assert result.returncode == 0, result.stderr  # v1：核心取格點節點，近岸距離由遮罩補算
    with np.load(track_output / "track_observations.npz") as observations:
        assert np.all(np.isfinite(observations["core_coast_distance_m"]))

    # 混用 v1 與 v2 → detection_config 不一致，明確拒絕
    mixed_root = tmp_path / "mixed"
    for month in ("202401", "202402", "202403"):
        result = _run_detect_shards(three_month_root, month=month, output_dir=mixed_root / month)
        assert result.returncode == 0, result.stderr
    _downgrade_shard_to_v1(mixed_root / "202401")
    refused = _run_cli("track-period", "--shards-root", str(mixed_root), "--domain-id", DOMAIN_ID, "--output-dir", str(tmp_path / "track_mixed"), "--months", "202401-202403")
    assert refused.returncode != 0 and "detection_config" in (refused.stderr + refused.stdout)

    # 不受支援的 schema 版本
    manifest_path = mixed_root / "202402" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "traps.period.shard.v0"
    from traps.period import _write_manifest_with_checksum

    _write_manifest_with_checksum(mixed_root / "202402", payload)
    bad = _run_cli("track-period", "--shards-root", str(mixed_root), "--domain-id", DOMAIN_ID, "--output-dir", str(tmp_path / "track_bad"), "--months", "202402")
    assert bad.returncode != 0 and "schema_version" in (bad.stderr + bad.stdout)
