"""WP-H：代表 TRAP、方法比較、長壽命 track、track 通過數與傳播速度圖的單元與合成端到端測試。"""

from __future__ import annotations

import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_aggregate import (  # noqa: F401
    FAKE_FONT,
    _manifest,
    _namespace,
    _synthetic_figure_data,
    built_run,
    fake_font,
)

from traps import aggregate as agg
from traps import aggregate_figures, example_figures
from traps.report_font import FIGURE_TEXT, REQUIRED_FIGURE_GLYPHS

HOUR_NS = 3_600_000_000_000


def test_select_representative_track_rules() -> None:
    cands = [
        {"track_id": 1, "lifetime_hours": 10.0, "mean_core_s1_sinv": -1e-5},
        {"track_id": 2, "lifetime_hours": 30.0, "mean_core_s1_sinv": -1e-5},
        {"track_id": 3, "lifetime_hours": 30.0, "mean_core_s1_sinv": -3e-5},
        {"track_id": 4, "lifetime_hours": 26.0, "mean_core_s1_sinv": -9e-5},
    ]
    chosen, criterion = agg.select_representative_track(cands, 24.0)
    assert chosen["track_id"] == 3 and criterion == "ge_threshold"  # 同壽命取核心最強
    chosen, criterion = agg.select_representative_track(cands[:1], 24.0)
    assert chosen["track_id"] == 1 and criterion == "longest_available"
    assert agg.select_representative_track([], 24.0) == (None, "no_candidate")


def test_fit_track_velocity_exact_on_constant_drift() -> None:
    times = np.arange(6, dtype=np.int64) * HOUR_NS
    x = 1000.0 * np.arange(6)  # 1 km/h 東向 = 24 km/day
    y = -500.0 * np.arange(6)  # 0.5 km/h 南向 = -12 km/day
    fit = agg.fit_track_velocity(times, x, y)
    assert fit is not None
    assert fit.cx_kmday == pytest.approx(24.0) and fit.cy_kmday == pytest.approx(-12.0)
    assert fit.resid_rms_m == pytest.approx(0.0, abs=1e-9)
    assert fit.displacement_m == pytest.approx(np.hypot(1000.0, 500.0) * 5)  # 擬合速度 × 觀測時間跨度
    assert fit.n_obs == 6
    assert agg.fit_track_velocity(times[:1], x[:1], y[:1]) is None
    assert agg.fit_track_velocity(np.zeros(3, dtype=np.int64), x[:3], y[:3]) is None  # 時間跨度為零


def test_fit_track_velocity_robust_to_half_cell_jitter_and_uses_actual_time() -> None:
    """位置抖動 ≤ 半格（500 m，格距 1 km）時，整條 track 的擬合速度仍逼近真值；殘差 RMS 反映抖動量。"""

    hours = np.arange(48)
    times = hours.astype(np.int64) * HOUR_NS
    jitter = 400.0 * np.where(hours % 2 == 0, 1.0, -1.0)
    x = 125.0 * hours + jitter  # 3 km/day
    y = -50.0 * hours - jitter
    fit = agg.fit_track_velocity(times, x, y)
    assert fit.cx_kmday == pytest.approx(3.0, rel=0.03) and fit.cy_kmday == pytest.approx(-1.2, rel=0.03)
    assert 300.0 < fit.resid_rms_m < 600.0
    # 跨時間缺口：用實際 UTC 差而不是觀測序號
    gap_times = np.array([0, 1, 3], dtype=np.int64) * HOUR_NS
    gap_fit = agg.fit_track_velocity(gap_times, np.array([0.0, 1000.0, 3000.0]), np.zeros(3))
    assert gap_fit.cx_kmday == pytest.approx(1000.0 / 3600.0 * 86.4)


def test_stationary_classification_and_propagation_columns() -> None:
    """擬合位移 < 1 格為靜止：只有移動 track 進入速度統計；靜止比例與殘差 RMS 另列。"""

    shape = (3, 3)
    grid = {"shape": shape, "x_m_axis": np.array([0.0, 1000.0, 2000.0]), "y_m_axis": np.array([0.0, 1000.0, 2000.0])}
    site = np.ones(shape, dtype=bool)
    t0 = 1_704_100_000_000_000_000
    hours = np.arange(6)
    # track 1：只有 ±100 m 抖動（靜止）；track 2：東向 100 m/h（600 m/day... 5 h 位移 500 m 仍靜止）；track 3：東向 400 m/h（位移 2 km 移動）
    observations = {
        "track_id": np.repeat([1, 2, 3], 6).astype(np.int64),
        "time_utc_ns": np.tile(hours, 3).astype(np.int64) * HOUR_NS + t0,
        "core_x_m": np.concatenate([1000.0 + 100.0 * (-1.0) ** hours, 100.0 * hours, 400.0 * hours]),
        "core_y_m": np.concatenate([np.full(6, 1000.0), np.full(6, 500.0), np.full(6, 700.0)]),
        "core_s1_sinv": np.full(18, -1e-5),
    }
    tracks = {i: {"left_censored": False, "right_censored": False, "lifetime_hours": 6.0, "start_time_utc_ns": t0} for i in (1, 2, 3)}
    acc: dict = defaultdict(agg._new_site_accumulator)
    agg._process_track_lifetimes(
        context={"observations": observations, "tracks_by_id": tracks}, site_masks_2d={"s": site}, grid=grid,
        site_accumulators=acc, spatial_accumulators={"s": {}},
    )
    fitted = np.asarray(acc[("s", "all")]["prop_tracks"])
    assert fitted.shape == (3, 5) and fitted[:, 4].tolist() == [1.0, 1.0, 0.0]
    record = agg._finalize_site_record(
        "s", "all", acc[("s", "all")], site_area_km2=1.0, site_name_zh="s", strong_threshold_sinv=None, hotspot_metrics={},
        bootstrap_reps=5, rng=np.random.default_rng(0),
    )
    assert record["prop_n_tracks"] == 3 and record["prop_n_moving"] == 1 and record["prop_n_stationary"] == 2
    assert record["prop_stationary_frac"] == pytest.approx(2 / 3)
    assert record["prop_cx_median_kmday"] == pytest.approx(400.0 * 24.0 / 1000.0)  # 只含移動 track
    assert record["prop_speed_median_kmday"] == pytest.approx(9.6) and record["prop_speed_p90_kmday"] == pytest.approx(9.6)
    assert record["prop_resid_rms_median_m"] is not None
    other = agg._finalize_site_record(
        "s", "DJF", acc[("s", "DJF")], site_area_km2=1.0, site_name_zh="s", strong_threshold_sinv=None, hotspot_metrics={},
        bootstrap_reps=5, rng=np.random.default_rng(0),
    )
    assert other["prop_n_tracks"] is None and other["prop_stationary_frac"] is None
    assert set(agg.PROPAGATION_COLUMNS) <= set(agg.SITE_STAT_FIELDNAMES) and "prop_n" not in agg.SITE_STAT_FIELDNAMES

    # 觀測數不足 max(3, --prop-min-obs) 的 track 不擬合
    acc2: dict = defaultdict(agg._new_site_accumulator)
    agg._process_track_lifetimes(
        context={"observations": observations, "tracks_by_id": tracks}, site_masks_2d={"s": site}, grid=grid,
        site_accumulators=acc2, spatial_accumulators={"s": {}}, prop_min_obs=7,
    )
    assert acc2[("s", "all")]["prop_tracks"] == []


def test_distinct_track_count_columns() -> None:
    shape = (3, 3)
    grid = {"shape": shape, "x_m_axis": np.array([-1.0, 0.0, 1.0]), "y_m_axis": np.array([-1.0, 0.0, 1.0])}
    site = np.ones(shape, dtype=bool)
    t0 = 1_704_100_000_000_000_000
    # track 1 兩次經過中心格（只計一次）；track 2 經過中心格與 (0,0)
    observations = {
        "track_id": np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        "time_utc_ns": np.array([0, 1, 2, 0, 1, 2], dtype=np.int64) * HOUR_NS + t0,
        "core_x_m": np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0]),
        "core_y_m": np.array([0.0, 0.1, 0.0, -1.0, 0.0, 1.0]),
        "core_s1_sinv": np.full(6, -1e-5),
    }
    tracks = {
        1: {"left_censored": False, "right_censored": False, "lifetime_hours": 5.0, "start_time_utc_ns": t0},
        2: {"left_censored": False, "right_censored": True, "lifetime_hours": 9.0, "start_time_utc_ns": t0},
    }
    acc: dict = defaultdict(agg._new_site_accumulator)
    spatial: dict = {"s": {}}
    agg._process_track_lifetimes(
        context={"observations": observations, "tracks_by_id": tracks}, site_masks_2d={"s": site}, grid=grid,
        site_accumulators=acc, spatial_accumulators=spatial, long_track_hours=4.0,
    )
    stratum = spatial["s"]["DJF"]
    assert stratum["track_count"][1, 1] == 2 and stratum["track_count"][0, 0] == 1
    assert stratum["long_track_count"][1, 1] == 1 and stratum["long_track_count"][0, 0] == 0  # 截尾 track 2 不算長壽命
    assert len(acc[("s", "all")]["prop_tracks"]) == 2  # 兩條 track 各一筆擬合


def test_verify_recomputed_core_pass_and_failure() -> None:
    det = SimpleNamespace(core_index_ij=(5, 7), is_qualified=True)
    assert example_figures.verify_recomputed_core([det], 0, 5, 8) == 1
    with pytest.raises(ValueError, match="超過容許"):
        example_figures.verify_recomputed_core([det], 0, 5, 9)
    with pytest.raises(ValueError, match="找不到"):
        example_figures.verify_recomputed_core([det], 3, 5, 7)
    with pytest.raises(ValueError, match="不是合格"):
        example_figures.verify_recomputed_core([SimpleNamespace(core_index_ij=(5, 7), is_qualified=False)], 0, 5, 7)


def test_pick_later_observations_and_labels() -> None:
    times = np.arange(10, dtype=np.int64) * HOUR_NS
    assert example_figures.pick_later_observations(times) == [3, 6, 9]
    assert example_figures.pick_later_observations(times[:2]) == [1]
    assert example_figures.relative_time_label(6 * 3600.0, day_unit=False) == "+6 h"
    assert example_figures.relative_time_label(3 * 86400.0, day_unit=True) == "+3 天"
    assert example_figures.uses_day_unit(24.0) and not example_figures.uses_day_unit(1.0)


def test_rose_histogram_excludes_zero_speed_from_north_bin() -> None:
    """零速度（atan2(0, 0) 被歸到正北）沒有方向，明確排除，也不進分母。"""

    cx = np.array([0.0, 0.0, 0.0, 10.0, np.nan])
    cy = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    centres, freq, speed = example_figures.rose_histogram(cx, cy)
    assert freq[0] == 0.0 and np.isnan(speed[0])  # 北扇區沒有被零速度灌水
    assert freq[4] == pytest.approx(1.0) and speed[4] == pytest.approx(10.0)
    assert freq.sum() == pytest.approx(1.0)
    _, empty_freq, empty_speed = example_figures.rose_histogram(np.zeros(4), np.zeros(4))
    assert empty_freq.sum() == 0.0 and np.all(np.isnan(empty_speed))


def test_rose_histogram() -> None:
    centres, freq, speed = example_figures.rose_histogram(np.array([0.0, 0.0, 10.0]), np.array([5.0, 5.0, 0.0]))
    assert centres.size == 16 and freq.sum() == pytest.approx(1.0)
    assert freq[0] == pytest.approx(2 / 3) and freq[4] == pytest.approx(1 / 3)  # 北、東
    assert speed[0] == pytest.approx(5.0) and speed[4] == pytest.approx(10.0)


@pytest.mark.filterwarnings("ignore")
def test_aggregate_end_to_end_wp_h(built_run, tmp_path: Path, fake_font) -> None:  # noqa: ANN001
    root, track_root, detect_root = built_run
    out = tmp_path / "agg"
    moved = tmp_path / "moved_surface"
    shutil.copytree(root / "ocm_surface", moved)
    ns = _namespace(track_root, detect_root, out, long_track_hours=1.0, surface_root=moved, no_example_figures=False)
    assert agg.run_aggregate(ns) == 0
    for site in ("nanwan", "gongliao", "guishan"):
        for kind in ("example", "method_comparison", "long_tracks", "track_density", "propagation"):
            assert (out / "figures" / site / f"site_{site}_{kind}.png").stat().st_size > 1_000, (site, kind)
        archive = np.load(out / "spatial" / site / "spatial_all.npz")
        assert {"track_count", "long_track_count"} <= set(archive.files)
    manifest = _manifest(out)
    examples = {site: info["example"] for site, info in manifest["site_track_figures"].items()}
    ok = [info for info in examples.values() if info["status"] == "ok"]
    assert ok and all(info["recomputed_core_offset_cells"] <= 1 for info in ok)
    assert all(str(moved) in info["source_month_dir"] for info in ok)  # 使用覆寫的 --surface-root
    assert any(info["status"] == "no_eligible_track" for info in examples.values())
    assert manifest["long_track_hours"] == 1.0
    with (out / "site_statistics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    all_rows = [r for r in rows if r["stratum"] == "all"]
    assert all(r["prop_n_tracks"] != "" for r in all_rows) and all(r["prop_n_tracks"] == "" for r in rows if r["stratum"] != "all")
    assert any(int(r["prop_n_tracks"]) > 0 for r in all_rows)
    assert all(r["prop_stationary_frac"] != "" for r in all_rows if int(r["prop_n_tracks"]) > 0)
    assert manifest["coastal_buffer_km"] == 10.0 and manifest["propagation"]["min_observations"] == 3
    assert manifest["core_subgrid"] == {"method": "quadratic_3x3", "refine": True}
    for site in ("nanwan", "gongliao", "guishan"):
        archive = np.load(out / "spatial" / site / "spatial_all.npz")
        assert {"coast_distance_m", "coast_buffer_mask", "coast_buffer_m"} <= set(archive.files)
        assert (out / "figures" / site / f"site_{site}_coastal_diagnostic.png").stat().st_size > 1_000

    # --no-example-figures：只略過兩張快照圖
    out2 = tmp_path / "agg2"
    assert agg.run_aggregate(_namespace(track_root, detect_root, out2, no_example_figures=True)) == 0
    assert not (out2 / "figures" / "guishan" / "site_guishan_example.png").exists()
    assert (out2 / "figures" / "guishan" / "site_guishan_propagation.png").is_file()

    # 來源被更動（海岸遮罩 SHA-256 不符）時明確失敗
    mask = next((moved / "northeast_taiwan_common_cache_v3" / "grid").glob("analysis_ocean_mask_*.npy"))
    np.save(mask, ~np.load(mask))
    with pytest.raises(ValueError, match="SHA-256"):
        agg.run_aggregate(_namespace(track_root, detect_root, tmp_path / "agg3", surface_root=moved))


# ---------------------------------------------------------------------------
# 版面與字元
# ---------------------------------------------------------------------------


def _synthetic_snapshot(data: aggregate_figures.SiteFigureData) -> example_figures.ExampleSnapshot:
    ny, nx = data.site_mask.shape
    x = np.arange(nx) * 500.0
    y = np.arange(ny) * 500.0
    xx, yy = np.meshgrid(x, y)
    u = -2e-4 * (xx - x[8]) * np.exp(-((xx - x[8]) ** 2 + (yy - y[9]) ** 2) / 4e6)
    v = -2e-4 * (yy - y[9]) * np.exp(-((xx - x[8]) ** 2 + (yy - y[9]) ** 2) / 4e6)
    curve = np.column_stack([np.full(9, x[8]), np.linspace(y[6], y[12], 9)])
    later = [
        example_figures.LaterObservation(dt_seconds=k * 6 * 3600.0, curve_m=curve + [500.0 * k, 0.0], core_m=(x[8] + 500.0 * k, y[9]))
        for k in (1, 2, 3)
    ]
    return example_figures.ExampleSnapshot(
        track_id=7, time_utc_ns=1_704_067_200_000_000_000, month="202401", snapshot_index=0, detection_index=0,
        lifetime_hours=18.0, n_observations=4, criterion="longest_available", core_offset_cells=0, x_m_axis=x, y_m_axis=y,
        u=u, v=v, valid=np.ones((ny, nx), bool), s1_sinv=-np.exp(-((xx - x[8]) ** 2 + (yy - y[9]) ** 2) / 4e6) * 1e-5,
        derivative_valid=np.ones((ny, nx), bool), curves_m=[curve], cores_m=[(x[8], y[9])], chosen_curve_m=curve,
        chosen_core_m=(x[8], y[9]), path_m=np.array([[x[8] + 500.0 * k, y[9]] for k in range(4)]), later=later,
    )


def _synthetic_tracks(data: aggregate_figures.SiteFigureData) -> example_figures.TrackFigureInputs:
    ny, nx = data.site_mask.shape
    rng = np.random.default_rng(1)
    tracks = [(24.0 + 6 * i, np.linspace(1000, 5000, 5) + 100 * i, np.linspace(1500, 3000, 5) + 50 * i) for i in range(6)]
    cx, cy = rng.normal(2.0, 3.0, 200), rng.normal(-1.0, 3.0, 200)
    return example_figures.TrackFigureInputs(
        x_m_axis=np.arange(nx) * 500.0, y_m_axis=np.arange(ny) * 500.0, long_track_hours=24.0, long_tracks=tracks, n_long_total=2500,
        prop_cx_kmday=cx, prop_cy_kmday=cy, prop_track_n=40, nominal_step_hours=1.0, null_long_mean=3.5, null_n_seeds=2,
    )


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("empty", [False, True])
def test_wp_h_figures_glyphs_and_no_text_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty: bool) -> None:
    import matplotlib.figure
    import matplotlib.text

    data = _synthetic_figure_data()
    data.spatial_all = {**data.spatial_all, "track_count": np.where(data.site_mask, 2, 0), "long_track_count": np.where(data.site_mask, 1, 0)}
    snap = None if empty else _synthetic_snapshot(data)
    tracks = _synthetic_tracks(data)
    if empty:
        tracks.long_tracks, tracks.n_long_total = [], 0
        tracks.prop_cx_kmday = tracks.prop_cy_kmday = np.zeros(0)
    captured: list[tuple[str, list[tuple[str, object]]]] = []
    original = matplotlib.figure.Figure.savefig

    def _spy(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        boxes = [
            (t.get_text(), t.get_window_extent(renderer))
            for t in self.findobj(matplotlib.text.Text)
            if t.get_visible() and t.get_text().strip() and t.get_window_extent(renderer).width > 0
        ]
        captured.append((str(args[0]), boxes))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _spy)
    paths = example_figures.render_snapshot_figures(
        tmp_path, data, snap, font_selection=FAKE_FONT, coastline=None, state_label="MODEL_ONLY", param_id="p", period_label="202401-202512", dpi=50,
    ) + example_figures.render_track_figures(
        tmp_path, data, tracks, font_selection=FAKE_FONT, coastline=None, state_label="PILOT_ONLY", param_id="p", period_label="202401-202512", dpi=50,
    )
    assert len(paths) == 5 and all((tmp_path / p).is_file() for p in paths)
    missing: set[str] = set()
    for _name, boxes in captured:
        for text, _ in boxes:
            plain = re.sub(r"\$[^$]*\$", "", text)
            missing |= {c for c in plain if c not in REQUIRED_FIGURE_GLYPHS and c not in "\n\t"}
    assert not missing, sorted(missing)
    for key in ("example_title", "method_note_streamline", "propagation_fit_note", "track_pass_note"):
        assert set(FIGURE_TEXT[key]) <= REQUIRED_FIGURE_GLYPHS
    # 代表圖：圖例文字與註腳文字（及軸標籤）互不重疊
    name, boxes = next(item for item in captured if item[0].endswith("_example.png"))
    legend_labels = {FIGURE_TEXT["example_velocity"], FIGURE_TEXT["example_all_curves"], FIGURE_TEXT["example_chosen_curve"]}
    legend = [b for t, b in boxes if t in legend_labels]
    footer = [b for t, b in boxes if "n=" in t or FIGURE_TEXT["science_model_only"] in t or "PILOT" in t or t.startswith("track ")]
    xlabel = [b for t, b in boxes if t == FIGURE_TEXT["lon"]]
    if not empty:
        assert legend and footer and xlabel
    for group_a, group_b in ((legend, footer), (legend, xlabel), (footer, xlabel)):
        for a in group_a:
            for b in group_b:
                assert not a.overlaps(b)
    # 長壽命圖：超過 2000 條時標註，並附 null 平均
    long_texts = [t for n, bx in captured if n.endswith("_long_tracks.png") for t, _ in bx]
    if not empty:
        assert any(FIGURE_TEXT["long_capped_a"] in t for t in long_texts) and any(FIGURE_TEXT["null_long_mean"] in t for t in long_texts)
