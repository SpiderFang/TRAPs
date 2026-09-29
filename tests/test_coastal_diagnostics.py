"""WP-I：近岸診斷、近岸外（offshore）統計、代表 track 選取與新圖面的單元與合成端到端測試。

端到端用 ``test_period`` 的南灣合成 domain，但把 analysis ocean mask 的西側五欄設為陸地：合成 TRAP 核心沿
y=0 從 x≈0 緩慢東移，距最近陸地格點約 1.65–1.86 km，因此 ``--coastal-buffer-km 2.0`` 時全部是近岸型、
``1.5`` 時全部是近岸外，兩種分類都能驗證。
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from test_aggregate import (  # noqa: F401
    FAKE_FONT,
    _manifest,
    _namespace,
    _read_csv_rows,
    _synthetic_figure_data,
    fake_font,
)
from test_cli_pipeline import _run_cli
from test_period import (
    DOMAIN_ID as C_DOMAIN_ID,
    FEB_HOURS,
    FEB_LOW_COVERAGE_HOUR,
    JAN_HOURS,
    MAR_HOURS,
    _ocean_mask_path,
    _run_detect_shards,
    _write_month,
    _write_shared_grid,
)

from traps import aggregate as agg
from traps import aggregate_figures, example_figures
from traps.report_font import FIGURE_TEXT, REQUIRED_FIGURE_GLYPHS

HOUR_NS = 3_600_000_000_000


# ---------------------------------------------------------------------------
# 單元：代表 track 選取
# ---------------------------------------------------------------------------


def _cand(track_id: int, lifetime: float, coast_m: float | None, s1: float = -1e-5) -> dict:
    return {"track_id": track_id, "lifetime_hours": lifetime, "mean_core_s1_sinv": s1, "mean_coast_m": coast_m}


def test_representative_prefers_offshore_track_exceeding_null_p99() -> None:
    cands = [
        _cand(1, 100.0, 500.0),  # 最長但近岸型
        _cand(2, 60.0, 8000.0),  # 近岸外、超過 p99
        _cand(3, 20.0, 9000.0),  # 近岸外、未超過 p99
        _cand(4, 80.0, 3000.0),  # 恰在緩衝帶上：不小於緩衝帶 → 近岸外
    ]
    chosen, criterion = agg.select_representative_track(cands, 24.0, coastal_buffer_m=3000.0, null_p99_hours=30.0)
    assert chosen["track_id"] == 4 and criterion == "offshore_exceeds_null"
    chosen, criterion = agg.select_representative_track(cands, 24.0, coastal_buffer_m=3000.0, null_p99_hours=None)
    assert chosen["track_id"] == 4 and criterion == "offshore_longest"
    chosen, criterion = agg.select_representative_track(cands, 24.0, coastal_buffer_m=3000.0, null_p99_hours=80.0)  # 需嚴格大於
    assert chosen["track_id"] == 1 and criterion == "fallback_coastal"  # 沒有近岸外者超過 p99 → 整體最長（近岸型）


def test_representative_fallback_labels_and_legacy_rule() -> None:
    only_coastal = [_cand(1, 40.0, 100.0), _cand(2, 30.0, 200.0)]
    chosen, criterion = agg.select_representative_track(only_coastal, 24.0, coastal_buffer_m=3000.0, null_p99_hours=10.0)
    assert chosen["track_id"] == 1 and criterion == "fallback_coastal"
    below_null = [_cand(1, 40.0, 9000.0), _cand(2, 90.0, 100.0)]
    chosen, criterion = agg.select_representative_track(below_null, 24.0, coastal_buffer_m=3000.0, null_p99_hours=200.0)
    assert chosen["track_id"] == 2 and criterion == "fallback_coastal"
    chosen, criterion = agg.select_representative_track([_cand(1, 40.0, 9000.0)], 24.0, coastal_buffer_m=3000.0, null_p99_hours=200.0)
    assert chosen["track_id"] == 1 and criterion == "fallback_offshore_below_null"
    assert agg.select_representative_track([], 24.0, coastal_buffer_m=3000.0) == (None, "no_candidate")
    # 沒有 mean_coast_m 資訊的候選視為非近岸外，不會被誤選為「近岸外」
    assert agg.select_representative_track([_cand(1, 40.0, None)], 24.0, coastal_buffer_m=3000.0)[1] == "fallback_offshore_below_null"


def test_select_coastal_track_picks_longest_coastal_locked() -> None:
    cands = [_cand(1, 100.0, 9000.0), _cand(2, 50.0, 1000.0), _cand(3, 70.0, 2999.0), _cand(4, 70.0, 3000.0)]
    assert agg.select_coastal_track(cands, 3000.0)["track_id"] == 3
    assert agg.select_coastal_track([_cand(1, 10.0, 9000.0)], 3000.0) is None


# ---------------------------------------------------------------------------
# 單元：近岸／近岸外統計拆分
# ---------------------------------------------------------------------------


def test_track_coastal_split_counts_long_tracks_and_lifetimes() -> None:
    shape = (3, 3)
    grid = {"shape": shape, "x_m_axis": np.array([0.0, 1000.0, 2000.0]), "y_m_axis": np.array([0.0, 1000.0, 2000.0])}
    site = np.ones(shape, dtype=bool)
    t0 = 1_704_100_000_000_000_000
    n = 4
    observations = {
        "track_id": np.repeat([1, 2, 3], n).astype(np.int64),
        "time_utc_ns": np.tile(np.arange(n), 3).astype(np.int64) * HOUR_NS + t0,
        "core_x_m": np.full(3 * n, 1000.0),
        "core_y_m": np.full(3 * n, 1000.0),
        "core_s1_sinv": np.full(3 * n, -1e-5),
        # track 1：平均 1 km（近岸型）；track 2：平均 8 km；track 3：平均 2.9 km（近岸型，缺一筆 NaN 不影響平均）
        "core_coast_distance_m": np.array([500.0, 1500.0, 1000.0, 1000.0, 8000.0, 8000.0, 8000.0, 8000.0, 2900.0, 2900.0, np.nan, 2900.0]),
    }
    tracks = {
        1: {"left_censored": False, "right_censored": False, "lifetime_hours": 10.0, "start_time_utc_ns": t0},
        2: {"left_censored": False, "right_censored": False, "lifetime_hours": 30.0, "start_time_utc_ns": t0},
        3: {"left_censored": False, "right_censored": True, "lifetime_hours": 50.0, "start_time_utc_ns": t0},  # 截尾：不進壽命統計
    }
    acc: dict = defaultdict(agg._new_site_accumulator)
    agg._process_track_lifetimes(
        context={"observations": observations, "tracks_by_id": tracks}, site_masks_2d={"s": site}, grid=grid,
        site_accumulators=acc, spatial_accumulators={"s": {}}, long_track_hours=6.0, coast_buffer_m=3000.0,
    )
    a = acc[("s", "all")]
    assert a["track_lifetimes_coastal"] == [10.0] and a["track_lifetimes_offshore"] == [30.0]
    assert a["long_track_n"] == 2 and a["long_track_coastal_n"] == 1  # 未截尾且 >= 6 h：track 1、2
    assert a["coast_hist_long"].sum() == 2 * n
    born = {item["track_id"]: item for item in a["born_uncensored"]}
    assert born[1]["coastal"] and not born[2]["coastal"] and born[2]["mean_coast_m"] == pytest.approx(8000.0)
    record = agg._finalize_site_record(
        "s", "all", a, site_area_km2=1.0, site_name_zh="s", strong_threshold_sinv=None, hotspot_metrics={},
        bootstrap_reps=5, rng=np.random.default_rng(0), lifetime_thresholds_hours=[12.0, 24.0], coast_buffer_km=3.0,
    )
    assert record["frac_long_tracks_within_buffer"] == pytest.approx(0.5) and record["long_track_n_born_uncensored"] == 2
    assert record["offshore_track_n_uncensored"] == 1 and record["coastal_track_n_uncensored"] == 1
    assert record["offshore_track_frac_uncensored_ge_24h"] == 1.0 and record["coastal_track_frac_uncensored_ge_12h"] == 0.0
    assert record["coastal_buffer_km"] == 3.0
    assert record["track_frac_uncensored_ge_12h"] == pytest.approx(0.5)  # 全部未截尾：10 h 與 30 h

    # 沒有近岸資訊（舊觀測且未補算）：不輸出近岸欄位，也不把全部視為近岸外
    del observations["core_coast_distance_m"]
    plain: dict = defaultdict(agg._new_site_accumulator)
    agg._process_track_lifetimes(
        context={"observations": observations, "tracks_by_id": tracks}, site_masks_2d={"s": site}, grid=grid,
        site_accumulators=plain, spatial_accumulators={"s": {}}, long_track_hours=6.0, coast_buffer_m=3000.0,
    )
    assert not plain[("s", "all")]["coast_info"]
    no_coast = agg._finalize_site_record(
        "s", "all", plain[("s", "all")], site_area_km2=1.0, site_name_zh="s", strong_threshold_sinv=None, hotspot_metrics={},
        bootstrap_reps=5, rng=np.random.default_rng(0), coast_buffer_km=3.0,
    )
    assert "offshore_track_n_uncensored" not in no_coast and "frac_long_tracks_within_buffer" not in no_coast


def test_ensure_observation_coast_samples_domain_field() -> None:
    x_axis = np.arange(5) * 1000.0
    field = np.tile(np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0], dtype=np.float32), (5, 1))
    context = {"observations": {"core_x_m": np.array([500.0, 2500.0]), "core_y_m": np.array([1000.0, 1000.0])}}
    agg.ensure_observation_coast(context, {"coast_distance_m": field, "x_m_axis": x_axis, "y_m_axis": x_axis})
    assert context["observations"]["core_coast_distance_m"] == pytest.approx([500.0, 2500.0])
    kept = context["observations"]["core_coast_distance_m"]
    agg.ensure_observation_coast(context, {"coast_distance_m": field * 0, "x_m_axis": x_axis, "y_m_axis": x_axis})
    assert context["observations"]["core_coast_distance_m"] is kept  # 已存在時不覆寫


def test_hotspot_buffer_metrics_and_offshore_hotspot() -> None:
    shape = (4, 4)
    site_mask = np.ones(shape, dtype=bool)
    accumulator = agg._new_spatial_accumulator(shape)
    accumulator["n_valid"][:] = 10
    accumulator["curve_pass_count"][0, :] = [9, 9, 2, 1]  # 第 0 列前兩格最高（近岸帶內）
    accumulator["curve_pass_count"][2, :] = [1, 4, 6, 6]
    buffer_cells = np.zeros(shape, dtype=bool)
    buffer_cells[0, :2] = True
    buffer_cells[1, :] = True  # 近岸帶：第 0 列前兩格與整個第 1 列
    finalized = agg._finalize_spatial(shape, accumulator)
    hot = agg._site_hotspots(accumulator, finalized, site_mask, np.full(shape, 4.0e6), 0.85, coast_buffer_mask=buffer_cells)
    metrics = hot["metrics"]
    assert metrics["hotspot_cell_count"] == 4  # 門檻 0.6：0.9、0.9、0.6、0.6 四格皆 >= 門檻
    in_buffer = float(np.sum(hot["curve_mask"] & buffer_cells)) / float(hot["curve_mask"].sum())
    assert metrics["hotspot_frac_area_within_buffer"] == pytest.approx(in_buffer)
    # 近岸外熱點：只在站內且不在緩衝帶的格點重新取分位數（不含 0.9 的近岸帶格點）
    offshore_pool = site_mask & ~buffer_cells
    values = finalized["curve_pass_frequency"][offshore_pool]
    threshold = np.quantile(values, 0.85)
    expected_mask = offshore_pool & (finalized["curve_pass_frequency"] >= threshold) & (finalized["curve_pass_frequency"] > 0)
    assert metrics["offshore_hotspot_cell_count"] == int(expected_mask.sum())
    assert metrics["offshore_hotspot_mean_curve_pass_frequency"] == pytest.approx(float(finalized["curve_pass_frequency"][expected_mask].mean()))
    assert metrics["offshore_hotspot_mean_curve_pass_frequency"] < metrics["hotspot_mean_curve_pass_frequency"]
    # 未給遮罩：欄位為 None
    plain = agg._site_hotspots(accumulator, finalized, site_mask, np.full(shape, 4.0e6), 0.85)["metrics"]
    assert plain["hotspot_frac_area_within_buffer"] is None and plain["offshore_hotspot_cell_count"] is None


def test_null_group_columns_and_fieldnames() -> None:
    record = {"offshore_track_frac_uncensored_ge_12h": 0.6, "coastal_track_frac_uncensored_ge_12h": 0.9}
    columns = agg._null_group_columns(
        record, {"offshore": [[10.0, 20.0], [30.0, 40.0]], "coastal": [[5.0], []]}, [12.0]
    )
    assert columns["null_offshore_track_frac_uncensored_ge_12h"] == pytest.approx(np.mean([0.5, 1.0]))
    assert columns["excess_offshore_track_frac_uncensored_ge_12h"] == pytest.approx(0.6 - 0.75)
    assert columns["null_coastal_track_frac_uncensored_ge_12h"] == 0.0  # 空 seed 不納入平均
    names = agg.site_stat_fieldnames([12.0, 24.0], with_null=True)
    for expected in (
        "offshore_track_frac_uncensored_ge_24h", "coastal_track_frac_uncensored_ge_12h", "excess_offshore_track_frac_uncensored_ge_24h",
        "null_coastal_track_frac_uncensored_ge_12h", "frac_cores_within_buffer", "frac_strong_cores_within_buffer",
        "frac_long_tracks_within_buffer", "hotspot_frac_area_within_buffer", "offshore_strong_core_occurrence_rate",
        "offshore_hotspot_mean_curve_pass_frequency", "prop_stationary_frac",
    ):
        assert expected in names, expected
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# 圖面：近岸緩衝帶、近岸型長壽命 track、標籤、字元
# ---------------------------------------------------------------------------


def _coast_field_for(data: aggregate_figures.SiteFigureData) -> np.ndarray:
    ny, nx = data.site_mask.shape
    cols = np.arange(nx)
    return np.tile((cols * 1000.0).astype(np.float32), (ny, 1))  # 距西岸 = 欄數 km


def _capture_texts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[tuple[str, object]]]]:
    import matplotlib.figure
    import matplotlib.text

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
    return captured


def _coastal_data() -> aggregate_figures.SiteFigureData:
    data = _synthetic_figure_data()
    data.coast_distance_m = _coast_field_for(data)
    data.coast_buffer_km = 10.0
    edges = np.arange(0.0, 34.0, 2.0)  # 舊 shard：距離場飽和於 30 km，bin 15 為「≥ 30 km」溢位 bin
    rng = np.random.default_rng(4)
    data.coastal = aggregate_figures.CoastalDiagnosticData(
        buffer_km=10.0, hist_edges_km=edges, cap_km=30.0, detection_floor_km=2.6,
        hist_all=rng.integers(1, 50, 16), hist_strong=rng.integers(0, 20, 16), hist_long=rng.integers(0, 10, 16),
        lifetimes_offshore=[3.0, 8.0, 14.0, 30.0], lifetimes_coastal=[5.0, 40.0, 90.0],
        null_offshore=[[2.0, 4.0, 9.0], [3.0, 6.0]], null_coastal=[[2.0, 3.0], [4.0]],
    )
    data.spatial_all = {**data.spatial_all, "track_count": np.where(data.site_mask, 2, 0), "long_track_count": np.where(data.site_mask, 1, 0)}
    return data


def _snapshot_with_static_core(data: aggregate_figures.SiteFigureData, *, moved: bool) -> example_figures.ExampleSnapshot:
    from test_aggregate import _synthetic_figure_data as _unused  # noqa: F401

    ny, nx = data.site_mask.shape
    x = np.arange(nx) * 500.0
    y = np.arange(ny) * 500.0
    xx, yy = np.meshgrid(x, y)
    envelope = np.exp(-((xx - x[8]) ** 2 + (yy - y[9]) ** 2) / 4e6)
    curve = np.column_stack([np.full(9, x[8]), np.linspace(y[6], y[12], 9)])
    step = 2000.0 if moved else 3.0
    later = [
        example_figures.LaterObservation(dt_seconds=k * 6 * 3600.0, curve_m=curve + [step * k, 0.0], core_m=(x[8] + step * k, y[9]))
        for k in (1, 2, 3)
    ]
    return example_figures.ExampleSnapshot(
        track_id=7, time_utc_ns=1_704_067_200_000_000_000, month="202401", snapshot_index=0, detection_index=0, lifetime_hours=18.0,
        n_observations=4, criterion="fallback_coastal", core_offset_cells=0, x_m_axis=x, y_m_axis=y,
        u=-2e-4 * (xx - x[8]) * envelope, v=-2e-4 * (yy - y[9]) * envelope, valid=np.ones((ny, nx), bool),
        s1_sinv=-envelope * 1e-5, derivative_valid=np.ones((ny, nx), bool), curves_m=[curve], cores_m=[(x[8], y[9])],
        chosen_curve_m=curve, chosen_core_m=(x[8], y[9]), path_m=np.array([[x[8] + step * k, y[9]] for k in range(4)]),
        later=later, is_coastal=True, mean_coast_m=1200.0, variant="coastal",
    )


@pytest.mark.filterwarnings("ignore")
def test_static_core_labels_move_to_offset_list_and_never_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _coastal_data()
    captured = _capture_texts(monkeypatch)
    for moved, name in ((False, "static"), (True, "moved")):
        snap = _snapshot_with_static_core(data, moved=moved)
        example_figures.render_coastal_example(
            tmp_path / name, data, snap, font_selection=FAKE_FONT, coastline=None, state_label="MODEL_ONLY", param_id="p",
            period_label="202401-202512", dpi=50,
        )
    assert example_figures._later_labels_overlap(_snapshot_with_static_core(data, moved=False), data, aggregate_figures._view_extent(data), 4.0, 3.0)
    assert not example_figures._later_labels_overlap(_snapshot_with_static_core(data, moved=True), data, aggregate_figures._view_extent(data), 4.0, 3.0)
    by_dir = {name: boxes for path, boxes in captured for name in ("static", "moved") if f"/{name}/" in path}
    static_texts = [t for t, _ in by_dir["static"]]
    moved_texts = [t for t, _ in by_dir["moved"]]
    # 靜止：地圖上沒有單獨的「+6 h」文字，改列在圖例並附位移；有明確的「（近岸型）」註記
    assert not any(t in {"+6 h", "+12 h", "+18 h"} for t in static_texts)
    assert any("+6 h" in t and FIGURE_TEXT["displacement"] in t for t in static_texts)
    assert any(FIGURE_TEXT["coastal_tag"] in t for t in static_texts)
    assert any(t == "+6 h" for t in moved_texts)  # 有明顯位移時維持地圖標籤
    for boxes in by_dir.values():
        legend_and_labels = [b for t, b in boxes if t in {"+6 h", "+12 h", "+18 h"}]
        for index, first in enumerate(legend_and_labels):
            assert not any(first.overlaps(second) for second in legend_and_labels[index + 1 :])
    assert (tmp_path / "static" / "figures" / "gongliao" / "site_gongliao_example_coastal.png").is_file()
    assert example_figures.render_coastal_example(
        tmp_path / "none", data, None, font_selection=FAKE_FONT, coastline=None, state_label="MODEL_ONLY", param_id="p", period_label="x", dpi=50
    ) == []
    assert not (tmp_path / "none").exists()


@pytest.mark.filterwarnings("ignore")
def test_coastal_figures_glyphs_buffer_legend_and_line_styles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _coastal_data()
    captured = _capture_texts(monkeypatch)
    paths = aggregate_figures.render_site_figures(
        tmp_path, data, font_selection=FAKE_FONT, coastline=None, state_label="MODEL_ONLY", param_id="p", period_label="202401-202512", dpi=50
    )
    assert paths[-1] == "figures/gongliao/site_gongliao_coastal_diagnostic.png" and len(paths) == 6
    snap = _snapshot_with_static_core(data, moved=True)
    snap.variant, snap.is_coastal = "main", False
    snap.criterion = "offshore_exceeds_null"
    ny, nx = data.site_mask.shape
    tracks = example_figures.TrackFigureInputs(
        x_m_axis=np.arange(nx) * 500.0, y_m_axis=np.arange(ny) * 500.0, long_track_hours=24.0,
        long_tracks=[(24.0 + 6 * i, np.linspace(1000, 5000, 5) + 100 * i, np.linspace(1500, 3000, 5) + 50 * i) for i in range(6)],
        n_long_total=6, prop_cx_kmday=np.array([0.0, 2.0, -3.0, 4.0]), prop_cy_kmday=np.array([0.0, 1.0, 2.0, -1.0]), prop_track_n=4,
        nominal_step_hours=1.0, null_long_mean=3.5, null_n_seeds=2,
        long_coastal=[True, False, True, False, False, False], n_long_coastal_total=2, n_long_offshore_total=4,
        null_long_coastal_mean=1.5, null_long_offshore_mean=2.0, prop_stationary=np.array([True, False, False, False]),
        prop_resid_rms_m=np.array([120.0, 300.0, 250.0, 180.0]),
    )
    paths += example_figures.render_snapshot_figures(
        tmp_path, data, snap, font_selection=FAKE_FONT, coastline=None, state_label="MODEL_ONLY", param_id="p", period_label="202401-202512", dpi=50
    ) + example_figures.render_track_figures(
        tmp_path, data, tracks, font_selection=FAKE_FONT, coastline=None, state_label="MODEL_ONLY", param_id="p", period_label="202401-202512", dpi=50
    )
    for relative in paths:
        assert (tmp_path / relative).stat().st_size > 1_000, relative
    # 全部新增固定文字都在字元集內（圖面實際文字逐字檢查）
    missing: set[str] = set()
    texts_by_figure: dict[str, list[str]] = defaultdict(list)
    for name, boxes in captured:
        for text, _ in boxes:
            texts_by_figure[Path(name).name].append(text)
            missing |= {c for c in re.sub(r"\$[^$]*\$", "", text) if c not in REQUIRED_FIGURE_GLYPHS and c not in "\n\t"}
    assert not missing, sorted(missing)
    for key in [k for k in FIGURE_TEXT if k.startswith(("coast", "example_rule_", "propagation_", "long_coast", "later_", "coastal"))]:
        assert set(FIGURE_TEXT[key]) <= REQUIRED_FIGURE_GLYPHS, key
    # 地圖圖例都有「近岸 10 km 帶」
    label = f"{FIGURE_TEXT['coast_buffer_prefix']} 10 {FIGURE_TEXT['coast_buffer_suffix']}"
    for figure_name in (
        "site_gongliao_spatial.png", "site_gongliao_hotspots.png", "site_gongliao_example.png", "site_gongliao_method_comparison.png",
        "site_gongliao_long_tracks.png", "site_gongliao_track_density.png",
    ):
        assert label in texts_by_figure[figure_name], figure_name
    # 長壽命圖：近岸型（虛線）與近岸外圖例，以及近岸型／近岸外計數（實際與對照）
    long_texts = " ".join(texts_by_figure["site_gongliao_long_tracks.png"])
    assert FIGURE_TEXT["coastal_locked_legend"] in long_texts and FIGURE_TEXT["offshore_track_legend"] in long_texts
    assert FIGURE_TEXT["long_coast_real"] in long_texts and FIGURE_TEXT["long_coast_null"] in long_texts
    # 傳播圖：移動／靜止條數、靜止比例與殘差 RMS
    prop_texts = " ".join(texts_by_figure["site_gongliao_propagation.png"])
    assert f"{FIGURE_TEXT['propagation_moving']} 3" in prop_texts and f"{FIGURE_TEXT['propagation_stationary']} 1" in prop_texts
    assert FIGURE_TEXT["propagation_stationary_frac"] in prop_texts and FIGURE_TEXT["propagation_resid"] in prop_texts
    # 近岸診斷圖：兩個子圖標題與分組圖例
    diag_texts = " ".join(texts_by_figure["site_gongliao_coastal_diagnostic.png"])
    for key in ("coast_hist_title", "coast_ccdf_title", "coast_hist_all", "coast_hist_strong", "coast_hist_long", "coast_offshore_real", "coast_coastal_null"):
        assert FIGURE_TEXT[key] in diag_texts, key
    # 檢出下限標註與腳註；舊 shard 的最後一格以明確「≥ 30 km」標示
    assert f"{FIGURE_TEXT['coast_floor_a']} 2.6 km{FIGURE_TEXT['coast_floor_b']}" in diag_texts
    assert FIGURE_TEXT["coast_footnote"] in diag_texts and FIGURE_TEXT["coast_footnote"] in long_texts
    assert f"{FIGURE_TEXT['coast_cap_label']} 30 km" in diag_texts


@pytest.mark.filterwarnings("ignore")
def test_long_tracks_use_dashed_linestyle_for_coastal_locked(tmp_path: Path) -> None:
    """LineCollection 的 linestyle 清單：近岸型為虛線、近岸外為實線（不只靠顏色區分）。"""

    import matplotlib.collections

    data = _coastal_data()
    ny, nx = data.site_mask.shape
    tracks = example_figures.TrackFigureInputs(
        x_m_axis=np.arange(nx) * 500.0, y_m_axis=np.arange(ny) * 500.0, long_track_hours=24.0,
        long_tracks=[(30.0, np.linspace(1000, 3000, 4), np.full(4, 2000.0)), (40.0, np.linspace(1000, 3000, 4), np.full(4, 3000.0))],
        n_long_total=2, prop_cx_kmday=np.zeros(0), prop_cy_kmday=np.zeros(0), prop_track_n=0, nominal_step_hours=1.0,
        long_coastal=[True, False], n_long_coastal_total=1, n_long_offshore_total=1,
    )
    seen: list[list] = []
    original = matplotlib.collections.LineCollection.__init__

    def _spy(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        original(self, *args, **kwargs)
        seen.append(list(kwargs.get("linestyles", [])))

    matplotlib.collections.LineCollection.__init__ = _spy
    try:
        example_figures.render_long_tracks(tmp_path / "l.png", data, tracks, coastline=None, footer="f", dpi=50)
    finally:
        matplotlib.collections.LineCollection.__init__ = original
    styles = next(item for item in seen if item)
    # 由短到長疊繪：壽命 30 h（近岸型，虛線）先、40 h（近岸外，實線）後
    assert styles[0] != "solid" and styles[1] == "solid"


# ---------------------------------------------------------------------------
# 端到端：帶陸地的合成 domain
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coastal_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, Path]:
    base = tmp_path_factory.mktemp("coastal_module")
    root = base / "src"
    grid_dir = _write_shared_grid(root)
    mask = np.load(grid_dir / "analysis_ocean_mask_synthetic_v1.npy")
    mask[:, :5] = False  # 西側五欄為陸地
    np.save(grid_dir / "analysis_ocean_mask_synthetic_v1.npy", mask, allow_pickle=False)
    # 連續小時（無缺口）：第 1、5 小時改為均勻流（無 TRAP 核心），使中間的 track 兩端都是「自然出生／消失」，
    # 即未截尾且站內出生；首尾 track 仍為截尾，不會被計入未截尾統計。
    _write_month(root, month="202401", hours=list(range(1000, 1008)))
    _write_month(root, month="202402", hours=[1007, 1008])
    _write_month(root, month="202403", hours=[1008, 1009])
    month_dir = root / "ocm_surface" / C_DOMAIN_ID / "months" / "202401"
    for name in ("u_surface_mps.npy", "v_surface_mps.npy"):
        array = np.load(month_dir / name)
        array[[1, 5]] = 0.02 if name.startswith("u") else 0.0
        np.save(month_dir / name, array, allow_pickle=False)
    detect_root = base / "detect"
    for month in ("202401", "202402", "202403"):
        result = _run_detect_shards(root, month=month, output_dir=detect_root / C_DOMAIN_ID / month)
        assert result.returncode == 0, result.stderr
    track_root, null_root = base / "track", base / "null"
    common = ["track-period", "--shards-root", str(detect_root / C_DOMAIN_ID), "--domain-id", C_DOMAIN_ID, "--months", "202401-202403"]
    assert _run_cli(*common, "--output-dir", str(track_root / C_DOMAIN_ID)).returncode == 0
    assert _run_cli(*common, "--output-dir", str(null_root / C_DOMAIN_ID), "--shuffle-seed", "7").returncode == 0
    return root, track_root, null_root, detect_root


def _independent_core_stats(track_root: Path, detect_root: Path, buffer_m: float) -> dict[str, float]:
    """獨立於 aggregate 程式碼，只用 shard 與 canonical 軸重算：合格核心中距岸小於 buffer 的比例。"""

    with np.load(track_root / C_DOMAIN_ID / "canonical_snapshots.npz") as canonical:
        month, snapshot_index, coverage_pass = canonical["month"], canonical["snapshot_index"], canonical["coverage_pass"]
    near = total = 0
    for month_code in np.unique(month):
        valid = snapshot_index[(month == month_code) & coverage_pass]
        with np.load(detect_root / C_DOMAIN_ID / f"{int(month_code):06d}" / "detections.npz") as archive:
            selected = archive["is_qualified"] & np.isin(archive["snapshot_index"], valid)
            coast = archive["core_coast_distance_m"][selected]
        near += int(np.count_nonzero(coast < buffer_m))
        total += int(coast.size)
    return {"near": near, "total": total}


@pytest.mark.filterwarnings("ignore")
def test_end_to_end_coastal_and_offshore_classification(coastal_run, tmp_path: Path) -> None:  # noqa: ANN001
    _root, track_root, null_root, detect_root = coastal_run
    with np.load(detect_root / C_DOMAIN_ID / "202401" / "grid_reference.npz") as grid:
        field = grid["coast_distance_m"]
        assert field[:, :5].max() == 0.0 and field[10, 5] == pytest.approx(150.0)  # 陸地 0；緊鄰陸地的海域一格
    with np.load(track_root / C_DOMAIN_ID / "track_observations.npz") as observations:
        distances = observations["core_coast_distance_m"]
    assert 1_600.0 < distances.min() and distances.max() < 1_900.0

    results = {}
    for buffer_km in (2.0, 1.5):
        out = tmp_path / f"agg_{buffer_km}"
        namespace = _namespace(
            track_root, detect_root, out, no_figures=True, prop_min_obs=3, coastal_buffer_km=buffer_km, long_track_hours=1.0,
            null_track_root=[null_root], lifetime_thresholds_hours=[1.0, 2.0],
        )
        assert agg.run_aggregate(namespace) == 0
        rows = _read_csv_rows(out / "site_statistics.csv")
        results[buffer_km] = ({(r["site_id"], r["stratum"]): r for r in rows}, out)
        assert _manifest(out)["coastal_buffer_km"] == buffer_km

    coastal_rows, coastal_out = results[2.0]
    offshore_rows, offshore_out = results[1.5]
    row_c = coastal_rows[("nanwan", "all")]
    row_o = offshore_rows[("nanwan", "all")]
    # 近岸帶 2 km：全部核心在帶內；1.5 km：全部在帶外
    expected_c = _independent_core_stats(track_root, detect_root, 2000.0)
    expected_o = _independent_core_stats(track_root, detect_root, 1500.0)
    assert expected_c["near"] == expected_c["total"] > 0 and expected_o["near"] == 0
    assert float(row_c["frac_cores_within_buffer"]) == pytest.approx(expected_c["near"] / expected_c["total"]) == 1.0
    assert float(row_o["frac_cores_within_buffer"]) == 0.0
    assert float(row_c["frac_strong_cores_within_buffer"]) == 1.0 and float(row_o["frac_strong_cores_within_buffer"]) == 0.0
    assert float(row_c["offshore_strong_core_occurrence_rate"]) == 0.0
    assert float(row_o["offshore_strong_core_occurrence_rate"]) == pytest.approx(float(row_o["strong_core_occurrence_rate"]))
    assert float(row_c["frac_long_tracks_within_buffer"]) == 1.0 and float(row_o["frac_long_tracks_within_buffer"]) == 0.0
    assert int(row_c["offshore_track_n_uncensored"]) == 0 and int(row_c["coastal_track_n_uncensored"]) == int(row_c["track_n_uncensored"])
    assert int(row_o["coastal_track_n_uncensored"]) == 0 and int(row_o["offshore_track_n_uncensored"]) == int(row_o["track_n_uncensored"])
    # 近岸外熱點：緩衝帶內格點被排除，近岸帶 2 km 時仍有帶外格點；熱點面積比例在 [0, 1]
    assert 0.0 <= float(row_c["hotspot_frac_area_within_buffer"]) <= 1.0
    assert row_o["offshore_track_frac_uncensored_ge_1h"] == row_o["track_frac_uncensored_ge_1h"]
    assert row_c["null_offshore_track_n_uncensored"] != "" and row_o["null_coastal_track_n_uncensored"] != ""
    assert row_o["excess_offshore_track_frac_uncensored_ge_1h"] != ""
    # 季節分層也有近岸比例；傳播欄位只在 all
    assert coastal_rows[("nanwan", "DJF")]["frac_cores_within_buffer"] == "1.0"
    assert coastal_rows[("nanwan", "DJF")]["prop_n_tracks"] == ""
    # 傳播：合成核心 14 h 只移動 ~200 m（< 1 格 = 150 m 的...）以一格 150 m 判定：擬合位移 200 m 大於一格 → 移動
    assert int(row_c["prop_n_tracks"]) >= 1 and float(row_c["prop_stationary_frac"]) in (0.0, 1.0)

    archive = np.load(coastal_out / "spatial" / "nanwan" / "spatial_all.npz")
    site_mask = archive["site_mask"]
    assert np.array_equal(archive["coast_buffer_mask"], site_mask & (field < 2000.0))
    assert np.array_equal(np.isfinite(archive["coast_distance_m"]), site_mask)
    assert float(archive["coast_buffer_m"]) == 2000.0
    # 舊 (--coastal-buffer-km 未給) 預設 3 km
    default_out = tmp_path / "agg_default"
    assert agg.run_aggregate(_namespace(track_root, detect_root, default_out, no_figures=True)) == 0
    assert _manifest(default_out)["coastal_buffer_km"] == 10.0


@pytest.mark.filterwarnings("ignore")
def test_end_to_end_figures_coastal_example_and_diagnostic(coastal_run, tmp_path: Path, fake_font) -> None:  # noqa: ANN001
    _root, track_root, null_root, detect_root = coastal_run
    out = tmp_path / "fig_coastal"
    namespace = _namespace(
        track_root, detect_root, out, coastal_buffer_km=2.0, long_track_hours=1.0, null_track_root=[null_root],
        lifetime_thresholds_hours=[1.0, 2.0], no_example_figures=False,
    )
    assert agg.run_aggregate(namespace) == 0
    figures = out / "figures" / "nanwan"
    for kind in ("coastal_diagnostic", "example", "example_coastal", "long_tracks", "propagation", "spatial", "hotspots", "track_density", "method_comparison"):
        assert (figures / f"site_nanwan_{kind}.png").stat().st_size > 1_000, kind
    manifest = _manifest(out)
    info = manifest["site_track_figures"]["nanwan"]
    assert info["example"]["status"] == "ok" and info["example"]["criterion"] == "fallback_coastal" and info["example"]["coastal"] is True
    assert info["example_coastal"]["status"] == "ok" and info["example_coastal"]["mean_coast_distance_m"] < 2000.0
    assert info["n_long_tracks_offshore"] == 0 and info["n_long_tracks_coastal"] == info["n_long_tracks"] > 0
    assert manifest["propagation"]["stationary_cell_m"][C_DOMAIN_ID] == pytest.approx(150.0)
    assert manifest["shard_schema_versions"][C_DOMAIN_ID] == ["traps.period.shard.v2"]
    assert all(str(path.name) in {a["path"].split("/")[-1] for a in manifest["artifacts"]} for path in figures.glob("*.png"))

    # 帶外：無近岸型 track → 不輸出 example_coastal，代表 track 為近岸外（是否超過對照決定準則）
    out2 = tmp_path / "fig_offshore"
    assert agg.run_aggregate(
        _namespace(track_root, detect_root, out2, coastal_buffer_km=1.5, long_track_hours=1.0, null_track_root=[null_root], no_example_figures=False)
    ) == 0
    assert not (out2 / "figures" / "nanwan" / "site_nanwan_example_coastal.png").exists()
    info2 = _manifest(out2)["site_track_figures"]["nanwan"]
    assert info2["example_coastal"] == {"status": "none"}
    assert info2["example"]["criterion"] in {"offshore_exceeds_null", "fallback_offshore_below_null"} and info2["example"]["coastal"] is False
    assert (out2 / "figures" / "nanwan" / "site_nanwan_coastal_diagnostic.png").is_file()

    # --resume：已完成則略過；--coastal-buffer-km 非法值被拒絕
    assert agg.run_aggregate(_namespace(track_root, detect_root, out, coastal_buffer_km=2.0, resume=True)) == 0
    with pytest.raises(ValueError, match="coastal-buffer-km"):
        agg.run_aggregate(_namespace(track_root, detect_root, tmp_path / "bad", coastal_buffer_km=-1.0))
    with pytest.raises(ValueError, match="coastal-buffer-km"):
        agg.run_aggregate(_namespace(track_root, detect_root, tmp_path / "bad2", coastal_buffer_km=250.0))


def test_cli_registers_coastal_and_prop_flags() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    agg.register_subcommands(parser.add_subparsers())
    args = parser.parse_args(["aggregate", "--track-root", "a", "--detect-root", "b", "--output-dir", "c", "--coastal-buffer-km", "5", "--prop-min-obs", "6"])
    assert args.coastal_buffer_km == 5.0 and args.prop_min_obs == 6
    defaults = parser.parse_args(["aggregate", "--track-root", "a", "--detect-root", "b", "--output-dir", "c"])
    assert defaults.coastal_buffer_km == 10.0 and defaults.prop_min_obs is None


def test_coast_hist_edges_cover_full_cap_with_overflow_bin_and_no_spike() -> None:
    """直方圖範圍涵蓋整個距離場上限，末端另有溢位 bin；遠離海岸的核心不再全堆進 30 km 的最後一格。"""

    from traps.period import COAST_DISTANCE_CAP_M

    assert COAST_DISTANCE_CAP_M >= 200_000.0
    edges = agg.COAST_HIST_EDGES_KM
    assert edges[0] == 0.0 and edges[-2] == COAST_DISTANCE_CAP_M / 1000.0 and edges[-1] > edges[-2]
    values = np.clip(np.array([1.0, 12.0, 45.0, 90.0, 150.0]), 0.0, edges[-1] - 1.0e-9)
    counts = np.histogram(values, bins=edges)[0]
    assert counts[-1] == 0 and counts[15] == 0 and counts.sum() == 5  # 30 km 處沒有堆積


def test_coastal_panel_a_has_no_free_text_and_floor_is_legend_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.figure
    import matplotlib.text

    seen: dict[str, object] = {}
    original = matplotlib.figure.Figure.savefig

    def _spy(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if str(args[0]).endswith("d.png"):
            seen["fig"] = self
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _spy)
    data = _coastal_data()
    aggregate_figures.render_coastal_diagnostic(tmp_path / "d.png", data, footer="f", dpi=50)
    fig = seen["fig"]
    ax = fig.axes[0]
    assert ax.get_xlabel() == FIGURE_TEXT["coast_dist_xlabel"] if "coast_dist_xlabel" in FIGURE_TEXT else True
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert any(t.startswith("檢出下限 ≈ 2.6 km") and "此範圍無法定位核心" in t for t in legend_texts)
    allowed = {id(ax.title), id(ax.xaxis.label), id(ax.yaxis.label), id(ax.xaxis.offsetText), id(ax.yaxis.offsetText)}
    allowed |= {id(t) for t in legend.findobj(matplotlib.text.Text)}
    fig.canvas.draw()
    for axis in (ax.xaxis, ax.yaxis):
        allowed |= {id(t) for t in axis.findobj(matplotlib.text.Text)}
    extra = [t.get_text() for t in ax.findobj(matplotlib.text.Text) if id(t) not in allowed and t.get_text().strip()]
    assert all(t.startswith("≥") for t in extra), extra  # 僅允許溢位 bin 的座標標記，不得有檢出下限等自由文字
    assert not any("檢出下限" in t for t in extra)


def test_prop_min_obs_auto_depends_on_step() -> None:
    assert agg.auto_prop_min_obs(1.0) == 7
    assert agg.auto_prop_min_obs(3.0) == 4
    assert agg.auto_prop_min_obs(24.0) == 3
    assert agg.resolve_prop_min_obs(None, 1.0) == 7
    assert agg.resolve_prop_min_obs(None, 24.0) == 3
    assert agg.resolve_prop_min_obs(5, 1.0) == 5
    assert agg.resolve_prop_min_obs(2, 1.0) == 3


@pytest.mark.filterwarnings("ignore")
def test_propagation_footnote_uses_effective_min_obs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _coastal_data()
    ny, nx = data.site_mask.shape
    captured = _capture_texts(monkeypatch)
    tracks = example_figures.TrackFigureInputs(
        x_m_axis=np.arange(nx) * 500.0, y_m_axis=np.arange(ny) * 500.0, long_track_hours=24.0,
        long_tracks=[(30.0, np.linspace(1000, 3000, 4), np.full(4, 2000.0))], n_long_total=1,
        prop_cx_kmday=np.array([1.0, 2.0]), prop_cy_kmday=np.array([0.5, -1.0]), prop_track_n=2, nominal_step_hours=1.0,
        prop_stationary=np.array([False, False]), prop_resid_rms_m=np.array([100.0, 200.0]), prop_min_obs=7,
    )
    example_figures.render_track_figures(
        tmp_path, data, tracks, font_selection=FAKE_FONT, coastline=None, state_label="MODEL_ONLY", param_id="p",
        period_label="202401-202512", dpi=50,
    )
    texts = [t for name, boxes in captured if Path(name).name.endswith("_propagation.png") for t, _ in boxes]
    joined = " ".join(texts)
    assert texts and "至少 3 個觀測" not in joined
    assert "（至少 7 個觀測）" in joined and "{min_obs}" not in joined
