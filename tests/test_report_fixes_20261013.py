"""2026-10-13 圖面與統計修正的合成資料測試。

涵蓋：long_tracks 色條標籤與跨站共用色階、平均未截尾壽命面板的單位轉換、
累積天數還原與地圖、逐時／逐日熱點重疊統計與地圖，以及由 aggregate 產物
還原近岸距離場。全部使用合成資料，不讀取正式結果、不需要 SERVER。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_aggregate import FAKE_FONT, _synthetic_figure_data
from test_example_figures import _synthetic_tracks

from traps import aggregate as agg
from traps import aggregate_figures, example_figures
from traps.aggregate_reload import reconstruct_domain_coast_distance
from traps.cumulative_days import cumulative_pass_days, site_valid_days, site_valid_days_summary
from traps.hotspot_overlap import (
    CSV_COLUMNS,
    check_same_grid,
    dilate_mask,
    overlap_statistics,
    site_overlap,
    write_overlap_outputs,
)
from traps.report_font import FIGURE_TEXT, REQUIRED_FIGURE_GLYPHS, cjk_font_context


def _capture_figures(monkeypatch: pytest.MonkeyPatch) -> list:
    """攔截 Figure.savefig 並保留 Figure 供檢查（不改變輸出）。"""

    import matplotlib.figure

    captured: list = []
    original = matplotlib.figure.Figure.savefig

    def _spy(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        captured.append((str(args[0]), self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _spy)
    return captured


def _colorbar_axes(figure, label: str):  # noqa: ANN001, ANN201
    """以 y 標籤找垂直色條 axes（色條標籤是 ylabel）。"""

    matches = [axis for axis in figure.axes if axis.get_ylabel() == label]
    assert len(matches) == 1, [axis.get_ylabel() for axis in figure.axes]
    return matches[0]


# ---------------------------------------------------------------------------
# 1. long_tracks 色條
# ---------------------------------------------------------------------------


def test_shared_long_track_colour_max_rounding() -> None:
    day = example_figures.shared_long_track_colour_max_hours([1632.0, 2352.0, 1416.0], nominal_step_hours=24.0)
    assert day == 2400.0  # 98 天 -> 100 天
    assert example_figures.shared_long_track_colour_max_hours([2400.0], nominal_step_hours=24.0) == 2400.0
    assert example_figures.shared_long_track_colour_max_hours([2424.0], nominal_step_hours=24.0) == 2640.0  # 101 天 -> 110 天
    assert example_figures.shared_long_track_colour_max_hours([84.0, 259.0], nominal_step_hours=1.0) == 264.0  # 12 h 倍數
    assert example_figures.shared_long_track_colour_max_hours([24.0], nominal_step_hours=1.0) == 24.0
    assert example_figures.shared_long_track_colour_max_hours([], nominal_step_hours=24.0) is None
    assert example_figures.shared_long_track_colour_max_hours([float("nan")], nominal_step_hours=24.0) is None


def test_aggregate_long_track_colour_range_uses_all_sites() -> None:
    records = {
        "a": [{"lifetime_hours": 200.0}, {"lifetime_hours": 100.0}],  # 100 < 門檻 168 不計
        "b": [{"lifetime_hours": 2352.0}],
        "c": [],
    }
    result = agg._long_track_colour_range(records, long_track_hours=168.0, nominal_step_hours=24.0)
    assert result["vmin_hours"] == 168.0 and result["vmax_hours"] == 2400.0
    assert result["per_site_max_lifetime_hours"] == {"a": 200.0, "b": 2352.0, "c": None}
    none = agg._long_track_colour_range({"a": []}, long_track_hours=168.0, nominal_step_hours=24.0)
    assert none["vmax_hours"] is None


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    ("step", "label_key", "threshold", "lifetimes", "explicit", "expected"),
    [
        (24.0, "long_tracks_colorbar_days", 168.0, (168.0, 480.0, 1632.0), 2400.0, (7.0, 100.0)),
        (24.0, "long_tracks_colorbar_days", 168.0, (168.0, 480.0, 1632.0), None, (7.0, 68.0)),
        (1.0, "long_tracks_colorbar_hours", 24.0, (24.0, 30.0, 48.0), 48.0, (24.0, 48.0)),
        (1.0, "long_tracks_colorbar_hours", 24.0, (24.0, 30.0, 48.0), None, (24.0, 48.0)),
    ],
)
def test_long_tracks_colorbar_label_and_explicit_vmax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step, label_key, threshold, lifetimes, explicit, expected  # noqa: ANN001
) -> None:
    data = _synthetic_figure_data()
    tracks = _synthetic_tracks(data)
    base = list(tracks.long_tracks)[0]
    tracks.long_tracks = [(life, base[1], base[2]) for life in lifetimes]
    tracks.n_long_total = len(lifetimes)
    tracks.long_track_hours = threshold
    tracks.nominal_step_hours = step
    tracks.colour_max_hours = explicit
    captured = _capture_figures(monkeypatch)
    with cjk_font_context(FAKE_FONT):
        example_figures.render_long_tracks(tmp_path / "long.png", data, tracks, coastline=None, footer="", dpi=50)
    figure = captured[0][1]
    cbar = _colorbar_axes(figure, FIGURE_TEXT[label_key])
    assert cbar.get_ylim() == pytest.approx(expected)
    # 舊標籤「壽命 (天)／(小時)」不得再出現在色條。
    assert not [axis for axis in figure.axes if axis.get_ylabel() in (FIGURE_TEXT["lifetime_days"], FIGURE_TEXT["lifetime_hours"])]


def test_long_tracks_explicit_vmax_below_longest_track_is_rejected(tmp_path: Path) -> None:
    data = _synthetic_figure_data()
    tracks = _synthetic_tracks(data)
    tracks.long_tracks = [(100.0, *list(tracks.long_tracks)[0][1:])]
    tracks.colour_max_hours = 50.0
    with cjk_font_context(FAKE_FONT), pytest.raises(ValueError, match="colour_max_hours"):
        example_figures.render_long_tracks(tmp_path / "long.png", data, tracks, coastline=None, footer="", dpi=50)


def test_new_text_is_covered_by_required_glyphs() -> None:
    for key in (
        "long_tracks_colorbar_days", "long_tracks_colorbar_hours", "cumulative_days_title", "cumulative_days_colorbar",
        "cumulative_days_subtitle_a", "cumulative_days_subtitle_b", "hotspot_overlap_title", "hotspot_overlap_hourly_only",
        "hotspot_overlap_daily_only", "hotspot_overlap_both", "hotspot_overlap_site_cells", "hotspot_overlap_cells",
        "hotspot_overlap_hourly", "hotspot_overlap_daily", "hotspot_overlap_intersection",
    ):
        assert set(FIGURE_TEXT[key]) <= REQUIRED_FIGURE_GLYPHS, key
    assert FIGURE_TEXT["long_tracks_colorbar_days"] == "單一結構連續存在天數"
    assert FIGURE_TEXT["long_tracks_colorbar_hours"] == "單一結構連續存在時數"


# ---------------------------------------------------------------------------
# 2. 空間圖「平均未截尾壽命」顯示單位
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("step", "unit", "scale"), [(24.0, "天", 24.0), (1.0, "小時", 1.0)])
def test_spatial_mean_lifetime_panel_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: float, unit: str, scale: float) -> None:
    data = _synthetic_figure_data()
    data.nominal_step_hours = step
    data.colour_limits = {**data.colour_limits, "mean_lifetime_hours": [24.0, 240.0]}
    stored = {key: np.array(value, copy=True) for key, value in data.spatial_all.items()}
    captured = _capture_figures(monkeypatch)
    with cjk_font_context(FAKE_FONT):
        aggregate_figures.render_spatial(tmp_path / "spatial.png", data, coastline=None, footer="", dpi=50)
    figure = captured[0][1]
    cbar = _colorbar_axes(figure, unit)
    assert cbar.get_ylim() == pytest.approx((24.0 / scale, 240.0 / scale))
    # 只改顯示：儲存的欄位與色階仍是小時，且沒有被就地修改。
    for key, value in stored.items():
        np.testing.assert_array_equal(data.spatial_all[key], value)
    assert data.colour_limits["mean_lifetime_hours"] == [24.0, 240.0]


# ---------------------------------------------------------------------------
# 3. 累積天數
# ---------------------------------------------------------------------------


def test_cumulative_pass_days_recovers_integer_counts() -> None:
    n_valid = np.array([[715, 715, 0], [715, 10, 715]])
    counts_true = np.array([[489, 0, 0], [715, 3, 1]])
    frequency = np.where(n_valid > 0, counts_true / np.where(n_valid > 0, n_valid, 1), np.nan)
    result = cumulative_pass_days(frequency, n_valid)
    expected = np.where(n_valid > 0, counts_true, np.nan).astype(float)
    np.testing.assert_array_equal(result, expected)
    assert np.isnan(result[0, 2])


def test_cumulative_pass_days_rejects_inconsistent_inputs() -> None:
    with pytest.raises(ValueError, match="整數"):
        cumulative_pass_days(np.array([[0.5]]), np.array([[715]]))
    with pytest.raises(ValueError, match="n_valid"):
        cumulative_pass_days(np.array([[1.5]]), np.array([[2]]))
    with pytest.raises(ValueError, match="形狀"):
        cumulative_pass_days(np.zeros((2, 2)), np.zeros((2, 3), dtype=int))


def test_site_valid_days() -> None:
    n_valid = np.array([[715, 700], [0, 715]])
    mask = np.array([[True, True], [False, True]])
    assert site_valid_days(n_valid, mask) == 715
    assert site_valid_days_summary(n_valid, mask) == {"min": 700, "max": 715, "uniform": False}
    assert site_valid_days(n_valid, np.zeros((2, 2), bool)) == 0


@pytest.mark.filterwarnings("ignore")
def test_render_cumulative_days_writes_png_with_expected_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _synthetic_figure_data()
    data.nominal_step_hours = 24.0
    counts = np.where(data.site_mask, np.round(data.spatial_all["curve_pass_frequency"] * 715), np.nan)
    captured = _capture_figures(monkeypatch)
    with cjk_font_context(FAKE_FONT):
        path = aggregate_figures.render_cumulative_days(
            tmp_path / "site_gongliao_cumulative_days.png", data, counts=counts, valid_days=715, coastline=None, dpi=50
        )
    assert path.stat().st_size > 1000
    figure = captured[0][1]
    cbar = _colorbar_axes(figure, FIGURE_TEXT["cumulative_days_colorbar"])
    assert cbar.get_ylim() == pytest.approx((0.0, float(np.nanmax(counts))))
    texts = [artist.get_text() for artist in figure.findobj(__import__("matplotlib").text.Text)]
    assert any("715" in text and FIGURE_TEXT["cumulative_days_subtitle_a"] in text for text in texts)


# ---------------------------------------------------------------------------
# 4. 熱點重疊
# ---------------------------------------------------------------------------


def test_dilate_mask_chebyshev() -> None:
    mask = np.zeros((7, 7), bool)
    mask[3, 3] = True
    one = dilate_mask(mask, 1)
    assert one.sum() == 9 and one[2:5, 2:5].all()
    assert dilate_mask(mask, 2).sum() == 25
    np.testing.assert_array_equal(dilate_mask(mask, 0), mask)
    edge = np.zeros((3, 3), bool)
    edge[0, 0] = True
    assert dilate_mask(edge, 1).sum() == 4  # 邊界不會繞回
    with pytest.raises(ValueError):
        dilate_mask(mask, -1)


def test_overlap_statistics_known_values() -> None:
    valid = np.ones((10, 10), bool)
    hourly = np.zeros((10, 10), bool)
    daily = np.zeros((10, 10), bool)
    hourly[0, 0:4] = True  # 4 格
    daily[0, 2:6] = True  # 4 格；交集 2 格
    stats = overlap_statistics(hourly, daily, valid)
    assert (stats["n_valid_cells"], stats["n_hourly"], stats["n_daily"]) == (100, 4, 4)
    assert stats["n_intersection"] == 2 and stats["n_union"] == 6
    assert stats["jaccard"] == pytest.approx(2 / 6)
    assert stats["pct_hourly_in_daily"] == pytest.approx(50.0) and stats["pct_daily_in_hourly"] == pytest.approx(50.0)
    # 1 格容許：hourly 的 (0,0)(0,1) 離 daily 起點 (0,2) 分別 2、1 格 -> 只有 (0,1) 新增命中
    assert stats["pct_hourly_within_1_of_daily"] == pytest.approx(75.0)
    assert stats["pct_hourly_within_2_of_daily"] == pytest.approx(100.0)
    assert stats["chance_hourly_fraction"] == pytest.approx(0.04) and stats["chance_pct_hourly_in_daily"] == pytest.approx(4.0)
    assert stats["chance_jaccard"] == pytest.approx(0.04 * 0.04 / (0.08 - 0.0016))
    # 容許 1 格的隨機基準 = daily 膨脹後佔有效格點的比例（2 列 x 6 欄 = 12 格）
    assert stats["chance_pct_hourly_within_1_of_daily"] == pytest.approx(12.0)


def test_overlap_statistics_restricts_to_valid_and_handles_empty() -> None:
    valid = np.zeros((4, 4), bool)
    stats = overlap_statistics(np.ones((4, 4), bool), np.ones((4, 4), bool), valid)
    assert stats["n_hourly"] == 0 and stats["jaccard"] is None and stats["pct_hourly_in_daily"] is None


def _spatial(freq: np.ndarray, mask: np.ndarray, quantile: float = 0.9) -> dict[str, np.ndarray]:
    field = np.where(mask, freq, np.nan)
    hot, _ = agg._top_quantile_mask(field, mask, quantile)
    return {
        "lon": np.linspace(120.0, 121.0, mask.shape[1]),
        "lat": np.linspace(24.0, 25.0, mask.shape[0]),
        "site_mask": mask,
        "curve_pass_frequency": field,
        "hotspot_curve_mask": hot,
    }


def test_site_overlap_end_to_end_and_grid_check(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    mask = np.ones((20, 20), bool)
    mask[:, 15:] = False
    base = rng.random((20, 20))
    hourly = _spatial(base, mask)
    daily = _spatial(base + 0.02 * rng.random((20, 20)), mask)
    row, masks = site_overlap(hourly, daily, site_id="x", quantile=0.9)
    assert row["same_grid"] and row["stored_masks_match"]
    assert row["n_valid_cells"] == int(mask.sum()) == 300
    assert row["n_hourly"] == int(hourly["hotspot_curve_mask"].sum()) and row["n_daily"] == int(daily["hotspot_curve_mask"].sum())
    assert row["n_intersection"] == int((masks["hourly"] & masks["daily"]).sum())
    assert 0.0 <= row["jaccard"] <= 1.0

    csv_path, json_path = write_overlap_outputs([row], tmp_path / "out", metadata={"note": "synthetic"})
    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(CSV_COLUMNS)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["sites"][0]["n_intersection"] == row["n_intersection"] and payload["metadata"]["note"] == "synthetic"

    shifted = dict(daily)
    shifted["lon"] = daily["lon"] + 0.001
    assert not check_same_grid(hourly, shifted)["same"]
    with pytest.raises(ValueError, match="格網不同"):
        site_overlap(hourly, shifted, site_id="x", quantile=0.9)


def test_site_overlap_flags_stored_mask_mismatch() -> None:
    mask = np.ones((10, 10), bool)
    freq = np.random.default_rng(1).random((10, 10))
    hourly, daily = _spatial(freq, mask), _spatial(freq, mask)
    daily["hotspot_curve_mask"] = ~daily["hotspot_curve_mask"]
    row, _ = site_overlap(hourly, daily, site_id="x", quantile=0.9)
    assert row["stored_masks_match"] is False


@pytest.mark.filterwarnings("ignore")
def test_render_hotspot_overlap_writes_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _synthetic_figure_data()
    hourly = data.hotspot_curve_mask.copy()
    daily = np.roll(hourly, 1, axis=1) & data.site_mask
    captured = _capture_figures(monkeypatch)
    with cjk_font_context(FAKE_FONT):
        path = aggregate_figures.render_hotspot_overlap(
            tmp_path / "site_gongliao_hotspot_overlap.png", data, hourly_mask=hourly, daily_mask=daily, jaccard=0.5, coastline=None, dpi=50
        )
    assert path.stat().st_size > 1000
    import matplotlib.text

    texts = [artist.get_text() for artist in captured[0][1].findobj(matplotlib.text.Text) if artist.get_text().strip()]
    plain = "".join(texts)
    assert FIGURE_TEXT["hotspot_overlap_both"] in plain and "Jaccard 0.50" in plain
    missing = {c for c in plain if c not in REQUIRED_FIGURE_GLYPHS and c not in "\n\t"}
    assert not missing, sorted(missing)


# ---------------------------------------------------------------------------
# 5. 近岸距離場還原
# ---------------------------------------------------------------------------


def test_reconstruct_domain_coast_distance_unions_sites_and_fills_land_with_zero() -> None:
    mask_a = np.zeros((3, 4), bool)
    mask_b = np.zeros((3, 4), bool)
    mask_a[:, 0:2] = True
    mask_b[:, 2:3] = True  # 第 3 欄之後（最後一欄）是陸地
    field_a = np.where(mask_a, 5000.0, np.nan).astype(np.float32)
    field_b = np.where(mask_b, 12000.0, np.nan).astype(np.float32)
    full = reconstruct_domain_coast_distance(
        {"a": {"site_mask": mask_a, "coast_distance_m": field_a}, "b": {"site_mask": mask_b, "coast_distance_m": field_b}}
    )
    assert full.dtype == np.float32
    np.testing.assert_array_equal(full[:, 0:2], 5000.0)
    np.testing.assert_array_equal(full[:, 2], 12000.0)
    np.testing.assert_array_equal(full[:, 3], 0.0)
    with pytest.raises(ValueError, match="互斥"):
        reconstruct_domain_coast_distance(
            {"a": {"site_mask": mask_a, "coast_distance_m": field_a}, "b": {"site_mask": mask_a, "coast_distance_m": field_a}}
        )
