"""2026-10-29 v51：壽命分布圖（kind lifetime）

* 說明段不再有「由 3 小時以上之 N 倍增至 6 小時以上之 N 倍」，全文沒有「6 小時以上之」；
* x = 3 小時處兩條曲線各有一個同曲線色小實心點（3–4 pt、無邊線）與比例標示（文字等於正文／表的數字、同曲線色、8 pt、無白邊），位於點右上方且不壓到曲線、參考線與軸框；
* 五站共用軸範圍，且上下限等於首尾刻度值（x 1–300、刻度 1、3、10、30、100、300；y 0.0001–100 %、刻度 100…0.0001），五站所有曲線點都在範圍內；
* gate 抓得到標示文字與表值不同、標示壓到曲線、軸上下限不等於刻度、五站範圍不一致。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from test_aggregate import FAKE_FONT
from test_lifetime_figures_20261028 import FIGS, SITES, _gate, _gate_inputs, _generator, _need_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from traps import lifetime_compare as LC  # noqa: E402
from traps.report_font import cjk_font_context  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------------- 說明段
def test_report_text_has_no_lifetime_distribution_paragraph_or_random_control_wording() -> None:
    """v62 (2026-10-30, user): the lifetime-distribution figures and their paragraphs, the coast classification and the random control are removed from the report text; the figure files stay on disk (and are still checked below)."""
    _need_data()
    gen = _generator()
    md = gen.build()
    for word in ("6 小時以上之", "3 小時以上", "隨機", "對照", "壽命分布", "近岸型", "近岸外"):
        assert word not in md, word
    for name in ("lifetime_sentence", "lifetime_caption", "coast_clause", "RAND_DEF", "RAND_POINTER", "THRESHOLD_BASIS"):
        assert not hasattr(gen, name), name


def test_lifetime_row_and_overall_conclusion_use_the_p90_hours() -> None:
    """v62: the 結構壽命 row of 表2–表6 and the 整體結論 carry the p90 lifetime hours (the smallest k with a cumulative share of 90 % or more), taken from site_statistics."""
    _need_data()
    gen = _generator()
    for k, site in enumerate(gen.SITES):
        p90 = gen.ST[site]['p90_hours']
        row = gen.site_texts(site, k, 2 + 3 * k)["rows"]
        assert row["結構壽命"][2] == f"{gen.hours(p90)} 小時以內", site
        assert "近岸與近岸外" not in row
        assert gen.bullet(site, k + 1).endswith(f"；90% 之結構壽命為 {gen.hours(p90)} 小時以內。"), site


# --------------------------------------------------------------------------- 軸
def test_axis_limits_are_the_first_and_last_ticks_and_cover_all_points() -> None:
    _need_data()
    assert LC.X_LIM == (LC.X_TICKS[0], LC.X_TICKS[-1]) == (1.0, 300.0) and LC.X_TICKS == (1, 3, 10, 30, 100, 300)
    assert LC.Y_LIM_PCT == (LC.Y_TICKS_PCT[-1], LC.Y_TICKS_PCT[0]) == (1.0e-4, 100.0) and LC.Y_TICKS_PCT == (100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001)
    hist = LC.load_hist()
    for site in SITES:
        curves = LC.site_curves(hist, site)
        for key in ("real", "null_mean"):
            values = LC.percent_axis(curves[key])
            positive = values[values > 0]
            assert LC.Y_LIM_PCT[0] <= positive.min() and positive.max() <= LC.Y_LIM_PCT[1] + 1e-9, (site, key)
        assert LC.X_LIM[0] <= curves["ks"].min() and curves["ks"].max() <= LC.X_LIM[1], site


def test_rendered_axes_are_flush_with_ticks_and_identical_for_five_sites() -> None:
    _need_data()
    hist = LC.load_hist()
    seen = set()
    for site in SITES:
        with cjk_font_context(FAKE_FONT):
            fig = LC.render(LC.site_curves(hist, site), plt)
        axis = fig.axes[0]
        xlim, ylim = tuple(axis.get_xlim()), tuple(axis.get_ylim())
        assert xlim == (axis.get_xticks()[0], axis.get_xticks()[-1]) and ylim == (min(axis.get_yticks()), max(axis.get_yticks()))
        labels = [t.get_text() for t in axis.get_xticklabels()] + [t.get_text() for t in axis.get_yticklabels()]
        assert labels == ["1", "3", "10", "30", "100", "300", "100", "10", "1", "0.1", "0.01", "0.001", "0.0001"]
        seen.add((xlim, ylim, tuple(axis.get_xticks()), tuple(axis.get_yticks())))
        plt.close(fig)
    assert len(seen) == 1


# --------------------------------------------------------------------------- 標示
def test_marks_and_labels_match_the_table_values_and_style() -> None:
    _need_data()
    gen = _generator()
    hist = LC.load_hist()
    for site in SITES:
        with cjk_font_context(FAKE_FONT):
            fig = LC.render(LC.site_curves(hist, site), plt)
        axis = fig.axes[0]
        want = {"real": gen.pct_small(gen.ST[site]["f3"]), "null": gen.pct_small(gen.ST[site]["null_f3"])}
        assert LC.share_label(gen.ST[site]["f3"]) == want["real"] and LC.share_label(gen.ST[site]["null_f3"]) == want["null"]
        by_text = {t.get_text(): t for t in axis.texts}
        assert sorted(by_text) == sorted(want.values())
        for series, colour in (("real", LC.REAL_COLOUR), ("null", LC.NULL_COLOUR)):
            text = by_text[want[series]]
            assert matplotlib.colors.to_hex(text.get_color()) == colour and text.get_fontsize() == 8.0 and text.get_path_effects() == []
            assert text.get_bbox_patch() is None      # 無底框、無白邊
        dots = [line for line in axis.get_lines() if line.get_marker() == "o"]
        assert len(dots) == 2
        for dot in dots:
            assert 3.0 <= dot.get_markersize() <= 4.0 and dot.get_markeredgewidth() == 0 and list(dot.get_xdata()) == [3]
            assert matplotlib.colors.to_hex(dot.get_markerfacecolor()) == matplotlib.colors.to_hex(dot.get_color())
            assert matplotlib.colors.to_hex(dot.get_color()) in (LC.REAL_COLOUR, LC.NULL_COLOUR)
        clearance = fig._traps_lifetime_facts["label_clearance_pt"]  # noqa: SLF001
        assert all(v >= LC.LABEL_CLEARANCE_PT for item in clearance.values() for v in item.values()), site
        plt.close(fig)


def test_delivered_sidecars_carry_the_annotations() -> None:
    _need_data()
    gen = _generator()
    for site in SITES:
        path = FIGS / "r1_hourly" / site / f"site_{site}_lifetime.json"
        if not path.is_file():
            pytest.skip("交付圖組不在此環境")
        side = json.loads(path.read_text(encoding="utf-8"))
        notes = {a["series"]: a for a in side["annotations"]}
        assert notes["real"]["text"] == gen.pct_small(gen.ST[site]["f3"]) and notes["null"]["text"] == gen.pct_small(gen.ST[site]["null_f3"])
        assert all(a["x_hours"] == 3 and a["white_edge"] is False and a["marker_edge_width_pt"] == 0 and a["font_pt"] == 8.0 for a in notes.values())
        assert side["axes"]["xlim"] == [1.0, 300.0] and side["axes"]["ylim_percent"] == [0.0001, 100.0]


# --------------------------------------------------------------------------- gate
def _failures(tmp_path: Path, mutate) -> list[str]:  # noqa: ANN001
    gate = _gate()
    records, manifest = _gate_inputs(tmp_path)
    mutate(records, Path(records[0]["absolute_path"]).with_suffix(".json"))
    return gate.lifetime_failures(tmp_path, records, manifest)


def test_gate_accepts_the_delivered_labels_and_axes(tmp_path: Path) -> None:
    _need_data()
    assert _failures(tmp_path, lambda records, side: None) == []


def test_gate_detects_label_text_that_differs_from_the_table(tmp_path: Path) -> None:
    _need_data()

    def mutate(records, _side):  # noqa: ANN001
        for item in records[0]["texts"]:
            if item["text"].endswith("%") and "(%)" not in item["text"]:
                item["text"] = "25.03%"
                break

    errors = _failures(tmp_path, mutate)
    assert any("圖內" in m for m in errors)


def test_gate_detects_missing_or_extra_percent_text(tmp_path: Path) -> None:
    _need_data()
    errors = _failures(tmp_path, lambda records, _s: records[0]["texts"].append({"text": "99.9%", "bbox": [0, 0, 10, 10], "outside": False}))
    assert any("圖內百分比標示" in m or "多出" in m for m in errors)


def test_gate_detects_label_on_top_of_a_curve_or_the_reference_line(tmp_path: Path) -> None:
    _need_data()

    def on_curve(records, _side):  # noqa: ANN001
        for item in records[0]["texts"]:
            if item["text"].endswith("%") and "(%)" not in item["text"]:
                item["bbox"][1] -= 20.0      # 下移 20 px：壓到曲線
                break

    assert any("曲線" in m or "不在點的右上方" in m for m in _failures(tmp_path / "a", on_curve))

    def on_line(records, _side):  # noqa: ANN001
        for item in records[0]["texts"]:
            if item["text"].endswith("%") and "(%)" not in item["text"]:
                item["bbox"][0] -= 12.0
                break

    assert any("參考線" in m or "不在點的右上方" in m for m in _failures(tmp_path / "b", on_line))


def test_gate_detects_axis_limit_not_on_a_tick_and_unequal_sites(tmp_path: Path) -> None:
    _need_data()

    def off_tick(_records, side_path):  # noqa: ANN001
        side = json.loads(side_path.read_text(encoding="utf-8"))
        side["axes"]["xlim"] = [0.8, 300.0]
        side_path.write_text(json.dumps(side), encoding="utf-8")

    errors = _failures(tmp_path, off_tick)
    assert any("切齊刻度" in m for m in errors) and any("座標軸範圍" in m for m in errors)


def test_gate_detects_wrong_annotation_record(tmp_path: Path) -> None:
    _need_data()

    def mutate(_records, side_path):  # noqa: ANN001
        side = json.loads(side_path.read_text(encoding="utf-8"))
        side["annotations"][0]["marker_diameter_pt"] = 6.0
        side["annotations"][1]["colour"] = "#000000"
        side_path.write_text(json.dumps(side), encoding="utf-8")

    errors = _failures(tmp_path, mutate)
    assert any("3–4 pt" in m for m in errors) and any("曲線色" in m for m in errors)
