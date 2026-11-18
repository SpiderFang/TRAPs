"""2026-10-27 v46：簡要報告聚集熱區圖與圖1 的核心黑點縮小（2.0 pt 加 0.4 pt 白邊 → 1.2 pt、無邊線），圖例黑點同步縮小（5.0 pt → 3.0 pt、無邊線）；
紅色熱區格不再被黑點蓋住。以合成資料畫圖，從 SVG 讀回實際的標記直徑，鎖定新的大小。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from lxml import etree
from test_concise_figures_20261016 import _write_options
from test_unfiltered_hotspots_20261027 import _data_with_small_pieces, _expected_split, _gate, _render, _with_outside_cells

from traps import concise_style

pytestmark = pytest.mark.filterwarnings("ignore")

SVG_NS = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"


def _marker_diameters(svg: Path) -> dict[str, list[float]]:
    """SVG 內填 #111111（黑點）或 #7f7f7f（灰點）的標記：地圖上與圖例中各自的直徑（pt）與是否有邊線；``grey`` 另記灰點的地圖直徑。"""
    gate = _gate()
    root = etree.parse(str(svg)).getroot()
    paths = {p.get("id"): p for p in root.iter(f"{{{SVG_NS}}}path") if p.get("id")}
    out: dict[str, list[float]] = {"map": [], "legend": [], "stroked": [], "grey": [], "legend_grey": []}
    for use in root.iter(f"{{{SVG_NS}}}use"):
        style = (use.get("style") or "").replace(" ", "").lower()
        grey = "fill:#7f7f7f" in style
        if "fill:#111111" not in style and not grey:
            continue
        path = paths[use.get(f"{{{XLINK}}}href").lstrip("#")]
        x0, y0, x1, y1 = gate._path_bbox(path.get("d"))  # noqa: SLF001
        where = "legend" if any((anc.get("id") or "").startswith("legend") for anc in use.iterancestors()) else "map"
        out["legend_grey" if grey and where == "legend" else "grey" if grey else where].append(round(x1 - x0, 3))
        if "stroke:" in style and "stroke:none" not in style:
            out["stroked"].append(1.0)
    return out


def test_core_dot_constants_are_locked() -> None:
    assert concise_style.HOTSPOT_CORE_DOT_PT == 1.2 and concise_style.HOTSPOT_CORE_DOT_EDGE_PT == 0.0
    assert concise_style.HOTSPOT_CORE_LEGEND_PT == 3.0 and concise_style.HOTSPOT_CORE_LEGEND_EDGE_PT == 0.0
    dot, legend = concise_style.hotspot_core_dot_kwargs(), concise_style.hotspot_core_legend_kwargs()
    assert dot["markersize"] == 1.2 and dot["markeredgewidth"] == 0.0 and dot["markerfacecolor"] == "#111111"
    grey_dot, grey_legend = concise_style.hotspot_core_dot_kwargs(outside=True), concise_style.hotspot_core_legend_kwargs(outside=True)      # v65：灰點大小、無邊線與黑點相同，只有顏色不同
    assert (grey_dot["markersize"], grey_dot["markeredgewidth"]) == (1.2, 0.0) and (grey_legend["markersize"], grey_legend["markeredgewidth"]) == (3.0, 0.0)
    assert grey_dot["markerfacecolor"] == grey_dot["markeredgecolor"] == grey_legend["markerfacecolor"] == "#7f7f7f"
    assert legend["markersize"] == 3.0 and legend["markeredgewidth"] == 0.0 and legend["markerfacecolor"] == dot["markerfacecolor"]
    assert dot["markeredgecolor"] == legend["markeredgecolor"] == dot["markerfacecolor"]      # 沒有白邊
    assert legend["markersize"] / concise_style.HOTSPOT_CORE_DOT_PT == pytest.approx(2.5)


def test_concise_render_draws_small_dots_without_white_edge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_options(tmp_path, monkeypatch, limits={"hotspots": {"curve_pass_frequency": [0.0, 0.35]}})
    data, _big_curve, _big_core = _data_with_small_pieces()
    data = _with_outside_cells(data)
    _render(tmp_path, data)
    sizes = _marker_diameters(tmp_path / "figures/wide/site_wide_hotspots.svg")
    black, grey = _expected_split(data)
    assert sizes["map"] and set(sizes["map"]) == {concise_style.HOTSPOT_CORE_DOT_PT}      # 黑點 1.2 pt
    assert sizes["grey"] and set(sizes["grey"]) == {concise_style.HOTSPOT_CORE_DOT_PT}      # v65：灰點同大小
    assert (len(sizes["map"]), len(sizes["grey"])) == (int(black.sum()), int(grey.sum()))      # 個數見 test_dot_rule_20261102.py
    assert sizes["legend"] == [concise_style.HOTSPOT_CORE_LEGEND_PT] and sizes["legend_grey"] == [concise_style.HOTSPOT_CORE_LEGEND_PT]
    assert not sizes["stroked"]


def test_formal_render_keeps_the_old_dot_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """正式圖組（未啟用 concise）不變：黑點仍依格寬決定（2–9 pt）、legend 5 pt。"""
    monkeypatch.setenv("TRAPS_VECTOR_REPORT", "1")
    monkeypatch.delenv(concise_style.ENV_OPTIONS, raising=False)
    data, _big_curve, _big_core = _data_with_small_pieces()
    _render(tmp_path, data)
    sizes = _marker_diameters(tmp_path / "figures/wide/site_wide_hotspots.svg")
    assert sizes["map"] and min(sizes["map"]) >= 2.0 - 1.0e-6
    assert sizes["legend"] == [5.0]
