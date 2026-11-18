"""報告向量圖輸出的合成驗收。

測試只建立小型 Matplotlib Figure，不讀取正式 2024--2025 結果，也不執行 SERVER。
驗收重點是 SVG 不含點陣資源、PDF/SVG 與 PNG 共用同一組有限值／NaN 資料，以及
報告版面紀錄能保存圖幅、最小字級與越界檢查結果。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from traps.vector_figures import VECTOR_PREVIEW_DPI, save_figure


def test_vector_save_has_no_raster_and_preserves_nan_data(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """真正向量輸出不得含 image/data URI，且輸出前後的 masked NaN 資料不變。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setenv("TRAPS_VECTOR_REPORT", "1")
    values = np.array([[0.0, np.nan, 2.0], [3.0, 4.0, np.nan]], dtype=np.float64)
    original = values.copy()
    figure, axis = plt.subplots(figsize=(6.3, 4.2))
    mesh = axis.pcolormesh(np.ma.masked_invalid(values), shading="auto", cmap="viridis")
    figure.colorbar(mesh, ax=axis).set_label("值")
    axis.set_xlabel("經度", fontsize=8)
    axis.set_ylabel("緯度", fontsize=8)
    output = tmp_path / "figures" / "synthetic" / "vector.png"
    written = save_figure(figure, plt, output, dpi=60)

    assert written == [output, output.with_suffix(".svg"), output.with_suffix(".pdf")]
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
    # 報告模式即使 renderer 以 150 dpi 呼叫，也要保留足以置入報告的 PNG 預覽解析度。
    from PIL import Image

    with Image.open(output) as preview:
        assert preview.info.get("dpi", (0, 0))[0] >= VECTOR_PREVIEW_DPI - 1
    svg = output.with_suffix(".svg").read_text(encoding="utf-8")
    assert "<image" not in svg.lower()
    assert "data:image" not in svg.lower()
    # PDF 也不得落回 Matplotlib 的點陣 image XObject；文字、網格與色階均應保留向量語意。
    assert b"/Subtype /Image" not in output.with_suffix(".pdf").read_bytes()
    assert np.array_equal(values, original, equal_nan=True)
    layout = json.loads((tmp_path / "vector_figure_layout.json").read_text(encoding="utf-8"))
    record = layout["figures"][0]
    assert record["figure_size_inch"] == [6.3, 4.2]
    assert record["minimum_visible_font_pt"] >= 8.0
    assert record["text_overflow_count"] == 0
    assert record["axes_overflow_count"] == 0


def test_overview_style_path_keeps_layout_record_inside_output(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """A 區 overview 的兩層產品路徑也應把版面紀錄留在 output 目錄內。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setenv("TRAPS_VECTOR_REPORT", "1")
    figure, axis = plt.subplots(figsize=(6.3, 3.0))
    axis.plot([0.0, 1.0], [0.0, 1.0])
    output = tmp_path / "overview_out" / "r1_hourly" / "domainA_overview.png"
    save_figure(figure, plt, output, dpi=60)
    assert (tmp_path / "overview_out" / "vector_figure_layout.json").is_file()
    assert not (tmp_path / "vector_figure_layout.json").exists()


def test_vector_mode_is_opt_in(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """未設定明確旗標時，儲存層只輸出既有 PNG，不產生向量副檔名。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.delenv("TRAPS_VECTOR_REPORT", raising=False)
    figure, axis = plt.subplots(figsize=(2.0, 1.5))
    axis.plot([0.0, 1.0], [0.0, 1.0])
    output = tmp_path / "ordinary.png"
    assert save_figure(figure, plt, output, dpi=40) == [output]
    assert output.is_file()
    assert not output.with_suffix(".svg").exists()
    assert not output.with_suffix(".pdf").exists()


def test_representative_renderers_have_report_width_and_print_font(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """以實際成果 renderer 驗證 SVG 寬度與置入 6.3 英吋後的最小字級。

    只驗證幾何與輸出封裝，不讀取正式 2024--2025 資料：合成資料仍會經過 spatial、
    lifetime、method comparison 與 propagation 的完整繪圖函式，因此可抓到「手造
    6.3 英吋 Figure 通過，但 renderer 又把圖幅推寬」的回歸。資料陣列本身不在此測試
    中被修改；正式數值與遮罩仍由 aggregate/track 輸入契約負責。
    """

    monkeypatch.setenv("TRAPS_VECTOR_REPORT", "1")
    from test_aggregate import _synthetic_figure_data
    from test_example_figures import _synthetic_snapshot, _synthetic_tracks
    from traps import aggregate_figures, example_figures

    data = _synthetic_figure_data()
    data.spatial_all = {
        **data.spatial_all,
        "track_count": np.where(data.site_mask, 2, 0),
        "long_track_count": np.where(data.site_mask, 1, 0),
    }
    tracks = _synthetic_tracks(data)
    snapshot = _synthetic_snapshot(data)
    site_dir = tmp_path / "figures" / "gongliao"
    renderers = (
        ("spatial", lambda path: aggregate_figures.render_spatial(path, data, coastline=None, footer="", dpi=60)),
        ("lifetime", lambda path: aggregate_figures.render_lifetime(path, data, footer="", dpi=60)),
        (
            "method_comparison",
            lambda path: example_figures.render_method_comparison(path, data, snapshot, coastline=None, footer="", dpi=60),
        ),
        ("propagation", lambda path: example_figures.render_propagation(path, data, tracks, footer="", dpi=60)),
    )
    for stem, renderer in renderers:
        renderer(site_dir / f"site_gongliao_{stem}.png")

    layout = json.loads((tmp_path / "vector_figure_layout.json").read_text(encoding="utf-8"))
    records = {Path(item["path"]).stem: item for item in layout["figures"]}
    for stem, _renderer in renderers:
        record = records[f"site_gongliao_{stem}"]
        assert record["figure_size_inch"][0] == pytest.approx(6.3)
        assert record["print_width_inch"] == pytest.approx(6.3)
        assert record["print_minimum_visible_font_pt"] >= 8.0
        assert record["text_overflow_count"] == 0
        assert record["axes_overflow_count"] == 0
        svg = (site_dir / f"site_gongliao_{stem}.svg").read_text(encoding="utf-8")
        width = re.search(r'<svg[^>]*\bwidth="([0-9.]+)pt"', svg)
        assert width is not None and float(width.group(1)) / 72.0 == pytest.approx(6.3, abs=1.0e-6)
