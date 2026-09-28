"""主圖與色條等高排版的幾何與 PNG 契約測試。

測試直接建立含 ``aspect='equal'`` 的 Matplotlib axes，讓 renderer 真正執行
layout，再量測主圖／色條的 figure 座標框。這可捕捉只看 GridSpec 欄位、未考慮
等比例縮放的錯誤；兩個色條使用不同主圖與不同色階，確認共用 helper 不會混用
色階或誤改另一個面板。PNG 檢查則確認最終 figure 經過一次再次 draw 後仍能正常
輸出，代表人工 y0/height 沒有在 savefig 階段被 layout engine 覆寫。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traps.colorbar_layout import align_colorbars_to_axes


def _assert_vertical_edges_match(figure: object, main_axis: object, colorbar_axis: object) -> None:
    """以 point 量測主圖／色條上下邊界，實際驗收誤差不得超過 0.5 point。"""

    main_position = main_axis.get_position()
    colorbar_position = colorbar_axis.get_position()
    figure_height_points = figure.get_figheight() * 72.0
    assert abs(main_position.y0 - colorbar_position.y0) * figure_height_points <= 0.5
    assert (
        abs(
            (main_position.y0 + main_position.height)
            - (colorbar_position.y0 + colorbar_position.height)
        )
        * figure_height_points
        <= 0.5
    )


def test_equal_aspect_multiple_colorbars_align_and_write_png(tmp_path: Path) -> None:
    """等比例雙面板的實際 axes 上下界對齊，且最終畫布可寫成有效 PNG。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.5), constrained_layout=True)
    grid = np.arange(36, dtype=np.float64).reshape(6, 6)
    mesh_left = axes[0].pcolormesh(np.arange(7), np.arange(7), grid, cmap="viridis", vmin=0.0, vmax=35.0)
    mesh_right = axes[1].pcolormesh(
        np.arange(7) * 0.25,
        np.arange(7) * 1.75,
        grid,
        cmap="magma",
        vmin=0.0,
        vmax=35.0,
    )
    colorbar_left = figure.colorbar(mesh_left, ax=axes[0], pad=0.02)
    colorbar_right = figure.colorbar(mesh_right, ax=axes[1], pad=0.02)
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].set_aspect("equal", adjustable="box")
    axes[0].set_title("map A")
    axes[1].set_title("map B")
    align_colorbars_to_axes(
        figure,
        ((axes[0], colorbar_left.ax), (axes[1], colorbar_right.ax)),
    )
    figure.canvas.draw()

    _assert_vertical_edges_match(figure, axes[0], colorbar_left.ax)
    _assert_vertical_edges_match(figure, axes[1], colorbar_right.ax)
    # 第二次 draw 也應保留同一幾何，這是 constrained_layout 被固定的直接證據。
    figure.canvas.draw()
    _assert_vertical_edges_match(figure, axes[0], colorbar_left.ax)
    _assert_vertical_edges_match(figure, axes[1], colorbar_right.ax)

    output = tmp_path / "colorbar_alignment.png"
    figure.savefig(output, dpi=120, format="png")
    plt.close(figure)
    assert output.is_file()
    assert output.stat().st_size > 1_000
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_alignment_preserves_horizontal_bounds_after_layout_is_fixed() -> None:
    """固定版面後只修改色條上下界，不讓 box_aspect 縮窄原色條。"""

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=False)
    mesh = axis.pcolormesh(np.arange(7), np.arange(7), np.ones((6, 6)), shading="auto")
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    axis.set_aspect("equal", adjustable="box")
    figure.canvas.draw()
    original = colorbar.ax.get_position()

    align_colorbars_to_axes(figure, ((axis, colorbar.ax),))
    aligned = colorbar.ax.get_position()
    assert abs(aligned.x0 - original.x0) <= 1.0e-10
    assert abs(aligned.width - original.width) <= 1.0e-10
    _assert_vertical_edges_match(figure, axis, colorbar.ax)
    plt.close(figure)
