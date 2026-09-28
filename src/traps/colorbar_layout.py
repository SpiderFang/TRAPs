"""TRAP 成果圖的主圖／色條幾何排版工具。

Matplotlib 在 ``set_aspect('equal', adjustable='box')`` 之後，主圖實際座標框
可能小於其 GridSpec 或 subplot 原先保留的欄位。若色條沿用欄位高度，色條就會伸入
標題、座標標籤或註腳區域。這個模組沿用相鄰 ``OCM-SVD-Analysis`` 的排版原則：
先讓 renderer 完成 layout，再以主圖 axes 的實際 ``Bbox`` 重設色條的垂直位置與
高度；色階、資料與色條的水平位置都由呼叫端保留。若呼叫端採用 Matplotlib
自動刻度，色條縮短後 renderer 可能重新選取可讀的顯示刻度；本模組不改
``vmin/vmax``、locator 或 formatter 的設定。

本模組只處理 Matplotlib axes 幾何，不知道 TRAP 數值、flow domain 或海岸線資料。
因此任何研究圖都可以共用同一個 helper，而不會把視覺排版規則混進科學計算流程。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def align_colorbars_to_axes(
    figure: Any,
    pairs: Iterable[tuple[Any, Any]],
) -> None:
    """把每條色條的實際上下邊界對齊其主圖 axes。

    Parameters
    ----------
    figure:
        建立色條與主圖的同一個 Matplotlib ``Figure``。函式會使用其 canvas
        觸發一次完整 renderer layout，並在對齊後再繪製一次，以確保輸出 PNG
        和後續 ``get_position()`` 量測看到相同的幾何結果。
    pairs:
        ``(main_axis, colorbar_axis)`` 序列。每個主圖 axes 可以是等比例地圖或
        一般數值圖；色條 axes 的 x 位置與寬度會保留，只更新 y0 與 height。

    Notes
    -----
    ``constrained_layout`` 會在下一次 draw 時重新管理 axes。第一次 draw 完成
    後停用 layout engine，才能讓明確寫入色條的 y0/height 在 ``savefig`` 時保留。
    這個停用只影響目前 figure 的最後排版階段，不會改變資料、vmin/vmax、色階
    或主圖 limits；自動 tick locator 可依新的色條高度選取不同顯示刻度。空的
    ``pairs`` 不會觸發 draw，方便沒有色條的概念圖安全共用
    呼叫邏輯。

    Raises
    ------
    TypeError
        ``figure`` 沒有可繪製 canvas，或任何 pair 不是二元素序列。
    ValueError
        主圖／色條不屬於同一個 figure，或其目前座標框沒有正值寬高。
    """

    pair_list = tuple(pairs)
    if not pair_list:
        return

    canvas = getattr(figure, "canvas", None)
    if canvas is None or not callable(getattr(canvas, "draw", None)):
        raise TypeError("figure 必須提供可呼叫 draw() 的 Matplotlib canvas")

    # ``aspect='equal'`` 與 constrained_layout 的實際 axes 位置必須先由
    # renderer 計算；這一步不能以 figure/subplot 的預設欄位位置替代。
    canvas.draw()

    # 色條位置是最後的人工幾何校正，因此固定第一次 draw 已完成的 layout。
    # Matplotlib >= 3.8 提供此 API；保留無 API 的相容分支，讓 helper 在測試替身
    # 或舊版 renderer 中仍能先完成位置寫入，而不把 layout engine 當作資料錯誤。
    get_layout_engine = getattr(figure, "get_layout_engine", None)
    set_layout_engine = getattr(figure, "set_layout_engine", None)
    if callable(get_layout_engine) and callable(set_layout_engine) and get_layout_engine() is not None:
        set_layout_engine(None)

    for pair_index, pair in enumerate(pair_list):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                "pairs 必須是 (main_axis, colorbar_axis) 二元素 tuple；"
                f"第 {pair_index} 項不符合"
            )
        main_axis, colorbar_axis = pair
        if getattr(main_axis, "figure", figure) is not figure:
            raise ValueError(f"第 {pair_index} 項的主圖 axes 不屬於 figure")
        if getattr(colorbar_axis, "figure", figure) is not figure:
            raise ValueError(f"第 {pair_index} 項的色條 axes 不屬於 figure")

        main_position = main_axis.get_position()
        colorbar_position = colorbar_axis.get_position()
        if main_position.width <= 0.0 or main_position.height <= 0.0:
            raise ValueError(f"第 {pair_index} 項的主圖 axes 沒有正值座標框")
        if colorbar_position.width <= 0.0 or colorbar_position.height <= 0.0:
            raise ValueError(f"第 {pair_index} 項的色條 axes 沒有正值座標框")

        # Matplotlib 的垂直色條預設 box_aspect 可能在下次 draw 依新高度
        # 自行縮窄色條；先解除該幾何比例，才可真正保留已排好的 x0 與 width。
        # 色階與 label 不受這個 axes 排版設定影響；自動刻度可依新高度調整
        # 顯示密度，因此不在這裡強制沿用可能重疊的舊 tick 清單。
        colorbar_axis.set_box_aspect(None)
        colorbar_axis.set_position(
            [
                colorbar_position.x0,
                main_position.y0,
                colorbar_position.width,
                main_position.height,
            ]
        )

    # 輸出前再次 draw，讓 renderer／文字 bbox 與最終 axes 幾何同步；呼叫端可在
    # 此後直接 savefig，無需自行猜測 constrained_layout 是否覆寫人工位置。
    canvas.draw()


__all__ = ["align_colorbars_to_axes"]
