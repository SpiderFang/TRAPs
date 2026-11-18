"""TRAPs 報告圖的 opt-in 向量輸出與版面檢查工具。

本模組只處理繪圖輸出的格式與版面證據，不參與 TRAP 核心、曲線、追蹤或統計計算。
輸入是已由既有 renderer 建立的 Matplotlib ``Figure``；輸出保留原 PNG 預覽，並在
``TRAPS_VECTOR_REPORT=1`` 時另外寫出同名 SVG 與 PDF。SVG／PDF 都直接由 Figure
中的折線、色塊、文字與座標軸重新輸出，不能把既有 PNG 包入容器，也不能以點陣圖描圖。

SVG 使用 path glyph，讓繁體中文字型在沒有相同字型的閱讀環境仍保留可見字形；PDF
則使用可嵌入字型。所有 artist 的 rasterized 狀態會被明確關閉，並對 SVG 內容做
``<image>``、data URI 與外部 href gate。版面紀錄保存圖幅、最小可見字級、文字及
axes 畫框是否越界，供全期圖組驗收使用。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

VECTOR_REPORT_ENV = "TRAPS_VECTOR_REPORT"
"""啟用正式報告向量圖模式的明確環境變數名稱。只有值為 ``1`` 才會啟用。"""

VECTOR_FORMATS = ("svg", "pdf")
"""報告模式的兩種無損向量格式；PNG 仍保留作為相容預覽。"""

VECTOR_PREVIEW_DPI = 300
"""報告模式 PNG 相容預覽的最低解析度；向量 SVG/PDF 不受此值限制。"""

VECTOR_REPORT_WIDTH_INCH = 6.3
"""Word 報告欄寬的固定列印寬度；報告模式 Figure 一律以此寬度產生。"""

VECTOR_LAYOUT_SCHEMA = "traps.vector_figure_layout.v1"
"""單一 aggregate 輸出根目錄的向量版面紀錄 schema。"""

_EXTERNAL_HREF_RE = re.compile(r"(?:xlink:)?href=([\"'])(?!#|data:|about:)", re.IGNORECASE)


def vector_report_enabled() -> bool:
    """回傳是否啟用報告向量圖模式。

    不讀取其他環境變數或隱含設定，避免一般分析流程在未明確選擇時改變輸出格式。
    ``TRAPS_VECTOR_REPORT`` 的值必須完全等於 ``1``；任何其他值都保持舊流程。
    """

    return os.environ.get(VECTOR_REPORT_ENV) == "1"


def _disable_rasterization(figure: Any) -> None:
    """關閉 Figure 內所有可見及隱藏 artist 的 rasterization 設定。

    Matplotlib 的 colorbar 可能建立額外的 ``QuadMesh``／collection；只檢查主要
    axes 會漏掉這些物件，因此使用 ``findobj`` 遞迴掃描整個 Figure。若某個 artist
    沒有 ``set_rasterized`` 介面則略過，因為它不具備可設定的點陣化狀態。
    """

    for artist in figure.findobj():
        setter = getattr(artist, "set_rasterized", None)
        if callable(setter):
            setter(False)
        axis = getattr(artist, "set_rasterization_zorder", None)
        if callable(axis):
            axis(None)


def _temporary_vector_rcparams() -> tuple[Any, dict[str, Any]]:
    """設定 SVG path glyph 與 PDF 可嵌入字型，並回傳可還原的 rcParams。"""

    import matplotlib

    keys = ("svg.fonttype", "pdf.fonttype", "pdf.use14corefonts")
    previous = {key: matplotlib.rcParams[key] for key in keys}
    matplotlib.rcParams["svg.fonttype"] = "path"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["pdf.use14corefonts"] = False
    return matplotlib, previous


def _restore_rcparams(matplotlib: Any, previous: Mapping[str, Any]) -> None:
    """還原向量輸出暫時變更的 Matplotlib 全域設定。"""

    for key, value in previous.items():
        matplotlib.rcParams[key] = value


def _svg_is_vector_only(path: Path) -> None:
    """拒絕含有點陣圖或外部資源引用的 SVG。

    ``<use href="#glyph"/>`` 是 path glyph 的合法內部引用，保留；外部檔案、data
    URI 及 ``<image>`` 都會使成果不符合真正向量圖要求，因此在寫檔後立即 fail closed。
    """

    payload = path.read_text(encoding="utf-8")
    if re.search(r"<image\b|data:image", payload, flags=re.IGNORECASE):
        raise ValueError(f"SVG 含點陣圖內容，拒絕報告輸出: {path}")
    if _EXTERNAL_HREF_RE.search(payload):
        raise ValueError(f"SVG 含外部資源引用，拒絕報告輸出: {path}")


def _figure_root(output_path: Path) -> Path:
    """由成果圖路徑推回版面紀錄根目錄。

    aggregate 圖通常位於 ``<aggregate>/figures/<site>/<name>.png``，因此以
    ``figures`` 目錄的父層作為根；A 區 overview 則是
    ``<output>/<run>/<name>.png``，退回使用其產品輸出目錄，避免把版面紀錄誤寫到
    repository 根目錄或目前工作目錄。這個根目錄只存驗收 JSON，不參與科學資料計算。
    """

    resolved = output_path.expanduser().resolve()
    for parent in resolved.parents:
        if parent.name == "figures":
            return parent.parent
    # A 區 overview 的產品資料夾名稱固定為這兩個既有產品 ID；以其上一層
    # 合併兩個產品的版面紀錄。其他單檔輸出則留在檔案所在資料夾，避免猜測
    # 任意父層是專案根目錄。
    if resolved.parent.name in {"r1_hourly", "d1_daily25h"} and resolved.parent.parent is not None:
        return resolved.parent.parent
    return resolved.parent


def off_view_tick_text_ids_of(figure: Any) -> set[int]:
    """回傳 view 範圍外 tick label 的 ``id``；這些 Text 不會成為輸出圖面的可見內容。

    Matplotlib 會為目前 view 範圍以外的 log／linear tick 建立 Text artist；這些
    tick 的 ``get_visible()`` 仍可能是 True，但實際不在座標軸內，也不會被繪出。
    只排除可由 tick location 證實在 view 外的標籤，保留真正碰到 Figure 邊界的標籤，
    避免版面 gate 用假陽性掩蓋實際裁切。``_layout_record`` 與發布工具的
    ``figure_layout_check`` 共用此函式，兩個 gate 因此對越界的定義一致。
    """

    ids: set[int] = set()
    for axis in figure.axes:
        for ticks, bounds in (
            ((*axis.xaxis.get_major_ticks(), *axis.xaxis.get_minor_ticks()), axis.get_xlim()),
            ((*axis.yaxis.get_major_ticks(), *axis.yaxis.get_minor_ticks()), axis.get_ylim()),
        ):
            lo, hi = sorted((float(bounds[0]), float(bounds[1])))
            tolerance = max(abs(hi - lo) * 1.0e-12, 1.0e-12)
            for tick in ticks:
                get_loc = getattr(tick, "get_loc", None)
                if not callable(get_loc):
                    continue
                try:
                    location = float(get_loc())
                except (TypeError, ValueError):
                    continue
                if lo - tolerance <= location <= hi + tolerance:
                    continue
                for label in (getattr(tick, "label1", None), getattr(tick, "label2", None)):
                    if label is not None:
                        ids.add(id(label))
    return ids


def axes_boxes_inch_of(figure: Any) -> list[list[float]]:
    """各可見 axes 的 ``[x, y, 寬, 高]``（英吋，左下角起算）；供統一版面標準 gate 比對。"""

    width, height = (float(value) for value in figure.get_size_inches())
    boxes: list[list[float]] = []
    for axis in figure.axes:
        if not axis.get_visible():
            continue
        position = axis.get_position()
        boxes.append(
            [
                round(float(position.x0) * width, 4),
                round(float(position.y0) * height, 4),
                round(float(position.width) * width, 4),
                round(float(position.height) * height, 4),
            ]
        )
    return boxes


def _layout_record(figure: Any, output_path: Path) -> dict[str, Any]:
    """擷取圖幅、最小字級及文字／axes 越界資訊。

    bbox 以 Figure canvas 的 pixel 座標檢查；若 renderer 尚未建立，先 draw。圖幅以
    英吋記錄，最小字級以 point 記錄，因此可直接和報告 6.3 英吋基準及 8 pt 門檻比對。
    """

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width_px, height_px = (float(value) for value in figure.bbox.size)
    def outside(box: Any) -> bool:
        return bool(box.x0 < -1.0 or box.y0 < -1.0 or box.x1 > width_px + 1.0 or box.y1 > height_px + 1.0)

    off_view_tick_text_ids = off_view_tick_text_ids_of(figure)

    texts: list[dict[str, Any]] = []
    font_sizes: list[float] = []
    for text in figure.findobj():
        # ``Figure.findobj`` 也會回傳 OffsetBox/TextArea 等複合 artist；這些物件
        # 可能提供 ``get_text`` 卻沒有 Text artist 的字級或 window bbox 介面，故
        # 必須逐一確認公開方法可呼叫，避免版面驗證本身改變正式 render 的結果。
        get_visible = getattr(text, "get_visible", None)
        get_text = getattr(text, "get_text", None)
        get_fontsize = getattr(text, "get_fontsize", None)
        get_window_extent = getattr(text, "get_window_extent", None)
        if not all(callable(method) for method in (get_visible, get_text, get_fontsize, get_window_extent)):
            continue
        if not get_visible():
            continue
        if id(text) in off_view_tick_text_ids:
            continue
        value = str(get_text())
        if not value.strip():
            continue
        box = get_window_extent(renderer=renderer)
        size = float(get_fontsize())
        font_sizes.append(size)
        texts.append({"text": value, "bbox": [float(box.x0), float(box.y0), float(box.width), float(box.height)], "outside": outside(box)})

    axes: list[dict[str, Any]] = []
    for axis in figure.axes:
        if not axis.get_visible():
            continue
        box = axis.get_tightbbox(renderer)
        axes.append({"bbox": [float(box.x0), float(box.y0), float(box.width), float(box.height)], "outside": outside(box)})

    fig_w, fig_h = (float(value) for value in figure.get_size_inches())
    print_scale = VECTOR_REPORT_WIDTH_INCH / fig_w if vector_report_enabled() and fig_w > 0.0 else 1.0
    from .concise_style import concise_checks_of

    concise_checks = concise_checks_of(figure)
    return {
        "path": str(output_path),
        "figure_size_inch": [fig_w, fig_h],
        "minimum_visible_font_pt": min(font_sizes) if font_sizes else None,
        "print_width_inch": VECTOR_REPORT_WIDTH_INCH if vector_report_enabled() else fig_w,
        "print_scale": print_scale,
        "print_minimum_visible_font_pt": min(font_sizes) * print_scale if font_sizes else None,
        "text_overflow_count": sum(bool(item["outside"]) for item in texts),
        "axes_overflow_count": sum(bool(item["outside"]) for item in axes),
        "texts": texts,
        "axes": axes,
        "axes_boxes_inch": axes_boxes_inch_of(figure),
        **({"concise": concise_checks} if concise_checks is not None else {}),
        "vector_formats": [output_path.with_suffix(f".{suffix}").name for suffix in VECTOR_FORMATS],
    }


def _write_layout_record(output_path: Path, record: Mapping[str, Any]) -> None:
    """以原子替換更新 aggregate 根目錄的 ``vector_figure_layout.json``。

    layout record 是驗收證據，不列入科學數值計算；同一張圖重畫時依相對路徑覆寫，
    讓續跑或局部測試不會累積重複紀錄。寫檔失敗會回報，避免向量圖已生成但沒有可檢查的尺寸證據。
    """

    root = _figure_root(output_path)
    path = root / "vector_figure_layout.json"
    payload: dict[str, Any]
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"既有向量版面紀錄無法讀取: {path}") from error
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}
    records = payload.get("figures")
    if not isinstance(records, list):
        records = []
    relative = str(output_path.resolve().relative_to(root.resolve())) if output_path.resolve().is_relative_to(root.resolve()) else output_path.name
    records = [item for item in records if isinstance(item, dict) and item.get("path") != relative]
    updated = dict(record)
    updated["path"] = relative
    records.append(updated)
    records.sort(key=lambda item: str(item.get("path", "")))
    payload = {"schema_version": VECTOR_LAYOUT_SCHEMA, "figures": records, "figure_count": len(records)}
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_figure(figure: Any, plt: Any, output_path: Path, dpi: int) -> list[Path]:
    """依 opt-in 設定保存 Figure，回傳實際寫出的檔案清單。

    未啟用報告模式時只寫入呼叫端指定的 PNG，維持既有行為。啟用後則先以至少
    300 dpi 的 PNG 保存相容預覽，再以同一個 Figure 直接輸出 SVG／PDF，最後寫入
    版面紀錄；即使舊呼叫端傳入較低 dpi，也不會讓 6.3 英吋報告圖退化成低解析預覽。
    所有輸出共用同一批 artist 與數值陣列，不會改變色階、NaN、海岸遮罩或統計母體。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not vector_report_enabled():
        figure.savefig(output_path, dpi=dpi)
        plt.close(figure)
        return [output_path]

    _disable_rasterization(figure)
    matplotlib, previous = _temporary_vector_rcparams()
    written = [output_path]
    try:
        figure.savefig(output_path, dpi=max(int(dpi), VECTOR_PREVIEW_DPI), format="png")
        for suffix in VECTOR_FORMATS:
            vector_path = output_path.with_suffix(f".{suffix}")
            figure.savefig(vector_path, format=suffix)
            written.append(vector_path)
            if suffix == "svg":
                _svg_is_vector_only(vector_path)
        record = _layout_record(figure, output_path)
        _write_layout_record(output_path, record)
    finally:
        _restore_rcparams(matplotlib, previous)
        plt.close(figure)
    return written


__all__ = [
    "VECTOR_FORMATS",
    "VECTOR_LAYOUT_SCHEMA",
    "VECTOR_PREVIEW_DPI",
    "VECTOR_REPORT_WIDTH_INCH",
    "VECTOR_REPORT_ENV",
    "axes_boxes_inch_of",
    "off_view_tick_text_ids_of",
    "save_figure",
    "vector_report_enabled",
]
