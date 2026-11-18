"""簡要報告「壽命分布與隨機對照比較圖」（圖種 ``lifetime``，2026-10-28 v49）。

每個地點一張單圖：縱軸為「壽命達該時數以上之比例」（互補累積分布 P(壽命 ≥ x)，百分比、對數刻度），橫軸為壽命（小時，對數刻度）；
實線為實際、灰色虛線為時間置換隨機對照（兩組亂數種子的平均）。資料為 ``outputs/concise_figures_20261016/lifetime_hist_r1_hourly.json``
（各地點未截尾強核心結構壽命的次數分布，實際與兩組隨機對照；由 .203 server 的 track 目錄以 ``traps.aggregate._null_track_records_by_site``
的同一規則取出，n 與壽命 ≥ 3／6／12／24 小時之比例已與 ``site_statistics.json`` 逐位核對）。

繪法（與完整版 ``aggregate_figures.render_lifetime`` 的差異）：壽命為整數小時，P(壽命 ≥ k) 只在整數 k 有定義，本圖逐整數 k 取值後以折線連接。
完整版把實際分布以逐筆樣本畫成「後接」階梯（``where='post'``），而隨機對照以唯一值格點畫成同樣的「後接」階梯：實際曲線因此在 [k, k+1) 間畫出 P(≥ k+1)
（向左偏移一小時），隨機對照則畫出 P(≥ k)；兩者的定義與母體並無不同，只是階梯慣例不一致（完整版圖的 1–2 小時段隨機對照看似高於實際即由此而來）。
本圖兩條曲線以同一個函式 :func:`survival` 取值，x = 3 小時的值等於結論摘要表的「壽命 3 小時以上之比例（實際／隨機對照）」。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

KIND = "lifetime"
ROOT = Path(__file__).resolve().parents[2]
HIST_PATH = ROOT / "outputs" / "concise_figures_20261016" / "lifetime_hist_r1_hourly.json"
SITES = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")
NULL_KEYS = ("null_s1", "null_s2")
REFERENCE_HOURS = 3
SECOND_HOURS = 6

# --- 圖面規格（五個地點相同）
# v51（2026-10-29，使用者要求）：軸的上下限本身就是一個有標示的刻度值（切齊刻度，如經緯度地圖）；五個地點共用同一組範圍與刻度。
# x：1–300 小時、刻度 1、3、10、30、100、300（南灣實際最長 259 小時，v50 的刻度只到 192）；y：0.0001–100 %、刻度 100、10、1、0.1、0.01、0.001、0.0001
# （五站所有曲線點含隨機對照最小值 0.000326 % 都在範圍內）。曲線起點 (1 小時, 100 %) 因而落在軸框左上角。
X_TICKS = (1, 3, 10, 30, 100, 300)
X_LIM = (float(X_TICKS[0]), float(X_TICKS[-1]))
Y_TICKS_PCT = (100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001)
Y_LIM_PCT = (Y_TICKS_PCT[-1], Y_TICKS_PCT[0])
REAL_COLOUR = "#c00000"      # 與代表性 TRAP 圖之起始曲線同色
NULL_COLOUR = "#6c757d"
REAL_LINEWIDTH_PT = 1.8      # 同代表性 TRAP 起始曲線
NULL_LINEWIDTH_PT = 1.5
NULL_DASHES = (0, (4.0, 2.5))
REFERENCE_LINE_COLOUR = "#d9d9d9"      # 3 小時處之極淡垂直參考線
REFERENCE_LINE_WIDTH_PT = 0.9
REAL_LABEL = "實際"
NULL_LABEL = "隨機對照"
X_LABEL = "壽命 (小時)"
Y_LABEL = "壽命達該時數以上之比例 (%)"
# v51：x = 3 小時處兩條曲線各標一個實心點與其比例（數字格式同正文與結論摘要表：1 % 以上兩位小數、未滿 1 % 三位有效數字）
MARKER_DIAMETER_PT = 3.5          # 小實心點，直徑 3–4 pt，同曲線色、無邊線（精簡版圖一律不用白邊）
MARKER_EDGE_WIDTH_PT = 0.0
LABEL_FONT_PT = 8.0               # 同刻度標籤
LABEL_OFFSET_PT = (4.0, 2.0)      # 文字左下角相對於點：右 4 pt、上 2 pt（點的右上方）；五個地點、兩條曲線同一規則
LABEL_CLEARANCE_PT = 1.0          # 文字外框與任何曲線、3 小時參考線、軸框的最小距離（render 時量測，不足即報錯）
SEED_BAND_MAX_RELATIVE_SPREAD = 0.35
"""兩組種子的相對差（在兩組都有結構、壽命 ≤ 6 小時的範圍內）不超過此值就不畫色帶，只畫平均（五個地點的最大相對差 0.04–0.30，折合對數軸上不足一條線寬，所以五張圖一律只畫平均）。"""


def load_hist(path: Path | None = None) -> dict[str, Any]:
    payload = json.loads((path or HIST_PATH).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "traps.lifetime_hist.v1", payload["schema_version"]
    return payload["hist"]


def survival(values: Any, counts: Any, ks: Any) -> np.ndarray:
    """P(壽命 ≥ k)：由次數分布（唯一壽命值與次數）在整數 k 取值。"""

    v, c = np.asarray(values, dtype=float), np.asarray(counts, dtype=float)
    total = float(c.sum())
    return np.array([float(c[v >= k].sum()) / total for k in np.asarray(ks, dtype=float)])


def site_curves(hist: dict[str, Any], site: str) -> dict[str, Any]:
    """一個地點的曲線：k＝1…實際最長壽命（整數小時）、實際、兩組種子與其平均（皆為比例，0–1）。"""

    real = hist["real"][site]
    nulls = [hist[key][site] for key in NULL_KEYS]
    for entry in (real, *nulls):
        assert all(float(x).is_integer() for x in entry["values"]), site
    max_real = int(max(real["values"]))
    max_null = int(max(max(entry["values"]) for entry in nulls))
    ks = np.arange(1, max_real + 1)
    real_s = survival(real["values"], real["counts"], ks)
    null_seed = [survival(entry["values"], entry["counts"], ks) for entry in nulls]
    null_mean = np.mean(null_seed, axis=0)
    return dict(site=site, ks=ks, real=real_s, null_seed=null_seed, null_mean=null_mean, n_real=int(real["n"]), n_null=[int(entry["n"]) for entry in nulls],
                max_real=max_real, max_null=max_null)


def site_facts(curves: dict[str, Any]) -> dict[str, Any]:
    """正文與 gate 用的事實：3 小時與 6 小時之比例、倍數、最長壽命、「各壽命皆高於」與「倍數遞增」之判定、兩組種子之差。"""

    ks, real, null = curves["ks"], curves["real"], curves["null_mean"]
    at = {int(k): i for i, k in enumerate(ks)}
    i3, i6 = at[REFERENCE_HOURS], at[SECOND_HOURS]
    assert null[i6] > 0, curves["site"]
    mask = ks >= REFERENCE_HOURS
    pair = [curves["null_seed"][0][ks <= SECOND_HOURS], curves["null_seed"][1][ks <= SECOND_HOURS]]
    spread = float(np.max(np.abs(pair[0] - pair[1]) / np.maximum(pair[0], pair[1])))
    return dict(
        f3=float(real[i3]), null_f3=float(null[i3]), ratio3=float(real[i3] / null[i3]), f6=float(real[i6]), null_f6=float(null[i6]), ratio6=float(real[i6] / null[i6]),
        max_real_hours=int(curves["max_real"]), max_null_hours=int(curves["max_null"]), n_real=int(curves["n_real"]), n_null_mean=float(np.mean(curves["n_null"])),
        real_above_null_from_3h=bool(np.all(real[mask] > null[mask])),
        ratio_increasing_3_to_6h=bool(np.all(np.diff(real[(ks >= REFERENCE_HOURS) & (ks <= SECOND_HOURS)] / null[(ks >= REFERENCE_HOURS) & (ks <= SECOND_HOURS)]) > 0)),
        seed_max_relative_spread_to_6h=spread, draw_seed_band=bool(spread > SEED_BAND_MAX_RELATIVE_SPREAD),
    )


def share_label(fraction: float) -> str:
    """比例（0–1）的標示文字：與正文／結論摘要表同一格式（``generate_concise_results_md.pct_small``）：1 % 以上兩位小數，未滿 1 % 三位有效數字。"""

    p = 100.0 * float(fraction)
    return f"{p:.2f}%" if p >= 1 else f"{p:#.3g}%"


def percent_axis(values: np.ndarray) -> np.ndarray:
    return 100.0 * np.asarray(values, dtype=float)


def _tick_text(value: float) -> str:
    return f"{value:g}"


def style_axes(axis: Any) -> None:
    """座標軸：與地圖類精簡版圖相同的框線（#222222、1.1 pt）、向外刻度記號（3.5 pt）、8 pt 標示；無格線；對數刻度，不用科學記號。"""

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    from .concise_style import FRAME_COLOUR, FRAME_LINEWIDTH, TICK_LENGTH_PT

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*X_LIM)
    axis.set_ylim(*Y_LIM_PCT)
    axis.xaxis.set_major_locator(FixedLocator(list(X_TICKS)))
    axis.yaxis.set_major_locator(FixedLocator(list(Y_TICKS_PCT)))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: _tick_text(value)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: _tick_text(value)))
    axis.tick_params(
        axis="both", which="major", direction="out", length=TICK_LENGTH_PT, width=0.8, color=FRAME_COLOUR,
        labelsize=8, pad=2, bottom=True, left=True, top=False, right=False, labelbottom=True, labelleft=True,
    )
    axis.grid(False)
    axis.set_xlabel(X_LABEL, fontsize=8, labelpad=2)
    axis.set_ylabel(Y_LABEL, fontsize=8, labelpad=2)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LINEWIDTH)
        spine.set_edgecolor(FRAME_COLOUR)
        spine.set_zorder(20)
    axis._traps_role = "line"  # noqa: SLF001  版面紀錄用；不是地圖軸，不進 concise 地圖檢查


def reference_marks(curves: dict[str, Any]) -> list[dict[str, Any]]:
    """x = 3 小時處的兩個標示：實際與隨機對照（兩組種子平均）之比例（百分比）、同曲線色、標示文字（等於正文與結論摘要表的數字）。"""

    i3 = int(np.nonzero(curves["ks"] == REFERENCE_HOURS)[0][0])
    out = []
    for series, colour, key in (("real", REAL_COLOUR, "real"), ("null", NULL_COLOUR, "null_mean")):
        fraction = float(curves[key][i3])
        out.append(dict(series=series, colour=colour, fraction=fraction, y_percent=100.0 * fraction, text=share_label(fraction)))
    return out


def label_clearances(fig: Any, axis: Any, curves: dict[str, Any], marks: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """每個標示文字外框（含字型下伸部）到各曲線折線、3 小時參考線、軸框的最短距離（pt）；任一小於 :data:`LABEL_CLEARANCE_PT` 即報錯（文字不可壓到曲線、參考線或圖框）。"""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    axes_box = axis.get_window_extent(renderer)
    ref_x = float(axis.transData.transform((REFERENCE_HOURS, 1.0))[0])
    ks = curves["ks"]
    polylines = {}
    for name, values in (("real", curves["real"]), ("null", curves["null_mean"])):
        keep = values > 0
        polylines[name] = axis.transData.transform(np.column_stack([ks[keep], percent_axis(values[keep])]))
    out: dict[str, dict[str, float]] = {}
    for mark in marks:
        box = mark["artist"].get_window_extent(renderer)
        x0, y0, x1, y1 = float(box.x0), float(box.y0), float(box.x1), float(box.y1)

        def to_rect(points: np.ndarray) -> float:
            dx = np.maximum(np.maximum(x0 - points[:, 0], points[:, 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - points[:, 1], points[:, 1] - y1), 0.0)
            return float(np.min(np.hypot(dx, dy)))

        dists = {}
        for name, poly in polylines.items():      # 折線以每段 200 點取樣
            samples = np.column_stack([np.concatenate([np.linspace(a[0], b[0], 200) for a, b in zip(poly[:-1], poly[1:])]),
                                       np.concatenate([np.linspace(a[1], b[1], 200) for a, b in zip(poly[:-1], poly[1:])])])
            dists[f"curve_{name}"] = to_rect(samples)
        dists["reference_line"] = x0 - ref_x      # 文字在參考線右側，左緣到線的距離
        dists["axes_frame"] = float(min(x0 - axes_box.x0, axes_box.x1 - x1, y0 - axes_box.y0, axes_box.y1 - y1))
        dists = {key: round(value * to_pt, 3) for key, value in dists.items()}
        for key, value in dists.items():
            assert value >= LABEL_CLEARANCE_PT, (curves["site"], mark["series"], key, value)
        out[mark["series"]] = dists
    return out


def _lifetime_axes():
    from . import figure_standard as std

    base = std.CONCISE_MAP_AXES
    return std.Box(base.x, std.LIFETIME_AXES_Y_INCH, std.WIDTH_INCH - 0.14 - base.x, base.h)


def render(curves: dict[str, Any], plt: Any) -> Any:
    """畫一個地點的圖（無標題）：畫布 6.3 × 3.98 英吋（v50）、主軸框左緣、框高與上緣（距畫布頂端 0.20 英吋）同聚集熱區圖（``CONCISE_MAP_AXES``）、右緣延伸到畫布右側留 0.14 英吋（折線圖沒有色條，不留空白邊），
    圖例在軸框下方（淺灰外框、8 pt，同其他精簡版圖）；圖例下緣距畫布底緣 0.0956 英吋、x 軸標籤底緣到圖例框頂緣約 0.18 英吋，與地圖類圖相同（因此畫布比地圖類圖矮 0.15 英吋）。"""

    from matplotlib.lines import Line2D

    from . import figure_standard as std

    facts = site_facts(curves)
    fig = plt.figure(figsize=(std.WIDTH_INCH, std.LIFETIME_CANVAS_HEIGHT_INCH))
    axis = std.add_box_axes(fig, _lifetime_axes())
    style_axes(axis)
    ks = curves["ks"]
    axis.axvline(REFERENCE_HOURS, color=REFERENCE_LINE_COLOUR, linewidth=REFERENCE_LINE_WIDTH_PT, zorder=1)
    null_mask = curves["null_mean"] > 0
    if facts["draw_seed_band"]:
        low = np.minimum(*curves["null_seed"])
        high = np.maximum(*curves["null_seed"])
        band = (low > 0) & (high > 0)
        axis.fill_between(ks[band], percent_axis(low[band]), percent_axis(high[band]), color=NULL_COLOUR, alpha=0.25, linewidth=0, zorder=2)
    axis.plot(ks[null_mask], percent_axis(curves["null_mean"][null_mask]), color=NULL_COLOUR, linestyle=NULL_DASHES, linewidth=NULL_LINEWIDTH_PT, solid_capstyle="butt", zorder=3)
    axis.plot(ks, percent_axis(curves["real"]), color=REAL_COLOUR, linewidth=REAL_LINEWIDTH_PT, solid_capstyle="butt", zorder=4)
    marks = reference_marks(curves)
    for mark in marks:      # v51: x = 3 小時處的實心點（同曲線色、無邊線）與其比例；文字同曲線色、8 pt、無白邊或底框，放在點的右上方
        axis.plot([REFERENCE_HOURS], [mark["y_percent"]], linestyle="none", marker="o", markersize=MARKER_DIAMETER_PT, markerfacecolor=mark["colour"], markeredgecolor=mark["colour"],
                  markeredgewidth=MARKER_EDGE_WIDTH_PT, color=mark["colour"], zorder=5, clip_on=False)
        mark["artist"] = axis.annotate(mark["text"], xy=(REFERENCE_HOURS, mark["y_percent"]), xytext=LABEL_OFFSET_PT, textcoords="offset points", ha="left", va="bottom",
                                       fontsize=LABEL_FONT_PT, color=mark["colour"], zorder=6, annotation_clip=False)
    facts["label_clearance_pt"] = label_clearances(fig, axis, curves, marks)
    handles = [
        Line2D([], [], color=REAL_COLOUR, linewidth=REAL_LINEWIDTH_PT),
        Line2D([], [], color=NULL_COLOUR, linestyle=NULL_DASHES, linewidth=NULL_LINEWIDTH_PT),
    ]
    box = _lifetime_axes()
    std.legend_below(fig, fig.get_size_inches()[1], handles, [REAL_LABEL, NULL_LABEL], ncol=2, framed=True, center_x_inch=box.x + 0.5 * box.w)
    fig._traps_lifetime_facts = facts  # noqa: SLF001
    fig._traps_lifetime_marks = [{k: v for k, v in m.items() if k != "artist"} for m in marks]  # noqa: SLF001
    return fig


__all__ = [
    "KIND", "SITES", "HIST_PATH", "X_LIM", "X_TICKS", "Y_LIM_PCT", "Y_TICKS_PCT", "REAL_COLOUR", "NULL_COLOUR", "REAL_LABEL", "NULL_LABEL", "X_LABEL", "Y_LABEL",
    "load_hist", "survival", "site_curves", "site_facts", "render", "style_axes", "share_label", "reference_marks", "MARKER_DIAMETER_PT", "LABEL_FONT_PT", "LABEL_OFFSET_PT",
]
