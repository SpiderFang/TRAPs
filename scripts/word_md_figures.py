"""Figures drawn for the Word method report (pipeline flowchart, 25-h filter response).

Output is JPG (<= 1600 px wide) under docs/reports/word/figures/.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from word_md_common import FIG_DIR  # noqa: E402


def _font():
    for name in ("Noto Serif CJK TC", "Noto Sans CJK TC", "Noto Sans CJK JP"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return name
    raise RuntimeError("no CJK font available for figure text")


def _save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / (stem + ".png")
    fig.savefig(png, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    jpg = FIG_DIR / (stem + ".jpg")
    im.save(jpg, quality=90)
    png.unlink()
    return jpg


def pipeline_flowchart():
    _font()
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F2F2", fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec="#333333", lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(p, q, label=None, dashed=False, lx=0.0, ly=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.0,
                                     color="#333333", linestyle="--" if dashed else "-"))
        if label:
            ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=8, ha="center",
                    va="center", color="#444444",
                    bbox=dict(fc="white", ec="none", pad=0.6))

    # left column (main chain)
    box(0.3, 10.6, 4.4, 1.1, "網路儲存（NAS）：表層月份快取\n（資料格式 3.0.0，1 公里規則網格）")
    box(0.3, 8.7, 4.4, 1.1, "伺服器本機固態硬碟\n（11 GB，SHA-256 逐檔核對）")
    box(0.3, 5.7, 4.4, 1.7, "階段 1　區域月份檢出\n96 個區域月份平行\n應變率 → s₁、e₂ → 核心 → 曲線", fc="#E7E6E6", bold=True)
    box(0.3, 3.2, 4.4, 1.7, "階段 2　期間追蹤\n標準時間軸 → 強核心篩選\n→ 串流追蹤 → 壽命與截尾", fc="#E7E6E6", bold=True)
    box(0.3, 0.5, 4.4, 1.7, "階段 3　彙整\n逐站統計、熱點、近岸診斷\n逐日區塊重抽樣、成果圖", fc="#E7E6E6", bold=True)
    # right column
    box(5.6, 8.7, 4.1, 1.1, "日產品建置\n25 小時置中平均")
    box(5.6, 6.0, 4.1, 1.1, "海岸遮罩附屬檔\n（校正海岸線版本）")
    box(5.6, 3.2, 4.1, 1.7, "隨機時間置換對照追蹤\n（亂數種子 1、2）\n同一追蹤設定、置換時間槽", fc="#FFFFFF")
    box(5.6, 0.5, 4.1, 1.7, "輸出\n逐站統計檔、逐格空間陣列\n成果圖、執行紀錄檔", fc="#FFFFFF")

    arrow((2.5, 10.6), (2.5, 9.8), "雜湊值驗證複製", lx=1.25)
    arrow((2.5, 8.7), (2.5, 7.4))
    arrow((4.7, 9.25), (5.6, 9.25), "")
    arrow((7.65, 8.7), (4.7, 7.0), "逐日資料", lx=0.55, ly=0.1)
    arrow((5.6, 6.55), (4.7, 6.55), "")
    arrow((2.5, 5.7), (2.5, 4.9), "分片檔：快照與檢出", lx=1.55)
    arrow((2.5, 3.2), (2.5, 2.2), "結構軌跡表與觀測", lx=1.1)
    arrow((4.7, 4.05), (5.6, 4.05), "", dashed=True)
    arrow((7.65, 3.2), (4.7, 1.5), "對照追蹤結果", dashed=True, lx=0.6, ly=0.35)
    arrow((4.7, 1.35), (5.6, 1.35))
    return _save(fig, "fig_pipeline")


def filter_response():
    _font()
    Ts = [12 + 0.05 * i for i in range(0, 1)]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))

    def H(T, N=25):
        return abs(math.sin(math.pi * N / T) / (N * math.sin(math.pi / T)))

    xs = [2 + i * 0.25 for i in range(0, int((200 - 2) / 0.25) + 1)]
    xs = [x for x in xs if abs(math.sin(math.pi / x)) > 1e-9]
    ax.semilogx(xs, [H(x) for x in xs], color="#1f4e79", lw=1.4)
    marks = [("M2", 12.4206), ("S2", 12.0), ("N2", 12.6583), ("K1", 23.9345), ("O1", 25.8193), ("P1", 24.0659),
             ("3 日", 72.0), ("7 日", 168.0)]
    for name, T in marks:
        ax.plot([T], [H(T)], "o", ms=4, color="#c00000")
    ax.annotate("M2、S2、N2\n約 0.6%–4%", xy=(12.4, H(12.4206)), xytext=(4.2, 0.45), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.annotate("K1、O1、P1\n約 3%–4%", xy=(24.5, H(24.0659)), xytext=(30, 0.3), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.annotate("3 日 0.813", xy=(72, H(72)), xytext=(45, 0.55), fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.annotate("7 日 0.964", xy=(168, H(168)), xytext=(95, 0.7), fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.set_xlabel("訊號週期 T（小時）")
    ax.set_ylabel("振幅響應 H(T)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(2, 200)
    ax.set_xticks([3, 6, 12, 24, 48, 72, 168])
    ax.set_xticklabels(["3", "6", "12", "24", "48", "72", "168"])
    ax.grid(True, which="major", color="#cccccc", lw=0.5)
    return _save(fig, "fig_filter_response")


def trim_white(im, pad=12, thresh=245):
    """Crop near-white margins (PIL bbox on the inverted near-white mask), keep a small pad."""
    from PIL import ImageChops
    g = im.convert("L").point(lambda v: 255 if v < thresh else 0)
    bb = g.getbbox()
    if not bb:
        return im
    l, t, r, b = bb
    return im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height)))


def convert_site_figures(sites):
    """Copy the selected result figures as JPG (<=1600 px wide)."""
    from word_md_common import RESULTS, RUNS

    out = {}
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for run_key, kinds in (("hourly", ["lifetime"]), ("daily", ["hotspots", "spatial", "lifetime", "long_tracks", "example"])):
        run = RUNS[run_key]
        for sid, _, _ in sites:
            for kind in kinds:
                src = RESULTS / run / "aggregate_report" / "figures" / sid / f"site_{sid}_{kind}.png"
                dst = FIG_DIR / f"{sid}_{run_key}_{kind}.jpg"
                im = trim_white(Image.open(src).convert("RGB"))
                if im.width > 1600:
                    im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
                im.save(dst, quality=85, optimize=True)
                out[(sid, run_key, kind)] = f"figures/{dst.name}"
    return out
