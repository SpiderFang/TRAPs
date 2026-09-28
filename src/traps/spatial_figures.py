"""TRAP 逐格空間統計的成果圖。

本模組採用 Kunz 等人空間分布四分圖與 Ghosh 等人出現日數圖的資訊安排，將本專案
實際可計算的 TRAP 統計放入同一張 2×2 地理圖：逐格有效 snapshot 分母、核心吸引
強度、曲線通過率，以及未截尾 track 的平均觀測跨度。首末檢出時間差只受取樣
解析度與 catalog 時窗支持，不能稱為物理壽命。圖面不放入尚未取得的獨立漂流器、海廢
紀錄或潮相分層；截尾 track 只在圖下注記與 SpatialSummary metadata 中保留為資料
狀態，不被畫成完整生命週期。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .coastline import CoastlineGeometry, draw_geographic_coastline
from .colorbar_layout import align_colorbars_to_axes
from .figures import _site_font_properties
from .spatial import SpatialSummary


def _load_matplotlib() -> tuple[Any, Any]:
    """延遲載入無顯示器 matplotlib，避免匯入空間統計時要求 GUI。"""

    try:
        if not os.environ.get("MPLCONFIGDIR"):
            cache_dir = Path(tempfile.gettempdir()) / "traps-matplotlib"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - 由環境依賴 gate 覆蓋
        raise RuntimeError("空間成果圖需要 matplotlib；請安裝 traps-analysis 的繪圖相依") from error
    return matplotlib, plt


def _display_field(summary: SpatialSummary, field: np.ndarray) -> np.ndarray:
    """把零有效分母格點轉為 NaN，避免圖面把資料缺口誤標成零。"""

    array = np.asarray(field, dtype=np.float64)
    denominator = np.asarray(summary.valid_snapshot_count, dtype=np.int64)
    return np.where(denominator > 0, array, np.nan)


def render_spatial_summary_figure(
    summary: SpatialSummary,
    output_path: str | Path,
    *,
    display_name: str | None = None,
    coastline: CoastlineGeometry | None = None,
    dpi: int = 150,
) -> Path:
    """輸出 Kunz C2／Ghosh D1 借鏡的 TRAP 空間四分圖。

    ``summary`` 必須由 :func:`traps.spatial.build_spatial_summary` 建立，因為該物件已
    保存 catalog 與 ``SurfaceMonth`` 的時間對應、逐格分子／分母及完整／截尾 track
    規則。四個分圖的單位分別是 count、``s^-1``、百分比與 hour；地理軸直接使用
    OCM 驗證過的 WGS84 ``lon``／``lat`` 軸。``display_name`` 由已核對的研究設計
    提供，C 區顯示南灣但保留上游 domain ID 作 metadata 追溯。若沒有任何完整 track，右下圖保留空白
    並明示資料不足，不以 0 小時代替。

    圖面是模型內部 TRAP 結構統計，`PILOT_ONLY` 等狀態會寫入標題；它不代表海廢
    濃度、污染來源或獨立觀測驗證。若提供已驗證的 ``coastline``，每個經緯度
    分圖只疊加與 OCM flow-domain bbox 相交的 outline；海岸線不會改變數值欄位、
    ocean mask 或圖面軸範圍。
    """

    if not isinstance(summary, SpatialSummary):
        raise TypeError("summary 必須是 SpatialSummary")
    _, plt = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lon = np.asarray(summary.lon, dtype=np.float64)
    lat = np.asarray(summary.lat, dtype=np.float64)
    fields = (
        ("Valid snapshot denominator", _display_field(summary, summary.valid_snapshot_count), "count", "viridis"),
        ("Mean core attraction", _display_field(summary, summary.mean_abs_core_s1_sinv), r"$|s_1|$ ($s^{-1}$)", "magma"),
        ("TRAP curve passage", _display_field(summary, summary.curve_pass_rate_percent), "snapshot passage (%)", "YlOrRd"),
        ("Uncensored track observed span", _display_field(summary, summary.mean_complete_track_duration_hours), "mean observed span (h)", "plasma"),
    )
    # 預留底部註腳及上方來源狀態，避免四張地理圖的經緯度標籤與註腳碰撞。
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), constrained_layout=False)
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.13, top=0.86, wspace=0.30, hspace=0.50)
    axes_flat = tuple(axes.flat)
    meshes: list[Any] = []
    colorbar_pairs: list[tuple[Any, Any]] = []
    for axis, (title, field, colorbar_label, cmap) in zip(axes_flat, fields, strict=True):
        masked = np.ma.masked_invalid(field)
        # 有效分母與通過率使用固定物理範圍，避免全域皆四時刻有效時自動色階
        # 縮成 3.6–4.4 或不同區域百分比色階不可比較。
        limits = {"Valid snapshot denominator": (0.0, float(len(summary.snapshot_indices))), "TRAP curve passage": (0.0, 100.0)}
        kwargs = {"vmin": limits[title][0], "vmax": limits[title][1]} if title in limits else {}
        mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap=cmap, **kwargs)
        meshes.append(mesh)
        colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
        colorbar.set_label(colorbar_label)
        colorbar_pairs.append((axis, colorbar.ax))
        axis.set_title(title)
        axis.set_xlabel("Longitude (°E)")
        axis.set_ylabel("Latitude (°N)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2, linewidth=0.5)
        if coastline is not None:
            # bbox 由 OCM 地理軸決定；GeoJSON 僅作視覺輪廓，域外 feature 不得
            # 透過 matplotlib autoscale 把四分圖拉離既定 flow domain。
            draw_geographic_coastline(
                axis,
                coastline,
                bbox=(float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))),
                color="#263238",
                linewidth=0.65,
                zorder=6.0,
            )

    # 四個地理 axes 都設定為等比例；必須等所有色條建立且所有 axes 完成設定後
    # 一次量測，才能把每條色條鎖在其對應主圖的真實 y0/height，而不是 GridSpec
    # 預留欄位的高度。這不會改變任何統計欄位或各面板的色階政策。
    align_colorbars_to_axes(figure, colorbar_pairs)

    # 沒有完整 track 時，NaN 圖面本身不會向讀者說明原因；顯式文字把它與物理零值區分。
    if not np.any(np.isfinite(summary.mean_complete_track_duration_hours)):
        axes_flat[3].text(
            0.5,
            0.5,
            "No uncensored track spans\n(data unavailable; not 0 h)",
            transform=axes_flat[3].transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

    state = str(summary.science_status or summary.state_label)
    figure.suptitle(
        f"TRAP spatial summary | {display_name or summary.domain_id} | {summary.month} | {state}\n"
        "Kunz C2 / Ghosh D1 presentation analogue; model structural statistics",
        fontproperties=_site_font_properties(),
        fontsize=13,
    )
    figure.text(
        0.5, 0.04,
        f"UTC snapshots: {len(summary.snapshot_indices)} | passage counts each valid cell once per snapshot | "
        "span = first-to-last detection (0 h unresolved)",
        ha="center", fontsize=8,
    )
    coastline_note = "coastline overlay: CRS84 outline only" if coastline is not None else "coastline overlay unavailable"
    figure.text(0.5, 0.02, f"Model TRAP structure only; no independent debris observation overlay | {coastline_note}", ha="center", fontsize=8, color="#7f0000")
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


__all__ = ["render_spatial_summary_figure"]
