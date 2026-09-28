"""TRAPs 快照與五站統計成果圖。

本模組只處理已由 ``detection`` 和 CLI 驗證過的資料，不重新計算物理量。快照圖
把 `s1`、速度向量、核心及 e2 曲線放在同一張局部公尺座標圖，並在標題保留 UTC
時間與輸入狀態。摘要圖則明確畫出各站出現率的分子/分母、吸引強度與可用的壽命
統計；若尚未有跨時刻 tracking，壽命欄位會留白並標示資料不足，不會把單一快照
誤稱為壽命結果。
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .colorbar_layout import align_colorbars_to_axes
from .models import SnapshotResult

if TYPE_CHECKING:
    from .coastline import CoastlineGeometry


def _load_matplotlib() -> tuple[Any, Any]:
    """延遲載入 matplotlib 並固定無顯示器 backend，讓 SERVER CLI 可批次輸出。"""

    try:
        # SERVER/受限工作區的 home 可能不可寫；把 font cache 放到暫存區可避免每個
        # subprocess 都重新建立 cache，也不會改動專案或使用者設定檔。
        if not os.environ.get("MPLCONFIGDIR"):
            cache_dir = Path(tempfile.gettempdir()) / "traps-matplotlib"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - 由環境依賴 gate 覆蓋
        raise RuntimeError("繪圖需要 matplotlib；請安裝 traps-analysis 的繪圖相依") from error
    return matplotlib, plt


def _time_label(time_utc_ns: int | None) -> str:
    """把 UTC nanosecond 時間轉成穩定、可讀的標題字串。"""

    if time_utc_ns is None:
        return "UTC time unavailable"
    import datetime as _datetime

    timestamp = _datetime.datetime.fromtimestamp(int(time_utc_ns) / 1.0e9, tz=_datetime.UTC)
    return timestamp.isoformat(timespec="seconds")


def _site_font_properties() -> Any | None:
    """尋找能完整顯示五站繁中字形的既有字型。

    圖面只讀取系統既有字型；不同執行主機不保證有相同家族，因此逐字檢查
    glyph coverage。找不到時呼叫端改用 ASCII 站點 ID，不輸出缺字方框，也不
    下載字型或把本機絕對路徑寫入科學資料。
    """

    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    required = "貢寮龜山島西側新竹外海南灣連江"
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        face = FT2Font(str(path))
        if all(face.get_char_index(ord(character)) for character in required):
            return font_manager.FontProperties(fname=str(path))
    return None


def _metric_coastline_axes(snapshot: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """取得海岸線疊圖所需的 WGS84 與局部公尺座標，並先做軸契約檢查。

    海岸線 GeoJSON 的幾何座標是 CRS84 經緯度；成果圖的底圖仍使用 OCM
    snapshot 自己的局部公尺座標。因此這裡只核對兩套座標的存在性、有限值與
    rectilinear 軸對齊關係，再把原始 snapshot 軸傳給 ``draw_metric_coastline``
    進行轉換。這個檢查不會改變 ``valid``、ocean mask 或任何 TRAP 數值；若輸入
    缺少任一軸或軸折返，立即回報可定位的 ValueError，避免把岸線畫到錯誤位置。
    """

    required = ("lon", "lat", "x_m", "y_m")
    missing = [name for name in required if not hasattr(snapshot, name)]
    if missing:
        raise ValueError(f"coastline 疊圖需要 snapshot 欄位: {', '.join(missing)}")
    lon_axis = np.asarray(snapshot.lon, dtype=np.float64)
    lat_axis = np.asarray(snapshot.lat, dtype=np.float64)
    x_axis = np.asarray(snapshot.x_m, dtype=np.float64)
    y_axis = np.asarray(snapshot.y_m, dtype=np.float64)
    if lon_axis.ndim != 1 or lat_axis.ndim != 1:
        raise ValueError("coastline 疊圖需要一維 snapshot.lon 與 snapshot.lat 軸")
    if lon_axis.size < 2 or lat_axis.size < 2:
        raise ValueError("coastline 疊圖的 lon/lat 軸至少需要兩個格點")
    if not np.all(np.isfinite(lon_axis)) or not np.all(np.isfinite(lat_axis)):
        raise ValueError("coastline 疊圖的 lon/lat 軸必須全部有限")
    if not (np.all(np.diff(lon_axis) > 0.0) or np.all(np.diff(lon_axis) < 0.0)):
        raise ValueError("coastline 疊圖的 lon 軸必須嚴格單調")
    if not (np.all(np.diff(lat_axis) > 0.0) or np.all(np.diff(lat_axis) < 0.0)):
        raise ValueError("coastline 疊圖的 lat 軸必須嚴格單調")
    if x_axis.ndim not in (1, 2) or y_axis.ndim not in (1, 2):
        raise ValueError("coastline 疊圖的 x_m/y_m 必須是一維軸或二維 rectilinear 網格")
    if x_axis.ndim == 2:
        if x_axis.shape != y_axis.shape:
            raise ValueError("coastline 疊圖的 x_m/y_m 二維 shape 必須一致")
        if not np.allclose(x_axis, x_axis[0, :][None, :], rtol=0.0, atol=1.0e-6):
            raise ValueError("coastline 疊圖的 x_m 必須是 rectilinear 軸對齊網格")
        if not np.allclose(y_axis, y_axis[:, 0][:, None], rtol=0.0, atol=1.0e-6):
            raise ValueError("coastline 疊圖的 y_m 必須是 rectilinear 軸對齊網格")
        x_check = x_axis[0, :]
        y_check = y_axis[:, 0]
    else:
        x_check = x_axis
        y_check = y_axis
    if not np.all(np.isfinite(x_check)) or not np.all(np.isfinite(y_check)):
        raise ValueError("coastline 疊圖的 x_m/y_m 軸必須全部有限")
    if not (np.all(np.diff(x_check) > 0.0) or np.all(np.diff(x_check) < 0.0)):
        raise ValueError("coastline 疊圖的 x_m 軸必須嚴格單調")
    if not (np.all(np.diff(y_check) > 0.0) or np.all(np.diff(y_check) < 0.0)):
        raise ValueError("coastline 疊圖的 y_m 軸必須嚴格單調")
    return lon_axis, lat_axis, x_axis, y_axis


def render_snapshot_figure(
    snapshot: Any,
    result: SnapshotResult,
    output_path: str | Path,
    *,
    state_label: str,
    source_label: str,
    max_arrows: int = 900,
    dpi: int = 150,
    coastline: CoastlineGeometry | None = None,
) -> Path:
    """輸出一張帶 s1、速度向量與 TRAP 曲線的單時刻 PNG。

    ``snapshot`` 需提供 `time_utc_ns`、`u`、`v`、`valid`、`x_m`、`y_m` 欄位；這個
    介面刻意與 ``ocm.SurfaceSnapshot`` 對齊，但不直接依賴 OCM 模組，便於合成資料
    和後續讀取器重用。s1 顏色只顯示 ``result.derivative_valid`` 內的格點，速度
    箭頭只取有限且有效的格點，避免圖面暗示 mask 外存在流速。若提供
    ``coastline``，其 CRS84 向量輪廓會使用同一 snapshot 的經緯度與局部公尺軸
    轉換後疊在底圖上；這是 visual-only 圖面資訊，不會改變 ocean mask、flow
    domain 或 TRAP 檢出值。
    """

    _, plt = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(snapshot.x_m)
    y = np.asarray(snapshot.y_m)
    u = np.asarray(snapshot.u, dtype=np.float64)
    v = np.asarray(snapshot.v, dtype=np.float64)
    valid = np.asarray(snapshot.valid, dtype=bool)
    s1 = np.ma.masked_where(~result.derivative_valid, result.s1_sinv)

    figure, axis = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
    mesh = axis.pcolormesh(x, y, s1, shading="auto", cmap="coolwarm")
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label(r"$s_1$ ($s^{-1}$)")

    if coastline is not None:
        # 岸線需使用當前 snapshot 的座標，而不能重用另一個 domain 的 grid；
        # zorder 低於後續曲線與核心，確保視覺參考線不遮蔽 TRAP 證據。
        lon_axis, lat_axis, x_axis, y_axis = _metric_coastline_axes(snapshot)
        from .coastline import draw_metric_coastline

        draw_metric_coastline(
            axis,
            coastline,
            lon_axis=lon_axis,
            lat_axis=lat_axis,
            x_axis=x_axis,
            y_axis=y_axis,
            color="#263238",
            linewidth=0.75,
            zorder=1.5,
        )

    finite = valid & np.isfinite(u) & np.isfinite(v) & result.derivative_valid
    valid_indices = np.argwhere(finite)
    if valid_indices.size:
        stride = max(1, int(np.ceil(np.sqrt(valid_indices.shape[0] / max(1, max_arrows)))))
        selected = valid_indices[::stride]
        rows = selected[:, 0]
        cols = selected[:, 1]
        axis.quiver(
            x[rows, cols],
            y[rows, cols],
            u[rows, cols],
            v[rows, cols],
            color="0.20",
            alpha=0.65,
            width=0.0018,
            angles="xy",
            scale_units="xy",
            scale=None,
            label="surface velocity",
        )

    for detection_index, detection in enumerate(result.detections, start=1):
        points = detection.curve_points_m
        line_label = "TRAP curve" if detection_index == 1 else None
        axis.plot(points[:, 0], points[:, 1], color="#111111", linewidth=1.8, label=line_label)
        axis.scatter(
            [detection.core_xy_m[0]],
            [detection.core_xy_m[1]],
            color="#e31a1c",
            edgecolor="white",
            linewidth=0.6,
            s=35,
            zorder=4,
            label="negative local s1 core" if detection_index == 1 else None,
        )

    time_label = _time_label(getattr(snapshot, "time_utc_ns", None))
    # 長 domain ID 放在第二行，避免圖標題被右側 s1 色階裁切。
    axis.set_title(f"TRAP surface attraction | {time_label} | {state_label}\n{source_label}", fontsize=10)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal", adjustable="box")
    # constrained_layout 與等比例局部公尺 axes 完成縮放後，色條必須對齊主圖
    # 的實際座標框；這只改畫布排版，不改 s1 色階、TRAP 曲線或速度資料。
    align_colorbars_to_axes(figure, ((axis, colorbar.ax),))
    axis.grid(alpha=0.2, linewidth=0.5)
    handles, _labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best", fontsize=8)
    if coastline is not None:
        # GeoJSON 僅有平面輪廓，不能被解讀成水深、潮位或新的 ocean mask；
        # 將 OSM/ODbL attribution 留在單張 PNG 內，避免圖脫離 manifest 後失去來源。
        figure.text(
            0.5,
            0.012,
            "OpenStreetMap (ODbL) coastline overlay | visual-only; no ocean mask, flow domain, or TRAP values changed",
            ha="center",
            fontsize=7,
            color="#455a64",
        )
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


def render_site_summary_figure(
    site_statistics: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    state_label: str,
    dpi: int = 150,
) -> Path:
    """輸出目前有研究面積之站點出現率、吸引強度與未截尾 track 觀測跨度圖。

    每一筆 station row 應至少含 `study_site_name_zh`、`occurrence_rate`、
    `occurrence_numerator`、`occurrence_denominator`、`mean_abs_s1_sinv` 及
    `mean_complete_track_duration_hours`。單區 catalog 的其他三區面積是 0，不能畫
    成 0/0 的站點比較；左／右截尾 track 不能併入未截尾平均。時長是首末檢出
    差，單時刻的 0 小時僅代表時間解析度不足，不能稱為物理壽命。若主機缺繁中
    字型，x 軸退回可讀的 ASCII site ID，避免圖面出現缺字方框。
    """

    _, plt = _load_matplotlib()
    rows = [dict(row) for row in site_statistics if float(row.get("site_area_m2") or 0.0) > 0.0]
    if not rows:
        raise ValueError("沒有具有效統計面積的站點可繪圖")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    font_properties = _site_font_properties()
    names = [
        str(row.get("study_site_name_zh") or row.get("study_site_id"))
        if font_properties is not None else str(row.get("study_site_id"))
        for row in rows
    ]
    occurrence = [
        float(row["occurrence_rate"]) * 100.0
        if row.get("occurrence_rate") is not None
        else np.nan
        for row in rows
    ]
    intensity = [
        float(row["mean_abs_s1_sinv"]) if row.get("mean_abs_s1_sinv") is not None else np.nan
        for row in rows
    ]
    lifetime = [
        float(row["mean_complete_track_duration_hours"])
        if row.get("mean_complete_track_duration_hours") is not None
        else np.nan
        for row in rows
    ]

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 5.6), constrained_layout=True)
    positions = np.arange(len(names), dtype=float)
    axes[0].bar(positions, occurrence, color="#377eb8")
    axes[0].set_ylabel("Occurrence rate (%)")
    axes[0].set_ylim(bottom=0.0)
    for index, row in enumerate(rows):
        numerator = row.get("occurrence_numerator", 0)
        denominator = row.get("occurrence_denominator", 0)
        axes[0].text(index, occurrence[index] if np.isfinite(occurrence[index]) else 0.0, f"{numerator}/{denominator}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(positions, intensity, color="#e41a1c")
    axes[1].set_ylabel(r"Mean $|s_1|$ ($s^{-1}$)")
    axes[1].set_ylim(bottom=0.0)

    axes[2].bar(positions, lifetime, color="#4daf4a")
    axes[2].set_ylabel("Mean uncensored observed span (h)")
    axes[2].set_ylim(bottom=0.0)
    if not np.any(np.isfinite(lifetime)):
        axes[2].text(0.5, 0.5, "No complete tracks", transform=axes[2].transAxes, ha="center", va="center")

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(names, rotation=25, ha="right", fontproperties=font_properties)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    figure.suptitle(f"TRAP site summary | {state_label}")
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


__all__ = ["render_site_summary_figure", "render_snapshot_figure"]
