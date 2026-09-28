"""依文獻圖型整理 TRAP 方法比較與結構生命週期圖。

本模組只接受已通過上游資料檢查的 ``SurfaceSnapshot``、``SnapshotResult`` 或
CLI ``catalog``，負責將本專案的數值欄位排成可審核的研究圖面。圖型借鏡已保存於
``data/literature`` 的研究問題與分圖安排，但所有曲線、散度與壽命數值都重新由
本專案輸入計算，不複製原文圖像或外部研究的觀測值；track 圖呈現的是可見觀測跨度，
不是可直接辨識的物理存活時間。

三類輸出有明確的證據邊界：

* ``render_mechanism_figure`` 是原創幾何示意，所有箭頭是概念向量，不能解讀成
  OCM 流場、觀測漂流器或海廢物體軌跡。
* ``render_method_comparison_figure`` 的流線是單一 UTC snapshot 的 Eulerian
  瞬時診斷；TRAP 曲線來自 ``SnapshotResult``，散度由同一張表層速度場計算，三者
  都不是獨立觀測。
* ``render_lifetime_distribution_figure`` 與
  ``render_lifecycle_intensity_figure`` 只把左右邊界及資料缺口截尾的 track 排除
  於完整觀測跨度統計；截尾樣本會另列為觀測跨度下界。即使未截尾，首末檢出時間差
  仍受 snapshot 時間解析度限制。這些 track 是 Eulerian TRAP 結構的追蹤，不是
  Lagrangian 粒子 residence time。

所有圖標題與註腳保留 UTC、局部公尺座標、物理單位及 ``state_label``；短期試算
應傳入 ``PILOT_ONLY``，概念示意預設使用 ``CONCEPT_ONLY``。本模組不接受或補造潮
相、獨立漂流器、海廢觀測等附加資料，這些條件圖由後續資料介面負責。
"""

from __future__ import annotations

import datetime as _datetime
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .colorbar_layout import align_colorbars_to_axes
from .figures import _load_matplotlib, _metric_coastline_axes, _site_font_properties
from .models import SnapshotResult
from .ocm import SurfaceSnapshot

if TYPE_CHECKING:
    from .coastline import CoastlineGeometry

NANOSECONDS_PER_HOUR = 3_600_000_000_000
"""UTC nanosecond 時間轉成小時的固定倍率。"""


def _require(condition: bool, message: str) -> None:
    """在輸入不符合圖面資料契約時立即停止，避免輸出誤導性圖。"""

    if not condition:
        raise ValueError(message)


def _time_label(time_utc_ns: int | None) -> str:
    """將 UTC nanosecond 時間轉成圖標題使用的 ISO-8601 字串。"""

    if time_utc_ns is None:
        return "UTC time unavailable"
    timestamp = _datetime.datetime.fromtimestamp(int(time_utc_ns) / 1.0e9, tz=_datetime.UTC)
    return timestamp.isoformat(timespec="seconds")


def _output_path(output_path: str | Path) -> Path:
    """建立單一輸出父目錄並回傳路徑；不清除既有檔案。"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _surface_axes(
    snapshot: SurfaceSnapshot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """驗證 ``SurfaceSnapshot`` 並回傳單調 x/y 軸、速度與有效遮罩。

    OCM 目前以 `(lat, lon)` 的二維公尺格網保存 ``x_m``、``y_m``；函式也接受同樣
    語意的一維軸，方便合成測試。輸出速度仍排列為 `(ny, nx)`，第一軸代表 y/row、
    第二軸代表 x/column。非軸對齊或座標不遞增的網格不能直接供 matplotlib
    ``streamplot`` 使用，會被拒絕而不是默默重排。
    """

    _require(isinstance(snapshot, SurfaceSnapshot), "snapshot 必須是 SurfaceSnapshot")
    u = np.asarray(snapshot.u, dtype=np.float64)
    v = np.asarray(snapshot.v, dtype=np.float64)
    valid = np.asarray(snapshot.valid, dtype=bool)
    _require(u.ndim == 2 and v.shape == u.shape and valid.shape == u.shape, "u/v/valid shape 必須一致且為二維")
    x = np.asarray(snapshot.x_m, dtype=np.float64)
    y = np.asarray(snapshot.y_m, dtype=np.float64)
    if x.ndim == 1:
        _require(x.size == u.shape[1], "x_m 一維長度與速度場不符")
        x_axis = x
    else:
        _require(x.ndim == 2 and x.shape == u.shape, "x_m 必須是一維軸或與速度場同 shape")
        x_axis = x[0, :]
        _require(np.allclose(x, x_axis[None, :], rtol=0.0, atol=1.0e-6), "x_m 必須是軸對齊 rectilinear 座標")
    if y.ndim == 1:
        _require(y.size == u.shape[0], "y_m 一維長度與速度場不符")
        y_axis = y
    else:
        _require(y.ndim == 2 and y.shape == u.shape, "y_m 必須是一維軸或與速度場同 shape")
        y_axis = y[:, 0]
        _require(np.allclose(y, y_axis[:, None], rtol=0.0, atol=1.0e-6), "y_m 必須是軸對齊 rectilinear 座標")
    _require(x_axis.size >= 3 and y_axis.size >= 3, "方法比較至少需要 3 x 3 格點")
    _require(np.all(np.isfinite(x_axis)) and np.all(np.isfinite(y_axis)), "x_m/y_m 必須全部有限")
    _require(np.all(np.diff(x_axis) > 0.0) and np.all(np.diff(y_axis) > 0.0), "x_m/y_m 必須嚴格遞增")
    return x_axis, y_axis, u, v, valid


def _masked_field(field: np.ndarray, valid: np.ndarray) -> np.ma.MaskedArray:
    """將非有限或無效海域遮蔽，避免把 mask 外畫成物理結果。"""

    array = np.asarray(field, dtype=np.float64)
    return np.ma.masked_where(~valid | ~np.isfinite(array), array)


def _divergence_sinv(
    u: np.ndarray,
    v: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """以公尺制中心差分估計表層速度散度，單位為 ``s^-1``。

    ``np.gradient`` 會在有效邊界使用單邊差分；與海岸相鄰的梯度若受無效格點影響，
    仍會由 ``valid`` 遮蔽。這張圖只做單時刻方法對照，不把散度當作 TRAP 法向
    吸引量，也不宣稱它是獨立觀測。
    """

    u_safe = np.where(valid & np.isfinite(u), u, np.nan)
    v_safe = np.where(valid & np.isfinite(v), v, np.nan)
    du_dy, du_dx = np.gradient(u_safe, y_axis, x_axis, edge_order=1)
    dv_dy, dv_dx = np.gradient(v_safe, y_axis, x_axis, edge_order=1)
    del du_dy, dv_dx
    divergence = du_dx + dv_dy
    return np.where(valid & np.isfinite(divergence), divergence, np.nan)


def _plot_trap_curves(axis: Any, result: SnapshotResult) -> int:
    """在方法比較圖上畫合格／未合格曲線，並回傳曲線數量。"""

    count = 0
    qualified_count = 0
    for detection in result.detections:
        points = np.asarray(detection.curve_points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            continue
        qualified = bool(detection.is_qualified)
        axis.plot(
            points[:, 0],
            points[:, 1],
            color="#111111" if qualified else "#777777",
            linewidth=1.7 if qualified else 1.0,
            linestyle="-" if qualified else "--",
            alpha=0.95 if qualified else 0.65,
            label="qualified TRAP curve" if qualified and qualified_count == 0 else None,
        )
        axis.scatter(
            [float(detection.core_xy_m[0])],
            [float(detection.core_xy_m[1])],
            s=30,
            color="#e31a1c" if qualified else "#999999",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label="TRAP core" if qualified and qualified_count == 0 else None,
        )
        count += 1
        if qualified:
            qualified_count += 1
    return count


def render_mechanism_figure(
    output_path: str | Path,
    *,
    state_label: str = "CONCEPT_ONLY",
    source_label: str = "TRAP conceptual mechanism",
    dpi: int = 150,
) -> Path:
    """輸出原創 TRAP 法向吸引與沿線輸送的概念示意圖。

    圖中的速度場以人工指定的公尺／秒箭頭展示幾何關係，沒有讀取 OCM 或觀測資料。
    中央黑線代表概念性的吸引骨架，藍色箭頭表示往骨架的法向分量，橘色箭頭表示
    沿線輸送；兩者不能被解讀成粒子路徑。標題與註腳明確寫出 ``CONCEPT_ONLY``
    或呼叫端傳入的科學狀態，避免示意圖混入正式結果圖組。
    """

    _, plt = _load_matplotlib()
    output = _output_path(output_path)
    x = np.linspace(-20_000.0, 20_000.0, 21)
    y = np.linspace(-10_000.0, 10_000.0, 11)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    # 此處只建立可視化概念向量；量級與方向不能當作 OCM 速度資料或觀測值。
    normal_v = -0.000012 * grid_y
    tangent_u = 0.12 + 0.03 * np.cos(grid_x / 7_000.0)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), constrained_layout=False)
    # 底部註腳包含方法邊界，需預留空間避免與 x 軸標籤重疊或被裁切。
    figure.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.20, wspace=0.22)
    axis = axes[0]
    axis.quiver(
        grid_x,
        grid_y,
        np.zeros_like(grid_x),
        normal_v,
        color="#377eb8",
        alpha=0.75,
        angles="xy",
        scale_units="xy",
        scale=0.0001,
        width=0.002,
        label="normal attraction (concept)",
    )
    axis.quiver(
        grid_x,
        grid_y,
        tangent_u,
        np.zeros_like(grid_y),
        color="#e66101",
        alpha=0.75,
        angles="xy",
        scale_units="xy",
        scale=0.0001,
        width=0.002,
        label="along-line transport (concept)",
    )
    axis.plot(x, np.zeros_like(x), color="#111111", linewidth=2.2, label="conceptual TRAP skeleton")
    axis.scatter([0.0], [0.0], color="#e31a1c", edgecolor="white", zorder=4, label="conceptual core")
    axis.set_title("Geometry of an attracting profile")
    axis.set_xlabel("x (m; conceptual)")
    axis.set_ylabel("y (m; conceptual)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(loc="upper left", fontsize=8)

    axis = axes[1]
    axis.plot([-1_000.0, 1_000.0], [0.0, 0.0], color="#111111", linewidth=2.2)
    axis.annotate(
        "normal attraction\n(v toward profile)",
        xy=(0.0, 0.0),
        xytext=(-750.0, 750.0),
        arrowprops={"arrowstyle": "->", "color": "#377eb8", "linewidth": 1.8},
        color="#255c91",
        ha="center",
        fontsize=9,
    )
    axis.annotate(
        "along-line transport\n(u along profile)",
        xy=(650.0, 0.0),
        xytext=(350.0, -750.0),
        arrowprops={"arrowstyle": "->", "color": "#e66101", "linewidth": 1.8},
        color="#a23b00",
        ha="center",
        fontsize=9,
    )
    axis.scatter([0.0], [0.0], color="#e31a1c", edgecolor="white", zorder=4)
    axis.set_xlim(-1_100.0, 1_100.0)
    axis.set_ylim(-1_100.0, 1_100.0)
    axis.set_title("Normal/tangential decomposition")
    axis.set_xlabel("local profile coordinate (m; conceptual)")
    axis.set_ylabel("normal coordinate (m; conceptual)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2, linewidth=0.5)

    figure.suptitle(f"TRAP mechanism schematic | {state_label}\n{source_label}", fontsize=11)
    figure.text(
        0.5,
        0.01,
        "CONCEPTUAL SCHEMATIC ONLY • no OCM value • no observation • arrows are not particle trajectories",
        ha="center",
        fontsize=8,
        color="#7f0000",
    )
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


def render_method_comparison_figure(
    snapshot: SurfaceSnapshot,
    result: SnapshotResult,
    output_path: str | Path,
    *,
    state_label: str,
    source_label: str,
    stream_density: float = 1.0,
    dpi: int = 150,
    coastline: CoastlineGeometry | None = None,
) -> Path:
    """輸出同一 OCM UTC snapshot 的 TRAP、瞬時流線與散度三分圖。

    左圖只畫 ``SnapshotResult`` 的 s1 與 e2 TRAP 曲線；中圖畫 Eulerian 瞬時流線，
    明確標成 ``instantaneous``，不能視為粒子或漂流器觀測軌跡；右圖則由表層
    ``u/v`` 公尺制速度計算 ``∇·u``，單位為 ``s^-1``。三圖共用同一時間與局部公尺
    座標，這使方法差異可追溯，但不提供獨立驗證。若提供 ``coastline``，三個
    局部公尺分圖都會疊加同一份 CRS84 海岸線向量輪廓；轉換使用此 snapshot 的
    WGS84 經緯度與局部公尺軸，並以較低 z-order 放在 TRAP 曲線與核心下方。
    海岸線是 visual-only OSM/ODbL 參考，不會改變 ocean mask、flow domain 或數值。
    """

    _, plt = _load_matplotlib()
    _require(np.isfinite(stream_density) and stream_density > 0.0, "stream_density 必須是正有限值")
    x_axis, y_axis, u, v, valid = _surface_axes(snapshot)
    _require(isinstance(result, SnapshotResult), "result 必須是 SnapshotResult")
    _require(result.grid_shape == u.shape, "SnapshotResult 與 SurfaceSnapshot shape 不符")
    output = _output_path(output_path)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing="xy")
    divergence = _divergence_sinv(u, v, x_axis, y_axis, valid)
    coastline_axes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    if coastline is not None:
        # 三個分圖共用同一份經緯度／公尺軸轉換，避免各分圖因重算而產生不同岸線。
        coastline_axes = _metric_coastline_axes(snapshot)
        from .coastline import draw_metric_coastline

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), constrained_layout=False)
    # 方法限制註腳必須獨立於三個 x 軸，否則縮放後會壓到座標單位。
    figure.subplots_adjust(left=0.04, right=0.96, top=0.78, bottom=0.20, wspace=0.36)
    s1 = _masked_field(result.s1_sinv, result.derivative_valid)
    s1_finite = np.asarray(s1.compressed(), dtype=np.float64)
    if s1_finite.size:
        s1_limit = float(np.nanpercentile(np.abs(s1_finite), 98.0))
        s1_limit = max(s1_limit, 1.0e-12)
    else:
        s1_limit = 1.0e-12
    mesh = axes[0].pcolormesh(
        grid_x,
        grid_y,
        s1,
        shading="auto",
        cmap="coolwarm",
        vmin=-s1_limit,
        vmax=s1_limit,
    )
    colorbar = figure.colorbar(mesh, ax=axes[0], pad=0.02)
    colorbar.set_label(r"$s_1$ ($s^{-1}$)")
    if coastline_axes is not None:
        lon_axis, lat_axis, metric_x, metric_y = coastline_axes
        draw_metric_coastline(
            axes[0],
            coastline,
            lon_axis=lon_axis,
            lat_axis=lat_axis,
            x_axis=metric_x,
            y_axis=metric_y,
            color="#263238",
            linewidth=0.75,
            zorder=1.5,
        )
    curve_count = _plot_trap_curves(axes[0], result)
    if curve_count == 0:
        axes[0].text(0.5, 0.5, "No TRAP detections", transform=axes[0].transAxes, ha="center", va="center")
    axes[0].set_title("TRAP detector: s1 and e2 curves")

    stream_u = np.ma.masked_where(~valid | ~np.isfinite(u), u)
    stream_v = np.ma.masked_where(~valid | ~np.isfinite(v), v)
    try:
        axes[1].streamplot(
            x_axis,
            y_axis,
            stream_u,
            stream_v,
            density=float(stream_density),
            color="#2166ac",
            linewidth=0.8,
        )
        stream_label = "instantaneous Eulerian streamlines"
    except (TypeError, ValueError):
        # 部分遮罩或極小網格可能令 matplotlib 的 streamplot 插值失敗；退回有效格點
        # 箭頭仍保留方法對照，但標題說明它不是完整流線，避免假裝成功繪製。
        finite = valid & np.isfinite(u) & np.isfinite(v)
        axes[1].quiver(grid_x[finite], grid_y[finite], u[finite], v[finite], color="#2166ac", alpha=0.7)
        stream_label = "instantaneous Eulerian vectors (streamline fallback)"
    if coastline_axes is not None:
        lon_axis, lat_axis, metric_x, metric_y = coastline_axes
        draw_metric_coastline(
            axes[1],
            coastline,
            lon_axis=lon_axis,
            lat_axis=lat_axis,
            x_axis=metric_x,
            y_axis=metric_y,
            color="#263238",
            linewidth=0.75,
            zorder=1.5,
        )
    axes[1].set_title("Eulerian flow diagnostic")
    axes[1].text(0.02, 0.02, stream_label, transform=axes[1].transAxes, fontsize=8, color="#174a7e")

    finite_divergence = divergence[np.isfinite(divergence)]
    if finite_divergence.size:
        divergence_limit = float(np.nanpercentile(np.abs(finite_divergence), 98.0))
        divergence_limit = max(divergence_limit, 1.0e-12)
    else:
        divergence_limit = 1.0e-12
    divergence_mesh = axes[2].pcolormesh(
        grid_x,
        grid_y,
        _masked_field(divergence, valid),
        shading="auto",
        cmap="RdBu_r",
        vmin=-divergence_limit,
        vmax=divergence_limit,
    )
    divergence_colorbar = figure.colorbar(divergence_mesh, ax=axes[2], pad=0.02)
    divergence_colorbar.set_label(r"$\nabla\cdot u$ ($s^{-1}$)")
    if coastline_axes is not None:
        lon_axis, lat_axis, metric_x, metric_y = coastline_axes
        draw_metric_coastline(
            axes[2],
            coastline,
            lon_axis=lon_axis,
            lat_axis=lat_axis,
            x_axis=metric_x,
            y_axis=metric_y,
            color="#263238",
            linewidth=0.75,
            zorder=1.5,
        )
    axes[2].set_title("Velocity divergence")

    for axis in axes:
        axis.set_xlabel("x (m; local metric)")
        axis.set_ylabel("y (m; local metric)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2, linewidth=0.5)
    time_label = _time_label(snapshot.time_utc_ns)
    figure.suptitle(
        f"OCM single-snapshot method comparison | {time_label} UTC | {state_label}\n{source_label}",
        fontproperties=_site_font_properties(),
        fontsize=11,
    )
    method_note = (
        "Streamlines are instantaneous Eulerian diagnostics, not observed particle/drifter "
        "trajectories; all fields are model-flow diagnostics"
    )
    if coastline is not None:
        method_note += "\nOpenStreetMap (ODbL) coastline overlay is visual-only; no ocean mask, flow domain, or TRAP values changed"
    figure.text(
        0.5,
        0.01,
        method_note,
        ha="center",
        fontsize=8,
        color="#7f0000",
    )
    # aspect='equal' 完成後主圖的實際 axes 高度可能小於 subplot 欄位高度；依
    # 相鄰 OCM-SVD-Analysis 的繪圖法，以已完成 layout 的主圖框對齊兩條色條。
    # 這只移動色條 axes 的上下位置與高度，不改變色階設定、流場或 TRAP
    # 資料；自動刻度可依縮短後的色條重新選取，避免標籤重疊。
    align_colorbars_to_axes(
        figure,
        (
            (axes[0], colorbar.ax),
            (axes[2], divergence_colorbar.ax),
        ),
    )
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


def _load_catalog(catalog: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    """讀取已載入或 JSON 路徑形式的 catalog，保留原始欄位語意。"""

    if isinstance(catalog, Mapping):
        return catalog
    path = Path(catalog)
    _require(path.is_file(), f"找不到 catalog JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"catalog JSON 無法讀取: {path}: {error}") from error
    _require(isinstance(payload, Mapping), "catalog JSON 頂層必須是物件")
    return payload


def _track_is_complete(row: Mapping[str, Any]) -> bool:
    """依既有 tracking flags 判斷 track 是否可納入完整觀測跨度統計。

    完整 track 必須沒有研究期左／右截尾，也不能帶資料缺口；函式不會因缺少旗標而
    推測完整，避免把舊版或不完整 catalog 的 duration 誤當成物理存活時間。
    """

    if bool(row.get("left_censored")) or bool(row.get("right_censored")):
        return False
    flags = {str(flag) for flag in row.get("quality_flags", ()) or ()}
    if "contains_data_gap" in flags or str(row.get("termination_reason")) in {
        "data_gap",
        "time_gap",
        "study_period_end",
    }:
        return False
    return "left_censored" in row and "right_censored" in row


def summarize_track_lifetimes(catalog: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """分開整理完整 track 的觀測跨度與截尾 observed span。

    回傳的 ``complete_duration_hours`` 是未截尾 track 的首末檢出時間差，可用於完整
    觀測跨度分布；它仍不是精確物理壽命。
    ``censored_observed_span_hours`` 是截至研究窗口或資料缺口的可見跨度，只能以
    下界語意另行呈現。若所有 track 都截尾，完整陣列保持空集合並回傳
    ``lifetime_status='no_complete_track_duration'``，不以零代替資料不足。另回傳
    catalog 能辨識的 median snapshot interval，供圖說揭露時間解析度。
    """

    payload = _load_catalog(catalog)
    raw_tracks = payload.get("tracks", ())
    _require(isinstance(raw_tracks, Sequence), "catalog.tracks 必須是清單")
    complete_records: list[dict[str, Any]] = []
    censored_records: list[dict[str, Any]] = []
    for raw in raw_tracks:
        _require(isinstance(raw, Mapping), "catalog.tracks 每筆必須是物件")
        duration_seconds = raw.get("duration_seconds")
        try:
            duration_hours = float(duration_seconds) / 3_600.0
        except (TypeError, ValueError):
            continue
        if not np.isfinite(duration_hours) or duration_hours < 0.0:
            continue
        record = {
            "track_id": int(raw.get("track_id", len(complete_records) + len(censored_records))),
            "duration_hours": duration_hours,
            "termination_reason": str(raw.get("termination_reason", "unknown")),
            "left_censored": bool(raw.get("left_censored")),
            "right_censored": bool(raw.get("right_censored")),
            "quality_flags": [str(flag) for flag in raw.get("quality_flags", ()) or ()],
        }
        if _track_is_complete(raw):
            complete_records.append(record)
        else:
            censored_records.append(record)
    raw_snapshots = payload.get("snapshots", ())
    snapshot_times: list[int] = []
    if isinstance(raw_snapshots, Sequence):
        for raw_snapshot in raw_snapshots:
            if isinstance(raw_snapshot, Mapping) and raw_snapshot.get("time_utc_ns") is not None:
                try:
                    snapshot_times.append(int(raw_snapshot["time_utc_ns"]))
                except (TypeError, ValueError):
                    continue
    interval_hours = np.diff(np.asarray(sorted(set(snapshot_times)), dtype=np.int64)) / NANOSECONDS_PER_HOUR
    interval_hours = interval_hours[np.isfinite(interval_hours) & (interval_hours > 0.0)]
    sampling_interval_hours = float(np.median(interval_hours)) if interval_hours.size else None
    return {
        "state_label": payload.get("state_label"),
        "science_status": payload.get("science_status") or payload.get("state_label"),
        "source": payload.get("source"),
        "complete_tracks": complete_records,
        "censored_tracks": censored_records,
        "complete_duration_hours": [record["duration_hours"] for record in complete_records],
        "complete_observed_span_hours": [record["duration_hours"] for record in complete_records],
        "censored_observed_span_hours": [record["duration_hours"] for record in censored_records],
        "complete_track_count": len(complete_records),
        "censored_track_count": len(censored_records),
        "sampling_interval_hours": sampling_interval_hours,
        "duration_semantics": "first_to_last_detection_observed_span; physical_lifetime_unresolved",
        "lifetime_status": "complete_tracks_available" if complete_records else "no_complete_track_duration",
    }


def render_lifetime_distribution_figure(
    catalog: Mapping[str, Any] | str | Path,
    output_path: str | Path,
    *,
    state_label: str | None = None,
    source_label: str | None = None,
    dpi: int = 150,
) -> Path:
    """輸出完整／截尾 track 的首末檢出觀測跨度分布比較圖。

    完整樣本以實線填色 histogram 顯示；截尾樣本以斜線空心 histogram 顯示，並在
    圖例寫明它們是 observed span lower bounds。完整 histogram 的統計樣本永遠不含
    截尾 track；若沒有完整 track，圖中只顯示資料不足訊息與截尾樣本，並保留
    ``PILOT_ONLY`` 或其他科學狀態。所有 duration 都是首末檢出時間差，不能當作
    精確物理存活時間。
    """

    _, plt = _load_matplotlib()
    summary = summarize_track_lifetimes(catalog)
    output = _output_path(output_path)
    complete = np.asarray(summary["complete_duration_hours"], dtype=np.float64)
    censored = np.asarray(summary["censored_observed_span_hours"], dtype=np.float64)
    values = np.concatenate((complete, censored)) if complete.size or censored.size else np.asarray([0.0])
    if values.size == 1 or np.allclose(values, values[0]):
        center = float(values[0])
        bins = np.asarray([max(0.0, center - 0.5), center + 0.5], dtype=np.float64)
    else:
        bins = np.histogram_bin_edges(values, bins="auto")
        if bins.size < 2:
            bins = np.asarray([0.0, float(np.max(values)) + 1.0], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(10.0, 6.0), constrained_layout=False)
    figure.subplots_adjust(left=0.10, right=0.98, top=0.76, bottom=0.20)
    if complete.size:
        axis.hist(
            complete,
            bins=bins,
            color="#377eb8",
            alpha=0.70,
            label=f"complete observed span (n={complete.size})",
        )
    if censored.size:
        axis.hist(
            censored,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color="#e66101",
            hatch="//",
            label=f"censored observed span lower bound (n={censored.size})",
        )
    if not complete.size:
        axis.text(
            0.5,
            0.55,
            "No complete tracks: observed-span distribution unavailable",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#8c2d04",
        )
    if not complete.size and not censored.size:
        axis.text(0.5, 0.44, "No track records", transform=axis.transAxes, ha="center", va="center")
    axis.set_xlabel("Eulerian TRAP observed span (h; first-to-last detection)")
    axis.set_ylabel("Track count")
    sampling_label = (
        f"{summary['sampling_interval_hours']:.3g}"
        if summary["sampling_interval_hours"] is not None
        else "unknown"
    )
    # 長 domain ID 與 UTC 來源另放於置中副標，避免軸標題超出圖邊界而裁掉樣本數。
    axis.set_title(
        f"TRAP structure observed-span distribution | {state_label or summary['science_status']}\n"
        f"complete={summary['complete_track_count']} | censored={summary['censored_track_count']} | sampling Δt={sampling_label} h",
        fontsize=11,
    )
    figure.text(0.5, 0.88, source_label or "single catalog", ha="center", fontsize=8, fontproperties=_site_font_properties())
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    handles, _labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best", fontsize=8)
    figure.text(
        0.5,
        0.04,
        "Complete sample excludes censored tracks; first-to-last detection span is resolution limited.",
        ha="center",
        fontsize=8,
        color="#7f0000",
    )
    figure.text(0.5, 0.02, "Observed TRAP span is not particle residence time or exact physical lifetime", ha="center", fontsize=8, color="#7f0000")
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


def _lifecycle_records(
    catalog: Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any], dict[int, Mapping[str, Any]]]:
    """由 catalog 建立 track 與逐 detection 的生命週期索引。"""

    payload = _load_catalog(catalog)
    raw_tracks = payload.get("tracks", ())
    raw_detections = payload.get("detections", ())
    _require(
        isinstance(raw_tracks, Sequence) and isinstance(raw_detections, Sequence),
        "catalog tracks/detections 必須是清單",
    )
    tracks: dict[int, Mapping[str, Any]] = {}
    for raw in raw_tracks:
        if isinstance(raw, Mapping) and raw.get("track_id") is not None:
            tracks[int(raw["track_id"])] = raw
    return payload, tracks


def render_lifecycle_intensity_figure(
    catalog: Mapping[str, Any] | str | Path,
    output_path: str | Path,
    *,
    state_label: str | None = None,
    source_label: str | None = None,
    dpi: int = 150,
) -> Path:
    """輸出逐 track 的 TRAP 沿生命週期平均 ``|s1|`` 圖。

    每個點來自 catalog detection 的 UTC 時間與 ``curve_mean_abs_s1_sinv``；x 軸是
    該 Eulerian track 相對第一個觀測的年齡（小時），y 軸單位為 ``s^-1``。完整與
    截尾 track 使用不同線型，且截尾 track 只表示可見期間。這張圖不把任一 detection
    當成粒子位置，也不以曲線生命週期推論海廢停留時間。
    """

    _, plt = _load_matplotlib()
    payload, tracks = _lifecycle_records(catalog)
    lifetime_summary = summarize_track_lifetimes(payload)
    output = _output_path(output_path)
    rows_by_track: dict[int, list[tuple[int, float]]] = {}
    raw_detections = payload.get("detections", ())
    for raw in raw_detections:
        if not isinstance(raw, Mapping) or raw.get("track_id") is None:
            continue
        try:
            track_id = int(raw["track_id"])
            time_utc_ns = int(raw["time_utc_ns"])
            intensity = float(raw["curve_mean_abs_s1_sinv"])
        except (TypeError, ValueError, KeyError):
            continue
        if not np.isfinite(intensity):
            continue
        rows_by_track.setdefault(track_id, []).append((time_utc_ns, intensity))

    figure, axis = plt.subplots(figsize=(10.0, 6.0), constrained_layout=False)
    figure.subplots_adjust(left=0.10, right=0.98, top=0.76, bottom=0.20)
    complete_count = 0
    censored_count = 0
    for track_id in sorted(rows_by_track):
        track = tracks.get(track_id)
        if track is None:
            continue
        rows = sorted(rows_by_track[track_id], key=lambda item: item[0])
        if not rows:
            continue
        start_ns = int(track.get("start_time_utc_ns", rows[0][0]))
        age_hours = np.asarray([(time_ns - start_ns) / NANOSECONDS_PER_HOUR for time_ns, _ in rows], dtype=np.float64)
        intensity = np.asarray([value for _, value in rows], dtype=np.float64)
        complete = _track_is_complete(track)
        if complete:
            complete_count += 1
            style = {
                "color": "#377eb8",
                "linestyle": "-",
                "alpha": 0.82,
                "label": "complete track" if complete_count == 1 else None,
            }
        else:
            censored_count += 1
            style = {
                "color": "#e66101",
                "linestyle": "--",
                "alpha": 0.82,
                "label": "censored track" if censored_count == 1 else None,
            }
        axis.plot(age_hours, intensity, marker="o", markersize=3.4, linewidth=1.2, **style)

    if complete_count == 0 and censored_count == 0:
        axis.text(0.5, 0.5, "No track-linked intensity records", transform=axis.transAxes, ha="center", va="center")
    axis.set_xlabel("Age since first observed detection (h; UTC track sequence)")
    axis.set_ylabel(r"Mean curve $|s_1|$ ($s^{-1}$)")
    sampling_label = (
        f"{lifetime_summary['sampling_interval_hours']:.3g}"
        if lifetime_summary["sampling_interval_hours"] is not None
        else "unknown"
    )
    axis.set_title(
        "TRAP intensity through observed life cycle | "
        f"{state_label or payload.get('science_status') or payload.get('state_label')}\n"
        f"complete tracks={complete_count} | censored tracks={censored_count} | sampling Δt={sampling_label} h",
        fontsize=11,
    )
    figure.text(0.5, 0.88, source_label or "single catalog", ha="center", fontsize=8, fontproperties=_site_font_properties())
    axis.grid(alpha=0.25, linewidth=0.6)
    handles, _labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best", fontsize=8)
    figure.text(
        0.5,
        0.04,
        "Each point is a model Eulerian TRAP detection at recorded UTC; age is snapshot-resolution limited.",
        ha="center",
        fontsize=8,
        color="#7f0000",
    )
    figure.text(0.5, 0.02, "Censored curves show observed span only; no particle residence time is inferred", ha="center", fontsize=8, color="#7f0000")
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


# 以短名稱提供給後續 CLI 整合，實際契約仍由上方具名函式說明。
render_lifetime_figure = render_lifetime_distribution_figure
render_lifecycle_figure = render_lifecycle_intensity_figure


__all__ = [
    "render_lifecycle_figure",
    "render_lifecycle_intensity_figure",
    "render_lifetime_distribution_figure",
    "render_lifetime_figure",
    "render_mechanism_figure",
    "render_method_comparison_figure",
    "summarize_track_lifetimes",
]
