"""``aggregate`` 的逐站「代表 TRAP」與 track 診斷圖（WP-H）。

每個研究站各自獨立（不與其他站比較、色階只取自該站資料），輸出到 ``figures/<site_id>/``：

- ``site_<id>_example.png``（Kunz et al. 2024 Fig. 2 風格）：該站代表 track 在起始時刻的 s1、
  表層流速箭頭、全部合格 TRAP 曲線（淡色），代表 track 的起始曲線與核心（紅色粗線），以及該 track
  後續核心路徑與約 1/3、2/3 壽命與最後觀測的曲線（較淺色，附相對起始的時間標籤）；
- ``site_<id>_method_comparison.png``（Serra et al. 2020 Fig. 4d 風格）：同一 snapshot 的
  (a) s1 與 TRAP、(b) 瞬時流線與 TRAP、(c) 水平散度與 TRAP；
- ``site_<id>_long_tracks.png``（Kunz Fig. 4b 風格）：長壽命 track 的核心路徑，以壽命著色；
- ``site_<id>_track_density.png``（Kunz Fig. 4c 風格）：通過各格的不同 track 數；
- ``site_<id>_propagation.png``（Kunz Fig. 5c 類比）：逐 track 線性擬合傳播速度（位移小於 1 格者為靜止，
  不入直方圖與方向玫瑰）；標示移動／靜止條數、靜止比例與殘差 RMS；
- ``site_<id>_example_coastal.png``（WP-I）：代表 track 退回近岸型時另列的近岸型代表；
  ``site_<id>_long_tracks.png`` 以虛線畫近岸型、實線畫近岸外，所有地圖疊加近岸緩衝帶輪廓。

前兩張需重開 OCM 月份並以該 shard 的偵測設定重算單一 snapshot（:func:`prepare_example_snapshot`）：來源路徑
與海岸遮罩 sidecar 取自 shard manifest 的來源 provenance，並核對 SHA-256；重算的代表核心必須與 shard 已存核心
相差不超過 1 格，否則明確失敗。圖面只描述模型內部「表層短期吸引潛勢」，流線不是粒子軌跡，散度也不等於 TRAP 吸引量。

所有固定文字來自 :data:`traps.report_font.FIGURE_TEXT`，並在 :func:`traps.report_font.cjk_font_context` 內繪製。
"""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .aggregate_figures import (
    _BOTTOM_LABEL_INCH,
    _CBAR_PAD_INCH,
    _CBAR_RIGHT_INCH,
    _CBAR_WIDTH_INCH,
    _LEFT_INCH,
    _MIN_FIGURE_WIDTH,
    S1_DAY_LABEL,
    SiteFigureData,
    _add_inch_axes,
    _draw_coast_buffer,
    _fit_panel,
    _footer,
    _footer_height_inch,
    _load_matplotlib,
    _place_footer,
    _prepare_map_axis,
    _save,
    _view_extent,
    coast_buffer_label,
    map_data_aspect,
)
from .colorbar_layout import align_colorbars_to_axes
from .coastline import CoastlineGeometry
from .report_font import FIGURE_TEXT as T
from .report_font import CJKFontSelection, cjk_font_context

SECONDS_PER_DAY = 86_400.0
CORE_MATCH_TOLERANCE_CELLS = 1
"""重算核心與 shard 已存核心允許的最大格距（Chebyshev）。"""
LATER_CURVE_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0, 1.0)
"""後續曲線取樣位置：約 1/3、2/3 壽命與最後觀測。"""
LATER_CURVE_COLOURS = ("#d84a4a", "#ee8a75", "#f7bfae")
"""後續曲線的較淺紅色階（由深至淺）。"""
ROSE_SECTORS = 16
MAX_QUIVER_ARROWS = 350

EXAMPLE_FIGURE_KINDS = ("example", "method_comparison")
"""需要重開 OCM snapshot 的圖（``--no-example-figures`` 略過）。"""
TRACK_FIGURE_KINDS = ("long_tracks", "track_density", "propagation")
"""只用 track／spatial 產物的圖。"""


# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------


@dataclass
class LaterObservation:
    """代表 track 的一個後續觀測：相對起始的時間、已存曲線與核心（公尺）。"""

    dt_seconds: float
    curve_m: np.ndarray
    core_m: tuple[float, float]


@dataclass
class ExampleSnapshot:
    """代表 track 起始時刻的重算 snapshot 與 track 資訊（全部欄位為公尺制格網上的完整場）。"""

    track_id: int
    time_utc_ns: int
    month: str
    snapshot_index: int
    detection_index: int
    lifetime_hours: float
    n_observations: int
    criterion: str
    core_offset_cells: int
    x_m_axis: np.ndarray
    y_m_axis: np.ndarray
    u: np.ndarray
    v: np.ndarray
    valid: np.ndarray
    s1_sinv: np.ndarray
    derivative_valid: np.ndarray
    curves_m: list[np.ndarray]
    cores_m: list[tuple[float, float]]
    chosen_curve_m: np.ndarray
    chosen_core_m: tuple[float, float]
    path_m: np.ndarray
    later: list[LaterObservation] = field(default_factory=list)
    source_month_dir: str = ""
    """實際重開的 OCM 月份目錄（寫入 manifest）。"""
    is_coastal: bool = False
    """代表 track 是否為近岸型（平均核心距岸小於緩衝帶）。"""
    mean_coast_m: float | None = None
    """代表 track 全部觀測的平均核心距岸距離（m）；沒有近岸資訊時為 ``None``。"""
    variant: str = "main"
    """``main``（一般代表圖）或 ``coastal``（近岸型代表圖，標題不同）。"""


@dataclass
class TrackFigureInputs:
    """track 診斷圖的輸入（已限制在該站；長壽命 track 已依壽命由長到短排序並截斷）。"""

    x_m_axis: np.ndarray
    y_m_axis: np.ndarray
    long_track_hours: float
    long_tracks: Sequence[tuple[float, np.ndarray, np.ndarray]]
    """``(壽命小時, core_x_m, core_y_m)``，最多 ``LONG_TRACK_DRAW_LIMIT`` 條、由最長開始。"""
    n_long_total: int
    prop_cx_kmday: np.ndarray
    prop_cy_kmday: np.ndarray
    prop_track_n: int
    nominal_step_hours: float
    null_long_mean: float | None = None
    null_n_seeds: int = 0
    long_coastal: Sequence[bool] = ()
    """與 ``long_tracks`` 平行：該 track 是否近岸型（平均核心距岸小於緩衝帶）；空表示沒有近岸資訊。"""
    n_long_coastal_total: int | None = None
    n_long_offshore_total: int | None = None
    null_long_coastal_mean: float | None = None
    null_long_offshore_mean: float | None = None
    prop_stationary: np.ndarray | None = None
    """與 ``prop_cx_kmday`` 平行；True 為靜止 track（擬合位移小於 1 格）。``None`` 表示全部視為移動。"""
    prop_resid_rms_m: np.ndarray | None = None
    """每條已擬合 track 的位置殘差 RMS（m）。"""
    prop_min_obs: int | None = None
    """實際使用的最少擬合觀測數（顯示於副標題）。"""


# ---------------------------------------------------------------------------
# 幾何與格式小工具
# ---------------------------------------------------------------------------


class MetricToLonLat:
    """公尺軸 → 經緯度軸的線性內插（與 :func:`traps.coastline.draw_metric_coastline` 同一映射）。"""

    def __init__(self, x_m_axis: np.ndarray, y_m_axis: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> None:
        x_axis = np.asarray(x_m_axis, dtype=np.float64)
        y_axis = np.asarray(y_m_axis, dtype=np.float64)
        lon_axis = np.asarray(lon, dtype=np.float64)
        lat_axis = np.asarray(lat, dtype=np.float64)
        if x_axis.size != lon_axis.size or y_axis.size != lat_axis.size:
            raise ValueError("公尺軸與經緯度軸長度必須相同")
        if x_axis[-1] < x_axis[0]:
            x_axis, lon_axis = x_axis[::-1], lon_axis[::-1]
        if y_axis[-1] < y_axis[0]:
            y_axis, lat_axis = y_axis[::-1], lat_axis[::-1]
        self._x, self._lon, self._y, self._lat = x_axis, lon_axis, y_axis, lat_axis

    def __call__(self, x_m: Any, y_m: Any) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.interp(np.asarray(x_m, dtype=np.float64), self._x, self._lon),
            np.interp(np.asarray(y_m, dtype=np.float64), self._y, self._lat),
        )


def time_label_utc(time_utc_ns: int) -> str:
    """UTC 時間字串（到分，``YYYY-MM-DD HH:MM``；圖面另標 UTC）。"""

    timestamp = _datetime.datetime.fromtimestamp(int(time_utc_ns) / 1.0e9, tz=_datetime.UTC)
    return timestamp.strftime("%Y-%m-%d %H:%M")


def uses_day_unit(nominal_step_hours: float) -> bool:
    """名目步長為 24 h 的整數倍時，以「天」顯示時間與壽命。"""

    step = float(nominal_step_hours)
    return step >= 24.0 and abs(step / 24.0 - round(step / 24.0)) < 1.0e-9


def relative_time_label(dt_seconds: float, *, day_unit: bool) -> str:
    """相對起始的時間標籤：``+6 h`` 或 ``+3 天``（一位小數，去除多餘零）。"""

    if day_unit:
        return f"+{round(dt_seconds / SECONDS_PER_DAY, 1):g} {T['day_unit']}"
    return f"+{round(dt_seconds / 3_600.0, 1):g} h"


def duration_text(hours: float, *, day_unit: bool) -> str:
    """壽命文字：``24 h`` 或 ``3 天``。"""

    if day_unit:
        return f"{round(float(hours) / 24.0, 1):g} {T['day_unit']}"
    return f"{round(float(hours), 1):g} h"


def padded_view(data: SiteFigureData, pad_fraction: float = 0.10) -> tuple[slice, slice]:
    """站點分析區外接框外擴 ``pad_fraction``（至少 2 格）的 (row, col) 切片。"""

    rows = np.nonzero(np.any(data.site_mask, axis=1))[0]
    cols = np.nonzero(np.any(data.site_mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return _view_extent(data)
    span = max(int(rows[-1] - rows[0]) + 1, int(cols[-1] - cols[0]) + 1)
    return _view_extent(data, margin_cells=max(2, int(round(pad_fraction * span))))


def pick_later_observations(times_utc_ns: np.ndarray, fractions: Sequence[float] = LATER_CURVE_FRACTIONS) -> list[int]:
    """在起始觀測（索引 0）之後，挑出最接近各壽命比例時刻的觀測索引（遞增、不重複；最後一個為末觀測）。"""

    times = np.asarray(times_utc_ns, dtype=np.int64)
    if times.size < 2:
        return []
    start, end = float(times[0]), float(times[-1])
    chosen: list[int] = []
    for fraction in fractions:
        target = start + float(fraction) * (end - start)
        index = int(np.argmin(np.abs(times[1:].astype(np.float64) - target))) + 1
        if index not in chosen:
            chosen.append(index)
    return sorted(chosen)


def verify_recomputed_core(
    detections: Sequence[Any],
    detection_index: int,
    stored_row: int,
    stored_col: int,
    *,
    tolerance_cells: int = CORE_MATCH_TOLERANCE_CELLS,
) -> int:
    """核對重算的第 ``detection_index`` 個 detection 核心與 shard 已存核心格點在容許範圍內。

    回傳兩者的 Chebyshev 格距；找不到該 detection、未合格或格距大於 ``tolerance_cells`` 時拋出
    ``ValueError``（表示來源資料／偵測設定與 shard 不一致，不可用來畫代表圖）。
    """

    if not 0 <= int(detection_index) < len(detections):
        raise ValueError(f"重算 snapshot 只有 {len(detections)} 個 detection，找不到 detection_index={detection_index}")
    detection = detections[int(detection_index)]
    if not bool(detection.is_qualified):
        raise ValueError(f"重算的 detection_index={detection_index} 不是合格 TRAP，與 shard 記錄不一致")
    row, col = int(detection.core_index_ij[0]), int(detection.core_index_ij[1])
    offset = max(abs(row - int(stored_row)), abs(col - int(stored_col)))
    if offset > int(tolerance_cells):
        raise ValueError(
            f"重算核心格點 ({row}, {col}) 與 shard 已存核心 ({int(stored_row)}, {int(stored_col)}) 相差 {offset} 格，"
            f"超過容許的 {int(tolerance_cells)} 格：來源 OCM 資料或偵測設定與 shard 不一致"
        )
    return offset


# ---------------------------------------------------------------------------
# 重開 OCM 月份並重算代表 snapshot
# ---------------------------------------------------------------------------


def resolve_source_paths(
    shard_manifest: Mapping[str, Any], *, domain_id: str, month: str, surface_root: Path | None
) -> tuple[Path, Path | None]:
    """由 shard manifest 的來源 provenance 解析 OCM 月份目錄與海岸遮罩 sidecar 路徑。

    未指定 ``surface_root`` 時直接使用 manifest 記錄的路徑；指定時月份目錄為
    ``surface_root/<domain_id>/months/<month>``，遮罩優先取 ``surface_root/<domain_id>/grid/<原檔名>``，找不到才用
    manifest 路徑。任何解析結果都必須存在，否則拋出含候選路徑的 ``ValueError``。
    """

    source = shard_manifest.get("source") or {}
    mask_record = source.get("coastline_mask") or {}
    recorded_month_dir = source.get("month_dir")
    recorded_mask = mask_record.get("path")
    if surface_root is not None:
        month_dir = Path(surface_root) / domain_id / "months" / month
        candidates = []
        if recorded_mask:
            candidates.append(Path(surface_root) / domain_id / "grid" / Path(str(recorded_mask)).name)
            candidates.append(Path(str(recorded_mask)))
    else:
        if not recorded_month_dir:
            raise ValueError(f"shard manifest（{domain_id}/{month}）沒有記錄 source.month_dir；請以 --surface-root 指定 OCM 路徑")
        month_dir = Path(str(recorded_month_dir))
        candidates = [Path(str(recorded_mask))] if recorded_mask else []
    if not month_dir.is_dir():
        raise ValueError(f"重開 OCM 月份失敗：目錄不存在 {month_dir}（可用 --surface-root 指定新的 OCM 路徑）")
    mask_path: Path | None = None
    if recorded_mask:
        mask_path = next((path for path in candidates if path.is_file()), None)
        if mask_path is None:
            raise ValueError(f"重開 OCM 月份失敗：找不到海岸遮罩 sidecar，候選 {[str(path) for path in candidates]}")
    return month_dir, mask_path


def _stored_detection(
    detect_root: Path, domain_id: str, month: str, snapshot_index: int, detection_index: int, cache: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    """讀 shard ``detections.npz`` 中指定 (snapshot, detection) 的核心格點與 ragged 曲線（公尺）。"""

    if month not in cache:
        with np.load(detect_root / domain_id / month / "detections.npz") as archive:
            cache[month] = {
                name: np.array(archive[name])
                for name in ("snapshot_index", "detection_index", "core_row", "core_col", "curve_offsets", "curve_points_m", "is_qualified")
            }
    arrays = cache[month]
    rows = np.nonzero((arrays["snapshot_index"] == int(snapshot_index)) & (arrays["detection_index"] == int(detection_index)))[0]
    if rows.size != 1:
        raise ValueError(f"shard {domain_id}/{month} 找不到唯一的 detection（snapshot={snapshot_index}, detection={detection_index}）")
    row = int(rows[0])
    lo, hi = int(arrays["curve_offsets"][row]), int(arrays["curve_offsets"][row + 1])
    return {
        "core_row": int(arrays["core_row"][row]),
        "core_col": int(arrays["core_col"][row]),
        "curve_m": np.asarray(arrays["curve_points_m"][lo:hi], dtype=np.float64),
        "is_qualified": bool(arrays["is_qualified"][row]),
    }


def prepare_example_snapshot(
    *,
    detect_root: Path,
    domain_id: str,
    shard_manifests: Mapping[str, Mapping[str, Any]],
    state_label: str,
    surface_root: Path | None,
    observations: Mapping[str, np.ndarray],
    start: int,
    stop: int,
    track_id: int,
    lifetime_hours: float,
    criterion: str,
    coast_buffer_m: float | None = None,
    variant: str = "main",
) -> ExampleSnapshot:
    """重開代表 track 起始月份、重算起始 snapshot 並核對核心，組成 :class:`ExampleSnapshot`。

    起始觀測是 ``observations`` 陣列 ``[start:stop)`` 的第一列（track 的第一個觀測，且在站內）。流程：
    (1) 依該觀測所屬月份的 shard manifest 解析 OCM 月份目錄與海岸遮罩（核對 metadata／遮罩 SHA-256 與 UTC 時刻）；
    (2) 用該 shard 的 ``detection_config`` 對這一個 snapshot 重跑 :func:`traps.detection.detect_snapshot`；
    (3) 重算核心必須與 shard 已存核心相差不超過 1 格；(4) 後續觀測的曲線取自各自月份 shard 的已存曲線。
    """

    from .detection import detect_snapshot
    from .models import DetectionConfig
    from .ocm import _sha256_file, open_month

    month_values = np.asarray(observations["month"][start:stop], dtype=np.int64)
    times = np.asarray(observations["time_utc_ns"][start:stop], dtype=np.int64)
    snapshots = np.asarray(observations["snapshot_index"][start:stop], dtype=np.int64)
    detections = np.asarray(observations["detection_index"][start:stop], dtype=np.int64)
    core_x = np.asarray(observations["core_x_m"][start:stop], dtype=np.float64)
    core_y = np.asarray(observations["core_y_m"][start:stop], dtype=np.float64)

    birth_month = f"{int(month_values[0]):06d}"
    manifest = shard_manifests.get(birth_month)
    if manifest is None:
        raise ValueError(f"代表 track 起始月份 {domain_id}/{birth_month} 沒有已驗證的 shard manifest")
    source = manifest.get("source") or {}
    month_dir, mask_path = resolve_source_paths(manifest, domain_id=domain_id, month=birth_month, surface_root=surface_root)

    recorded_meta_sha = source.get("month_metadata_sha256")
    if recorded_meta_sha and _sha256_file(month_dir / "metadata.json") != recorded_meta_sha:
        raise ValueError(f"OCM 月份 metadata 的 SHA-256 與 shard 記錄不符（來源已變更）: {month_dir / 'metadata.json'}")
    recorded_mask_sha = (source.get("coastline_mask") or {}).get("sha256")
    if mask_path is not None and recorded_mask_sha and _sha256_file(mask_path) != recorded_mask_sha:
        raise ValueError(f"海岸遮罩 sidecar 的 SHA-256 與 shard 記錄不符（來源已變更）: {mask_path}")

    month = open_month(
        month_dir,
        allow_trial_pilot=(state_label == "PILOT_ONLY"),
        expected_domain_id=domain_id,
        analysis_ocean_mask_path=mask_path,
    )
    snapshot = month.snapshot(int(snapshots[0]))
    if int(snapshot.time_utc_ns) != int(times[0]):
        raise ValueError(
            f"重開的 OCM snapshot 時刻 {time_label_utc(snapshot.time_utc_ns)} 與代表 track 起始觀測 {time_label_utc(int(times[0]))} 不符"
        )
    config = DetectionConfig(**dict(manifest["detection_config"]))
    result = detect_snapshot(snapshot.u, snapshot.v, snapshot.valid, snapshot.x_m, snapshot.y_m, config)

    cache: dict[str, dict[str, np.ndarray]] = {}
    stored = _stored_detection(detect_root, domain_id, birth_month, int(snapshots[0]), int(detections[0]), cache)
    offset = verify_recomputed_core(result.detections, int(detections[0]), stored["core_row"], stored["core_col"])

    later: list[LaterObservation] = []
    for index in pick_later_observations(times):
        obs_month = f"{int(month_values[index]):06d}"
        later_stored = _stored_detection(detect_root, domain_id, obs_month, int(snapshots[index]), int(detections[index]), cache)
        later.append(
            LaterObservation(
                dt_seconds=float(times[index] - times[0]) / 1.0e9,
                curve_m=later_stored["curve_m"],
                core_m=(float(core_x[index]), float(core_y[index])),
            )
        )

    mean_coast_m: float | None = None
    if "core_coast_distance_m" in observations:
        coast_values = np.asarray(observations["core_coast_distance_m"][start:stop], dtype=np.float64)
        if np.any(np.isfinite(coast_values)):
            mean_coast_m = float(np.nanmean(coast_values))
    is_coastal = bool(coast_buffer_m is not None and mean_coast_m is not None and mean_coast_m < float(coast_buffer_m))

    chosen = result.detections[int(detections[0])]
    curves = [np.asarray(item.curve_points_m, dtype=np.float64) for item in result.detections if item.is_qualified]
    return ExampleSnapshot(
        track_id=int(track_id),
        time_utc_ns=int(times[0]),
        month=birth_month,
        snapshot_index=int(snapshots[0]),
        detection_index=int(detections[0]),
        lifetime_hours=float(lifetime_hours),
        n_observations=int(times.size),
        criterion=criterion,
        core_offset_cells=int(offset),
        x_m_axis=np.asarray(snapshot.x_m, dtype=np.float64)[0, :].copy(),
        y_m_axis=np.asarray(snapshot.y_m, dtype=np.float64)[:, 0].copy(),
        u=np.asarray(snapshot.u, dtype=np.float64),
        v=np.asarray(snapshot.v, dtype=np.float64),
        valid=np.asarray(snapshot.valid, dtype=bool),
        s1_sinv=np.asarray(result.s1_sinv, dtype=np.float64),
        derivative_valid=np.asarray(result.derivative_valid, dtype=bool),
        curves_m=curves,
        cores_m=[(float(item.core_xy_m[0]), float(item.core_xy_m[1])) for item in result.detections if item.is_qualified],
        chosen_curve_m=np.asarray(chosen.curve_points_m, dtype=np.float64),
        chosen_core_m=(float(chosen.core_xy_m[0]), float(chosen.core_xy_m[1])),
        path_m=np.column_stack([core_x, core_y]),
        later=later,
        source_month_dir=str(month_dir),
        is_coastal=is_coastal,
        mean_coast_m=mean_coast_m,
        variant=variant,
    )


# ---------------------------------------------------------------------------
# 共用繪圖小件
# ---------------------------------------------------------------------------


def _symmetric_limit(values: np.ndarray) -> float:
    """對稱色階上限：有限值絕對值的 98 百分位（至少極小正數）。"""

    finite = np.asarray(values, dtype=np.float64)
    finite = np.abs(finite[np.isfinite(finite)])
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, 98.0))
    return limit if limit > 0.0 else max(float(finite.max()), 1.0e-12)


def _empty_state(axis: Any, message: str) -> None:
    """資料不足時在地圖／座標軸中央顯示說明（不畫假資料）。"""

    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", fontsize=10, color="#7f0000", wrap=True)


def _sequence_title(figure: Any, fig_w: float, fig_h: float, title: str, subtitle: str | None) -> None:
    figure.text(0.5, 1.0 - 0.10 / fig_h, title, ha="center", va="top", fontsize=13)
    if subtitle:
        figure.text(0.5, 1.0 - 0.42 / fig_h, subtitle, ha="center", va="top", fontsize=9)


def _draw_quiver(axis: Any, lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray, usable: np.ndarray, dlon_dx: np.ndarray, dlat_dy: np.ndarray) -> None:
    """在視窗內畫抽稀後的速度箭頭（方向以經緯度圖面等比例換算，長度自動縮放）。"""

    indices = np.argwhere(usable)
    if indices.size == 0:
        return
    stride = max(1, int(np.ceil(np.sqrt(indices.shape[0] / float(MAX_QUIVER_ARROWS)))))
    selected = indices[::stride]
    rows, cols = selected[:, 0], selected[:, 1]
    axis.quiver(
        lon[cols],
        lat[rows],
        u[rows, cols] * dlon_dx[cols],
        v[rows, cols] * dlat_dy[rows],
        color="0.15",
        alpha=0.7,
        width=0.0022,
        angles="xy",
        scale_units="xy",
        scale=None,
        zorder=3,
    )


def _local_scale(lon_axis: np.ndarray, lat_axis: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每欄 ``dlon/dx``、每列 ``dlat/dy``：把公尺速度換成經緯度圖面上的位移速度。"""

    return np.gradient(lon_axis, x_axis), np.gradient(lat_axis, y_axis)


def _plot_curves_lonlat(axis: Any, curves_m: Sequence[np.ndarray], mapper: MetricToLonLat, **style: Any) -> None:
    for curve in curves_m:
        if curve.shape[0] < 2:
            continue
        lon_c, lat_c = mapper(curve[:, 0], curve[:, 1])
        axis.plot(lon_c, lat_c, **style)


def _single_map_layout(
    data: SiteFigureData, view: tuple[slice, slice], footer: str, *, legend_height: float, max_width: float = 7.4, max_height: float = 6.0
) -> dict[str, float]:
    panel_w, panel_h = _fit_panel(map_data_aspect(data, view), max_width, max_height)
    content_w = _LEFT_INCH + panel_w + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH
    footer_h = _footer_height_inch(footer)
    fig_w = max(content_w, _MIN_FIGURE_WIDTH)
    title_h = 0.78
    fig_h = 0.1 + title_h + panel_h + _BOTTOM_LABEL_INCH + legend_height + footer_h
    return {
        "fig_w": fig_w,
        "fig_h": fig_h,
        "x0": (fig_w - content_w) / 2.0 + _LEFT_INCH,
        "y0": footer_h + legend_height + _BOTTOM_LABEL_INCH,
        "panel_w": panel_w,
        "panel_h": panel_h,
        "footer_h": footer_h,
        "legend_height": legend_height,
    }


# ---------------------------------------------------------------------------
# 1. 代表 TRAP（Kunz 2024 Fig. 2 風格）
# ---------------------------------------------------------------------------


_CRITERION_RULE_KEYS = {
    "offshore_exceeds_null": "example_rule_offshore",
    "offshore_longest": "example_rule_offshore_nonull",
    "fallback_coastal": "example_rule_fallback_coast",
    "fallback_offshore_below_null": "example_rule_fallback_coast",
    "coastal_longest": "example_rule_coastal",
}
"""WP-I 代表 track 選取準則對應的圖面規則文字 key。"""


def _example_tag(snap: ExampleSnapshot) -> str:
    """代表圖副標題的明確註記：近岸型，或壽命未超過隨機對照 p99 的整體最長者。"""

    if snap.is_coastal:
        return T["coastal_tag"]
    if snap.criterion == "fallback_offshore_below_null":
        return T["below_null_tag"]
    return ""


def _example_footer(base_footer: str, snap: ExampleSnapshot | None, *, day_unit: bool) -> str:
    if snap is None:
        return base_footer
    rule = T["example_rule"]
    if snap.criterion == "ge_threshold":
        rule += f"（{T['example_rule_threshold']}）"
    elif snap.criterion == "longest_available":
        rule += f"（{T['example_rule_fallback']}）"
    elif snap.criterion in _CRITERION_RULE_KEYS:
        rule = T[_CRITERION_RULE_KEYS[snap.criterion]]
    coast_text = ""
    if snap.mean_coast_m is not None:
        coast_text = f"  {T['coast_dist_xlabel'].split(' (')[0]} {snap.mean_coast_m / 1000.0:.1f} km"
    return (
        f"{base_footer}\ntrack {snap.track_id}  {T['example_birth']} {time_label_utc(snap.time_utc_ns)} UTC  "
        f"{T['example_lifetime']} {duration_text(snap.lifetime_hours, day_unit=day_unit)}  {snap.n_observations} {T['example_observations']}"
        f"{coast_text}\n{rule}\n{T['refined_core_note']}"
    )


LABEL_MIN_SEPARATION_INCH = 0.55
"""後續核心的圖面距離小於此值（英吋）時，相對時間標籤會彼此重疊，改以圖例清單列出。"""


def _later_labels_overlap(snap: ExampleSnapshot, data: SiteFigureData, view: tuple[slice, slice], panel_w: float, panel_h: float) -> bool:
    """後續核心（含起始核心）在圖面上是否有任兩點距離小於 :data:`LABEL_MIN_SEPARATION_INCH`。"""

    if not snap.later:
        return False
    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    mapper = MetricToLonLat(snap.x_m_axis, snap.y_m_axis, data.lon, data.lat)
    points_m = np.asarray([snap.chosen_core_m, *[item.core_m for item in snap.later]], dtype=np.float64)
    lon_p, lat_p = mapper(points_m[:, 0], points_m[:, 1])
    lon_span = max(float(lon.max() - lon.min()), 1.0e-12)
    lat_span = max(float(lat.max() - lat.min()), 1.0e-12)
    xy = np.column_stack([(lon_p - lon.min()) / lon_span * panel_w, (lat_p - lat.min()) / lat_span * panel_h])
    for first in range(1, xy.shape[0]):
        for second in range(0, first):
            if float(np.hypot(*(xy[first] - xy[second]))) < LABEL_MIN_SEPARATION_INCH:
                return True
    return False


def render_example(
    output_path: Path,
    data: SiteFigureData,
    snap: ExampleSnapshot | None,
    *,
    coastline: CoastlineGeometry | None,
    footer: str,
    day_unit: bool = False,
    dpi: int = 150,
) -> Path:
    """代表 track 起始 snapshot：s1、速度箭頭、TRAP 曲線、代表曲線與核心、後續曲線與核心路徑。

    ``snap`` 為 ``None``（本站沒有站內出生的未截尾 track）時只畫空白地圖並標示原因，不畫假資料。
    地圖範圍是站點分析區外接框外擴；圖例與註腳各佔獨立版面帶，彼此不重疊。
    """

    matplotlib, plt = _load_matplotlib()
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as path_effects

    view = padded_view(data)
    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    full_footer = _example_footer(footer, snap, day_unit=day_unit)
    layout = _single_map_layout(data, view, full_footer, legend_height=0.62)
    # 後續核心幾乎沒有移動時，地圖上的相對時間標籤會彼此重疊：改為圖例中的位移清單。
    list_labels = snap is not None and _later_labels_overlap(snap, data, view, layout["panel_w"], layout["panel_h"])
    if list_labels:
        layout = _single_map_layout(data, view, full_footer, legend_height=0.86)
    figure = plt.figure(figsize=(layout["fig_w"], layout["fig_h"]))
    axis = _add_inch_axes(figure, layout["x0"], layout["y0"], layout["panel_w"], layout["panel_h"])
    cax = _add_inch_axes(figure, layout["x0"] + layout["panel_w"] + _CBAR_PAD_INCH, layout["y0"], _CBAR_WIDTH_INCH, layout["panel_h"])

    handles: list[Any] = []
    labels: list[str] = []
    if snap is None:
        blank = axis.pcolormesh(lon, lat, np.ma.masked_all((lat.size, lon.size)), shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
        colorbar = figure.colorbar(blank, cax=cax)
        colorbar.set_label(S1_DAY_LABEL)
        _prepare_map_axis(axis, data, view, coastline)
        _empty_state(axis, T["example_no_track"])
        subtitle = T["example_no_track_short"]
    else:
        mapper = MetricToLonLat(snap.x_m_axis, snap.y_m_axis, data.lon, data.lat)
        s1_day = np.where(snap.derivative_valid, snap.s1_sinv * SECONDS_PER_DAY, np.nan)[rows, cols]
        limit = _symmetric_limit(s1_day)
        mesh = axis.pcolormesh(lon, lat, np.ma.masked_invalid(s1_day), shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        colorbar = figure.colorbar(mesh, cax=cax)
        colorbar.set_label(S1_DAY_LABEL)
        dlon_dx, dlat_dy = _local_scale(data.lon, data.lat, snap.x_m_axis, snap.y_m_axis)
        usable = (snap.valid & np.isfinite(snap.u) & np.isfinite(snap.v))[rows, cols]
        _draw_quiver(axis, lon, lat, snap.u[rows, cols], snap.v[rows, cols], usable, dlon_dx[cols], dlat_dy[rows])
        handles.append(Line2D([], [], color="0.15", marker=r"$\rightarrow$", linestyle="none", markersize=11))
        labels.append(T["example_velocity"])

        _plot_curves_lonlat(axis, snap.curves_m, mapper, color="0.1", linewidth=0.9, alpha=0.35, zorder=4)
        handles.append(Line2D([], [], color="0.1", linewidth=1.0, alpha=0.5))
        labels.append(T["example_all_curves"])

        halo = [path_effects.withStroke(linewidth=3.6, foreground="white")]
        for rank, later_obs in enumerate(snap.later):
            colour = LATER_CURVE_COLOURS[round(rank * (len(LATER_CURVE_COLOURS) - 1) / max(len(snap.later) - 1, 1))]
            curve_lon, curve_lat = mapper(later_obs.curve_m[:, 0], later_obs.curve_m[:, 1])
            axis.plot(curve_lon, curve_lat, color=colour, linewidth=1.9, zorder=5.5, path_effects=halo)
            core_lon, core_lat = mapper([later_obs.core_m[0]], [later_obs.core_m[1]])
            axis.plot(core_lon, core_lat, marker="o", markersize=4, color=colour, markeredgecolor="white", markeredgewidth=0.5, zorder=6)
            if list_labels:
                shift_km = float(np.hypot(later_obs.core_m[0] - snap.chosen_core_m[0], later_obs.core_m[1] - snap.chosen_core_m[1])) / 1000.0
                handles.append(Line2D([], [], color=colour, linewidth=1.9, marker="o", markersize=4))
                labels.append(f"{relative_time_label(later_obs.dt_seconds, day_unit=day_unit)}（{T['displacement']} {shift_km:.2f} km）")
            else:
                axis.annotate(
                    relative_time_label(later_obs.dt_seconds, day_unit=day_unit),
                    xy=(float(core_lon[0]), float(core_lat[0])),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="#5f1010",
                    zorder=9,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
        if snap.later and not list_labels:
            handles.append(Line2D([], [], color=LATER_CURVE_COLOURS[1], linewidth=1.9))
            labels.append(T["example_later_curves"])

        if snap.path_m.shape[0] >= 2:
            path_lon, path_lat = mapper(snap.path_m[:, 0], snap.path_m[:, 1])
            axis.plot(path_lon, path_lat, color="#7f1d1d", linewidth=1.1, linestyle=":", marker=".", markersize=3, zorder=6.2)
            handles.append(Line2D([], [], color="#7f1d1d", linewidth=1.1, linestyle=":", marker=".", markersize=3))
            labels.append(T["example_core_path"])

        chosen_lon, chosen_lat = mapper(snap.chosen_curve_m[:, 0], snap.chosen_curve_m[:, 1])
        axis.plot(chosen_lon, chosen_lat, color="#c00000", linewidth=3.0, zorder=7, path_effects=halo)
        core_lon, core_lat = mapper([snap.chosen_core_m[0]], [snap.chosen_core_m[1]])
        axis.plot(core_lon, core_lat, marker="o", markersize=7.5, color="#c00000", markeredgecolor="white", markeredgewidth=1.0, linestyle="none", zorder=8)
        handles.append(Line2D([], [], color="#c00000", linewidth=3.0, marker="o", markersize=6))
        labels.append(T["example_chosen_curve"])
        buffer_handle = _prepare_map_axis(axis, data, view, coastline)
        if buffer_handle is not None:
            handles.append(buffer_handle)
            labels.append(coast_buffer_label(data))
        subtitle = (
            f"track {snap.track_id}｜{T['example_birth']} {time_label_utc(snap.time_utc_ns)} UTC｜"
            f"{T['example_lifetime']} {duration_text(snap.lifetime_hours, day_unit=day_unit)}{_example_tag(snap)}"
        )

    align_colorbars_to_axes(figure, ((axis, cax),))
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, layout["footer_h"] / layout["fig_h"]),
            ncol=3,
            fontsize=8,
            frameon=False,
            columnspacing=1.6,
            handlelength=2.2,
        )
    title_key = "coastal_example_title" if (snap is not None and snap.variant == "coastal") else "example_title"
    _sequence_title(figure, layout["fig_w"], layout["fig_h"], f"{data.site_name_zh} — {T[title_key]}", subtitle)
    _place_footer(figure, full_footer)
    _save(figure, plt, output_path, dpi)
    return output_path


# ---------------------------------------------------------------------------
# 2. 方法比較（Serra 2020 Fig. 4d 風格）
# ---------------------------------------------------------------------------


def _uniform_axis(axis: np.ndarray) -> np.ndarray | None:
    """等間距（相對誤差 1e-3 內）的一維軸回傳嚴格等距版本，否則 ``None``（streamplot 需要等距軸）。"""

    if axis.size < 3:
        return None
    steps = np.diff(axis)
    if np.all(steps > 0) and np.allclose(steps, steps.mean(), rtol=1.0e-3, atol=0.0):
        return np.linspace(float(axis[0]), float(axis[-1]), axis.size)
    return None


def render_method_comparison(
    output_path: Path,
    data: SiteFigureData,
    snap: ExampleSnapshot | None,
    *,
    coastline: CoastlineGeometry | None,
    footer: str,
    day_unit: bool = False,
    dpi: int = 150,
) -> Path:
    """同一 snapshot 的三分圖：(a) s1 與 TRAP、(b) 流線與 TRAP、(c) 水平散度與 TRAP。

    三個分圖共用地圖範圍與 TRAP 曲線；散度以公尺制中心差分計算（沿用
    :func:`traps.literature_figures._divergence_sinv`）並換算為 day⁻¹，色階對稱。流線用 matplotlib
    ``streamplot`` 畫在有效海域（遮罩處中止），速度依經緯度圖面等比例換算方向。註腳說明流線不是粒子軌跡、
    散度不同於 TRAP 吸引。``snap`` 為 ``None`` 時只標示無代表 track。
    """

    from .literature_figures import _divergence_sinv

    matplotlib, plt = _load_matplotlib()
    view = padded_view(data)
    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    notes = f"{T['method_note_streamline']}{T['method_note_divergence']}"
    full_footer = _example_footer(footer, snap, day_unit=day_unit) if snap is not None else footer
    note_h = 0.22

    panel_w, panel_h = _fit_panel(map_data_aspect(data, view), 4.3, 4.6)
    pitch_cbar = _LEFT_INCH + panel_w + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + 0.95
    pitch_plain = _LEFT_INCH + panel_w + 0.25
    content_w = 2.0 * pitch_cbar + pitch_plain + 0.2
    footer_h = _footer_height_inch(full_footer)
    fig_w = max(content_w, _MIN_FIGURE_WIDTH)
    title_h = 0.78
    fig_h = 0.1 + title_h + 0.30 + panel_h + _BOTTOM_LABEL_INCH + note_h + footer_h
    y0 = footer_h + note_h + _BOTTOM_LABEL_INCH
    figure = plt.figure(figsize=(fig_w, fig_h))
    x_start = (fig_w - content_w) / 2.0
    positions = [x_start + _LEFT_INCH, x_start + pitch_cbar + _LEFT_INCH, x_start + pitch_cbar + pitch_plain + _LEFT_INCH]
    axes = [_add_inch_axes(figure, x, y0, panel_w, panel_h) for x in positions]
    caxes = [
        _add_inch_axes(figure, positions[0] + panel_w + _CBAR_PAD_INCH, y0, _CBAR_WIDTH_INCH, panel_h),
        None,
        _add_inch_axes(figure, positions[2] + panel_w + _CBAR_PAD_INCH, y0, _CBAR_WIDTH_INCH, panel_h),
    ]
    pairs: list[tuple[Any, Any]] = []
    buffer_handles: list[Any] = []
    titles = (T["method_panel_a"], T["method_panel_b"], T["method_panel_c"])
    subtitle: str | None = None

    if snap is None:
        for axis, cax, title in zip(axes, caxes, titles, strict=True):
            _prepare_map_axis(axis, data, view, coastline)
            axis.set_title(title, fontsize=10)
            if cax is not None:
                blank = axis.pcolormesh(lon, lat, np.ma.masked_all((lat.size, lon.size)), shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
                figure.colorbar(blank, cax=cax)
                pairs.append((axis, cax))
        _empty_state(axes[1], T["example_no_track"])
        subtitle = T["example_no_track_short"]
    else:
        mapper = MetricToLonLat(snap.x_m_axis, snap.y_m_axis, data.lon, data.lat)
        subtitle = (
            f"track {snap.track_id}｜{T['example_birth']} {time_label_utc(snap.time_utc_ns)} UTC｜"
            f"{T['example_lifetime']} {duration_text(snap.lifetime_hours, day_unit=day_unit)}"
        )
        # (a) s1
        s1_day = np.where(snap.derivative_valid, snap.s1_sinv * SECONDS_PER_DAY, np.nan)[rows, cols]
        limit = _symmetric_limit(s1_day)
        mesh_a = axes[0].pcolormesh(lon, lat, np.ma.masked_invalid(s1_day), shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        colorbar_a = figure.colorbar(mesh_a, cax=caxes[0])
        colorbar_a.set_label(S1_DAY_LABEL)
        pairs.append((axes[0], caxes[0]))
        # (c) 水平散度
        divergence = _divergence_sinv(snap.u, snap.v, snap.x_m_axis, snap.y_m_axis, snap.valid) * SECONDS_PER_DAY
        divergence_view = divergence[rows, cols]
        div_limit = _symmetric_limit(divergence_view)
        mesh_c = axes[2].pcolormesh(lon, lat, np.ma.masked_invalid(divergence_view), shading="auto", cmap="RdBu_r", vmin=-div_limit, vmax=div_limit)
        colorbar_c = figure.colorbar(mesh_c, cax=caxes[2])
        colorbar_c.set_label(T["divergence_label"] + r" $\nabla\cdot\mathbf{u}$ ($\mathrm{day^{-1}}$)")
        pairs.append((axes[2], caxes[2]))
        # (b) 流線
        dlon_dx, dlat_dy = _local_scale(data.lon, data.lat, snap.x_m_axis, snap.y_m_axis)
        usable = (snap.valid & np.isfinite(snap.u) & np.isfinite(snap.v))[rows, cols]
        u_plane = np.ma.masked_where(~usable, snap.u[rows, cols] * dlon_dx[cols][None, :])
        v_plane = np.ma.masked_where(~usable, snap.v[rows, cols] * dlat_dy[rows][:, None])
        lon_uniform, lat_uniform = _uniform_axis(lon), _uniform_axis(lat)
        stream_ok = False
        if lon_uniform is not None and lat_uniform is not None and bool(usable.any()):
            try:
                axes[1].streamplot(lon_uniform, lat_uniform, u_plane, v_plane, density=1.2, color="#2166ac", linewidth=0.8, arrowsize=0.8)
                stream_ok = True
            except (TypeError, ValueError):
                stream_ok = False
        if not stream_ok:
            _draw_quiver(axes[1], lon, lat, snap.u[rows, cols], snap.v[rows, cols], usable, dlon_dx[cols], dlat_dy[rows])
        for axis in axes:
            _plot_curves_lonlat(axis, snap.curves_m, mapper, color="#111111", linewidth=1.3, alpha=0.9, zorder=5)
            if snap.cores_m:
                cores = np.asarray(snap.cores_m, dtype=np.float64)
                core_lon, core_lat = mapper(cores[:, 0], cores[:, 1])
                axis.scatter(core_lon, core_lat, s=14, color="#e31a1c", edgecolor="white", linewidth=0.4, zorder=6)
            buffer_handle = _prepare_map_axis(axis, data, view, coastline)
            if buffer_handle is not None:
                buffer_handles.append(buffer_handle)
        for axis, title in zip(axes, titles, strict=True):
            axis.set_title(title, fontsize=10)
        axes[0].plot([], [], color="#111111", linewidth=1.3, label=T["trap_curve"])
        axes[0].plot([], [], marker="o", markersize=4, color="#e31a1c", linestyle="none", label=T["trap_core"])
        axes[0].legend(handles=[*axes[0].get_legend_handles_labels()[0], *buffer_handles[:1]], loc="lower right", fontsize=7)

    align_colorbars_to_axes(figure, tuple(pairs))
    _sequence_title(figure, fig_w, fig_h, f"{data.site_name_zh} — {T['method_title']}", subtitle)
    figure.text(0.5, (footer_h + note_h / 2.0) / fig_h, notes, ha="center", va="center", fontsize=8, color="#7f0000")
    _place_footer(figure, full_footer)
    _save(figure, plt, output_path, dpi)
    return output_path


# ---------------------------------------------------------------------------
# 3. 長壽命 track 核心路徑（Kunz Fig. 4b 風格）
# ---------------------------------------------------------------------------


def render_long_tracks(
    output_path: Path, data: SiteFigureData, tracks: TrackFigureInputs, *, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150
) -> Path:
    """長壽命 track 核心路徑，依壽命著色（由短到長疊繪，最長在最上層），起點小圓點。"""

    from matplotlib.collections import LineCollection

    matplotlib, plt = _load_matplotlib()
    day_unit = uses_day_unit(tracks.nominal_step_hours)
    scale = 24.0 if day_unit else 1.0
    view = _view_extent(data)
    rows, cols = view
    mapper = MetricToLonLat(tracks.x_m_axis, tracks.y_m_axis, data.lon, data.lat)
    drawn = list(tracks.long_tracks)
    n_total = int(tracks.n_long_total)
    extra_lines = []
    if n_total > len(drawn):
        extra_lines.append(f"{T['long_capped_a']} {len(drawn)} {T['long_capped_b']} {n_total} {T['long_capped_c']}")
    if tracks.null_long_mean is not None:
        extra_lines.append(f"{T['null_long_mean']} {tracks.null_long_mean:.1f} {T['n_tracks']}（n_seed={tracks.null_n_seeds}）")
    if tracks.n_long_coastal_total is not None and tracks.n_long_offshore_total is not None:
        extra_lines.append(
            f"{T['long_coast_real']} {int(tracks.n_long_coastal_total)} {T['n_tracks']}，"
            f"{T['long_coast_offshore']} {int(tracks.n_long_offshore_total)} {T['n_tracks']}"
        )
        if tracks.null_long_coastal_mean is not None and tracks.null_long_offshore_mean is not None:
            extra_lines.append(
                f"{T['long_coast_null']} {tracks.null_long_coastal_mean:.1f} {T['n_tracks']}，"
                f"{T['long_coast_offshore']} {tracks.null_long_offshore_mean:.1f} {T['n_tracks']}（n_seed={tracks.null_n_seeds}）"
            )
    if tracks.n_long_coastal_total is not None and tracks.n_long_offshore_total is not None:
        extra_lines.append(T["coast_footnote"])
    full_footer = footer + "".join(f"\n{line}" for line in extra_lines)
    layout = _single_map_layout(data, view, full_footer, legend_height=0.0, max_width=7.0, max_height=6.0)
    figure = plt.figure(figsize=(layout["fig_w"], layout["fig_h"]))
    axis = _add_inch_axes(figure, layout["x0"], layout["y0"], layout["panel_w"], layout["panel_h"])
    cax = _add_inch_axes(figure, layout["x0"] + layout["panel_w"] + _CBAR_PAD_INCH, layout["y0"], _CBAR_WIDTH_INCH, layout["panel_h"])
    lifetimes = np.asarray([item[0] for item in drawn], dtype=np.float64) / scale
    vmin = float(tracks.long_track_hours) / scale
    vmax = max(float(lifetimes.max()) if lifetimes.size else vmin, vmin + 1.0e-9)
    if lifetimes.size and vmax <= vmin:
        vmax = vmin + 1.0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    legend_handles: list[Any] = []
    if drawn:
        from matplotlib.lines import Line2D

        coastal_flags = list(tracks.long_coastal) if len(tracks.long_coastal) == len(drawn) else [False] * len(drawn)
        order = sorted(range(len(drawn)), key=lambda index: drawn[index][0])
        segments, colours, starts, styles = [], [], [], []
        for index in order:
            lifetime, x_m, y_m = drawn[index]
            lon_t, lat_t = mapper(x_m, y_m)
            segments.append(np.column_stack([lon_t, lat_t]))
            colours.append(cmap(norm(lifetime / scale)))
            starts.append((lon_t[0], lat_t[0]))
            styles.append((0, (3.0, 2.0)) if coastal_flags[index] else "solid")
        axis.add_collection(LineCollection(segments, colors=colours, linewidths=0.9, alpha=0.9, zorder=4, linestyles=styles))
        start_array = np.asarray(starts)
        axis.scatter(start_array[:, 0], start_array[:, 1], s=6, color="black", edgecolor="white", linewidth=0.3, zorder=5, label=T["long_tracks_start"])
        legend_handles.extend(axis.get_legend_handles_labels()[0])
        if any(coastal_flags):
            legend_handles.append(Line2D([], [], color="0.25", linewidth=1.2, linestyle="-", label=T["offshore_track_legend"]))
            legend_handles.append(Line2D([], [], color="0.25", linewidth=1.2, linestyle=(0, (3.0, 2.0)), label=T["coastal_locked_legend"]))
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(mappable, cax=cax)
    colorbar.set_label(T["lifetime_days"] if day_unit else T["lifetime_hours"])
    buffer_handle = _prepare_map_axis(axis, data, view, coastline)
    if buffer_handle is not None:
        legend_handles.append(buffer_handle)
    if legend_handles:
        axis.legend(handles=legend_handles, loc="lower right", fontsize=7)
    if not drawn:
        _empty_state(axis, T["no_data"])
    threshold_text = duration_text(tracks.long_track_hours, day_unit=day_unit)
    subtitle = f"{T['long_tracks_criterion']} {threshold_text}｜n = {n_total} {T['n_tracks']}"
    align_colorbars_to_axes(figure, ((axis, cax),))
    _sequence_title(figure, layout["fig_w"], layout["fig_h"], f"{data.site_name_zh} — {T['long_tracks_title']}", subtitle)
    _place_footer(figure, full_footer)
    _save(figure, plt, output_path, dpi)
    return output_path


# ---------------------------------------------------------------------------
# 4. track 通過格點數（Kunz Fig. 4c 風格）
# ---------------------------------------------------------------------------


def render_track_density(
    output_path: Path, data: SiteFigureData, tracks: TrackFigureInputs, *, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150
) -> Path:
    """兩個並排面板：通過各格的不同 track 數（全部真實 track）與長壽命 track 數。"""

    matplotlib, plt = _load_matplotlib()
    from matplotlib.ticker import MaxNLocator

    day_unit = uses_day_unit(tracks.nominal_step_hours)
    view = _view_extent(data)
    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    fields = (
        (f"{T['track_pass_all']}", "track_count", "YlGnBu"),
        (f"{T['track_pass_long']} {duration_text(tracks.long_track_hours, day_unit=day_unit)}", "long_track_count", "YlOrRd"),
    )
    full_footer = footer + "\n" + T["track_pass_note"]
    panel_w, panel_h = _fit_panel(map_data_aspect(data, view), 4.6, 4.2)
    column_pitch = _LEFT_INCH + panel_w + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH
    footer_h = _footer_height_inch(full_footer)
    fig_w = max(2.0 * column_pitch, _MIN_FIGURE_WIDTH)
    title_h = 0.78
    fig_h = 0.1 + title_h + 0.3 + panel_h + _BOTTOM_LABEL_INCH + footer_h
    x_offset = (fig_w - 2.0 * column_pitch) / 2.0
    y0 = footer_h + _BOTTOM_LABEL_INCH
    figure = plt.figure(figsize=(fig_w, fig_h))
    pairs: list[tuple[Any, Any]] = []
    for index, (title, key, cmap) in enumerate(fields):
        x0 = x_offset + index * column_pitch + _LEFT_INCH
        axis = _add_inch_axes(figure, x0, y0, panel_w, panel_h)
        cax = _add_inch_axes(figure, x0 + panel_w + _CBAR_PAD_INCH, y0, _CBAR_WIDTH_INCH, panel_h)
        counts = np.asarray(data.spatial_all[key], dtype=np.float64)[rows, cols]
        inside = np.asarray(data.site_mask, dtype=bool)[rows, cols]
        masked = np.ma.masked_where(~inside | ~np.isfinite(counts), counts)
        vmax = max(float(masked.max()) if masked.count() else 1.0, 1.0)
        mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        colorbar = figure.colorbar(mesh, cax=cax)
        colorbar.set_label(T["n_tracks"])
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()
        axis.set_title(title, fontsize=10)
        buffer_handle = _prepare_map_axis(axis, data, view, coastline)
        if buffer_handle is not None:
            axis.legend(handles=[buffer_handle], loc="lower right", fontsize=7)
        pairs.append((axis, cax))
    align_colorbars_to_axes(figure, tuple(pairs))
    _sequence_title(figure, fig_w, fig_h, f"{data.site_name_zh} — {T['track_density_title']}", None)
    _place_footer(figure, full_footer)
    _save(figure, plt, output_path, dpi)
    return output_path


# ---------------------------------------------------------------------------
# 5. 傳播速度（Kunz Fig. 5c 類比）
# ---------------------------------------------------------------------------


def rose_histogram(cx_kmday: np.ndarray, cy_kmday: np.ndarray, sectors: int = ROSE_SECTORS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """去向玫瑰圖統計：回傳 (扇區中心方位角度（自北順時針）, 各扇區頻率, 各扇區平均速率)。

    方位角 = ``atan2(cx, cy)``（0° 北、90° 東）；扇區以正北為中心對稱切分。平均速率為該扇區
    ``hypot(cx, cy)`` 的平均，空扇區為 NaN；頻率總和為 1（無樣本時全為 0）。速率為零（或非有限）的樣本
    沒有方向（``atan2(0, 0)`` 會被誤歸到正北），因此明確排除，不計入任何扇區與分母。
    """

    cx = np.asarray(cx_kmday, dtype=np.float64)
    cy = np.asarray(cy_kmday, dtype=np.float64)
    moving = np.isfinite(cx) & np.isfinite(cy) & (np.hypot(cx, cy) > 0.0)
    cx, cy = cx[moving], cy[moving]
    width = 360.0 / sectors
    centres = np.arange(sectors) * width
    frequency = np.zeros(sectors)
    mean_speed = np.full(sectors, np.nan)
    if cx.size:
        bearing = np.degrees(np.arctan2(cx, cy)) % 360.0
        index = np.floor((bearing + width / 2.0) / width).astype(int) % sectors
        speed = np.hypot(cx, cy)
        for sector in range(sectors):
            selected = index == sector
            frequency[sector] = float(selected.sum()) / cx.size
            if selected.any():
                mean_speed[sector] = float(speed[selected].mean())
    return centres, frequency, mean_speed


def render_propagation(
    output_path: Path, data: SiteFigureData, tracks: TrackFigureInputs, *, footer: str, dpi: int = 150
) -> Path:
    """核心傳播速度：每條 track 的線性擬合速度分量直方圖（含中位數線）與去向玫瑰圖，只含移動 track。

    速度來自每條 track 核心位置 ``x(t)、y(t)`` 的最小平方線性擬合（至少 3 個觀測）；擬合位移小於 1 格的
    track 視為靜止，不進入直方圖與玫瑰圖（零速度沒有方向），但在副標題明確標示移動／靜止條數與靜止比例，
    註腳附殘差 RMS 中位數。
    """

    matplotlib, plt = _load_matplotlib()
    all_cx = np.asarray(tracks.prop_cx_kmday, dtype=np.float64)
    all_cy = np.asarray(tracks.prop_cy_kmday, dtype=np.float64)
    stationary = (
        np.asarray(tracks.prop_stationary, dtype=bool) if tracks.prop_stationary is not None else np.zeros(all_cx.shape, dtype=bool)
    )
    cx, cy = all_cx[~stationary], all_cy[~stationary]
    n = int(cx.size)
    n_stationary = int(stationary.sum())
    n_fitted = int(all_cx.size)
    stationary_fraction = (n_stationary / n_fitted) if n_fitted else float("nan")
    subtitle = (
        f"{T['propagation_moving']} {n}／{T['propagation_stationary']} {n_stationary}"
        + (f"（{T['propagation_stationary_frac']} {100.0 * stationary_fraction:.0f}%）" if n_fitted else "")
        + f"｜{T['propagation_stationary_def']}"
        + (f"｜{T['propagation_min_obs_a']} {int(tracks.prop_min_obs)} {T['propagation_min_obs_b']}" if tracks.prop_min_obs else "")
    )
    resid_text = ""
    if tracks.prop_resid_rms_m is not None and np.size(tracks.prop_resid_rms_m):
        resid_text = f"  {T['propagation_resid']} {float(np.median(tracks.prop_resid_rms_m)):.0f} {T['unit_m']}"
    full_footer = f"{footer}\n{T['propagation_fitted']} {n_fitted} {T['propagation_tracks']}{resid_text}\n{T['propagation_fit_note'].format(min_obs=int(tracks.prop_min_obs) if tracks.prop_min_obs else 3)}"
    footer_h = _footer_height_inch(full_footer)
    hist_w, hist_h = 3.5, 3.1
    rose_size = 3.1
    fig_w = _LEFT_INCH + hist_w + 0.35 + hist_w + 0.35 + 0.55 + rose_size + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH
    title_h = 0.78
    fig_h = 0.1 + title_h + 0.3 + max(hist_h, rose_size) + _BOTTOM_LABEL_INCH + footer_h
    y0 = footer_h + _BOTTOM_LABEL_INCH
    figure = plt.figure(figsize=(fig_w, fig_h))

    x_cx = _LEFT_INCH
    x_cy = x_cx + hist_w + 0.35 + 0.15
    x_rose = x_cy + hist_w + 0.35 + 0.55
    ax_cx = _add_inch_axes(figure, x_cx, y0, hist_w, hist_h)
    ax_cy = _add_inch_axes(figure, x_cy, y0, hist_w, hist_h)
    if n:
        edges = np.histogram_bin_edges(np.concatenate([cx, cy]), bins="fd")
        if edges.size > 61:
            edges = np.linspace(float(edges[0]), float(edges[-1]), 61)
    else:
        edges = np.linspace(-1.0, 1.0, 11)
    for axis, values, colour, title in (
        (ax_cx, cx, "#1f77b4", T["propagation_east"]),
        (ax_cy, cy, "#d95f02", T["propagation_north"]),
    ):
        axis.set_title(f"{title} (n={n})", fontsize=10)
        axis.set_xlabel("km/day")
        axis.grid(alpha=0.25, linewidth=0.5)
        if n:
            axis.hist(values, bins=edges, color=colour, alpha=0.85, edgecolor="white", linewidth=0.4)
            median = float(np.median(values))
            axis.axvline(median, color="#111111", linestyle="--", linewidth=1.3, label=f"{T['propagation_median']} {median:.1f} km/day")
            axis.axvline(0.0, color="0.5", linewidth=0.7)
            axis.legend(loc="upper right", fontsize=7)
        else:
            _empty_state(axis, T["no_data"])
    ax_cx.set_ylabel(T["propagation_samples"])

    rose_rect = (x_rose / fig_w, y0 / fig_h, rose_size / fig_w, rose_size / fig_h)
    ax_rose = figure.add_axes(rose_rect, projection="polar")
    cax = _add_inch_axes(figure, x_rose + rose_size + _CBAR_PAD_INCH, y0, _CBAR_WIDTH_INCH, rose_size)
    centres, frequency, mean_speed = rose_histogram(cx, cy)
    finite_speed = mean_speed[np.isfinite(mean_speed)]
    vmax = float(finite_speed.max()) if finite_speed.size else 1.0
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=max(vmax, 1.0e-9))
    cmap = plt.get_cmap("viridis")
    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)
    if n:
        colours = [cmap(norm(value)) if np.isfinite(value) else (0.85, 0.85, 0.85, 1.0) for value in mean_speed]
        ax_rose.bar(np.radians(centres), frequency, width=np.radians(360.0 / ROSE_SECTORS) * 0.92, color=colours, edgecolor="white", linewidth=0.5)
    ax_rose.set_thetagrids(np.arange(0, 360, 45), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=8)
    ax_rose.tick_params(axis="y", labelsize=7)
    mean_all = float(np.mean(np.hypot(cx, cy))) if n else float("nan")
    rose_title = T["propagation_rose"] + (f"\n{T['propagation_mean_speed']} {mean_all:.1f} km/day" if n else "")
    ax_rose.set_title(rose_title, fontsize=10, pad=14)
    colorbar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label(f"{T['propagation_mean_speed']} (km/day)")
    if not n:
        ax_rose.text(0.5, 0.5, T["no_data"], transform=ax_rose.transAxes, ha="center", va="center", color="#7f0000")
    ax_rose.set_ylabel("")
    align_colorbars_to_axes(figure, ((ax_rose, cax),))
    _sequence_title(figure, fig_w, fig_h, f"{data.site_name_zh} — {T['propagation_title']}", subtitle)
    _place_footer(figure, full_footer)
    _save(figure, plt, output_path, dpi)
    return output_path


# ---------------------------------------------------------------------------
# 入口：整組輸出
# ---------------------------------------------------------------------------


def _require_font(font_selection: CJKFontSelection | None) -> CJKFontSelection:
    if font_selection is None:
        raise ValueError("沒有合格的 CJK 字型，拒絕輸出成果圖（可改用 --no-figures）")
    return font_selection


def _shared_footer(data: SiteFigureData, font_selection: CJKFontSelection, state_label: str, param_id: str, period_label: str) -> str:
    return _footer(period_label, state_label, param_id, data.n_valid_snapshots, font_selection.family)


def render_snapshot_figures(
    output_dir: Path,
    data: SiteFigureData,
    snap: ExampleSnapshot | None,
    *,
    font_selection: CJKFontSelection | None,
    coastline: CoastlineGeometry | None,
    state_label: str,
    param_id: str,
    period_label: str,
    nominal_step_hours: float = 1.0,
    dpi: int = 150,
) -> list[str]:
    """輸出 ``example`` 與 ``method_comparison`` 兩張圖，回傳相對 ``output_dir`` 的路徑。"""

    font = _require_font(font_selection)
    footer = _shared_footer(data, font, state_label, param_id, period_label)
    site_dir = output_dir / "figures" / data.site_id
    stem = f"site_{data.site_id}"
    day_unit = uses_day_unit(nominal_step_hours)
    with cjk_font_context(font):
        render_example(site_dir / f"{stem}_example.png", data, snap, coastline=coastline, footer=footer, day_unit=day_unit, dpi=dpi)
        render_method_comparison(site_dir / f"{stem}_method_comparison.png", data, snap, coastline=coastline, footer=footer, day_unit=day_unit, dpi=dpi)
    return [f"figures/{data.site_id}/{stem}_{kind}.png" for kind in EXAMPLE_FIGURE_KINDS]


def render_coastal_example(
    output_dir: Path,
    data: SiteFigureData,
    snap: ExampleSnapshot | None,
    *,
    font_selection: CJKFontSelection | None,
    coastline: CoastlineGeometry | None,
    state_label: str,
    param_id: str,
    period_label: str,
    nominal_step_hours: float = 1.0,
    dpi: int = 150,
) -> list[str]:
    """輸出 ``site_<id>_example_coastal.png``（最長近岸型 track 的代表圖）；沒有近岸型 track（``snap`` 為 ``None``）時不輸出。"""

    if snap is None:
        return []
    font = _require_font(font_selection)
    footer = _shared_footer(data, font, state_label, param_id, period_label)
    site_dir = output_dir / "figures" / data.site_id
    with cjk_font_context(font):
        render_example(
            site_dir / f"site_{data.site_id}_example_coastal.png", data, snap, coastline=coastline, footer=footer,
            day_unit=uses_day_unit(nominal_step_hours), dpi=dpi,
        )
    return [f"figures/{data.site_id}/site_{data.site_id}_example_coastal.png"]


def render_track_figures(
    output_dir: Path,
    data: SiteFigureData,
    tracks: TrackFigureInputs,
    *,
    font_selection: CJKFontSelection | None,
    coastline: CoastlineGeometry | None,
    state_label: str,
    param_id: str,
    period_label: str,
    dpi: int = 150,
) -> list[str]:
    """輸出 ``long_tracks``、``track_density``、``propagation`` 三張圖，回傳相對 ``output_dir`` 的路徑。"""

    font = _require_font(font_selection)
    footer = _shared_footer(data, font, state_label, param_id, period_label)
    site_dir = output_dir / "figures" / data.site_id
    stem = f"site_{data.site_id}"
    with cjk_font_context(font):
        render_long_tracks(site_dir / f"{stem}_long_tracks.png", data, tracks, coastline=coastline, footer=footer, dpi=dpi)
        render_track_density(site_dir / f"{stem}_track_density.png", data, tracks, coastline=coastline, footer=footer, dpi=dpi)
        render_propagation(site_dir / f"{stem}_propagation.png", data, tracks, footer=footer, dpi=dpi)
    return [f"figures/{data.site_id}/{stem}_{kind}.png" for kind in TRACK_FIGURE_KINDS]


__all__ = [
    "EXAMPLE_FIGURE_KINDS",
    "TRACK_FIGURE_KINDS",
    "ExampleSnapshot",
    "LaterObservation",
    "MetricToLonLat",
    "TrackFigureInputs",
    "pick_later_observations",
    "prepare_example_snapshot",
    "relative_time_label",
    "render_coastal_example",
    "render_snapshot_figures",
    "render_track_figures",
    "resolve_source_paths",
    "rose_histogram",
    "uses_day_unit",
    "verify_recomputed_core",
]
