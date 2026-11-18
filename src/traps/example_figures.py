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

正式成果圖的版面下緣僅保留圖例或各座標軸自身的圖例；期間、狀態、來源、字型、
選取規則與方法說明等 metadata/footer 不再繪製。代表圖與方法比較圖的上方副標題
保留起始 UTC 與壽命，但不顯示 ``track_id``；``track_id`` 仍由資料結構與 manifest
保存，以維持可追溯性。
"""

from __future__ import annotations

import datetime as _datetime
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import math

import numpy as np

from . import concise_style
from . import figure_standard as std
from .aggregate_figures import (
    concise_map_boxes,
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
    _load_matplotlib,
    _prepare_map_axis,
    _save,
    _view_extent,
    coast_buffer_label,
    map_data_aspect,
    standard_map_frame,
    style_colorbar,
)
from .colorbar_layout import align_colorbars_to_axes
from .coastline import CoastlineGeometry
from .report_font import FIGURE_TEXT as T
from .report_font import CJKFontSelection, cjk_font_context
from .vector_figures import VECTOR_REPORT_WIDTH_INCH, vector_report_enabled

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
    zoom: dict[str, Any] = field(default_factory=dict)
    """縮放代表圖（簡要報告）的選取說明：邊界距離、候選數、放寬與否；一般圖為空。"""


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
    colour_max_hours: float | None = None
    """``long_tracks`` 色階上限（小時）；``None`` 時沿用「該站自己最長的 track」。

    同一產品要讓五站共用色階時，由 caller（``aggregate`` 或重繪工具）以
    :func:`shared_long_track_colour_max_hours` 跨站計算後傳入；色階下限固定為
    ``long_track_hours``。小於本站最長 track 時視為呼叫錯誤，避免顏色被截斷。
    """


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


def shared_long_track_colour_max_hours(
    max_lifetimes_hours: Sequence[float], *, nominal_step_hours: float
) -> float | None:
    """跨站共用的 ``long_tracks`` 色階上限（小時）。

    取各站最長 track 壽命的最大值，再進位到整齊的刻度：以「天」顯示的產品
    （名目步長為 24 h 的整數倍）進位到 10 天的倍數，逐時產品進位到 12 小時的倍數。
    沒有任何有限值（所有站都沒有長壽命 track）時回傳 ``None``，由 renderer 沿用
    各站自己的上限。只改顏色對應的上下限，不影響任何 track 或統計。
    """

    values = [float(value) for value in max_lifetimes_hours if value is not None and np.isfinite(float(value))]
    if not values:
        return None
    if uses_day_unit(nominal_step_hours):
        step_hours = 10.0 * 24.0
    else:
        step_hours = 12.0
    return float(np.ceil(max(values) / step_hours - 1.0e-9) * step_hours)


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


def later_curve_label(dt_seconds: float, *, day_unit: bool) -> str:
    """圖例的後續曲線標籤：逐時為「3 小時後曲線」，逐日維持「+3 天 曲線」。"""

    if day_unit:
        return f"{relative_time_label(dt_seconds, day_unit=True)} 曲線"
    return f"{round(dt_seconds / 3_600.0, 1):g} 小時後曲線"


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
    later_fractions: Sequence[float] = LATER_CURVE_FRACTIONS,
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
    for index in pick_later_observations(times, later_fractions):
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
    """在固定列印寬度內配置主標題與副標題。

    報告模式的傳播統計副標題可能同時包含移動、靜止、比例及擬合條件；先依
    可讀寬度換行，再依主標題行數下移副標題，避免長字串侵入圖框或互相覆蓋。
    換行只改文字排版，所有數值與欄位內容仍完整保留。
    """

    report_mode = vector_report_enabled()
    title_text = textwrap.fill(title, width=30 if report_mode else 80, break_long_words=False, break_on_hyphens=False)
    title_lines = title_text.count("\n") + 1
    figure.text(0.5, 1.0 - 0.10 / fig_h, title_text, ha="center", va="top", fontsize=12 if report_mode else 13)
    if subtitle:
        subtitle_text = textwrap.fill(subtitle, width=50 if report_mode else 100, break_long_words=False, break_on_hyphens=False)
        subtitle_offset = 0.42 + 0.22 * max(title_lines - 1, 0)
        figure.text(0.5, 1.0 - subtitle_offset / fig_h, subtitle_text, ha="center", va="top", fontsize=9)


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
    data: SiteFigureData,
    view: tuple[slice, slice],
    footer: str = "",
    *,
    legend_height: float,
    max_width: float = 7.4,
    max_height: float = 6.0,
) -> dict[str, float]:
    """計算單一地圖圖面的英吋配置，讓下方只保留圖例所需的空間。

    ``data`` 與 ``view`` 決定地圖面板及色條的實際寬高；``legend_height`` 是圖例位於
    面板下方時的獨立版面帶高度。正式成果不再繪製 metadata、描述文字或 footer，
    因此下緣只保留經度標籤 ``_BOTTOM_LABEL_INCH`` 與圖例帶，不呼叫共用 footer
    高度計算；``footer`` 位置參數則為舊呼叫端保留，不影響版面。回傳的
    ``legend_y`` 是 figure 座標中的圖例錨點，固定在最下緣，
    使圖例不會侵入經度標籤區；輸入資料的科學值與地圖比例不在此函式中改動。
    """

    del footer  # 保留原本第三個位置參數，讓外部 renderer/測試不因移除 footer 而失效。
    report_mode = vector_report_enabled()
    report_panel_width = max(
        0.5,
        VECTOR_REPORT_WIDTH_INCH - _LEFT_INCH - _CBAR_PAD_INCH - _CBAR_WIDTH_INCH - _CBAR_RIGHT_INCH,
    )
    panel_w, panel_h = _fit_panel(
        map_data_aspect(data, view),
        min(max_width, report_panel_width) if report_mode else max_width,
        min(max_height, 4.6) if report_mode else max_height,
    )
    content_w = _LEFT_INCH + panel_w + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH
    fig_w = max(content_w, VECTOR_REPORT_WIDTH_INCH if report_mode else _MIN_FIGURE_WIDTH)
    title_h = 0.78
    # 沒有 figure-level 圖例的單地圖（例如 long_tracks）仍需為經度標籤保留
    # 額外下邊界；否則固定 6.3 英吋後 x tick 會落到 Figure 外。代表圖有圖例帶
    # 時沿用其既有高度，避免重複增加空白。
    report_bottom_extra = 0.24 if report_mode and legend_height <= 0.0 else 0.0
    fig_h = 0.1 + title_h + panel_h + _BOTTOM_LABEL_INCH + legend_height + report_bottom_extra
    return {
        "fig_w": fig_w,
        "fig_h": fig_h,
        "x0": (fig_w - content_w) / 2.0 + _LEFT_INCH,
        "y0": legend_height + _BOTTOM_LABEL_INCH + report_bottom_extra,
        "panel_w": panel_w,
        "panel_h": panel_h,
        "legend_y": 0.0,
        "legend_height": legend_height,
    }


# ---------------------------------------------------------------------------
# 1. 代表 TRAP（Kunz 2024 Fig. 2 風格）
# ---------------------------------------------------------------------------


def _example_tag(snap: ExampleSnapshot) -> str:
    """代表圖副標題的明確註記：近岸型，或壽命未超過隨機對照 p99 的整體最長者。"""

    if snap.is_coastal:
        return T["coastal_tag"]
    if snap.criterion == "fallback_offshore_below_null":
        return T["below_null_tag"]
    return ""


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
    地圖範圍是站點分析區外接框外擴；正式成果下方只保留圖例，起始 UTC 與壽命放在
    上方副標題，``track_id`` 僅保留在輸入資料及 manifest，不以文字呈現在圖面。
    ``footer`` 參數保留給既有呼叫端，但不再繪製 metadata、描述文字或 footer。
    """

    matplotlib, plt = _load_matplotlib()
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as path_effects

    base_view = padded_view(data)
    report_mode = vector_report_enabled()
    limits = None
    view = base_view
    concise = report_mode and concise_style.concise_enabled()
    if concise:
        data = replace(data, site_name_zh=concise_style.display_site_name(data.site_id, data.site_name_zh))
    if report_mode:
        # 標準 multi 畫布（6.3×7.0）：高地圖＋垂直 s1 色條＋固定圖例帶；固定軸框、擴張經緯度範圍。
        view, limits = standard_map_frame(data, base_view, std.EXAMPLE_AXES)
    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    if report_mode:
        layout = {
            "fig_w": std.WIDTH_INCH,
            "fig_h": std.MULTI_HEIGHT_INCH,
            "panel_w": std.EXAMPLE_AXES.w,
            "panel_h": std.EXAMPLE_AXES.h,
        }
        list_labels = snap is not None and _later_labels_overlap(snap, data, view, layout["panel_w"], layout["panel_h"])
        figure = plt.figure(figsize=(layout["fig_w"], layout["fig_h"]))
        axis = std.add_box_axes(figure, std.EXAMPLE_AXES)
        cax = std.add_box_axes(figure, std.EXAMPLE_CBAR)
    else:
        layout = _single_map_layout(data, view, legend_height=0.62)
        # 後續核心幾乎沒有移動時，地圖上的相對時間標籤會彼此重疊：改為圖例中的位移清單。
        list_labels = snap is not None and _later_labels_overlap(snap, data, view, layout["panel_w"], layout["panel_h"])
        if list_labels:
            layout = _single_map_layout(data, view, legend_height=0.86)
        figure = plt.figure(figsize=(layout["fig_w"], layout["fig_h"]))
        axis = _add_inch_axes(figure, layout["x0"], layout["y0"], layout["panel_w"], layout["panel_h"])
        cax = _add_inch_axes(figure, layout["x0"] + layout["panel_w"] + _CBAR_PAD_INCH, layout["y0"], _CBAR_WIDTH_INCH, layout["panel_h"])

    handles: list[Any] = []
    labels: list[str] = []
    if snap is None:
        blank = axis.pcolormesh(lon, lat, np.ma.masked_all((lat.size, lon.size)), shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
        colorbar = figure.colorbar(blank, cax=cax)
        colorbar.set_label(S1_DAY_LABEL)
        if report_mode:
            style_colorbar(colorbar, S1_DAY_LABEL)
        _prepare_map_axis(axis, data, view, coastline, limits=limits)
        _empty_state(axis, T["example_no_track"])
        subtitle = T["example_no_track_short"]
    else:
        mapper = MetricToLonLat(snap.x_m_axis, snap.y_m_axis, data.lon, data.lat)
        s1_full = np.where(snap.derivative_valid, snap.s1_sinv * SECONDS_PER_DAY, np.nan)
        s1_day = s1_full[rows, cols]
        # 色階只取原本（未擴張）視窗內的 s1，擴張地圖範圍不改變色階。
        limit = _symmetric_limit(s1_full[base_view[0], base_view[1]])
        s1_extend = "neither"
        if concise:
            # 簡要報告圖組：五站共用同一個對稱 s1 色階；資料超出時以延伸箭頭標示。
            concise_style.record_limit("example", data.site_id, "s1_day", (-limit, limit))
            shared = concise_style.shared_limits("example", "s1_day")
            if shared is not None:
                limit = float(max(abs(shared[0]), abs(shared[1])))
            s1_extend = "neither"  # 簡要報告：色條兩端平整，超出的值以端點色飽和
        mesh = axis.pcolormesh(lon, lat, np.ma.masked_invalid(s1_day), shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        colorbar = figure.colorbar(mesh, cax=cax, extend=s1_extend)
        colorbar.set_label(S1_DAY_LABEL)
        if report_mode:
            style_colorbar(colorbar, S1_DAY_LABEL)
            if concise:
                concise_style.style_colorbar_shared(
                    colorbar, -limit, limit, S1_DAY_LABEL, name="s1_day", extend=s1_extend, clipped=concise_style.clipped_fraction(s1_day, -limit, limit)
                )
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
        buffer_handle = _prepare_map_axis(axis, data, view, coastline, limits=limits)
        if buffer_handle is not None:
            handles.append(buffer_handle)
            labels.append(coast_buffer_label(data))
        subtitle = (
            f"{T['example_birth']} {time_label_utc(snap.time_utc_ns)} UTC｜"
            f"{T['example_lifetime']} {duration_text(snap.lifetime_hours, day_unit=day_unit)}{_example_tag(snap)}"
        )

    title_key = "coastal_example_title" if (snap is not None and snap.variant == "coastal") else "example_title"
    if report_mode:
        std.legend_below(figure, layout["fig_h"], handles, labels, ncol=3)
        std.draw_title_block(figure, layout["fig_h"], f"{data.site_name_zh} — {T[title_key]}", subtitle)
    else:
        align_colorbars_to_axes(figure, ((axis, cax),))
        if handles:
            figure.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, layout["legend_y"]),
                ncol=3,
                fontsize=8,
                frameon=False,
                columnspacing=1.6,
                handlelength=2.2,
            )
        _sequence_title(figure, layout["fig_w"], layout["fig_h"], f"{data.site_name_zh} — {T[title_key]}", subtitle)
    _save(figure, plt, output_path, dpi)
    return output_path


# ---------------------------------------------------------------------------
# 1b. 縮放代表 TRAP（簡要報告：Kunz et al. 2024 Fig. 2 風格，單一 TRAP 的放大視窗）
# ---------------------------------------------------------------------------

KM_PER_DEGREE = 111.195
"""每度緯度的公里數（地球平均半徑 6371 km）；視窗大小以此換算，經度另乘 cos 緯度。"""
ZOOM_MARGIN_KM = 3.0
"""後續位置的外接框與視窗邊緣至少保留的距離（公里）。"""
ZOOM_LEGEND_LATER_ALPHAS = (0.85, 0.5)
ZOOM_LATER_COLOUR = "#f0806a"


def track_edge_distance_m(core_x_m: np.ndarray, core_y_m: np.ndarray, x_m_axis: np.ndarray, y_m_axis: np.ndarray) -> float:
    """整條核心路徑到分析區矩形外緣（格網最外緣）的最小距離（公尺）；負值代表路徑超出格網。"""

    x_low, x_high = float(np.min(x_m_axis)), float(np.max(x_m_axis))
    y_low, y_high = float(np.min(y_m_axis)), float(np.max(y_m_axis))
    cx = np.asarray(core_x_m, dtype=np.float64)
    cy = np.asarray(core_y_m, dtype=np.float64)
    return float(min(np.min(cx) - x_low, x_high - np.max(cx), np.min(cy) - y_low, y_high - np.max(cy)))


def select_zoom_track(
    born_records: Sequence[Mapping[str, Any]],
    observations: Mapping[str, np.ndarray],
    x_m_axis: np.ndarray,
    y_m_axis: np.ndarray,
    *,
    min_lifetime_hours: float,
    min_edge_km: Sequence[float] = (10.0, 5.0),
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    """縮放代表圖的選取規則（只影響呈現，不影響任何統計）。

    候選為站內出生、起訖皆在觀測期內（``born_records``）且壽命不小於 ``min_lifetime_hours`` 的 track。
    在整條核心路徑距分析區矩形外緣都不小於 ``min_edge_km[0]`` 公里者中取壽命最長，同壽命取平均核心 s1
    最強（最負），再取較小 ``track_id``；沒有任何 track 符合時依序放寬到 ``min_edge_km`` 的下一個距離。
    海岸線鄰近不排除。回傳 ``(選中的紀錄或 None, 說明字典)``。
    """

    scored: list[tuple[Mapping[str, Any], float]] = []
    for item in born_records:
        if float(item["lifetime_hours"]) < float(min_lifetime_hours) - 1.0e-9:
            continue
        start, stop = int(item["start"]), int(item["stop"])
        distance = track_edge_distance_m(observations["core_x_m"][start:stop], observations["core_y_m"][start:stop], x_m_axis, y_m_axis)
        scored.append((item, distance))
    info: dict[str, Any] = {
        "rule": "born_in_site_uncensored_lifetime_ge_min_longest_with_core_ge_edge_km_tiebreak_stronger_mean_s1",
        "min_lifetime_hours": float(min_lifetime_hours),
        "n_candidates": len(scored),
        "edge_thresholds_km": [float(value) for value in min_edge_km],
    }
    for rank, threshold in enumerate(min_edge_km):
        eligible = [(item, distance) for item, distance in scored if distance >= float(threshold) * 1000.0]
        if not eligible:
            continue
        chosen, distance = min(
            eligible, key=lambda pair: (-float(pair[0]["lifetime_hours"]), float(pair[0]["mean_core_s1_sinv"]), int(pair[0]["track_id"]))
        )
        info.update(
            {
                "status": "ok",
                "edge_threshold_used_km": float(threshold),
                "relaxed": bool(rank > 0),
                "n_eligible": len(eligible),
                "min_edge_distance_km": distance / 1000.0,
            }
        )
        return chosen, info
    info.update({"status": "no_track_meets_edge_rule", "relaxed": True, "n_eligible": 0})
    return None, info


def _polyline_length_m(points_m: np.ndarray) -> float:
    """折線長度（公尺）；少於兩點為 0。"""

    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))))


def _distance_to_mask_m(mask: np.ndarray, x_m_axis: np.ndarray, y_m_axis: np.ndarray) -> np.ndarray:
    """每個格點到最近 ``mask`` 為真格點中心的距離（公尺）；``mask`` 全為假時回傳 inf。以 numpy 分塊暴力計算（不需 scipy）。"""

    mask = np.asarray(mask, dtype=bool)
    out = np.full(mask.shape, np.inf)
    if not mask.any():
        return out
    rows, cols = np.nonzero(mask)
    hx = np.asarray(x_m_axis, dtype=np.float64)[cols]
    hy = np.asarray(y_m_axis, dtype=np.float64)[rows]
    gx = np.asarray(x_m_axis, dtype=np.float64)
    for row in range(mask.shape[0]):
        dy2 = (float(y_m_axis[row]) - hy) ** 2
        dist2 = (gx[:, None] - hx[None, :]) ** 2 + dy2[None, :]
        out[row] = np.sqrt(dist2.min(axis=1))
    return out


def hourly_candidate_features(
    born_records: Sequence[Mapping[str, Any]],
    observations: Mapping[str, np.ndarray],
    x_m_axis: np.ndarray,
    y_m_axis: np.ndarray,
    hotspot_mask: np.ndarray,
    *,
    detect_root: Path | None,
    domain_id: str,
    min_lifetime_hours: float,
    later_fractions: Sequence[float] = (1.0 / 3.0, 2.0 / 3.0),
    with_curves: bool = True,
) -> list[dict[str, Any]]:
    """逐時代表圖候選 track 的特徵表（只讀；不影響任何統計）。

    候選為 ``born_records`` 中壽命不小於 ``min_lifetime_hours`` 者。每筆含：壽命、觀測數、平均／最強核心 s1、起始核心位置與所在格是否為
    熱點、起始核心到最近熱點格點的距離、核心路徑到分析區矩形外緣的距離、起訖位移與路徑外接框、平均距岸距離，
    以及（``with_curves``）起始與後續時刻（``later_fractions``）已存曲線的長度與外接框，用來判斷曲線是否清楚可見。
    """

    from .period import _nearest_indices

    distance_m = _distance_to_mask_m(hotspot_mask, x_m_axis, y_m_axis)
    cache: dict[str, dict[str, np.ndarray]] = {}
    rows_out: list[dict[str, Any]] = []
    for item in born_records:
        if float(item["lifetime_hours"]) < float(min_lifetime_hours) - 1.0e-9:
            continue
        start, stop = int(item["start"]), int(item["stop"])
        core_x = np.asarray(observations["core_x_m"][start:stop], dtype=np.float64)
        core_y = np.asarray(observations["core_y_m"][start:stop], dtype=np.float64)
        s1_values = np.asarray(observations["core_s1_sinv"][start:stop], dtype=np.float64)
        col0 = int(_nearest_indices(x_m_axis, core_x[:1])[0])
        row0 = int(_nearest_indices(y_m_axis, core_y[:1])[0])
        times = np.asarray(observations["time_utc_ns"][start:stop], dtype=np.int64)
        row: dict[str, Any] = {
            "track_id": int(item["track_id"]),
            "lifetime_hours": float(item["lifetime_hours"]),
            "n_obs": int(item["n_obs"]),
            "mean_core_s1_sinv": float(item["mean_core_s1_sinv"]),
            "min_core_s1_sinv": float(np.min(s1_values)),
            "start_time_utc_ns": int(times[0]),
            "start_core_m": [float(core_x[0]), float(core_y[0])],
            "start_in_hotspot": bool(hotspot_mask[row0, col0]),
            "start_dist_to_hotspot_km": float(distance_m[row0, col0]) / 1000.0,
            "edge_distance_km": track_edge_distance_m(core_x, core_y, x_m_axis, y_m_axis) / 1000.0,
            "displacement_km": float(np.hypot(core_x[-1] - core_x[0], core_y[-1] - core_y[0])) / 1000.0,
            "path_extent_km": [float(np.ptp(core_x)) / 1000.0, float(np.ptp(core_y)) / 1000.0],
            "max_offset_from_start_km": float(np.max(np.hypot(core_x - core_x[0], core_y - core_y[0]))) / 1000.0,
            "mean_coast_m": item.get("mean_coast_m"),
            "coastal": bool(item.get("coastal")),
            "step_hours_regular": bool(np.all(np.diff(times) % 3_600_000_000_000 == 0)) if times.size > 1 else True,
        }
        if with_curves and detect_root is not None:
            indices = [0, *pick_later_observations(times, later_fractions)]
            curves = []
            bbox_points = [np.column_stack([core_x, core_y])]
            for index in indices:
                month = f"{int(observations['month'][start + index]):06d}"
                stored = _stored_detection(
                    detect_root, domain_id, month, int(observations["snapshot_index"][start + index]),
                    int(observations["detection_index"][start + index]), cache,
                )
                curve = stored["curve_m"]
                extent = [float(np.ptp(curve[:, 0])) / 1000.0, float(np.ptp(curve[:, 1])) / 1000.0] if curve.shape[0] else [0.0, 0.0]
                if curve.shape[0]:
                    bbox_points.append(np.asarray(curve, dtype=np.float64).reshape(-1, 2))
                curves.append(
                    {
                        "obs_index": int(index),
                        "dt_hours": float(times[index] - times[0]) / 3.6e12,
                        "length_km": _polyline_length_m(curve) / 1000.0,
                        "n_points": int(curve.shape[0]),
                        "extent_km": extent,
                    }
                )
            row["curves"] = curves
            stacked = np.vstack(bbox_points)
            row["track_bbox_m"] = [
                float(stacked[:, 0].min()), float(stacked[:, 1].min()), float(stacked[:, 0].max()), float(stacked[:, 1].max())
            ]
        rows_out.append(row)
    return rows_out


HOURLY_SELECTION_DEFAULTS: dict[str, Any] = {
    "tiers_hours": [24.0, 12.0],
    "hotspot_max_distance_km": 3.0,
    "min_curve_length_km": [6.0],
    "max_offset_km": 12.0,
    "min_edge_km": [5.0],
    "require_window": False,
    "max_mean_coast_km": None,
}
"""逐時代表圖的選取參數（可由選項檔 ``example_zoom`` 覆寫）。"""


def select_hourly_track(
    features: Sequence[Mapping[str, Any]],
    *,
    n_born_uncensored: int,
    tiers_hours: Sequence[float] = HOURLY_SELECTION_DEFAULTS["tiers_hours"],
    hotspot_max_distance_km: float = HOURLY_SELECTION_DEFAULTS["hotspot_max_distance_km"],
    min_curve_length_km: Sequence[float] = HOURLY_SELECTION_DEFAULTS["min_curve_length_km"],
    max_offset_km: float = HOURLY_SELECTION_DEFAULTS["max_offset_km"],
    min_edge_km: Sequence[float] = HOURLY_SELECTION_DEFAULTS["min_edge_km"],
    require_window: bool = HOURLY_SELECTION_DEFAULTS["require_window"],
    max_mean_coast_km: float | None = HOURLY_SELECTION_DEFAULTS["max_mean_coast_km"],
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    """逐時代表圖的選取規則（只影響呈現，不影響任何統計）。

    候選為站內出生的未截尾 track（``born_records`` 本身即強核心結構）。條件：
    (1) 起始核心所在格是站內曲線通過頻率前 10% 熱點（圖面所示、濾除小連通塊後的熱點），或距最近熱點格點中心不超過
    ``hotspot_max_distance_km``（「緊鄰」）；(2) 整條核心路徑距分析區外緣不小於 ``min_edge_km``，且核心離起點最遠不超過
    ``max_offset_km``（路徑與曲線放得進視窗）；(3) 起始與兩個後續時刻的曲線長度都不小於 ``min_curve_length_km``（曲線清楚可見）。
    壽命依 ``tiers_hours`` 由高到低分層：第一個有符合者的層（例如 ≥24 h，沒有則 ≥12 h）內挑選，順序為
    起始核心在熱點內者優先、再取三個時刻中最短曲線最長者、再取平均核心 s1 最強（最負）、最後取較小 ``track_id``。
    ``min_curve_length_km`` 與 ``min_edge_km`` 是依序放寬的候選門檻（第一個產生候選者生效，``relaxed`` 記錄是否放寬）。
    ``require_window`` 為真時，候選須帶有可行的地圖視窗（``item["window"]``：含站點錨點與海岸線，見 :func:`find_example_window`）。
    ``max_mean_coast_km`` 不為 ``None`` 時，候選的平均距岸距離不得超過此值（要求靠近海岸的站使用）。
    壽命層優先於門檻放寬：先在最高壽命層內依序嘗試 ``min_curve_length_km`` × ``min_edge_km`` 的各組門檻（曲線門檻為外層），
    該層全部組合都沒有候選才降到下一個壽命層。
    """

    info: dict[str, Any] = {
        "rule": "hourly_born_uncensored_in_or_next_to_hotspot_visible_curves_tiered_lifetime_clearest_curve",
        "tiers_hours": [float(v) for v in tiers_hours],
        "hotspot_max_distance_km": float(hotspot_max_distance_km),
        "min_curve_length_km_options": [float(v) for v in min_curve_length_km],
        "max_offset_km": float(max_offset_km),
        "min_edge_km_options": [float(v) for v in min_edge_km],
        "require_window": bool(require_window),
        "max_mean_coast_km": None if max_mean_coast_km is None else float(max_mean_coast_km),
        "n_born_uncensored": int(n_born_uncensored),
        "n_candidates_min_lifetime": len(features),
        "pool_sizes": {},
    }

    def _min_curve(item: Mapping[str, Any]) -> float:
        return min(float(curve["length_km"]) for curve in item["curves"])

    pairs = [(curve, edge) for curve in min_curve_length_km for edge in min_edge_km]
    for tier in tiers_hours:
        for rank, (curve_threshold, edge_threshold) in enumerate(pairs):
            pool = [
                item
                for item in features
                if float(item["lifetime_hours"]) >= float(tier) - 1.0e-9
                and float(item["start_dist_to_hotspot_km"]) <= float(hotspot_max_distance_km) + 1.0e-9
                and float(item["edge_distance_km"]) >= float(edge_threshold)
                and float(item["max_offset_from_start_km"]) <= float(max_offset_km)
                and item.get("curves")
                and _min_curve(item) >= float(curve_threshold)
                and bool(item.get("step_hours_regular", True))
                and bool(item.get("start_in_site", True))
                and (not require_window or item.get("window") is not None)
                and (
                    max_mean_coast_km is None
                    or (item.get("mean_coast_m") is not None and float(item["mean_coast_m"]) <= float(max_mean_coast_km) * 1000.0)
                )
            ]
            info["pool_sizes"][f"curve>={curve_threshold:g}km,edge>={edge_threshold:g}km,life>={tier:g}h"] = len(pool)
            if not pool:
                continue
            chosen = min(
                pool,
                key=lambda item: (
                    not bool(item["start_in_hotspot"]),
                    -_min_curve(item),
                    float(item["mean_core_s1_sinv"]),
                    int(item["track_id"]),
                ),
            )
            info.update(
                {
                    "status": "ok",
                    "tier_used_hours": float(tier),
                    "min_curve_length_km_used": float(curve_threshold),
                    "edge_threshold_used_km": float(edge_threshold),
                    "min_edge_distance_km": float(chosen["edge_distance_km"]),
                    "relaxed": bool(rank > 0),
                    "n_eligible": len(pool),
                    "chosen_features": dict(chosen),
                }
            )
            return chosen, info
    info.update({"status": "no_track_meets_rule", "relaxed": True, "n_eligible": 0})
    return None, info


EXAMPLE_WINDOW_DEFAULTS: dict[str, Any] = {
    "lon_step_deg": 0.25,
    "lon_intervals": 3,
    "lat_step_deg": 0.15,
    "lat_intervals": 3,
    "fallback_specs": [[0.25, 3, 0.05, 9], [0.25, 4, 0.2, 3], [0.5, 2, 0.3, 2], [0.25, 2, 0.1, 3]],
    "max_aspect_error": 0.03,
    "axes_aspect": 4.5 / 2.98,
    "track_margin_km": 3.0,
    "anchor_margin_km": 3.0,
    "anchor_right_margin_km": 7.0,
    "min_coast_km": 12.0,
    "coast_cap_km": 40.0,
    "key_patch_fraction": [0.76, 0.0, 1.0, 0.16],
}
"""逐時代表圖的視窗規則（選項檔 ``window_lattice.standard_frame`` 為規格表名稱，例如 ``"zoom"`` 時，「入選」視窗再擴成標準圖框，見 :func:`standardize_example_window`；
各候選的視窗與選取規則不受影響）：視窗是刻度格線上的矩形（經度 0.25° 的 3 格＝0.75°、緯度 0.15° 的 3 格＝0.45°，約 75 × 50 km，
長寬比與標準軸框〔4.5 × 2.98 英吋〕相同、誤差 ≤ 3%），
兩軸兩端都是標示刻度；必須含站點錨點（四周與右側留白，標示站名）、足夠長的海岸線（含島嶼），且整條 TRAP 的核心路徑與
三個時刻的曲線（外加 ``track_margin_km``）都在窗內，並避開右下角 1 m/s 參考箭頭的保留區。"""


def coast_segments(coastline: CoastlineGeometry | None, bbox: Sequence[float]) -> np.ndarray:
    """``bbox = (min_lon, min_lat, max_lon, max_lat)`` 附近海岸線的線段表 ``(N, 4)``：``[lon0, lat0, lon1, lat1]``。"""

    if coastline is None:
        return np.zeros((0, 4))
    parts = []
    for ring in coastline.rings:
        if ring[:, 0].max() >= bbox[0] and ring[:, 0].min() <= bbox[2] and ring[:, 1].max() >= bbox[1] and ring[:, 1].min() <= bbox[3]:
            parts.append(np.hstack([ring[:-1], ring[1:]]))
    return np.vstack(parts) if parts else np.zeros((0, 4))


def coast_km_in_bbox(segments: np.ndarray, bbox: Sequence[float]) -> float:
    """視窗內（兩端點都在窗內）的海岸線折線長度（公里）。"""

    if segments.shape[0] == 0:
        return 0.0
    lon0, lat0, lon1, lat1 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
    inside = (
        (lon0 >= bbox[0]) & (lon0 <= bbox[2]) & (lat0 >= bbox[1]) & (lat0 <= bbox[3])
        & (lon1 >= bbox[0]) & (lon1 <= bbox[2]) & (lat1 >= bbox[1]) & (lat1 <= bbox[3])
    )
    cos_lat = max(float(np.cos(np.radians(0.5 * (bbox[1] + bbox[3])))), 0.05)
    return float(np.hypot((lon1 - lon0)[inside] * KM_PER_DEGREE * cos_lat, (lat1 - lat0)[inside] * KM_PER_DEGREE).sum())


def find_example_window(
    track_bbox_lonlat: Sequence[float],
    anchor_lonlat: Sequence[float],
    domain_bbox: Sequence[float],
    segments: np.ndarray,
    params: Mapping[str, Any] | None = None,
    *,
    coast_cache: dict[tuple[float, float], float] | None = None,
) -> dict[str, Any] | None:
    """逐時代表圖的視窗（見 :data:`EXAMPLE_WINDOW_DEFAULTS`）；沒有可行視窗時回傳 ``None``。

    ``track_bbox_lonlat``＝TRAP 核心路徑與三個時刻曲線的外接框 ``(min_lon, min_lat, max_lon, max_lat)``；
    ``domain_bbox`` 為分析網格範圍（視窗須完整在其內，圖面不出現無資料的空白邊）。可行視窗中取海岸線較長者（上限
    ``coast_cap_km``），再取視窗中心最接近 TRAP 外接框中心者。
    """

    opts = {**EXAMPLE_WINDOW_DEFAULTS, **(dict(params) if params else {})}
    specs = [(float(opts["lon_step_deg"]), int(opts["lon_intervals"]), float(opts["lat_step_deg"]), int(opts["lat_intervals"]))]
    specs += [(float(v[0]), int(v[1]), float(v[2]), int(v[3])) for v in opts.get("fallback_specs", [])]
    for spec in specs:
        found = _find_example_window_spec(
            track_bbox_lonlat, anchor_lonlat, domain_bbox, segments, opts, spec, coast_cache=coast_cache if spec == specs[0] else None
        )
        if found is not None:
            return found
    return None


def standardize_example_window(found: Mapping[str, Any], segments: np.ndarray, frame_name: str) -> dict[str, Any]:
    """逐時代表圖視窗 → 標準圖框（2026-10-22）：視窗的選取（錨點、海岸線、軌跡與曲線放得進、保留區、長寬比）與 track 的選取仍照
    :func:`find_example_window` 與 :func:`select_hourly_track`；選定 track 之後（只對入選者）再以 :func:`traps.concise_style.standard_frame`
    往外擴到規格表（``FRAME_SPECS[frame_name]``，放大圖 0.8° × 0.5°）的跨度，四邊落在 0.1° 的整數倍上並完整包住原視窗；
    因此不影響哪個 track 被選上（案例不變），只改顯示範圍。原視窗另存為 ``lattice_window``；海岸線長度與長寬比誤差依新視窗重算
    （長寬比誤差相對於放大圖的主圖框 :data:`traps.figure_standard.CONCISE_ZOOM_AXES`，24.1° 參考緯度長寬比）。
    原視窗比規格跨度大時（備援規格的視窗）``ValueError``。"""

    from . import concise_style

    xlim, ylim, frame_info = concise_style.standard_frame(frame_name, found["xlim"], found["ylim"])
    spec = concise_style.FRAME_SPECS[frame_name]
    bbox = (xlim[0], ylim[0], xlim[1], ylim[1])
    # 標準圖框的長寬比誤差相對於放大圖的 LBT 規格主圖框（4.352 × 2.98 英吋；視窗的選取仍用 EXAMPLE_WINDOW_DEFAULTS["axes_aspect"]，案例不受影響）
    aspect = concise_style.frame_aspect_error(xlim, ylim, std.CONCISE_ZOOM_AXES.w, std.CONCISE_ZOOM_AXES.h)
    result = dict(found)
    result.update(
        {
            "xlim": [float(xlim[0]), float(xlim[1])],
            "ylim": [float(ylim[0]), float(ylim[1])],
            "coast_km_in_window": coast_km_in_bbox(segments, bbox),
            "lon_step_deg": float(spec["lon_step"]),
            "lat_step_deg": float(spec["lat_step"]),
            "lon_intervals": int(round(spec["lon_span"] / spec["lon_step"])),
            "lat_intervals": int(round(spec["lat_span"] / spec["lat_step"])),
            "aspect_error": aspect,
            "standard_frame": frame_name,
            "frame": frame_info,
            "lattice_window": {"xlim": [float(v) for v in found["xlim"]], "ylim": [float(v) for v in found["ylim"]], "coast_km_in_window": float(found["coast_km_in_window"]),
                               "aspect_error": float(found["aspect_error"])},
        }
    )
    return result


def _find_example_window_spec(
    track_bbox_lonlat: Sequence[float],
    anchor_lonlat: Sequence[float],
    domain_bbox: Sequence[float],
    segments: np.ndarray,
    opts: Mapping[str, Any],
    spec: tuple[float, int, float, int],
    *,
    coast_cache: dict[tuple[float, float], float] | None = None,
) -> dict[str, Any] | None:
    """:func:`find_example_window` 的單一格線規格（經度步長、格數、緯度步長、格數）；視窗長寬比須接近軸框（``max_aspect_error``）。"""

    lon_step, lon_n, lat_step, lat_n = spec
    span_lon, span_lat = lon_step * lon_n, lat_step * lat_n
    t_lon0, t_lat0, t_lon1, t_lat1 = (float(v) for v in track_bbox_lonlat)
    a_lon, a_lat = float(anchor_lonlat[0]), float(anchor_lonlat[1])
    centre_lat = 0.5 * (t_lat0 + t_lat1)
    km_lon = KM_PER_DEGREE * max(float(np.cos(np.radians(centre_lat))), 0.05)
    track_margin_lon, track_margin_lat = float(opts["track_margin_km"]) / km_lon, float(opts["track_margin_km"]) / KM_PER_DEGREE
    a_margin_lon, a_margin_lat = float(opts["anchor_margin_km"]) / km_lon, float(opts["anchor_margin_km"]) / KM_PER_DEGREE
    a_right_lon = float(opts["anchor_right_margin_km"]) / km_lon
    key_fraction = [float(v) for v in opts["key_patch_fraction"]]
    first_x = math.floor((min(t_lon0 - track_margin_lon, a_lon - a_margin_lon) - span_lon) / lon_step)
    last_x = math.ceil(max(t_lon1 + track_margin_lon, a_lon + a_margin_lon) / lon_step)
    first_y = math.floor((min(t_lat0 - track_margin_lat, a_lat - a_margin_lat) - span_lat) / lat_step)
    last_y = math.ceil(max(t_lat1 + track_margin_lat, a_lat + a_margin_lat) / lat_step)
    best: tuple[float, float, dict[str, Any]] | None = None
    for ix in range(first_x, last_x + 1):
        x0 = round(ix * lon_step, 6)
        x1 = round(x0 + span_lon, 6)
        if x0 < domain_bbox[0] - 1.0e-9 or x1 > domain_bbox[2] + 1.0e-9:
            continue
        if not (t_lon0 - track_margin_lon >= x0 and t_lon1 + track_margin_lon <= x1 and a_lon - a_margin_lon >= x0 and a_lon + a_right_lon <= x1):
            continue
        for iy in range(first_y, last_y + 1):
            y0 = round(iy * lat_step, 6)
            y1 = round(y0 + span_lat, 6)
            if y0 < domain_bbox[1] - 1.0e-9 or y1 > domain_bbox[3] + 1.0e-9:
                continue
            if not (t_lat0 - track_margin_lat >= y0 and t_lat1 + track_margin_lat <= y1 and a_lat - a_margin_lat >= y0 and a_lat + a_margin_lat <= y1):
                continue
            # 右下角參考箭頭保留區（以視窗比例）：TRAP 外接框與錨點都不在其中。
            def in_key(lon_v: float, lat_v: float) -> bool:
                fx, fy = (lon_v - x0) / span_lon, (lat_v - y0) / span_lat
                return key_fraction[0] <= fx <= key_fraction[2] and key_fraction[1] <= fy <= key_fraction[3]

            if any(in_key(px, py) for px in (t_lon0, t_lon1) for py in (t_lat0, t_lat1)) or in_key(a_lon, a_lat):
                continue
            if coast_cache is not None and (x0, y0) in coast_cache:
                coast = coast_cache[(x0, y0)]
            else:
                coast = coast_km_in_bbox(segments, (x0, y0, x1, y1))
                if coast_cache is not None:
                    coast_cache[(x0, y0)] = coast
            if coast < float(opts["min_coast_km"]):
                continue
            window_aspect = (span_lon * KM_PER_DEGREE * math.cos(math.radians(0.5 * (y0 + y1)))) / (span_lat * KM_PER_DEGREE)
            window_aspect_error = window_aspect / float(opts["axes_aspect"]) - 1.0
            if abs(window_aspect_error) > float(opts["max_aspect_error"]):
                continue
            centre_distance = math.hypot(
                (0.5 * (t_lon0 + t_lon1) - 0.5 * (x0 + x1)) * km_lon, (0.5 * (t_lat0 + t_lat1) - 0.5 * (y0 + y1)) * KM_PER_DEGREE
            )
            score = (-min(coast, float(opts["coast_cap_km"])), centre_distance)
            if best is None or score < (best[0], best[1]):
                best = (
                    score[0], score[1],
                    {
                        "xlim": [x0, x1], "ylim": [y0, y1], "coast_km_in_window": coast, "anchor_in_window": True,
                        "anchor_lonlat": [a_lon, a_lat], "track_to_window_centre_km": centre_distance,
                        "lon_step_deg": lon_step, "lat_step_deg": lat_step, "lon_intervals": lon_n, "lat_intervals": lat_n,
                        "aspect_error": window_aspect_error,
                    },
                )
    return None if best is None else best[2]


SITE_LABEL_DIRECTIONS = (
    ("W", -1, 0), ("E", 1, 0), ("N", 0, 1), ("S", 0, -1), ("NW", -1, 1), ("SW", -1, -1), ("NE", 1, 1), ("SE", 1, -1),
)
"""站名相對錨點的八個方位（dx、dy 為 -1／0／1）。"""

SITE_LABEL_GAP_PT = 6.0
SITE_LABEL_CLEARANCE_INCH = 0.04
SITE_LABEL_HALO_INCH = 0.03


def _is_blank_cell(field: np.ndarray, lon_axis: np.ndarray, lat_axis: np.ndarray, lon_v: float, lat_v: float) -> bool:
    """最近格點是否沒有場值（陸地或分析區外，圖面為白底）。"""

    row = int(np.abs(np.asarray(lat_axis) - lat_v).argmin())
    col = int(np.abs(np.asarray(lon_axis) - lon_v).argmin())
    return not bool(np.isfinite(field[row, col]))


def _rect_distance(points: np.ndarray, rect: Sequence[float]) -> float:
    """點群到矩形 ``(x0, y0, x1, y1)`` 的最短距離（點在矩形內為 0）；沒有點時回傳無限大。"""

    if points.shape[0] == 0:
        return float("inf")
    dx = np.maximum.reduce([rect[0] - points[:, 0], np.zeros(points.shape[0]), points[:, 0] - rect[2]])
    dy = np.maximum.reduce([rect[1] - points[:, 1], np.zeros(points.shape[0]), points[:, 1] - rect[3]])
    return float(np.hypot(dx, dy).min())


def _rects_overlap(first: Sequence[float], second: Sequence[float], pad: float = 0.0) -> bool:
    return not (first[2] + pad < second[0] or second[2] + pad < first[0] or first[3] + pad < second[1] or second[3] + pad < first[1])


def _place_site_label(
    figure: Any,
    axis: Any,
    text: str,
    anchor: Sequence[float],
    limits: tuple[Sequence[float], Sequence[float]],
    *,
    polylines: Sequence[tuple[np.ndarray, np.ndarray]],
    points: Sequence[tuple[float, float]],
    extra_artists: Sequence[Any],
    land_probe: Any,
    key_patch: Sequence[float] | None = None,
) -> dict[str, Any]:
    """站名放在錨點旁：八個方位中，標示框（含白色描邊）完整在軸內、與 TRAP 曲線／核心路徑／核心點／日數標籤的距離
    ≥ ``SITE_LABEL_CLEARANCE_INCH``、不碰 1 m/s 參考箭頭保留區者才有效；有效者取「落在陸地（無場值）側的比例高、離軌跡遠」者。
    以實際文字外框（渲染後量測）判斷；沒有有效方位時取離軌跡最遠者並回報 ``valid=False``。"""

    import matplotlib.patheffects as path_effects

    key_patch = ZOOM_KEY_PATCH_INCH if key_patch is None else key_patch  # 預設 MAP1 軸框的保留區；簡要報告傳入較窄軸框的保留區
    axis.apply_aspect()
    renderer = figure.canvas.get_renderer()
    dpi = float(figure.dpi)
    transform = axis.transData
    inverse = transform.inverted()

    def to_inch(lon_v: Any, lat_v: Any) -> np.ndarray:
        return transform.transform(np.column_stack([np.atleast_1d(lon_v), np.atleast_1d(lat_v)]).astype(float)) / dpi

    samples = []
    for lon_v, lat_v in polylines:
        xy = to_inch(lon_v, lat_v)
        samples.append(xy)
        for first, second in zip(xy[:-1], xy[1:], strict=True):
            count = max(int(np.hypot(*(second - first)) / 0.01), 1)
            samples.append(first + (second - first) * (np.arange(1, count) / count)[:, None])
    if points:
        samples.append(to_inch([p[0] for p in points], [p[1] for p in points]))
    obstacles = np.vstack(samples) if samples else np.zeros((0, 2))
    box = axis.get_window_extent(renderer)
    axes_rect = (box.x0 / dpi, box.y0 / dpi, box.x1 / dpi, box.y1 / dpi)
    key_rect = (
        axes_rect[0] + key_patch[0], axes_rect[1] + key_patch[1],
        axes_rect[0] + key_patch[2], axes_rect[1] + key_patch[3],
    )
    extra_rects = []
    for artist in extra_artists:
        ext = artist.get_window_extent(renderer)
        extra_rects.append((ext.x0 / dpi, ext.y0 / dpi, ext.x1 / dpi, ext.y1 / dpi))
    halo = [path_effects.withStroke(linewidth=2.4, foreground="white")]
    results = []
    for name, dx, dy in SITE_LABEL_DIRECTIONS:
        annotation = axis.annotate(
            text, (anchor[0], anchor[1]), xytext=(dx * SITE_LABEL_GAP_PT, dy * SITE_LABEL_GAP_PT), textcoords="offset points",
            ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
            fontsize=8, fontweight="bold", color="#1b2a30", zorder=9, path_effects=halo,
        )
        ext = annotation.get_window_extent(renderer)
        rect = (ext.x0 / dpi - SITE_LABEL_HALO_INCH, ext.y0 / dpi - SITE_LABEL_HALO_INCH, ext.x1 / dpi + SITE_LABEL_HALO_INCH, ext.y1 / dpi + SITE_LABEL_HALO_INCH)
        annotation.remove()
        inside = rect[0] >= axes_rect[0] + 0.02 and rect[1] >= axes_rect[1] + 0.02 and rect[2] <= axes_rect[2] - 0.02 and rect[3] <= axes_rect[3] - 0.02
        clearance = _rect_distance(obstacles, rect)
        hits_key = _rects_overlap(rect, key_rect, 0.03)
        hits_labels = any(_rects_overlap(rect, other, 0.02) for other in extra_rects)
        probes = [(rect[0] + (rect[2] - rect[0]) * fx, rect[1] + (rect[3] - rect[1]) * fy) for fx in np.linspace(0.1, 0.9, 5) for fy in (0.2, 0.5, 0.8)]
        lonlat = inverse.transform(np.asarray(probes) * dpi)
        land = float(np.mean([bool(land_probe(float(lon_v), float(lat_v))) for lon_v, lat_v in lonlat]))
        valid = bool(inside and not hits_key and not hits_labels and clearance >= SITE_LABEL_CLEARANCE_INCH)
        results.append({
            "direction": name, "dx": dx, "dy": dy, "valid": valid, "inside_axes": bool(inside), "overlaps_key": bool(hits_key),
            "overlaps_later_labels": bool(hits_labels), "clearance_to_track_inch": round(clearance, 4), "land_fraction": round(land, 3),
            "bbox_axes_inch": [round(rect[0] - axes_rect[0], 3), round(rect[1] - axes_rect[1], 3), round(rect[2] - axes_rect[0], 3), round(rect[3] - axes_rect[1], 3)],
        })
    pool = [item for item in results if item["valid"]] or [item for item in results if item["inside_axes"] and not item["overlaps_key"]] or results
    chosen = max(pool, key=lambda item: (2.0 * item["land_fraction"] + min(item["clearance_to_track_inch"], 0.3) / 0.3, -abs(item["dx"]) - abs(item["dy"])))
    axis.annotate(
        text, (anchor[0], anchor[1]), xytext=(chosen["dx"] * SITE_LABEL_GAP_PT, chosen["dy"] * SITE_LABEL_GAP_PT), textcoords="offset points",
        ha="left" if chosen["dx"] > 0 else ("right" if chosen["dx"] < 0 else "center"),
        va="bottom" if chosen["dy"] > 0 else ("top" if chosen["dy"] < 0 else "center"),
        fontsize=8, fontweight="bold", color="#1b2a30", zorder=9, path_effects=halo,
    )
    return {
        **{key: chosen[key] for key in ("direction", "valid", "inside_axes", "overlaps_key", "overlaps_later_labels", "clearance_to_track_inch", "land_fraction", "bbox_axes_inch")},
        "min_clearance_required_inch": SITE_LABEL_CLEARANCE_INCH, "n_valid_directions": sum(1 for item in results if item["valid"]),
        "candidates": [{key: item[key] for key in ("direction", "valid", "land_fraction", "clearance_to_track_inch")} for item in results],
    }


def zoom_window(
    data: SiteFigureData,
    snap: ExampleSnapshot,
    *,
    window_km: Sequence[float],
    margin_km: float = ZOOM_MARGIN_KM,
    fixed_limits: tuple[Sequence[float], Sequence[float]] | None = None,
) -> dict[str, Any]:
    """縮放視窗（經緯度範圍）：大小固定的實際公里數，以起始核心為中心。

    起始曲線、後續曲線與核心的外接框若不能完整放進以起始核心為中心的視窗（各邊保留 ``margin_km``），
    改以外接框中心為中心；最後把視窗平移到分析區矩形之內（視窗小於分析區時），避免出現空白邊界。
    """

    mapper = MetricToLonLat(snap.x_m_axis, snap.y_m_axis, data.lon, data.lat)
    width_km, height_km = float(window_km[0]), float(window_km[1])
    points_m = [np.asarray(snap.chosen_curve_m, dtype=np.float64), np.asarray([snap.chosen_core_m], dtype=np.float64)]
    for later_obs in snap.later:
        points_m.append(np.asarray(later_obs.curve_m, dtype=np.float64))
        points_m.append(np.asarray([later_obs.core_m], dtype=np.float64))
    stacked = np.vstack([block.reshape(-1, 2) for block in points_m])
    lon_p, lat_p = mapper(stacked[:, 0], stacked[:, 1])
    core_lon, core_lat = (float(value[0]) for value in mapper([snap.chosen_core_m[0]], [snap.chosen_core_m[1]]))
    domain_lon = (float(np.min(data.lon)), float(np.max(data.lon)))
    domain_lat = (float(np.min(data.lat)), float(np.max(data.lat)))
    if fixed_limits is not None:
        # 視窗已由選取規則決定（刻度格線上的矩形，見 find_example_window）：只回報實際大小與位置檢查。
        fx, fy = (float(fixed_limits[0][0]), float(fixed_limits[0][1])), (float(fixed_limits[1][0]), float(fixed_limits[1][1]))
        centre_lon_f, centre_lat_f = 0.5 * (fx[0] + fx[1]), 0.5 * (fy[0] + fy[1])
        km_lon_f = KM_PER_DEGREE * max(float(np.cos(np.radians(centre_lat_f))), 0.05)
        return {
            "xlim": fx,
            "ylim": fy,
            "centre_lonlat": (centre_lon_f, centre_lat_f),
            "centred_on": "tick_lattice",
            "shift_to_keep_inside_domain_km": 0.0,
            "positions_inside_window": bool(lon_p.min() >= fx[0] and lon_p.max() <= fx[1] and lat_p.min() >= fy[0] and lat_p.max() <= fy[1]),
            "window_km": [(fx[1] - fx[0]) * km_lon_f, (fy[1] - fy[0]) * KM_PER_DEGREE],
            "start_core_lonlat": (core_lon, core_lat),
        }

    def spans(centre_lat: float) -> tuple[float, float]:
        return width_km / (KM_PER_DEGREE * max(float(np.cos(np.radians(centre_lat))), 0.05)), height_km / KM_PER_DEGREE

    def inside(centre_lon: float, centre_lat: float) -> bool:
        span_lon, span_lat = spans(centre_lat)
        margin_lat = margin_km / KM_PER_DEGREE
        margin_lon = margin_km / (KM_PER_DEGREE * max(float(np.cos(np.radians(centre_lat))), 0.05))
        return bool(
            lon_p.min() >= centre_lon - 0.5 * span_lon + margin_lon
            and lon_p.max() <= centre_lon + 0.5 * span_lon - margin_lon
            and lat_p.min() >= centre_lat - 0.5 * span_lat + margin_lat
            and lat_p.max() <= centre_lat + 0.5 * span_lat - margin_lat
        )

    centre_lon, centre_lat, centred_on = core_lon, core_lat, "core"
    if not inside(centre_lon, centre_lat):
        centre_lon, centre_lat, centred_on = 0.5 * float(lon_p.min() + lon_p.max()), 0.5 * float(lat_p.min() + lat_p.max()), "bbox"
    centre_lat_unclamped, centre_lon_unclamped = centre_lat, centre_lon
    for _ in range(2):
        span_lon, span_lat = spans(centre_lat)
        if span_lon <= domain_lon[1] - domain_lon[0]:
            centre_lon = min(max(centre_lon, domain_lon[0] + 0.5 * span_lon), domain_lon[1] - 0.5 * span_lon)
        if span_lat <= domain_lat[1] - domain_lat[0]:
            centre_lat = min(max(centre_lat, domain_lat[0] + 0.5 * span_lat), domain_lat[1] - 0.5 * span_lat)
    span_lon, span_lat = spans(centre_lat)
    shift_km = float(
        np.hypot(
            (centre_lon - centre_lon_unclamped) * KM_PER_DEGREE * np.cos(np.radians(centre_lat)),
            (centre_lat - centre_lat_unclamped) * KM_PER_DEGREE,
        )
    )
    xlim = (centre_lon - 0.5 * span_lon, centre_lon + 0.5 * span_lon)
    ylim = (centre_lat - 0.5 * span_lat, centre_lat + 0.5 * span_lat)
    bbox_inside = bool(lon_p.min() >= xlim[0] and lon_p.max() <= xlim[1] and lat_p.min() >= ylim[0] and lat_p.max() <= ylim[1])
    return {
        "xlim": xlim,
        "ylim": ylim,
        "centre_lonlat": (centre_lon, centre_lat),
        "centred_on": centred_on,
        "shift_to_keep_inside_domain_km": shift_km,
        "positions_inside_window": bbox_inside,
        "window_km": [width_km, height_km],
        "start_core_lonlat": (core_lon, core_lat),
    }


ZOOM_KEY_ANCHOR_INCH = (4.08, 0.16)
"""1 m/s 參考箭頭的箭頭尖端位置（以地圖軸左下角為原點的英吋；QuiverKey 的 ``labelpos="E"`` 以尖端定位，箭頭向左延伸、文字在右側）。"""

ZOOM_KEY_PATCH_INCH = (3.52, 0.0, 4.50, 0.34)
"""參考箭頭的固定保留區 ``(x0, y0, x1, y1)``（地圖軸英吋，右下角；五站相同位置與大小）：任何會與此區重疊的流速箭頭都不畫。
這是 MAP1 軸框（寬 4.50 英吋）的值；簡要報告改用較窄的軸框 :data:`traps.figure_standard.CONCISE_ZOOM_AXES` 時，
保留區與箭頭尖端對齊右緣（距右緣的英吋數不變），見 :func:`zoom_key_geometry`。"""

_MAP1_ZOOM_WIDTH_INCH = 4.50
"""``ZOOM_KEY_*`` 常數對應的軸框寬度（英吋）。"""


def zoom_boxes() -> tuple[std.Box, std.Box]:
    """放大圖的地圖軸框與色條框：簡要報告（concise）用 LBT 規格的 :data:`traps.figure_standard.CONCISE_ZOOM_AXES`／``CONCISE_ZOOM_CBAR``
    （左緣、下緣、框高、色條寬與間距同大圖；寬度依 24.1° 參考緯度長寬比），其餘用 MAP1。"""

    if concise_style.concise_enabled():
        return std.CONCISE_ZOOM_AXES, std.CONCISE_ZOOM_CBAR
    return std.MAP1_AXES, std.MAP1_CBAR


def zoom_key_geometry(axes_width: float) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    """寬 ``axes_width`` 英吋的軸框內，1 m/s 參考箭頭的尖端位置與保留區：與右緣的距離同 MAP1 軸框（寬 4.50 英吋）的 :data:`ZOOM_KEY_ANCHOR_INCH`、:data:`ZOOM_KEY_PATCH_INCH`。"""

    shift = float(axes_width) - _MAP1_ZOOM_WIDTH_INCH
    anchor = (ZOOM_KEY_ANCHOR_INCH[0] + shift, ZOOM_KEY_ANCHOR_INCH[1])
    patch = (ZOOM_KEY_PATCH_INCH[0] + shift, ZOOM_KEY_PATCH_INCH[1], ZOOM_KEY_PATCH_INCH[2] + shift, ZOOM_KEY_PATCH_INCH[3])
    return (round(anchor[0], 6), anchor[1]), tuple(round(v, 6) for v in patch)  # type: ignore[return-value]


ZOOM_ARROW_PAD_INCH = 0.05
"""判斷箭頭是否與保留區重疊時，箭頭線段外框向外加的寬度（涵蓋箭頭頭部寬度）。"""

ZOOM_KEY_HALO_PT = 2.0
"""參考箭頭文字外框的餘裕（pt）：v47 起圖面已無白色描邊，此值只用於量測外框（保留區判定與版面紀錄維持不變），不再對應任何繪出的描邊。"""


def _arrows_hit_rect(
    tail_x: np.ndarray, tail_y: np.ndarray, delta_x: np.ndarray, delta_y: np.ndarray, rect: Sequence[float], pad: float
) -> np.ndarray:
    """每支箭頭（尾端到箭頭尖端的線段，外框加 ``pad``）是否與矩形 ``rect = (x0, y0, x1, y1)`` 相交（單位皆為英吋）。"""

    x_low = np.minimum(tail_x, tail_x + delta_x) - pad
    x_high = np.maximum(tail_x, tail_x + delta_x) + pad
    y_low = np.minimum(tail_y, tail_y + delta_y) - pad
    y_high = np.maximum(tail_y, tail_y + delta_y) + pad
    return (x_high >= rect[0]) & (x_low <= rect[2]) & (y_high >= rect[1]) & (y_low <= rect[3])


def _arrows_clearance_to_box(
    tail_x: np.ndarray, tail_y: np.ndarray, delta_x: np.ndarray, delta_y: np.ndarray, box: Sequence[float]
) -> list[float]:
    """每支箭頭（線段外框，不加寬）到矩形 ``box`` 的最短間距（英吋）；0 以下代表外框相交。"""

    x_low, x_high = np.minimum(tail_x, tail_x + delta_x), np.maximum(tail_x, tail_x + delta_x)
    y_low, y_high = np.minimum(tail_y, tail_y + delta_y), np.maximum(tail_y, tail_y + delta_y)
    gap_x = np.maximum(box[0] - x_high, x_low - box[2])
    gap_y = np.maximum(box[1] - y_high, y_low - box[3])
    clearance = np.where((gap_x <= 0.0) & (gap_y <= 0.0), -1.0, np.hypot(np.maximum(gap_x, 0.0), np.maximum(gap_y, 0.0)))
    return [float(value) for value in clearance]


def _vector_inch_geometry(
    lon_pts: np.ndarray, lat_pts: np.ndarray, u: np.ndarray, v: np.ndarray, xlim: Sequence[float], ylim: Sequence[float], inch_per_m_s: float,
    box: std.Box | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """流速箭頭的尾端位置與位移，換成地圖軸英吋（``angles="uv"``：圖面方向就是 (u, v) 方向，長度 = 速度 × 比例）；``box`` 為地圖軸框（預設 MAP1）。"""

    box = std.MAP1_AXES if box is None else box
    tail_x = (np.asarray(lon_pts, dtype=np.float64) - xlim[0]) / (xlim[1] - xlim[0]) * box.w
    tail_y = (np.asarray(lat_pts, dtype=np.float64) - ylim[0]) / (ylim[1] - ylim[0]) * box.h
    return tail_x, tail_y, np.asarray(u, dtype=np.float64) * inch_per_m_s, np.asarray(v, dtype=np.float64) * inch_per_m_s


def _measure_reference_key(figure: Any, axis: Any, key: Any, length_inch: float, box_size: Sequence[float] | None = None) -> dict[str, Any]:
    """參考箭頭（箭頭加文字）在地圖軸內的外框，單位為以左下角為原點的軸英吋；須先 ``canvas.draw()`` 讓文字定位。
    ``box_size`` 為地圖軸框 ``(寬, 高)`` 英吋（預設 MAP1）；箭頭尖端與保留區由 :func:`zoom_key_geometry` 依軸框寬度決定。"""

    axes_w, axes_h = (std.MAP1_AXES.w, std.MAP1_AXES.h) if box_size is None else (float(box_size[0]), float(box_size[1]))
    anchor_inch, patch = zoom_key_geometry(axes_w)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    text_box = key.text.get_window_extent(renderer)
    inverse = axis.transAxes.inverted()
    (fx0, fy0), (fx1, fy1) = inverse.transform((text_box.x0, text_box.y0)), inverse.transform((text_box.x1, text_box.y1))
    halo = ZOOM_KEY_HALO_PT / 72.0 / 2.0
    text_in = [fx0 * axes_w - halo, fy0 * axes_h - halo, fx1 * axes_w + halo, fy1 * axes_h + halo]
    tip_x, tip_y = anchor_inch
    arrow_in = [tip_x - length_inch - ZOOM_ARROW_PAD_INCH, tip_y - ZOOM_ARROW_PAD_INCH, tip_x + ZOOM_ARROW_PAD_INCH, tip_y + ZOOM_ARROW_PAD_INCH]
    box = [min(text_in[0], arrow_in[0]), min(text_in[1], arrow_in[1]), max(text_in[2], arrow_in[2]), max(text_in[3], arrow_in[3])]
    return {
        "bbox_axes_inch": [round(value, 4) for value in box],
        "patch_axes_inch": list(patch),
        "axes_size_inch": [axes_w, axes_h],
        "inside_axes": bool(box[0] >= 0.0 and box[1] >= 0.0 and box[2] <= axes_w and box[3] <= axes_h),
        "inside_patch": bool(box[0] >= patch[0] - 1.0e-9 and box[1] >= patch[1] - 1.0e-9 and box[2] <= patch[2] + 1.0e-9 and box[3] <= patch[3] + 1.0e-9),
        "reference_length_inch": length_inch,
    }


def _sample_vector_grid(
    lon_axis: np.ndarray, lat_axis: np.ndarray, xlim: Sequence[float], ylim: Sequence[float], n_cols: int, n_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """視窗內均勻的 ``n_cols × n_rows`` 箭頭位置（取最近的網格點）：回傳 (列索引, 欄索引) 一維陣列。"""

    xs = xlim[0] + (np.arange(n_cols) + 0.5) * (xlim[1] - xlim[0]) / n_cols
    ys = ylim[0] + (np.arange(n_rows) + 0.5) * (ylim[1] - ylim[0]) / n_rows
    cols = np.array([int(np.argmin(np.abs(lon_axis - value))) for value in xs])
    rows = np.array([int(np.argmin(np.abs(lat_axis - value))) for value in ys])
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    return grid_rows.ravel(), grid_cols.ravel()


def render_example_zoom(
    output_path: Path,
    data: SiteFigureData,
    snap: ExampleSnapshot,
    *,
    coastline: CoastlineGeometry | None,
    zoom: Mapping[str, Any],
    day_unit: bool = True,
    dpi: int = 150,
) -> Path:
    """縮放代表 TRAP 圖（簡要報告）：單一 6.3×4.4 英吋畫布，以代表 TRAP 為中心的固定公里視窗。

    內容：視窗內 s1（五站共用對稱色階）、抽稀的表層流速箭頭（全站共用箭頭長度比例，附參考箭頭）、
    代表 TRAP 起始曲線（紅線）與核心（紅點）、同一 TRAP 兩個後續時刻的曲線（鮭魚紅，透明度遞增，標示天數）
    與核心路徑；不畫其他 TRAP 曲線。v47 起圖面一律無白邊：曲線不加白色描邊、核心點無邊線、參考箭頭與文字不加白色描邊，圖例與圖上一致。
    """

    matplotlib, plt = _load_matplotlib()
    from matplotlib.lines import Line2D

    data = replace(data, site_name_zh=concise_style.display_site_name(data.site_id, data.site_name_zh))
    concise_on = concise_style.concise_enabled()
    axes_box, cbar_box = zoom_boxes()  # 簡要報告：LBT 規格的主圖框與色條框（同大圖的左緣、下緣、框高、色條寬與間距）；其餘 MAP1
    key_anchor, key_patch = zoom_key_geometry(axes_box.w)
    lattice = (snap.zoom or {}).get("window") if getattr(snap, "zoom", None) else None
    if lattice:
        window = zoom_window(data, snap, window_km=zoom["window_km"], fixed_limits=(lattice["xlim"], lattice["ylim"]))
    else:
        window = zoom_window(data, snap, window_km=zoom["window_km"])
        if concise_style.concise_enabled():
            window["xlim"], window["ylim"], _snap_info = concise_style.fit_limits_to_frame(window["xlim"], window["ylim"], axes_box.w, axes_box.h)
            window["xlim"], window["ylim"] = concise_style.centred_frame("zoom", window["xlim"], window["ylim"])  # 後備：規格表的標準放大圖框（中心對齊 0.1° 格線）
    xlim, ylim = window["xlim"], window["ylim"]
    view = std.limits_slices(data.lon, data.lat, xlim, ylim)
    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    figure = plt.figure(figsize=(std.WIDTH_INCH, std.SINGLE_HEIGHT_INCH))
    axis = std.add_box_axes(figure, axes_box)
    cax = std.add_box_axes(figure, cbar_box)

    mapper = MetricToLonLat(snap.x_m_axis, snap.y_m_axis, data.lon, data.lat)
    s1_full = np.where(snap.derivative_valid, snap.s1_sinv * SECONDS_PER_DAY, np.nan)
    s1_day = s1_full[rows, cols]
    # 視窗內「有場值／陸地／其餘空白」的面積比例（場值格＝有效的 s1 格；陸地由海岸線多邊形判定）。
    from .extent_coverage import build_coverage

    field_fractions = build_coverage(
        data.lon, data.lat, np.isfinite(s1_full), coastline, (xlim[0], xlim[1], ylim[0], ylim[1]), margin=0.0
    ).fractions(xlim, ylim)
    window_rows, window_cols = std.limits_slices(data.lon, data.lat, xlim, ylim, margin_cells=0)
    natural_limit = _symmetric_limit(s1_full[window_rows, window_cols])
    concise_style.record_limit("example_zoom", data.site_id, "s1_day", (-natural_limit, natural_limit))
    limit = natural_limit
    shared = concise_style.shared_limits("example_zoom", "s1_day")
    if shared is not None:
        limit = float(max(abs(shared[0]), abs(shared[1])))
    in_window = s1_full[window_rows, window_cols]
    s1_extend = "neither"  # 色條兩端平整；超出共用色階的值以端點色飽和（比例記錄於 clipped_fraction）
    mesh = axis.pcolormesh(lon, lat, np.ma.masked_invalid(s1_day), shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    colorbar = figure.colorbar(mesh, cax=cax, extend=s1_extend)
    style_colorbar(colorbar, S1_DAY_LABEL)
    concise_style.style_colorbar_shared(
        colorbar, -limit, limit, S1_DAY_LABEL, name="s1_day", extend=s1_extend, clipped=concise_style.clipped_fraction(in_window, -limit, limit)
    )

    # 抽稀流速箭頭：視窗內 15×10，長度比例（英吋 / (m/s)）五站相同。
    n_cols, n_rows = (int(value) for value in zoom["vector_grid"])
    sample_rows, sample_cols = _sample_vector_grid(data.lon, data.lat, xlim, ylim, n_cols, n_rows)
    usable = snap.valid[sample_rows, sample_cols] & np.isfinite(snap.u[sample_rows, sample_cols]) & np.isfinite(snap.v[sample_rows, sample_cols])
    sample_rows, sample_cols = sample_rows[usable], sample_cols[usable]
    inch_per_m_s = float(zoom["vector_inch_per_m_s"])
    reference = float(zoom["vector_reference_m_s"])
    # 參考箭頭放在地圖內右下角（固定保留區，五站相同）：會與保留區重疊的流速箭頭不畫，其餘取樣與長度比例不變。
    tail_x, tail_y, delta_x, delta_y = _vector_inch_geometry(
        data.lon[sample_cols], data.lat[sample_rows], snap.u[sample_rows, sample_cols], snap.v[sample_rows, sample_cols], xlim, ylim, inch_per_m_s, axes_box
    )
    hits_key = _arrows_hit_rect(tail_x, tail_y, delta_x, delta_y, key_patch, ZOOM_ARROW_PAD_INCH)
    n_suppressed = int(hits_key.sum())
    sample_rows, sample_cols = sample_rows[~hits_key], sample_cols[~hits_key]
    tail_x, tail_y, delta_x, delta_y = tail_x[~hits_key], tail_y[~hits_key], delta_x[~hits_key], delta_y[~hits_key]
    quiver = axis.quiver(
        data.lon[sample_cols], data.lat[sample_rows], snap.u[sample_rows, sample_cols], snap.v[sample_rows, sample_cols],
        color="0.12", alpha=0.85, angles="uv", scale_units="inches", scale=1.0 / inch_per_m_s, width=0.0032, zorder=3,
    )

    handles: list[Any] = []
    labels: list[str] = []
    later_points: list[tuple[float, float, str]] = []
    later_curves_lonlat: list[tuple[np.ndarray, np.ndarray]] = []
    for rank, later_obs in enumerate(snap.later):
        alpha = ZOOM_LEGEND_LATER_ALPHAS[min(rank, len(ZOOM_LEGEND_LATER_ALPHAS) - 1)]
        curve_lon, curve_lat = mapper(later_obs.curve_m[:, 0], later_obs.curve_m[:, 1])
        later_curves_lonlat.append((np.asarray(curve_lon, dtype=float), np.asarray(curve_lat, dtype=float)))
        axis.plot(curve_lon, curve_lat, color=ZOOM_LATER_COLOUR, linewidth=3.2, alpha=alpha, zorder=5.5, solid_capstyle="round")
        core_lon_l, core_lat_l = mapper([later_obs.core_m[0]], [later_obs.core_m[1]])
        axis.plot(core_lon_l, core_lat_l, marker="o", markersize=3.5, color=ZOOM_LATER_COLOUR, alpha=alpha, markeredgewidth=0, linestyle="none", zorder=6)
        day_text = relative_time_label(later_obs.dt_seconds, day_unit=day_unit)
        later_points.append((float(core_lon_l[0]), float(core_lat_l[0]), day_text))
        handles.append(Line2D([], [], color=ZOOM_LATER_COLOUR, linewidth=3.0, alpha=alpha))
        labels.append(later_curve_label(later_obs.dt_seconds, day_unit=day_unit))
    if snap.path_m.shape[0] >= 2:
        path_lon, path_lat = mapper(snap.path_m[:, 0], snap.path_m[:, 1])
        axis.plot(path_lon, path_lat, color="#7f1d1d", linewidth=1.0, linestyle=":", zorder=6.2)
    chosen_lon, chosen_lat = mapper(snap.chosen_curve_m[:, 0], snap.chosen_curve_m[:, 1])
    axis.plot(chosen_lon, chosen_lat, color="#c00000", linewidth=1.8, zorder=7)
    core_lon, core_lat = mapper([snap.chosen_core_m[0]], [snap.chosen_core_m[1]])
    axis.plot(core_lon, core_lat, marker="o", markersize=6.5, color="#c00000", markeredgewidth=0, linestyle="none", zorder=8)
    handles.insert(0, Line2D([], [], color="#c00000", linewidth=1.8, marker="o", markersize=6.5, markeredgewidth=0))      # v47: 與圖上核心同大（6.5 pt、無邊線）；原 5.5 pt 加預設 1 pt 同色邊線，圖例上看到的直徑同為 6.5 pt
    labels.insert(0, "起始時刻 TRAP 曲線與核心")
    if snap.path_m.shape[0] >= 2:
        handles.append(Line2D([], [], color="#7f1d1d", linewidth=1.0, linestyle=":"))
        labels.append("核心路徑")

    # 簡要報告：底圖樣式與大圖相同（LBT run09）：黑色海岸線 0.35 pt、不畫經緯度格線
    buffer_handle = _prepare_map_axis(axis, data, view, coastline, limits=(xlim, ylim), lbt_style=concise_on)
    anchor_lon, anchor_lat = data.anchor_lonlat
    anchor_in_window = bool(xlim[0] <= anchor_lon <= xlim[1] and ylim[0] <= anchor_lat <= ylim[1])
    if buffer_handle is not None:
        handles.append(buffer_handle)
        labels.append(coast_buffer_label(data))

    # 日數標籤：後續核心彼此太近（圖面距離小於 0.45 英吋）時只列在圖例，避免文字互相覆蓋。
    span_lon, span_lat = xlim[1] - xlim[0], ylim[1] - ylim[0]
    xy_in = np.asarray(
        [[(lon_v - xlim[0]) / span_lon * axes_box.w, (lat_v - ylim[0]) / span_lat * axes_box.h] for lon_v, lat_v, _ in later_points]
        + [[(float(core_lon[0]) - xlim[0]) / span_lon * axes_box.w, (float(core_lat[0]) - ylim[0]) / span_lat * axes_box.h]]
    )
    separated = all(
        float(np.hypot(*(xy_in[first] - xy_in[second]))) >= 0.45 for first in range(1, len(xy_in)) for second in range(first)
    )
    later_annotations: list[Any] = []
    if separated:
        for lon_v, lat_v, day_text in later_points:
            later_annotations.append(
                axis.annotate(
                    day_text, xy=(lon_v, lat_v), xytext=(4, 4), textcoords="offset points", fontsize=8, color="#5f1010", zorder=9,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
            )

    # 站名標示：站點錨點（星號）旁、盡量在陸地側、與 TRAP 曲線／核心／路徑／日數標籤／1 m/s 參考箭頭保留區都不重疊。
    site_label_info: dict[str, Any] | None = None
    if anchor_in_window and not concise_style.show_site_names():
        site_label_info = {"omitted": True, "reason": "簡要報告圖內不印地點名稱（圖說說明地點），只保留錨點星號記號"}
    elif anchor_in_window:
        path_lonlat = (np.asarray(path_lon, dtype=float), np.asarray(path_lat, dtype=float)) if snap.path_m.shape[0] >= 2 else None
        site_label_info = _place_site_label(
            figure, axis, data.site_name_zh, (anchor_lon, anchor_lat), (xlim, ylim),
            polylines=[(np.asarray(chosen_lon, dtype=float), np.asarray(chosen_lat, dtype=float)), *later_curves_lonlat, *([path_lonlat] if path_lonlat else [])],
            points=[(float(core_lon[0]), float(core_lat[0])), *[(lon_v, lat_v) for lon_v, lat_v, _ in later_points]],
            extra_artists=later_annotations,
            land_probe=lambda lon_v, lat_v: _is_blank_cell(s1_full, data.lon, data.lat, lon_v, lat_v),
            key_patch=key_patch,
        )

    # 參考箭頭（1 m/s）：地圖內右下角、不加底框、不加白色描邊（與起始曲線、核心一致，v47 起圖面一律無白邊）；與流速箭頭的重疊已由保留區排除。
    tip_x, tip_y = key_anchor
    key = axis.quiverkey(
        quiver, tip_x / axes_box.w, tip_y / axes_box.h, reference, f"{reference:g} m/s", labelpos="E", coordinates="axes",
        labelsep=0.05, fontproperties={"size": 8}, zorder=10,
    )
    key.text.set_gid("reference_key_label")  # 文字在 SVG 內以字形路徑保存、沒有字串註解；用 gid 讓 gate 找到並確認它在地圖軸內
    key.vector.set_gid("reference_key_arrow")
    key_info = _measure_reference_key(figure, axis, key, reference * inch_per_m_s, (axes_box.w, axes_box.h))
    key_box = key_info["bbox_axes_inch"]
    clearance = _arrows_clearance_to_box(tail_x, tail_y, delta_x, delta_y, key_box)
    key_info["arrows_overlapping_key"] = int(sum(1 for value in clearance if value <= 0.0))
    key_info["min_arrow_clearance_inch"] = round(float(min(clearance)), 4) if len(clearance) else None
    key_info["arrows_suppressed"] = n_suppressed
    figure._traps_example_key = key_info  # noqa: SLF001

    start_text = time_label_utc(snap.time_utc_ns)[:10] if day_unit else f"{time_label_utc(snap.time_utc_ns)} UTC"
    subtitle = f"起始 {start_text}｜壽命 {duration_text(snap.lifetime_hours, day_unit=day_unit)}"
    # 簡要報告：圖例與大圖相同（淺灰外框、白底、不透明度 0.8，置中於主圖框下方）
    std.legend_below(figure, std.SINGLE_HEIGHT_INCH, handles, labels, ncol=3, framed=concise_on, center_x_inch=(axes_box.x + 0.5 * axes_box.w) if concise_on else None)
    std.draw_title_block(figure, std.SINGLE_HEIGHT_INCH, f"{data.site_name_zh} — 代表性 TRAP", subtitle)
    concise_style.record_note(
        "example_zoom",
        data.site_id,
        {
            **{key_: value for key_, value in snap.zoom.items()},
            "track_id": snap.track_id,
            "start_time_utc": time_label_utc(snap.time_utc_ns),
            "lifetime_hours": snap.lifetime_hours,
            "n_observations": snap.n_observations,
            "mean_coast_distance_m": snap.mean_coast_m,
            "coastal": bool(snap.is_coastal),
            "window_xlim": [float(v) for v in xlim],
            "window_ylim": [float(v) for v in ylim],
            "window_km": window["window_km"],
            "anchor_in_window": anchor_in_window,
            "site_label": site_label_info,
            "field_fractions": field_fractions,
            "anchor_lonlat": [float(anchor_lon), float(anchor_lat)],
            "coast_km_in_window": float((snap.zoom or {}).get("window", {}).get("coast_km_in_window", float("nan"))) if (snap.zoom or {}).get("window") else None,
            "window_aspect_error": float(
                (xlim[1] - xlim[0]) * max(float(np.cos(np.radians(0.5 * (ylim[0] + ylim[1])))), 0.05) / (ylim[1] - ylim[0]) / (axes_box.w / axes_box.h) - 1.0
            ),
            "axes_box_inch": [axes_box.w, axes_box.h],
            "colourbar_box_inch": [cbar_box.w, cbar_box.h],
            "window_centre_lonlat": [float(v) for v in window["centre_lonlat"]],
            "window_centred_on": window["centred_on"],
            "window_shift_to_keep_inside_domain_km": window["shift_to_keep_inside_domain_km"],
            "later_positions_inside_window": window["positions_inside_window"],
            "start_core_lonlat": [float(v) for v in window["start_core_lonlat"]],
            "later_offsets": [text for _lon, _lat, text in later_points],
            "natural_s1_limit": natural_limit,
            "s1_limit_used": limit,
            "vector_inch_per_m_s": inch_per_m_s,
            "vector_reference_m_s": reference,
            "vector_arrows_drawn": int(sample_rows.size),
            "vector_arrows_suppressed_for_key": n_suppressed,
            "reference_key": key_info,
        },
    )
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
    ``streamplot`` 畫在有效海域（遮罩處中止），速度依經緯度圖面等比例換算方向。
    ``snap`` 為 ``None`` 時只標示無代表 track；正式成果不再在三分圖下方繪製紅色說明、
    metadata 或 footer，``footer`` 參數僅為相容既有呼叫介面而保留。
    """

    from .literature_figures import _divergence_sinv

    matplotlib, plt = _load_matplotlib()
    base_view = padded_view(data)
    report_mode = vector_report_enabled()
    limits = None
    view = base_view
    if report_mode:
        # 標準 multi 畫布（6.3×7.0）：(a)(b) 在上排、(c) 在下排左側、圖例在下排右側空格；
        # 三個面板同一軸框大小，固定軸框、擴張經緯度範圍，色階只取原視窗內的場。
        view, limits = standard_map_frame(data, base_view, std.METHOD_A)
    rows, cols = view
    lon, lat = data.lon[cols], data.lat[rows]
    if report_mode:
        fig_w, fig_h = std.WIDTH_INCH, std.MULTI_HEIGHT_INCH
        figure = plt.figure(figsize=(fig_w, fig_h))
        axes = [std.add_box_axes(figure, box) for box in (std.METHOD_A, std.METHOD_B, std.METHOD_C)]
        caxes = [std.add_box_axes(figure, std.METHOD_A_CBAR), None, std.add_box_axes(figure, std.METHOD_C_CBAR)]
        panel_w = std.METHOD_A.w
    else:
        panel_w, panel_h = _fit_panel(map_data_aspect(data, view), 4.3, 4.6)
        pitch_cbar = _LEFT_INCH + panel_w + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + 0.95
        pitch_plain = _LEFT_INCH + panel_w + 0.25
        content_w = 2.0 * pitch_cbar + pitch_plain + 0.2
        fig_w = max(content_w, _MIN_FIGURE_WIDTH)
        title_h = 0.78
        # 下緣只保留經度標籤所需高度；移除舊版紅色 notes 後，面板直接下移以消除空白。
        fig_h = 0.1 + title_h + 0.30 + panel_h + _BOTTOM_LABEL_INCH
        y0 = _BOTTOM_LABEL_INCH
        x_start = (fig_w - content_w) / 2.0
        positions = [x_start + _LEFT_INCH, x_start + pitch_cbar + _LEFT_INCH, x_start + pitch_cbar + pitch_plain + _LEFT_INCH]
        y_positions = [y0, y0, y0]
        figure = plt.figure(figsize=(fig_w, fig_h))
        axes = [_add_inch_axes(figure, x, y, panel_w, panel_h) for x, y in zip(positions, y_positions, strict=True)]
        caxes = [
            _add_inch_axes(figure, positions[0] + panel_w + _CBAR_PAD_INCH, y_positions[0], _CBAR_WIDTH_INCH, panel_h),
            None,
            _add_inch_axes(figure, positions[2] + panel_w + _CBAR_PAD_INCH, y_positions[2], _CBAR_WIDTH_INCH, panel_h),
        ]
    pairs: list[tuple[Any, Any]] = []
    buffer_handles: list[Any] = []
    titles = (T["method_panel_a"], T["method_panel_b"], T["method_panel_c"])
    subtitle: str | None = None

    if snap is None:
        for axis, cax, title in zip(axes, caxes, titles, strict=True):
            _prepare_map_axis(axis, data, view, coastline, limits=limits, max_ticks=3 if report_mode else None)
            axis.set_title(title, fontsize=10)
            if cax is not None:
                blank = axis.pcolormesh(lon, lat, np.ma.masked_all((lat.size, lon.size)), shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
                blank_bar = figure.colorbar(blank, cax=cax)
                if report_mode:
                    style_colorbar(blank_bar)
                pairs.append((axis, cax))
        _empty_state(axes[1], T["example_no_track"])
        subtitle = T["example_no_track_short"]
    else:
        mapper = MetricToLonLat(snap.x_m_axis, snap.y_m_axis, data.lon, data.lat)
        subtitle = (
            f"{T['example_birth']} {time_label_utc(snap.time_utc_ns)} UTC｜"
            f"{T['example_lifetime']} {duration_text(snap.lifetime_hours, day_unit=day_unit)}"
        )
        # (a) s1
        s1_full = np.where(snap.derivative_valid, snap.s1_sinv * SECONDS_PER_DAY, np.nan)
        s1_day = s1_full[rows, cols]
        limit = _symmetric_limit(s1_full[base_view[0], base_view[1]])
        mesh_a = axes[0].pcolormesh(lon, lat, np.ma.masked_invalid(s1_day), shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        colorbar_a = figure.colorbar(mesh_a, cax=caxes[0])
        colorbar_a.set_label(S1_DAY_LABEL, fontsize=8, labelpad=2)
        colorbar_a.ax.tick_params(labelsize=8, pad=1)
        pairs.append((axes[0], caxes[0]))
        # (c) 水平散度
        divergence = _divergence_sinv(snap.u, snap.v, snap.x_m_axis, snap.y_m_axis, snap.valid) * SECONDS_PER_DAY
        divergence_view = divergence[rows, cols]
        div_limit = _symmetric_limit(divergence[base_view[0], base_view[1]])
        mesh_c = axes[2].pcolormesh(lon, lat, np.ma.masked_invalid(divergence_view), shading="auto", cmap="RdBu_r", vmin=-div_limit, vmax=div_limit)
        colorbar_c = figure.colorbar(mesh_c, cax=caxes[2])
        colorbar_c.set_label(T["divergence_label"] + r" $\nabla\cdot\mathbf{u}$ ($\mathrm{day^{-1}}$)", fontsize=8, labelpad=2)
        colorbar_c.ax.tick_params(labelsize=8, pad=1)
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
            buffer_handle = _prepare_map_axis(axis, data, view, coastline, limits=limits, max_ticks=3 if report_mode else None)
            if buffer_handle is not None:
                buffer_handles.append(buffer_handle)
        for axis, title in zip(axes, titles, strict=True):
            axis.set_title(title, fontsize=10)
            axis.tick_params(labelsize=8, pad=1)
            axis.xaxis.label.set_size(8)
            axis.yaxis.label.set_size(8)
        axes[0].plot([], [], color="#111111", linewidth=1.3, label=T["trap_curve"])
        axes[0].plot([], [], marker="o", markersize=4, color="#e31a1c", linestyle="none", label=T["trap_core"])
        legend_handles = [*axes[0].get_legend_handles_labels()[0], *buffer_handles[:1]]
        if report_mode:
            legend_labels = [handle.get_label() for handle in legend_handles]
            figure.legend(
                legend_handles,
                legend_labels,
                loc="center",
                bbox_to_anchor=(std.METHOD_LEGEND_CENTER[0] / fig_w, std.METHOD_LEGEND_CENTER[1] / fig_h),
                ncol=1,
                fontsize=std.TEXT_FONT_PT,
                frameon=False,
                handlelength=1.6,
            )
        else:
            axes[0].legend(handles=legend_handles, loc="lower right", fontsize=7)

    if report_mode:
        # 上排兩張地圖使用相同經緯度範圍與 grid，但 (b) 的緯度刻度與 ylabel
        # 只會重複 (a) 的資訊，且會侵入 (a) 色條保留區；保留 tick locator
        # 與 grid，僅關閉 (b) 左側文字，讓 (a) 的完整地理軸成為唯一標示。
        axes[1].set_ylabel("")
        axes[1].tick_params(axis="y", labelleft=False)
        std.draw_title_block(figure, fig_h, f"{data.site_name_zh} — {T['method_title']}", subtitle)
    else:
        align_colorbars_to_axes(figure, tuple(pairs))
        _sequence_title(figure, fig_w, fig_h, f"{data.site_name_zh} — {T['method_title']}", subtitle)
    _save(figure, plt, output_path, dpi)
    return output_path


# ---------------------------------------------------------------------------
# 3. 長壽命 track 核心路徑（Kunz Fig. 4b 風格）
# ---------------------------------------------------------------------------


def render_long_tracks(
    output_path: Path, data: SiteFigureData, tracks: TrackFigureInputs, *, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150
) -> Path:
    """長壽命 track 核心路徑，依壽命著色（由短到長疊繪，最長在最上層），起點小圓點。

    色條標籤為「單一結構連續存在天數／時數」（單一 TRAP 結構從出生到消失的連續時間，不是
    整個研究期的累計）。色階下限為長壽命門檻；上限預設為本站最長 track，亦可由
    ``tracks.colour_max_hours`` 明示（同一產品五站共用，見
    :func:`shared_long_track_colour_max_hours`）。

    圖面保留上方的壽命門檻與 track 數統計、地圖圖例及座標軸；原先位於下方的
    metadata、隨機對照與近岸分類說明不再繪製。``footer`` 參數保留既有共用入口
    的相容性，但不會改變圖面內容。
    """

    from matplotlib.collections import LineCollection

    matplotlib, plt = _load_matplotlib()
    day_unit = uses_day_unit(tracks.nominal_step_hours)
    scale = 24.0 if day_unit else 1.0
    view = _view_extent(data)
    report_mode = vector_report_enabled()
    limits = None
    if report_mode:
        # 標準 single 畫布（6.3×4.4）與 MAP1 軸框；固定軸框、擴張經緯度範圍。
        view, limits = standard_map_frame(data, view, concise_map_boxes()[0])  # 簡要報告：LBT 長寬比的主圖框；其餘 MAP1
    rows, cols = view
    concise = report_mode and concise_style.concise_enabled()
    if concise:
        data = replace(data, site_name_zh=concise_style.display_site_name(data.site_id, data.site_name_zh))
    mapper = MetricToLonLat(tracks.x_m_axis, tracks.y_m_axis, data.lon, data.lat)
    drawn = list(tracks.long_tracks)
    n_total = int(tracks.n_long_total)
    if report_mode:
        layout = {"fig_w": std.WIDTH_INCH, "fig_h": std.SINGLE_HEIGHT_INCH}
        figure = plt.figure(figsize=(layout["fig_w"], layout["fig_h"]))
        axis = std.add_box_axes(figure, concise_map_boxes()[0])
        cax = std.add_box_axes(figure, concise_map_boxes()[1])
    else:
        layout = _single_map_layout(data, view, legend_height=0.0, max_width=7.0, max_height=6.0)
        figure = plt.figure(figsize=(layout["fig_w"], layout["fig_h"]))
        axis = _add_inch_axes(figure, layout["x0"], layout["y0"], layout["panel_w"], layout["panel_h"])
        cax = _add_inch_axes(figure, layout["x0"] + layout["panel_w"] + _CBAR_PAD_INCH, layout["y0"], _CBAR_WIDTH_INCH, layout["panel_h"])
    lifetimes = np.asarray([item[0] for item in drawn], dtype=np.float64) / scale
    vmin = float(tracks.long_track_hours) / scale
    if tracks.colour_max_hours is not None:
        # 跨站共用色階：上限由 caller 以五站最長壽命進位後給定；不得小於本站最長 track。
        vmax = float(tracks.colour_max_hours) / scale
        if lifetimes.size and float(lifetimes.max()) > vmax + 1.0e-9:
            raise ValueError("colour_max_hours 小於本站最長 track 壽命，色階會截斷資料")
        vmax = max(vmax, vmin + 1.0e-9)
    else:
        vmax = max(float(lifetimes.max()) if lifetimes.size else vmin, vmin + 1.0e-9)
        if lifetimes.size and vmax <= vmin:
            vmax = vmin + 1.0
    if concise:
        concise_style.record_limit("long_tracks", data.site_id, "lifetime", (vmin, vmax))
        shared_limit = concise_style.shared_limits("long_tracks", "lifetime")
        if shared_limit is not None:
            # 簡要報告：跨站指定色階（天）；下限必須等於長壽命門檻，上限不得截斷任何 track。
            if abs(shared_limit[0] - vmin) > 1.0e-9:
                raise ValueError(f"long_tracks 共用色階下限 {shared_limit[0]} 與長壽命門檻 {vmin} 不同")
            if lifetimes.size and float(lifetimes.max()) > shared_limit[1] + 1.0e-9:
                raise ValueError("long_tracks 共用色階上限小於本站最長 track 壽命，色階會截斷資料")
            vmax = float(shared_limit[1])
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
    # 色條標籤明示是「單一結構」的連續存在時間，避免被讀成整個研究期的累計天數。
    colorbar_label = T["long_tracks_colorbar_days"] if day_unit else T["long_tracks_colorbar_hours"]
    colorbar.set_label(colorbar_label)
    if report_mode:
        style_colorbar(colorbar, colorbar_label)
        if concise:
            concise_style.style_colorbar_shared(
                colorbar, vmin, vmax, colorbar_label, name="lifetime", extend="neither", tick_values=concise_style.shared_ticks("long_tracks", "lifetime")
            )
    buffer_handle = _prepare_map_axis(axis, data, view, coastline, limits=limits)
    if buffer_handle is not None:
        legend_handles.append(buffer_handle)
    if not report_mode and legend_handles:
        axis.legend(handles=legend_handles, loc="lower right", fontsize=7)
    if not drawn:
        _empty_state(axis, T["no_data"])
    threshold_text = duration_text(tracks.long_track_hours, day_unit=day_unit)
    subtitle = f"{T['long_tracks_criterion']} {threshold_text}｜n = {n_total} {T['n_tracks']}"
    title = f"{data.site_name_zh} — {T['long_tracks_title']}"
    if report_mode:
        std.legend_below(figure, layout["fig_h"], legend_handles, [handle.get_label() for handle in legend_handles], ncol=3)
        std.draw_title_block(figure, layout["fig_h"], title, subtitle)
    else:
        align_colorbars_to_axes(figure, ((axis, cax),))
        _sequence_title(figure, layout["fig_w"], layout["fig_h"], title, subtitle)
    _save(figure, plt, output_path, dpi)
    return output_path


# ---------------------------------------------------------------------------
# 4. track 通過格點數（Kunz Fig. 4c 風格）
# ---------------------------------------------------------------------------


def render_track_density(
    output_path: Path, data: SiteFigureData, tracks: TrackFigureInputs, *, coastline: CoastlineGeometry | None, footer: str, dpi: int = 150
) -> Path:
    """繪製通過格點的不同 track 數與長壽命 track 數兩張地圖。

    一般分析輸出維持兩個並排面板；報告向量模式則改為上下排列，並將各自色條放在
    地圖下方，以固定 6.3 英吋列印寬度保留 8 pt 以上的軸標籤。兩種版面都直接使用
    ``spatial_all`` 的整數計數與 ``site_mask``，不改變真實 track 分母、NaN 遮罩或
    色階上限。原先下方的計數方法說明、metadata 與 footer 移除，以免成果圖出現
    額外描述文字；``footer`` 參數仍保留供既有呼叫介面相容。
    """

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
    if vector_report_enabled():
        # 標準 multi 畫布（6.3×7.0）：兩張地圖上下排列，各自右側垂直色條，軸框位置與大小固定
        # （地圖寬度、x 位置與 MAP1 相同）；固定軸框、擴張經緯度範圍。兩個 counts 陣列只重新
        # 指定繪圖位置，遮罩、vmax 與整數刻度規則完全沿用原流程。
        view, limits = standard_map_frame(data, view, std.DENSITY_1)
        rows, cols = view
        lon, lat = data.lon[cols], data.lat[rows]
        fig_w, fig_h = std.WIDTH_INCH, std.MULTI_HEIGHT_INCH
        figure = plt.figure(figsize=(fig_w, fig_h))
        buffer_handles: list[Any] = []
        for index, ((title, key, cmap), box) in enumerate(zip(fields, (std.DENSITY_1, std.DENSITY_2), strict=True)):
            axis = std.add_box_axes(figure, box)
            cax = std.add_box_axes(figure, std.Box(std.DENSITY_CBAR_X, box.y, std.DENSITY_CBAR_W, box.h))
            counts = np.asarray(data.spatial_all[key], dtype=np.float64)[rows, cols]
            inside = np.asarray(data.site_mask, dtype=bool)[rows, cols]
            masked = np.ma.masked_where(~inside | ~np.isfinite(counts), counts)
            vmax = max(float(masked.max()) if masked.count() else 1.0, 1.0)
            mesh = axis.pcolormesh(lon, lat, masked, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax)
            colorbar = figure.colorbar(mesh, cax=cax)
            colorbar.locator = MaxNLocator(integer=True)
            colorbar.update_ticks()
            style_colorbar(colorbar, T["n_tracks"])
            axis.set_title(title, fontsize=std.SUBTITLE_FONT_PT, pad=3)
            buffer_handle = _prepare_map_axis(axis, data, view, coastline, limits=limits)
            if buffer_handle is not None:
                buffer_handles.append(buffer_handle)
        if buffer_handles:
            std.legend_below(figure, fig_h, [buffer_handles[0]], [coast_buffer_label(data)], ncol=1)
        std.draw_title_block(figure, fig_h, f"{data.site_name_zh} — {T['track_density_title']}")
        _save(figure, plt, output_path, dpi)
        return output_path

    panel_w, panel_h = _fit_panel(map_data_aspect(data, view), 4.6, 4.2)
    column_pitch = _LEFT_INCH + panel_w + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH
    fig_w = max(2.0 * column_pitch, _MIN_FIGURE_WIDTH)
    title_h = 0.78
    fig_h = 0.1 + title_h + 0.3 + panel_h + _BOTTOM_LABEL_INCH
    x_offset = (fig_w - 2.0 * column_pitch) / 2.0
    y0 = _BOTTOM_LABEL_INCH
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
            axis.legend(handles=[buffer_handle], loc="lower right", fontsize=8 if vector_report_enabled() else 7)
        pairs.append((axis, cax))
    align_colorbars_to_axes(figure, tuple(pairs))
    _sequence_title(figure, fig_w, fig_h, f"{data.site_name_zh} — {T['track_density_title']}", None)
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
    報告向量模式將兩個速度直方圖置於上排、去向玫瑰圖置於下排，固定 Figure 寬度為
    6.3 英吋；一般分析輸出仍維持原有並排版面。兩種版面都只保留各軸本身的圖例，
    不再繪製殘差、metadata 或方法註腳；``footer`` 參數保留既有共用入口的相容性，
    但不會改變圖面內容。
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
    if vector_report_enabled():
        # 標準 multi 畫布（6.3×7.0）：兩個速度分量直方圖在上排、去向玫瑰與色條在下排，
        # 位置與大小固定。速度樣本、FD bins、靜止排除與玫瑰統計完全不變。
        fig_w, fig_h = std.WIDTH_INCH, std.MULTI_HEIGHT_INCH
        figure = plt.figure(figsize=(fig_w, fig_h))
        ax_cx = std.add_box_axes(figure, std.PROP_HIST_X)
        ax_cy = std.add_box_axes(figure, std.PROP_HIST_Y)
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
            axis.set_title(f"{title} (n={n})", fontsize=10, pad=3)
            axis.set_xlabel("km/day", fontsize=8)
            axis.tick_params(labelsize=8, pad=1)
            axis.grid(alpha=0.25, linewidth=0.5)
            if n:
                axis.hist(values, bins=edges, color=colour, alpha=0.85, edgecolor="white", linewidth=0.4)
                median = float(np.median(values))
                axis.axvline(median, color="#111111", linestyle="--", linewidth=1.3, label=f"{T['propagation_median']} {median:.1f} km/day")
                axis.axvline(0.0, color="0.5", linewidth=0.7)
                axis.legend(loc="upper right", fontsize=8)
            else:
                _empty_state(axis, T["no_data"])
        ax_cx.set_ylabel(T["propagation_samples"], fontsize=8)

        ax_rose = std.add_box_axes(figure, std.PROP_ROSE, projection="polar")
        cax = std.add_box_axes(figure, std.PROP_ROSE_CBAR)
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
        ax_rose.tick_params(axis="y", labelsize=8)
        mean_all = float(np.mean(np.hypot(cx, cy))) if n else float("nan")
        rose_title = T["propagation_rose"] + (f"\n{T['propagation_mean_speed']} {mean_all:.1f} km/day" if n else "")
        ax_rose.set_title(rose_title, fontsize=10, pad=12)
        colorbar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        colorbar.set_label(f"{T['propagation_mean_speed']} (km/day)", fontsize=8, labelpad=2)
        colorbar.ax.tick_params(labelsize=8, pad=1)
        if not n:
            ax_rose.text(0.5, 0.5, T["no_data"], transform=ax_rose.transAxes, ha="center", va="center", color="#7f0000")
        ax_rose.set_ylabel("")
        std.draw_title_block(figure, fig_h, f"{data.site_name_zh} — {T['propagation_title']}", subtitle)
        _save(figure, plt, output_path, dpi)
        return output_path

    hist_w, hist_h = 3.5, 3.1
    rose_size = 3.1
    fig_w = _LEFT_INCH + hist_w + 0.35 + hist_w + 0.35 + 0.55 + rose_size + _CBAR_PAD_INCH + _CBAR_WIDTH_INCH + _CBAR_RIGHT_INCH
    title_h = 0.78
    fig_h = 0.1 + title_h + 0.3 + max(hist_h, rose_size) + _BOTTOM_LABEL_INCH
    y0 = _BOTTOM_LABEL_INCH
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
            axis.legend(loc="upper right", fontsize=8 if vector_report_enabled() else 7)
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
    ax_rose.tick_params(axis="y", labelsize=8 if vector_report_enabled() else 7)
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
    """建立既有入口所需的 footer 字串，但由成果圖繪製函式刻意不呈現。

    aggregate 呼叫鏈仍以 ``footer`` 參數傳遞期間、狀態、參數與有效快照數，
    因此保留這個資料介面可避免改動 manifest／共用入口；本模組的正式圖面已移除
    footer 與其預留高度，實際可見內容只由圖例、座標軸、主標題及上方統計組成。
    """

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
    zoom = concise_style.example_zoom_options() if (vector_report_enabled() and concise_style.concise_enabled()) else None
    if zoom is not None and snap is not None:
        # 簡要報告：只輸出縮放代表圖（不輸出方法比較）。
        with cjk_font_context(font):
            render_example_zoom(site_dir / f"{stem}_example.png", data, snap, coastline=coastline, zoom=zoom, day_unit=day_unit, dpi=dpi)
        return [f"figures/{data.site_id}/{stem}_example.png"]
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
    "render_example_zoom",
    "select_zoom_track",
    "track_edge_distance_m",
    "zoom_window",
    "relative_time_label",
    "render_coastal_example",
    "render_snapshot_figures",
    "render_track_figures",
    "resolve_source_paths",
    "rose_histogram",
    "shared_long_track_colour_max_hours",
    "uses_day_unit",
    "verify_recomputed_core",
]
