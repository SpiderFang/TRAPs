"""TRAP 逐格空間統計與可追溯的圖面資料結構。

本模組把單一 flow domain 的 ``catalog.json`` 與已通過上游輸入檢查的
``SurfaceMonth`` 對齊，建立 Kunz 等人空間分布圖及 Ghosh 等人出現日數圖可借鏡的
逐格統計。catalog 只保存檢出摘要，月份快取則提供每個 UTC snapshot 的實際有效
遮罩與地理座標；兩者必須逐時對應，不能以 catalog 的全域 coverage 數字代替每個格點
的有效分母。

曲線通過率的分子是「該格有至少一條合格 TRAP 曲線通過的 UTC snapshot 數」，同一
snapshot 即使有多條曲線重疊也只計一次；分母是該格在選定 snapshot 序列中有效的時刻
數。完整 track 壽命只使用沒有左右截尾的 track，截尾 track 的觀測跨度另存為下界
資訊，絕不混入未截尾 track 的平均觀測跨度。首末檢出跨度並非物理壽命；單時刻
track 的 0 小時代表時間解析度不足。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ocm import SurfaceMonth
from .tracking import densify_polyline


def _require(condition: bool, message: str) -> None:
    """在空間統計輸入契約不成立時立即停止，避免靜默產生錯誤熱區。"""

    if not condition:
        raise ValueError(message)


def _readonly_array(value: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """建立不可寫的結果陣列，避免繪圖或後處理改寫統計證據。"""

    array = np.asarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SpatialSummary:
    """一個 flow domain 的逐格 TRAP 空間統計。

    所有二維欄位均使用 ``(lat, lon)`` 排列，與 ``SurfaceMonth.snapshot().valid``
    相同；``lon``、``lat`` 是 WGS84 經緯度軸，供成果圖直接建立地理座標。欄位中
    的 ``NaN`` 表示沒有可用分母或沒有該類型的 track，不能當作物理上的零。

    ``mean_complete_track_duration_hours`` 是為維持欄位相容而保留的名稱，只由左右
    皆未截尾 track 填入，數值代表首末檢出的觀測跨度，並非物理壽命；
    ``mean_censored_track_observed_span_hours`` 是截尾 track 的可觀測首尾跨度下界，
    僅供資料狀態說明，不能解讀為完整生命週期。
    """

    domain_id: str
    month: str
    state_label: str
    science_status: str
    snapshot_indices: tuple[int, ...]
    time_utc_ns: tuple[int, ...]
    lon: np.ndarray
    lat: np.ndarray
    valid_snapshot_count: np.ndarray
    core_count: np.ndarray
    mean_abs_core_s1_sinv: np.ndarray
    curve_pass_count: np.ndarray
    curve_pass_rate: np.ndarray
    complete_track_count: np.ndarray
    censored_track_count: np.ndarray
    mean_complete_track_duration_hours: np.ndarray
    mean_censored_track_observed_span_hours: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """核對座標、欄位 shape 與時間索引，並鎖定所有數值陣列。"""

        lon = _readonly_array(self.lon, dtype=np.float64)
        lat = _readonly_array(self.lat, dtype=np.float64)
        _require(lon.ndim == 1 and lat.ndim == 1, "SpatialSummary lon/lat 必須是一維座標軸")
        _require(np.all(np.isfinite(lon)) and np.all(np.diff(lon) > 0.0), "SpatialSummary lon 必須嚴格遞增")
        _require(np.all(np.isfinite(lat)) and np.all(np.diff(lat) > 0.0), "SpatialSummary lat 必須嚴格遞增")
        shape = (lat.size, lon.size)
        field_names = (
            "valid_snapshot_count",
            "core_count",
            "mean_abs_core_s1_sinv",
            "curve_pass_count",
            "curve_pass_rate",
            "complete_track_count",
            "censored_track_count",
            "mean_complete_track_duration_hours",
            "mean_censored_track_observed_span_hours",
        )
        for name in field_names:
            field = _readonly_array(getattr(self, name), dtype=np.float64 if "mean" in name or name == "curve_pass_rate" else np.int64)
            _require(field.shape == shape, f"SpatialSummary.{name} shape 必須是 {shape}")
            object.__setattr__(self, name, field)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "lat", lat)
        indices = tuple(int(index) for index in self.snapshot_indices)
        times = tuple(int(value) for value in self.time_utc_ns)
        _require(len(indices) == len(times) and len(indices) > 0, "SpatialSummary 必須保存非空且平行的 snapshot 時間軸")
        _require(len(set(indices)) == len(indices), "SpatialSummary snapshot_indices 不可重複")
        object.__setattr__(self, "snapshot_indices", indices)
        object.__setattr__(self, "time_utc_ns", times)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """回傳二維統計場的 ``(lat, lon)`` shape。"""

        return (int(self.lat.size), int(self.lon.size))

    @property
    def curve_pass_rate_percent(self) -> np.ndarray:
        """回傳圖面使用的百分比 view；原始 ``curve_pass_rate`` 仍是 0–1 比例。"""

        return _readonly_array(self.curve_pass_rate * 100.0, dtype=np.float64)

    def field_units(self) -> dict[str, str]:
        """回傳每個輸出欄位的單位，供圖說、JSON metadata 與後續 CLI 使用。"""

        return {
            "valid_snapshot_count": "count",
            "core_count": "count",
            "mean_abs_core_s1_sinv": "s^-1",
            "curve_pass_count": "count",
            "curve_pass_rate": "fraction (0-1)",
            "complete_track_count": "count",
            "censored_track_count": "count",
            "mean_complete_track_duration_hours": "hour",
            "mean_censored_track_observed_span_hours": "hour lower bound",
        }

    def to_metadata(self) -> dict[str, Any]:
        """回傳不含大型陣列的 JSON 相容統計契約與來源摘要。"""

        return {
            "schema_version": "spatial_summary_v1",
            "domain_id": self.domain_id,
            "month": self.month,
            "state_label": self.state_label,
            "science_status": self.science_status,
            "snapshot_indices": list(self.snapshot_indices),
            "time_utc_ns": list(self.time_utc_ns),
            "coordinate_reference": "WGS84 longitude/latitude axes from verified SurfaceMonth",
            "coordinate_units": "degrees",
            "array_order": "(lat, lon)",
            "field_units": self.field_units(),
            "field_definitions": {
                "valid_snapshot_count": "每格 valid=True 的 UTC snapshot 數，作為逐格分母",
                "mean_abs_core_s1_sinv": "該格合格核心 |s1| 的平均值；無核心為 NaN",
                "curve_pass_count": "有至少一條合格曲線通過該格的 UTC snapshot 數；同時刻去重",
                "curve_pass_rate": "curve_pass_count / valid_snapshot_count；零分母為 NaN",
                "mean_complete_track_duration_hours": "觸及該格且左右皆未截尾 track 的平均首末檢出觀測跨度；非物理壽命，單時刻 0 小時表示解析度不足",
                "mean_censored_track_observed_span_hours": "觸及該格之截尾 track 的平均觀測跨度下界",
            },
            "curve_rasterization": self.metadata.get("curve_rasterization", {}),
            "lifetime_rule": self.metadata.get("lifetime_rule", {}),
        }


def _rectilinear_axes(month: SurfaceMonth) -> tuple[np.ndarray, np.ndarray, float]:
    """從 SurfaceMonth 取出可明確格網化的公尺軸與半格距取樣步長。

    曲線資料使用局部公尺座標；只有在 ``x_m``、``y_m`` 可化為軸對齊規則格網時，
    才能以最近格點方式建立可重現的空間分布。若上游是曲線或旋轉格網，本函式拒絕
    產圖，避免把座標誤當成經緯度直角網格。
    """

    x_grid = np.asarray(month.x_m, dtype=np.float64)
    y_grid = np.asarray(month.y_m, dtype=np.float64)
    _require(x_grid.ndim == y_grid.ndim == 2, "SurfaceMonth x_m/y_m 必須是二維格網")
    x_axis = x_grid[0, :]
    y_axis = y_grid[:, 0]
    _require(np.all(np.isfinite(x_axis)) and np.all(np.diff(x_axis) > 0.0), "x_m 第一列必須是嚴格遞增軸")
    _require(np.all(np.isfinite(y_axis)) and np.all(np.diff(y_axis) > 0.0), "y_m 第一欄必須是嚴格遞增軸")
    _require(np.allclose(x_grid, x_axis[None, :], atol=1.0e-6, rtol=0.0), "x_m 不是軸對齊直角格網")
    _require(np.allclose(y_grid, y_axis[:, None], atol=1.0e-6, rtol=0.0), "y_m 不是軸對齊直角格網")
    spacings = np.concatenate((np.diff(x_axis), np.diff(y_axis)))
    _require(np.all(np.isfinite(spacings)) and np.all(spacings > 0.0), "公尺格距必須為正有限值")
    return x_axis, y_axis, float(np.min(spacings) / 2.0)


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """將公尺座標轉成最近格點 index，並回傳仍在網格內的布林遮罩。"""

    finite = np.isfinite(values)
    right = np.searchsorted(axis, np.where(finite, values, axis[0]), side="left")
    right = np.clip(right, 0, axis.size - 1)
    left = np.clip(right - 1, 0, axis.size - 1)
    choose_left = np.abs(values - axis[left]) <= np.abs(values - axis[right])
    indices = np.where(choose_left, left, right).astype(np.int64)
    inside = finite & (values >= axis[0]) & (values <= axis[-1])
    return indices, inside


def _curve_cells(
    points_m: np.ndarray,
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    valid_mask: np.ndarray,
    max_step_m: float,
) -> np.ndarray:
    """將一條 TRAP 曲線加密後映射成該 snapshot 的有效格點 flat index。

    加密步長不大於最小格距的一半，端點與線段內部均保留；每一個 sample 取最近格點，
    再套用該時刻 valid mask。這個規則可避免只查核心或曲線端點而漏掉穿越格點，也會
    明確排除曲線落在海岸／缺值格點的部分。它是規則格網上的統計 rasterization，
    不是把曲線宣稱成觀測面積。
    """

    points = np.asarray(points_m, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 2 and points.shape[0] >= 1, "curve_points_m 必須是 (n, 2)")
    _require(np.all(np.isfinite(points)), "curve_points_m 不可含非有限座標")
    dense = densify_polyline(points, max_step_m)
    cols, inside_x = _nearest_axis_indices(x_axis, dense[:, 0])
    rows, inside_y = _nearest_axis_indices(y_axis, dense[:, 1])
    inside = inside_x & inside_y
    if not np.any(inside):
        return np.empty(0, dtype=np.int64)
    rows = rows[inside]
    cols = cols[inside]
    keep = valid_mask[rows, cols]
    if not np.any(keep):
        return np.empty(0, dtype=np.int64)
    # ``ravel_multi_index`` 以 C-order 將 (row, column) 轉成二維欄位的 flat index；
    # 先 unique 可使同一條曲線在同一格的多個加密 sample 只保留一次。
    return np.unique(np.ravel_multi_index((rows[keep], cols[keep]), valid_mask.shape))


def _point_cell(
    point_m: Sequence[float],
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    valid_mask: np.ndarray,
) -> int | None:
    """把核心公尺座標轉成一個仍有效的 flat index；出界或無效時回傳 ``None``。"""

    _require(len(point_m) == 2 and all(value is not None for value in point_m), "core_xy_m 欄位缺失")
    point = np.asarray(point_m, dtype=np.float64)
    _require(point.shape == (2,) and np.all(np.isfinite(point)), "core_xy_m 必須是兩個有限座標")
    cols, inside_x = _nearest_axis_indices(x_axis, point[0:1])
    rows, inside_y = _nearest_axis_indices(y_axis, point[1:2])
    row = int(rows[0])
    col = int(cols[0])
    if not bool(inside_x[0] and inside_y[0]) or not bool(valid_mask[row, col]):
        return None
    return int(np.ravel_multi_index((row, col), valid_mask.shape))


def _row_int(row: Mapping[str, Any], name: str) -> int:
    """讀取 catalog row 的整數欄位並給出一致的 gate 錯誤。"""

    try:
        value = row[name]
        if isinstance(value, bool):
            raise TypeError
        return int(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"catalog row 缺少有效整數欄位 {name}: {row!r}") from error


def _validate_catalog_alignment(catalog: Mapping[str, Any], month: SurfaceMonth) -> tuple[list[int], list[int], dict[int, Mapping[str, Any]]]:
    """核對 catalog 的 domain、月份、snapshot index 與 SurfaceMonth 時間軸。"""

    source = catalog.get("source")
    _require(isinstance(source, Mapping), "catalog.source 必須是物件")
    source_domain = source.get("domain_id")
    source_month = source.get("month")
    if source_domain is not None:
        _require(str(source_domain) == month.domain_id, f"catalog domain 與 SurfaceMonth 不符: {source_domain!r} != {month.domain_id!r}")
    if source_month is not None:
        _require(str(source_month) == month.month, f"catalog month 與 SurfaceMonth 不符: {source_month!r} != {month.month!r}")
    raw_snapshots = catalog.get("snapshots")
    _require(isinstance(raw_snapshots, Sequence) and not isinstance(raw_snapshots, (str, bytes)) and len(raw_snapshots) > 0, "catalog.snapshots 必須是非空清單")
    snapshot_rows: dict[int, Mapping[str, Any]] = {}
    for raw in raw_snapshots:
        _require(isinstance(raw, Mapping), "catalog.snapshots 每項必須是物件")
        index = _row_int(raw, "snapshot_index")
        _require(index not in snapshot_rows, f"catalog snapshot_index 重複: {index}")
        _require(0 <= index < month.snapshot_count, f"catalog snapshot_index 超出 SurfaceMonth: {index}")
        time_value = _row_int(raw, "time_utc_ns")
        _require(time_value == int(month.time_utc_ns[index]), f"catalog snapshot {index} 的 UTC 時間與 SurfaceMonth 不符")
        snapshot_rows[index] = raw
    indices = sorted(snapshot_rows)
    return indices, [int(month.time_utc_ns[index]) for index in indices], snapshot_rows


def _detection_rows(catalog: Mapping[str, Any], selected: set[int]) -> dict[int, list[Mapping[str, Any]]]:
    """只保留合格 detection，並依 snapshot index 分組供逐時去重。"""

    raw_detections = catalog.get("detections", [])
    _require(isinstance(raw_detections, Sequence) and not isinstance(raw_detections, (str, bytes)), "catalog.detections 必須是清單")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for raw in raw_detections:
        _require(isinstance(raw, Mapping), "catalog.detections 每項必須是物件")
        index = _row_int(raw, "snapshot_index")
        _require(index in selected, f"detection snapshot_index 不在 catalog.snapshots: {index}")
        # catalog 可以保留候選或品質不合格項，但空間通過率契約只採 is_qualified=True。
        if raw.get("is_qualified") is True:
            grouped[index].append(raw)
    return grouped


def _track_records(catalog: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    """建立 track_id 到 catalog 摘要的索引，並檢查壽命欄位可解讀。"""

    raw_tracks = catalog.get("tracks", [])
    _require(isinstance(raw_tracks, Sequence) and not isinstance(raw_tracks, (str, bytes)), "catalog.tracks 必須是清單")
    records: dict[int, Mapping[str, Any]] = {}
    for raw in raw_tracks:
        _require(isinstance(raw, Mapping), "catalog.tracks 每項必須是物件")
        track_id = _row_int(raw, "track_id")
        _require(track_id not in records, f"catalog track_id 重複: {track_id}")
        duration = raw.get("duration_seconds")
        _require(duration is not None and np.isfinite(float(duration)) and float(duration) >= 0.0, f"track {track_id} duration_seconds 無效")
        _require(isinstance(raw.get("left_censored"), bool) and isinstance(raw.get("right_censored"), bool), f"track {track_id} 缺少截尾旗標")
        records[track_id] = raw
    return records


def build_spatial_summary(catalog: Mapping[str, Any], month: SurfaceMonth) -> SpatialSummary:
    """由單區 catalog 與已驗證 SurfaceMonth 建立逐格空間統計。

    參數
    ------
    catalog:
        CLI `detect` 產生的單一 sensitivity case catalog。其 `snapshots` 必須和
        `month.time_utc_ns` 對齊；`detections` 可同時包含合格與不合格項，但只有
        `is_qualified=True` 會進入曲線與核心統計。
    month:
        已通過 `open_month()` schema、domain、速度有限值與海岸 sidecar 檢查的
        `SurfaceMonth`。它提供每個 UTC 時刻的 valid mask 及地理經緯度軸。

    回傳
    ------
    SpatialSummary
        2D 欄位均為 `(lat, lon)`；每格分母為有效 snapshot 數。曲線 pass count
        以「每個 snapshot 每個格點最多一次」計算；完整與截尾 track 分別保存。

    限制
    ------
    目前只接受軸對齊的 rectilinear 公尺格網。跨月份 track 接續不在此函式中處理，
    所有壽命只代表單一 catalog 的觀測視窗；因此 `PILOT_ONLY` 仍會原樣傳遞。
    """

    _require(isinstance(catalog, Mapping), "catalog 必須是 mapping")
    _require(isinstance(month, SurfaceMonth), "month 必須是已驗證的 SurfaceMonth")
    indices, times, _snapshot_rows = _validate_catalog_alignment(catalog, month)
    selected = set(indices)
    grouped = _detection_rows(catalog, selected)
    tracks = _track_records(catalog)
    x_axis, y_axis, max_step_m = _rectilinear_axes(month)
    shape = month.spatial_shape
    valid_count = np.zeros(shape, dtype=np.int64)
    core_count = np.zeros(shape, dtype=np.int64)
    core_strength_sum = np.zeros(shape, dtype=np.float64)
    curve_pass_count = np.zeros(shape, dtype=np.int64)

    # 先建立逐格有效分母；即使全域 coverage 未達 CLI 門檻，仍保留實際每格可用數，
    # 讓後續圖面能明確呈現資料支援範圍而不是把缺值變成零出現率。
    snapshot_valid_masks: dict[int, np.ndarray] = {}
    for index in indices:
        valid = np.asarray(month.snapshot(index).valid, dtype=bool)
        snapshot_valid_masks[index] = valid
        valid_count += valid.astype(np.int64)

    # 每個 snapshot 使用 set-like bool field 合併所有合格曲線，確保同時刻重疊曲線
    # 不會把出現次數重複累加；曲線點只在該時刻 valid 格點上計數。
    for index, rows in grouped.items():
        valid = snapshot_valid_masks[index]
        passed_flat: set[int] = set()
        for row in rows:
            curve_cells = _curve_cells(
                row.get("curve_points_m"),
                x_axis=x_axis,
                y_axis=y_axis,
                valid_mask=valid,
                max_step_m=max_step_m,
            )
            passed_flat.update(int(value) for value in curve_cells)
            core_cell = _point_cell(
                (row.get("core_x_m"), row.get("core_y_m")),
                x_axis=x_axis,
                y_axis=y_axis,
                valid_mask=valid,
            )
            if core_cell is not None:
                row_core = float(row.get("core_s1_sinv"))
                _require(np.isfinite(row_core), "qualified detection core_s1_sinv 必須有限")
                core_strength_sum.flat[core_cell] += abs(row_core)
                core_count.flat[core_cell] += 1
        if passed_flat:
            flat = np.fromiter(sorted(passed_flat), dtype=np.int64)
            curve_pass_count.flat[flat] += 1

    mean_core = np.full(shape, np.nan, dtype=np.float64)
    np.divide(core_strength_sum, core_count, out=mean_core, where=core_count > 0)
    curve_rate = np.full(shape, np.nan, dtype=np.float64)
    np.divide(curve_pass_count, valid_count, out=curve_rate, where=valid_count > 0)

    complete_count = np.zeros(shape, dtype=np.int64)
    censored_count = np.zeros(shape, dtype=np.int64)
    complete_duration_sum = np.zeros(shape, dtype=np.float64)
    censored_span_sum = np.zeros(shape, dtype=np.float64)
    track_cells: dict[int, set[int]] = defaultdict(set)
    track_rows = [row for rows in grouped.values() for row in rows if row.get("track_id") is not None]
    for row in track_rows:
        track_id = _row_int(row, "track_id")
        _require(track_id in tracks, f"detection 指向不存在的 track_id: {track_id}")
        snapshot_index = _row_int(row, "snapshot_index")
        cells = _curve_cells(
            row.get("curve_points_m"),
            x_axis=x_axis,
            y_axis=y_axis,
            valid_mask=snapshot_valid_masks[snapshot_index],
            max_step_m=max_step_m,
        )
        track_cells[track_id].update(int(value) for value in cells)

    # 一條 track 的多個觀測只對每格計一次，否則長 track 會同時扭曲空間出現數與壽命圖。
    for track_id, cells in track_cells.items():
        track = tracks[track_id]
        duration_hours = float(track["duration_seconds"]) / 3_600.0
        if not bool(track["left_censored"]) and not bool(track["right_censored"]):
            complete_count.flat[list(cells)] += 1
            complete_duration_sum.flat[list(cells)] += duration_hours
        else:
            censored_count.flat[list(cells)] += 1
            censored_span_sum.flat[list(cells)] += duration_hours

    mean_complete_duration = np.full(shape, np.nan, dtype=np.float64)
    np.divide(
        complete_duration_sum,
        complete_count,
        out=mean_complete_duration,
        where=complete_count > 0,
    )
    mean_censored_span = np.full(shape, np.nan, dtype=np.float64)
    np.divide(censored_span_sum, censored_count, out=mean_censored_span, where=censored_count > 0)

    state_label = str(catalog.get("state_label") or month.state_label)
    _require(state_label == month.state_label, f"catalog state_label 與 SurfaceMonth 不符: {state_label!r} != {month.state_label!r}")
    science_status = str(catalog.get("science_status") or state_label)
    metadata = {
        "curve_rasterization": {
            "coordinate_units": "local metric metres",
            "method": "densify polyline then nearest rectilinear cell",
            "max_step_m": max_step_m,
            "valid_mask_applied_per_snapshot": True,
            "same_snapshot_curve_union": True,
        },
        "lifetime_rule": {
            "complete": "left_censored=false and right_censored=false",
            "censored": "either censor flag true; duration is observed span lower bound",
            "duration_semantics": "first-to-last detection observed span, not physical lifetime; zero hours is snapshot-resolution limited",
            "cross_month_tracks": "not joined; this summary is limited to one catalog",
        },
        "catalog_parameters": dict(catalog.get("parameters") or {}),
    }
    return SpatialSummary(
        domain_id=month.domain_id,
        month=month.month,
        state_label=state_label,
        science_status=science_status,
        snapshot_indices=tuple(indices),
        time_utc_ns=tuple(times),
        lon=np.asarray(month.lon),
        lat=np.asarray(month.lat),
        valid_snapshot_count=valid_count,
        core_count=core_count,
        mean_abs_core_s1_sinv=mean_core,
        curve_pass_count=curve_pass_count,
        curve_pass_rate=curve_rate,
        complete_track_count=complete_count,
        censored_track_count=censored_count,
        mean_complete_track_duration_hours=mean_complete_duration,
        mean_censored_track_observed_span_hours=mean_censored_span,
        metadata=metadata,
    )


__all__ = ["SpatialSummary", "build_spatial_summary"]
