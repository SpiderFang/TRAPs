"""OCM schema 3.0 表層月份快取的唯讀存取介面。

本模組只允許 TRAP 分析讀取上游標準化的 ``ocm_surface`` NumPy 快取。所有大型陣列
都以 ``mmap_mode="r"`` 開啟，呼叫端透過 :class:`SurfaceMonth.snapshot` 逐時取得資料，
避免把整個月份複製到記憶體。月份 metadata、靜態 grid 與逐時欄位在開啟時會先做
fail-closed 驗證；這樣資料缺檔、domain 混用、時間軸錯誤或 valid/finite 不一致時，
錯誤會在數值檢出前被揭露。

``mask_static.npy`` 只代表 OCM/SCHISM 網格支撐與 bathymetry 有效性，不等於真實海岸
線遮罩。若呼叫端明確提供 ``analysis_ocean_mask_path``，本模組會再驗證該 sidecar 是
``bool``、shape 相同且為 ``mask_static`` 的子集；未提供時會在 ``analysis_mask_status``
中保留 ``coastline_mask_missing``，不把結果宣稱為已完成海岸線 QC。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = "3.0.0"
"""本專案允許的 OCM surface metadata schema 版本。"""

STANDARD_CACHE_KINDS = frozenset({"standard_month", "standard_partial_month"})
"""可直接進入正式輸入檢查的月份種類；partial 仍會保留其來源缺日資訊。"""

TRIAL_CACHE_KIND = "trial_partial_month"
"""上游以少量日檔產生的 pilot 快取種類，必須明示允許才可讀取。"""

_REQUIRED_MONTH_FILES = {
    "time_utc_ns": "time_utc_ns.npy",
    "u": "u_surface_mps.npy",
    "v": "v_surface_mps.npy",
    "valid": "valid_mask_surface.npy",
    "qc_flags": "qc_flags.npy",
}
_REQUIRED_GRID_FILES = {
    "lon": "lon.npy",
    "lat": "lat.npy",
    "x_m": "x_m.npy",
    "y_m": "y_m.npy",
    "cell_area_m2": "cell_area_m2.npy",
    "mask_static": "mask_static.npy",
}


def _require(condition: bool, message: str) -> None:
    """在輸入資料契約不成立時立即拋出 ``ValueError``。

    這個小型 guard 集中所有 fail-closed 條件的例外型別與訊息格式，讓 CLI、測試和
    上游批次能辨識「輸入不合格」而不是繼續使用部分資料。訊息只描述目前檢查的欄位，
    不會掩蓋原始 path 或 domain 來源。
    """

    if not condition:
        raise ValueError(message)


def _load_npy(path: Path) -> np.ndarray:
    """以唯讀 memory map 開啟一個純 NumPy ``.npy`` 陣列。

    ``allow_pickle=False`` 是固定的安全與可重現性限制；此函式不接受 object array，
    也不會把資料轉成一般 ndarray。回傳值通常是唯讀 ``np.memmap``，因此後續 shape、
    dtype 及必要的有限值掃描可以在不複製整個月份的情況下完成。
    """

    _require(path.is_file(), f"缺少 OCM 快取陣列: {path}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"無法以唯讀 NumPy 格式讀取 {path}: {error}") from error
    _require(isinstance(array, np.ndarray), f"OCM 快取必須是 ndarray: {path}")
    _require(array.dtype != np.dtype("O"), f"OCM 快取不可使用 object dtype: {path}")
    return array


def _sha256_file(path: Path) -> str:
    """以分塊方式計算 sidecar 的 SHA-256，避免把外部檔案整批讀入記憶體。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_metadata_shape(metadata: Mapping[str, Any], name: str) -> tuple[int, ...] | None:
    """取出 metadata 內欄位的 shape 宣告；缺少宣告時交由實際陣列 gate 檢查。

    上游 schema 3.0 正常會保存 ``arrays`` 宣告，但保留缺省分支可讓早期合成快取在
    仍通過實際檔案驗證的前提下被用於單元測試。只要 metadata 明列 shape，本函式就會
    強制核對，避免 metadata 與實際資料悄悄分叉。
    """

    arrays = metadata.get("arrays")
    if arrays is None:
        return None
    _require(isinstance(arrays, Mapping), "metadata.arrays 必須是物件")
    declaration = arrays.get(name)
    if declaration is None:
        return None
    _require(isinstance(declaration, Mapping), f"metadata.arrays.{name} 必須是物件")
    shape = declaration.get("shape")
    if shape is None:
        return None
    _require(isinstance(shape, (list, tuple)), f"metadata.arrays.{name}.shape 必須是清單")
    try:
        return tuple(int(value) for value in shape)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata.arrays.{name}.shape 含非整數欄位") from error


def _validate_valid_slices(valid: np.ndarray, field: np.ndarray, name: str, static_mask: np.ndarray) -> None:
    """逐時檢查 valid/finite/static 關係，避免建立整月暫存布林陣列。

    上游月份可能含數千個時刻；若對整個三維 memmap 一次呼叫 ``isfinite``，即使資料
    沒有被永久載入，也會建立同尺寸的暫存陣列造成不必要的尖峰記憶體。逐時切片仍保留
    fail-closed 語意，同時把峰值限制在單一 `(lat, lon)` 格點層。
    """

    for time_index in range(int(field.shape[0])):
        valid_slice = valid[time_index]
        _require(not np.any(valid_slice & ~static_mask), f"OCM valid 格點不可落在 mask_static 外 (time={time_index})")
        _require(not np.any(valid_slice & ~np.isfinite(field[time_index])), f"OCM valid 格點不可有非有限 {name} (time={time_index})")


@dataclass(frozen=True)
class SurfaceSnapshot:
    """單一 UTC 時刻的表層流場與完整靜態網格參考。

    ``u``、``v``、``valid`` 是 shape ``(lat, lon)`` 的單時刻 view；``x_m``、``y_m``、
    ``cell_area_m2`` 是同 shape 的公尺制靜態網格，``lon`` 和 ``lat`` 是一維座標軸。
    速度與座標不在此物件中複製，呼叫端應在處理完此時刻後釋放 reference。``valid``
    已同時套用逐時 OCM valid、``mask_static`` 以及選用的 analysis ocean sidecar。
    """

    index: int
    time_utc_ns: int
    u: np.ndarray
    v: np.ndarray
    valid: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    cell_area_m2: np.ndarray


@dataclass
class SurfaceMonth:
    """已通過 schema gate 的 OCM 表層月份唯讀快取。

    物件保存的是 memory-map reference，不會在初始化時合併或複製整月速度陣列。公開
    的 ``snapshot(index)`` 是下游數值模組應使用的主要入口；它以一時刻 view 供 TRAP
    梯度、骨架與時序模組處理。``pilot_only`` 若為真，任何輸出都必須沿用
    ``PILOT_ONLY`` 標籤，不能被當成 2024–2025 正式月份。
    """

    month_dir: Path
    metadata: dict[str, Any]
    grid_metadata: dict[str, Any]
    _time_utc_ns: np.ndarray
    _u: np.ndarray
    _v: np.ndarray
    _valid: np.ndarray
    _qc_flags: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    cell_area_m2: np.ndarray
    mask_static: np.ndarray
    analysis_ocean_mask: np.ndarray
    analysis_mask_status: str
    coastline_mask_provenance: dict[str, Any]
    pilot_only: bool = False

    @property
    def domain_id(self) -> str:
        """回傳 metadata 記錄的 OCM flow-domain ID。"""

        return str(self.metadata["domain"]["domain_id"])

    @property
    def month(self) -> str:
        """回傳 ``YYYYMM`` 月份識別字串。"""

        return str(self.metadata["month"])

    @property
    def cache_kind(self) -> str:
        """回傳 upstream cache kind，供報告區分完整、partial 與 pilot。"""

        return str(self.metadata["cache_kind"])

    @property
    def status(self) -> str:
        """回傳上游 metadata 的 ready/trial_ready 狀態。"""

        return str(self.metadata["status"])

    @property
    def state_label(self) -> str:
        """回傳應附加到產物的狀態標籤。

        pilot 月份固定回傳 ``PILOT_ONLY``；正式月份回傳 ``READY``。海岸線 sidecar 的
        缺失另外由 ``analysis_mask_status`` 表達，避免一個標籤同時承擔兩種品質含義。
        """

        return "PILOT_ONLY" if self.pilot_only else "READY"

    @property
    def time_utc_ns(self) -> np.ndarray:
        """回傳嚴格遞增的 UTC nanosecond 時間軸唯讀 view。"""

        return self._time_utc_ns

    @property
    def snapshot_count(self) -> int:
        """回傳月份可逐時讀取的 time 維度長度。"""

        return int(self._time_utc_ns.size)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """回傳固定的 ``(lat, lon)`` 空間 shape。"""

        return tuple(int(value) for value in self._u.shape[1:])

    @property
    def provenance(self) -> dict[str, Any]:
        """回傳小型輸入與海岸遮罩 provenance 摘要，不複製大型陣列。"""

        return {
            "month_dir": str(self.month_dir),
            "domain_id": self.domain_id,
            "month": self.month,
            "cache_schema_version": self.metadata.get("cache_schema_version"),
            "cache_kind": self.cache_kind,
            "state_label": self.state_label,
            "analysis_mask_status": self.analysis_mask_status,
            "coastline_mask": dict(self.coastline_mask_provenance),
        }

    def snapshot(self, index: int) -> SurfaceSnapshot:
        """逐時取得一個表層 snapshot，並合併靜態與動態有效遮罩。

        `index` 是月份陣列的零起算 time index；函式只建立一張該時刻的 bool valid view，
        不會載入其他時刻的速度。逐時有效格點必須同時符合上游 `valid_mask_surface`、
        OCM `mask_static` 與明確指定的 coastline analysis ocean mask；因此 TRAP 微分
        流程若要跨月或跨時刻，必須在外層顯式逐一呼叫本函式並保留每時 coverage。
        """

        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("snapshot index 必須是整數")
        index = int(index)
        if not 0 <= index < self.snapshot_count:
            raise IndexError(f"snapshot index 超出範圍: {index} / {self.snapshot_count}")
        # 逐時 AND 只建立一張小型有效遮罩；速度、時間與靜態陣列仍指向唯讀 mmap。
        valid = np.asarray(self._valid[index], dtype=bool) & self.analysis_ocean_mask
        return SurfaceSnapshot(
            index=index,
            time_utc_ns=int(self._time_utc_ns[index]),
            u=self._u[index],
            v=self._v[index],
            valid=valid,
            x_m=self.x_m,
            y_m=self.y_m,
            lon=self.lon,
            lat=self.lat,
            cell_area_m2=self.cell_area_m2,
        )


def _validate_metadata(
    metadata: dict[str, Any],
    month_dir: Path,
    *,
    expected_domain_id: str | None,
    domain_id_alias: str | None,
    allow_trial_pilot: bool,
) -> tuple[str, bool]:
    """驗證月份 metadata 的 schema、狀態、domain 與月份識別。

    ``trial_partial_month`` 的 ``trial_ready`` 狀態只在呼叫端傳入
    ``allow_trial_pilot=True`` 時接受，並回傳 ``pilot_only=True``。其他未列入上游
    schema 的 cache kind 或 status 一律拒絕，防止未完成快取被誤當成正式輸入。
    """

    _require(metadata.get("cache_schema_version") == CACHE_SCHEMA_VERSION, "OCM cache_schema_version 必須是 3.0.0")
    cache_kind = metadata.get("cache_kind")
    _require(isinstance(cache_kind, str), "OCM metadata 缺少 cache_kind")
    status = metadata.get("status")
    _require(isinstance(status, str), "OCM metadata 缺少 status")
    if cache_kind in STANDARD_CACHE_KINDS:
        _require(status == "ready", f"正式 OCM cache 的 status 必須是 ready，目前為 {status!r}")
        pilot_only = False
    elif cache_kind == TRIAL_CACHE_KIND:
        _require(status in {"trial_ready", "ready"}, "trial_partial_month 的 status 必須是 trial_ready 或 ready")
        _require(allow_trial_pilot, "trial_partial_month 只能在 allow_trial_pilot=True 時讀取")
        pilot_only = True
    else:
        raise ValueError(f"不支援的 OCM cache_kind: {cache_kind!r}")

    domain = metadata.get("domain")
    _require(isinstance(domain, Mapping), "OCM metadata.domain 必須是物件")
    current_domain_id = domain.get("domain_id")
    _require(isinstance(current_domain_id, str) and current_domain_id, "OCM metadata.domain.domain_id 缺失")
    if expected_domain_id is not None and domain_id_alias is not None and expected_domain_id != domain_id_alias:
        raise ValueError("expected_domain_id 與 domain_id 不能同時指定不同值")
    required_domain = expected_domain_id or domain_id_alias
    if required_domain is not None:
        _require(current_domain_id == required_domain, f"OCM domain 不符：需要 {required_domain!r}，實際為 {current_domain_id!r}")

    month = metadata.get("month")
    _require(isinstance(month, str) and len(month) == 6 and month.isdigit(), "OCM metadata.month 必須是 YYYYMM")
    _require(month_dir.name == month, f"月份目錄名稱 {month_dir.name!r} 與 metadata.month {month!r} 不一致")
    return str(current_domain_id), pilot_only


def _validate_grid_metadata(grid_metadata: dict[str, Any], domain_id: str, shape: tuple[int, int]) -> None:
    """核對靜態 grid metadata 的 schema、domain 與宣告 shape。"""

    _require(grid_metadata.get("cache_schema_version") == CACHE_SCHEMA_VERSION, "grid metadata schema 必須是 3.0.0")
    domain = grid_metadata.get("domain")
    _require(isinstance(domain, Mapping), "grid metadata.domain 必須是物件")
    _require(domain.get("domain_id") == domain_id, "surface 與 grid metadata 的 domain_id 不一致")
    declared_shape = grid_metadata.get("grid_shape")
    if declared_shape is not None:
        _require(tuple(int(value) for value in declared_shape) == shape, "grid metadata.grid_shape 與陣列 shape 不一致")


def _load_analysis_mask(
    path_value: str | Path | None,
    static_mask: np.ndarray,
    domain_id: str,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """讀取可選 analysis ocean sidecar 並回傳 mask、狀態與 provenance。

    sidecar 必須是與 ``mask_static`` 同 shape 的 bool `.npy`，且不得把原生靜態無效格點
    重新標成有效。若同資料夾存在同 tag 的 JSON metadata，會一併保存其 path、tag、
    cache schema 與 SHA-256；JSON 缺失時仍拒絕默認正式海岸 QC，而是明列
    ``provided_metadata_missing``。若提供 metadata，還須核對 domain、grid shape 與
    schema，避免不同區域恰好具有相同陣列尺寸時誤用海岸遮罩。
    """

    if path_value is None:
        return static_mask, "coastline_mask_missing", {
            "path": None,
            "metadata_path": None,
            "metadata_sha256": None,
            "semantics": "未指定 analysis_ocean_mask sidecar；結果仍需明示 coastline_mask_missing。",
        }

    path = Path(path_value)
    sidecar = _load_npy(path)
    _require(sidecar.dtype == np.dtype(bool), "analysis ocean mask sidecar dtype 必須是 bool")
    _require(sidecar.shape == static_mask.shape, "analysis ocean mask sidecar shape 必須等於 mask_static")
    _require(not np.any(sidecar & ~static_mask), "analysis ocean mask 必須是 mask_static 的子集")

    tag = path.stem.removeprefix("analysis_ocean_mask_")
    # 上游 sidecar 產生器以 analysis_land_mask_<tag>.json 作為兩張 mask 的共同
    # metadata；同時接受與 ocean 檔同名的 JSON，方便其他明示建立流程保存自己的摘要。
    metadata_candidates = (
        path.with_name(f"analysis_land_mask_{tag}.json"),
        path.with_name(f"analysis_ocean_mask_{tag}.json"),
        path.with_suffix(".json"),
    )
    metadata_path = next((candidate for candidate in metadata_candidates if candidate.is_file()), metadata_candidates[0])
    metadata_payload: dict[str, Any] | None = None
    metadata_sha256: str | None = None
    metadata_status = "provided_metadata_missing"
    if metadata_path.is_file():
        try:
            decoded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"analysis ocean mask metadata 無法讀取: {metadata_path}: {error}") from error
        _require(isinstance(decoded, dict), "analysis ocean mask metadata 必須是 JSON 物件")
        _require(decoded.get("cache_schema_version") == CACHE_SCHEMA_VERSION, "海岸 sidecar schema 與 OCM 不符")
        sidecar_domain = decoded.get("domain")
        _require(isinstance(sidecar_domain, Mapping) and sidecar_domain.get("domain_id") == domain_id, "海岸 sidecar domain_id 與月份不符")
        sidecar_arrays = decoded.get("arrays")
        _require(isinstance(sidecar_arrays, Mapping) and tuple(sidecar_arrays.get("shape_lat_lon", ())) == static_mask.shape, "海岸 sidecar grid shape 與月份不符")
        metadata_payload = decoded
        metadata_sha256 = _sha256_file(metadata_path)
        metadata_status = "provided"

    provenance = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "metadata_path": str(metadata_path) if metadata_path.is_file() else None,
        "metadata_sha256": metadata_sha256,
        "version_tag": tag,
        "metadata": metadata_payload,
    }
    return sidecar, metadata_status, provenance


def open_month(
    month_dir: str | Path,
    *,
    allow_trial_pilot: bool = False,
    expected_domain_id: str | None = None,
    domain_id: str | None = None,
    analysis_ocean_mask_path: str | Path | None = None,
) -> SurfaceMonth:
    """以唯讀 memory map 開啟並驗證一個 OCM surface 月份目錄。

    輸入目錄 layout 固定為 ``ocm_surface/<domain_id>/months/<YYYYMM>``，其同層
    ``grid/`` 必須提供 lon、lat、x_m、y_m、cell_area_m2、mask_static 及 grid metadata。
    月份資料必須符合 schema 3.0.0，並通過 time 嚴格遞增、速度 shape、finite-valid、
    動態遮罩不越過原生 static mask 等 gate。函式不讀取 raw NetCDF、不補 NaN、不將
    整月陣列複製到記憶體。

    ``expected_domain_id`` 是主要 domain gate；``domain_id`` 是相容的別名，兩者同時
    指定時必須相同。``trial_partial_month`` 僅在 ``allow_trial_pilot=True`` 可讀，回傳
    物件的 ``pilot_only`` 與 ``state_label`` 會固定標為 ``PILOT_ONLY``。若未提供
    ``analysis_ocean_mask_path``，物件仍可供工程測試使用，但 ``analysis_mask_status``
    會是 ``coastline_mask_missing``，正式成果不得忽略此 provenance。
    """

    month_dir = Path(month_dir)
    _require(month_dir.is_dir(), f"OCM 月份目錄不存在: {month_dir}")
    metadata_path = month_dir / "metadata.json"
    _require(metadata_path.is_file(), f"缺少 OCM 月份 metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"無法讀取 OCM 月份 metadata: {metadata_path}: {error}") from error
    _require(isinstance(metadata, dict), "OCM 月份 metadata 必須是 JSON 物件")
    resolved_domain_id, pilot_only = _validate_metadata(
        metadata,
        month_dir,
        expected_domain_id=expected_domain_id,
        domain_id_alias=domain_id,
        allow_trial_pilot=allow_trial_pilot,
    )

    grid_dir = month_dir.parent.parent / "grid"
    _require(grid_dir.is_dir(), f"缺少 OCM 靜態 grid 目錄: {grid_dir}")
    grid_metadata_path = grid_dir / "metadata.json"
    _require(grid_metadata_path.is_file(), f"缺少 OCM 靜態 grid metadata: {grid_metadata_path}")
    try:
        grid_metadata = json.loads(grid_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"無法讀取 OCM grid metadata: {grid_metadata_path}: {error}") from error
    _require(isinstance(grid_metadata, dict), "OCM grid metadata 必須是 JSON 物件")

    month_arrays = {name: _load_npy(month_dir / filename) for name, filename in _REQUIRED_MONTH_FILES.items()}
    grid_arrays = {name: _load_npy(grid_dir / filename) for name, filename in _REQUIRED_GRID_FILES.items()}
    time_utc_ns = month_arrays["time_utc_ns"]
    u = month_arrays["u"]
    v = month_arrays["v"]
    dynamic_valid = month_arrays["valid"]
    qc_flags = month_arrays["qc_flags"]
    lon = grid_arrays["lon"]
    lat = grid_arrays["lat"]
    x_m = grid_arrays["x_m"]
    y_m = grid_arrays["y_m"]
    cell_area_m2 = grid_arrays["cell_area_m2"]
    static_mask = grid_arrays["mask_static"]

    _require(u.dtype == np.dtype("float32") and v.dtype == np.dtype("float32"), "OCM u/v 必須是 float32")
    _require(time_utc_ns.dtype == np.dtype("int64"), "OCM time_utc_ns 必須是 int64")
    _require(dynamic_valid.dtype == np.dtype(bool), "OCM valid_mask_surface 必須是 bool")
    _require(qc_flags.dtype == np.dtype("uint16"), "OCM qc_flags 必須是 uint16")
    _require(static_mask.dtype == np.dtype(bool), "OCM mask_static 必須是 bool")
    _require(u.ndim == 3 and v.shape == u.shape, "OCM u/v 必須是相同的 (time, lat, lon) 三維陣列")
    _require(dynamic_valid.shape == u.shape and qc_flags.shape == u.shape, "OCM valid/qc shape 必須與 u/v 完全一致")
    _require(time_utc_ns.shape == (u.shape[0],), "OCM time 軸長度必須等於 u/v time 維度")
    spatial_shape = tuple(int(value) for value in u.shape[1:])
    _require(static_mask.shape == spatial_shape, "OCM mask_static shape 必須等於 u/v 空間 shape")
    _require(lon.ndim == 1 and lat.ndim == 1, "OCM lon/lat 必須是一維座標軸")
    _require(lon.shape == (spatial_shape[1],) and lat.shape == (spatial_shape[0],), "OCM lon/lat 軸長度不符")
    _require(x_m.shape == y_m.shape == cell_area_m2.shape == spatial_shape, "OCM x_m/y_m/cell_area_m2 shape 不符")
    _validate_grid_metadata(grid_metadata, resolved_domain_id, spatial_shape)

    for name, array in (("u", u), ("v", v), ("lon", lon), ("lat", lat), ("x_m", x_m), ("y_m", y_m), ("cell_area_m2", cell_area_m2)):
        _require(np.issubdtype(array.dtype, np.floating), f"OCM {name} 必須是浮點陣列")
    _require(np.all(np.diff(time_utc_ns) > 0), "OCM time_utc_ns 必須嚴格遞增")
    _require(np.all(np.isfinite(lon)) and np.all(np.diff(lon) > 0), "OCM lon 必須有限且嚴格遞增")
    _require(np.all(np.isfinite(lat)) and np.all(np.diff(lat) > 0), "OCM lat 必須有限且嚴格遞增")
    _require(np.all(np.isfinite(x_m)) and np.all(np.isfinite(y_m)), "OCM x_m/y_m 不可含非有限值")
    _require(np.all(np.isfinite(cell_area_m2[static_mask])) and np.all(cell_area_m2[static_mask] > 0), "OCM 有效格點 cell_area_m2 必須為正且有限")
    _validate_valid_slices(dynamic_valid, u, "u", static_mask)
    _validate_valid_slices(dynamic_valid, v, "v", static_mask)

    for field_name, field in (("time_utc_ns", time_utc_ns), ("u_surface_mps", u), ("v_surface_mps", v), ("valid_mask_surface", dynamic_valid)):
        declared_shape = _array_metadata_shape(metadata, f"{field_name}.npy")
        if declared_shape is not None:
            _require(declared_shape == tuple(int(value) for value in field.shape), f"metadata {field_name} shape 與實際陣列不一致")

    analysis_mask, analysis_mask_status, coastline_provenance = _load_analysis_mask(
        analysis_ocean_mask_path, static_mask, resolved_domain_id
    )
    return SurfaceMonth(
        month_dir=month_dir,
        metadata=metadata,
        grid_metadata=grid_metadata,
        _time_utc_ns=time_utc_ns,
        _u=u,
        _v=v,
        _valid=dynamic_valid,
        _qc_flags=qc_flags,
        x_m=x_m,
        y_m=y_m,
        lon=lon,
        lat=lat,
        cell_area_m2=cell_area_m2,
        mask_static=static_mask,
        analysis_ocean_mask=analysis_mask,
        analysis_mask_status=analysis_mask_status,
        coastline_mask_provenance=coastline_provenance,
        pilot_only=pilot_only,
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "STANDARD_CACHE_KINDS",
    "TRIAL_CACHE_KIND",
    "SurfaceMonth",
    "SurfaceSnapshot",
    "open_month",
]
