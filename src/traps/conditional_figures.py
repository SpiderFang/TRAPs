"""條件資料介面、fail-closed gate 與文獻借鏡成果圖。

本模組處理三種需要額外 provenance 的表格：

* 潮相表：每個有效研究站點與 catalog UTC snapshot 必須各有一筆，並明示
  ``flood``／``ebb`` 及 ``spring``／``neap``。
* 短期軌跡表：每個點保存個體 ID、UTC、經緯度、來源類型與 QC；
  ``observed_drifter`` 只有在 metadata 明確確認獨立性後才可標成獨立觀測，
  ``model_particle`` 永遠只作模型內部比較。
* 海廢紀錄表：每筆紀錄保存 UTC 起訖期間、位置、數量、採樣 effort 與 QC；
  疊圖只呈現時間／空間共現，不宣稱污染來源。

所有輸入都是 UTF-8 CSV，且必須有同 stem JSON metadata。metadata 鎖定 CSV
SHA-256、UTC timezone、來源、schema、單位與獨立性狀態。任何缺檔、缺欄、時區不明、
SHA 不符、重複鍵、QC 失敗或與 catalog 時間窗無法對齊的情況，都回傳
``DATA_REQUIRED`` 或 ``QC_BLOCKED`` 報告；圖函式只接受 ``READY``，不建立空白或
補值的科學圖。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np

from .coastline import CoastlineGeometry, draw_geographic_coastline
from .colorbar_layout import align_colorbars_to_axes
from .figures import _load_matplotlib
from .ocm import SurfaceMonth
from .spatial import SpatialSummary

GateStatus: TypeAlias = Literal["READY", "DATA_REQUIRED", "QC_BLOCKED"]
"""條件資料 gate 的固定狀態集合。"""

SOURCE_KINDS = frozenset({"observed_drifter", "model_particle"})
"""短期軌跡允許的來源類型；其他名稱不可進入圖面。"""

_TABLE_SCHEMAS = {
    "tidal_phase": "traps.conditional.tidal_phase.v1",
    "trajectory": "traps.conditional.trajectory.v1",
    "marine_debris": "traps.conditional.marine_debris.v1",
}
_REQUIRED_UNITS = {
    "tidal_phase": ("time_utc_ns",),
    "trajectory": ("time_utc_ns", "lon", "lat"),
    "marine_debris": ("start_time_utc_ns", "end_time_utc_ns", "lon", "lat", "count", "effort"),
}


class ConditionalGateError(ValueError):
    """圖面因條件資料 gate 未通過而拒絕產生時的明確例外。

    ``report.status`` 保留 ``DATA_REQUIRED`` 或 ``QC_BLOCKED``，呼叫端可以直接將
    ``report.to_dict()`` 寫入 `figure_availability.json`，不用從例外字串猜測原因。
    """

    def __init__(self, report: ConditionalGateReport) -> None:
        self.report = report
        details = "; ".join((*report.missing, *report.issues)) or "未提供 gate 詳情"
        super().__init__(f"{report.kind} gate={report.status}: {details}")


def _require(condition: bool, message: str) -> None:
    """在內部計算契約不成立時停止，避免把壞資料畫成科學結果。"""

    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    """分塊計算輸入 CSV 的 SHA-256，供 metadata provenance 核對。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_path(csv_path: Path) -> Path:
    """回傳與 CSV 同 stem 的 JSON metadata 路徑。"""

    return csv_path.with_suffix(".json")


def _is_nonempty_source(value: Any) -> bool:
    """判斷 metadata source 是否為可追溯的非空字串或物件。"""

    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    return False


def _report(
    kind: str,
    status: GateStatus,
    *,
    csv_path: Path | None = None,
    metadata_path: Path | None = None,
    table: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    missing: Sequence[str] = (),
    issues: Sequence[str] = (),
) -> ConditionalGateReport:
    """集中建立不可變 gate 報告，保證所有 loader 回傳同一種狀態結構。"""

    return ConditionalGateReport(
        kind=kind,
        status=status,
        csv_path=str(csv_path) if csv_path is not None else None,
        metadata_path=str(metadata_path) if metadata_path is not None else None,
        table=table,
        metadata=dict(metadata) if metadata is not None else None,
        missing=tuple(str(item) for item in missing),
        issues=tuple(str(item) for item in issues),
    )


@dataclass(frozen=True, slots=True)
class ConditionalGateReport:
    """三類條件表格的 gate 結果。

    ``status='READY'`` 時 ``table`` 是對應的 typed table；其他狀態一定沒有 table。
    ``missing`` 描述需要補齊的資料或 metadata，``issues`` 描述已存在但不合格的
    欄位／來源／QC。這個區分讓後續 CLI 能分別報告「資料尚未提供」和「資料已提供但
    不能用」。
    """

    kind: str
    status: GateStatus
    csv_path: str | None
    metadata_path: str | None
    table: Any | None
    metadata: Mapping[str, Any] | None
    missing: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """確保 blocked 報告不會偷偷攜帶可繪圖 table。"""

        _require(self.status in {"READY", "DATA_REQUIRED", "QC_BLOCKED"}, "未知條件資料 gate 狀態")
        if self.status != "READY":
            _require(self.table is None, "blocked gate 不可攜帶 table")
        if self.status == "READY":
            _require(self.table is not None, "READY gate 必須攜帶 table")

    @property
    def ready(self) -> bool:
        """回傳是否可進入成果圖函式。"""

        return self.status == "READY"

    def require_ready(self) -> Any:
        """若 gate 未通過則以保留狀態的例外停止；通過時回傳 typed table。"""

        if not self.ready:
            raise ConditionalGateError(self)
        return self.table

    def to_dict(self) -> dict[str, Any]:
        """回傳可直接寫入 availability JSON 的摘要，不展開表格資料。"""

        return {
            "kind": self.kind,
            "status": self.status,
            "csv_path": self.csv_path,
            "metadata_path": self.metadata_path,
            "missing": list(self.missing),
            "issues": list(self.issues),
            "metadata": dict(self.metadata) if self.metadata is not None else None,
        }


@dataclass(frozen=True, slots=True)
class TidalPhaseRecord:
    """一個站點在一個 UTC snapshot 的潮相與大小潮標籤。"""

    site_id: str
    time_utc_ns: int
    phase: Literal["flood", "ebb"]
    spring_neap: Literal["spring", "neap"]
    qc_pass: bool


@dataclass(frozen=True, slots=True)
class TidalPhaseTable:
    """通過完整站點×UTC 覆蓋檢查的潮相表。"""

    records: tuple[TidalPhaseRecord, ...]
    metadata: Mapping[str, Any]
    csv_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """一個短期軌跡點，位置是 WGS84 經緯度。"""

    trajectory_id: str
    time_utc_ns: int
    lon: float
    lat: float
    source_kind: Literal["observed_drifter", "model_particle"]
    qc_pass: bool


@dataclass(frozen=True, slots=True)
class TrajectoryTable:
    """通過來源類型與觀測獨立性 metadata 檢查的軌跡表。"""

    records: tuple[TrajectoryRecord, ...]
    metadata: Mapping[str, Any]
    csv_sha256: str
    source_kind: Literal["observed_drifter", "model_particle"]


@dataclass(frozen=True, slots=True)
class MarineDebrisRecord:
    """一筆具有 UTC 起訖期間、位置、數量與採樣 effort 的海廢紀錄。"""

    record_id: str
    start_time_utc_ns: int
    end_time_utc_ns: int
    lon: float
    lat: float
    count: float
    effort: float
    qc_pass: bool


@dataclass(frozen=True, slots=True)
class MarineDebrisTable:
    """通過時間、數量、effort 與 QC 檢查的海廢紀錄表。"""

    records: tuple[MarineDebrisRecord, ...]
    metadata: Mapping[str, Any]
    csv_sha256: str


@dataclass(frozen=True, slots=True)
class TidalOccurrence:
    """依潮相與大小潮分層的站點 TRAP 出現率分子／分母。"""

    site_id: str
    phase: Literal["flood", "ebb"]
    spring_neap: Literal["spring", "neap"]
    numerator: int
    denominator: int

    @property
    def occurrence_rate(self) -> float | None:
        """分母為零時保留 ``None``，不把無資料誤畫成 0%。"""

        return self.numerator / self.denominator if self.denominator else None

    def to_record(self) -> dict[str, Any]:
        """轉為 JSON/CSV 友善的分層統計列。"""

        return {
            "site_id": self.site_id,
            "phase": self.phase,
            "spring_neap": self.spring_neap,
            "occurrence_numerator": self.numerator,
            "occurrence_denominator": self.denominator,
            "occurrence_rate": self.occurrence_rate,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryPointComparison:
    """軌跡點與同時刻最近合格 TRAP 曲線的距離結果。"""

    trajectory_id: str
    time_utc_ns: int
    lon: float
    lat: float
    distance_to_trap_m: float | None


@dataclass(frozen=True, slots=True)
class TrajectoryComparison:
    """一條短期軌跡的 TRAP 距離、停留與相對運動摘要。"""

    trajectory_id: str
    source_kind: Literal["observed_drifter", "model_particle"]
    points: tuple[TrajectoryPointComparison, ...]
    dwell_hours_within_threshold: float
    mean_distance_to_trap_m: float | None
    mean_distance_change_rate_mps: float | None
    mean_observed_speed_mps: float | None
    observed_span_hours: float


def _parse_bool(value: Any, field: str) -> bool:
    """解析 CSV 的 true/false 欄位，拒絕空字串與模糊數值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field} 必須是 true 或 false")


def _parse_int(value: Any, field: str) -> int:
    """解析 UTC nanosecond 或識別欄位的整數值，拒絕 bool。"""

    if isinstance(value, bool) or value is None or str(value).strip() == "":
        raise ValueError(f"{field} 必須是整數")
    parsed = int(str(value).strip())
    return parsed


def _parse_float(value: Any, field: str, *, minimum: float | None = None, strict_positive: bool = False) -> float:
    """解析有限浮點欄位並套用位置、數量或 effort 的下界。"""

    if value is None or str(value).strip() == "":
        raise ValueError(f"{field} 不可為空")
    parsed = float(str(value).strip())
    if not math.isfinite(parsed):
        raise ValueError(f"{field} 必須是有限數值")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field} 不可小於 {minimum}")
    if strict_positive and parsed <= 0.0:
        raise ValueError(f"{field} 必須大於 0")
    return parsed


def _metadata_gate(
    csv_path: Path | None,
    *,
    kind: str,
    required_columns: Sequence[str],
    rows: list[dict[str, str]] | None,
) -> tuple[ConditionalGateReport | None, dict[str, Any] | None]:
    """執行共用檔案、CSV 欄位、metadata、SHA、UTC 與 provenance gate。"""

    if csv_path is None:
        return _report(kind, "DATA_REQUIRED", missing=(f"{kind}_csv", f"{kind}_metadata_json")), None
    if not csv_path.is_file():
        return _report(kind, "DATA_REQUIRED", csv_path=csv_path, missing=(str(csv_path),)), None
    metadata_path = _metadata_path(csv_path)
    if not metadata_path.is_file():
        return _report(kind, "DATA_REQUIRED", csv_path=csv_path, metadata_path=metadata_path, missing=(str(metadata_path),)), None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return _report(kind, "QC_BLOCKED", csv_path=csv_path, metadata_path=metadata_path, issues=(f"metadata JSON 無法讀取: {error}",)), None
    issues: list[str] = []
    if not isinstance(metadata, Mapping):
        issues.append("metadata 頂層必須是 JSON object")
        metadata = {}
    expected_schema = _TABLE_SCHEMAS[kind]
    if metadata.get("schema_version") != expected_schema:
        issues.append(f"schema_version 必須是 {expected_schema}")
    if metadata.get("table_kind") != kind:
        issues.append(f"table_kind 必須是 {kind}")
    if metadata.get("timezone") != "UTC":
        issues.append("timezone 必須明確是 UTC")
    if not _is_nonempty_source(metadata.get("source")):
        issues.append("metadata.source 不可為空")
    independence = metadata.get("independence_status")
    if not isinstance(independence, str) or not independence.strip():
        issues.append("metadata.independence_status 不可為空")
    units = metadata.get("units")
    if not isinstance(units, Mapping) or not units:
        issues.append("metadata.units 必須是非空 object")
    else:
        for unit_field in _REQUIRED_UNITS[kind]:
            unit_value = units.get(unit_field)
            if not isinstance(unit_value, str) or not unit_value.strip():
                issues.append(f"metadata.units 缺少 {unit_field}")
    try:
        actual_sha = _sha256_file(csv_path)
    except OSError as error:
        actual_sha = ""
        issues.append(f"CSV SHA-256 無法計算: {error}")
    if metadata.get("csv_sha256") != actual_sha:
        issues.append("metadata.csv_sha256 與 CSV 實際 SHA-256 不符")
    if rows is None:
        issues.append("CSV rows 尚未讀取")
    else:
        if not rows:
            return _report(kind, "DATA_REQUIRED", csv_path=csv_path, metadata_path=metadata_path, metadata=metadata, missing=("CSV data rows",)), None
        available_columns = set(rows[0])
        missing_columns = [column for column in required_columns if column not in available_columns]
        if missing_columns:
            issues.append(f"CSV 缺少欄位: {', '.join(missing_columns)}")
    if issues:
        return _report(kind, "QC_BLOCKED", csv_path=csv_path, metadata_path=metadata_path, metadata=metadata, issues=issues), None
    return None, dict(metadata)


def _read_csv(path: Path | None, *, kind: str, required_columns: Sequence[str]) -> tuple[ConditionalGateReport | None, list[dict[str, str]] | None, dict[str, Any] | None]:
    """以 UTF-8 讀取 CSV 並先執行共用 metadata gate。"""

    if path is None:
        report, metadata = _metadata_gate(None, kind=kind, required_columns=required_columns, rows=None)
        return report, None, metadata
    path = Path(path)
    if not path.is_file():
        report, metadata = _metadata_gate(path, kind=kind, required_columns=required_columns, rows=None)
        return report, None, metadata
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return _report(kind, "QC_BLOCKED", csv_path=path, metadata_path=_metadata_path(path), issues=(f"UTF-8 CSV 無法讀取: {error}",)), None, None
    report, metadata = _metadata_gate(path, kind=kind, required_columns=required_columns, rows=rows)
    return report, rows, metadata


def _parse_catalog_sites_times(catalog: Mapping[str, Any]) -> tuple[set[str], set[int], dict[int, bool]]:
    """取出 catalog 中有統計面積的站點與 coverage 通過的 UTC snapshot。"""

    raw_sites = catalog.get("site_statistics")
    raw_snapshots = catalog.get("snapshots")
    _require(isinstance(raw_sites, Sequence) and isinstance(raw_snapshots, Sequence), "catalog 缺少 site_statistics 或 snapshots")
    sites: set[str] = set()
    for raw in raw_sites:
        _require(isinstance(raw, Mapping), "catalog.site_statistics 每項必須是 object")
        if float(raw.get("site_area_m2") or 0.0) > 0.0:
            site_id = str(raw.get("study_site_id") or raw.get("site_id") or "")
            _require(site_id, "有效站點缺少 study_site_id")
            sites.add(site_id)
    times: set[int] = set()
    coverage: dict[int, bool] = {}
    for raw in raw_snapshots:
        _require(isinstance(raw, Mapping), "catalog.snapshots 每項必須是 object")
        time = _parse_int(raw.get("time_utc_ns"), "catalog time_utc_ns")
        coverage_value = raw.get("coverage_pass")
        _require(isinstance(coverage_value, bool), "catalog snapshot coverage_pass 必須是 bool")
        coverage[time] = coverage_value
        if coverage_value:
            times.add(time)
    return sites, times, coverage


def read_tidal_phase_table(path: str | Path | None, catalog: Mapping[str, Any]) -> ConditionalGateReport:
    """讀取並驗證潮相 CSV 與同 stem metadata。

    gate 需要完整覆蓋 catalog 中所有有效站點與 ``coverage_pass=True`` 的 UTC 時刻，
    即 expected keys 為 ``(site_id, time_utc_ns)`` 的笛卡兒積。重複、額外 key、任何
    ``qc_pass=false``、未知 phase／大小潮標籤或零分母分層都回傳 ``QC_BLOCKED``；檔案
    或 metadata 缺失回傳 ``DATA_REQUIRED``。
    """

    required = ("site_id", "time_utc_ns", "phase", "spring_neap", "qc_pass")
    report, rows, metadata = _read_csv(Path(path) if path is not None else None, kind="tidal_phase", required_columns=required)
    if report is not None:
        return report
    assert rows is not None and metadata is not None
    issues: list[str] = []
    missing_requirements: list[str] = []
    try:
        records: list[TidalPhaseRecord] = []
        keys: set[tuple[str, int]] = set()
        for raw in rows:
            site_id = str(raw["site_id"]).strip()
            time = _parse_int(raw["time_utc_ns"], "time_utc_ns")
            phase = str(raw["phase"]).strip().lower()
            spring_neap = str(raw["spring_neap"]).strip().lower()
            qc_pass = _parse_bool(raw["qc_pass"], "qc_pass")
            if not site_id or phase not in {"flood", "ebb"} or spring_neap not in {"spring", "neap"}:
                raise ValueError("site_id、phase 或 spring_neap 值不合法")
            key = (site_id, time)
            if key in keys:
                raise ValueError(f"潮相 key 重複: {key}")
            keys.add(key)
            records.append(TidalPhaseRecord(site_id, time, phase, spring_neap, qc_pass))
        sites, valid_times, _coverage = _parse_catalog_sites_times(catalog)
        expected = {(site_id, time) for site_id in sites for time in valid_times}
        missing = expected - keys
        extra = keys - expected
        if missing:
            missing_requirements.append(f"潮相未完整涵蓋有效站點×UTC: {sorted(missing)[:5]}")
        if extra:
            issues.append(f"潮相含 catalog 外的站點或時間: {sorted(extra)[:5]}")
        if any(not record.qc_pass for record in records):
            issues.append("潮相表含 qc_pass=false")
        if not expected:
            issues.append("catalog 沒有有效站點×UTC 分母")
    except (KeyError, TypeError, ValueError) as error:
        issues.append(str(error))
        records = []
    if missing_requirements and not issues:
        return _report("tidal_phase", "DATA_REQUIRED", csv_path=Path(path) if path is not None else None, metadata_path=_metadata_path(Path(path)) if path is not None else None, metadata=metadata, missing=missing_requirements)
    if issues:
        return _report("tidal_phase", "QC_BLOCKED", csv_path=Path(path) if path is not None else None, metadata_path=_metadata_path(Path(path)) if path is not None else None, metadata=metadata, issues=issues)
    table = TidalPhaseTable(tuple(records), metadata, _sha256_file(Path(path)))
    return _report("tidal_phase", "READY", csv_path=Path(path), metadata_path=_metadata_path(Path(path)), table=table, metadata=metadata)


def read_trajectory_table(
    path: str | Path | None,
    *,
    catalog: Mapping[str, Any] | None = None,
    expected_source_kind: str | None = None,
) -> ConditionalGateReport:
    """讀取軌跡 CSV，驗證來源類型與獨立性，並可檢查 catalog UTC 覆蓋。

    一個 CSV 必須只有一種 ``source_kind``，以免同一張圖把獨立觀測和模型粒子混稱。
    ``observed_drifter`` 必須有 metadata ``independence_status=confirmed_independent``；
    ``model_particle`` 只能使用 ``model_internal`` 或 ``not_independent``。傳入 catalog
    時至少要有一個共同 UTC 時刻，否則回傳 ``DATA_REQUIRED``。
    """

    required = ("trajectory_id", "time_utc_ns", "lon", "lat", "source_kind", "qc_pass")
    report, rows, metadata = _read_csv(Path(path) if path is not None else None, kind="trajectory", required_columns=required)
    if report is not None:
        return report
    assert rows is not None and metadata is not None
    issues: list[str] = []
    records: list[TrajectoryRecord] = []
    try:
        keys: set[tuple[str, int]] = set()
        for raw in rows:
            trajectory_id = str(raw["trajectory_id"]).strip()
            time = _parse_int(raw["time_utc_ns"], "time_utc_ns")
            lon = _parse_float(raw["lon"], "lon", minimum=-180.0)
            lat = _parse_float(raw["lat"], "lat", minimum=-90.0)
            _require(lon <= 180.0 and lat <= 90.0, "經緯度超出 WGS84 範圍")
            source_kind = str(raw["source_kind"]).strip()
            _require(source_kind in SOURCE_KINDS, f"source_kind 不支援: {source_kind}")
            qc_pass = _parse_bool(raw["qc_pass"], "qc_pass")
            _require(trajectory_id, "trajectory_id 不可為空")
            key = (trajectory_id, time)
            _require(key not in keys, f"trajectory key 重複: {key}")
            keys.add(key)
            records.append(TrajectoryRecord(trajectory_id, time, lon, lat, source_kind, qc_pass))
        source_kinds = {record.source_kind for record in records}
        _require(len(source_kinds) == 1, "單一軌跡 CSV 不可混合 observed_drifter 與 model_particle")
        source_kind = next(iter(source_kinds))
        if expected_source_kind is not None:
            _require(source_kind == expected_source_kind, f"source_kind 與呼叫端預期不符: {source_kind}")
        independence = str(metadata.get("independence_status"))
        if source_kind == "observed_drifter" and independence != "confirmed_independent":
            raise ValueError("observed_drifter 必須由 metadata 確認 independence_status=confirmed_independent")
        if source_kind == "model_particle" and independence not in {"model_internal", "not_independent"}:
            raise ValueError("model_particle 只能標示 model_internal 或 not_independent")
        if any(not record.qc_pass for record in records):
            raise ValueError("軌跡表含 qc_pass=false")
        if catalog is not None:
            _sites, catalog_times, _coverage = _parse_catalog_sites_times(catalog)
            if not ({record.time_utc_ns for record in records} & catalog_times):
                return _report("trajectory", "DATA_REQUIRED", csv_path=Path(path), metadata_path=_metadata_path(Path(path)), metadata=metadata, missing=("trajectory/catalog UTC overlap",))
    except (KeyError, TypeError, ValueError) as error:
        issues.append(str(error))
        source_kind = None
    if issues:
        return _report("trajectory", "QC_BLOCKED", csv_path=Path(path), metadata_path=_metadata_path(Path(path)), metadata=metadata, issues=issues)
    assert source_kind is not None
    table = TrajectoryTable(tuple(records), metadata, _sha256_file(Path(path)), source_kind)  # type: ignore[arg-type]
    return _report("trajectory", "READY", csv_path=Path(path), metadata_path=_metadata_path(Path(path)), table=table, metadata=metadata)


def read_marine_debris_table(path: str | Path | None, *, catalog: Mapping[str, Any] | None = None) -> ConditionalGateReport:
    """讀取海廢 CSV，驗證 UTC 期間、位置、數量、effort、重複 ID 與時間覆蓋。

    ``count`` 是觀測數量，``effort`` 必須為正值且其實際單位由 metadata.units 鎖定。
    這些紀錄可與 TRAP 空間通過率並列，但本模組不把共現轉為污染來源或因果判斷。
    """

    required = ("record_id", "start_time_utc_ns", "end_time_utc_ns", "lon", "lat", "count", "effort", "qc_pass")
    report, rows, metadata = _read_csv(Path(path) if path is not None else None, kind="marine_debris", required_columns=required)
    if report is not None:
        return report
    assert rows is not None and metadata is not None
    issues: list[str] = []
    records: list[MarineDebrisRecord] = []
    try:
        ids: set[str] = set()
        for raw in rows:
            record_id = str(raw["record_id"]).strip()
            start = _parse_int(raw["start_time_utc_ns"], "start_time_utc_ns")
            end = _parse_int(raw["end_time_utc_ns"], "end_time_utc_ns")
            _require(end >= start, "海廢 UTC end 必須不早於 start")
            lon = _parse_float(raw["lon"], "lon", minimum=-180.0)
            lat = _parse_float(raw["lat"], "lat", minimum=-90.0)
            _require(lon <= 180.0 and lat <= 90.0, "海廢經緯度超出 WGS84 範圍")
            count = _parse_float(raw["count"], "count", minimum=0.0)
            effort = _parse_float(raw["effort"], "effort", strict_positive=True)
            qc_pass = _parse_bool(raw["qc_pass"], "qc_pass")
            _require(record_id, "record_id 不可為空")
            _require(record_id not in ids, f"record_id 重複: {record_id}")
            ids.add(record_id)
            records.append(MarineDebrisRecord(record_id, start, end, lon, lat, count, effort, qc_pass))
        if any(not record.qc_pass for record in records):
            raise ValueError("海廢表含 qc_pass=false")
        if catalog is not None:
            _sites, catalog_times, _coverage = _parse_catalog_sites_times(catalog)
            if not catalog_times:
                raise ValueError("catalog 沒有可用 UTC 分母")
            catalog_start = min(catalog_times)
            catalog_end = max(catalog_times)
            if not any(record.end_time_utc_ns >= catalog_start and record.start_time_utc_ns <= catalog_end for record in records):
                return _report("marine_debris", "DATA_REQUIRED", csv_path=Path(path), metadata_path=_metadata_path(Path(path)), metadata=metadata, missing=("marine_debris/catalog UTC overlap",))
    except (KeyError, TypeError, ValueError) as error:
        issues.append(str(error))
    if issues:
        return _report("marine_debris", "QC_BLOCKED", csv_path=Path(path), metadata_path=_metadata_path(Path(path)), metadata=metadata, issues=issues)
    table = MarineDebrisTable(tuple(records), metadata, _sha256_file(Path(path)))
    return _report("marine_debris", "READY", csv_path=Path(path), metadata_path=_metadata_path(Path(path)), table=table, metadata=metadata)


def _catalog_snapshot_times(catalog: Mapping[str, Any]) -> tuple[int, ...]:
    """讀取 catalog snapshot UTC 序列，供條件圖對齊與標題使用。"""

    raw = catalog.get("snapshots")
    _require(isinstance(raw, Sequence), "catalog.snapshots 必須是清單")
    return tuple(_parse_int(item.get("time_utc_ns"), "catalog time_utc_ns") for item in raw if isinstance(item, Mapping))


def summarize_tidal_occurrence(report: ConditionalGateReport, catalog: Mapping[str, Any]) -> tuple[TidalOccurrence, ...]:
    """把 READY 潮相表依站點、漲退潮與大小潮計算 TRAP 出現率。

    分母只使用 catalog snapshot 的 ``coverage_pass=True`` 時刻；分子是該時刻是否有
    至少一筆合格 detection 的 ``study_site_ids`` 包含該站。這裡計算的是模型結構
    出現率，不是海廢觀測頻率。
    """

    table = report.require_ready()
    _require(isinstance(table, TidalPhaseTable), "report 必須是 TidalPhaseTable")
    sites, valid_times, coverage = _parse_catalog_sites_times(catalog)
    detections = catalog.get("detections")
    _require(isinstance(detections, Sequence), "catalog.detections 必須是清單")
    occurrence: set[tuple[str, int]] = set()
    for raw in detections:
        if not isinstance(raw, Mapping) or raw.get("is_qualified") is not True:
            continue
        time = _parse_int(raw.get("time_utc_ns"), "detection time_utc_ns")
        for site_id in raw.get("study_site_ids", ()) or ():
            occurrence.add((str(site_id), time))
    results: list[TidalOccurrence] = []
    for phase in ("flood", "ebb"):
        for spring_neap in ("spring", "neap"):
            for site_id in sorted(sites):
                records = [record for record in table.records if record.site_id == site_id and record.phase == phase and record.spring_neap == spring_neap and record.time_utc_ns in valid_times]
                denominator = sum(1 for record in records if coverage.get(record.time_utc_ns, False))
                numerator = sum(1 for record in records if (site_id, record.time_utc_ns) in occurrence)
                if denominator <= 0:
                    blocked = _report(
                        "tidal_phase",
                        "QC_BLOCKED",
                        csv_path=Path(report.csv_path) if report.csv_path else None,
                        metadata_path=Path(report.metadata_path) if report.metadata_path else None,
                        metadata=report.metadata,
                        issues=(f"潮相分層零分母: {site_id}/{phase}/{spring_neap}",),
                    )
                    raise ConditionalGateError(blocked)
                results.append(TidalOccurrence(site_id, phase, spring_neap, numerator, denominator))
    return tuple(results)


def _rectilinear_axes(month: SurfaceMonth) -> tuple[np.ndarray, np.ndarray]:
    """取出 SurfaceMonth 的軸對齊公尺座標軸，供曲線轉為經緯度。"""

    x = np.asarray(month.x_m, dtype=np.float64)
    y = np.asarray(month.y_m, dtype=np.float64)
    _require(x.ndim == y.ndim == 2, "SurfaceMonth x_m/y_m 必須是二維")
    x_axis = x[0]
    y_axis = y[:, 0]
    _require(np.allclose(x, x_axis[None, :], atol=1.0e-6, rtol=0.0), "x_m 必須是軸對齊格網")
    _require(np.allclose(y, y_axis[:, None], atol=1.0e-6, rtol=0.0), "y_m 必須是軸對齊格網")
    _require(np.all(np.diff(x_axis) > 0.0) and np.all(np.diff(y_axis) > 0.0), "x_m/y_m 必須嚴格遞增")
    return x_axis, y_axis


def _catalog_curves_lonlat(catalog: Mapping[str, Any], month: SurfaceMonth) -> dict[int, list[np.ndarray]]:
    """將 catalog 合格 TRAP 曲線由局部公尺座標轉成 WGS84 曲線。"""

    x_axis, y_axis = _rectilinear_axes(month)
    curves: dict[int, list[np.ndarray]] = defaultdict(list)
    raw = catalog.get("detections")
    _require(isinstance(raw, Sequence), "catalog.detections 必須是清單")
    for row in raw:
        if not isinstance(row, Mapping) or row.get("is_qualified") is not True:
            continue
        time = _parse_int(row.get("time_utc_ns"), "detection time_utc_ns")
        points = np.asarray(row.get("curve_points_m"), dtype=np.float64)
        _require(points.ndim == 2 and points.shape[1] == 2 and points.shape[0] >= 1, "catalog curve_points_m shape 不符")
        _require(np.all(points[:, 0] >= x_axis[0]) and np.all(points[:, 0] <= x_axis[-1]) and np.all(points[:, 1] >= y_axis[0]) and np.all(points[:, 1] <= y_axis[-1]), "catalog TRAP 曲線超出 SurfaceMonth 公尺網格")
        curves[time].append(np.column_stack((np.interp(points[:, 0], x_axis, month.lon), np.interp(points[:, 1], y_axis, month.lat))))
    return curves


def _point_to_polyline_distance_m(lon: float, lat: float, curve: np.ndarray) -> float:
    """以局部等距近似計算經緯度點到曲線的公尺距離。"""

    lat_scale = 111_320.0
    lon_scale = lat_scale * max(math.cos(math.radians(lat)), 1.0e-6)
    points = np.column_stack(((curve[:, 0] - lon) * lon_scale, (curve[:, 1] - lat) * lat_scale))
    if points.shape[0] == 1:
        return float(np.hypot(points[0, 0], points[0, 1]))
    start = points[:-1]
    vectors = points[1:] - start
    lengths = np.sum(vectors * vectors, axis=1)
    offsets = -start
    fraction = np.divide(np.sum(offsets * vectors, axis=1), lengths, out=np.zeros_like(lengths), where=lengths > 0.0)
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = start + fraction[:, None] * vectors
    return float(np.min(np.hypot(nearest[:, 0], nearest[:, 1])))


def summarize_trajectory_trap_comparison(
    report: ConditionalGateReport,
    catalog: Mapping[str, Any],
    month: SurfaceMonth,
    *,
    distance_threshold_m: float = 12_500.0,
    max_interval_hours: float = 6.0,
) -> tuple[TrajectoryComparison, ...]:
    """計算短期軌跡到同時刻 TRAP 的距離、停留與相對距離變化。

    ``dwell_hours_within_threshold`` 只累加相鄰兩點都在距離門檻內且時間間隔合理的
    觀測區段；``mean_distance_change_rate_mps`` 是到最近 TRAP 曲線距離的變化率，
    不是完整流體相對速度。對模型粒子而言所有指標只代表共用 OCM 強迫下的內部比較。
    """

    _require(np.isfinite(distance_threshold_m) and distance_threshold_m > 0.0, "distance_threshold_m 必須是正有限值")
    _require(np.isfinite(max_interval_hours) and max_interval_hours > 0.0, "max_interval_hours 必須是正有限值")
    table = report.require_ready()
    _require(isinstance(table, TrajectoryTable), "report 必須是 TrajectoryTable")
    curves_by_time = _catalog_curves_lonlat(catalog, month)
    catalog_times = set(_catalog_snapshot_times(catalog))
    _require(bool({record.time_utc_ns for record in table.records} & catalog_times), "軌跡與 catalog 沒有共同 UTC 時刻")
    by_id: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in table.records:
        by_id[record.trajectory_id].append(record)
    results: list[TrajectoryComparison] = []
    for trajectory_id, records in sorted(by_id.items()):
        ordered = sorted(records, key=lambda record: record.time_utc_ns)
        points: list[TrajectoryPointComparison] = []
        for record in ordered:
            curves = curves_by_time.get(record.time_utc_ns, ())
            distance = min((_point_to_polyline_distance_m(record.lon, record.lat, curve) for curve in curves), default=None)
            points.append(TrajectoryPointComparison(record.trajectory_id, record.time_utc_ns, record.lon, record.lat, distance))
        dwell_seconds = 0.0
        distance_changes: list[float] = []
        speeds: list[float] = []
        for first, second in pairwise(points):
            interval_seconds = (second.time_utc_ns - first.time_utc_ns) / 1.0e9
            if interval_seconds <= 0.0 or interval_seconds > max_interval_hours * 3_600.0:
                continue
            if first.distance_to_trap_m is not None and second.distance_to_trap_m is not None:
                if first.distance_to_trap_m <= distance_threshold_m and second.distance_to_trap_m <= distance_threshold_m:
                    dwell_seconds += interval_seconds
                distance_changes.append(abs(second.distance_to_trap_m - first.distance_to_trap_m) / interval_seconds)
            delta_lon = (second.lon - first.lon) * 111_320.0 * max(math.cos(math.radians((first.lat + second.lat) / 2.0)), 1.0e-6)
            delta_lat = (second.lat - first.lat) * 111_320.0
            speeds.append(float(math.hypot(delta_lon, delta_lat) / interval_seconds))
        finite_distances = [point.distance_to_trap_m for point in points if point.distance_to_trap_m is not None]
        results.append(
            TrajectoryComparison(
                trajectory_id=trajectory_id,
                source_kind=table.source_kind,
                points=tuple(points),
                dwell_hours_within_threshold=dwell_seconds / 3_600.0,
                mean_distance_to_trap_m=float(np.mean(finite_distances)) if finite_distances else None,
                mean_distance_change_rate_mps=float(np.mean(distance_changes)) if distance_changes else None,
                mean_observed_speed_mps=float(np.mean(speeds)) if speeds else None,
                observed_span_hours=(ordered[-1].time_utc_ns - ordered[0].time_utc_ns) / 3_600_000_000_000.0,
            )
        )
    return tuple(results)


def _state_label(catalog: Mapping[str, Any]) -> str:
    """取得圖面應顯示的科學狀態。"""

    return str(catalog.get("science_status") or catalog.get("state_label") or "STATUS_UNSPECIFIED")


def render_tidal_occurrence_figure(
    report: ConditionalGateReport,
    catalog: Mapping[str, Any],
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """輸出漲／退潮與大／小潮四組站點 TRAP 出現率（分子／分母）圖。"""

    occurrence = summarize_tidal_occurrence(report, catalog)
    _, plt = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sites = sorted({row.site_id for row in occurrence})
    # 四組站點的分子／分母標籤與旋轉站名需要固定留白，避免被下一列標題遮住。
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharey=True, constrained_layout=False)
    figure.subplots_adjust(left=0.09, right=0.97, bottom=0.15, top=0.85, wspace=0.20, hspace=0.55)
    for axis, phase in zip(axes.flat, ("flood", "ebb", "flood", "ebb"), strict=True):
        spring_neap = "spring" if axis in (axes.flat[0], axes.flat[1]) else "neap"
        rows = [row for row in occurrence if row.phase == phase and row.spring_neap == spring_neap]
        values = [100.0 * row.occurrence_rate if row.occurrence_rate is not None else np.nan for row in rows]
        positions = np.arange(len(sites), dtype=float)
        by_site = {row.site_id: row for row in rows}
        plotted = [values[[row.site_id for row in rows].index(site)] if site in by_site else np.nan for site in sites]
        axis.bar(positions, plotted, color="#377eb8" if phase == "flood" else "#e66101")
        for index, site in enumerate(sites):
            row = by_site.get(site)
            if row is not None:
                label_height = min(float(plotted[index]) + 1.0, 105.0) if np.isfinite(plotted[index]) else 1.0
                axis.text(index, label_height, f"{row.numerator}/{row.denominator}", ha="center", va="bottom", fontsize=8)
        axis.set_title(f"{phase.title()} | {spring_neap.title()}")
        axis.set_xticks(positions)
        axis.set_xticklabels(sites, rotation=25, ha="right")
        axis.set_ylim(0.0, 110.0)
        axis.set_ylabel("TRAP occurrence (%)")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"Conditional TRAP occurrence by tidal phase | {_state_label(catalog)}\nmodel structure occurrence; numerator/denominator shown")
    figure.text(0.5, 0.04, "Denominator = coverage-pass UTC snapshots; no debris concentration or source attribution", ha="center", fontsize=8, color="#7f0000")
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


def _render_trajectory_figure(
    report: ConditionalGateReport,
    catalog: Mapping[str, Any],
    month: SurfaceMonth,
    output_path: str | Path,
    *,
    distance_threshold_m: float,
    max_interval_hours: float,
    coastline: CoastlineGeometry | None,
    dpi: int,
) -> Path:
    """繪製軌跡／TRAP 共用圖，並以呼叫端的最大間隔計算停留摘要。

    最大間隔與 :func:`summarize_trajectory_trap_comparison` 使用同一公務參數，
    避免 CLI 覆寫門檻後圖與 JSON 摘要採用不同的相鄰點判準。
    """

    table = report.require_ready()
    _require(isinstance(table, TrajectoryTable), "report 必須是 TrajectoryTable")
    comparisons = summarize_trajectory_trap_comparison(
        report, catalog, month,
        distance_threshold_m=distance_threshold_m,
        max_interval_hours=max_interval_hours,
    )
    curves_by_time = _catalog_curves_lonlat(catalog, month)
    _, plt = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), constrained_layout=False)
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.23, top=0.78, wspace=0.25)
    for time, curves in curves_by_time.items():
        for curve_index, curve in enumerate(curves):
            axes[0].plot(curve[:, 0], curve[:, 1], color="#555555", linewidth=0.8, alpha=0.45, label="qualified TRAP curve" if time == next(iter(curves_by_time)) and curve_index == 0 else None)
    for comparison in comparisons:
        points = comparison.points
        axes[0].plot([point.lon for point in points], [point.lat for point in points], marker="o", markersize=3.5, linewidth=1.5, label=f"{comparison.trajectory_id} ({comparison.source_kind})")
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].set_title("Short-term trajectory and TRAP geometry")
    axes[0].grid(alpha=0.2)
    coastline_note = "coastline overlay: CRS84 outline only"
    if coastline is not None:
        # 先讓 TRAP 與軌跡決定既有圖面範圍，再把同一視窗傳給 coastline
        # clipping；draw_geographic_coastline 不會 autoscale，因此域外 polygon
        # 不會改變短期軌跡圖的 flow-domain／資料視窗。
        x_limits = axes[0].get_xlim()
        y_limits = axes[0].get_ylim()
        draw_geographic_coastline(
            axes[0],
            coastline,
            bbox=(min(x_limits), min(y_limits), max(x_limits), max(y_limits)),
            color="#263238",
            linewidth=0.7,
            zorder=6.0,
        )
    else:
        coastline_note = "coastline overlay unavailable"
    source_note = "independent observed drifter" if table.source_kind == "observed_drifter" else "model-particle internal comparison; not independent"
    axes[0].text(0.02, 0.02, source_note, transform=axes[0].transAxes, fontsize=8, color="#7f0000")
    for comparison in comparisons:
        finite = [(point.time_utc_ns - comparison.points[0].time_utc_ns) / 3_600_000_000_000.0 for point in comparison.points if point.distance_to_trap_m is not None]
        distances = [point.distance_to_trap_m for point in comparison.points if point.distance_to_trap_m is not None]
        if finite:
            axes[1].plot(finite, distances, marker="o", linewidth=1.2, label=comparison.trajectory_id)
    axes[1].axhline(distance_threshold_m, color="#e66101", linestyle="--", linewidth=1.0, label=f"dwell threshold ({distance_threshold_m / 1000.0:g} km)")
    axes[1].set_xlabel("Hours since first trajectory point (UTC)")
    axes[1].set_ylabel("Distance to nearest TRAP curve (m)")
    axes[1].set_title("TRAP distance and dwell threshold")
    axes[1].grid(alpha=0.2)
    handles, _labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(fontsize=8)
    has_distance = any(point.distance_to_trap_m is not None for comparison in comparisons for point in comparison.points)
    if not has_distance:
        axes[1].text(0.5, 0.5, "No matched qualified TRAP at trajectory times", transform=axes[1].transAxes, ha="center", va="center")
    state = _state_label(catalog)
    figure.suptitle(f"Trajectory/TRAP comparison | {state}\n{source_note}")
    figure.text(0.5, 0.05, f"Distance change rate and dwell are model-geometry diagnostics; they are not pollution-source evidence | {coastline_note}", ha="center", fontsize=8, color="#7f0000")
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


def render_trajectory_trap_comparison_figure(
    report: ConditionalGateReport,
    catalog: Mapping[str, Any],
    month: SurfaceMonth,
    output_path: str | Path,
    *,
    distance_threshold_m: float = 12_500.0,
    max_interval_hours: float = 6.0,
    coastline: CoastlineGeometry | None = None,
    dpi: int = 150,
) -> Path:
    """輸出短期軌跡/TRAP 對照圖；停留摘要使用指定的最大觀測間隔。

    ``coastline`` 僅疊加已驗證 CRS84 輪廓，不參與軌跡距離、停留或任何 TRAP
    統計；未提供時維持原有無岸線圖面。
    """

    return _render_trajectory_figure(report, catalog, month, output_path, distance_threshold_m=distance_threshold_m, max_interval_hours=max_interval_hours, coastline=coastline, dpi=dpi)


def render_drifter_trap_comparison_figure(
    report: ConditionalGateReport,
    catalog: Mapping[str, Any],
    month: SurfaceMonth,
    output_path: str | Path,
    *,
    distance_threshold_m: float = 12_500.0,
    max_interval_hours: float = 6.0,
    coastline: CoastlineGeometry | None = None,
    dpi: int = 150,
) -> Path:
    """輸出已確認獨立性的 observed drifter/TRAP 對照圖。

    此 wrapper 額外拒絕 model particle，讓報告腳本不會誤把共用 OCM 強迫的模型軌跡
    放進「獨立觀測驗證」圖。模型粒子應呼叫一般的
    :func:`render_trajectory_trap_comparison_figure`，並保留 internal 標籤。
    """

    table = report.require_ready()
    _require(isinstance(table, TrajectoryTable), "report 必須是 TrajectoryTable")
    if table.source_kind != "observed_drifter":
        raise ConditionalGateError(_report("trajectory", "QC_BLOCKED", issues=("drifter figure 需要 observed_drifter",)))
    return _render_trajectory_figure(report, catalog, month, output_path, distance_threshold_m=distance_threshold_m, max_interval_hours=max_interval_hours, coastline=coastline, dpi=dpi)


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """以單調鏈建立局部公尺座標的凸包；共線或重複點不代表可量測面積。"""

    unique = sorted({(float(x), float(y)) for x, y in points})
    _require(len(unique) >= 3, "粒子群至少需三個不同位置")

    def cross(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float]) -> float:
        """回傳三點的有向面積，用來剔除不構成凸包轉角的點。"""

        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    _require(hull.shape[0] >= 3, "粒子群共線，無法計算可解讀的面積變化")
    return hull


def _polygon_area_m2(hull: np.ndarray) -> float:
    """以 shoelace 公式計算凸包面積，輸入為同一局部切平面的公尺座標。"""

    return float(abs(np.dot(hull[:, 0], np.roll(hull[:, 1], -1)) - np.dot(hull[:, 1], np.roll(hull[:, 0], -1))) / 2.0)


def summarize_particle_patch_deformation(
    report: ConditionalGateReport,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """計算同一模型粒子群在共同 UTC 時刻的凸包面積變化。

    CSV 每個 ``trajectory_id`` 是一顆模型粒子；metadata ``patch_id`` 必須明示
    整張表屬於同一釋放群。只採 catalog 與粒子共同時刻、所有時刻皆存在的粒子，
    且至少需三顆非共線粒子和兩個時刻。經緯度以全群平均位置建立局部切平面，
    僅接受最大跨度 100 km 的粒子群。此量是凸包面積比的有限時間形變代理，
    不是 FTLE、流場變形梯度或獨立觀測驗證。
    """

    table = report.require_ready()
    _require(isinstance(table, TrajectoryTable), "report 必須是 TrajectoryTable")
    if table.source_kind != "model_particle":
        raise ConditionalGateError(_report("short_term_deformation", "DATA_REQUIRED", missing=("model_particle trajectory table",)))
    patch_id = table.metadata.get("patch_id")
    if not isinstance(patch_id, str) or not patch_id.strip():
        raise ConditionalGateError(_report("short_term_deformation", "QC_BLOCKED", issues=("metadata.patch_id 必須明示單一模型粒子釋放群",)))
    catalog_times = set(_catalog_snapshot_times(catalog))
    times = sorted({row.time_utc_ns for row in table.records} & catalog_times)
    if len(times) < 2:
        raise ConditionalGateError(_report("short_term_deformation", "DATA_REQUIRED", missing=("至少兩個與 catalog 相同的粒子群 UTC 時刻",)))
    by_time: dict[int, dict[str, TrajectoryRecord]] = defaultdict(dict)
    for row in table.records:
        if row.time_utc_ns in catalog_times:
            by_time[row.time_utc_ns][row.trajectory_id] = row
    particle_ids = sorted(set.intersection(*(set(by_time[time]) for time in times)))
    if len(particle_ids) < 3:
        raise ConditionalGateError(_report("short_term_deformation", "DATA_REQUIRED", missing=("至少三顆於共同 UTC 時刻持續追蹤的模型粒子",)))
    all_rows = [by_time[time][particle_id] for time in times for particle_id in particle_ids]
    reference_lon = float(np.mean([row.lon for row in all_rows]))
    reference_lat = float(np.mean([row.lat for row in all_rows]))
    x_scale = 111_320.0 * math.cos(math.radians(reference_lat))
    y_scale = 111_320.0
    # 同一局部切平面須涵蓋整段軌跡，而非只檢查每個時刻的群體寬度；
    # 否則整群漂移很遠時，個別時刻雖緊密仍會超出平面近似的適用範圍。
    all_x = np.asarray([(row.lon - reference_lon) * x_scale for row in all_rows], dtype=np.float64)
    all_y = np.asarray([(row.lat - reference_lat) * y_scale for row in all_rows], dtype=np.float64)
    if np.ptp(all_x) > 100_000.0 or np.ptp(all_y) > 100_000.0:
        raise ConditionalGateError(_report("short_term_deformation", "QC_BLOCKED", issues=("整段粒子群跨度超過局部切平面 100 km 限制",)))
    points_by_time: list[np.ndarray] = []
    hulls_by_time: list[np.ndarray] = []
    areas: list[float] = []
    for time in times:
        points = np.asarray(
            [[(by_time[time][particle_id].lon - reference_lon) * x_scale,
              (by_time[time][particle_id].lat - reference_lat) * y_scale]
             for particle_id in particle_ids],
            dtype=np.float64,
        )
        if np.ptp(points[:, 0]) > 100_000.0 or np.ptp(points[:, 1]) > 100_000.0:
            raise ConditionalGateError(_report("short_term_deformation", "QC_BLOCKED", issues=("粒子群跨度超過局部切平面 100 km 限制",)))
        try:
            hull = _convex_hull_xy(points)
        except ValueError as error:
            raise ConditionalGateError(_report("short_term_deformation", "QC_BLOCKED", issues=(str(error),))) from error
        area = _polygon_area_m2(hull)
        if area <= 0.0:
            raise ConditionalGateError(_report("short_term_deformation", "QC_BLOCKED", issues=("粒子群凸包面積為零",)))
        points_by_time.append(points)
        hulls_by_time.append(hull)
        areas.append(area)
    return {
        "patch_id": patch_id,
        "source_kind": table.source_kind,
        "science_status": _state_label(catalog),
        "time_utc_ns": times,
        "particle_ids": particle_ids,
        "particle_count": len(particle_ids),
        "reference_lon": reference_lon,
        "reference_lat": reference_lat,
        "local_projection": "equirectangular tangent-plane approximation, <=100 km extent",
        "area_m2": areas,
        "final_to_initial_area_ratio": areas[-1] / areas[0],
        "interpretation": "model_particle convex-hull area change proxy; not FTLE, deformation gradient, or independent validation",
        "_points_by_time": points_by_time,
        "_hulls_by_time": hulls_by_time,
    }


def render_particle_patch_deformation_figure(
    report: ConditionalGateReport,
    catalog: Mapping[str, Any],
    month: SurfaceMonth,
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """繪製模型粒子群形狀、同時刻 TRAP 曲線與凸包面積比。

    TRAP 曲線從已核對的 ``SurfaceMonth`` 轉為 WGS84，再放入粒子群局部公尺軸；
    缺少同時刻合格曲線時明示，避免把不同時刻結構硬疊為同時觀測。
    """

    summary = summarize_particle_patch_deformation(report, catalog)
    _, plt = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=False)
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.20, top=0.80, wspace=0.28)
    curves_by_time = _catalog_curves_lonlat(catalog, month)
    x_scale = 111_320.0 * math.cos(math.radians(summary["reference_lat"]))
    # 地圖視窗依實際粒子群大小決定；TRAP 曲線僅畫近群體的片段，避免遠端
    # 合格曲線讓粒子 patch 被壓成不可辨識的一條線。
    all_patch_points_km = np.vstack(summary["_points_by_time"]) / 1_000.0
    x_min, y_min = np.min(all_patch_points_km, axis=0)
    x_max, y_max = np.max(all_patch_points_km, axis=0)
    margin_km = max(0.5, 0.25 * max(float(x_max - x_min), float(y_max - y_min)))
    view_x = (float(x_min - margin_km), float(x_max + margin_km))
    view_y = (float(y_min - margin_km), float(y_max + margin_km))
    curve_label_used = False
    for index, color, label in ((0, "#377eb8", "first common UTC"), (-1, "#e66101", "last common UTC")):
        time = summary["time_utc_ns"][index]
        for curve in curves_by_time.get(time, ()):
            curve_x_km = (curve[:, 0] - summary["reference_lon"]) * x_scale / 1_000.0
            curve_y_km = (curve[:, 1] - summary["reference_lat"]) * 111_320.0 / 1_000.0
            near = (curve_x_km >= view_x[0]) & (curve_x_km <= view_x[1]) & (curve_y_km >= view_y[0]) & (curve_y_km <= view_y[1])
            if np.count_nonzero(near) >= 2:
                axes[0].plot(np.where(near, curve_x_km, np.nan), np.where(near, curve_y_km, np.nan), color="0.45", linewidth=0.8, alpha=0.45, label="same-UTC TRAP" if not curve_label_used else None)
                curve_label_used = True
        points = summary["_points_by_time"][index]
        hull = summary["_hulls_by_time"][index]
        closed = np.vstack((hull, hull[0])) / 1_000.0
        axes[0].plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5, label=label)
        axes[0].scatter(points[:, 0] / 1_000.0, points[:, 1] / 1_000.0, color=color, s=22)
    axes[0].set_xlabel("Local x (km)")
    axes[0].set_ylabel("Local y (km)")
    axes[0].set_title("Same-patch model particle positions")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(*view_x)
    axes[0].set_ylim(*view_y)
    if not curve_label_used:
        axes[0].text(0.5, 0.03, "No nearby qualified TRAP at selected UTC times", transform=axes[0].transAxes, ha="center", fontsize=8, color="#7f0000")
    axes[0].legend(fontsize=8)
    ages = [(time - summary["time_utc_ns"][0]) / 3_600_000_000_000.0 for time in summary["time_utc_ns"]]
    ratios = np.asarray(summary["area_m2"], dtype=np.float64) / float(summary["area_m2"][0])
    axes[1].plot(ages, ratios, marker="o", color="#984ea3")
    axes[1].axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Hours since first common UTC")
    axes[1].set_ylabel("Convex-hull area / initial area")
    axes[1].set_title(f"Area ratio = {summary['final_to_initial_area_ratio']:.3g}")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle(f"Model particle patch deformation proxy | {summary['science_status']} | patch {summary['patch_id']}")
    figure.text(0.5, 0.05, "Shared forcing: model-internal comparison only; convex-hull area ratio is not FTLE or a deformation tensor", ha="center", fontsize=8, color="#7f0000")
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


def render_debris_trap_overlay_figure(
    report: ConditionalGateReport,
    catalog: Mapping[str, Any],
    spatial_summary: SpatialSummary,
    output_path: str | Path,
    *,
    coastline: CoastlineGeometry | None = None,
    dpi: int = 150,
) -> Path:
    """輸出海廢紀錄與 TRAP 空間通過率的時間／空間並列圖。

    背景是已計算的 ``SpatialSummary.curve_pass_rate_percent``，點大小呈現 count、
    顏色呈現 effort-normalized count。只有紀錄期間與 catalog snapshot 時間窗至少
    相交時才繪圖；圖說明確將結果限制為共現描述，不稱污染來源或因果關係。
    """

    table = report.require_ready()
    _require(isinstance(table, MarineDebrisTable), "report 必須是 MarineDebrisTable")
    _require(isinstance(spatial_summary, SpatialSummary), "spatial_summary 必須是 SpatialSummary")
    source = catalog.get("source")
    if isinstance(source, Mapping):
        if source.get("domain_id") is not None and str(source["domain_id"]) != spatial_summary.domain_id:
            blocked = _report("marine_debris", "QC_BLOCKED", issues=("catalog domain 與 spatial_summary 不符",))
            raise ConditionalGateError(blocked)
        if source.get("month") is not None and str(source["month"]) != spatial_summary.month:
            blocked = _report("marine_debris", "QC_BLOCKED", issues=("catalog month 與 spatial_summary 不符",))
            raise ConditionalGateError(blocked)
    times = _catalog_snapshot_times(catalog)
    _require(times, "catalog 沒有 snapshot 時間")
    start_time, end_time = min(times), max(times)
    records = [record for record in table.records if record.end_time_utc_ns >= start_time and record.start_time_utc_ns <= end_time]
    if not records:
        raise ConditionalGateError(_report("marine_debris", "DATA_REQUIRED", missing=("marine_debris/catalog UTC overlap",)))
    # 時間相交不足以支持空間共現；只保留落在已驗證 TRAP 地理網格內的紀錄。
    # 若所有紀錄都在 flow domain 之外，拒絕產圖，避免座標軸為了遠端觀測而
    # 放大到使模型背景縮成角落。
    records = [
        record for record in records
        if float(spatial_summary.lon[0]) <= record.lon <= float(spatial_summary.lon[-1])
        and float(spatial_summary.lat[0]) <= record.lat <= float(spatial_summary.lat[-1])
    ]
    if not records:
        raise ConditionalGateError(_report("marine_debris", "DATA_REQUIRED", missing=("marine_debris records within TRAP flow domain",)))
    _, plt = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.5, 7.0), constrained_layout=False)
    figure.subplots_adjust(left=0.10, right=0.88, bottom=0.16, top=0.88)
    field = np.ma.masked_invalid(np.asarray(spatial_summary.curve_pass_rate_percent, dtype=np.float64))
    mesh = axis.pcolormesh(spatial_summary.lon, spatial_summary.lat, field, shading="auto", cmap="YlOrRd", vmin=0.0, vmax=100.0)
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label("TRAP curve passage (%)")
    counts = np.asarray([record.count for record in records], dtype=np.float64)
    efforts = np.asarray([record.effort for record in records], dtype=np.float64)
    rate = counts / efforts
    sizes = 35.0 + 180.0 * counts / max(float(np.max(counts)), 1.0)
    scatter = axis.scatter([record.lon for record in records], [record.lat for record in records], s=sizes, c=rate, cmap="viridis", vmin=0.0, vmax=max(float(np.max(rate)), 1.0), edgecolor="black", linewidth=0.5, label="marine debris record")
    debris_colorbar = figure.colorbar(scatter, ax=axis, pad=0.08)
    units = table.metadata.get("units") if isinstance(table.metadata.get("units"), Mapping) else {}
    debris_colorbar.set_label(f"{units.get('count', 'count')} / {units.get('effort', 'effort')}")
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_title(f"Marine debris records with TRAP spatial passage | {_state_label(catalog)}")
    axis.grid(alpha=0.2)
    coastline_note = "coastline overlay: CRS84 outline only"
    if coastline is not None:
        # 海廢點已先限制在 TRAP flow-domain 地理範圍；沿用相同 bbox 畫 outline，
        # 不讓域外海岸線改變背景格網的軸範圍。
        draw_geographic_coastline(
            axis,
            coastline,
            bbox=(float(np.min(spatial_summary.lon)), float(np.min(spatial_summary.lat)), float(np.max(spatial_summary.lon)), float(np.max(spatial_summary.lat))),
            color="#263238",
            linewidth=0.75,
            zorder=6.0,
        )
    else:
        coastline_note = "coastline overlay unavailable"
    axis.legend(loc="best", fontsize=8)
    figure.text(0.5, 0.05, f"Temporal/spatial co-occurrence only; no pollution-source or causal attribution | {coastline_note}", ha="center", fontsize=8, color="#7f0000")
    # 兩條色條都描述同一張海廢/TRAP 空間主圖；在 scatter、legend 與 aspect
    # 設定完成後量測主圖的實際 axes 框，確保色條上下邊界與主圖一致。只調整
    # 色條 axes 的 y0/height，保留其水平位置、寬度、色階與資料 gate。
    align_colorbars_to_axes(
        figure,
        (
            (axis, colorbar.ax),
            (axis, debris_colorbar.ax),
        ),
    )
    figure.savefig(output, dpi=dpi, format="png")
    plt.close(figure)
    return output


__all__ = [
    "ConditionalGateError",
    "ConditionalGateReport",
    "GateStatus",
    "MarineDebrisRecord",
    "MarineDebrisTable",
    "TidalOccurrence",
    "TidalPhaseRecord",
    "TidalPhaseTable",
    "TrajectoryComparison",
    "TrajectoryPointComparison",
    "TrajectoryRecord",
    "TrajectoryTable",
    "read_marine_debris_table",
    "read_tidal_phase_table",
    "read_trajectory_table",
    "render_debris_trap_overlay_figure",
    "render_drifter_trap_comparison_figure",
    "render_particle_patch_deformation_figure",
    "render_tidal_occurrence_figure",
    "render_trajectory_trap_comparison_figure",
    "summarize_particle_patch_deformation",
    "summarize_tidal_occurrence",
    "summarize_trajectory_trap_comparison",
]
