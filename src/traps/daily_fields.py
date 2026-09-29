"""潮汐濾波日平均表層流場產品：``build-daily-fields``。

逐時 OCM 表層流以潮汐（M2/S2/K1/O1）為主，TRAP 文獻（Kunz et al. 2024）使用的是每日、
去潮汐後的快照。本模組由逐時 ``ocm_surface/<domain>/{grid,months/<YYYYMM>}`` 建立
**完全相同磁碟 schema** 的日產品，因此 ``detect-shards``、``track-period``、``aggregate``
不需任何修改即可在其上執行（只需 ``--nominal-time-step-hours 24`` 等時間步長參數）。

方法（``centered_boxcar_mean``）
    * 正式時間軸與 ``track-period`` 相同：依月份時序串接、UTC 穩定排序、重複 UTC 保留
      較晚月份（``sort_and_deduplicate_prefer_last``）；只讀 ``time/u/v/valid``（memory map），
      從不修改來源。
    * 每個 UTC 日的目標時刻 ``T = 當日 00:00 + sample_hour_utc``（預設 12:00）；視窗為 canonical
      逐時樣本中 ``|t - T| <= (window_hours - 1) / 2`` 小時者。25 小時視窗即當日 00:00 至隔日
      00:00（含兩端），可跨月份邊界讀取鄰月。視窗內實際存在的樣本數不足 ``min_hours`` 時，
      該日整日省略（缺口，永不補值）。
    * 格點僅在視窗內「所有實際存在的小時」皆有效時才有效（嚴格規則，使後續微分不混用不同
      樣本集合）；u、v 為存在小時的算術平均（float32），無效處為 NaN。

濾波性質：25 小時等權平均（Godin 式單次滑動平均）對 M2（12.42 h）、S2（12 h）與 K1（23.93 h）、
O1（25.82 h）分潮有很強的抑制（25 h 幾乎恰好整數倍於半日潮與接近日潮週期），但並非理想
低通，殘留的日潮與非潮汐高頻訊號可能不為零；相同資訊會寫入 metadata 的 ``derived_from.filter``。

輸出（每月 ``months/<YYYYMM>``）：``time_utc_ns``（該月各日目標時刻）、``u_surface_mps``、
``v_surface_mps``、``valid_mask_surface``、``qc_flags``（uint16；1 = 靜態無效格點、
2 = 當日無有效表層，靜態無效格點兩位元皆設），加上 ``metadata.json``
（``cache_kind = derived_daily_mean``、``status = ready``、schema 3.0.0）與 ``_SUCCESS``。
grid 目錄逐檔原樣複製，因此既有 ocean-mask sidecar 對日產品仍然有效。
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ocm import CACHE_SCHEMA_VERSION, DERIVED_DAILY_CACHE_KIND, STANDARD_CACHE_KINDS, SurfaceMonth, open_month

FILTER_METHOD = "centered_boxcar_mean"
CELL_VALIDITY_RULE = "valid_in_all_present_hours"
QC_STATIC_INVALID = 1
QC_NO_VALID_SURFACE = 2

HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS

_MONTH_FILES = {
    "time_utc_ns": "time_utc_ns.npy",
    "u": "u_surface_mps.npy",
    "v": "v_surface_mps.npy",
    "valid": "valid_mask_surface.npy",
    "qc_flags": "qc_flags.npy",
}
_KNOWN_MONTH_FILES = (*_MONTH_FILES.values(), "metadata.json", "_SUCCESS")


def _require(condition: bool, message: str) -> None:
    """輸入或參數契約不成立時拋出 ``ValueError``（由 CLI 轉成非零結束碼）。"""

    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    """分塊計算檔案 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """先寫 ``.tmp-`` 檔再原子改名，避免半寫的 ``.npy`` 被誤認完成。"""

    tmp_path = path.with_name(f".tmp-{path.name}")
    with tmp_path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """以臨時檔＋原子改名寫出 JSON。"""

    tmp_path = path.with_name(f".tmp-{path.name}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _date_of(time_ns: int) -> _datetime.date:
    """UTC nanosecond 轉成 UTC 日期。"""

    return (_datetime.datetime(1970, 1, 1) + _datetime.timedelta(microseconds=time_ns // 1000)).date()


def _day_start_ns(day: _datetime.date) -> int:
    """UTC 日期 00:00 的 epoch nanosecond。"""

    return (day - _datetime.date(1970, 1, 1)).days * DAY_NS


def _month_days(month: str) -> list[_datetime.date]:
    """回傳 ``YYYYMM`` 月份內的所有 UTC 日期。"""

    year, mon = int(month[:4]), int(month[4:6])
    day = _datetime.date(year, mon, 1)
    days: list[_datetime.date] = []
    while day.month == mon:
        days.append(day)
        day += _datetime.timedelta(days=1)
    return days


def build_canonical_hourly_axis(month_times: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """建立與 ``track-period`` 相同規則的 canonical 逐時時間軸。

    月份依 ``YYYYMM`` 遞增串接、以 UTC 穩定排序，重複 UTC 只保留最後（較晚月份）一列。
    回傳 ``time_utc_ns``、``month_code``（int）、``index``（月內 time index）三個等長陣列，
    以及被丟棄的重複列數 ``duplicates_removed``。
    """

    months = sorted(month_times)
    times = np.concatenate([np.asarray(month_times[m], dtype=np.int64) for m in months])
    codes = np.concatenate([np.full(np.asarray(month_times[m]).shape[0], int(m), dtype=np.int64) for m in months])
    indices = np.concatenate([np.arange(np.asarray(month_times[m]).shape[0], dtype=np.int64) for m in months])
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    keep = np.ones(sorted_times.shape[0], dtype=bool)
    if sorted_times.shape[0] > 1:
        keep[:-1] = sorted_times[:-1] != sorted_times[1:]
    chosen = order[keep]
    return {
        "time_utc_ns": times[chosen],
        "month_code": codes[chosen],
        "index": indices[chosen],
        "duplicates_removed": np.int64(int(order.shape[0] - chosen.shape[0])),
    }


def _plan_days(
    axis: dict[str, np.ndarray],
    months: Sequence[str],
    *,
    window_hours: float,
    min_hours: int,
    sample_hour_utc: int,
) -> dict[str, dict[str, Any]]:
    """只用時間軸為每個請求月份規劃逐日視窗（尚不讀取任何速度資料）。

    回傳 ``{month: {"days": [{date, target_ns, rows, present}], "omitted": [...]}}``，
    ``rows`` 是 canonical 軸上視窗內的列索引；目標時刻落在該月的日子才屬於該月。
    """

    times = axis["time_utc_ns"]
    half_ns = int(round((float(window_hours) - 1.0) / 2.0 * HOUR_NS))
    plan: dict[str, dict[str, Any]] = {}
    for month in months:
        days: list[dict[str, Any]] = []
        omitted: list[dict[str, Any]] = []
        for day in _month_days(month):
            target = _day_start_ns(day) + int(sample_hour_utc) * HOUR_NS
            lo = int(np.searchsorted(times, target - half_ns, side="left"))
            hi = int(np.searchsorted(times, target + half_ns, side="right"))
            present = hi - lo
            record = {"date": day.isoformat(), "target_ns": target, "rows": (lo, hi), "present": present}
            if present >= min_hours and present > 0:
                days.append(record)
            else:
                omitted.append({"date": day.isoformat(), "present_hours": present})
        plan[month] = {"days": days, "omitted": omitted}
    return plan


def _filter_record(window_hours: float, min_hours: int, sample_hour_utc: int) -> dict[str, Any]:
    """metadata 內濾波參數描述；resume 時逐欄比對。"""

    return {
        "method": FILTER_METHOD,
        "window_hours": window_hours,
        "min_hours": min_hours,
        "sample_hour_utc": sample_hour_utc,
        "cell_validity": CELL_VALIDITY_RULE,
        "note": "25 h 等權平均（Godin 式單次滑動平均）強烈抑制 M2/S2/K1/O1，但不是理想低通。",
    }


def _month_dir_verified(month_dir: Path, filt: dict[str, Any], source_records: list[dict[str, Any]]) -> bool:
    """--resume 判定：_SUCCESS 存在、濾波參數與來源 SHA-256 相符、五個陣列 SHA-256 未變。"""

    if not (month_dir / "_SUCCESS").is_file():
        return False
    try:
        metadata = json.loads((month_dir / "metadata.json").read_text(encoding="utf-8"))
        derived = metadata["derived_from"]
        if metadata.get("cache_kind") != DERIVED_DAILY_CACHE_KIND or derived["filter"] != filt:
            return False
        if derived["source_months"] != source_records:
            return False
        for name, expected in derived["artifact_sha256"].items():
            path = month_dir / name
            if not path.is_file() or _sha256_file(path) != expected:
                return False
        return set(derived["artifact_sha256"]) == set(_MONTH_FILES.values())
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _copy_grid(source_grid: Path, target_grid: Path) -> None:
    """逐檔原樣複製 grid 目錄（含 metadata 與 ocean-mask sidecar）；內容相同則不重寫。"""

    target_grid.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_grid.iterdir()):
        if not source.is_file():
            continue
        target = target_grid / source.name
        if target.is_file() and _sha256_file(target) == _sha256_file(source):
            continue
        tmp_path = target.with_name(f".tmp-{target.name}")
        shutil.copyfile(source, tmp_path)
        os.replace(tmp_path, target)


def _compute_month(
    opened: dict[int, SurfaceMonth],
    axis: dict[str, np.ndarray],
    days: list[dict[str, Any]],
    static_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """對一個輸出月份的所有日子計算平均場；回傳 time、u、v、valid、qc。"""

    ny, nx = static_mask.shape
    count = len(days)
    time_out = np.empty(count, dtype=np.int64)
    u_out = np.full((count, ny, nx), np.nan, dtype=np.float32)
    v_out = np.full((count, ny, nx), np.nan, dtype=np.float32)
    valid_out = np.zeros((count, ny, nx), dtype=bool)
    for position, day in enumerate(days):
        lo, hi = day["rows"]
        sum_u = np.zeros((ny, nx), dtype=np.float64)
        sum_v = np.zeros((ny, nx), dtype=np.float64)
        all_valid = np.ones((ny, nx), dtype=bool)
        for row in range(lo, hi):
            source = opened[int(axis["month_code"][row])]
            index = int(axis["index"][row])
            valid = np.asarray(source._valid[index], dtype=bool)
            all_valid &= valid
            u = np.asarray(source._u[index], dtype=np.float64)
            v = np.asarray(source._v[index], dtype=np.float64)
            # 無效格點的值可能是 NaN；先置 0 再由 all_valid 決定是否輸出，避免污染其他格點。
            sum_u += np.where(valid, u, 0.0)
            sum_v += np.where(valid, v, 0.0)
        present = float(hi - lo)
        valid_day = all_valid & static_mask
        u_out[position] = np.where(valid_day, sum_u / present, np.nan).astype(np.float32)
        v_out[position] = np.where(valid_day, sum_v / present, np.nan).astype(np.float32)
        valid_out[position] = valid_day
        time_out[position] = day["target_ns"]
    qc = np.zeros((count, ny, nx), dtype=np.uint16)
    qc[~valid_out] |= np.uint16(QC_NO_VALID_SURFACE)
    qc[:, ~static_mask] |= np.uint16(QC_STATIC_INVALID)
    return time_out, u_out, v_out, valid_out, qc


def _resolve_months(surface_domain_dir: Path, tokens: Sequence[str] | None) -> list[str]:
    """解析 ``--months``；未指定時取來源 months 目錄下全部 ``YYYYMM``。"""

    from .period import _expand_month_token

    if tokens:
        expanded: list[str] = []
        for token in tokens:
            expanded.extend(_expand_month_token(token))
        return sorted(set(expanded))
    found = sorted(p.name for p in (surface_domain_dir / "months").iterdir() if p.is_dir() and len(p.name) == 6 and p.name.isdigit())
    _require(bool(found), f"來源沒有任何月份目錄: {surface_domain_dir / 'months'}")
    return found


def run_build_daily_fields(args: argparse.Namespace) -> int:
    """建立潮汐濾波日平均表層產品（見模組說明）；回傳 0 表示成功。"""

    window_hours = float(args.window_hours)
    if window_hours == int(window_hours):
        window_hours = int(window_hours)
    min_hours = int(args.min_hours)
    sample_hour = int(args.sample_hour_utc)
    _require(window_hours >= 1, "--window-hours 必須 >= 1")
    _require(1 <= min_hours <= window_hours, "--min-hours 必須介於 1 與 window-hours 之間")
    _require(0 <= sample_hour <= 23, "--sample-hour-utc 必須是 0–23")

    surface_root = Path(args.surface_root)
    output_root = Path(args.output_root)
    domain_id = str(args.domain_id)
    source_domain = surface_root / domain_id
    target_domain = output_root / domain_id
    _require((source_domain / "grid").is_dir(), f"缺少來源 grid 目錄: {source_domain / 'grid'}")
    _require(source_domain.resolve() != target_domain.resolve(), "--output-root 不可與來源 --surface-root 相同")
    months = _resolve_months(source_domain, args.months)

    # 開啟並驗證所有請求月份（fail-closed gate）；只有 mmap，不複製資料。
    opened: dict[int, SurfaceMonth] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    for month in months:
        month_dir = source_domain / "months" / month
        _require(month_dir.is_dir(), f"來源缺少月份目錄: {month_dir}")
        surface = open_month(month_dir, expected_domain_id=domain_id)
        _require(surface.cache_kind in STANDARD_CACHE_KINDS, f"build-daily-fields 只接受逐時 standard 來源，{month} 為 {surface.cache_kind!r}")
        opened[int(month)] = surface
        metadata_path = month_dir / "metadata.json"
        source_meta[month] = {
            "month": month,
            "metadata_path": str(metadata_path),
            "metadata_sha256": _sha256_file(metadata_path),
        }
    first = opened[int(months[0])]
    for month in months[1:]:
        _require(opened[int(month)].spatial_shape == first.spatial_shape, f"月份 {month} 空間 shape 與 {months[0]} 不一致")

    axis = build_canonical_hourly_axis({m: np.asarray(opened[int(m)].time_utc_ns) for m in months})
    plan = _plan_days(axis, months, window_hours=window_hours, min_hours=min_hours, sample_hour_utc=sample_hour)
    filt = _filter_record(window_hours, min_hours, sample_hour)

    month_names = [m for m in months if plan[m]["days"]]
    if not args.resume:
        for month in months:
            month_dir = target_domain / "months" / month
            _require(not (month_dir.is_dir() and any(month_dir.iterdir())), f"輸出月份目錄已有內容，請改用 --resume 或新的 --output-root: {month_dir}")

    _copy_grid(source_domain / "grid", target_domain / "grid")
    static_mask = np.asarray(first.mask_static, dtype=bool)

    built: list[str] = []
    skipped: list[str] = []
    for month in months:
        entry = plan[month]
        if not entry["days"]:
            continue
        rows = [row for day in entry["days"] for row in range(day["rows"][0], day["rows"][1])]
        contributing = sorted({f"{int(axis['month_code'][row]):06d}" for row in rows})
        source_records = [source_meta[m] for m in contributing]
        month_dir = target_domain / "months" / month
        if args.resume and _month_dir_verified(month_dir, filt, source_records):
            skipped.append(month)
            continue
        month_dir.mkdir(parents=True, exist_ok=True)
        for name in _KNOWN_MONTH_FILES:
            candidate = month_dir / name
            if candidate.is_file():
                candidate.unlink()
        for tmp in month_dir.glob(".tmp-*"):
            tmp.unlink()

        time_out, u_out, v_out, valid_out, qc = _compute_month(opened, axis, entry["days"], static_mask)
        arrays = {
            _MONTH_FILES["time_utc_ns"]: time_out,
            _MONTH_FILES["u"]: u_out,
            _MONTH_FILES["v"]: v_out,
            _MONTH_FILES["valid"]: valid_out,
            _MONTH_FILES["qc_flags"]: qc,
        }
        for name, array in arrays.items():
            _atomic_save_npy(month_dir / name, array)

        source_meta_full = opened[int(contributing[0])].metadata
        metadata: dict[str, Any] = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_kind": DERIVED_DAILY_CACHE_KIND,
            "status": "ready",
            "month": month,
            "domain": source_meta_full["domain"],
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in arrays.items()
            },
            "derived_from": {
                "surface_root": str(surface_root),
                "source_months": source_records,
                "filter": filt,
                "present_hours_per_day": {day["date"]: day["present"] for day in entry["days"]},
                "omitted_days": entry["omitted"],
                "duplicates_removed_on_canonical_axis": int(axis["duplicates_removed"]),
                "artifact_sha256": {name: _sha256_file(month_dir / name) for name in arrays},
                "qc_flags_bits": {"1": "static_invalid", "2": "no_valid_surface"},
            },
        }
        if "units" in source_meta_full:
            metadata["units"] = source_meta_full["units"]
        _atomic_write_json(month_dir / "metadata.json", metadata)
        (month_dir / "_SUCCESS").write_text(_sha256_file(month_dir / "metadata.json") + "\n", encoding="utf-8")
        built.append(month)

    summary = {
        "operation": "build-daily-fields",
        "domain_id": domain_id,
        "output_domain_dir": str(target_domain),
        "months_requested": months,
        "months_built": built,
        "months_skipped_resume": skipped,
        "months_without_days": [m for m in months if m not in month_names],
        "days_written": sum(len(plan[m]["days"]) for m in months),
        "days_omitted": sum(len(plan[m]["omitted"]) for m in months),
        "filter": filt,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def register_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """掛載 ``build-daily-fields`` 子命令。"""

    parser = subparsers.add_parser("build-daily-fields", help="由逐時表層流場建立潮汐濾波日平均產品（相同 OCM schema）")
    parser.add_argument("--surface-root", type=Path, required=True, help="來源逐時 ocm_surface 根目錄")
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="輸出根目錄；產生 <root>/<domain>/{grid,months}")
    parser.add_argument("--months", nargs="+", help="YYYYMM 或 YYYYMM-YYYYMM；預設為來源全部月份")
    parser.add_argument("--window-hours", type=float, default=25, help="置中等權視窗長度（小時），預設 25")
    parser.add_argument("--min-hours", type=int, default=23, help="視窗內至少需有的樣本數，否則該日省略")
    parser.add_argument("--sample-hour-utc", type=int, default=12, help="每日目標時刻（UTC 時），預設 12")
    parser.add_argument("--resume", action="store_true", help="略過已驗證（_SUCCESS＋SHA-256）的月份")
    parser.set_defaults(handler=run_build_daily_fields)


__all__ = ["build_canonical_hourly_axis", "register_subcommands", "run_build_daily_fields"]
