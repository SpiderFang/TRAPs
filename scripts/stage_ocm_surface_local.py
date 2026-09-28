#!/usr/bin/env python3
"""CLI：把 SERVER NAS 上的 OCM surface 前處理資料，同步（staging）到本機 SSD。

背景：NAS（``/data``）走 NFS，頻寬約 100 Mb/s，且內容可能隨時被清理；TRAPs 在
SERVER 上執行大量平行 detect/track 工作時，必須先把要用到的 domain × month 檔案
複製一份到本機 SSD（``/home/mustlab/Workspace/traps-local-data``），依照與 NAS 相
同的 layout 排列，之後所有計算都只讀本機複本，不再直接打 NAS。

本檔案刻意只用標準函式庫，且語法與 stdlib API 都相容 Python 3.8：部署腳本
（``run_traps_server.sh`` 的 STAGE 模式）可能在 ``.venv``（Python 3.11+）建立
「之前」，用 SERVER 系統内建的 python3（3.8）先執行本腳本，把資料準備好。

用法範例（於 repo 根目錄執行）::

    python3 scripts/stage_ocm_surface_local.py \\
        --dest-root /home/mustlab/Workspace/traps-local-data \\
        --domain-id northeast_taiwan_common_cache_v3 \\
        --months 202401-202412 \\
        --jobs 8

不指定 ``--source-root`` 時預設為 SERVER 上標準的
``/data/OCM-Preprocessed-Data/preprocessed/ocm_surface``；不指定
``--domain-id`` 時預設處理四個研究區域 domain；不指定 ``--months`` 時預設
``202401-202512``。

每個 domain 複製完成後，會在 ``<dest-root>/.staged/<domain_id>.manifest.json``
寫入逐檔案的 size/sha256/來源資訊，並在確認 manifest 落地後，寫入空的
``<dest-root>/.staged/<domain_id>.done`` 標記檔（``run_traps_server.sh`` 的
PREPARE 模式會檢查這個標記檔是否存在）。

重跑安全：檔案若已存在於 dest，且 size 與 sha256（用 manifest 內的 hash cache，
以 path+size+mtime 為 key，避免每次重跑都整檔重算 hash）都與來源相符，會被跳過；
``--force-reverify`` 會強制重新計算所有雜湊值（仍不會重算來源的 hash，因為來源不
可信任被修改，只重算 dest 端）。任何檔案的 size 或 sha256 不相符，腳本都會以非零
狀態碼結束。

本腳本絕不刪除 dest 端既有檔案（僅刪除自己建立的暫存檔），也絕不修改來源。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_SOURCE_ROOT = "/data/OCM-Preprocessed-Data/preprocessed/ocm_surface"

DEFAULT_DOMAIN_IDS = (
    "northeast_taiwan_common_cache_v3",
    "hsinchu_cache_v3",
    "houwan_nmmba_cache_v3",
    "lienchiang_common_cache_v3",
)

DEFAULT_MONTHS_SPEC = "202401-202512"

# 每月固定要複製的檔案；eta_m.npy 另由 --include-eta 控制。
_REQUIRED_MONTH_FILES = (
    "metadata.json",
    "time_utc_ns.npy",
    "u_surface_mps.npy",
    "v_surface_mps.npy",
    "valid_mask_surface.npy",
    "qc_flags.npy",
)
# 存在才複製，不存在不算錯誤。
_OPTIONAL_MONTH_FILES = ("quality_report.json",)
_ETA_FILE = "eta_m.npy"

_HASH_CHUNK_SIZE = 1024 * 1024
_FREE_SPACE_SAFETY_FACTOR = 1.05


class StagingError(Exception):
    """單一檔案層級的 staging 失敗（size 或 sha256 不符、來源缺失等）。"""


def _parse_months(spec):
    """把 ``--months`` 字串（單一 YYYYMM、逗號分隔清單、或 YYYYMM-YYYYMM 範圍）展開。

    支援混合，例如 ``"202401-202403,202406,202412"``。回傳排序後、去重的
    ``YYYYMM`` 字串 list。
    """

    months = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = _parse_yyyymm(start_str.strip())
            end = _parse_yyyymm(end_str.strip())
            if end < start:
                raise ValueError(f"--months 範圍 {chunk!r} 結束早於開始")
            year, month = start
            while (year, month) <= end:
                months.add(f"{year:04d}{month:02d}")
                month += 1
                if month > 12:
                    month = 1
                    year += 1
        else:
            _parse_yyyymm(chunk)  # 驗證格式
            months.add(chunk)
    return sorted(months)


def _parse_yyyymm(text):
    if len(text) != 6 or not text.isdigit():
        raise ValueError(f"月份必須是 YYYYMM 格式，收到 {text!r}")
    year = int(text[:4])
    month = int(text[4:6])
    if not (1 <= month <= 12):
        raise ValueError(f"月份數字必須介於 01-12，收到 {text!r}")
    return (year, month)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="把 SERVER NAS 上的 OCM surface 前處理資料同步到本機 SSD。",
    )
    parser.add_argument(
        "--source-root",
        default=DEFAULT_SOURCE_ROOT,
        help=f"NAS 上 ocm_surface 根目錄，預設 {DEFAULT_SOURCE_ROOT}。",
    )
    parser.add_argument(
        "--dest-root",
        required=True,
        help="本機 SSD 目的地根目錄（例如 traps-local-data）；本腳本會在其下建立 ocm_surface/ 與 .staged/。",
    )
    parser.add_argument(
        "--domain-id",
        dest="domain_ids",
        action="append",
        default=None,
        help="要 staging 的 flow-domain ID；可重複指定。預設處理四個研究區域 domain。",
    )
    parser.add_argument(
        "--months",
        default=DEFAULT_MONTHS_SPEC,
        help=f"要 staging 的月份，YYYYMM 範圍或逗號清單，預設 {DEFAULT_MONTHS_SPEC}。",
    )
    parser.add_argument(
        "--include-eta",
        action="store_true",
        help="同時複製 eta_m.npy（預設不複製）。",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="平行複製的 worker 數，預設 4。",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只驗證 dest 端既有檔案的 size/sha256，不複製任何新檔案；缺檔視為失敗。",
    )
    parser.add_argument(
        "--force-reverify",
        action="store_true",
        help="忽略 manifest hash cache，強制重新計算 dest 端所有檔案的 sha256。",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs 必須 >= 1")
    return args


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(manifest_path):
    if not os.path.isfile(manifest_path):
        return {"domain_id": None, "files": {}}
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {"domain_id": None, "files": {}}
    files = payload.get("files")
    if not isinstance(files, dict):
        files = {}
    return {"domain_id": payload.get("domain_id"), "files": files}


def _cached_dest_hash(cache_entry, dest_path, force_reverify):
    """回傳 dest 檔案目前的 sha256；若 cache 命中（path+size+mtime 相符）且未強制
    重算，直接沿用 cache 內的 hash，避免每次重跑都整檔重讀。"""

    stat = os.stat(dest_path)
    size = stat.st_size
    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
    if (
        not force_reverify
        and cache_entry is not None
        and cache_entry.get("bytes") == size
        and cache_entry.get("dest_mtime_ns") == mtime_ns
        and cache_entry.get("sha256")
    ):
        return cache_entry["sha256"], size, mtime_ns
    return _sha256_file(dest_path), size, mtime_ns


def _discover_domain_files(source_root, domain_id, months, include_eta):
    """列出一個 domain 需要複製的 (relative_path, source_abs_path) 清單。

    grid/ 下的所有檔案全部複製；每個月份下只複製固定清單中存在的檔案。缺少必要
    的每月檔案會直接丟出 StagingError（qc/quality_report 例外，允許缺）。
    """

    domain_source_root = os.path.join(source_root, domain_id)
    entries = []

    grid_dir = os.path.join(domain_source_root, "grid")
    if not os.path.isdir(grid_dir):
        raise StagingError(f"domain {domain_id!r} 來源缺少 grid/：{grid_dir}")
    for name in sorted(os.listdir(grid_dir)):
        source_path = os.path.join(grid_dir, name)
        if os.path.isfile(source_path):
            entries.append((f"{domain_id}/grid/{name}", source_path))

    for month in months:
        month_dir = os.path.join(domain_source_root, "months", month)
        if not os.path.isdir(month_dir):
            raise StagingError(f"domain {domain_id!r} 缺少月份目錄：{month_dir}")
        for name in _REQUIRED_MONTH_FILES:
            source_path = os.path.join(month_dir, name)
            if not os.path.isfile(source_path):
                raise StagingError(f"domain {domain_id!r} 月份 {month} 缺少必要檔案：{source_path}")
            entries.append((f"{domain_id}/months/{month}/{name}", source_path))
        for name in _OPTIONAL_MONTH_FILES:
            source_path = os.path.join(month_dir, name)
            if os.path.isfile(source_path):
                entries.append((f"{domain_id}/months/{month}/{name}", source_path))
        if include_eta:
            source_path = os.path.join(month_dir, _ETA_FILE)
            if not os.path.isfile(source_path):
                raise StagingError(f"domain {domain_id!r} 月份 {month} 缺少 --include-eta 要求的 {_ETA_FILE}")
            entries.append((f"{domain_id}/months/{month}/{_ETA_FILE}", source_path))

    return entries


def _check_free_space(dest_root, entries):
    """確認 dest 所在檔案系統的可用空間，足以容納所有來源檔案總大小 * 1.05。"""

    total_bytes = 0
    for _relative_path, source_path in entries:
        total_bytes += os.path.getsize(source_path)
    required_bytes = int(total_bytes * _FREE_SPACE_SAFETY_FACTOR)

    os.makedirs(dest_root, exist_ok=True)
    usage = shutil.disk_usage(dest_root)
    if usage.free < required_bytes:
        raise StagingError(
            "dest 空間不足："
            f"需要約 {required_bytes / 1e9:.2f} GB（含 5% 安全係數），"
            f"目前可用 {usage.free / 1e9:.2f} GB（{dest_root}）"
        )
    return total_bytes


def _copy_one_file(relative_path, source_path, dest_root, cache_entry, verify_only, force_reverify):
    """複製（或驗證）單一檔案，回傳描述本次結果的 dict。

    複製流程：寫到同目錄的暫存檔名 -> fsync -> os.replace 原子換名，避免中途失敗
    留下半寫檔案覆蓋掉原本可能已存在的正確 dest 檔案。
    """

    dest_path = os.path.join(dest_root, relative_path)
    source_size = os.path.getsize(source_path)

    if verify_only:
        if not os.path.isfile(dest_path):
            raise StagingError(f"--verify-only：dest 缺少檔案 {relative_path}")
        dest_hash, dest_size, dest_mtime_ns = _cached_dest_hash(cache_entry, dest_path, force_reverify)
        if dest_size != source_size:
            raise StagingError(f"{relative_path} size 不符：dest={dest_size} source={source_size}")
        source_hash = _sha256_file(source_path)
        if dest_hash != source_hash:
            raise StagingError(f"{relative_path} sha256 不符")
        return {
            "relative_path": relative_path,
            "bytes": dest_size,
            "sha256": dest_hash,
            "dest_mtime_ns": dest_mtime_ns,
            "source_path": source_path,
            "source_mtime": os.path.getmtime(source_path),
            "copied": False,
            "skipped": True,
        }

    if os.path.isfile(dest_path):
        dest_hash, dest_size, dest_mtime_ns = _cached_dest_hash(cache_entry, dest_path, force_reverify)
        if dest_size == source_size:
            source_hash = _sha256_file(source_path)
            if dest_hash == source_hash:
                return {
                    "relative_path": relative_path,
                    "bytes": dest_size,
                    "sha256": dest_hash,
                    "dest_mtime_ns": dest_mtime_ns,
                    "source_path": source_path,
                    "source_mtime": os.path.getmtime(source_path),
                    "copied": False,
                    "skipped": True,
                }
        # size/hash 不符：視為需要重新複製（可能是先前中斷的半寫檔）。

    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    tmp_path = os.path.join(dest_dir, f".{os.path.basename(dest_path)}.tmp{os.getpid()}")
    try:
        with open(source_path, "rb") as source_handle, open(tmp_path, "wb") as tmp_handle:
            shutil.copyfileobj(source_handle, tmp_handle, length=_HASH_CHUNK_SIZE)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
        os.replace(tmp_path, dest_path)
    except BaseException:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise

    dest_size = os.path.getsize(dest_path)
    if dest_size != source_size:
        raise StagingError(f"{relative_path} 複製後 size 不符：dest={dest_size} source={source_size}")
    dest_hash = _sha256_file(dest_path)
    source_hash = _sha256_file(source_path)
    if dest_hash != source_hash:
        raise StagingError(f"{relative_path} 複製後 sha256 不符")
    stat = os.stat(dest_path)
    dest_mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
    return {
        "relative_path": relative_path,
        "bytes": dest_size,
        "sha256": dest_hash,
        "dest_mtime_ns": dest_mtime_ns,
        "source_path": source_path,
        "source_mtime": os.path.getmtime(source_path),
        "copied": True,
        "skipped": False,
    }


def _stage_domain(source_root, domain_id, months, include_eta, dest_root, jobs, verify_only, force_reverify):
    """Staging 單一 domain；回傳 (success, bytes_copied, files_done, error_message)。"""

    ocm_surface_dest_root = os.path.join(dest_root, "ocm_surface")
    staged_dir = os.path.join(dest_root, ".staged")
    manifest_path = os.path.join(staged_dir, f"{domain_id}.manifest.json")
    done_marker_path = os.path.join(staged_dir, f"{domain_id}.done")

    try:
        entries = _discover_domain_files(source_root, domain_id, months, include_eta)
    except StagingError as error:
        print(
            json.dumps({"domain_id": domain_id, "status": "failed", "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return False, 0, 0, str(error)

    if not verify_only:
        try:
            _check_free_space(ocm_surface_dest_root, entries)
        except StagingError as error:
            print(
                json.dumps({"domain_id": domain_id, "status": "failed", "error": str(error)}, ensure_ascii=False),
                file=sys.stderr,
            )
            return False, 0, 0, str(error)

    existing_manifest = _load_manifest(manifest_path)
    existing_files = existing_manifest["files"]

    files_by_month = {}
    for relative_path, source_path in entries:
        parts = relative_path.split("/")
        month = parts[2] if len(parts) >= 3 and parts[1] == "months" else "grid"
        files_by_month.setdefault(month, []).append((relative_path, source_path))

    bytes_copied = 0
    files_done = 0
    results = {}
    errors = []

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for month in sorted(files_by_month):
            month_entries = files_by_month[month]
            futures = {}
            for relative_path, source_path in month_entries:
                cache_entry = existing_files.get(relative_path)
                future = executor.submit(
                    _copy_one_file,
                    relative_path,
                    source_path,
                    ocm_surface_dest_root,
                    cache_entry,
                    verify_only,
                    force_reverify,
                )
                futures[future] = relative_path

            month_bytes = 0
            month_errors = []
            for future in as_completed(futures):
                relative_path = futures[future]
                try:
                    result = future.result()
                except StagingError as error:
                    month_errors.append(f"{relative_path}: {error}")
                    continue
                except OSError as error:
                    month_errors.append(f"{relative_path}: {error}")
                    continue
                results[relative_path] = result
                if result["copied"]:
                    month_bytes += result["bytes"]
                files_done += 1

            bytes_copied += month_bytes
            if month_errors:
                errors.extend(month_errors)
            print(
                json.dumps(
                    {
                        "domain_id": domain_id,
                        "month": month,
                        "status": "failed" if month_errors else "ok",
                        "files": len(month_entries),
                        "errors": month_errors if month_errors else None,
                    },
                    ensure_ascii=False,
                )
            )

    if errors:
        for message in errors:
            print(json.dumps({"domain_id": domain_id, "status": "failed", "error": message}, ensure_ascii=False), file=sys.stderr)
        return False, bytes_copied, files_done, "; ".join(errors)

    if verify_only:
        return True, bytes_copied, files_done, None

    manifest_files = {}
    for relative_path, _source_path in entries:
        result = results[relative_path]
        manifest_files[relative_path] = {
            "bytes": result["bytes"],
            "sha256": result["sha256"],
            "dest_mtime_ns": result["dest_mtime_ns"],
            "source_path": result["source_path"],
            "source_mtime": result["source_mtime"],
            "copied_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    os.makedirs(staged_dir, exist_ok=True)
    manifest_payload = {
        "domain_id": domain_id,
        "months": months,
        "include_eta": include_eta,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": manifest_files,
    }
    tmp_manifest_path = manifest_path + f".tmp{os.getpid()}"
    try:
        with open(tmp_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_manifest_path, manifest_path)
    except BaseException:
        try:
            if os.path.isfile(tmp_manifest_path):
                os.remove(tmp_manifest_path)
        except OSError:
            pass
        raise

    with open(done_marker_path, "w", encoding="utf-8") as handle:
        handle.write("")
        handle.flush()
        os.fsync(handle.fileno())

    return True, bytes_copied, files_done, None


def main(argv=None):
    args = _parse_args(argv)
    try:
        months = _parse_months(args.months)
    except ValueError as error:
        print(f"--months 解析失敗: {error}", file=sys.stderr)
        return 2
    if not months:
        print("--months 未產生任何月份", file=sys.stderr)
        return 2

    domain_ids = args.domain_ids or list(DEFAULT_DOMAIN_IDS)

    start_time = time.time()
    total_bytes = 0
    total_files = 0
    failed_domains = []

    for domain_id in domain_ids:
        try:
            success, bytes_copied, files_done, error_message = _stage_domain(
                source_root=args.source_root,
                domain_id=domain_id,
                months=months,
                include_eta=args.include_eta,
                dest_root=args.dest_root,
                jobs=args.jobs,
                verify_only=args.verify_only,
                force_reverify=args.force_reverify,
            )
        except Exception:  # noqa: BLE001 - 任何未預期例外都不能讓其他 domain 被跳過的原因不明
            traceback.print_exc(file=sys.stderr)
            failed_domains.append(domain_id)
            continue

        total_bytes += bytes_copied
        total_files += files_done
        if not success:
            failed_domains.append(domain_id)

    elapsed = time.time() - start_time
    mb_per_s = (total_bytes / 1e6 / elapsed) if elapsed > 0 else 0.0
    summary = {
        "status": "failed" if failed_domains else "ok",
        "domains_requested": domain_ids,
        "domains_failed": failed_domains,
        "months": months,
        "files_processed": total_files,
        "bytes_copied": total_bytes,
        "elapsed_seconds": round(elapsed, 2),
        "mb_per_s": round(mb_per_s, 2),
        "verify_only": args.verify_only,
    }
    print(json.dumps(summary, ensure_ascii=False))

    return 1 if failed_domains else 0


if __name__ == "__main__":
    raise SystemExit(main())
