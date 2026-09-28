"""``scripts/stage_ocm_surface_local.py`` 的契約測試。

腳本本身刻意只用 stdlib，且語法要相容 Python 3.8（有可能在 .venv 建立前，由
SERVER 系統內建的 python3 執行），所以這裡用 ``importlib`` 動態載入該檔案，而不
是把 scripts/ 加進 sys.path 當成套件 import——避免和專案其他測試共用的
``pythonpath = ["src"]`` 設定混淆。

測試只操作 ``tmp_path`` 下的合成 fixture 目錄，模擬 NAS 端
``ocm_surface/<domain>/{grid, months/<YYYYMM>}`` layout；不連線 SERVER、不讀取真
實資料。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "stage_ocm_surface_local.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage_ocm_surface_local", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stage = _load_module()

_DOMAIN_ID = "synthetic_domain"
_MONTHS = ["202401", "202402"]

_MONTH_FILES = (
    "metadata.json",
    "quality_report.json",
    "time_utc_ns.npy",
    "u_surface_mps.npy",
    "v_surface_mps.npy",
    "valid_mask_surface.npy",
    "qc_flags.npy",
)


def _write_fake_source(source_root: Path, domain_id: str, months, include_eta: bool = False) -> None:
    grid_dir = source_root / domain_id / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    (grid_dir / "lat.npy").write_bytes(b"lat-bytes")
    (grid_dir / "lon.npy").write_bytes(b"lon-bytes")
    (grid_dir / "metadata.json").write_text(json.dumps({"domain_id": domain_id}), encoding="utf-8")

    for month in months:
        month_dir = source_root / domain_id / "months" / month
        month_dir.mkdir(parents=True, exist_ok=True)
        for name in _MONTH_FILES:
            content = f"{domain_id}-{month}-{name}".encode("utf-8")
            (month_dir / name).write_bytes(content)
        if include_eta:
            (month_dir / "eta_m.npy").write_bytes(f"{domain_id}-{month}-eta".encode("utf-8"))


class TestParseMonths:
    def test_single_range(self):
        assert stage._parse_months("202401-202403") == ["202401", "202402", "202403"]

    def test_range_across_year_boundary(self):
        assert stage._parse_months("202411-202502") == ["202411", "202412", "202501", "202502"]

    def test_mixed_list_and_range(self):
        result = stage._parse_months("202401-202402,202406, 202412")
        assert result == ["202401", "202402", "202406", "202412"]

    def test_deduplicates_and_sorts(self):
        result = stage._parse_months("202403,202401,202402,202401")
        assert result == ["202401", "202402", "202403"]

    def test_invalid_month_number_raises(self):
        with pytest.raises(ValueError):
            stage._parse_months("202413")

    def test_reversed_range_raises(self):
        with pytest.raises(ValueError):
            stage._parse_months("202403-202401")

    def test_bad_format_raises(self):
        with pytest.raises(ValueError):
            stage._parse_months("2024-01")


class TestStageDomain:
    def test_copies_all_files_and_writes_manifest_and_marker(self, tmp_path: Path):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        success, bytes_copied, files_done, error = stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )

        assert success is True
        assert error is None
        assert bytes_copied > 0
        # grid 3 檔 + 2 個月 * 7 檔 (含 quality_report) = 17
        assert files_done == 17

        dest_ocm_root = dest_root / "ocm_surface" / _DOMAIN_ID
        assert (dest_ocm_root / "grid" / "lat.npy").read_bytes() == b"lat-bytes"
        assert (dest_ocm_root / "months" / "202401" / "u_surface_mps.npy").exists()
        # eta 未要求，不應被複製
        assert not (dest_ocm_root / "months" / "202401" / "eta_m.npy").exists()

        manifest_path = dest_root / ".staged" / f"{_DOMAIN_ID}.manifest.json"
        done_path = dest_root / ".staged" / f"{_DOMAIN_ID}.done"
        assert manifest_path.is_file()
        assert done_path.is_file()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["domain_id"] == _DOMAIN_ID
        assert len(manifest["files"]) == 17
        for relative_path, entry in manifest["files"].items():
            dest_file = dest_root / "ocm_surface" / relative_path
            assert dest_file.is_file()
            assert entry["bytes"] == dest_file.stat().st_size
            digest = hashlib.sha256(dest_file.read_bytes()).hexdigest()
            assert entry["sha256"] == digest

    def test_include_eta_copies_eta_file(self, tmp_path: Path):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS, include_eta=True)

        success, _bytes_copied, files_done, error = stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=True,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )

        assert success is True
        assert error is None
        assert files_done == 19  # 17 + 2 個月各一個 eta_m.npy
        for month in _MONTHS:
            eta_path = dest_root / "ocm_surface" / _DOMAIN_ID / "months" / month / "eta_m.npy"
            assert eta_path.is_file()

    def test_missing_required_month_file_fails_and_writes_no_marker(self, tmp_path: Path):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)
        # 移除一個必要檔案。
        (source_root / _DOMAIN_ID / "months" / "202401" / "u_surface_mps.npy").unlink()

        success, _bytes_copied, _files_done, error = stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )

        assert success is False
        assert error is not None
        done_path = dest_root / ".staged" / f"{_DOMAIN_ID}.done"
        assert not done_path.exists()

    def test_rerun_skips_unchanged_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        first = stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )
        assert first[0] is True

        call_count = {"n": 0}
        original_sha256_file = stage._sha256_file

        def counting_sha256_file(path):
            call_count["n"] += 1
            return original_sha256_file(path)

        monkeypatch.setattr(stage, "_sha256_file", counting_sha256_file)

        second = stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )
        assert second[0] is True
        assert second[1] == 0  # 沒有新複製任何 bytes

        # source hash 仍會逐檔重算，但 dest 端應該命中 cache（call_count 遠少於
        # 2 倍檔案數，若完全沒 cache，每個檔案都要重算 source+dest 兩次）。
        assert call_count["n"] <= 17  # 只有 source 端被重算，dest 端全部命中 cache

    def test_force_reverify_rehashes_dest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )

        call_count = {"n": 0}
        original_sha256_file = stage._sha256_file

        def counting_sha256_file(path):
            call_count["n"] += 1
            return original_sha256_file(path)

        monkeypatch.setattr(stage, "_sha256_file", counting_sha256_file)

        stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=True,
        )
        # force_reverify: 每個檔案 source + dest 都要重算 -> 2 * 17
        assert call_count["n"] == 34

    def test_tampered_dest_file_is_recopied(self, tmp_path: Path):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )

        target = dest_root / "ocm_surface" / _DOMAIN_ID / "grid" / "lat.npy"
        target.write_bytes(b"corrupted")

        success, bytes_copied, _files_done, error = stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )
        assert success is True
        assert error is None
        assert bytes_copied > 0
        assert target.read_bytes() == b"lat-bytes"

    def test_verify_only_missing_dest_fails(self, tmp_path: Path):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        success, _bytes_copied, _files_done, error = stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=True,
            force_reverify=False,
        )
        assert success is False
        assert error is not None

    def test_verify_only_after_successful_copy_passes(self, tmp_path: Path):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )

        success, _bytes_copied, _files_done, error = stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=True,
            force_reverify=False,
        )
        assert success is True
        assert error is None

    def test_never_deletes_unrelated_dest_files(self, tmp_path: Path):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        sentinel = dest_root / "ocm_surface" / _DOMAIN_ID / "grid" / "unrelated_extra_file.txt"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("do not touch")

        stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )

        assert sentinel.is_file()
        assert sentinel.read_text() == "do not touch"

    def test_no_leftover_tmp_files_after_success(self, tmp_path: Path):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )

        tmp_files = [p for p in dest_root.rglob("*.tmp*")]
        assert tmp_files == []

    def test_insufficient_free_space_refuses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        class _FakeUsage:
            free = 1  # 遠小於任何合成 fixture 的大小

        monkeypatch.setattr(stage.shutil, "disk_usage", lambda _path: _FakeUsage())

        success, bytes_copied, _files_done, error = stage._stage_domain(
            source_root=str(source_root),
            domain_id=_DOMAIN_ID,
            months=_MONTHS,
            include_eta=False,
            dest_root=str(dest_root),
            jobs=2,
            verify_only=False,
            force_reverify=False,
        )
        assert success is False
        assert bytes_copied == 0
        assert error is not None
        assert "空間不足" in error


class TestMainCli:
    def test_main_end_to_end_success(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, _DOMAIN_ID, _MONTHS)

        exit_code = stage.main(
            [
                "--source-root",
                str(source_root),
                "--dest-root",
                str(dest_root),
                "--domain-id",
                _DOMAIN_ID,
                "--months",
                "202401-202402",
                "--jobs",
                "2",
            ]
        )
        assert exit_code == 0

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().splitlines() if line]
        summary = json.loads(lines[-1])
        assert summary["status"] == "ok"
        assert summary["domains_failed"] == []
        assert summary["files_processed"] == 17
        assert summary["bytes_copied"] > 0

        done_path = dest_root / ".staged" / f"{_DOMAIN_ID}.done"
        assert done_path.is_file()

    def test_main_reports_nonzero_on_missing_domain(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        # 完全不建立來源 domain 目錄。

        exit_code = stage.main(
            [
                "--source-root",
                str(source_root),
                "--dest-root",
                str(dest_root),
                "--domain-id",
                _DOMAIN_ID,
                "--months",
                "202401",
            ]
        )
        assert exit_code == 1

    def test_main_rejects_bad_months(self, tmp_path: Path):
        exit_code = stage.main(
            [
                "--dest-root",
                str(tmp_path / "dest"),
                "--months",
                "not-a-month",
            ]
        )
        assert exit_code == 2

    def test_main_multiple_domains_partial_failure(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        source_root = tmp_path / "source"
        dest_root = tmp_path / "dest"
        _write_fake_source(source_root, "domain_ok", _MONTHS)
        # domain_missing 完全沒有來源資料。

        exit_code = stage.main(
            [
                "--source-root",
                str(source_root),
                "--dest-root",
                str(dest_root),
                "--domain-id",
                "domain_ok",
                "--domain-id",
                "domain_missing",
                "--months",
                "202401-202402",
            ]
        )
        assert exit_code == 1

        done_ok = dest_root / ".staged" / "domain_ok.done"
        done_missing = dest_root / ".staged" / "domain_missing.done"
        assert done_ok.is_file()
        assert not done_missing.exists()


# ---------------------------------------------------------------------------
# scripts/run_traps_server.sh — 輕量煙霧測試
# ---------------------------------------------------------------------------
#
# 本檔案是 stage_ocm_surface_local.py 的測試，但 run_traps_server.sh 沒有自己的
# 測試檔（依 WP-C 規格只新增這一個 tests/ 檔案），所以順帶在此驗證 bash 語法與
# STATUS 模式對一個假 workspace 的行為，不呼叫任何需要 SERVER/NAS/.venv 的模式。

_RUN_SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_traps_server.sh"

pytestmark_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")


@pytestmark_bash
class TestRunTrapsServerShellScript:
    def test_bash_syntax_check(self):
        result = subprocess.run(
            ["bash", "-n", str(_RUN_SERVER_SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_usage_on_missing_mode_exits_nonzero(self):
        result = subprocess.run(
            ["bash", str(_RUN_SERVER_SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "MODE" in result.stderr

    def test_unknown_mode_exits_nonzero(self):
        result = subprocess.run(
            ["bash", str(_RUN_SERVER_SCRIPT), "NOT_A_MODE"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_status_against_fake_workspace(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        runs = workspace / "traps-runs"
        param_id = "r0_default"
        detect_root = runs / "runs" / param_id / "detect"
        track_root = runs / "runs" / param_id / "track"

        domain = "synthetic_domain"
        # 兩個月都有 _SUCCESS，track 也有 _SUCCESS -> STATUS 應能印出且不噴例外。
        for month in ("202401", "202402"):
            shard_dir = detect_root / domain / month
            shard_dir.mkdir(parents=True, exist_ok=True)
            (shard_dir / "_SUCCESS").write_text("")
        (track_root / domain).mkdir(parents=True, exist_ok=True)
        (track_root / domain / "_SUCCESS").write_text("")

        env = dict(os.environ)
        env.update(
            {
                "TRAPS_WORKSPACE": str(workspace),
                "TRAPS_RUNS": str(runs),
                "TRAPS_DOMAINS": domain,
                "TRAPS_MONTHS": "202401-202402",
                "TRAPS_PARAM_ID": param_id,
            }
        )

        result = subprocess.run(
            ["bash", str(_RUN_SERVER_SCRIPT), "STATUS"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "TRAPs STATUS" in result.stdout
        assert domain in result.stdout
        assert "2/2 _SUCCESS shards" in result.stdout
        assert f"{domain:<40} done" in result.stdout or "done" in result.stdout

    def test_status_reports_not_started_when_no_runs_dir(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        runs = workspace / "traps-runs"
        domain = "synthetic_domain"

        env = dict(os.environ)
        env.update(
            {
                "TRAPS_WORKSPACE": str(workspace),
                "TRAPS_RUNS": str(runs),
                "TRAPS_DOMAINS": domain,
                "TRAPS_MONTHS": "202401",
                "TRAPS_PARAM_ID": "r0_default",
            }
        )

        result = subprocess.run(
            ["bash", str(_RUN_SERVER_SCRIPT), "STATUS"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "0/1 _SUCCESS shards" in result.stdout
        assert "not-started" in result.stdout

    def test_prepare_fails_cleanly_against_empty_fake_workspace(self, tmp_path: Path):
        # PREPARE 對完全沒有 repo/.venv/staging 的假 workspace 應該以每項 FAIL 收
        # 場並回傳非零，而不是拋出未預期的 bash 錯誤（-e/-u 下常見的雷區）。
        workspace = tmp_path / "workspace"
        runs = workspace / "traps-runs"
        domain = "synthetic_domain"

        env = dict(os.environ)
        env.update(
            {
                "TRAPS_WORKSPACE": str(workspace),
                "TRAPS_REPO": str(workspace / "traps-repo"),
                "TRAPS_INPUTS": str(workspace / "traps-inputs"),
                "TRAPS_LOCAL_DATA": str(workspace / "traps-local-data"),
                "TRAPS_RUNS": str(runs),
                "TRAPS_DOMAINS": domain,
                "TRAPS_MONTHS": "202401",
                "TRAPS_MIN_FREE_GB": "999999999",
            }
        )

        result = subprocess.run(
            ["bash", str(_RUN_SERVER_SCRIPT), "PREPARE"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert "FAIL:" in result.stdout
