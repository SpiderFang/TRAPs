"""TRAPs CLI 的合成月份端到端測試。

fixture 只模擬 OCM schema 3.0 的小型表層 cache，速度場是可重現的局地吸引場，
不代表 2024--2025 正式資料。測試涵蓋 preflight、trial gate、流式 detection、
tracking/站點 metrics、sensitivity case、PNG/GeoJSON 及輸出雜湊。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN_ID = "houwan_nmmba_cache_v3"


def _write_json(path: Path, payload: dict) -> None:
    """以 UTF-8 寫入測試 metadata，保持與上游 sidecar 相同的 JSON 編碼。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_month(root: Path, *, cache_kind: str = "standard_month", status: str = "ready") -> Path:
    """建立兩時刻、41x41 公尺網格的局地合成 OCM 月份。"""

    month_dir = root / "ocm_surface" / DOMAIN_ID / "months" / "202501"
    grid_dir = root / "ocm_surface" / DOMAIN_ID / "grid"
    month_dir.mkdir(parents=True)
    grid_dir.mkdir(parents=True)
    size = 41
    spacing_m = 100.0
    axis_m = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) * spacing_m
    x_m, y_m = np.meshgrid(axis_m, axis_m, indexing="xy")
    envelope = np.exp(-(x_m * x_m + y_m * y_m) / 900.0**2)
    u_one = -2.0e-3 * x_m * envelope
    v_one = -1.0e-3 * y_m * envelope
    u = np.stack((u_one, u_one), axis=0).astype(np.float32)
    v = np.stack((v_one, v_one), axis=0).astype(np.float32)
    valid = np.ones(u.shape, dtype=bool)
    qc_flags = np.ones(u.shape, dtype=np.uint16)
    time = np.array([1_725_096_000_000_000_000, 1_725_099_600_000_000_000], dtype=np.int64)
    static = np.ones((size, size), dtype=bool)
    lon_axis = 120.763161 + axis_m / 100_000.0
    lat_axis = 21.946577 + axis_m / 100_000.0
    cell_area = np.full((size, size), spacing_m**2, dtype=np.float32)

    for name, array in {
        "time_utc_ns.npy": time,
        "u_surface_mps.npy": u,
        "v_surface_mps.npy": v,
        "valid_mask_surface.npy": valid,
        "qc_flags.npy": qc_flags,
    }.items():
        np.save(month_dir / name, array, allow_pickle=False)
    for name, array in {
        "lon.npy": lon_axis.astype(np.float64),
        "lat.npy": lat_axis.astype(np.float64),
        "x_m.npy": x_m.astype(np.float64),
        "y_m.npy": y_m.astype(np.float64),
        "cell_area_m2.npy": cell_area,
        "mask_static.npy": static,
    }.items():
        np.save(grid_dir / name, array, allow_pickle=False)

    _write_json(
        month_dir / "metadata.json",
        {
            "status": status,
            "cache_kind": cache_kind,
            "cache_schema_version": "3.0.0",
            "month": "202501",
            "domain": {
                "domain_id": DOMAIN_ID,
                "name_zh": "南灣合成海域",
                "lon_min": 120.16671,
                "lon_max": 121.62,
                "lat_min": 21.550844,
                "lat_max": 22.449156,
            },
        },
    )
    _write_json(
        grid_dir / "metadata.json",
        {
            "cache_schema_version": "3.0.0",
            "domain": {"domain_id": DOMAIN_ID},
            "grid_shape": [size, size],
        },
    )
    # 正式輸出只接受可回溯的海岸 sidecar；合成場沒有陸地，因此分析海域等於 static。
    np.save(grid_dir / "analysis_ocean_mask_synthetic_v1.npy", static, allow_pickle=False)
    _write_json(
        grid_dir / "analysis_land_mask_synthetic_v1.json",
        {
            "cache_schema_version": "3.0.0",
            "domain": {"domain_id": DOMAIN_ID},
            "arrays": {"shape_lat_lon": [size, size]},
        },
    )
    return month_dir


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """以 `python -m traps` 執行 CLI，確保測試驗證實際模組入口。"""

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "traps", *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_preflight_and_detect_write_traceable_outputs(tmp_path: Path) -> None:
    """標準月份可 preflight、流式檢出，並寫出 metrics、track、GeoJSON、圖與雜湊。"""

    month_dir = _write_month(tmp_path)
    preflight_dir = tmp_path / "preflight"
    preflight = _run_cli(
        "preflight",
        "--month-dir",
        str(month_dir),
        "--output-root",
        str(preflight_dir),
    )
    assert preflight.returncode == 0, preflight.stderr
    assert json.loads(preflight.stdout)["state_label"] == "READY"
    assert (preflight_dir / "preflight.sha256").is_file()

    # 上游 static mask 並非海岸線；正式檢出缺 sidecar 必須及早拒絕。
    no_coast = _run_cli("detect", "--month-dir", str(month_dir), "--output-root", str(tmp_path / "no_coast"))
    assert no_coast.returncode != 0
    assert "--ocean-mask" in no_coast.stderr

    output_dir = tmp_path / "detect"
    detect = _run_cli(
        "detect",
        "--month-dir",
        str(month_dir),
        "--ocean-mask",
        str(month_dir.parent.parent / "grid" / "analysis_ocean_mask_synthetic_v1.npy"),
        "--output-root",
        str(output_dir),
        "--render-snapshots",
        "--render-summary",
        "--min-coverage",
        "0.95",
    )
    assert detect.returncode == 0, detect.stderr
    catalog = json.loads((output_dir / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["state_label"] == "READY"
    assert len(catalog["snapshots"]) == 2
    assert len(catalog["site_statistics"]) == 5
    assert catalog["parameters"]["tracking_config"]["max_missing_frames"] == 1
    assert (output_dir / "tracks.json").is_file()
    assert (output_dir / "catalog.geojson").is_file()
    # GeoJSON 必須輸出 WGS84 經緯度；公尺制原始曲線只留在 catalog.json。
    geojson = json.loads((output_dir / "catalog.geojson").read_text(encoding="utf-8"))
    assert geojson["coordinate_units"] == "degrees"
    for feature in geojson["features"]:
        assert all(120.0 < point[0] < 122.0 and 21.0 < point[1] < 23.0 for point in feature["geometry"]["coordinates"])
    assert (output_dir / "figures" / "site_summary.png").is_file()
    assert len(list((output_dir / "figures").glob("snapshot_*.png"))) == 2

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        artifact_path = output_dir / artifact["path"]
        assert artifact_path.is_file()
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"]
    # 相同 output root 不得覆寫或讓舊圖被新 manifest 誤列為本次產物。
    repeated = _run_cli(
        "detect", "--month-dir", str(month_dir), "--ocean-mask",
        str(month_dir.parent.parent / "grid" / "analysis_ocean_mask_synthetic_v1.npy"),
        "--output-root", str(output_dir), "--snapshot-index", "0",
    )
    assert repeated.returncode != 0
    assert "輸出目錄已有內容" in repeated.stderr


def test_trial_requires_explicit_flag_and_sensitivity_has_eight_cases(tmp_path: Path) -> None:
    """trial cache 預設拒絕；允許後八個 sensitivity case 全部保留 PILOT_ONLY。"""

    month_dir = _write_month(tmp_path, cache_kind="trial_partial_month", status="trial_ready")
    output_dir = tmp_path / "trial"
    rejected = _run_cli(
        "detect",
        "--month-dir",
        str(month_dir),
        "--output-root",
        str(output_dir),
        "--snapshot-index",
        "0",
    )
    assert rejected.returncode != 0
    assert "allow_trial_pilot" in rejected.stderr

    allowed = _run_cli(
        "detect",
        "--month-dir",
        str(month_dir),
        "--output-root",
        str(output_dir),
        "--allow-trial-pilot",
        "--snapshot-index",
        "0",
        "--sensitivity",
    )
    assert allowed.returncode == 0, allowed.stderr
    root_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["state_label"] == "PILOT_ONLY"
    assert len(root_manifest["cases"]) == 8
    assert all(case["state_label"] == "PILOT_ONLY" for case in root_manifest["cases"])
    assert root_manifest["parameters"]["robust_status"] == "not_evaluated"


def test_summary_and_render_can_reuse_catalog(tmp_path: Path) -> None:
    """summary/render 只讀 catalog 即可重建站點表與摘要圖。"""

    month_dir = _write_month(tmp_path)
    detect_dir = tmp_path / "detect"
    detected = _run_cli(
        "detect",
        "--surface-root",
        str(tmp_path / "ocm_surface"),
        "--domain-id",
        DOMAIN_ID,
        "--month",
        "202501",
        "--ocean-mask",
        str(month_dir.parent.parent / "grid" / "analysis_ocean_mask_synthetic_v1.npy"),
        "--output-root",
        str(detect_dir),
        "--snapshot-index",
        "0",
    )
    assert detected.returncode == 0, detected.stderr
    summary_dir = tmp_path / "summary"
    summary = _run_cli(
        "summary",
        "--catalog",
        str(detect_dir / "catalog.json"),
        "--output-root",
        str(summary_dir),
        "--render",
    )
    assert summary.returncode == 0, summary.stderr
    single_rows = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))["rows"]
    assert [row["study_site_id"] for row in single_rows] == ["nanwan"]
    assert (summary_dir / "site_summary.png").is_file()

    # 四區比較使用多份同時窗 catalog；此合成第二份資料只驗證站點路由與分母，
    # 不代表另一個真實 flow domain 已完成 TRAP 計算。
    second = json.loads((detect_dir / "catalog.json").read_text(encoding="utf-8"))
    nanwan_row = dict(next(row for row in second["site_statistics"] if row["study_site_id"] == "nanwan"))
    for row in second["site_statistics"]:
        row["site_area_m2"] = 0.0
    hsinchu_row = next(row for row in second["site_statistics"] if row["study_site_id"] == "hsinchu")
    hsinchu_row.update({key: value for key, value in nanwan_row.items() if key not in {"site_id", "study_site_id", "study_site_name_zh"}})
    hsinchu_row.update({"site_id": "hsinchu", "study_site_id": "hsinchu", "study_site_name_zh": "新竹外海"})
    second["source"]["domain_id"] = "hsinchu_cache_v3"
    second_path = tmp_path / "hsinchu_catalog.json"
    second_path.write_text(json.dumps(second, ensure_ascii=False), encoding="utf-8")
    combined_dir = tmp_path / "combined"
    combined = _run_cli(
        "summary", "--catalog", str(detect_dir / "catalog.json"), str(second_path),
        "--output-root", str(combined_dir), "--render",
    )
    assert combined.returncode == 0, combined.stderr
    combined_rows = json.loads((combined_dir / "summary.json").read_text(encoding="utf-8"))["rows"]
    assert [row["study_site_id"] for row in combined_rows] == ["hsinchu", "nanwan"]

    render_dir = tmp_path / "render"
    rendered = _run_cli(
        "render",
        "--kind",
        "snapshot",
        "--month-dir",
        str(month_dir),
        "--ocean-mask",
        str(month_dir.parent.parent / "grid" / "analysis_ocean_mask_synthetic_v1.npy"),
        "--snapshot-index",
        "0",
        "--output-root",
        str(render_dir),
    )
    assert rendered.returncode == 0, rendered.stderr
    assert (render_dir / "snapshot_000000.png").is_file()
