"""literature-figures CLI 的合成端到端測試。

測試沿用 ``test_cli_pipeline`` 的小型 OCM cache fixture，讓 CLI 真的從 catalog
重開月份、grid 與 analysis mask provenance，再檢查文獻借鏡圖與 availability。
fixture 只有兩個 UTC snapshot，所有輸出仍是合成／短期試算，不能代表正式海況或
長期壽命估計。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from test_cli_pipeline import DOMAIN_ID, _run_cli, _write_month


def _build_detect_catalog(tmp_path: Path) -> tuple[Path, Path]:
    """建立帶有可回溯海岸 sidecar 的合成 catalog。

    literature-figures 需要 detect 寫出的完整 source provenance，因此測試不直接
    拼裝 catalog，而是先走既有 detect CLI。回傳月份目錄與 detect 成果目錄，供後續
    測試檢查來源雜湊及圖面路徑。
    """

    month_dir = _write_month(tmp_path)
    sidecar = month_dir.parent.parent / "grid" / "analysis_ocean_mask_synthetic_v1.npy"
    detect_dir = tmp_path / "detect"
    detected = _run_cli(
        "detect",
        "--month-dir",
        str(month_dir),
        "--ocean-mask",
        str(sidecar),
        "--output-root",
        str(detect_dir),
    )
    assert detected.returncode == 0, detected.stderr
    return month_dir, detect_dir


def test_literature_figures_emit_model_outputs_and_data_gates(tmp_path: Path) -> None:
    """單區 catalog 應產生模型圖，附加資料缺少時只標 DATA_REQUIRED。

    這個測試同時確認 mechanism 圖保留 CONCEPT_ONLY、method comparison 使用首個
    snapshot、壽命圖有完整／截尾統計摘要，以及 manifest 的每一筆 artifact 都可由
    實際檔案重新計算 SHA-256。潮相、軌跡與海廢沒有提供檔案時，不應產生空白科學圖。
    """

    _month_dir, detect_dir = _build_detect_catalog(tmp_path)
    output_dir = tmp_path / "literature"
    result = _run_cli(
        "literature-figures",
        "--catalog",
        str(detect_dir / "catalog.json"),
        "--output-root",
        str(output_dir),
        "--trajectory-distance-threshold-m",
        "20000",
        "--trajectory-max-interval-hours",
        "3",
    )
    assert result.returncode == 0, result.stderr

    availability = json.loads((output_dir / "figure_availability.json").read_text(encoding="utf-8"))
    assert availability["operation"] == "literature-figures"
    assert availability["catalog_count"] == 1
    assert availability["parameters"]["trajectory_distance_threshold_m"] == 20000.0
    assert availability["parameters"]["trajectory_distance_threshold_status"] == "CLI_OVERRIDE"
    assert availability["parameters"]["trajectory_max_interval_hours"] == 3.0
    assert availability["parameters"]["trajectory_max_interval_status"] == "CLI_OVERRIDE"

    mechanism = next(item for item in availability["figures"] if item["kind"] == "mechanism")
    assert mechanism["status"] == "CONCEPT_ONLY"
    assert (output_dir / mechanism["path"]).is_file()

    domain = availability["domains"][DOMAIN_ID]
    assert set(domain["figures"]) == {
        "method_comparison",
        "spatial_summary",
        "observed_span_distribution",
        "lifecycle_intensity",
    }
    # 可追溯的 JSON 統計契約另列 summaries，不應被誤當成 PNG 圖件。
    assert set(domain["summaries"]) == {
        "spatial_summary_metadata",
        "lifetime_summary",
    }
    for relative_path in domain["figures"].values():
        assert (output_dir / relative_path).is_file()
    for relative_path in domain["summaries"].values():
        assert (output_dir / relative_path).is_file()
    assert domain["conditional"]["tidal_phase"]["status"] == "DATA_REQUIRED"
    assert domain["conditional"]["trajectory"]["status"] == "DATA_REQUIRED"
    assert domain["conditional"]["short_term_deformation"]["status"] == "DATA_REQUIRED"
    assert domain["conditional"]["marine_debris"]["status"] == "DATA_REQUIRED"
    assert not (output_dir / DOMAIN_ID / "tidal_occurrence.png").exists()
    assert not (output_dir / DOMAIN_ID / "trajectory_trap_comparison.png").exists()
    assert not (output_dir / DOMAIN_ID / "debris_trap_overlay.png").exists()

    lifetime = json.loads(
        (output_dir / DOMAIN_ID / "lifetime_summary.json").read_text(encoding="utf-8")
    )
    assert lifetime["duration_semantics"].startswith("first_to_last_detection_observed_span")
    assert "complete_track_count" in lifetime
    assert "censored_track_count" in lifetime

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        artifact_path = output_dir / artifact["path"]
        assert artifact_path.is_file()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]


def test_literature_figures_marks_optional_qc_failure_without_science_figure(tmp_path: Path) -> None:
    """附加 CSV 存在但 metadata 不合約時應回報 QC_BLOCKED 且不畫潮相圖。"""

    _month_dir, detect_dir = _build_detect_catalog(tmp_path)
    tide_path = tmp_path / "tidal.csv"
    tide_path.write_text("time_utc_ns,phase\n1725096000000000000,ebb\n", encoding="utf-8")
    # 故意提供不完整 JSON；此情況代表資料檔在場但 provenance/QC 尚未通過。
    tide_path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "literature_qc"
    result = _run_cli(
        "literature-figures",
        "--catalog",
        str(detect_dir / "catalog.json"),
        "--output-root",
        str(output_dir),
        "--tide-table",
        str(tide_path),
    )
    assert result.returncode == 0, result.stderr

    availability = json.loads((output_dir / "figure_availability.json").read_text(encoding="utf-8"))
    tidal = availability["domains"][DOMAIN_ID]["conditional"]["tidal_phase"]
    assert tidal["status"] == "QC_BLOCKED"
    assert not (output_dir / DOMAIN_ID / "tidal_occurrence.png").exists()
    # 其他沒有提供的條件資料仍各自回報缺件，不應被潮相 QC 失敗連帶改寫。
    assert availability["domains"][DOMAIN_ID]["conditional"]["trajectory"]["status"] == "DATA_REQUIRED"
    assert availability["domains"][DOMAIN_ID]["conditional"]["marine_debris"]["status"] == "DATA_REQUIRED"


def test_literature_figures_rejects_changed_source_metadata(tmp_path: Path) -> None:
    """catalog 引用的月份 metadata 被改動後，來源 SHA gate 應拒絕整批圖組。"""

    month_dir, detect_dir = _build_detect_catalog(tmp_path)
    metadata_path = month_dir / "metadata.json"
    original = metadata_path.read_text(encoding="utf-8")
    metadata_path.write_text(original + "\n", encoding="utf-8")
    result = _run_cli(
        "literature-figures",
        "--catalog",
        str(detect_dir / "catalog.json"),
        "--output-root",
        str(tmp_path / "literature_blocked"),
    )
    assert result.returncode != 0
    assert "month metadata SHA-256 不符" in result.stderr


def test_literature_figures_rejects_duplicate_domain_catalog(tmp_path: Path) -> None:
    """同一 flow domain 重複輸入時，CLI 不應把同一證據算成兩區。"""

    _month_dir, detect_dir = _build_detect_catalog(tmp_path)
    catalog_path = detect_dir / "catalog.json"
    result = _run_cli(
        "literature-figures",
        "--catalog",
        str(catalog_path),
        str(catalog_path),
        "--output-root",
        str(tmp_path / "literature_duplicate"),
    )
    assert result.returncode != 0
    assert "不可重複提供同一 flow domain" in result.stderr
