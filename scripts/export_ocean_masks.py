#!/usr/bin/env python3
"""CLI：把 SERVER 端海岸線校正網格批次匯出成 TRAP analysis ocean mask sidecar。

用法範例（於 repo 根目錄執行）::

    python3 scripts/export_ocean_masks.py \\
        --surface-root /data/OCM-Preprocessed-Data/preprocessed/ocm_surface \\
        --corrected-root /data/OCM-Preprocessed-Data/preprocessed/ocm_surface_coastline_corrected_v2 \\
        --output-root /data/OCM-Preprocessed-Data/preprocessed/analysis_ocean_masks \\
        --input-manifest /data/OCM-Preprocessed-Data/preprocessed/ocm_surface_coastline_corrected_v2/coastline_corrected_surface_input_manifest.json

不指定 ``--domain-id`` 時預設處理四個研究區域 domain。每個 domain 成功後會印出
一行 JSON（scope、cell 計數與輸出路徑）；任何 domain 失敗都會印出錯誤到 stderr
並以非零狀態碼結束，不影響其他已成功 domain 的輸出檔案。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path and (_SRC_DIR / "traps").is_dir():
    # 允許在未 `pip install -e .` 的環境下直接以 repo 根目錄執行本腳本。
    sys.path.insert(0, str(_SRC_DIR))

from traps.ocean_mask import export_ocean_mask_from_corrected_grid  # noqa: E402

DEFAULT_DOMAIN_IDS = (
    "northeast_taiwan_common_cache_v3",
    "hsinchu_cache_v3",
    "houwan_nmmba_cache_v3",
    "lienchiang_common_cache_v3",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析命令列參數。"""

    parser = argparse.ArgumentParser(
        description="把 SERVER 端 coastline_corrected_v2 校正網格匯出成 TRAP analysis ocean mask sidecar。",
    )
    parser.add_argument(
        "--surface-root",
        required=True,
        type=Path,
        help="標準 OCM surface 前處理根目錄，內含 <domain_id>/grid。",
    )
    parser.add_argument(
        "--corrected-root",
        required=True,
        type=Path,
        help="海岸線校正輸出根目錄，內含 <domain_id>/grid。",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=None,
        help="可選的 coastline_corrected_surface_input_manifest.json 路徑，提供時會核對雜湊與格點計數。",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="sidecar 輸出根目錄；每個 domain 會寫入 <output-root>/<domain_id>/。",
    )
    parser.add_argument(
        "--domain-id",
        dest="domain_ids",
        action="append",
        default=None,
        help="要處理的 flow-domain ID；可重複指定。預設處理四個研究區域 domain。",
    )
    parser.add_argument(
        "--tag",
        default="coastline_corrected_v2",
        help="sidecar 檔名版本標籤，預設 coastline_corrected_v2。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """執行批次匯出；回傳 process exit code（任何 domain 失敗即非零）。"""

    args = _parse_args(argv)
    domain_ids = args.domain_ids or list(DEFAULT_DOMAIN_IDS)

    failures: list[str] = []
    for domain_id in domain_ids:
        surface_grid_dir = args.surface_root / domain_id / "grid"
        corrected_grid_dir = args.corrected_root / domain_id / "grid"
        output_dir = args.output_root / domain_id
        try:
            metadata = export_ocean_mask_from_corrected_grid(
                surface_grid_dir=surface_grid_dir,
                corrected_grid_dir=corrected_grid_dir,
                domain_id=domain_id,
                output_dir=output_dir,
                tag=args.tag,
                input_manifest_path=args.input_manifest,
            )
        except (ValueError, OSError) as error:
            failures.append(domain_id)
            print(
                json.dumps({"domain_id": domain_id, "status": "failed", "error": str(error)}, ensure_ascii=False),
                file=sys.stderr,
            )
            continue

        arrays = metadata["arrays"]
        print(
            json.dumps(
                {
                    "domain_id": domain_id,
                    "status": "ok",
                    "shape_lat_lon": arrays["shape_lat_lon"],
                    "static_ocean_cell_count": arrays["static_ocean_cell_count"],
                    "analysis_ocean_cell_count": arrays["analysis_ocean_cell_count"],
                    "coastline_land_cells_inside_static_mask": arrays["coastline_land_cells_inside_static_mask"],
                    "outputs": metadata["outputs"],
                },
                ensure_ascii=False,
            )
        )

    if failures:
        print(f"匯出失敗的 domain: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
