"""從 SERVER 端已有的海岸線校正網格匯出分析用海洋遮罩 sidecar。

SERVER 上 ``ocm_surface_coastline_corrected_v2`` 目錄已經把海岸線多邊形套用在
原始 ``ocm_surface`` 靜態網格上，產生「校正後」``mask_static.npy``（原始
``mask_static`` 與 ``~coastline_land_mask`` 的交集）。本模組不重新計算任何海岸
幾何，只負責：

1. 核對兩份網格（標準輸入網格與校正網格）的座標、shape、domain 與 schema 完全
   一致，且校正遮罩確實是原始遮罩的子集；
2. 在有提供時，核對校正 metadata 內記錄的來源 hash，以及（若提供）上游
   manifest 內對應 region 的 hash 與格點計數，確保匯出內容與 SERVER 端 manifest
   所描述的產物一致；
3. 把校正遮罩原封不動匯出成 :mod:`traps.ocm` 認得的
   ``analysis_ocean_mask_<tag>.npy`` / ``analysis_land_mask_<tag>.npy`` /
   ``analysis_land_mask_<tag>.json`` sidecar 三件組。

本模組不修改、不重新分類任何海洋/陸地格點；輸出的遮罩在數值上與校正網格的
``mask_static.npy`` 逐位元相同，只是換成 :func:`traps.ocm.open_month` 的
``analysis_ocean_mask_path`` 介面所要求的檔名與 metadata 格式。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = "3.0.0"
"""本模組要求輸入與輸出 metadata 都遵守的 schema 版本，與 :mod:`traps.ocm` 一致。"""

_GRID_ARRAY_FILES = ("lon.npy", "lat.npy", "x_m.npy", "y_m.npy")
"""兩份網格必須逐位元相同的座標陣列檔名。"""


def _require(condition: bool, message: str) -> None:
    """在 sidecar 匯出前置條件不成立時立即拋出 ``ValueError``。"""

    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    """以分塊方式計算檔案的 SHA-256，避免把大型陣列整批讀入記憶體。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npy_full(path: Path) -> np.ndarray:
    """讀取一個必要的 ``.npy`` 陣列；缺檔或非 ndarray 一律 fail closed。"""

    _require(path.is_file(), f"缺少必要陣列: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"無法讀取 {path}: {error}") from error
    _require(isinstance(array, np.ndarray), f"{path} 必須是 ndarray")
    return array


def _load_grid_metadata(grid_dir: Path) -> dict[str, Any]:
    """讀取網格目錄的 ``metadata.json``。"""

    metadata_path = grid_dir / "metadata.json"
    _require(metadata_path.is_file(), f"缺少網格 metadata: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"無法讀取網格 metadata: {metadata_path}: {error}") from error
    _require(isinstance(payload, dict), f"{metadata_path} 必須是 JSON 物件")
    return payload


def _require_domain_and_schema(metadata: Mapping[str, Any], grid_dir: Path, domain_id: str) -> None:
    """核對網格 metadata 的 schema 版本與 domain_id。"""

    _require(
        metadata.get("cache_schema_version") == CACHE_SCHEMA_VERSION,
        f"{grid_dir} 的 cache_schema_version 必須是 {CACHE_SCHEMA_VERSION}",
    )
    domain = metadata.get("domain")
    _require(isinstance(domain, Mapping), f"{grid_dir} 的 metadata.domain 必須是物件")
    _require(
        domain.get("domain_id") == domain_id,
        f"{grid_dir} 的 metadata.domain.domain_id 與要求的 {domain_id!r} 不符",
    )


def _find_manifest_region(manifest: Mapping[str, Any], domain_id: str, manifest_path: Path) -> Mapping[str, Any]:
    """從 SERVER manifest 找出對應 ``flow_domain_id`` 的 region 描述。"""

    regions = manifest.get("regions")
    _require(isinstance(regions, list), f"{manifest_path} 缺少 regions 清單")
    matches = [region for region in regions if isinstance(region, Mapping) and region.get("flow_domain_id") == domain_id]
    _require(len(matches) == 1, f"{manifest_path} 內找不到唯一符合 flow_domain_id={domain_id!r} 的 region")
    return matches[0]


def export_ocean_mask_from_corrected_grid(
    *,
    surface_grid_dir: str | Path,
    corrected_grid_dir: str | Path,
    domain_id: str,
    output_dir: str | Path,
    tag: str = "coastline_corrected_v2",
    input_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """把 SERVER 端海岸線校正網格匯出成 TRAP analysis ocean mask sidecar。

    參數
    ------
    surface_grid_dir:
        標準（未校正）OCM surface 網格目錄，即 ``ocm_surface/<domain_id>/grid``。
    corrected_grid_dir:
        校正網格目錄，即
        ``ocm_surface_coastline_corrected_v2/<domain_id>/grid``。
    domain_id:
        期望的 OCM flow-domain ID；兩份網格的 metadata 都必須與此一致。
    output_dir:
        sidecar 輸出目錄；必須是新目錄或現存空目錄，避免悄悄覆寫既有 sidecar。
    tag:
        sidecar 檔名版本標籤，預設 ``coastline_corrected_v2``。
    input_manifest_path:
        可選的 SERVER ``coastline_corrected_surface_input_manifest.json``；提供時
        會核對其中對應 region 的雜湊與格點計數。

    回傳
    ------
    dict
        寫入 ``analysis_land_mask_<tag>.json`` 的完整 metadata 內容。

    驗證與輸出行為見模組docstring；任何不一致都會在寫檔前以 ``ValueError``
    fail closed，不會留下部分寫入的 sidecar。
    """

    surface_dir = Path(surface_grid_dir)
    corrected_dir = Path(corrected_grid_dir)
    out_dir = Path(output_dir)

    surface_metadata = _load_grid_metadata(surface_dir)
    corrected_metadata = _load_grid_metadata(corrected_dir)
    _require_domain_and_schema(surface_metadata, surface_dir, domain_id)
    _require_domain_and_schema(corrected_metadata, corrected_dir, domain_id)

    # --- 座標一致性：兩份網格必須描述同一個實體格網 ---
    for filename in _GRID_ARRAY_FILES:
        surface_array = _load_npy_full(surface_dir / filename)
        corrected_array = _load_npy_full(corrected_dir / filename)
        _require(
            surface_array.shape == corrected_array.shape,
            f"{filename} 在兩份網格間的 shape 不一致: {surface_array.shape} vs {corrected_array.shape}",
        )
        _require(
            np.array_equal(surface_array, corrected_array),
            f"{filename} 在兩份網格間必須逐位元相同",
        )

    surface_mask_path = surface_dir / "mask_static.npy"
    corrected_mask_path = corrected_dir / "mask_static.npy"
    surface_static = _load_npy_full(surface_mask_path)
    corrected_static = _load_npy_full(corrected_mask_path)
    _require(surface_static.dtype == np.dtype(bool), f"{surface_mask_path} dtype 必須是 bool")
    _require(corrected_static.dtype == np.dtype(bool), f"{corrected_mask_path} dtype 必須是 bool")
    _require(surface_static.shape == corrected_static.shape, "兩份網格的 mask_static shape 不一致")
    _require(not np.any(corrected_static & ~surface_static), "校正後 mask_static 必須是原始 mask_static 的子集")

    original_path = corrected_dir / "mask_static_original.npy"
    if original_path.is_file():
        original_array = _load_npy_full(original_path)
        _require(
            original_array.dtype == np.dtype(bool) and np.array_equal(original_array, surface_static),
            f"{original_path} 必須與標準網格 mask_static 相同",
        )

    land_mask_path = corrected_dir / "coastline_land_mask.npy"
    if land_mask_path.is_file():
        land_mask = _load_npy_full(land_mask_path)
        _require(land_mask.dtype == np.dtype(bool) and land_mask.shape == surface_static.shape, f"{land_mask_path} shape/dtype 不符")
        _require(
            np.array_equal(corrected_static, surface_static & ~land_mask),
            "校正後 mask_static 必須等於 (標準 mask_static & ~coastline_land_mask)",
        )

    surface_mask_sha256 = _sha256_file(surface_mask_path)
    corrected_mask_sha256 = _sha256_file(corrected_mask_path)

    coastline_correction = corrected_metadata.get("coastline_correction")
    if isinstance(coastline_correction, Mapping):
        declared_source_sha256 = coastline_correction.get("source_mask_static_sha256")
        if declared_source_sha256 is not None:
            _require(
                declared_source_sha256 == surface_mask_sha256,
                "校正 metadata 的 source_mask_static_sha256 與標準網格 mask_static 檔案雜湊不符",
            )

    manifest_payload: dict[str, Any] | None = None
    manifest_sha256: str | None = None
    manifest_path: Path | None = None
    if input_manifest_path is not None:
        manifest_path = Path(input_manifest_path)
        _require(manifest_path.is_file(), f"找不到 input manifest: {manifest_path}")
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"無法讀取 input manifest: {manifest_path}: {error}") from error
        _require(isinstance(manifest_payload, dict), f"{manifest_path} 必須是 JSON 物件")
        region = _find_manifest_region(manifest_payload, domain_id, manifest_path)
        _require(
            region.get("corrected_mask_static_sha256") == corrected_mask_sha256,
            f"{manifest_path} 內 region 的 corrected_mask_static_sha256 與校正 mask_static 檔案雜湊不符",
        )
        surface_ocean_cells = int(np.count_nonzero(surface_static))
        corrected_ocean_cells = int(np.count_nonzero(corrected_static))
        _require(
            region.get("original_static_ocean_cell_count") == surface_ocean_cells,
            f"{manifest_path} 內 region 的 original_static_ocean_cell_count 與標準網格格點數不符",
        )
        _require(
            region.get("corrected_static_ocean_cell_count") == corrected_ocean_cells,
            f"{manifest_path} 內 region 的 corrected_static_ocean_cell_count 與校正網格格點數不符",
        )
        manifest_sha256 = _sha256_file(manifest_path)

    _require(not out_dir.exists() or (out_dir.is_dir() and not any(out_dir.iterdir())), f"輸出目錄必須是新目錄或空目錄: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_land_mask = surface_static & ~corrected_static
    shape_lat_lon = [int(value) for value in corrected_static.shape]
    static_ocean_cell_count = int(np.count_nonzero(surface_static))
    analysis_ocean_cell_count = int(np.count_nonzero(corrected_static))
    coastline_land_cells = int(np.count_nonzero(analysis_land_mask))

    ocean_mask_filename = f"analysis_ocean_mask_{tag}.npy"
    land_mask_filename = f"analysis_land_mask_{tag}.npy"
    metadata_filename = f"analysis_land_mask_{tag}.json"
    ocean_mask_out_path = out_dir / ocean_mask_filename
    land_mask_out_path = out_dir / land_mask_filename
    metadata_out_path = out_dir / metadata_filename

    _atomic_save_npy(ocean_mask_out_path, corrected_static)
    _atomic_save_npy(land_mask_out_path, analysis_land_mask)

    sources: dict[str, Any] = {
        "surface_grid_dir": str(surface_dir),
        "corrected_grid_dir": str(corrected_dir),
        "surface_mask_static_sha256": surface_mask_sha256,
        "corrected_mask_static_sha256": corrected_mask_sha256,
        "corrected_grid_metadata_sha256": _sha256_file(corrected_dir / "metadata.json"),
        "coastline_correction": dict(coastline_correction) if isinstance(coastline_correction, Mapping) else None,
        "input_manifest_path": str(manifest_path) if manifest_path is not None else None,
        "input_manifest_sha256": manifest_sha256,
    }

    metadata_document: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_kind": "coastline_corrected_export",
        "version_tag": tag,
        "domain": dict(surface_metadata.get("domain", {})),
        "arrays": {
            "shape_lat_lon": shape_lat_lon,
            "static_ocean_cell_count": static_ocean_cell_count,
            "analysis_ocean_cell_count": analysis_ocean_cell_count,
            "coastline_land_cells_inside_static_mask": coastline_land_cells,
        },
        "sources": sources,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "semantics": (
            "本 sidecar 原封不動重新匯出 SERVER 端 coastline_corrected_v2 校正後的"
            "海洋遮罩，未做任何數值修改；只定義哪些格點屬於數值上有效的海域，"
            "不會改變任何月份的 u/v/valid 陣列內容。"
        ),
        "outputs": {
            "analysis_ocean_mask_path": str(ocean_mask_out_path),
            "analysis_land_mask_path": str(land_mask_out_path),
            "metadata_path": str(metadata_out_path),
        },
    }

    _atomic_write_text(metadata_out_path, json.dumps(metadata_document, ensure_ascii=False, indent=2))

    return metadata_document


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """以「暫存檔 + os.replace」方式原子寫入一個 ``.npy`` 陣列。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    # ``suffix=".npy"`` 確保 ``np.save`` 不會另外附加副檔名，暫存檔名因此固定可預期。
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".npy")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.save(tmp_path, array, allow_pickle=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    """以「暫存檔 + os.replace」方式原子寫入文字檔（用於 JSON metadata）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)


__all__ = ["CACHE_SCHEMA_VERSION", "export_ocean_mask_from_corrected_grid"]
