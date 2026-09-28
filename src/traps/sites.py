"""TRAP 四流場區域與五研究站的統計遮罩。

本模組把研究設計中的 flow-domain、anchor 與站名轉成可直接套用在 OCM 規則格網的
``bool`` 遮罩。統計母域永遠是上游完整 flow domain；A 區因貢寮與龜山島共用同一
domain，才依兩個已核定 anchor 的公尺距離做互斥 Voronoi 分割，且 tie 固定歸貢寮。
B、C、D 則把各自完整 domain 的分析靜態海域遮罩交給唯一研究站。這個分割只決定
統計歸屬，不裁切 OCM 流場微分、TRAP 骨架積分或上游 cache 邊界。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
"""WGS84 近似球半徑；此值只用來比較 anchor 距離，不取代上游公尺網格。"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_DESIGN_PATH = PROJECT_ROOT / "configs" / "study_design.json"
"""專案內固定研究設計檔的位置；站點 ID 與名稱不從呼叫端臨時推導。"""


def _require(condition: bool, message: str) -> None:
    """在研究設計或幾何輸入不符合契約時立即停止。"""

    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=4)
def load_study_design(path: str | Path = DEFAULT_STUDY_DESIGN_PATH) -> dict[str, Any]:
    """讀取並驗證 TRAP 四 domain／五站研究設計 JSON。

    設計檔保存了上游 flow-domain bbox、五站 anchor、C 區的「南灣」顯示名稱與仍不可
    變更的 ``houwan_nmmba_cache_v3`` 產品 ID。函式只讀取小型 JSON；來源檔案的 path
    與 SHA-256 由設定檔保留，供報告及 manifest 追溯。回傳的 dict 是解析後資料，不含
    任何大型網格陣列；呼叫端不應在此處修改內容。
    """

    design_path = Path(path)
    _require(design_path.is_file(), f"找不到 TRAP study design: {design_path}")
    try:
        payload = json.loads(design_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"無法讀取 TRAP study design: {design_path}: {error}") from error
    _require(isinstance(payload, dict), "study_design.json 必須是 JSON 物件")
    _require(payload.get("schema_version") == "1.0.0", "study_design.json schema_version 必須是 1.0.0")
    _require(payload.get("source_path_base") == "project_root", "source_references.path 必須相對於 TRAPs 專案根目錄")

    source_references = payload.get("source_references")
    _require(isinstance(source_references, list) and source_references, "study_design.json 缺少 source_references")
    for index, source in enumerate(source_references):
        _require(isinstance(source, Mapping), f"source_references[{index}] 必須是物件")
        _require(isinstance(source.get("path"), str) and source.get("path"), f"source_references[{index}] 缺少 path")
        digest = source.get("sha256")
        _require(isinstance(digest, str) and len(digest) == 64, f"source_references[{index}].sha256 必須是 SHA-256")

    domains = payload.get("flow_domains")
    _require(isinstance(domains, list) and len(domains) == 4, "study_design.json 必須明列四個 flow domain")
    domain_ids: set[str] = set()
    for index, domain in enumerate(domains):
        _require(isinstance(domain, Mapping), f"flow_domains[{index}] 必須是物件")
        domain_id = domain.get("flow_domain_id")
        _require(isinstance(domain_id, str) and domain_id not in domain_ids, f"flow_domains[{index}] domain_id 重複或缺失")
        domain_ids.add(domain_id)
        bbox = domain.get("bbox_lon_lat")
        _require(isinstance(bbox, list) and len(bbox) == 4, f"flow_domains[{index}].bbox_lon_lat 必須有四個值")
        _require(float(bbox[0]) < float(bbox[1]) and float(bbox[2]) < float(bbox[3]), f"flow_domains[{index}] bbox 上下界不合法")

    sites = payload.get("study_sites")
    _require(isinstance(sites, list) and len(sites) == 5, "study_design.json 必須明列五個研究站")
    site_ids: set[str] = set()
    for index, site in enumerate(sites):
        _require(isinstance(site, Mapping), f"study_sites[{index}] 必須是物件")
        site_id = site.get("study_site_id")
        _require(isinstance(site_id, str) and site_id not in site_ids, f"study_sites[{index}] site_id 重複或缺失")
        site_ids.add(site_id)
        _require(site.get("flow_domain_id") in domain_ids, f"study_sites[{index}] 指向不存在的 flow domain")
        anchor = site.get("anchor_lonlat")
        _require(isinstance(anchor, list) and len(anchor) == 2, f"study_sites[{index}].anchor_lonlat 必須有經緯度")
        _require(all(np.isfinite(float(value)) for value in anchor), f"study_sites[{index}].anchor_lonlat 必須有限")

    nanwan = next((site for site in sites if site.get("study_site_id") == "nanwan"), None)
    _require(nanwan is not None, "study_design.json 缺少 nanwan 站")
    _require(nanwan.get("study_site_name_zh") == "南灣", "C 區研究站顯示名稱必須是南灣")
    _require(nanwan.get("flow_domain_id") == "houwan_nmmba_cache_v3", "南灣必須沿用 houwan flow_domain_id")
    return payload


def _grid_coordinates(lon: np.ndarray, lat: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """把一維或二維 lon/lat 轉成與 ``static_mask`` 同 shape 的格點座標。

    上游 OCM cache 保存一維規則座標軸；接受二維 meshgrid 只是為了讓離線測試與報告
    重用相同幾何函式。函式不將座標轉成經度 0–360，也不自動排序，因為排序會使遮罩
    與流場索引失去一一對應。
    """

    if lon.ndim == 1 and lat.ndim == 1:
        _require(shape == (lat.size, lon.size), "一維 lon/lat 長度與 static_mask shape 不符")
        return np.meshgrid(lon, lat, indexing="xy")
    _require(lon.ndim == 2 and lat.ndim == 2 and lon.shape == lat.shape == shape, "二維 lon/lat 必須與 static_mask 同 shape")
    return lon, lat


def _haversine_distance_m(lon: np.ndarray, lat: np.ndarray, anchor_lon: float, anchor_lat: float) -> np.ndarray:
    """以球面 haversine 計算每個格點到 anchor 的公尺距離。

    A 區 domain 約為數十至百公里，球面距離足以穩定表達「最近 anchor」且不依賴額外
    GIS 套件；距離只用來比較兩站，不會被解讀為正式測地學產品或取代 OCM 的 x/y。
    """

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    anchor_lon_rad = np.deg2rad(float(anchor_lon))
    anchor_lat_rad = np.deg2rad(float(anchor_lat))
    dlon = lon_rad - anchor_lon_rad
    dlat = lat_rad - anchor_lat_rad
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(anchor_lat_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def site_masks(
    domain_id: str,
    lon: np.ndarray,
    lat: np.ndarray,
    static_mask: np.ndarray,
    *,
    study_design_path: str | Path = DEFAULT_STUDY_DESIGN_PATH,
) -> dict[str, np.ndarray]:
    """建立五站統計歸屬遮罩，保留完整 flow-domain 母域。

    ``domain_id`` 必須是研究設計中的四個上游產品 ID；``lon``、``lat`` 是規則格網
    一維軸或與 ``static_mask`` 同 shape 的二維 meshgrid，而 ``static_mask`` 必須是
    bool。回傳字典固定包含五個 station ID，非該 domain 的站點遮罩全為 False。

    A 區的 ``northeast_taiwan_common_cache_v3`` 同時包含貢寮與龜山島，兩者以 anchor
    球面距離的最近站分割所有 static 有效格點；兩距離完全相等時使用
    ``d_gongliao <= d_guishan``，因此 tie 固定歸貢寮。B/C/D 不縮小成 12.5 km core，
    各自站點直接得到完整 flow-domain 的 static mask。呼叫端若已明確建立
    ``analysis_ocean_mask``，應把該遮罩傳入此參數，以避免把 coastline sidecar 以外的
    static 格點納入站點分母。
    """

    design = load_study_design(study_design_path)
    domains = {domain["flow_domain_id"]: domain for domain in design["flow_domains"]}
    _require(domain_id in domains, f"不支援的 flow_domain_id: {domain_id!r}")
    static = np.asarray(static_mask)
    _require(static.dtype == np.dtype(bool), "static_mask 必須是 bool")
    _require(static.ndim == 2, "static_mask 必須是 (lat, lon) 二維陣列")
    lon_array = np.asarray(lon)
    lat_array = np.asarray(lat)
    _require(np.issubdtype(lon_array.dtype, np.floating) and np.issubdtype(lat_array.dtype, np.floating), "lon/lat 必須是浮點陣列")
    grid_lon, grid_lat = _grid_coordinates(lon_array, lat_array, static.shape)

    masks = {site_id: np.zeros(static.shape, dtype=bool) for site_id in ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")}
    domain_sites = [site for site in design["study_sites"] if site["flow_domain_id"] == domain_id]
    _require(domain_sites, f"flow domain 沒有對應研究站: {domain_id!r}")
    if domain_id == "northeast_taiwan_common_cache_v3":
        _require({site["study_site_id"] for site in domain_sites} == {"gongliao", "guishan"}, "A 區研究站契約不符")
        anchors = {site["study_site_id"]: site["anchor_lonlat"] for site in domain_sites}
        distance_gongliao = _haversine_distance_m(grid_lon, grid_lat, *anchors["gongliao"])
        distance_guishan = _haversine_distance_m(grid_lon, grid_lat, *anchors["guishan"])
        # <= 明確處理 tie，保證 A 區每個有效格點只歸一站且站點順序可重現。
        masks["gongliao"] = static & (distance_gongliao <= distance_guishan)
        masks["guishan"] = static & ~masks["gongliao"]
    else:
        _require(len(domain_sites) == 1, f"非 A 區 flow domain 應只有一個研究站: {domain_id!r}")
        masks[domain_sites[0]["study_site_id"]] = static.copy()

    if domain_id == "northeast_taiwan_common_cache_v3":
        a_union = masks["gongliao"] | masks["guishan"]
        _require(np.array_equal(a_union, static), "A 區站點遮罩未完整覆蓋 static mask")
    return masks


__all__ = ["DEFAULT_STUDY_DESIGN_PATH", "EARTH_RADIUS_M", "load_study_design", "site_masks"]
