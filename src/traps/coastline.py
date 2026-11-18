"""CRS84 GeoJSON 海岸線的可追溯視覺疊圖介面。

本模組只把海岸線當作成果圖上的輪廓線。GeoJSON 沒有提供水深、潮位、分析
海洋遮罩或流場邊界，因此載入與繪圖都不會改變 OCM ``SurfaceMonth``、TRAP
偵測、研究區域或統計分母。每條 ring 會先通過 CRS、座標有限性與閉合性檢查，
繪圖時再依圖面 bbox 裁切；這個順序可避免域外 polygon 自動放大 matplotlib
軸範圍，也可在南灣等較小 flow domain 中保留可讀的岸線輪廓。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np


class CoastlineValidationError(ValueError):
    """GeoJSON 海岸線不符合 CRS84 或幾何資料契約時的明確例外。"""


@dataclass(frozen=True, slots=True)
class CoastlineGeometry:
    """已驗證的 CRS84 海岸線及其 provenance 摘要。

    ``rings`` 保留所有 Polygon 外環與內環，且每個陣列都是 ``(N, 2)`` 的
    ``[longitude, latitude]`` 閉合座標；它僅供輪廓疊圖使用。其他欄位可由
    :meth:`to_metadata` 直接序列化為 JSON，方便 CLI manifest 記錄實際檔案、
    SHA-256、CRS、feature/ring 數及全檔案 bounds。
    """

    path: str
    sha256: str
    crs: str
    feature_count: int
    ring_count: int
    point_count: int
    bounds: tuple[float, float, float, float]
    rings: tuple[np.ndarray, ...] = field(repr=False, compare=False)

    @property
    def visual_overlay_only(self) -> bool:
        """固定回傳視覺疊圖界線，避免被誤當成物理海岸資料。"""

        return True

    def to_metadata(self) -> dict[str, Any]:
        """回傳不含 numpy 陣列的 JSON 友善 provenance。"""

        return {
            "path": self.path,
            "sha256": self.sha256,
            "crs": self.crs,
            "feature_count": self.feature_count,
            "ring_count": self.ring_count,
            "point_count": self.point_count,
            "bounds": list(self.bounds),
            "visual_overlay_only": True,
            "source_semantics": "CRS84 vector coastline outline only; no bathymetry, tide, or ocean-mask semantics",
        }


_CRS84_NAMES = frozenset(
    {
        "CRS84",
        "OGC:CRS84",
        "urn:ogc:def:crs:OGC:1.3:CRS84",
        "urn:ogc:def:crs:OGC::CRS84",
    }
)


def _sha256(path: Path) -> str:
    """以分塊方式計算輸入 GeoJSON，避免讀大檔時複製不必要的 bytes。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    """集中把輸入錯誤轉成可由 CLI 顯示的 domain-specific 例外。"""

    if not condition:
        raise CoastlineValidationError(message)


def _crs84_name(document: dict[str, Any]) -> str:
    """解析 GeoJSON 舊式 ``crs.name``，並拒絕軸順序不明的 CRS。

    來源檔明確使用 ``urn:ogc:def:crs:OGC:1.3:CRS84``。即使 EPSG:4326 在許多
    軟體中常被當作經緯度，本研究仍拒絕它，因為 EPSG:4326 的軸順序語義可能
    不同，不能在沒有額外轉換證據時直接套用到地理圖。
    """

    crs = document.get("crs")
    _require(isinstance(crs, dict), "GeoJSON 必須明確提供 CRS84 crs object")
    _require(crs.get("type") == "name", "GeoJSON crs.type 必須是 name")
    properties = crs.get("properties")
    _require(isinstance(properties, dict), "GeoJSON crs.properties 必須是 object")
    name = properties.get("name")
    _require(isinstance(name, str) and name in _CRS84_NAMES, f"只接受 CRS84 GeoJSON，收到 {name!r}")
    return name


def _as_ring(raw_ring: Any, *, feature_index: int, polygon_index: int, ring_index: int) -> np.ndarray:
    """驗證單一 ring 的形狀、有限性、經緯度範圍與閉合性。"""

    _require(isinstance(raw_ring, list), f"feature {feature_index} polygon {polygon_index} ring {ring_index} 必須是座標清單")
    _require(len(raw_ring) >= 4, f"feature {feature_index} polygon {polygon_index} ring {ring_index} 至少需要四個座標")
    coordinates: list[tuple[float, float]] = []
    for point_index, raw_point in enumerate(raw_ring):
        _require(
            isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2,
            f"feature {feature_index} polygon {polygon_index} ring {ring_index} point {point_index} 必須是 [lon, lat]",
        )
        try:
            lon = float(raw_point[0])
            lat = float(raw_point[1])
        except (TypeError, ValueError) as error:
            raise CoastlineValidationError(
                f"feature {feature_index} polygon {polygon_index} ring {ring_index} point {point_index} 經緯度不可轉為數值"
            ) from error
        _require(np.isfinite(lon) and np.isfinite(lat), f"feature {feature_index} ring {ring_index} 含非有限經緯度")
        _require(-180.0 <= lon <= 180.0, f"feature {feature_index} ring {ring_index} 經度超出 [-180, 180]")
        _require(-90.0 <= lat <= 90.0, f"feature {feature_index} ring {ring_index} 緯度超出 [-90, 90]")
        coordinates.append((lon, lat))
    ring = np.asarray(coordinates, dtype=np.float64)
    _require(np.array_equal(ring[0], ring[-1]), f"feature {feature_index} polygon {polygon_index} ring {ring_index} 必須閉合")
    return ring


def _geometry_rings(geometry: Any, *, feature_index: int) -> list[np.ndarray]:
    """展開 Polygon/MultiPolygon 成閉合 ring，維持 GeoJSON 原始幾何語義。"""

    _require(isinstance(geometry, dict), f"feature {feature_index} geometry 必須是 object")
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    rings: list[np.ndarray] = []
    if geometry_type == "Polygon":
        _require(isinstance(coordinates, list), f"feature {feature_index} Polygon coordinates 必須是清單")
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        _require(isinstance(coordinates, list), f"feature {feature_index} MultiPolygon coordinates 必須是清單")
        polygons = coordinates
    else:
        raise CoastlineValidationError(f"feature {feature_index} 只接受 Polygon/MultiPolygon，收到 {geometry_type!r}")
    _require(bool(polygons), f"feature {feature_index} 幾何不可為空")
    for polygon_index, polygon in enumerate(polygons):
        _require(isinstance(polygon, list) and bool(polygon), f"feature {feature_index} polygon {polygon_index} 必須含 ring")
        for ring_index, raw_ring in enumerate(polygon):
            rings.append(_as_ring(raw_ring, feature_index=feature_index, polygon_index=polygon_index, ring_index=ring_index))
    return rings


def load_coastline(path: str | Path) -> CoastlineGeometry:
    """載入並嚴格驗證 CRS84 Polygon/MultiPolygon 海岸線。

    輸入是 GeoJSON ``FeatureCollection``；輸出保留每個閉合 ring 與可序列化
    provenance。檔案不存在、JSON 損壞、CRS 缺失／非 CRS84、非 Polygon 幾何、
    未閉合 ring、非有限值或超出經緯度範圍都會拒絕。此函式不進行座標重投影，
    因而不會靜默把 EPSG:4326 或其他 CRS 當成 CRS84。
    """

    coastline_path = Path(path).expanduser()
    _require(coastline_path.is_file(), f"海岸線 GeoJSON 不存在: {coastline_path}")
    file_sha256 = _sha256(coastline_path)
    try:
        document = json.loads(coastline_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CoastlineValidationError(f"海岸線 GeoJSON 無法讀取: {coastline_path}") from error
    _require(isinstance(document, dict) and document.get("type") == "FeatureCollection", "海岸線必須是 GeoJSON FeatureCollection")
    crs_name = _crs84_name(document)
    features = document.get("features")
    _require(isinstance(features, list) and bool(features), "海岸線 FeatureCollection 不可沒有 feature")
    rings: list[np.ndarray] = []
    for feature_index, feature in enumerate(features):
        _require(isinstance(feature, dict) and feature.get("type", "Feature") == "Feature", f"feature {feature_index} 必須是 GeoJSON Feature")
        rings.extend(_geometry_rings(feature.get("geometry"), feature_index=feature_index))
    _require(bool(rings), "海岸線沒有可繪製 ring")
    all_points = np.vstack(rings)
    bounds = (
        float(np.min(all_points[:, 0])),
        float(np.min(all_points[:, 1])),
        float(np.max(all_points[:, 0])),
        float(np.max(all_points[:, 1])),
    )
    return CoastlineGeometry(
        path=str(coastline_path.resolve()),
        sha256=file_sha256,
        crs=crs_name,
        feature_count=len(features),
        ring_count=len(rings),
        point_count=int(all_points.shape[0]),
        bounds=bounds,
        rings=tuple(rings),
    )


def _validate_bbox(bbox: Sequence[float], *, name: str = "bbox") -> tuple[float, float, float, float]:
    """驗證 ``(min_x, min_y, max_x, max_y)``，避免反向或無限圖面。"""

    # 呼叫端常直接傳入 numpy 軸運算結果；因此不限制為 list/tuple，只要求它
    # 能提供四個可轉成 float 的值，避免把合法的 ndarray 誤判為不支援的型別。
    try:
        _require(len(bbox) == 4, f"{name} 必須是四元素 bbox")
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError) as error:
        raise CoastlineValidationError(f"{name} 必須含有限數值") from error
    _require(all(np.isfinite(value) for value in values), f"{name} 必須含有限數值")
    _require(values[0] < values[2] and values[1] < values[3], f"{name} 必須滿足 min < max")
    return values


def _clip_segment(start: np.ndarray, end: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    """以 Liang-Barsky 算法裁切線段，回傳 bbox 內的端點。"""

    xmin, ymin, xmax, ymax = bbox
    delta = end - start
    t_min, t_max = 0.0, 1.0
    constraints = ((-delta[0], start[0] - xmin), (delta[0], xmax - start[0]), (-delta[1], start[1] - ymin), (delta[1], ymax - start[1]))
    for p_value, q_value in constraints:
        if abs(float(p_value)) <= 1.0e-15:
            if q_value < 0.0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            if ratio > t_max:
                return None
            t_min = max(t_min, ratio)
        else:
            if ratio < t_min:
                return None
            t_max = min(t_max, ratio)
    if t_min > t_max:
        return None
    return start + t_min * delta, start + t_max * delta


def _clip_ring(ring: np.ndarray, bbox: tuple[float, float, float, float]) -> list[np.ndarray]:
    """把閉合 ring 分割成 bbox 內的連續折線片段。"""

    fragments: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for start, end in pairwise(ring):
        clipped = _clip_segment(start, end, bbox)
        if clipped is None:
            if len(current) >= 2:
                fragments.append(np.asarray(current, dtype=np.float64))
            current = []
            continue
        first, last = clipped
        if not current or not np.allclose(current[-1], first, atol=1.0e-12, rtol=0.0):
            if len(current) >= 2:
                fragments.append(np.asarray(current, dtype=np.float64))
            current = [first]
        if not np.allclose(current[-1], last, atol=1.0e-12, rtol=0.0):
            current.append(last)
    if len(current) >= 2:
        fragments.append(np.asarray(current, dtype=np.float64))
    return fragments


def _draw_fragments(axis: Any, fragments: Iterable[np.ndarray], *, color: str, linewidth: float, zorder: float) -> int:
    """將已裁切片段繪成 outline，並停用 autoscale 防止域外線段改變軸。"""

    _require(np.isfinite(linewidth) and linewidth > 0.0, "linewidth 必須是正有限值")
    drawn = 0
    for fragment in fragments:
        # 完整 ring 的首尾本來就相同；只有所有點完全重疊時才是零長度片段。
        if fragment.shape[0] < 2 or np.allclose(fragment, fragment[0], atol=1.0e-12, rtol=0.0):
            continue
        axis.plot(
            fragment[:, 0],
            fragment[:, 1],
            color=color,
            linewidth=linewidth,
            zorder=zorder,
            clip_on=True,
            scalex=False,
            scaley=False,
        )
        drawn += 1
    return drawn


def draw_geographic_coastline(
    axis: Any,
    coastline: CoastlineGeometry,
    *,
    bbox: Sequence[float],
    color: str = "#263238",
    linewidth: float = 0.7,
    zorder: float = 6.0,
) -> int:
    """在經緯度軸上繪製 bbox 內的海岸線輪廓。

    ``bbox`` 為 ``(min_lon, min_lat, max_lon, max_lat)``，只控制要繪製的地理
    範圍，不會呼叫 ``set_xlim``／``set_ylim``。因此呼叫端原有的 OCM flow-domain
    軸界線保持不變；回傳值是實際畫出的折線片段數，零表示該圖面沒有相交
    coastline ring。
    """

    _require(isinstance(coastline, CoastlineGeometry), "coastline 必須是 CoastlineGeometry")
    geographic_bbox = _validate_bbox(bbox)
    fragments = (fragment for ring in coastline.rings for fragment in _clip_ring(ring, geographic_bbox))
    return _draw_fragments(axis, fragments, color=color, linewidth=linewidth, zorder=zorder)


LAND_FILL_COLOUR = "#d9d9d9"
"""陸地填色（淺灰，與 Lagrangian-Ensemble-Backtracking 的流場範圍圖一致）；海岸線描邊畫在其上。"""


def draw_land_fill(
    axis: Any,
    coastline: CoastlineGeometry,
    *,
    bbox: Sequence[float],
    facecolor: str = LAND_FILL_COLOUR,
    zorder: float = 0.5,
) -> int:
    """在經緯度軸上以 ``facecolor`` 填滿 bbox 內的陸地多邊形（只填色、不描邊；海岸線另由 :func:`draw_geographic_coastline` 疊在其上）。

    海岸線檔的 ring 皆為外環（順時針、沒有內環），整批合成一條複合路徑以非零纏繞規則填色。與 bbox 不相交的 ring 略過，相交者以
    ``Path.clip_to_bbox`` 裁到 bbox 內，向量輸出不帶整條台灣本島的頂點。回傳實際填色的 ring 數；不改動軸範圍。
    """

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath
    from matplotlib.transforms import Bbox

    _require(isinstance(coastline, CoastlineGeometry), "coastline 必須是 CoastlineGeometry")
    geographic_bbox = _validate_bbox(bbox)
    clip_box = Bbox([[geographic_bbox[0], geographic_bbox[1]], [geographic_bbox[2], geographic_bbox[3]]])
    vertices: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    filled = 0
    for ring in coastline.rings:
        ring_min, ring_max = ring.min(axis=0), ring.max(axis=0)
        if ring_max[0] < geographic_bbox[0] or ring_min[0] > geographic_bbox[2] or ring_max[1] < geographic_bbox[1] or ring_min[1] > geographic_bbox[3]:
            continue
        inside = bool(ring_min[0] >= geographic_bbox[0] and ring_max[0] <= geographic_bbox[2] and ring_min[1] >= geographic_bbox[1] and ring_max[1] <= geographic_bbox[3])
        path = MplPath(ring, closed=True)
        if not inside:
            path = path.clip_to_bbox(clip_box)
        if len(path.vertices) < 3:
            continue
        vertices.append(np.asarray(path.vertices, dtype=np.float64))
        codes.append(np.asarray(path.codes, dtype=np.uint8) if path.codes is not None else np.r_[MplPath.MOVETO, [MplPath.LINETO] * (len(path.vertices) - 2), MplPath.CLOSEPOLY].astype(np.uint8))
        filled += 1
    if filled:
        patch = PathPatch(MplPath(np.concatenate(vertices), np.concatenate(codes)), facecolor=facecolor, edgecolor="none", linewidth=0.0, zorder=zorder, clip_on=True)
        axis.add_patch(patch)
    return filled


def _monotonic_axis(value: Any, *, name: str, from_grid: bool = False) -> np.ndarray:
    """取出一維且嚴格單調的軸；支援 OCM 的軸對齊二維 grid。"""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and from_grid:
        if name.startswith("x"):
            _require(np.allclose(array, array[0:1, :], atol=1.0e-6, rtol=0.0), f"{name} 必須是軸對齊格網")
            array = array[0, :]
        else:
            _require(np.allclose(array, array[:, 0:1], atol=1.0e-6, rtol=0.0), f"{name} 必須是軸對齊格網")
            array = array[:, 0]
    _require(array.ndim == 1 and array.size >= 2, f"{name} 必須是一維且至少含兩點")
    _require(np.all(np.isfinite(array)), f"{name} 必須全部有限")
    differences = np.diff(array)
    _require(np.all(differences > 0.0) or np.all(differences < 0.0), f"{name} 必須嚴格單調")
    return array


def draw_metric_coastline(
    axis: Any,
    coastline: CoastlineGeometry,
    *,
    lon_axis: Any,
    lat_axis: Any,
    x_axis: Any,
    y_axis: Any,
    color: str = "#263238",
    linewidth: float = 0.7,
    zorder: float = 6.0,
) -> int:
    """使用同一 OCM snapshot 的軸，把 CRS84 岸線疊到局部公尺圖。

    ``lon_axis``／``lat_axis`` 是 WGS84 直線軸，``x_axis``／``y_axis`` 是同一
    snapshot 的局部公尺軸；後兩者也可傳入 OCM 的二維軸對齊 ``x_m``／``y_m``。
    對應的地理軸與公尺軸必須等長，否則無法逐格建立可信的座標映射。
    先以經緯度 domain bbox 裁切，再以線性內插映射，避免域外岸線被 ``interp``
    夾到邊界而製造假海岸。函式不調整 matplotlib 軸範圍，回傳畫出的公尺折線片段數。
    """

    _require(isinstance(coastline, CoastlineGeometry), "coastline 必須是 CoastlineGeometry")
    lon_values = _monotonic_axis(lon_axis, name="lon_axis")
    lat_values = _monotonic_axis(lat_axis, name="lat_axis")
    x_values = _monotonic_axis(x_axis, name="x_axis", from_grid=True)
    y_values = _monotonic_axis(y_axis, name="y_axis", from_grid=True)
    # 地理與公尺軸是同一網格的兩種座標表示；長度不同時即使各自單調，
    # 內插也會把海岸線映到錯誤位置，必須在實際繪圖前明確拒絕。
    _require(lon_values.size == x_values.size, "lon_axis 與 x_axis 長度必須相同")
    _require(lat_values.size == y_values.size, "lat_axis 與 y_axis 長度必須相同")
    geographic_bbox = (float(np.min(lon_values)), float(np.min(lat_values)), float(np.max(lon_values)), float(np.max(lat_values)))
    metric_bbox = (float(np.min(x_values)), float(np.min(y_values)), float(np.max(x_values)), float(np.max(y_values)))

    # np.interp 需要遞增輸入；輸出軸可遞減，仍能保留 snapshot 自身的座標方向。
    lon_for_interp, x_for_interp = (lon_values, x_values) if lon_values[0] < lon_values[-1] else (lon_values[::-1], x_values[::-1])
    lat_for_interp, y_for_interp = (lat_values, y_values) if lat_values[0] < lat_values[-1] else (lat_values[::-1], y_values[::-1])
    metric_fragments: list[np.ndarray] = []
    for ring in coastline.rings:
        for fragment in _clip_ring(ring, geographic_bbox):
            mapped = np.column_stack((np.interp(fragment[:, 0], lon_for_interp, x_for_interp), np.interp(fragment[:, 1], lat_for_interp, y_for_interp)))
            # 轉換後再次裁切，處理浮點內插在 bbox 邊界留下的微小超界值。
            metric_fragments.extend(_clip_ring(mapped, metric_bbox))
    return _draw_fragments(axis, metric_fragments, color=color, linewidth=linewidth, zorder=zorder)


__all__ = [
    "CoastlineGeometry",
    "CoastlineValidationError",
    "draw_geographic_coastline",
    "draw_metric_coastline",
    "load_coastline",
]
