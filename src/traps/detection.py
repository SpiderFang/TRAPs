"""以公尺制表層速度快照檢出 TRAP 核心與 e2 骨架。

實作只接受二維、軸對齊且近似 rectilinear 的公尺格網。速度梯度使用完整的
中心差分 stencil；中心點、四個正交鄰點及必要的局部極小值鄰域只要有一格
無效，就不產生導數或曲線採樣。這個保守規則刻意避免把海岸或缺值當成 0，
也避免曲線穿過遮罩缺口。輸入是 `(row=y, column=x)` 陣列，`u/v` 的單位由
上游 OCM schema 定義，輸出的應變率 `s1/s2` 單位為 `s⁻¹`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .models import DetectionConfig, SnapshotResult, TrapDetection


class _RectilinearGrid(NamedTuple):
    """已驗證的公尺座標軸與格距，供差分及曲線積分共用。"""

    x_axis_m: np.ndarray
    y_axis_m: np.ndarray
    dx_m: float
    dy_m: float


class _Sample(NamedTuple):
    """曲線某一公尺座標的雙線性場採樣值。"""

    s1_sinv: float
    e2_x: float
    e2_y: float


@dataclass(frozen=True, slots=True)
class _BranchResult:
    """e2 單向積分的內部結果，保留端點停止原因。"""

    points_m: np.ndarray
    strengths_sinv: np.ndarray
    stop_reason: str


def detect_snapshot(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    config: DetectionConfig,
) -> SnapshotResult:
    """計算一個表層速度 snapshot 的 TRAP 候選與雙向 e2 曲線。

    參數
    ------
    u, v:
        形狀為 `(ny, nx)` 的東向與北向（或上游契約定義的 x/y）速度陣列。
        無效位置的值不會被填零；即使 `valid=True` 而速度是 NaN，也會被降為
        `input_valid_mask=False`，使相鄰差分和曲線採樣停止。
    valid:
        上游逐時有效遮罩。此遮罩不會被擴張，且任何差分 stencil 或曲線雙線性
        採樣只要跨過 False 格點就拒絕。
    x_m, y_m:
        一維軸或與速度同形的二維座標，單位為 m。二維輸入必須可化為
        `x[row, col] = x_axis[col]`、`y[row, col] = y_axis[row]`；非單調或
        非軸對齊格網會明確丟出 `ValueError`，避免把曲線當成規則公尺格網。
    config:
        所有核心、方向退化、相對強度、單調回升、轉角、弧長和品質門檻。

    回傳
    ------
    SnapshotResult
        保存 `s1/s2`、應變張量分量、`e2`、導數有效遮罩、接受的核心候選及
        每條曲線的弧長與停止原因。結果陣列是唯讀的，方便後續追蹤模組建立
        可回溯的 manifest。

    限制
    ------
    目前只實作無平滑的中心差分。這不是把陸地或缺值補成 0；若設定指定平滑，
    本函式會拒絕執行，待日後加入只在有效海域內的保守平滑後再放開。
    """

    if not isinstance(config, DetectionConfig):
        raise TypeError("config 必須是 DetectionConfig")
    u_array, v_array, valid_array, grid = _validate_inputs(u, v, valid, x_m, y_m, config)
    usable = valid_array & np.isfinite(u_array) & np.isfinite(v_array)
    smoothed_u, smoothed_v = _mask_safe_graph_diffusion(
        u_array,
        v_array,
        usable,
        radius_cells=config.smoothing_radius_cells,
        alpha=config.smoothing_diffusion_alpha,
    )

    sxx, sxy, syy, s1, s2, e2_x, e2_y, derivative_valid = _compute_strain(
        smoothed_u, smoothed_v, usable, grid, config
    )
    candidates = _find_core_candidates(s1, derivative_valid, grid, config)

    detections: list[TrapDetection] = []
    for row, col in candidates:
        detections.append(
            _build_detection(
                row,
                col,
                grid,
                s1,
                e2_x,
                e2_y,
                derivative_valid,
                config,
            )
        )

    core_candidate_mask = np.zeros_like(derivative_valid, dtype=bool)
    for row, col in candidates:
        core_candidate_mask[row, col] = True

    qc_flags: list[str] = []
    if np.any(valid_array & ~usable):
        qc_flags.append("nonfinite_velocity_marked_invalid")
    if not np.any(derivative_valid):
        qc_flags.append("no_valid_derivative_cells")
    if not candidates:
        qc_flags.append("no_negative_local_minimum_core")
    if config.smoothing_radius_cells:
        qc_flags.append("smoothing_applied_graph_diffusion")

    smoothing_sigma = _equivalent_smoothing_sigma_m(grid, config)

    return SnapshotResult(
        s1_sinv=s1,
        s2_sinv=s2,
        sxx_sinv=sxx,
        sxy_sinv=sxy,
        syy_sinv=syy,
        e2_x=e2_x,
        e2_y=e2_y,
        derivative_valid=derivative_valid,
        core_candidate_mask=core_candidate_mask,
        input_valid_mask=usable,
        detections=tuple(detections),
        config=config,
        grid_spacing_m=(grid.dx_m, grid.dy_m),
        qc_flags=tuple(qc_flags),
        smoothing_radius_cells=config.smoothing_radius_cells,
        smoothing_equivalent_sigma_m=smoothing_sigma,
        smoothing_method=(config.smoothing_method if config.smoothing_radius_cells else "none"),
    )


def _validate_inputs(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    config: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _RectilinearGrid]:
    """檢查 shape、有限性與公尺座標，拒絕無法安全差分的輸入。"""

    u_array = np.asarray(u, dtype=np.float64)
    v_array = np.asarray(v, dtype=np.float64)
    valid_array = np.asarray(valid, dtype=bool)
    if u_array.ndim != 2 or v_array.ndim != 2 or valid_array.ndim != 2:
        raise ValueError("u、v、valid 必須是二維 `(ny, nx)` 陣列")
    if u_array.shape != v_array.shape or u_array.shape != valid_array.shape:
        raise ValueError("u、v、valid 的 shape 必須完全一致")
    if min(u_array.shape) < 3:
        raise ValueError("中心差分至少需要 3x3 格點")

    x_raw = np.asarray(x_m, dtype=np.float64)
    y_raw = np.asarray(y_m, dtype=np.float64)
    y_size, x_size = u_array.shape
    x_axis = _extract_axis(x_raw, (y_size, x_size), axis_name="x_m", axis=1, tolerance=config.grid_coordinate_tolerance_m)
    y_axis = _extract_axis(y_raw, (y_size, x_size), axis_name="y_m", axis=0, tolerance=config.grid_coordinate_tolerance_m)
    _validate_monotonic_axis(x_axis, "x_m", config.grid_coordinate_tolerance_m)
    _validate_monotonic_axis(y_axis, "y_m", config.grid_coordinate_tolerance_m)

    dx = float(np.median(np.abs(np.diff(x_axis))))
    dy = float(np.median(np.abs(np.diff(y_axis))))
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("x_m/y_m 必須提供正的公尺格距")
    return u_array, v_array, valid_array, _RectilinearGrid(x_axis, y_axis, dx, dy)


def _mask_safe_graph_diffusion(
    u: np.ndarray,
    v: np.ndarray,
    usable: np.ndarray,
    *,
    radius_cells: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在有效格點四鄰接圖上做有限次 explicit diffusion。

    每次更新採用 ``(1-alpha) * center + alpha * mean(valid neighbours)``。鄰點
    平均的分母只計算同時為 ``usable`` 的格點，因此 mask 外的 NaN 不會被填入，
    一列無效陸地也會切斷兩側連通元件。擴散迭代數等於設定的格點尺度，故資料
    最多沿 graph shortest path 傳播該數量的邊；這是保守的 geodesic 平滑，與
    在完整矩形陣列上固定標準差、會跨越陸地的 Gaussian normalized convolution
    不同。

    回傳的無效位置保持 NaN（或原本的非有限值），而有效位置始終只由同一個
    有效圖元件的有限速度值組成。``alpha=0.5`` 時，均勻內部網格的每軸等效
    Gaussian 標準差約為 ``sqrt(radius/4) * grid_spacing``；邊界與非均勻格網
    會使這個值只具近似意義，實際尺度由結果模型另行記錄。
    """

    if radius_cells == 0:
        # 0 格是敏感度基線；不建立額外副本，保留上游 memory-map 的逐時讀取特性。
        return u, v

    current_u = np.full(u.shape, np.nan, dtype=np.float64)
    current_v = np.full(v.shape, np.nan, dtype=np.float64)
    current_u[usable] = u[usable]
    current_v[usable] = v[usable]
    ny, nx = u.shape
    neighbour_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for _ in range(radius_cells):
        neighbour_sum_u = np.zeros_like(current_u)
        neighbour_sum_v = np.zeros_like(current_v)
        neighbour_count = np.zeros(u.shape, dtype=np.float64)
        for row_offset, col_offset in neighbour_offsets:
            target_rows = slice(max(0, row_offset), min(ny, ny + row_offset))
            target_cols = slice(max(0, col_offset), min(nx, nx + col_offset))
            source_rows = slice(max(0, -row_offset), min(ny, ny - row_offset))
            source_cols = slice(max(0, -col_offset), min(nx, nx - col_offset))
            source_usable = usable[source_rows, source_cols]
            target_sum_u = neighbour_sum_u[target_rows, target_cols]
            target_sum_v = neighbour_sum_v[target_rows, target_cols]
            target_count = neighbour_count[target_rows, target_cols]
            source_u = current_u[source_rows, source_cols]
            source_v = current_v[source_rows, source_cols]
            target_sum_u[source_usable] += source_u[source_usable]
            target_sum_v[source_usable] += source_v[source_usable]
            target_count[source_usable] += 1.0

        updated_u = current_u.copy()
        updated_v = current_v.copy()
        has_neighbour = usable & (neighbour_count > 0.0)
        mean_u = np.divide(
            neighbour_sum_u,
            neighbour_count,
            out=np.zeros_like(neighbour_sum_u),
            where=neighbour_count > 0.0,
        )
        mean_v = np.divide(
            neighbour_sum_v,
            neighbour_count,
            out=np.zeros_like(neighbour_sum_v),
            where=neighbour_count > 0.0,
        )
        updated_u[has_neighbour] = (
            (1.0 - alpha) * current_u[has_neighbour] + alpha * mean_u[has_neighbour]
        )
        updated_v[has_neighbour] = (
            (1.0 - alpha) * current_v[has_neighbour] + alpha * mean_v[has_neighbour]
        )
        current_u = updated_u
        current_v = updated_v

    return current_u, current_v


def _equivalent_smoothing_sigma_m(
    grid: _RectilinearGrid, config: DetectionConfig
) -> tuple[float, float]:
    """以隨機漫步二階矩估計 graph diffusion 的等效 Gaussian sigma。

    四鄰接 explicit step 的單次 x/y 變異數為 ``alpha*h²/2``；乘上有限迭代數
    後得到此近似。這只是尺度記錄，不代表 graph kernel 已經是 Gaussian，尤其
    在海岸和窄通道附近仍會受有效鄰點數與連通幾何影響。
    """

    if config.smoothing_radius_cells == 0:
        return (0.0, 0.0)
    factor = np.sqrt(config.smoothing_diffusion_alpha * config.smoothing_radius_cells / 2.0)
    return (float(factor * grid.dx_m), float(factor * grid.dy_m))


def _extract_axis(
    coordinates: np.ndarray,
    shape: tuple[int, int],
    *,
    axis_name: str,
    axis: int,
    tolerance: float,
) -> np.ndarray:
    """從一維或可分離二維座標抽出單調軸，拒絕曲線或旋轉格網。

    只允許軸對齊 rectilinear 格網，是因為本模組以 `du/dx`、`du/dy` 的公尺中心
    差分定義應變率；若座標是曲線或旋轉格網，直接套用這些分母會產生沒有明確
    物理意義的導數，應由未來專門的 metric-tensor 模組處理。
    """

    y_size, x_size = shape
    expected_length = x_size if axis == 1 else y_size
    if coordinates.ndim == 1:
        if coordinates.size != expected_length:
            raise ValueError(f"{axis_name} 一維長度必須等於速度場對應軸長度")
        extracted = coordinates
    elif coordinates.ndim == 2:
        if coordinates.shape != shape:
            raise ValueError(f"{axis_name} 二維 shape 必須與速度場一致")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"{axis_name} 必須全部是有限公尺座標")
        extracted = coordinates[0, :] if axis == 1 else coordinates[:, 0]
        expected = extracted[None, :] if axis == 1 else extracted[:, None]
        if not np.allclose(coordinates, expected, rtol=0.0, atol=tolerance):
            raise ValueError(f"{axis_name} 不是容許誤差內的軸對齊 rectilinear 公尺格網")
    else:
        raise ValueError(f"{axis_name} 必須是一維軸或與速度場同形的二維座標")
    if not np.all(np.isfinite(extracted)):
        raise ValueError(f"{axis_name} 必須全部是有限公尺座標")
    return np.asarray(extracted, dtype=np.float64)


def _validate_monotonic_axis(axis: np.ndarray, name: str, tolerance: float) -> None:
    """要求座標嚴格單調，並拒絕重複或折返位置。"""

    difference = np.diff(axis)
    if np.any(np.abs(difference) <= tolerance):
        raise ValueError(f"{name} 含有重複或小於座標容許值的格距")
    increasing = np.all(difference > 0.0)
    decreasing = np.all(difference < 0.0)
    if not (increasing or decreasing):
        raise ValueError(f"{name} 必須嚴格單調；非單調公尺格網不予計算")


def _compute_strain(
    u: np.ndarray,
    v: np.ndarray,
    usable: np.ndarray,
    grid: _RectilinearGrid,
    config: DetectionConfig,
) -> tuple[np.ndarray, ...]:
    """用完整中心差分 stencil 計算 S、特徵值及 e2。

    每個中心點的四個一階差分鄰點和中心速度都必須在 ``usable`` 內；因此即使
    無效格點只有一列寬，也不會被相鄰的中心差分跨過。輸出未通過 stencil 或
    特徵值間距門檻的位置保留 NaN/False，這些 NaN 是輸出遮罩的明示結果，並非
    對輸入缺值的填補。
    """

    shape = u.shape
    sxx = np.full(shape, np.nan, dtype=np.float64)
    sxy = np.full(shape, np.nan, dtype=np.float64)
    syy = np.full(shape, np.nan, dtype=np.float64)
    s1 = np.full(shape, np.nan, dtype=np.float64)
    s2 = np.full(shape, np.nan, dtype=np.float64)
    e2_x = np.full(shape, np.nan, dtype=np.float64)
    e2_y = np.full(shape, np.nan, dtype=np.float64)
    derivative_valid = np.zeros(shape, dtype=bool)
    if shape[0] < 3 or shape[1] < 3:
        return sxx, sxy, syy, s1, s2, e2_x, e2_y, derivative_valid

    center_usable = usable[1:-1, 1:-1]
    stencil_usable = (
        center_usable
        & usable[:-2, 1:-1]
        & usable[2:, 1:-1]
        & usable[1:-1, :-2]
        & usable[1:-1, 2:]
    )
    if not np.any(stencil_usable):
        return sxx, sxy, syy, s1, s2, e2_x, e2_y, derivative_valid

    x_left = grid.x_axis_m[:-2]
    x_right = grid.x_axis_m[2:]
    y_lower = grid.y_axis_m[:-2]
    y_upper = grid.y_axis_m[2:]
    dx_two = x_right - x_left
    dy_two = y_upper - y_lower
    # 軸座標在此已驗證為嚴格單調，因此分母不會為零；保留符號可支援遞減軸。
    du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / dx_two[None, :]
    dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / dx_two[None, :]
    du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / dy_two[:, None]
    dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / dy_two[:, None]

    sxx_inner = du_dx
    syy_inner = dv_dy
    sxy_inner = 0.5 * (du_dy + dv_dx)
    trace_half = 0.5 * (sxx_inner + syy_inner)
    half_difference = 0.5 * (sxx_inner - syy_inner)
    radius = np.hypot(half_difference, sxy_inner)
    s1_inner = trace_half - radius
    s2_inner = trace_half + radius
    eigen_gap = 2.0 * radius

    sxx[1:-1, 1:-1][stencil_usable] = sxx_inner[stencil_usable]
    sxy[1:-1, 1:-1][stencil_usable] = sxy_inner[stencil_usable]
    syy[1:-1, 1:-1][stencil_usable] = syy_inner[stencil_usable]
    s1[1:-1, 1:-1][stencil_usable] = s1_inner[stencil_usable]
    s2[1:-1, 1:-1][stencil_usable] = s2_inner[stencil_usable]

    finite = np.isfinite(s1_inner) & np.isfinite(s2_inner) & np.isfinite(eigen_gap)
    eigen_valid = stencil_usable & finite & (eigen_gap >= config.min_eigenvalue_gap_sinv)

    # 對稱 2x2 矩陣的 lambda2 向量可寫為 (b, lambda2-a)。當 b 幾乎為零且
    # 這個表示退化時，改用 (lambda2-d, b)，可穩定取得軸向特徵向量。
    first_x = sxy_inner
    first_y = s2_inner - sxx_inner
    first_norm = np.hypot(first_x, first_y)
    second_x = s2_inner - syy_inner
    second_y = sxy_inner
    second_norm = np.hypot(second_x, second_y)
    use_first = first_norm >= second_norm
    raw_x = np.where(use_first, first_x, second_x)
    raw_y = np.where(use_first, first_y, second_y)
    norm = np.hypot(raw_x, raw_y)
    vector_valid = eigen_valid & (norm > 0.0) & np.isfinite(norm)
    normalized_x = np.divide(raw_x, norm, out=np.full_like(raw_x, np.nan), where=norm > 0.0)
    normalized_y = np.divide(raw_y, norm, out=np.full_like(raw_y, np.nan), where=norm > 0.0)
    e2_x[1:-1, 1:-1][vector_valid] = normalized_x[vector_valid]
    e2_y[1:-1, 1:-1][vector_valid] = normalized_y[vector_valid]
    derivative_valid[1:-1, 1:-1] = vector_valid
    return sxx, sxy, syy, s1, s2, e2_x, e2_y, derivative_valid


def _find_core_candidates(
    s1: np.ndarray,
    derivative_valid: np.ndarray,
    grid: _RectilinearGrid,
    config: DetectionConfig,
) -> list[tuple[int, int]]:
    """尋找負的局部 s1 極小值並按公尺距離做確定性去重。"""

    radius = config.local_min_radius_cells
    ny, nx = s1.shape
    raw: list[tuple[int, int]] = []
    # 局部極小值比較範圍要求全 stencil 有效，避免在核心附近以無效值作比較。
    for row in range(radius, ny - radius):
        for col in range(radius, nx - radius):
            if not derivative_valid[row, col] or not np.isfinite(s1[row, col]):
                continue
            local_valid = derivative_valid[row - radius : row + radius + 1, col - radius : col + radius + 1]
            local_values = s1[row - radius : row + radius + 1, col - radius : col + radius + 1]
            if not np.all(local_valid) or not np.all(np.isfinite(local_values)):
                continue
            center = float(s1[row, col])
            if center >= config.min_core_s1_sinv:
                continue
            # 允許數值相等的鄰點留在 plateau，但必須至少有一個嚴格較高鄰點，
            # 否則整片常數區會被每個格點當成核心。
            local_minimum = float(np.min(local_values))
            if center <= local_minimum + config.local_min_tolerance_sinv and np.any(
                local_values > center + config.local_min_tolerance_sinv
            ):
                raw.append((row, col))

    raw.sort(key=lambda index: (float(s1[index]), index[0], index[1]))
    accepted: list[tuple[int, int]] = []
    separation = config.core_min_separation_m
    # 分離距離為零時，局部極小值 stencil 已完成最小去重；非零時才加入
    # 以公尺計的 NMS，並使用排序後的較強核心優先規則確保結果可重現。
    for index in raw:
        if separation > 0.0:
            row, col = index
            x = grid.x_axis_m[col]
            y = grid.y_axis_m[row]
            too_close = any(
                np.hypot(x - grid.x_axis_m[other_col], y - grid.y_axis_m[other_row]) < separation
                for other_row, other_col in accepted
            )
            if too_close:
                continue
        accepted.append(index)
    return accepted


def _build_detection(
    row: int,
    col: int,
    grid: _RectilinearGrid,
    s1: np.ndarray,
    e2_x: np.ndarray,
    e2_y: np.ndarray,
    derivative_valid: np.ndarray,
    config: DetectionConfig,
) -> TrapDetection:
    """從一個核心向 e2 正反方向積分並建立可序列化的檢出模型。"""

    core_xy = (float(grid.x_axis_m[col]), float(grid.y_axis_m[row]))
    core_s1 = float(s1[row, col])
    forward = _integrate_branch(
        core_xy,
        sign=1.0,
        core_s1_sinv=core_s1,
        grid=grid,
        s1=s1,
        e2_x=e2_x,
        e2_y=e2_y,
        derivative_valid=derivative_valid,
        config=config,
    )
    backward = _integrate_branch(
        core_xy,
        sign=-1.0,
        core_s1_sinv=core_s1,
        grid=grid,
        s1=s1,
        e2_x=e2_x,
        e2_y=e2_y,
        derivative_valid=derivative_valid,
        config=config,
    )

    # 反向分支在輸出時反轉，讓曲線方向固定為 backward -> core -> forward；
    # 這樣交會、Hausdorff 距離和圖面標註不會受 e2 的任意正負號影響。
    points = np.vstack((backward.points_m[::-1], forward.points_m[1:]))
    strengths = np.concatenate((backward.strengths_sinv[::-1], forward.strengths_sinv[1:]))
    length = float(np.sum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))))
    mean_s1 = float(np.mean(strengths))
    mean_abs_s1 = float(np.mean(np.abs(strengths)))
    is_qualified = points.shape[0] >= config.min_curve_points and length >= config.min_curve_length_m
    qc_flags: list[str] = []
    if points.shape[0] < config.min_curve_points:
        qc_flags.append("curve_too_few_points")
    if length < config.min_curve_length_m:
        qc_flags.append("curve_too_short")
    if forward.stop_reason == "invalid_mask":
        qc_flags.append("forward_reached_invalid_mask")
    if backward.stop_reason == "invalid_mask":
        qc_flags.append("backward_reached_invalid_mask")

    return TrapDetection(
        core_index_ij=(row, col),
        core_xy_m=core_xy,
        core_s1_sinv=core_s1,
        curve_points_m=points,
        curve_s1_sinv=strengths,
        curve_length_m=length,
        curve_mean_s1_sinv=mean_s1,
        curve_mean_abs_s1_sinv=mean_abs_s1,
        forward_stop_reason=forward.stop_reason,
        backward_stop_reason=backward.stop_reason,
        is_qualified=is_qualified,
        qc_flags=tuple(qc_flags),
    )


def _integrate_branch(
    start_xy: tuple[float, float],
    *,
    sign: float,
    core_s1_sinv: float,
    grid: _RectilinearGrid,
    s1: np.ndarray,
    e2_x: np.ndarray,
    e2_y: np.ndarray,
    derivative_valid: np.ndarray,
    config: DetectionConfig,
) -> _BranchResult:
    """沿連續化 e2 向單一方向積分，並保守處理 mask、強度與轉角。"""

    points: list[tuple[float, float]] = [start_xy]
    strengths: list[float] = [core_s1_sinv]
    initial = _sample_field(start_xy[0], start_xy[1], grid, s1, e2_x, e2_y, derivative_valid)
    if initial is None:
        return _BranchResult(np.asarray(points), np.asarray(strengths), "invalid_mask")
    previous_tangent = np.asarray((sign * initial.e2_x, sign * initial.e2_y), dtype=np.float64)
    previous_strength = core_s1_sinv
    total_length = 0.0
    safe_step = 0.5 * min(grid.dx_m, grid.dy_m)
    requested_step = config.integration_step_m if config.integration_step_m is not None else safe_step
    # 即使使用者指定較大步長，也切成不超過半格的 substeps，避免跳過一列無效格點。
    substep = min(float(requested_step), safe_step)
    if not np.isfinite(substep) or substep <= 0.0:
        return _BranchResult(np.asarray(points), np.asarray(strengths), "invalid_step")

    for _ in range(config.max_integration_steps):
        if total_length >= config.max_curve_length_m - 1.0e-12:
            return _BranchResult(np.asarray(points), np.asarray(strengths), "max_curve_length")
        current = np.asarray(points[-1], dtype=np.float64)
        current_sample = _sample_field(
            current[0], current[1], grid, s1, e2_x, e2_y, derivative_valid
        )
        if current_sample is None:
            return _BranchResult(np.asarray(points), np.asarray(strengths), "invalid_mask")
        tangent = _oriented_tangent(
            sign * np.asarray((current_sample.e2_x, current_sample.e2_y)), previous_tangent
        )
        angle = _angle_deg(previous_tangent, tangent)
        if angle > config.max_turn_angle_deg:
            return _BranchResult(np.asarray(points), np.asarray(strengths), "curvature_limit")

        remaining = config.max_curve_length_m - total_length
        step = min(substep, remaining)
        midpoint = current + 0.5 * step * tangent
        midpoint_sample = _sample_field(
            midpoint[0], midpoint[1], grid, s1, e2_x, e2_y, derivative_valid
        )
        if midpoint_sample is None:
            reason = _classify_outside_or_mask(midpoint, grid, derivative_valid)
            return _BranchResult(np.asarray(points), np.asarray(strengths), reason)
        midpoint_tangent = _oriented_tangent(
            sign * np.asarray((midpoint_sample.e2_x, midpoint_sample.e2_y)), tangent
        )
        if _angle_deg(tangent, midpoint_tangent) > config.max_turn_angle_deg:
            return _BranchResult(np.asarray(points), np.asarray(strengths), "curvature_limit")

        target = current + step * midpoint_tangent
        target_sample = _sample_field(target[0], target[1], grid, s1, e2_x, e2_y, derivative_valid)
        if target_sample is None:
            reason = _classify_outside_or_mask(target, grid, derivative_valid)
            return _BranchResult(np.asarray(points), np.asarray(strengths), reason)
        target_strength = target_sample.s1_sinv
        if target_strength >= config.curve_s1_stop_sinv:
            return _BranchResult(np.asarray(points), np.asarray(strengths), "s1_threshold")
        if abs(target_strength) < abs(core_s1_sinv) * config.curve_min_relative_strength:
            return _BranchResult(np.asarray(points), np.asarray(strengths), "relative_strength_threshold")
        if target_strength < previous_strength - config.curve_monotonic_tolerance_sinv:
            return _BranchResult(np.asarray(points), np.asarray(strengths), "s1_not_monotonic")

        target_tuple = (float(target[0]), float(target[1]))
        segment_length = float(np.hypot(target[0] - current[0], target[1] - current[1]))
        if segment_length <= 0.0:
            return _BranchResult(np.asarray(points), np.asarray(strengths), "zero_step")
        points.append(target_tuple)
        strengths.append(float(target_strength))
        total_length += segment_length
        previous_tangent = midpoint_tangent
        previous_strength = float(target_strength)
        if total_length >= config.max_curve_length_m - 1.0e-12:
            return _BranchResult(np.asarray(points), np.asarray(strengths), "max_curve_length")

    return _BranchResult(np.asarray(points), np.asarray(strengths), "max_integration_steps")


def _sample_field(
    x: float,
    y: float,
    grid: _RectilinearGrid,
    s1: np.ndarray,
    e2_x: np.ndarray,
    e2_y: np.ndarray,
    derivative_valid: np.ndarray,
) -> _Sample | None:
    """在四個有效角點上雙線性採樣 s1/e2，並連續化向量正負號。"""

    x_bracket = _axis_bracket(grid.x_axis_m, x)
    y_bracket = _axis_bracket(grid.y_axis_m, y)
    if x_bracket is None or y_bracket is None:
        return None
    col, tx = x_bracket
    row, ty = y_bracket
    corners = ((row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1))
    if not all(derivative_valid[corner] for corner in corners):
        return None
    weights = ((1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty)
    strength_values = np.asarray([s1[corner] for corner in corners], dtype=np.float64)
    vectors = np.asarray([(e2_x[corner], e2_y[corner]) for corner in corners], dtype=np.float64)
    if not np.all(np.isfinite(strength_values)) or not np.all(np.isfinite(vectors)):
        return None
    reference = vectors[0]
    for index in range(1, vectors.shape[0]):
        if float(np.dot(reference, vectors[index])) < 0.0:
            vectors[index] *= -1.0
    weighted_vector = np.sum(vectors * np.asarray(weights)[:, None], axis=0)
    norm = float(np.hypot(weighted_vector[0], weighted_vector[1]))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        return None
    weighted_vector /= norm
    return _Sample(
        s1_sinv=float(np.dot(np.asarray(weights), strength_values)),
        e2_x=float(weighted_vector[0]),
        e2_y=float(weighted_vector[1]),
    )


def _axis_bracket(axis: np.ndarray, value: float) -> tuple[int, float] | None:
    """回傳座標所在的相鄰 cell 與局部比例，支援遞增或遞減軸。"""

    if not np.isfinite(value) or value < min(axis[0], axis[-1]) or value > max(axis[0], axis[-1]):
        return None
    increasing = axis[-1] > axis[0]
    ordered = axis if increasing else -axis
    target = value if increasing else -value
    index = int(np.searchsorted(ordered, target, side="right") - 1)
    index = min(max(index, 0), axis.size - 2)
    denominator = axis[index + 1] - axis[index]
    fraction = float((value - axis[index]) / denominator)
    if fraction < -1.0e-12 or fraction > 1.0 + 1.0e-12:
        return None
    return index, min(max(fraction, 0.0), 1.0)


def _oriented_tangent(vector: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """以相鄰切線內積連續化 e2 的正負號。"""

    result = np.asarray(vector, dtype=np.float64)
    if float(np.dot(result, previous)) < 0.0:
        result = -result
    norm = float(np.hypot(result[0], result[1]))
    return result / norm


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """回傳兩個單位向量的夾角，避免浮點誤差造成 arccos 越界。"""

    first_norm = float(np.hypot(first[0], first[1]))
    second_norm = float(np.hypot(second[0], second[1]))
    if first_norm <= 0.0 or second_norm <= 0.0:
        return 180.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _classify_outside_or_mask(
    point: np.ndarray,
    grid: _RectilinearGrid,
    derivative_valid: np.ndarray,
) -> str:
    """將採樣失敗區分成流場邊界和有效遮罩障礙，供 QC 使用。"""

    if (
        point[0] < min(grid.x_axis_m[0], grid.x_axis_m[-1])
        or point[0] > max(grid.x_axis_m[0], grid.x_axis_m[-1])
        or point[1] < min(grid.y_axis_m[0], grid.y_axis_m[-1])
        or point[1] > max(grid.y_axis_m[0], grid.y_axis_m[-1])
    ):
        return "outside_flow_domain"
    return "invalid_mask"
