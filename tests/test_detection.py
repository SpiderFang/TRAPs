"""TRAPs 數值核心的解析流場與遮罩邊界測試。

這些測試只驗證可由數學定義直接判定的核心性質：平移與剛體旋轉不應製造
吸引核心，解析對稱應變應得到正確特徵值，局地合成吸引場應能建立曲線，
而無效格點不得被中心差分或雙線性積分跨越。
"""

from __future__ import annotations

import numpy as np
import pytest

from traps import DetectionConfig, detect_snapshot


def _grid(size: int = 21, spacing_m: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """建立以公尺為單位、可直接對照解析導數的規則測試格網。"""

    axis = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) * spacing_m
    return np.meshgrid(axis, axis, indexing="xy")


def _config(**overrides: object) -> DetectionConfig:
    """提供測試使用的明示門檻，避免依賴生產設定檔的隱含值。"""

    values: dict[str, object] = {
        "min_core_s1_sinv": -1.0e-8,
        "min_eigenvalue_gap_sinv": 1.0e-12,
        "local_min_radius_cells": 1,
        "integration_step_m": 30.0,
        "max_curve_length_m": 1_000.0,
        "curve_min_relative_strength": 0.05,
        "curve_monotonic_tolerance_sinv": 2.0e-7,
        "min_curve_points": 3,
        "min_curve_length_m": 0.0,
    }
    values.update(overrides)
    return DetectionConfig(**values)


def test_translation_has_zero_strain_and_no_core() -> None:
    """均勻平移的速度梯度為零，不應產生負 s1 核心。"""

    x, y = _grid()
    u = np.full_like(x, 0.7)
    v = np.full_like(y, -0.2)
    result = detect_snapshot(u, v, np.ones_like(x, dtype=bool), x, y, _config())

    assert np.allclose(result.sxx_sinv[1:-1, 1:-1], 0.0)
    assert np.allclose(result.sxy_sinv[1:-1, 1:-1], 0.0)
    assert np.allclose(result.syy_sinv[1:-1, 1:-1], 0.0)
    assert not np.any(result.derivative_valid)
    assert result.detections == ()
    assert "no_negative_local_minimum_core" in result.qc_flags


def test_rigid_rotation_has_zero_symmetric_strain() -> None:
    """純剛體旋轉只有反對稱速度梯度，不應被誤判為吸引 TRAP。"""

    x, y = _grid()
    omega = 2.0e-3
    u = -omega * y
    v = omega * x
    result = detect_snapshot(u, v, np.ones_like(x, dtype=bool), x, y, _config())

    assert np.allclose(result.sxx_sinv[1:-1, 1:-1], 0.0, atol=1.0e-14)
    assert np.allclose(result.sxy_sinv[1:-1, 1:-1], 0.0, atol=1.0e-14)
    assert np.allclose(result.syy_sinv[1:-1, 1:-1], 0.0, atol=1.0e-14)
    assert result.detections == ()


def test_analytic_strain_returns_expected_eigenvalues_and_e2() -> None:
    """線性速度場的 S 特徵值與最大特徵方向應符合解析解。"""

    x, y = _grid()
    a, b, c = -2.0e-3, 3.0e-3, 5.0e-4
    # u=a*x+c*y、v=c*x+b*y 直接給出對稱應變 S=[[a,c],[c,b]]。
    u = a * x + c * y
    v = c * x + b * y
    result = detect_snapshot(u, v, np.ones_like(x, dtype=bool), x, y, _config())

    expected_s1 = 0.5 * (a + b) - np.hypot(0.5 * (a - b), c)
    expected_s2 = 0.5 * (a + b) + np.hypot(0.5 * (a - b), c)
    interior = (slice(2, -2), slice(2, -2))
    assert np.all(result.derivative_valid[interior])
    assert np.allclose(result.s1_sinv[interior], expected_s1, atol=1.0e-12)
    assert np.allclose(result.s2_sinv[interior], expected_s2, atol=1.0e-12)

    expected_e2 = np.array([c, expected_s2 - a], dtype=np.float64)
    expected_e2 /= np.linalg.norm(expected_e2)
    actual_e2 = result.e2[10, 10]
    assert abs(float(np.dot(actual_e2, expected_e2))) > 1.0 - 1.0e-12
    assert result.detections == ()


def test_localized_attraction_produces_core_and_curve() -> None:
    """局地高斯吸引場應檢出中心核心並沿 e2 形成可合格曲線。"""

    x, y = _grid(size=41, spacing_m=100.0)
    length_scale = 900.0
    envelope = np.exp(-(x * x + y * y) / length_scale**2)
    # a>b 使中心 e2 主要沿 y 軸，且中心 s1=-a 為負的局地極小值。
    u = -2.0e-3 * x * envelope
    v = -1.0e-3 * y * envelope
    result = detect_snapshot(
        u,
        v,
        np.ones_like(x, dtype=bool),
        x,
        y,
        _config(max_curve_length_m=1_200.0),
    )

    assert result.detections, "合成局地吸引場應至少有一個核心"
    center = min(result.detections, key=lambda detection: abs(detection.core_xy_m[0]) + abs(detection.core_xy_m[1]))
    assert center.core_index_ij == (20, 20)
    assert center.core_s1_sinv < 0.0
    assert center.curve_points_m.shape[1] == 2
    assert center.curve_length_m > 0.0
    assert center.is_qualified
    assert center.forward_stop_reason in {
        "relative_strength_threshold",
        "s1_threshold",
        "max_curve_length",
        "outside_flow_domain",
        "invalid_mask",
    }


def test_invalid_mask_blocks_difference_and_curve() -> None:
    """一列無效格點會使相鄰導數失效，曲線也不得跨越該列。"""

    x, y = _grid(size=31, spacing_m=100.0)
    u = -1.5e-3 * x
    v = -5.0e-4 * y
    valid = np.ones_like(x, dtype=bool)
    obstacle_col = x.shape[1] // 2
    valid[:, obstacle_col] = False
    result = detect_snapshot(u, v, valid, x, y, _config())

    # 中心差分不能從障礙列兩側取值；再遠一格的內點仍可保有完整 stencil。
    assert not np.any(result.derivative_valid[1:-1, obstacle_col - 1])
    assert not np.any(result.derivative_valid[1:-1, obstacle_col + 1])
    assert np.any(result.derivative_valid[2:-2, 2])
    for detection in result.detections:
        assert not np.any(np.isclose(detection.curve_points_m[:, 0], x[0, obstacle_col]))


def test_non_rectilinear_or_nonmonotonic_coordinates_are_rejected() -> None:
    """曲線格網及折返座標不能套用軸向公尺差分。"""

    x, y = _grid()
    valid = np.ones_like(x, dtype=bool)
    with pytest.raises(ValueError, match="rectilinear"):
        distorted_x = x.copy()
        distorted_x[3, 4] += 10.0
        detect_snapshot(x, y, valid, distorted_x, y, _config())

    with pytest.raises(ValueError, match="嚴格單調"):
        nonmonotonic_x = x[0].copy()
        nonmonotonic_x[5] = nonmonotonic_x[4] - 1.0
        detect_snapshot(x, y, valid, nonmonotonic_x, y, _config())


def test_descending_metric_axes_are_supported() -> None:
    """嚴格遞減但仍為 rectilinear 的公尺軸，導數符號和特徵值應保持正確。"""

    x, y = _grid()
    descending_x = x[:, ::-1]
    descending_y = y[::-1, :]
    u = -2.0e-3 * descending_x + 5.0e-4 * descending_y
    v = 5.0e-4 * descending_x + 3.0e-3 * descending_y
    result = detect_snapshot(
        u,
        v,
        np.ones_like(u, dtype=bool),
        descending_x[0],
        descending_y[:, 0],
        _config(),
    )
    expected_s1 = 0.5 * (-2.0e-3 + 3.0e-3) - np.hypot(2.5e-3, 5.0e-4)
    assert np.allclose(result.s1_sinv[2:-2, 2:-2], expected_s1, atol=1.0e-12)


@pytest.mark.parametrize("radius_cells", [0, 1, 2, 3])
def test_mask_safe_smoothing_records_scale_for_all_required_cases(radius_cells: int) -> None:
    """0/1/2/3 格敏感度案例都應可執行，且結果記錄 graph 與等效 sigma。"""

    x, y = _grid(size=21, spacing_m=100.0)
    u = -1.0e-3 * x * np.exp(-(x * x + y * y) / 1_000.0**2)
    v = -5.0e-4 * y * np.exp(-(x * x + y * y) / 1_000.0**2)
    result = detect_snapshot(
        u,
        v,
        np.ones_like(x, dtype=bool),
        x,
        y,
        _config(smoothing_radius_cells=radius_cells),
    )

    assert result.smoothing_radius_cells == radius_cells
    if radius_cells == 0:
        assert result.smoothing_method == "none"
        assert result.smoothing_equivalent_sigma_m == (0.0, 0.0)
    else:
        expected_sigma = 100.0 * np.sqrt(0.5 * radius_cells / 2.0)
        assert result.smoothing_method == "four_neighbor_graph_diffusion"
        assert np.allclose(result.smoothing_equivalent_sigma_m, (expected_sigma, expected_sigma))
        assert "smoothing_applied_graph_diffusion" in result.qc_flags


def test_graph_diffusion_does_not_mix_across_narrow_invalid_barrier() -> None:
    """三格 graph diffusion 不應把 barrier 左側速度傳到右側有效連通元件。"""

    x, y = _grid(size=31, spacing_m=100.0)
    barrier_col = x.shape[1] // 2
    u = np.zeros_like(x)
    u[:, :barrier_col] = 1.0
    u[:, barrier_col] = np.nan
    # 非零 v 梯度讓右側零 u 的 sxx=0 格點仍具有非退化 e2，便於檢查平滑結果。
    v = 1.0e-3 * y
    valid_with_barrier = np.ones_like(x, dtype=bool)
    valid_with_barrier[:, barrier_col] = False
    config = _config(smoothing_radius_cells=3)

    masked = detect_snapshot(u, v, valid_with_barrier, x, y, config)
    open_domain = detect_snapshot(
        np.nan_to_num(u, nan=0.0),
        v,
        np.ones_like(x, dtype=bool),
        x,
        y,
        config,
    )
    right_probe = barrier_col + 2

    # barrier 右側第二列已可計算完整中心差分，但三次 graph diffusion 仍只能看右側值。
    assert masked.derivative_valid[15, right_probe]
    assert masked.sxx_sinv[15, right_probe] == pytest.approx(0.0, abs=1.0e-14)
    # 對照組沒有 barrier，普通完整網格上的同一 graph diffusion 會產生跨界梯度。
    assert abs(float(open_domain.sxx_sinv[15, right_probe])) > 1.0e-8
    assert not masked.input_valid_mask[15, barrier_col]


def test_smoothing_scale_above_three_is_rejected() -> None:
    """研究計畫只核定 0/1/2/3 格，其他尺度不能悄悄進入正式分析。"""

    with pytest.raises(ValueError, match="0、1、2 或 3"):
        DetectionConfig(smoothing_radius_cells=4)


# ---------------------------------------------------------------------------
# WP-I：次網格核心定位（3x3 二次曲面）
# ---------------------------------------------------------------------------


def _rect_grid(size: int = 9, spacing_m: float = 1000.0):
    from traps.detection import _RectilinearGrid

    axis = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) * spacing_m
    return _RectilinearGrid(axis, axis.copy(), spacing_m, spacing_m), axis


def test_quadratic_refinement_recovers_off_node_minimum_below_one_metre() -> None:
    """解析二次碗（含交叉項）的真實極小值在格點外 0.3/-0.2 格；細化位置誤差 < 1 m。"""

    from traps.detection import refine_core_quadratic

    grid, axis = _rect_grid()
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    x0, y0 = 300.0, -200.0  # 相對格點 (4, 4) 的偏移，單位 m（格距 1000 m）
    dx, dy = xx - x0, yy - y0
    s1 = -2.0e-5 + 3.0e-11 * dx**2 + 2.0e-11 * dy**2 + 1.0e-11 * dx * dy
    valid = np.ones_like(s1, dtype=bool)
    refined = refine_core_quadratic(s1, valid, grid, 4, 4)
    assert refined is not None
    assert np.hypot(refined[0] - x0, refined[1] - y0) < 1.0


def test_quadratic_refinement_falls_back_for_saddle_flat_and_far_offset() -> None:
    """鞍點、Hessian 非正定、偏移超過 0.5 格、stencil 含無效格點時保留格點節點（回傳 None）。"""

    from traps.detection import refine_core_quadratic

    grid, axis = _rect_grid()
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    valid = np.ones(xx.shape, dtype=bool)
    saddle = 1.0e-11 * (xx**2 - yy**2)
    assert refine_core_quadratic(saddle, valid, grid, 4, 4) is None
    linear = 1.0e-9 * xx + 2.0e-9 * yy
    assert refine_core_quadratic(linear, valid, grid, 4, 4) is None  # Hessian 為零
    far = 1.0e-11 * ((xx - 900.0) ** 2 + yy**2)  # 極小值在 0.9 格外
    assert refine_core_quadratic(far, valid, grid, 4, 4) is None
    bowl = 1.0e-11 * (xx**2 + yy**2)
    invalid = valid.copy()
    invalid[3, 4] = False
    assert refine_core_quadratic(bowl, invalid, grid, 4, 4) is None
    assert refine_core_quadratic(bowl, valid, grid, 0, 4) is None  # 邊界沒有完整 3x3


def test_detection_records_refined_core_off_node_and_toggle() -> None:
    """局地吸引場中心在格點外：細化核心比格點節點更接近真實中心，並可用旗標關閉。"""

    x, y = _grid(size=41, spacing_m=100.0)
    x0, y0 = 30.0, -20.0
    envelope = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 900.0**2)
    u = -2.0e-3 * (x - x0) * envelope
    v = -1.0e-3 * (y - y0) * envelope
    valid = np.ones_like(x, dtype=bool)
    on = detect_snapshot(u, v, valid, x, y, _config(max_curve_length_m=1_200.0))
    off = detect_snapshot(u, v, valid, x, y, _config(max_curve_length_m=1_200.0, core_subgrid_refine=False))
    core = min(on.detections, key=lambda d: abs(d.core_grid_xy_m[0]) + abs(d.core_grid_xy_m[1]))
    node = min(off.detections, key=lambda d: abs(d.core_xy_m[0]) + abs(d.core_xy_m[1]))
    assert core.core_index_ij == node.core_index_ij == (20, 20)
    assert core.core_refined and core.core_grid_xy_m == (0.0, 0.0)
    assert not node.core_refined and node.core_xy_m == node.core_grid_xy_m == (0.0, 0.0)
    assert abs(core.core_xy_m[0]) <= 50.0 and abs(core.core_xy_m[1]) <= 50.0
    assert np.hypot(core.core_xy_m[0] - x0, core.core_xy_m[1] - y0) < np.hypot(x0, y0) / 3.0
    assert any(np.allclose(point, core.core_xy_m) for point in core.curve_points_m)  # 曲線從細化位置起算
    assert on.config.to_manifest_dict()["core_subgrid"] == "quadratic_3x3"
    assert on.config.to_manifest_dict()["core_subgrid_refine"] is True
    assert off.config.to_manifest_dict()["core_subgrid_refine"] is False
