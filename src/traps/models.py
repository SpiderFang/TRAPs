"""TRAPs 數值核心使用的資料模型與可重現門檻。

本模組只放置資料契約，不在這裡執行流場運算。將設定與輸出模型集中在單一檔案，
可以讓逐時輸入、骨架追蹤、站點統計和成果圖共用同一套欄位名稱，並避免把某個
分析腳本內的預設值誤當成研究結果。所有速度梯度及應變率的單位均為 ``s⁻¹``；
曲線座標和弧長使用輸入的公尺座標 ``m``。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _freeze_array(value: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """建立唯讀陣列，避免結果模型被下游統計意外改寫。

    ``s1``、特徵向量和曲線是同一次 snapshot 的證據。下游若需要做不同的
    視覺化轉換，應先複製陣列，而不是就地改動結果模型；這樣可維持 manifest
    及重現流程所需的輸出一致性。
    """

    array = np.asarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """單一 snapshot 的 TRAP 檢出、局部極小值及曲線積分門檻。

    所有會改變候選數量或曲線長度的值都在此處明列，讓每次執行可以直接以
    ``dataclasses.asdict(config)`` 寫入 manifest。``0.30`` 的相對強度預設值
    只是工程起始值；它不能直接沿用其他海域的尺度，正式分析必須做敏感度檢查。

    ``smoothing_radius_cells`` 可設為 0、1、2 或 3。非零尺度使用有效格點四鄰接
    圖上的有限次擴散，每次只把同一個有效連通元件的東西向、北向鄰點納入；因此
    不會像未帶 mask 的 Gaussian 卷積一樣跨過一列窄陸地。這個圖核只在規則網格
    內近似 Gaussian，等效 ``sigma`` 與邊界的鄰點數會改變，結果會保存方法和
    估計尺度，正式分析仍需將它與 0 格基線做敏感度比較。
    """

    # 候選核心需小於此負值；值的單位為 s⁻¹，而非無因次百分比。
    min_core_s1_sinv: float = -1.0e-8
    # 退化的 S 特徵值會使 e2 方向不唯一，低於此差值的格點不參與檢出與積分。
    min_eigenvalue_gap_sinv: float = 1.0e-12
    # 局部極小值比較半徑，以格點數表示；周圍 stencil 必須全部可用。
    local_min_radius_cells: int = 1
    # 局部極小值相對鄰域的最小顯著差，避免線性場的浮點殘差製造假核心。
    local_min_tolerance_sinv: float = 1.0e-12
    # 額外的核心空間去重距離；0 表示只使用局部極小值 stencil，不另行合併。
    core_min_separation_m: float = 0.0
    # 骨架積分固定步長；None 由資料格距決定，實際值會限制在半格距內以避免跳過 mask。
    integration_step_m: float | None = None
    # 從核心向外允許的最大總弧長，單位為 m。
    max_curve_length_m: float = 20_000.0
    # 防止壞資料讓積分無限循環的步數上限。
    max_integration_steps: int = 2_000
    # 採樣點的 s1 到達此值時停止；預設 0 代表只保留負 s1 的吸引區。
    curve_s1_stop_sinv: float = 0.0
    # 曲線上 |s1| 相對於核心 |s1| 的最低比例；需明列以便跨區域敏感度分析。
    curve_min_relative_strength: float = 0.30
    # s1 向外不嚴格單調回升時允許的絕對誤差，單位為 s⁻¹。
    curve_monotonic_tolerance_sinv: float = 1.0e-12
    # 相鄰骨架切線的最大轉角；超過時視為數值不可信並停止該分支。
    max_turn_angle_deg: float = 90.0
    # 合格曲線至少需有的採樣點數和弧長；核心點本身計入採樣點數。
    min_curve_points: int = 3
    min_curve_length_m: float = 0.0
    # 2D 座標需可化為單調、軸對齊的 rectilinear 公尺格網時使用的絕對誤差。
    grid_coordinate_tolerance_m: float = 1.0e-6
    # 以四鄰接 graph diffusion 的迭代次數表示平滑尺度；需求固定比較 0/1/2/3 格。
    smoothing_radius_cells: int = 0
    # 每次擴散把中心值與有效四鄰點平均值混合的比例；0.5 對應穩定的 explicit step。
    smoothing_diffusion_alpha: float = 0.5
    # 供 manifest 明確指出這不是未帶遮罩的 Gaussian normalized convolution。
    smoothing_method: str = "four_neighbor_graph_diffusion"
    # 是否以 3x3 有效 stencil 的二維二次曲面擬合，把核心從格點提升到次網格位置。
    # 關閉時核心位置退回格點節點（與 v1 行為相同）。
    core_subgrid_refine: bool = True
    # 供 manifest 明確記錄次網格核心定位方法；固定值，不可由使用者改寫。
    core_subgrid: str = "quadratic_3x3"

    def __post_init__(self) -> None:
        """驗證門檻，及早拒絕會產生無法解讀結果的設定。"""

        if not np.isfinite(self.min_core_s1_sinv) or self.min_core_s1_sinv >= 0.0:
            raise ValueError("min_core_s1_sinv 必須是有限負值（單位 s^-1）")
        if not np.isfinite(self.min_eigenvalue_gap_sinv) or self.min_eigenvalue_gap_sinv < 0.0:
            raise ValueError("min_eigenvalue_gap_sinv 必須是非負有限值")
        if self.local_min_radius_cells < 1:
            raise ValueError("local_min_radius_cells 必須至少為 1")
        if not np.isfinite(self.local_min_tolerance_sinv) or self.local_min_tolerance_sinv < 0.0:
            raise ValueError("local_min_tolerance_sinv 必須是非負有限值")
        if not np.isfinite(self.core_min_separation_m) or self.core_min_separation_m < 0.0:
            raise ValueError("core_min_separation_m 必須是非負有限值")
        if self.integration_step_m is not None and (
            not np.isfinite(self.integration_step_m) or self.integration_step_m <= 0.0
        ):
            raise ValueError("integration_step_m 必須是正有限值或 None")
        if not np.isfinite(self.max_curve_length_m) or self.max_curve_length_m <= 0.0:
            raise ValueError("max_curve_length_m 必須是正有限值")
        if self.max_integration_steps < 1:
            raise ValueError("max_integration_steps 必須至少為 1")
        if not np.isfinite(self.curve_s1_stop_sinv):
            raise ValueError("curve_s1_stop_sinv 必須是有限值")
        if not np.isfinite(self.curve_min_relative_strength) or not (
            0.0 <= self.curve_min_relative_strength <= 1.0
        ):
            raise ValueError("curve_min_relative_strength 必須落在 0 到 1 之間")
        if (
            not np.isfinite(self.curve_monotonic_tolerance_sinv)
            or self.curve_monotonic_tolerance_sinv < 0.0
        ):
            raise ValueError("curve_monotonic_tolerance_sinv 必須是非負有限值")
        if not np.isfinite(self.max_turn_angle_deg) or not (0.0 < self.max_turn_angle_deg <= 180.0):
            raise ValueError("max_turn_angle_deg 必須落在 (0, 180] 度")
        if self.min_curve_points < 1:
            raise ValueError("min_curve_points 必須至少為 1")
        if not np.isfinite(self.min_curve_length_m) or self.min_curve_length_m < 0.0:
            raise ValueError("min_curve_length_m 必須是非負有限值")
        if not np.isfinite(self.grid_coordinate_tolerance_m) or self.grid_coordinate_tolerance_m <= 0.0:
            raise ValueError("grid_coordinate_tolerance_m 必須是正有限值")
        if isinstance(self.smoothing_radius_cells, bool) or self.smoothing_radius_cells not in (0, 1, 2, 3):
            raise ValueError("smoothing_radius_cells 必須是 0、1、2 或 3 格")
        if not np.isfinite(self.smoothing_diffusion_alpha) or not (
            0.0 < self.smoothing_diffusion_alpha <= 1.0
        ):
            raise ValueError("smoothing_diffusion_alpha 必須落在 (0, 1] 之間")
        if self.smoothing_method != "four_neighbor_graph_diffusion":
            raise ValueError("smoothing_method 固定為 four_neighbor_graph_diffusion")
        if not isinstance(self.core_subgrid_refine, (bool, np.bool_)):
            raise ValueError("core_subgrid_refine 必須是布林值")
        if self.core_subgrid != "quadratic_3x3":
            raise ValueError("core_subgrid 固定為 quadratic_3x3")

    @property
    def min_core_separation_m(self) -> float:
        """提供較短的相容欄位名稱，供站點統計或舊版腳本讀取。"""

        return self.core_min_separation_m

    def to_manifest_dict(self) -> dict[str, Any]:
        """回傳可直接寫入 JSON manifest 的完整門檻字典。"""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrapDetection:
    """單一 snapshot 的候選 TRAP 核心與沿 e2 的雙向曲線。

    ``curve_points_m`` 由反向分支、核心、正向分支依序組成，核心只出現一次；
    每一列為 ``(x_m, y_m)``。``forward_stop_reason`` 與
    ``backward_stop_reason`` 保留兩端真正停止的原因，因為到邊界、遇到 mask、
    強度衰減和長度上限代表不同的科學品質狀態。``is_qualified`` 只表示通過
    設定的最小點數及長度門檻，不代表已完成觀測驗證。
    """

    core_index_ij: tuple[int, int]
    core_xy_m: tuple[float, float]
    core_s1_sinv: float
    curve_points_m: np.ndarray
    curve_s1_sinv: np.ndarray
    curve_length_m: float
    curve_mean_s1_sinv: float
    curve_mean_abs_s1_sinv: float
    forward_stop_reason: str
    backward_stop_reason: str
    is_qualified: bool
    qc_flags: tuple[str, ...] = field(default_factory=tuple)
    # 核心所在的格點節點座標；未提供時等於 ``core_xy_m``（舊版建構相容）。
    core_grid_xy_m: tuple[float, float] | None = None
    # ``core_xy_m`` 是否為次網格二次擬合後的位置（False 代表仍是格點節點）。
    core_refined: bool = False

    def __post_init__(self) -> None:
        """固定曲線陣列形狀與唯讀狀態，避免下游破壞檢出證據。"""

        index = tuple(int(value) for value in self.core_index_ij)
        if len(index) != 2:
            raise ValueError("core_index_ij 必須是 (row, column)")
        object.__setattr__(self, "core_index_ij", index)
        xy = tuple(float(value) for value in self.core_xy_m)
        if len(xy) != 2 or not all(np.isfinite(value) for value in xy):
            raise ValueError("core_xy_m 必須是兩個有限公尺座標")
        object.__setattr__(self, "core_xy_m", xy)
        if self.core_grid_xy_m is None:
            grid_xy = xy
        else:
            grid_xy = tuple(float(value) for value in self.core_grid_xy_m)
            if len(grid_xy) != 2 or not all(np.isfinite(value) for value in grid_xy):
                raise ValueError("core_grid_xy_m 必須是兩個有限公尺座標")
        object.__setattr__(self, "core_grid_xy_m", grid_xy)
        object.__setattr__(self, "core_refined", bool(self.core_refined))

        points = _freeze_array(self.curve_points_m, dtype=np.float64)
        strengths = _freeze_array(self.curve_s1_sinv, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("curve_points_m 必須是 (n, 2) 公尺座標陣列")
        if strengths.ndim != 1 or strengths.shape[0] != points.shape[0]:
            raise ValueError("curve_s1_sinv 必須與曲線點數一致")
        if points.shape[0] == 0:
            raise ValueError("曲線至少需要核心點")
        object.__setattr__(self, "curve_points_m", points)
        object.__setattr__(self, "curve_s1_sinv", strengths)
        object.__setattr__(self, "curve_length_m", float(self.curve_length_m))
        object.__setattr__(self, "curve_mean_s1_sinv", float(self.curve_mean_s1_sinv))
        object.__setattr__(self, "curve_mean_abs_s1_sinv", float(self.curve_mean_abs_s1_sinv))
        object.__setattr__(self, "qc_flags", tuple(str(flag) for flag in self.qc_flags))

    @property
    def core_ij(self) -> tuple[int, int]:
        """`core_index_ij` 的短別名，便於建立表格輸出。"""

        return self.core_index_ij

    @property
    def core_index(self) -> tuple[int, int]:
        """`core_index_ij` 的一般化別名，供表格與追蹤程式使用。"""

        return self.core_index_ij

    @property
    def curve_xy_m(self) -> np.ndarray:
        """`curve_points_m` 的語意化別名。"""

        return self.curve_points_m

    @property
    def line_length_m(self) -> float:
        """曲線弧長的相容別名。"""

        return self.curve_length_m

    @property
    def mean_s1_sinv(self) -> float:
        """沿線平均 s1 的相容別名。"""

        return self.curve_mean_s1_sinv

    @property
    def mean_abs_s1_sinv(self) -> float:
        """沿線平均 |s1| 的相容別名。"""

        return self.curve_mean_abs_s1_sinv

    @property
    def quality_flags(self) -> tuple[str, ...]:
        """`qc_flags` 的語意化別名。"""

        return self.qc_flags

    @property
    def qualified(self) -> bool:
        """`is_qualified` 的語意化別名。"""

        return self.is_qualified

    @property
    def stop_reasons(self) -> tuple[str, str]:
        """以 `(forward, backward)` 順序回傳兩端停止原因。"""

        return self.forward_stop_reason, self.backward_stop_reason


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """單一時間快照的應變場、核心候選和 TRAP 曲線結果。

    所有場陣列保持輸入的 ``(ny, nx)`` 排列；第一軸是 y/row，第二軸是 x/column。
    ``derivative_valid`` 已同時要求速度有限、輸入 ``valid`` 為真、中心差分 stencil
    完整且特徵值方向不退化。這個遮罩是後續統計能使用的最小可信範圍。
    """

    s1_sinv: np.ndarray
    s2_sinv: np.ndarray
    sxx_sinv: np.ndarray
    sxy_sinv: np.ndarray
    syy_sinv: np.ndarray
    e2_x: np.ndarray
    e2_y: np.ndarray
    derivative_valid: np.ndarray
    core_candidate_mask: np.ndarray
    input_valid_mask: np.ndarray
    detections: tuple[TrapDetection, ...]
    config: DetectionConfig
    grid_spacing_m: tuple[float, float]
    qc_flags: tuple[str, ...] = field(default_factory=tuple)
    smoothing_radius_cells: int = 0
    smoothing_equivalent_sigma_m: tuple[float, float] = (0.0, 0.0)
    smoothing_method: str = "none"

    def __post_init__(self) -> None:
        """檢查結果陣列同形，並將數值結果鎖成唯讀。"""

        arrays = {
            "s1_sinv": _freeze_array(self.s1_sinv, dtype=np.float64),
            "s2_sinv": _freeze_array(self.s2_sinv, dtype=np.float64),
            "sxx_sinv": _freeze_array(self.sxx_sinv, dtype=np.float64),
            "sxy_sinv": _freeze_array(self.sxy_sinv, dtype=np.float64),
            "syy_sinv": _freeze_array(self.syy_sinv, dtype=np.float64),
            "e2_x": _freeze_array(self.e2_x, dtype=np.float64),
            "e2_y": _freeze_array(self.e2_y, dtype=np.float64),
            "derivative_valid": _freeze_array(self.derivative_valid, dtype=bool),
            "core_candidate_mask": _freeze_array(self.core_candidate_mask, dtype=bool),
            "input_valid_mask": _freeze_array(self.input_valid_mask, dtype=bool),
        }
        shape = arrays["s1_sinv"].shape
        if len(shape) != 2:
            raise ValueError("SnapshotResult 的場陣列必須是二維")
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("SnapshotResult 的場陣列形狀必須一致")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        if len(self.grid_spacing_m) != 2 or not all(
            np.isfinite(float(value)) and float(value) > 0.0 for value in self.grid_spacing_m
        ):
            raise ValueError("grid_spacing_m 必須是兩個正有限格距")
        object.__setattr__(self, "grid_spacing_m", tuple(float(value) for value in self.grid_spacing_m))
        object.__setattr__(self, "detections", tuple(self.detections))
        object.__setattr__(self, "qc_flags", tuple(str(flag) for flag in self.qc_flags))
        if isinstance(self.smoothing_radius_cells, bool) or self.smoothing_radius_cells not in (0, 1, 2, 3):
            raise ValueError("SnapshotResult.smoothing_radius_cells 必須是 0、1、2 或 3")
        if len(self.smoothing_equivalent_sigma_m) != 2 or not all(
            np.isfinite(float(value)) and float(value) >= 0.0
            for value in self.smoothing_equivalent_sigma_m
        ):
            raise ValueError("smoothing_equivalent_sigma_m 必須是兩個非負有限公尺尺度")
        object.__setattr__(
            self,
            "smoothing_equivalent_sigma_m",
            tuple(float(value) for value in self.smoothing_equivalent_sigma_m),
        )
        object.__setattr__(self, "smoothing_method", str(self.smoothing_method))

    @property
    def e2(self) -> np.ndarray:
        """回傳 `(ny, nx, 2)` 的最大特徵向量，最後一軸為 `(x, y)`。"""

        return _freeze_array(np.stack((self.e2_x, self.e2_y), axis=-1), dtype=np.float64)

    @property
    def trap_detections(self) -> tuple[TrapDetection, ...]:
        """`detections` 的語意化別名，供追蹤模組使用。"""

        return self.detections

    @property
    def grid_shape(self) -> tuple[int, int]:
        """回傳場陣列的 `(ny, nx)` 形狀。"""

        return self.s1_sinv.shape

    @property
    def smoothing_scale_m(self) -> tuple[float, float]:
        """回傳 graph diffusion 對應的近似 Gaussian `sigma` 公尺尺度。"""

        return self.smoothing_equivalent_sigma_m
