"""``aggregate`` 成果圖使用的繁體中文字型解析與可重現選擇紀錄。

本模組沿用相鄰 ``Lagrangian-Ensemble-Backtracking`` 專案的 ``report_font`` 做法：
只接受 Matplotlib 字型管理器已能在執行環境找到的系統字型，依固定候選家族順序逐一
檢查；每個候選都必須通過 FreeType 字碼表（charmap）涵蓋 TRAP 圖面實際會使用的
繁體中文、ASCII 數字與符號，才讀取字型檔 bytes 計算 SHA-256。回傳的
:class:`CJKFontSelection` 只保存家族名稱、檔名 basename、檔案摘要與必要字元數量，
不含絕對路徑，可安全寫入 manifest 與圖說。

本模組不下載字型，也不把英文預設字型當成缺件替代品；沒有合格候選時只拋出固定的
``ValueError``（不含本機路徑），呼叫端必須在寫任何圖之前 fail closed，圖面才不會
出現方框（tofu）。必要字元集合由 :data:`FIGURE_TEXT`（TRAP 圖面全部固定 zh-TW
文字的唯一來源）、五個研究站顯示名稱與 ASCII／符號清單共同建立，圖面繪製只可
使用這裡登錄的文字。
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from matplotlib import font_manager
from matplotlib.ft2font import FT2Font

__all__ = [
    "CJK_FONT_CANDIDATES",
    "FIGURE_TEXT",
    "REQUIRED_FIGURE_GLYPHS",
    "STUDY_SITE_NAMES_ZH",
    "CJKFontSelection",
    "cjk_font_context",
    "resolve_cjk_font",
]

# 候選順序是可重現性契約的一部分：前面的字型只要完整涵蓋必要字元，就不可被後面的
# 系統字型取代。名稱保留原始 family label，不由作業系統自動排序。
CJK_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Source Han Sans TC",
    "PingFang TC",
    "Heiti TC",
    "Arial Unicode MS",
)

STUDY_SITE_NAMES_ZH: Final[tuple[str, ...]] = ("貢寮", "龜山島西側", "新竹外海", "南灣", "連江")
"""五個研究站的 zh-TW 顯示名稱（``configs/study_design.json`` 的 ``study_site_name_zh``）。"""

# TRAP aggregate 圖面所有固定文字。key 只供程式引用；值是實際畫在圖上的字串。
# 動態部分（數字、site 名稱、period、param_id、字型家族）另由 ASCII／站名清單涵蓋。
FIGURE_TEXT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "science_model_only": "MODEL_ONLY 表層短期吸引潛勢",
        "science_pilot_only": "PILOT_ONLY",
        "footer_period": "期間",
        "footer_valid": "有效快照",
        "footer_font": "字型",
        "footer_attribution": "© OpenStreetMap contributors / ODbL",
        "footer_hotspot_area": "熱點面積",
        "footer_hotspot_cells": "熱點格數",
        "footer_threshold": "門檻",
        "lon": "經度 (°E)",
        "lat": "緯度 (°N)",
        "fraction": "比例",
        "hour": "小時",
        "spatial_title": "空間分布",
        "curve_pass_frequency": "TRAP 曲線出現頻率",
        "core_frequency": "TRAP 核心出現頻率",
        "mean_core_s1": "平均核心吸引強度",
        "mean_lifetime": "完整軌跡平均壽命",
        "anchor": "研究地點位置",
        "hotspot_title": "熱點分布",
        "hotspot_top": "前",
        "hotspot_curve_legend": "聚集熱區（TRAP 曲線出現頻率前 10 %）",
        "hotspot_core_legend": "TRAP 核心最常出現的位置（核心出現頻率前 10 %）",
        "hotspot_core_legend_outside": "TRAP 核心最常出現的位置（聚集熱區外）",
        "footer_min_cells": "圖面熱點最小連通格數",
        "hotspot_area": "熱點面積",
        "seasonal_title": "季節變化",
        "season_DJF": "冬季 (DJF)",
        "season_MAM": "春季 (MAM)",
        "season_JJA": "夏季 (JJA)",
        "season_SON": "秋季 (SON)",
        "strong_core_occurrence": "強核心出現率",
        "mean_pass_fraction": "平均通過格點比例",
        "ci95": "95% 信賴區間",
        "lifetime_title": "壽命與吸引強度演化",
        "ccdf_title": "壽命互補累積分布",
        "lifetime_hours": "壽命 (小時)",
        "ccdf_ylabel": "P(壽命 ≥ x)",
        "uncensored": "完整軌跡",
        "all_tracks": "全部軌跡",
        "evolution_title": "吸引強度隨正規化年齡演化",
        "evolution_subtitle": "起訖皆在觀測期內且壽命",
        "normalized_age": "正規化年齡",
        "n_tracks": "條軌跡",
        "timeseries_title": "逐日時間序列",
        "strong_core_daily": "含強核心的有效快照比例",
        "pass_fraction_daily": "平均通過格點比例",
        "daily": "逐日",
        "running_mean_30d": "30 日滑動平均",
        "date": "日期",
        "no_data": "樣本數 0",
        "strong_threshold": "強核心門檻",
        "null_ccdf_legend": "隨機時間序對照",
        "null_band_legend": "跨 seed 最小–最大",
        "excess": "超額",
        "actual_short": "實際",
        "random_short": "隨機",
        "table_sep": "｜",
        "day_unit": "天",
        "lifetime_days": "壽命 (天)",
        "real_short": "真實",
        "null_short": "對照",
        "lifetime_refers_to": "壽命對象",
        "strong_core_structures": "強核心結構",
        "all_qualified_structures": "全部合格結構",
        "per_site_top": "各站前",
        # --- WP-H：代表 TRAP、方法比較、長壽命 track、track 密度、傳播速度 ---
        "example_title": "代表性 TRAP 與後續演化",
        "example_subtitle_track": "代表軌跡",
        "example_birth": "起始時刻",
        "example_lifetime": "壽命",
        "example_observations": "個觀測",
        "example_rule": "代表軌跡：核心在站內出生、起訖皆在觀測期內、壽命最長（同壽命取核心最強）",
        "example_rule_threshold": "壽命門檻內",
        "example_rule_fallback": "壽命皆在門檻以內時，取最長者",
        "example_all_curves": "合格 TRAP 曲線",
        "example_chosen_curve": "代表軌跡起始曲線與核心",
        "example_later_curves": "後續曲線（約 1/3、2/3 壽命與最後觀測）",
        "example_core_path": "核心路徑",
        "example_velocity": "表層流速",
        "example_no_track": "站內出生且起訖皆在觀測期內的軌跡數為 0（空白圖）",
        "example_no_track_short": "代表軌跡數 0",
        "method_title": "單時刻方法比較",
        "method_panel_a": "(a) s1 與 TRAP 曲線",
        "method_panel_b": "(b) 瞬時流線與 TRAP 曲線",
        "method_panel_c": "(c) 水平散度與 TRAP 曲線",
        "method_note_streamline": "流線是瞬時速度場的幾何線，與粒子軌跡是不同的量。",
        "method_note_divergence": "水平散度只反映流場輻合，與 TRAP 吸引（s1）是不同的量，兩者各自解讀。",
        "divergence_label": "水平散度",
        "trap_curve": "TRAP 曲線",
        "trap_core": "TRAP 核心",
        "long_tracks_title": "長壽命核心路徑",
        "long_tracks_criterion": "站內出生、起訖皆在觀測期內、壽命 ≥",
        "long_tracks_start": "路徑起點",
        "long_tracks_lifetime": "壽命",
        "long_tracks_colorbar_days": "單一結構連續存在天數",
        "long_tracks_colorbar_hours": "單一結構連續存在時數",
        "long_capped_a": "僅繪最長的",
        "long_capped_b": "條，共",
        "long_capped_c": "條",
        "null_long_mean": "隨機時間序對照（各 seed 平均）",
        "track_density_title": "軌跡通過格點數",
        "track_pass_all": "通過該格的不同軌跡數（全部）",
        "track_pass_long": "通過該格的長壽命軌跡數 ≥",
        "track_pass_note": "每條軌跡在每格計一次；核心格點取自觀測核心位置最近格點。",
        "propagation_title": "TRAP 核心傳播速度",
        "propagation_east": "東向分量 cx",
        "propagation_north": "北向分量 cy",
        "propagation_rose": "傳播方向（去向）",
        "propagation_median": "中位數",
        "propagation_count": "頻率",
        "propagation_mean_speed": "平均速率",
        "propagation_samples": "樣本數",
        "propagation_tracks": "條軌跡",
        "sample_unit": "個",
        # --- WP-I：次網格核心、近岸診斷、靜止／移動 track ---
        "coast_buffer_prefix": "近岸",
        "coast_buffer_suffix": "km 帶",
        "coastal_diag_title": "近岸診斷",
        "coast_dist_xlabel": "核心距最近陸地或分析區外格點 (km)",
        "coast_hist_all": "全部合格核心",
        "coast_hist_strong": "強核心",
        "coast_hist_long": "長壽命軌跡核心",
        "coast_hist_title": "(a) 核心距岸距離分布",
        "coast_ccdf_title": "(b) 壽命互補累積分布：近岸型與近岸外",
        "coast_hist_ylabel": "比例（各組正規化）",
        "coast_offshore_real": "近岸外（實際）",
        "coast_coastal_real": "近岸型（實際）",
        "coast_offshore_null": "近岸外（隨機對照）",
        "coast_coastal_null": "近岸型（隨機對照）",
        "coast_type_note": "近岸型：軌跡平均核心距岸 < 緩衝帶寬；近岸外：≥ 緩衝帶寬",
        "coast_cap_label": "≥",
        "coast_floor_a": "檢出下限 ≈",
        "coast_floor_b": "（核心定位自此起算）",
        "coast_footnote": "近岸帶以核心平均距岸判定；近岸型與近岸外各與隨機對照比較",
        "coast_no_tracks": "完整軌跡數 0",
        "coastal_locked_legend": "近岸型軌跡（虛線）",
        "offshore_track_legend": "近岸外軌跡（實線）",
        "long_coast_real": "實際長壽命軌跡：近岸型",
        "long_coast_offshore": "近岸外",
        "long_coast_null": "隨機對照（各 seed 平均）：近岸型",
        "coastal_example_title": "近岸型代表軌跡",
        "coastal_tag": "（近岸型）",
        "below_null_tag": "（壽命 ≤ 隨機對照 p99）",
        "example_rule_offshore": "代表軌跡：站內出生、起訖皆在觀測期內、近岸外，且壽命超過隨機對照 p99 的最長者",
        "example_rule_offshore_nonull": "代表軌跡：站內出生、起訖皆在觀測期內、近岸外的最長者",
        "example_rule_fallback_coast": "代表軌跡：近岸外且超過對照的軌跡數為 0 時，取整體最長者",
        "example_rule_coastal": "近岸型代表：站內出生、起訖皆在觀測期內、平均距岸 < 緩衝帶寬的最長者",
        "example_no_coastal": "近岸型長壽命軌跡數為 0",
        "later_list_title": "後續核心相對起始位移",
        "displacement": "位移",
        "refined_core_note": "核心位置為 3x3 二次曲面次網格定位",
        "propagation_moving": "移動軌跡",
        "propagation_stationary": "靜止軌跡",
        "propagation_stationary_def": "靜止：擬合位移在 1 格以內",
        "propagation_fit_note": "每條軌跡以核心位置 x(t)、y(t) 最小平方線性擬合（至少 {min_obs} 個觀測）；直方圖與玫瑰圖只含移動軌跡，描述模型內 TRAP 核心的位移，與水體或漂浮物速度是不同的量。",
        "propagation_resid": "殘差 RMS 中位數",
        "propagation_stationary_frac": "靜止比例",
        "propagation_min_obs_a": "擬合至少",
        "propagation_min_obs_b": "個觀測",
        "propagation_fitted": "已擬合軌跡",
        "unit_m": "m",
        # --- 2026-10-13 修正：累積天數地圖、逐時與逐日熱點重疊 ---
        "cumulative_days_title": "累積天數",
        "cumulative_days_colorbar": "累積天數（日）",
        "cumulative_days_subtitle_a": "站內有效日數共",
        "cumulative_days_subtitle_b": "日",
        "hotspot_overlap_title": "逐時與逐日熱點重疊",
        "hotspot_overlap_hourly_only": "僅逐時熱點",
        "hotspot_overlap_daily_only": "僅逐日熱點",
        "hotspot_overlap_both": "兩者皆為熱點",
        "hotspot_overlap_site_cells": "站內其他格點",
        "hotspot_overlap_cells": "格",
        "hotspot_overlap_hourly": "逐時",
        "hotspot_overlap_daily": "逐日",
        "hotspot_overlap_intersection": "交集",
    }
)

# ASCII 字母／數字與圖面用到的標點、數學與單位符號；含全形標點以涵蓋日後標題用字。
_ASCII_AND_SYMBOL_GLYPHS: Final[str] = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "%()[]-–—_/,.:;=+×≤≥<>°²±·© −（）：；，｜／"
)

# 以 frozenset 封存 Unicode code point inventory，避免 caller 改寫必要字元。
REQUIRED_FIGURE_GLYPHS: Final[frozenset[str]] = frozenset(
    "".join(FIGURE_TEXT.values()) + "".join(STUDY_SITE_NAMES_ZH) + _ASCII_AND_SYMBOL_GLYPHS
)

_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_NO_VALID_FONT_MESSAGE: Final[str] = (
    "找不到涵蓋 TRAP 圖面必要字元的合格繁體中文字型（候選：" + "、".join(CJK_FONT_CANDIDATES) + "）；"
    "圖面不可輸出方框字，請安裝 Noto Sans CJK TC 或改用 --no-figures"
)


@dataclass(frozen=True, slots=True)
class CJKFontSelection:
    """已通過必要字元檢查的繁體中文字型 immutable provenance record。

    ``family`` 是固定候選清單中的 family label；``filename`` 只允許 basename；
    ``file_sha256`` 是字型檔原始 bytes 的小寫 SHA-256；``required_glyph_count`` 是
    本模組必要 Unicode code point 的數量。此 record 只描述圖面工程設定，不代表任何
    海洋科學結果。
    """

    family: str
    filename: str
    file_sha256: str
    required_glyph_count: int

    def __post_init__(self) -> None:
        """建立 record 時阻擋錯誤型別、路徑欄位與非 canonical 摘要。"""

        if type(self.family) is not str or not self.family:
            raise TypeError("family 必須是非空原生 str")
        if self.family not in CJK_FONT_CANDIDATES:
            raise ValueError("family 必須精確屬於 CJK_FONT_CANDIDATES")
        if type(self.filename) is not str or not self.filename:
            raise TypeError("filename 必須是非空原生 str basename")
        if self.filename in {".", ".."} or "/" in self.filename or "\\" in self.filename or Path(self.filename).is_absolute():
            raise ValueError("filename 必須是不含目錄的 basename")
        if type(self.file_sha256) is not str or _SHA256_PATTERN.fullmatch(self.file_sha256) is None:
            raise ValueError("file_sha256 必須是 64 碼小寫 SHA-256")
        if type(self.required_glyph_count) is not int:
            raise TypeError("required_glyph_count 必須是原生 int，且不可是 bool")
        if self.required_glyph_count != len(REQUIRED_FIGURE_GLYPHS):
            raise ValueError("required_glyph_count 必須精確等於必要字元數量")

    def as_manifest_dict(self) -> dict[str, str | int]:
        """回傳可直接寫入 manifest 的純 JSON 字典（不含路徑）。"""

        return {
            "family": self.family,
            "filename": self.filename,
            "file_sha256": self.file_sha256,
            "required_glyph_count": self.required_glyph_count,
        }


def _is_ordinary_font_file(filename: Path) -> bool:
    """以 ``lstat`` 驗證候選目標是非符號連結的普通檔案；任何 I/O 錯誤都視為不合格。"""

    try:
        file_stat = os.lstat(filename)
    except OSError:
        return False
    return stat.S_ISREG(file_stat.st_mode) and not stat.S_ISLNK(file_stat.st_mode)


def _font_covers_required_glyphs(filename: Path) -> bool:
    """以 FreeType charmap 判斷字型是否涵蓋每一個必要 Unicode code point。

    charmap 的 key 是 Unicode code point，因此對每個必要字元用 ``ord`` 精確比對，
    不以 Matplotlib 的 fallback 字型結果代替缺少的字元。
    """

    charmap = FT2Font(str(filename)).get_charmap()
    return all(ord(glyph) in charmap for glyph in REQUIRED_FIGURE_GLYPHS)


def _sha256_font_bytes(filename: Path) -> str:
    """讀取字型檔完整 bytes 並回傳小寫 SHA-256。"""

    return hashlib.sha256(filename.read_bytes()).hexdigest()


def resolve_cjk_font() -> CJKFontSelection:
    """依固定候選順序解析可支援 TRAP 圖面字元的繁體中文字型。

    每輪以 ``font_manager.findfont(..., fallback_to_default=False)`` 做 family
    matching（避免找不到 CJK family 時被悄悄換成西文預設字型），再以 FreeType
    charmap 驗證全部必要 glyph，最後以檔案 bytes 產生摘要。所有候選都不存在、
    缺字或無法讀取時只拋出不含路徑的固定 ``ValueError``。
    """

    for family in CJK_FONT_CANDIDATES:
        try:
            filename = Path(
                font_manager.findfont(font_manager.FontProperties(family=[family]), fallback_to_default=False)
            )
            if not _is_ordinary_font_file(filename):
                continue
            if not _font_covers_required_glyphs(filename):
                continue
            return CJKFontSelection(
                family=family,
                filename=filename.name,
                file_sha256=_sha256_font_bytes(filename),
                required_glyph_count=len(REQUIRED_FIGURE_GLYPHS),
            )
        # font_manager、FreeType 與檔案 I/O 的底層例外可能包含絕對路徑；候選視為不合格
        # 並繼續，最終由公開邊界統一拋出固定訊息。
        except Exception:
            continue
    raise ValueError(_NO_VALID_FONT_MESSAGE)


@contextmanager
def cjk_font_context(selection: CJKFontSelection) -> Iterator[None]:
    """在區塊內把 Matplotlib 全域字型設為已驗證的 CJK 字型，離開時還原全域狀態。

    ``font.family`` 為 ``[選定家族, "DejaVu Sans"]``：CJK 字型負責中文，DejaVu 只
    補數學符號；``axes.unicode_minus`` 設為 ``False``，避免 CJK 字型缺少 U+2212
    造成軸刻度負號變方框。使用 ``rc_context``，不污染呼叫端的 rcParams。
    """

    import matplotlib

    with matplotlib.rc_context({"font.family": [selection.family, "DejaVu Sans"], "axes.unicode_minus": False}):
        yield
