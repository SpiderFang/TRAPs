"""Plain-Chinese post-processing of the generated report Markdown (LBT terminology).

Math ($...$, $$...$$), image lines and file-list table rows are left alone.  Code and file
names are replaced by plain descriptions; they reappear only in the closing 附註 tables.
"""
from __future__ import annotations

import re

CJK = "　-鿿＀-￯"


# code identifiers -> plain Chinese names (also used for the closing 附註 table)
ID_ZH = [
    ("min_core_s1_sinv", "核心最小量值"),
    ("min_eigenvalue_gap_sinv", "特徵值間隙門檻"),
    ("local_min_radius_cells", "局部極小值鄰域半徑"),
    ("local_min_tolerance_sinv", "局部極小值並列容許"),
    ("core_min_separation_m", "核心最小間距"),
    ("core_subgrid", "次網格核心修正"),
    ("curve_s1_stop_sinv", "曲線停止之 s₁ 門檻"),
    ("curve_min_relative_strength", "曲線相對強度下限"),
    ("curve_monotonic_tolerance_sinv", "曲線單調性容許"),
    ("max_curve_length_m", "曲線最大長度"),
    ("max_integration_steps", "最大積分步數"),
    ("max_turn_angle_deg", "最大轉角"),
    ("min_curve_points", "曲線最少點數"),
    ("integration_step_m", "積分步長"),
    ("smoothing_radius_cells", "平滑半徑"),
    ("min_coverage", "最低有效覆蓋率"),
    ("strong_quantile", "強核心分位數"),
    ("hotspot_quantile", "熱點分位數"),
    ("hotspot_min_cells", "熱點最小連通格數"),
    ("max_missing_frames", "最多缺失時刻數"),
    ("coast_distance.cap_m", "距岸距離場上限"),
    ("coastal_buffer_km", "近岸緩衝帶寬度"),
    ("propagation.min_observations", "傳播速度最少觀測數"),
    ("prop_stationary_frac", "靜止比例"),
    ("relative_strength_threshold", "相對強度門檻"),
    ("strong_core_thresholds", "強核心門檻紀錄"),
    ("track_core_filter", "結構軌跡核心篩選紀錄"),
    ("tracking_config", "追蹤設定"),
    ("detection_config", "檢出設定"),
    ("track_n_uncensored", "未截尾結構軌跡數"),
    ("duration_seconds", "首末觀測跨度"),
    ("recomputed_core_offset_cells", "重算核心偏移格數"),
]
ID_REPL = [(a, b) for a, b in ID_ZH]

SPECIAL = [
    ("東向分量檔名為 u_surface_mps、北向分量為 v_surface_mps，單位為每秒公尺", "分量為東向表層流速 u 與北向表層流速 v，單位為每秒公尺"),
    ("（座標 x_m、y_m；schema 3.0.0）", "（座標 x、y 之單位為公尺；資料格式版本 3.0.0）"),
    ("狀態為 ready", "狀態為可用"),
    ("標準等級、ready", "標準等級、可用"),
    ("| READY | READY |", "| 可用 | 可用 |"),
    ("sort_and_deduplicate_prefer_last （依 UTC 穩定排序，重複者保留較晚月份）", "「依 UTC 穩定排序、重複者保留較晚月份」之規則"),
    ("採 coastline_corrected_v2 版本", "採校正海岸線版本"),
    ("coastline_corrected_v2 版本", "校正海岸線版本"),
    ("與專案 GeoJSON 同一 SHA-256", "與專案海岸線向量檔同一 SHA-256"),
    ("由 scripts/export_ocean_masks.py 逐位元匯出為 analysis_ocean_mask_coastline_corrected_v2.npy 並附 metadata 附屬檔", "由遮罩匯出程式逐位元匯出為分析海域遮罩檔並附中繼資料附屬檔"),
    ("mask_static.npy 只代表", "靜態網格遮罩檔只代表"),
    ("專案內之 GeoJSON（© OpenStreetMap contributors，ODbL）", "專案內之海岸線向量檔（© OpenStreetMap contributors，ODbL）"),
    ("本計畫研究設計（configs/study_design.json）", "本計畫研究設計設定檔"),
    ("NAS 複製到 SERVER 本機 SSD", "網路儲存（NAS）複製到伺服器本機固態硬碟（SSD）"),
    ("NAS 只當來源", "網路儲存只當來源"),
    ("schema、來源與逐時遮罩之檢查採 fail-closed（不符即停止，不自行補救）", "資料格式、來源與逐時遮罩之檢查採嚴格停止原則（不符即停止，不自行補救）"),
    ("保守之 fail-closed 設計", "保守之嚴格停止設計"),
    ("同 schema 之日產品（cache_kind 為 derived_daily_mean）", "同資料格式之日產品（標註為日平均衍生產品）"),
    ("各月 metadata.json 之 derived_from.filter", "各月中繼資料之濾波紀錄"),
    ("同 schema，", "同資料格式，"),
    ("schema", "資料格式"),
    ("（time_gap、data_gap）", "（時間缺口、資料缺口）"),
    ("gap closing（", "缺口銜接（"),
    ("無效處為 NaN。三、grid 目錄", "無效處為無效值。三、網格目錄"),
    ("tracks.csv（每條結構軌跡之起訖、觀測數、截尾旗標與壽命）與逐觀測之 track_observations.npz，以及 tracking_summary 與壽命直方資料。", "結構軌跡表（每條結構軌跡之起訖、觀測數、截尾旗標與壽命）、逐觀測表、追蹤摘要與壽命直方資料。"),
    ("逐站統計檔／.json", "逐站統計檔"),
    ("逐格空間陣列（spatial/*.npz）", "逐格空間陣列檔"),
    ("階段 1 detect-shards", "階段 1（區域月份檢出）"),
    ("階段 2 track-period", "階段 2（期間追蹤）"),
    ("階段 3 aggregate", "階段 3（彙整）"),
    ("另有 build-daily-fields 建立日產品", "另有日產品建置步驟"),
    ("本計畫程式庫（src/traps 與 scripts）", "本計畫程式庫"),
    ("分片檔內容：snapshots.npz（逐時 coverage 與 packbits 有效遮罩）、detections.npz（核心、曲線 ragged array、curve_pass_cells、停止原因整數碼、次網格核心與距岸距離）、grid_reference.npz（含距岸距離場）。", "分片檔內容：快照檔（各快照之有效覆蓋率與壓縮有效遮罩）、檢出檔（核心、不等長之曲線陣列、曲線通過格點、停止原因整數碼、次網格核心與距岸距離）、網格參考檔（含距岸距離場）。"),
    ("執行紀錄檔.sha256", "執行紀錄檔之雜湊檔"),
    ("全部 artifact", "全部輸出檔"),
    ("aggregate 重新核對", "彙整步驟重新核對"),
    ("（拒絕 PILOT_ONLY 與正式資料混用）", "（拒絕試行資料與正式資料混用）"),
    ("_SUCCESS 標記與 --resume", "完成標記檔與續跑選項"),
    ("STOP_REASON_CODES", "停止原因代碼"),
    ("invalid_mask", "遮罩無效"),
    ("outside_flow_domain", "超出流場區域"),
    ("s1_not_monotonic", "s₁ 未單調回升"),
    ("s1_threshold", "s₁ 達門檻"),
    ("invalid_step", "步進無效"),
    ("zero_step", "零步長"),
    ("curve_pass_cells", "曲線通過格點"),
    ("detection_keys", "偵測鍵"),
    ("--shuffle-seed", "亂數種子參數"),
    ("track-period 以", "期間追蹤步驟以"),
    ("（null_model/time_shuffle", "（隨機對照"),
    ("標記 refined 為否", "標記為未精修"),
    ("（eddy-census）", "（渦旋普查）"),
    ("eddy-census 慣例", "渦旋普查慣例"),
    ("quadratic_3x3（refine 為 是）", "二次曲面 3×3 擬合（啟用精修）"),
    ("(pytest --collect-only)", "(自動化測試)"),
    ("（pytest --collect-only）", "（自動化測試蒐集）"),
    ("CLI", "命令列"),
    ("NaN", "無效值"),
    ("（分層為 all）", "（分層：全部）"),
    ("（分層 all）", "（分層：全部）"),
    ("全部（all）", "全部"),
    ("日區塊逐日區塊重抽樣", "逐日區塊重抽樣"),
    ("方括號內為", "括號內為"),
    ("方括號為", "括號內為"),
    ("aggregate_report 目錄", "彙整報告目錄"),
    ("（domains_missing 為空）", "（缺漏流場區域名單為空）"),
    ("（sites_missing 為空）", "（缺漏研究站名單為空）"),
    ("_SUCCESS 完成標記", "完成標記檔"),
    ("traps.period.aggregate.v2／0.1.0", "第 2 版／0.1.0"),
    ("執行紀錄檔 schema／套件版本", "彙整結果格式／套件版本"),
    ("熱點圖（hotspots）", "熱點圖"),
    ("壽命圖（lifetime）", "壽命圖"),
    ("長壽命路徑圖（long_tracks）", "長壽命路徑圖"),
    ("近岸外（offshore）指標", "近岸外指標"),
    ("aggregate 完成時間", "彙整完成時間"),
    ("aggregate 耗時", "彙整耗時"),
    ("aggregate 皆完成", "彙整皆完成"),
    ("aggregate（", "彙整（"),
    ("detect-shards", "區域月份檢出"),
    ("build-daily-fields", "日產品建置"),
    ("在 SERVER", "在伺服器"),
    ("SERVER", "伺服器"),
    ("執行環境為 伺服器", "執行環境為伺服器"),
]

# exact phrases first (longest first), then generic words
PHRASES = [
    ("日區塊 bootstrap", "逐日區塊重抽樣"),
    ("日區塊 逐日區塊重抽樣", "逐日區塊重抽樣"),
    ("時間置換 null 追蹤", "隨機時間置換對照追蹤"),
    ("時間置換 null", "隨機對照"),
    ("null model", "隨機對照"),
    ("flow domain", "流場區域"),
    ("domain-month", "區域月份"),
    ("逐時 r1_hourly", "逐時產品"),
    ("日產品 d1_daily25h", "日產品"),
    ("逐時產品 r1_hourly", "逐時產品"),
    ("（param_id 為 r1_hourly）", ""),
    ("（param_id 為 d1_daily25h）", ""),
    ("（r1_hourly、d1_daily25h）", "（逐時產品與日產品）"),
    ("param_id", "執行識別"),
    ("r1_hourly", "逐時產品"),
    ("d1_daily25h", "日產品"),
    ("offshore_exceeds_null", "近岸外且壽命超過隨機對照"),
    ("「僅模型資料（MODEL_ONLY）、無獨立觀測驗證」", "「僅模式資料、無獨立觀測驗證」"),
    ("僅模型資料（MODEL_ONLY）", "僅模式資料"),
    ("MODEL_ONLY", "僅模式資料"),
    ("manifest.json", "執行紀錄檔"),
    ("site_statistics.csv", "逐站統計檔"),
    ("site_statistics", "逐站統計檔"),
    ("manifest", "執行紀錄檔"),
    ("state_label", "狀態標記"),
    ("runtime_seconds", "執行耗時"),
    ("git_dirty", "程式修改標記"),
    ("git_commit", "程式版本"),
    ("stratum", "分層"),
    ("bootstrap", "逐日區塊重抽樣"),
    ("CCDF", "互補累積分布"),
    ("RMS", "均方根"),
    ("p99", "第 99 百分位數"),
    ("p90", "第 90 百分位數"),
    ("p10", "第 10 百分位數"),
    ("DJF", "冬季（12–2 月）"),
    ("MAM", "春季（3–5 月）"),
    ("JJA", "夏季（6–8 月）"),
    ("SON", "秋季（9–11 月）"),
    ("最近 anchor", "最近錨點"),
    ("anchor", "錨點"),
    ("worker", "工作程序"),
    ("shard", "分片檔"),
    ("sidecar", "附屬檔"),
    ("tracks", "結構軌跡"),
    ("track", "結構軌跡"),
    ("null", "隨機對照"),
    ("seed", "亂數種子"),
]
SPECIAL = [
    ("執行紀錄檔 schema／套件版本", "彙整結果格式／套件版本"),
    ("狀態（state_label）", "狀態"),
    ("狀態（狀態標記）", "狀態"),
] + SPECIAL + [
    ("（min_core_s1_sinv）", ""),
    ("（local_min_radius_cells，即", "（即"),
    ("（curve_min_relative_strength）", ""),
    ("（smoothing_radius_cells 為 0；", "（平滑半徑為 0；"),
    ("，對應 min_eigenvalue_gap_sinv", ""),
    ("（strong_quantile）", ""),
    ("（min_coverage）", ""),
    ("null_model 為 time_shuffle 與種子", "隨機對照方式為時間置換及亂數種子"),
    ("上標 real 為真實追蹤", "上標「真實」為真實追蹤"),
    ("只在 all 分層有值", "只在「全部」分層有值"),
    ("另有 coastal_diagnostic 圖", "另有近岸診斷圖"),
    ("另有 propagation 圖", "另有傳播速度圖"),
    ("quadratic_3x3（refine 為是）", "二次曲面 3×3 擬合（啟用精修）"),
    ("所有 artifact", "所有輸出檔"),
    ("| git commit ", "| "),
    ("程式版本程式版本", "程式版本"),
    ("aggregate 三階段", "彙整三階段"),
    ("至 aggregate）", "至彙整）"),
    ("（spatial）", ""), ("（example）", ""),
    ("（SERVER", "（伺服器"), ("gap closing", "缺口銜接")]
STAGE_B = SPECIAL + ID_REPL


def _rx(a):
    tail = r"(?![A-Za-z0-9_\-]|\.(?:csv|json|py|npz))" if a[-1].isascii() and a[-1].isalnum() else ""
    head = r"(?<![A-Za-z0-9_/.\-])" if a[0].isascii() and a[0].isalnum() else ""
    return re.compile(head + re.escape(a) + tail)


WORDRE_B = [(_rx(a), b) for a, b in STAGE_B]
WORDRE = [(re.compile(r"(?<![A-Za-z0-9_/.\-])" + re.escape(a) + r"(?![A-Za-z0-9_\-]|\.(?:csv|json|py|npz))"), b) for a, b in PHRASES]


def _apply(seg: str) -> str:
    sp = lambda t: re.sub(rf"(?<=[{CJK}]) (?=[{CJK}])", "", t)  # noqa: E731
    for rx, b in WORDRE:
        seg = rx.sub(b, seg)
    seg = sp(seg)
    for rx, b in WORDRE_B:
        seg = rx.sub(b, seg)
    return sp(seg)


MATH_SUB = {"core": "核心", "site": "站", "valid": "有效", "strong": "強", "coast": "岸", "dist": "距離", "rel": "相對",
            "null": "隨機", "last": "末", "fit": "擬合", "first": "首", "curve": "曲線", "cells": "格", "unc": "未截尾",
            "real": "真實", "pass": "通過", "gap": "間隙", "cell": "格", "cov": "覆蓋"}


def localize(text: str) -> str:
    out, in_math = [], False
    for line in text.split("\n"):
        if line.strip() == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_math or line.startswith("![]("):
            out.append(line)
            continue
        if line.startswith("|") and re.search(r"\.(py|sh)\b|\bSTAGE\b|uv\.lock", line):
            out.append(line)  # file-list rows
            continue
        parts = re.split(r"(\$[^$]+\$)", line)
        out.append("".join(p if p.startswith("$") and p.endswith("$") and len(p) > 1 else _apply(p) for p in parts))
    t = "\n".join(out)
    for a, b in [("distribution function, 互補累積分布）", "distribution function, CCDF）"),
                 ("root mean square, 均方根）", "root mean square, RMS）"),
                 ("（block 逐日區塊重抽樣）", "（block bootstrap）"),
                 ("block 逐日區塊重抽樣", "block bootstrap")]:
        t = t.replace(a, b)
    for en, zh in MATH_SUB.items():
        t = t.replace("\\mathrm{%s}" % en, "\\text{%s}" % zh)
        t = t.replace("\\operatorname{%s}" % en, "\\text{%s}" % zh)
    for en, zh in (("DJF", "冬季"), ("MAM", "春季"), ("JJA", "夏季"), ("SON", "秋季")):
        t = t.replace("| %s（" % zh, "| %s（" % zh)
        t = re.sub(r"^\| %s（[^）]*） \|" % zh, "| %s |" % zh, t, flags=re.M)
        t = re.sub(r"^\| %s \|" % en, "| %s |" % zh, t, flags=re.M)
    t = re.sub(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d):\d\d\+00:00", "\\1\u00a0\\2", t)
    t = t.replace("| 流場區域 | flow_domain_id | 研究站 |\n|---|---|---|", "| 流場區域 | 研究站 |\n|---|---|")
    t = re.sub(r"^\| ([A-D]) \| [a-z_0-9]+_cache_v3[^|]*\| ([^|]+) \|$", r"| \1 | \2 |", t, flags=re.M)
    return t


def _next_label(text: str, prefix: str) -> int:
    nums = [int(x) for x in re.findall(rf"\*\*{prefix}-(\d+)  ", text)]
    return max(nums) + 1


def method_notes(text: str) -> str:
    n = _next_label(text, "表甲")
    rows = [
        ("東向、北向表層流速 u、v", "u_surface_mps、v_surface_mps"),
        ("網格座標 x、y（公尺）", "x_m、y_m"),
        ("流場區域 A、B、C、D 之快取名稱", "northeast_taiwan_common_cache_v3、hsinchu_cache_v3、houwan_nmmba_cache_v3、lienchiang_common_cache_v3"),
        ("執行識別（逐時產品、日產品）", "param_id：r1_hourly、d1_daily25h"),
        ("執行紀錄檔、逐站統計檔", "manifest.json、site_statistics.csv／.json"),
        ("校正海岸線版本之遮罩", "coastline_corrected_v2"),
    ] + [(zh, code) for code, zh in ID_ZH if code not in ("track_core_filter", "strong_core_thresholds", "tracking_config", "detection_config", "track_n_uncensored", "duration_seconds", "recomputed_core_offset_cells", "relative_strength_threshold", "prop_stationary_frac")]
    lines = ["", "## （六）附註：程式與檔案中之參數名稱對照", "",
             f"　　本章內文以中文名稱敘述各項參數與檔案；與程式及輸出檔中之名稱對照見表甲-{n}，供複現與查核使用。", "",
             f"**表甲-{n}  中文名稱與程式、檔案名稱對照**", "", "| 本章用語 | 程式或檔案中之名稱 |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    lines += ["", "資料來源：本計畫程式庫與兩次執行之執行紀錄檔。", ""]
    return text.rstrip("\n") + "\n" + "\n".join(lines)


def results_notes(text: str) -> str:
    n = _next_label(text, "表乙")
    rows = [
        ("逐時產品、日產品（執行識別）", "r1_hourly、d1_daily25h"),
        ("逐站統計檔", "site_statistics.csv、site_statistics.json"),
        ("執行紀錄檔與雜湊檔", "manifest.json、manifest.sha256"),
        ("彙整結果格式", "traps.period.aggregate.v2"),
        ("狀態標記；僅模式資料", "state_label（READY）；MODEL_ONLY"),
        ("程式版本與修改標記", "git_commit、git_dirty"),
        ("結構軌跡；隨機對照；亂數種子", "track；null；seed（1、2）"),
        ("區域月份；分層", "domain-month；stratum（all、DJF、MAM、JJA、SON、年份）"),
        ("近岸外；近岸型；超額", "offshore；coastal；excess"),
        ("強核心門檻紀錄；結構軌跡核心篩選紀錄", "strong_core_thresholds；track_core_filter"),
        ("缺漏流場區域、缺漏研究站名單", "domains_missing、sites_missing"),
        ("成果圖檔名", "site_<站名>_{hotspots、spatial、lifetime、long_tracks、example}.png"),
    ]
    lines = ["", "## （五）附註：識別名稱與檔案名稱對照", "",
             f"　　本章內文以中文名稱敘述；與程式及輸出檔中之英文名稱對照見表乙-{n}。", "",
             f"**表乙-{n}  中文用語與程式、檔案名稱對照**", "", "| 本章用語 | 程式或檔案中之名稱 |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    lines += ["", "資料來源：本計畫程式庫與兩次執行之執行紀錄檔。", ""]
    return text.rstrip("\n") + "\n" + "\n".join(lines)


def L(label: str) -> str:
    return localize(label)
