"""Content of 四、TRAPs 執行成果 (LBT dialect Markdown); all numbers come from site_statistics / manifest."""
from __future__ import annotations

from word_md_common import (MINUS, RESULTS, RUNS, SEASONS, SITES, Doc, ci, cip, g, num, pct, ratio, sci)

CRIT = {
    "offshore_exceeds_null": "站內出生、未截尾、核心平均距岸不小於 10 公里（近岸外）且壽命大於時間置換 null 之 p99 者中壽命最長者",
    "offshore_longest": "站內出生、未截尾、近岸外者中壽命最長者",
    "fallback_coastal": "無近岸外且壽命超過 null 之 p99 之候選，退回壽命最長者（屬近岸型）",
    "fallback_offshore_below_null": "無壽命超過 null 之 p99 之近岸外候選，退回壽命最長者（近岸外但未超過 null）",
}


def m(x, n=4):
    return g(x, n)


def count_figs(run):
    return len(list((RESULTS / run / "aggregate_report" / "figures").glob("*/*.png")))


def build_results(stats, man, figmap):
    S, M = stats, man
    for rk in ("hourly", "daily"):
        for sid, _, _ in SITES:
            a = S[rk][sid]["all"]
            assert a["core_occurrence_rate"] == 1.0 and a["curve_occurrence_rate"] == 1.0

    def lifetimes(rk):
        return [3, 6, 12, 24] if rk == "hourly" else [48, 72, 168]

    def seasonal_text(sid, rk):
        rows = {s: S[rk][sid][s] for s in SEASONS}
        by_pass = sorted(SEASONS, key=lambda s: rows[s]["mean_pass_cell_fraction"])
        lo, hi = by_pass[0], by_pass[-1]
        lo_ci, hi_ci = rows[lo]["mean_pass_cell_fraction_ci95"], rows[hi]["mean_pass_cell_fraction_ci95"]
        sep = lo_ci[1] < hi_ci[0]
        by_strong = sorted(SEASONS, key=lambda s: rows[s]["strong_core_occurrence_rate"])
        slo = by_strong[0]
        name = "逐時產品" if rk == "hourly" else "日產品"
        t = (f"{name}之曲線通過格點比例以 {hi} 最大（{pct(rows[hi]['mean_pass_cell_fraction'])}）、{lo} 最小（{pct(rows[lo]['mean_pass_cell_fraction'])}），"
             f"兩者之 95% 信賴區間{'不重疊' if sep else '重疊'}；")
        vals = [rows[s]["strong_core_occurrence_rate"] for s in SEASONS]
        if min(vals) == max(vals):
            t += f"四季強核心出現率皆為 {pct(vals[0])}。"
        else:
            t += (f"強核心出現率以 {slo} 最低（{pct(rows[slo]['strong_core_occurrence_rate'])}），"
                  f"其餘季節依序為 " + "、".join(f"{s} {pct(rows[s]['strong_core_occurrence_rate'])}" for s in SEASONS if s != slo) + "。")
        return t

    def build(doc: Doc):
        P, H = doc.p, doc.h
        hm, dm = M["hourly"], M["daily"]
        doc.h(1, "四、TRAPs 執行成果")
        doc.note("本章為工項 2「瞬態吸引剖面（TRAPs）」之執行成果，併入全文時置於工項 1 成果之後；表號暫定「表乙-」，圖號暫定「圖乙-」，"
                 "併入時請重編；數值取自兩次正式執行之 site_statistics 與 manifest，未自行推算新數值。")
        P("本章彙整 TRAP 分析在 2024 年 1 月至 2025 年 12 月、五個研究站之實際執行成果。流場以兩種產品獨立分析：逐時產品（保留潮汐）與 25 小時置中平均之去潮汐日產品；"
          "兩者尺度不同，結果是兩個獨立之分析，不是同一量之兩次重複。各站彼此獨立統計，各有自己之分母、門檻、熱點與色階，本章不做站與站之排序或比較。"
          "所有結果為模型內之「表層短期吸引潛勢」（科學狀態 MODEL_ONLY），不是海廢濃度、滯留時間、海底廢聚集或污染來源。")
        P("本章用語如下。「結構軌跡」指同一 TRAP 結構在連續時刻被追蹤所形成的序列；「隨機時間置換對照」（簡稱「隨機對照」）指保留追蹤設定、僅把各可用時間槽之檢出隨機置換後重新追蹤之結果，用以估計「沒有真實時間連續性」時之壽命基準，"
          "共使用兩組亂數種子；「超額」為真實值減隨機對照值；「近岸型」指核心平均距岸小於 10 公里，「近岸外」指不小於 10 公里；「截尾」指壽命因資料缺口或研究期邊界而無法完整觀測，「未截尾」則否；"
          "「互補累積分布」（complementary cumulative distribution function, CCDF）指壽命不小於某值之比例隨該值之變化；「均方根」（root mean square, RMS）為殘差平方之平均再開方；"
          "「逐日區塊重抽樣」（block bootstrap）指以日為單位重抽樣以估計信賴區間；「快照」指某一時刻之流場；「分層」指依季節等條件切分之統計子集。每日（day⁻¹）為應變率之單位，其後僅以 day⁻¹ 表示。")
        P("本章各指標之意義如下。「出現率」以有效快照為分母，每個快照至多計一次；「強核心」為該站核心強度 $s_1$ 最強之 10%，$s_1$ 單位為每日（day⁻¹），越負越強；"
          "「曲線通過格點比例」為每個快照中，TRAP 曲線所經站內格點占站內格點之平均比例；「熱點」為該站曲線通過頻率最高之 10% 格點，是相對於該站自己分布之定義；"
          "「壽命」只指各站強核心結構之未截尾 track，慣例為末次觀測時間減首次觀測時間再加一個取樣間隔；「超額」為真實比例減時間置換 null 比例（兩個種子平均），「比值」為真實除以 null；"
          "方括號內為日區塊 bootstrap 之 95% 信賴區間（只有出現率與曲線通過格點比例附有）；「近岸帶」為距非分析海域格點 10 公里以內。")
        # headline numbers for 本節重點 (no cross-site ranking: only universal statements)
        h_med = {S["hourly"][s]["all"]["track_median_lifetime_hours"] for s, _, _ in SITES}
        assert h_med == {1.0}
        for sid_, _, _ in SITES:
            ah, ad = S["hourly"][sid_]["all"], S["daily"][sid_]["all"]
            assert all(ah[f"excess_track_frac_uncensored_ge_{h}h"] > 0 for h in (3, 6, 12)), sid_
            assert ad["excess_track_frac_uncensored_ge_168h"] > 0 and ad["ratio_track_frac_uncensored_ge_168h"] > 1, sid_
            assert ah["frac_strong_cores_within_buffer"] > ah["frac_cores_within_buffer"], sid_
            assert ad["frac_strong_cores_within_buffer"] > ad["frac_cores_within_buffer"], sid_
        medd_all = [S["daily"][s_]["all"]["track_median_lifetime_hours"] for s_, _, _ in SITES]
        assert min(medd_all) == 24 and max(medd_all) == 48
        P(f"本節重點：4 個流場區域 × 24 個月 = 96 個 domain-month 全數完成偵測、追蹤與彙整，逐時產品每站有 {num(S['hourly']['gongliao']['all']['n_valid_snapshots'])} 個有效快照、"
          f"日產品每站有 {num(S['daily']['gongliao']['all']['n_valid_snapshots'])} 個有效快照。各站核心出現率與曲線出現率在兩個產品中皆為 100%，不具區辨力，"
          "有資訊之頻率指標為強核心出現率與曲線通過格點比例，強度指標為強核心門檻與熱點內之 $s_1$。逐時產品強核心結構之壽命中位數在五站皆為 1 小時，"
          "日產品為 1–2 天（24–48 小時）；真實壽命在逐時 3、6、12 小時與日產品 168 小時（7 天）之各門檻皆高於時間置換 null，但日產品 2–3 天之壽命有相當部分可由機會串接解釋。"
          "強核心與長壽命 track 偏向集中於 10 公里近岸帶，且核心在距非海域格點約 2–3 公里內無法檢出，因此近岸結果須視為可能之地形鎖定，而非漂移之收斂帶。")

        # ============================================================ （一）
        H(2, "（一）執行規模與完成度")
        H(3, "1、資料規模與完成度")
        a_h, a_d = S["hourly"]["gongliao"]["all"], S["daily"]["gongliao"]["all"]
        n_png = (count_figs(RUNS["hourly"]), count_figs(RUNS["daily"]))
        P(f"本計畫資料為中央氣象署 OCM／SCHISM 表層流場，預處理為 1 公里規則公尺網格，涵蓋 4 個流場區域、5 個研究站，期間 2024 年 1 月至 2025 年 12 月，"
          f"共 4 × 24 = 96 個 domain-month，依專案記錄全數為標準等級且狀態為 ready。逐時產品另有 17,196 列去除 72 個月交界重複之 UTC 時刻後，為 {num(a_h['n_valid_snapshots'])} 個有效快照"
          f"（缺漏 420 個小時，含 48 個長於 1 小時之缺口，最長 50 小時，一律不補值）；日產品為 {num(a_d['n_valid_snapshots'])} 個有效快照（每個流場區域另有 16 日因視窗小時數少於 23 而省略）。"
          f"{doc.T('scale')}列出兩個產品之執行規模與完成度。兩次執行之 aggregate 皆完成，manifest 記錄無缺漏流場區域（domains_missing 為空）與無缺漏研究站（sites_missing 為空），"
          "且 _SUCCESS 完成標記與 manifest 內全部 artifact 之 SHA-256 已核對。")
        doc.table("scale", "兩個產品之執行規模與完成度",
                  ["項目", "逐時產品 r1_hourly", "日產品 d1_daily25h"],
                  [["流場區域數／研究站數", "4／5", "4／5"],
                   ["domain-month 數（4 區域 × 24 個月）", "96（全數標準等級、ready）", "96（由逐時產品衍生）"],
                   ["有效快照數（每站）", num(a_h["n_valid_snapshots"]), num(a_d["n_valid_snapshots"])],
                   ["流場區域缺漏／研究站缺漏", f"{len(hm['domains_missing'])}／{len(hm['sites_missing'])}", f"{len(dm['domains_missing'])}／{len(dm['sites_missing'])}"],
                   ["site_statistics 列數（5 站 × 7 個分層）", "35", "35"],
                   ["成果圖數量（5 站 × 12 種）", f"{n_png[0]}", f"{n_png[1]}"],
                   ["時間置換 null 種子", "1、2", "1、2"],
                   ["狀態（state_label）", hm["state_label"], dm["state_label"]],
                   ["manifest schema／套件版本", f"{hm['schema_version']}／{hm['package_version']}", f"{dm['schema_version']}／{dm['package_version']}"],
                   ["程式版本 git_commit（git_dirty）", f"{hm['git_commit'][:7]}（{'是' if hm['git_dirty'] else '否'}）", f"{dm['git_commit'][:7]}（{'是' if dm['git_dirty'] else '否'}）"],
                   ["aggregate 完成時間（UTC）", hm["created_at_utc"], dm["created_at_utc"]]],
                  "本計畫兩次執行之 manifest 與 site_statistics；缺漏與省略數量為專案記錄。")
        H(3, "2、執行耗時")
        P(f"執行環境為 SERVER（Ubuntu 20.04、56 核、125 GB 記憶體，Python 3.12）。{doc.T('runtime')}列出各階段耗時；逐時偵測、強核心追蹤與日產品全流程之耗時為專案記錄，"
          "aggregate 耗時取自 manifest 之 runtime_seconds。")
        doc.table("runtime", "各階段執行耗時",
                  ["階段", "逐時產品", "日產品", "來源"],
                  [["偵測（detect-shards，96 個 domain-month）", "約 45 分鐘（32–40 個平行 worker，約 0.9–1.1 秒每快照每核）", "—", "專案記錄"],
                   ["強核心追蹤與 null 追蹤", "約 1.5 分鐘", "—", "專案記錄"],
                   ["aggregate（逐站統計、bootstrap、成果圖）", f"{m(hm['runtime_seconds'], 4)} 秒", f"{m(dm['runtime_seconds'], 4)} 秒", "manifest runtime_seconds"],
                   ["日產品全流程（build-daily-fields 至 aggregate）", "—", "約 10 分鐘", "專案記錄"]],
                  "本計畫執行紀錄與兩次執行之 manifest；專案記錄之耗時為概略值。")
        H(3, "3、各站強核心門檻與追蹤保留數")
        P(f"{doc.T('thr')}列出各站強核心門檻（站內合格核心 $s_1$ 之 10% 分位數）、合格核心數與追蹤保留數。各站之門檻為該站自己之相對定義，此表只是各站數值之並列，不構成站間排序。")
        rows = []
        tf_h, tf_d = hm["track_core_filter"]["thresholds"], dm["track_core_filter"]["thresholds"]
        for sid, nm, _ in SITES:
            rows.append([nm, m(tf_h[sid]["threshold_day"], 4), f"{num(tf_h[sid]['qualified_core_count'])}／{num(tf_h[sid]['kept_core_count'])}",
                         m(tf_d[sid]["threshold_day"], 4), f"{num(tf_d[sid]['qualified_core_count'])}／{num(tf_d[sid]['kept_core_count'])}"])
        doc.table("thr", "各站強核心門檻與追蹤保留數",
                  ["站點", "逐時強核心門檻（每日）", "逐時合格／保留核心數", "日產品強核心門檻（每日）", "日產品合格／保留核心數"],
                  rows, "本計畫兩次執行之 manifest（track_core_filter）；保留數約為合格核心數之 10%。")

        # ============================================================ （二）
        H(2, "（二）逐站成果")
        P("本節逐站呈現兩個產品之結果。每站依序說明頻率、強度、生命週期、近岸與近岸外、傳播速度與季節分層，並列出一張關鍵統計表、一張壽命分數對照表與一張季節分層表；"
          "每站以六張圖呈現：日產品之熱點與空間分布、逐時與日產品之壽命、日產品之長壽命 track 與代表性 TRAP。")
        crits = {hm["site_track_figures"][sid_]["example"]["criterion"] for sid_, _, _ in SITES} | {dm["site_track_figures"][sid_]["example"]["criterion"] for sid_, _, _ in SITES}
        assert crits == {"offshore_exceeds_null"}, crits
        crit_name = "offshore_exceeds_null"
        P("各圖之標示如下。空間分布圖（spatial）為站內 2×2 面板，依序為曲線通過頻率、核心頻率、平均核心 $s_1$（每日）與平均未截尾壽命，虛線為 10 公里近岸帶，色階上下限為該站分析區之 2–98 百分位。"
          "熱點圖（hotspots）標出曲線通過頻率最高 10% 之格點，並疊加核心頻率前 10% 之格點。壽命圖（lifetime）左為壽命互補累積分布（CCDF），實線為未截尾、虛線含截尾、灰色為時間置換 null 與跨種子範圍；"
          "右為壽命不小於 12 小時（日產品 72 小時）之未截尾 track 之平均核心 $s_1$ 對正規化年齡。長壽命路徑圖（long_tracks）為站內出生、未截尾、壽命不小於 24 小時（日產品 168 小時）之核心路徑，依壽命著色，近岸型虛線、近岸外實線。"
          "代表性 TRAP 圖（example）顯示代表 track 起始時刻之 $s_1$、全部合格曲線（淡色）、代表曲線與核心（紅）及核心路徑；"
          f"代表 track 之選取準則為 {crit_name}，即{CRIT[crit_name]}。圖面文字為繁體中文。")

        fig_no = {}
        for sid, nm, dom in SITES:
            H_ = S["hourly"][sid]["all"]
            D_ = S["daily"][sid]["all"]
            mh = hm["site_track_figures"][sid]
            md_ = dm["site_track_figures"][sid]
            exd = md_["example"]
            n_unc_h, n_unc_d = H_["track_n_uncensored"], D_["track_n_uncensored"]
            lh, ld = H_["long_track_n_born_uncensored"], D_["long_track_n_born_uncensored"]
            doc.h(3, f"{[s[0] for s in SITES].index(sid) + 1}、{nm}")
            share = ("（與另一研究站共用同一流場，以最近 anchor 互斥分割統計歸屬）" if dom == "A" else "（使用完整流場區域之分析海域）")
            P(f"{nm}位於流場區域 {dom}{share}，站點分析區面積 {m(H_['site_area_km2'], 5)} 平方公里。逐時產品有 {num(H_['n_valid_snapshots'])} 個有效快照，日產品有 {num(D_['n_valid_snapshots'])} 個，皆為全部標準快照。"
              f"{doc.F(sid + 'hot')}至{doc.F(sid + 'ex')}與{doc.T(sid + 'stat')}至{doc.T(sid + 'season')}呈現{nm}之結果。")
            # frequency
            P(f"在頻率方面，逐時產品之核心出現率與曲線出現率皆為 100%（{num(H_['n_valid_snapshots'])} 個有效快照皆至少有一個核心與一條曲線），不具區辨力；"
              f"強核心出現率為 {pct(H_['strong_core_occurrence_rate'])}（{num(H_['strong_core_occurrence_numerator'])}／{num(H_['n_valid_snapshots'])}，95% 信賴區間 {ci(H_['strong_core_occurrence_rate_ci95'])}），"
              f"曲線通過格點比例為 {pct(H_['mean_pass_cell_fraction'])}（95% 信賴區間 {ci(H_['mean_pass_cell_fraction_ci95'])}），即每個快照約有 {pct(H_['mean_pass_cell_fraction'], 3)} 之站內格點被 TRAP 曲線通過。"
              f"日產品之核心出現率與曲線出現率同為 100%；強核心出現率為 {pct(D_['strong_core_occurrence_rate'])}（{num(D_['strong_core_occurrence_numerator'])}／{num(D_['n_valid_snapshots'])}，95% 信賴區間 {ci(D_['strong_core_occurrence_rate_ci95'])}），"
              f"曲線通過格點比例為 {pct(D_['mean_pass_cell_fraction'])}（95% 信賴區間 {ci(D_['mean_pass_cell_fraction_ci95'])}）。"
              f"核心密度逐時為 {m(H_['core_density_per_1000km2_per_snapshot'], 4)}、日產品為 {m(D_['core_density_per_1000km2_per_snapshot'], 4)} 個每 1,000 平方公里每快照。")
            # intensity
            P(f"在強度方面，逐時產品之強核心門檻為 {m(H_['strong_core_threshold_day'], 4)} 每日（{sci(H_['strong_core_threshold_sinv'], 4)} 每秒）；站內合格核心共 {num(H_['core_s1_count'])} 個，核心 $s_1$ 之中位數與 p10 為 {m(H_['core_s1_median_day'], 4)} 與 {m(H_['core_s1_p10_day'], 4)} 每日。"
              f"熱點共 {num(H_['hotspot_cell_count'])} 格、{m(H_['hotspot_area_km2'], 5)} 平方公里，熱點門檻（曲線通過頻率）為 {m(H_['hotspot_threshold_curve_pass_frequency'], 4)}，熱點內平均通過頻率 {m(H_['hotspot_mean_curve_pass_frequency'], 4)}，"
              f"熱點內核心平均 $s_1$ 為 {m(H_['hotspot_mean_core_s1_day'], 4)} 每日，該站強核心有 {pct(H_['hotspot_strong_core_fraction'])} 落在熱點格點內。"
              f"日產品之強核心門檻為 {m(D_['strong_core_threshold_day'], 4)} 每日（{sci(D_['strong_core_threshold_sinv'], 4)} 每秒），核心 $s_1$ 之中位數與 p10 為 {m(D_['core_s1_median_day'], 4)} 與 {m(D_['core_s1_p10_day'], 4)} 每日；"
              f"熱點 {num(D_['hotspot_cell_count'])} 格、{m(D_['hotspot_area_km2'], 5)} 平方公里，門檻 {m(D_['hotspot_threshold_curve_pass_frequency'], 4)}，熱點內平均通過頻率 {m(D_['hotspot_mean_curve_pass_frequency'], 4)}，"
              f"熱點內核心平均 $s_1$ 為 {m(D_['hotspot_mean_core_s1_day'], 4)} 每日，強核心有 {pct(D_['hotspot_strong_core_fraction'])} 落在熱點格點內。"
              "日產品之強核心門檻與核心強度量值小於逐時產品，與 25 小時平均抑制潮汐與高頻應變一致；熱點為相對定義，面積約為站內面積之 10%。")
            # lifetime hourly
            def life_clause(rec, h, rk):
                key = f"track_frac_uncensored_ge_{h}h"
                nul = rec[f"null_{key}"]
                r_ = rec.get(f"ratio_{key}")
                s = (f"壽命不小於 {h} 小時者，真實為 {pct(rec[key])}、null 為 {pct(nul)}、超額 {pct(rec['excess_' + key])}")
                s += f"、比值 {ratio(r_)}" if r_ is not None else "（null 為 0，無比值）"
                return s

            c3 = round(H_["track_frac_uncensored_ge_12h"] * n_unc_h)
            c12 = c3
            P(f"在生命週期方面，逐時產品共追蹤 {num(H_['track_n_tracks'])} 條強核心 track，其中未截尾 {num(n_unc_h)} 條；壽命平均 {m(H_['track_mean_lifetime_hours'], 4)} 小時、中位數 {m(H_['track_median_lifetime_hours'], 4)} 小時、p90 為 {m(H_['track_p90_lifetime_hours'], 4)} 小時；"
              f"時間置換 null 之壽命中位數與 p90 為 {m(H_['null_median_lifetime_hours'], 4)} 與 {m(H_['null_p90_lifetime_hours'], 4)} 小時。"
              f"{life_clause(H_, 3, 'hourly')}；{life_clause(H_, 6, 'hourly')}；{life_clause(H_, 12, 'hourly')}（約 {num(c12)} 條）；"
              f"壽命不小於 24 小時者僅 {num(lh)} 條（{pct(H_['track_frac_uncensored_ge_24h'])}，null 為 {pct(H_['null_track_frac_uncensored_ge_24h'])}）"
              + ("，樣本數不足（不大於 20 條），不宜解讀。" if lh <= 20 else "，樣本數尚可，但仍遠少於較短壽命之樣本。"))
            P(f"日產品共追蹤 {num(D_['track_n_tracks'])} 條強核心 track，其中未截尾 {num(n_unc_d)} 條；壽命平均 {m(D_['track_mean_lifetime_hours'], 4)} 小時、中位數 {m(D_['track_median_lifetime_hours'], 4)} 小時（{m(D_['track_median_lifetime_hours'] / 24, 3)} 日）、p90 為 {m(D_['track_p90_lifetime_hours'], 4)} 小時（{m(D_['track_p90_lifetime_hours'] / 24, 3)} 日）；"
              f"null 之壽命中位數與 p90 為 {m(D_['null_median_lifetime_hours'], 4)} 與 {m(D_['null_p90_lifetime_hours'], 4)} 小時。"
              f"{life_clause(D_, 48, 'daily')}；{life_clause(D_, 72, 'daily')}；{life_clause(D_, 168, 'daily')}。"
              f"站內出生、未截尾、壽命不小於 168 小時（7 天）之長壽命 track 有 {num(ld)} 條"
              + ("，樣本數不足（不大於 20 條），不宜解讀。" if ld <= 20 else "。")
              + ("壽命 48 與 72 小時之比值較低，表示 2–3 天之壽命有相當部分可由機會串接解釋。" if (D_["ratio_track_frac_uncensored_ge_48h"] < 3 and D_["ratio_track_frac_uncensored_ge_72h"] < 3) else ""))
            # coastal
            def coast_life(rec, rk):
                hs = lifetimes(rk)
                parts = []
                for h in hs:
                    parts.append(f"不小於 {h} 小時：近岸外 {pct(rec[f'offshore_track_frac_uncensored_ge_{h}h'])}（null {pct(rec[f'null_offshore_track_frac_uncensored_ge_{h}h'])}）、"
                                 f"近岸型 {pct(rec[f'coastal_track_frac_uncensored_ge_{h}h'])}（null {pct(rec[f'null_coastal_track_frac_uncensored_ge_{h}h'])}）")
                return "；".join(parts)

            cl_h = H_["frac_long_tracks_within_buffer"]
            cl_d = D_["frac_long_tracks_within_buffer"]
            P(f"在近岸與近岸外方面（緩衝帶 10 公里），逐時產品之全部核心有 {pct(H_['frac_cores_within_buffer'])} 位於近岸帶，強核心則有 {pct(H_['frac_strong_cores_within_buffer'])}；"
              f"日產品分別為 {pct(D_['frac_cores_within_buffer'])} 與 {pct(D_['frac_strong_cores_within_buffer'])}。強核心之近岸比例高於全部核心之近岸比例，顯示強核心偏向近岸。"
              f"近岸外強核心出現率（門檻不變）逐時為 {pct(H_['offshore_strong_core_occurrence_rate'])}、日產品為 {pct(D_['offshore_strong_core_occurrence_rate'])}。"
              f"熱點面積有 {pct(H_['hotspot_frac_area_within_buffer'])}（逐時）與 {pct(D_['hotspot_frac_area_within_buffer'])}（日產品）位於近岸帶；只在近岸外格點重取分位數之熱點，逐時為 {num(H_['offshore_hotspot_cell_count'])} 格、平均通過頻率 {m(H_['offshore_hotspot_mean_curve_pass_frequency'], 4)}，"
              f"日產品為 {num(D_['offshore_hotspot_cell_count'])} 格、平均通過頻率 {m(D_['offshore_hotspot_mean_curve_pass_frequency'], 4)}。")
            P(f"長壽命 track 位於近岸帶之比例，逐時產品為 {pct(cl_h)}（長壽命 track {num(lh)} 條{'，樣本數不足，不宜解讀' if lh <= 20 else ''}），日產品為 {pct(cl_d)}（{num(ld)} 條）。"
              f"日產品中，未截尾 track 近岸外 {num(D_['offshore_track_n_uncensored'])} 條、近岸型 {num(D_['coastal_track_n_uncensored'])} 條；壽命分數（真實與 null）依序為："
              f"{coast_life(D_, 'daily')}。逐時產品中，近岸外 {num(H_['offshore_track_n_uncensored'])} 條、近岸型 {num(H_['coastal_track_n_uncensored'])} 條；{coast_life(H_, 'hourly')}。"
              + (f"逐時近岸外不小於 24 小時者只有 {num(round(H_['offshore_track_frac_uncensored_ge_24h'] * H_['offshore_track_n_uncensored']))} 條，不宜解讀。" if lh <= 20 or H_["offshore_track_frac_uncensored_ge_24h"] * H_["offshore_track_n_uncensored"] <= 20 else ""))
            # propagation
            P(f"在傳播速度方面（模型內 TRAP 核心之位移，不是水體或漂浮物速度），逐時產品已擬合 {num(H_['prop_n_tracks'])} 條 track（移動 {num(H_['prop_n_moving'])} 條、靜止 {num(H_['prop_n_stationary'])} 條，靜止比例 {pct(H_['prop_stationary_frac'])}）；"
              f"移動 track 之速度中位數 {m(H_['prop_speed_median_kmday'], 4)}、p90 為 {m(H_['prop_speed_p90_kmday'], 4)} 公里每日，東向與北向分量中位數為 {m(H_['prop_cx_median_kmday'], 3)} 與 {m(H_['prop_cy_median_kmday'], 3)} 公里每日，位置擬合殘差 RMS 中位數 {m(H_['prop_resid_rms_median_m'], 4)} 公尺。"
              f"日產品已擬合 {num(D_['prop_n_tracks'])} 條（移動 {num(D_['prop_n_moving'])} 條、靜止 {num(D_['prop_n_stationary'])} 條，靜止比例 {pct(D_['prop_stationary_frac'])}），速度中位數 {m(D_['prop_speed_median_kmday'], 4)}、p90 為 {m(D_['prop_speed_p90_kmday'], 4)} 公里每日，"
              f"東向與北向分量中位數 {m(D_['prop_cx_median_kmday'], 3)} 與 {m(D_['prop_cy_median_kmday'], 3)} 公里每日，殘差 RMS 中位數 {m(D_['prop_resid_rms_median_m'], 4)} 公尺。逐時（至少 7 個觀測）與日產品（至少 3 個觀測）之樣本不可直接對照。")
            # seasonal
            P(f"在季節分層方面，{seasonal_text(sid, 'hourly')}{seasonal_text(sid, 'daily')}季節分層僅涵蓋兩個年度，只描述分層數值，不能推論氣候季節性或年際變異。")

            # tables --------------------------------------------------------
            def r2(label, hv, dv):
                return [label, hv, dv]

            stat_rows = [
                r2("站點分析區面積（平方公里）", m(H_["site_area_km2"], 5), m(D_["site_area_km2"], 5)),
                r2("有效快照數", num(H_["n_valid_snapshots"]), num(D_["n_valid_snapshots"])),
                r2("強核心門檻（每日；每秒）", f"{m(H_['strong_core_threshold_day'], 4)}；{sci(H_['strong_core_threshold_sinv'], 4)}", f"{m(D_['strong_core_threshold_day'], 4)}；{sci(D_['strong_core_threshold_sinv'], 4)}"),
                r2("強核心出現率（分子／分母；95% 信賴區間）", f"{pct(H_['strong_core_occurrence_rate'])}（{num(H_['strong_core_occurrence_numerator'])}／{num(H_['n_valid_snapshots'])}）<br>{cip(H_['strong_core_occurrence_rate_ci95'])}",
                   f"{pct(D_['strong_core_occurrence_rate'])}（{num(D_['strong_core_occurrence_numerator'])}／{num(D_['n_valid_snapshots'])}）<br>{cip(D_['strong_core_occurrence_rate_ci95'])}"),
                r2("核心出現率／曲線出現率", "100%／100%（無區辨力）", "100%／100%（無區辨力）"),
                r2("曲線通過格點比例（95% 信賴區間）", f"{pct(H_['mean_pass_cell_fraction'])}<br>{cip(H_['mean_pass_cell_fraction_ci95'])}", f"{pct(D_['mean_pass_cell_fraction'])}<br>{cip(D_['mean_pass_cell_fraction_ci95'])}"),
                r2("核心密度（個每 1,000 平方公里每快照）", m(H_["core_density_per_1000km2_per_snapshot"], 4), m(D_["core_density_per_1000km2_per_snapshot"], 4)),
                r2("合格核心總數", num(H_["core_s1_count"]), num(D_["core_s1_count"])),
                r2("核心 $s_1$ 中位數／p10（每日）", f"{m(H_['core_s1_median_day'], 4)}／{m(H_['core_s1_p10_day'], 4)}", f"{m(D_['core_s1_median_day'], 4)}／{m(D_['core_s1_p10_day'], 4)}"),
                r2("熱點面積（平方公里）（格數）", f"{m(H_['hotspot_area_km2'], 5)}（{num(H_['hotspot_cell_count'])}）", f"{m(D_['hotspot_area_km2'], 5)}（{num(D_['hotspot_cell_count'])}）"),
                r2("熱點門檻（曲線通過頻率）／熱點內平均頻率", f"{m(H_['hotspot_threshold_curve_pass_frequency'], 4)}／{m(H_['hotspot_mean_curve_pass_frequency'], 4)}", f"{m(D_['hotspot_threshold_curve_pass_frequency'], 4)}／{m(D_['hotspot_mean_curve_pass_frequency'], 4)}"),
                r2("熱點內核心平均 $s_1$（每日）", m(H_["hotspot_mean_core_s1_day"], 4), m(D_["hotspot_mean_core_s1_day"], 4)),
                r2("熱點內強核心比例（占該站強核心）", pct(H_["hotspot_strong_core_fraction"]), pct(D_["hotspot_strong_core_fraction"])),
                r2("熱點面積位於 10 公里近岸帶比例", pct(H_["hotspot_frac_area_within_buffer"]), pct(D_["hotspot_frac_area_within_buffer"])),
                r2("近岸外熱點格數／平均通過頻率", f"{num(H_['offshore_hotspot_cell_count'])}／{m(H_['offshore_hotspot_mean_curve_pass_frequency'], 4)}", f"{num(D_['offshore_hotspot_cell_count'])}／{m(D_['offshore_hotspot_mean_curve_pass_frequency'], 4)}"),
                r2("全部核心位於 10 公里近岸帶比例", pct(H_["frac_cores_within_buffer"]), pct(D_["frac_cores_within_buffer"])),
                r2("強核心位於 10 公里近岸帶比例", pct(H_["frac_strong_cores_within_buffer"]), pct(D_["frac_strong_cores_within_buffer"])),
                r2("近岸外強核心出現率", pct(H_["offshore_strong_core_occurrence_rate"]), pct(D_["offshore_strong_core_occurrence_rate"])),
                r2("強核心 track 數（全部／未截尾）", f"{num(H_['track_n_tracks'])}／{num(n_unc_h)}", f"{num(D_['track_n_tracks'])}／{num(n_unc_d)}"),
                r2("壽命平均／中位數／p90（小時）", f"{m(H_['track_mean_lifetime_hours'], 4)}／{m(H_['track_median_lifetime_hours'], 4)}／{m(H_['track_p90_lifetime_hours'], 4)}", f"{m(D_['track_mean_lifetime_hours'], 4)}／{m(D_['track_median_lifetime_hours'], 4)}／{m(D_['track_p90_lifetime_hours'], 4)}"),
                r2("null 壽命中位數／p90（小時）", f"{m(H_['null_median_lifetime_hours'], 4)}／{m(H_['null_p90_lifetime_hours'], 4)}", f"{m(D_['null_median_lifetime_hours'], 4)}／{m(D_['null_p90_lifetime_hours'], 4)}"),
                r2("長壽命 track 數（未截尾、站內出生）；位於近岸帶比例", f"{num(lh)}；{pct(cl_h)}" + ("（樣本數不足，不宜解讀）" if lh <= 20 else ""), f"{num(ld)}；{pct(cl_d)}" + ("（樣本數不足，不宜解讀）" if ld <= 20 else "")),
                r2("傳播：已擬合 track 數（移動／靜止）", f"{num(H_['prop_n_tracks'])}（{num(H_['prop_n_moving'])}／{num(H_['prop_n_stationary'])}）", f"{num(D_['prop_n_tracks'])}（{num(D_['prop_n_moving'])}／{num(D_['prop_n_stationary'])}）"),
                r2("傳播：靜止比例", pct(H_["prop_stationary_frac"]), pct(D_["prop_stationary_frac"])),
                r2("傳播：速度中位數／p90（公里每日，僅移動 track）", f"{m(H_['prop_speed_median_kmday'], 4)}／{m(H_['prop_speed_p90_kmday'], 4)}", f"{m(D_['prop_speed_median_kmday'], 4)}／{m(D_['prop_speed_p90_kmday'], 4)}"),
                r2("傳播：東向／北向分量中位數（公里每日）", f"{m(H_['prop_cx_median_kmday'], 3)}／{m(H_['prop_cy_median_kmday'], 3)}", f"{m(D_['prop_cx_median_kmday'], 3)}／{m(D_['prop_cy_median_kmday'], 3)}"),
                r2("傳播：位置擬合殘差 RMS 中位數（公尺）", m(H_["prop_resid_rms_median_m"], 4), m(D_["prop_resid_rms_median_m"], 4)),
            ]
            doc.table(sid + "stat", f"{nm}關鍵統計", ["指標", "逐時 r1_hourly", "日產品 d1_daily25h"], stat_rows,
                      f"本計畫 site_statistics（stratum 為 all）；{nm}站點分析區；熱點與強核心皆為本站相對定義；方括號為日區塊 bootstrap 之 95% 信賴區間。")

            life_rows = []
            for rk, rec, n_all in (("hourly", H_, n_unc_h), ("daily", D_, n_unc_d)):
                for h in lifetimes(rk):
                    k = f"track_frac_uncensored_ge_{h}h"
                    ok, ck = f"offshore_track_frac_uncensored_ge_{h}h", f"coastal_track_frac_uncensored_ge_{h}h"
                    lab = ("逐時" if rk == "hourly" else "日產品") + f" ≥ {h} 小時" + ("（樣本不足）" if (h in (24, 168) and (lh if rk == 'hourly' else ld) <= 20) else "")
                    r_ = rec.get("ratio_" + k)
                    life_rows.append([lab, "全部", pct(rec[k]), pct(rec["null_" + k]), pct(rec["excess_" + k]), ratio(r_) if r_ is not None else "—"])
                    life_rows.append(["", "近岸外", pct(rec[ok]), pct(rec["null_" + ok]), pct(rec["excess_" + ok]), "—"])
                    life_rows.append(["", "近岸型", pct(rec[ck]), pct(rec["null_" + ck]), pct(rec["excess_" + ck]), "—"])
            doc.table(sid + "life", f"{nm}壽命分數：真實與時間置換 null",
                      ["產品與門檻", "類別", "真實", "null", "超額", "比值"], life_rows,
                      f"本計畫 site_statistics；壽命分數為未截尾 track 中壽命不小於門檻者之比例，null 為兩個種子平均；隨機對照為 0 時比值不定義，以「—」表示；近岸外與近岸型各列之比值未計算（以「—」表示）；「樣本不足」表示該門檻之長壽命 track 不大於 20 條，樣本數不足，不宜解讀；"
                      f"未截尾 track 數：逐時真實 {num(n_unc_h)}（近岸外 {num(H_['offshore_track_n_uncensored'])}、近岸型 {num(H_['coastal_track_n_uncensored'])}），null 平均 {num(H_['null_n_uncensored'])}；"
                      f"日產品真實 {num(n_unc_d)}（近岸外 {num(D_['offshore_track_n_uncensored'])}、近岸型 {num(D_['coastal_track_n_uncensored'])}），null 平均 {num(D_['null_n_uncensored'])}。")

            seas_rows = []
            for s in SEASONS:
                for rk, lab in (("hourly", "逐時產品"), ("daily", "日產品")):
                    a = S[rk][sid][s]
                    seas_rows.append([s if rk == "hourly" else "", lab, num(a["n_valid_snapshots"]), pct(a["strong_core_occurrence_rate"]),
                                      f"{pct(a['mean_pass_cell_fraction'])}<br>{cip(a['mean_pass_cell_fraction_ci95'])}"])
            doc.table(sid + "season", f"{nm}季節分層",
                      ["季節", "產品", "快照數", "強核心出現率", "曲線通過格點比例（95% 信賴區間）"],
                      seas_rows, "本計畫 site_statistics（stratum 為 DJF、MAM、JJA、SON）；季節分層僅涵蓋兩個年度。")

            # figures -------------------------------------------------------
            f = lambda kind, rk: figmap[(sid, rk, kind)]
            doc.fig(sid + "hot", f"日產品{nm}：曲線通過頻率前 10% 熱點（{num(D_['hotspot_cell_count'])} 格，{m(D_['hotspot_area_km2'], 5)} 平方公里，門檻 {m(D_['hotspot_threshold_curve_pass_frequency'], 3)}）", f("hotspots", "daily"), "6.3in")
            doc.fig(sid + "spa", f"日產品{nm}：曲線通過頻率、核心頻率、平均核心 $s_1$（每日）與平均未截尾壽命（2×2；有效快照 n = {num(D_['n_valid_snapshots'])}；虛線為 10 公里近岸帶）", f("spatial", "daily"), "6.3in")
            doc.fig(sid + "lh", f"逐時{nm}：強核心結構壽命 CCDF（實線為未截尾，n = {num(n_unc_h)}；灰色為時間置換 null）與壽命不小於 12 小時 track 之平均 $s_1$ 隨正規化年齡之演化", f("lifetime", "hourly"), "6.3in")
            doc.fig(sid + "ld", f"日產品{nm}：壽命 CCDF（未截尾 n = {num(n_unc_d)}）與壽命不小於 72 小時 track 之平均 $s_1$ 隨正規化年齡之演化", f("lifetime", "daily"), "6.3in")
            doc.fig(sid + "lt", f"日產品{nm}：壽命不小於 168 小時（7 天）之核心路徑，共 {num(md_['n_long_tracks'])} 條（近岸型 {num(md_['n_long_tracks_coastal'])}、近岸外 {num(md_['n_long_tracks_offshore'])}）；null 各種子平均 {m(md_['null_long_track_mean'], 4)} 條"
                    f"（近岸型 {m(md_['null_long_track_coastal_mean'], 4)}、近岸外 {m(md_['null_long_track_offshore_mean'], 4)}）", f("long_tracks", "daily"), "6.3in")
            doc.fig(sid + "ex", f"日產品{nm}：代表性 TRAP（track {exd['track_id']}，起始 {exd['birth_time_utc']} UTC，壽命 {m(exd['lifetime_hours'], 4)} 小時，{exd['n_observations']} 個觀測，核心平均距岸 {m(exd['mean_coast_distance_m'] / 1000, 3)} 公里；選取準則 {exd['criterion']}；null 之 p99 壽命 {m(exd['null_p99_lifetime_hours'], 4)} 小時）", f("example", "daily"), "6.3in")

        # ============================================================ （三）
        H(2, "（三）綜合說明")
        H(3, "1、各站摘要")
        P("以下逐站摘要各站獨立之結果，各站之門檻與熱點皆為該站自己之相對定義，不作站間排序。")
        for sid, nm, dom in SITES:
            H_, D_ = S["hourly"][sid]["all"], S["daily"][sid]["all"]
            lh, ld = H_["long_track_n_born_uncensored"], D_["long_track_n_born_uncensored"]
            P(f"{nm}：逐時與日產品之強核心出現率為 {pct(H_['strong_core_occurrence_rate'])} 與 {pct(D_['strong_core_occurrence_rate'])}，曲線通過格點比例為 {pct(H_['mean_pass_cell_fraction'])} 與 {pct(D_['mean_pass_cell_fraction'])}；"
              f"熱點面積為 {m(H_['hotspot_area_km2'], 5)} 與 {m(D_['hotspot_area_km2'], 5)} 平方公里；日產品壽命不小於 168 小時者真實為 {pct(D_['track_frac_uncensored_ge_168h'])}、null 為 {pct(D_['null_track_frac_uncensored_ge_168h'])}（比值 {ratio(D_['ratio_track_frac_uncensored_ge_168h'])}）；"
              f"強核心位於近岸帶之比例為 {pct(H_['frac_strong_cores_within_buffer'])}（逐時）與 {pct(D_['frac_strong_cores_within_buffer'])}（日產品）；"
              f"日產品長壽命 track 位於近岸帶之比例為 {pct(D_['frac_long_tracks_within_buffer'])}（{num(ld)} 條）；逐時長壽命 track 僅 {num(lh)} 條" + ("，樣本數不足，不宜解讀。" if lh <= 20 else "。"))
        H(3, "2、逐時與日產品之時間尺度意義")
        p90s = sorted({S["hourly"][s]["all"]["track_p90_lifetime_hours"] for s, _, _ in SITES})
        p90d = sorted({S["daily"][s]["all"]["track_p90_lifetime_hours"] for s, _, _ in SITES})
        medd = sorted({S["daily"][s]["all"]["track_median_lifetime_hours"] for s, _, _ in SITES})
        P(f"逐時產品保留潮汐等次日變動，TRAP 結構之壽命以小時計：五站之壽命中位數皆為 1 小時，p90 為 {'、'.join(m(v) for v in p90s)} 小時。在核心 2 公里、曲線 4 公里之嚴格追蹤門檻下，"
          "壽命不小於 3 小時之比例仍高於時間置換 null，顯示逐時結構確有時間連續性；但超過 12 小時者極少，不小於 24 小時者更少。逐時結果含潮汐，其短壽命與潮汐週期之關係尚未用潮相資料分層檢驗（本計畫無潮相資料）。")
        P(f"日產品經 25 小時平均抑制主要分潮後，結構壽命上升到天之量級：壽命中位數為 {'、'.join(m(v) for v in medd)} 小時，p90 為 {'、'.join(m(v) for v in p90d)} 小時。"
          "這與 Kunz 等（2024）以逐日資料分析之時間尺度相近，但其研究域為北太平洋中尺度，門檻與壽命不可直接移植到本計畫約 1 公里之近岸網格。"
          f"日產品壽命不小於 7 天者高於 null（比值 {m(min(S['daily'][s_]['all']['ratio_track_frac_uncensored_ge_168h'] for s_, _, _ in SITES), 3)}–{m(max(S['daily'][s_]['all']['ratio_track_frac_uncensored_ge_168h'] for s_, _, _ in SITES), 3)} 倍），但不小於 48 與 72 小時者之比值只有 {m(min(S['daily'][s_]['all'][f'ratio_track_frac_uncensored_ge_{h_}h'] for s_, _, _ in SITES for h_ in (48, 72)), 3)}–{m(max(S['daily'][s_]['all'][f'ratio_track_frac_uncensored_ge_{h_}h'] for s_, _, _ in SITES for h_ in (48, 72)), 3)} 倍，說明 2–3 天之壽命有相當部分可由機會串接解釋，只有更長之壽命才與 null 明顯分離；"
          "因此解讀日產品結果時，建議以不小於 7 天之長壽命 track 與近岸外指標為主。")
        H(3, "3、近岸鎖定")
        near_ok = [nm for sid, nm, _ in SITES if S["daily"][sid]["all"]["frac_long_tracks_within_buffer"] > 0.7]
        near_no = [(nm, S["daily"][sid]["all"]["frac_long_tracks_within_buffer"]) for sid, nm, _ in SITES if S["daily"][sid]["all"]["frac_long_tracks_within_buffer"] <= 0.7]
        strong_gt = all(S[rk][sid]["all"]["frac_strong_cores_within_buffer"] > S[rk][sid]["all"]["frac_cores_within_buffer"] for rk in ("hourly", "daily") for sid, _, _ in SITES)
        P("核心在距非海域格點約 2–3 公里內無法偵測（檢出下限）。在可偵測範圍內，" + ("五站在兩個產品中，強核心位於近岸帶之比例皆高於該站全部核心之近岸比例（各站數值見關鍵統計表）；" if strong_gt else "強核心之近岸比例在部分站不高於全部核心；")
          + f"日產品長壽命 track 位於近岸帶之比例，{'、'.join(near_ok)}超過 70%" + ("，" + "、".join(f"{n}為 {pct(v)}" for n, v in near_no) + "，未超過 70%" if near_no else "") + "。"
          f"日產品之靜止比例（擬合位移小於 1 個格距者）於各站見關鍵統計表，許多長壽命 track 為位移小於 1 格之靜止型。"
          "這與「地形（岬角、島嶼）鎖定之應變場或邊界效應」之解讀一致：此類結構不是漂移之收斂帶，其與實際海廢聚集之關係須獨立觀測才能確認。保守之結論應以近岸外（offshore）指標為準。")
        P("逐時產品中，長壽命 track 之樣本數：" + "、".join(f"{nm} {num(S['hourly'][sid]['all']['long_track_n_born_uncensored'])} 條" for sid, nm, _ in SITES)
          + "；其中不大於 20 條者樣本數不足，不宜解讀。因此「長壽命者多在近岸」是日產品多數站之現象，不是逐時產品或每一站之通則。")
        H(3, "4、對時間置換 null 之穩健性")
        d_ok = all(S["daily"][sid]["all"][f"excess_{k}_track_frac_uncensored_ge_{h}h"] > 0 for sid, _, _ in SITES for k in ("offshore", "coastal") for h in (48, 72, 168))
        h_ok = all(S["hourly"][sid]["all"][f"excess_offshore_track_frac_uncensored_ge_{h}h"] > 0 for sid, _, _ in SITES for h in (3, 6, 12))
        P(("日產品：每一站之近岸外與近岸型壽命不小於 48、72、168 小時之比例皆高於其 null（超額皆為正，見各站壽命分數表）。" if d_ok else "日產品：部分站之近岸外或近岸型壽命比例未高於 null，詳見各站壽命分數表。")
          + ("逐時：近岸外壽命不小於 3、6、12 小時之比例在各站皆高於 null；" if h_ok else "逐時：部分站之近岸外壽命比例未高於 null；")
          + "不小於 24 小時者僅有少數 track，樣本極少，不宜解讀。追蹤全部核心時壽命大多可由機會解釋；本結果採「只追蹤各站前 10% 強核心加兩個種子之時間置換 null」才有超額。null 只有 2 個種子，且只檢驗「時間連續性」，不檢驗結構之物理意義。")

        # ============================================================ （四）
        H(2, "（四）限制與使用說明")
        P("一、僅模型資料（MODEL_ONLY）。沒有獨立之漂流器或海廢觀測，因此未進行 Serra 等（2020）圖 4a–c 與表 1–2 型式之驗證，也未與海廢記錄對照。結果只能稱為「表層短期吸引潛勢」，不得寫成海廢濃度、長期滯留、海底廢聚集或污染來源。SVD 與 TRAP 共用 OCM 強迫，其關聯屬機制探索，不是獨立驗證。")
        P("二、壽命之機會成分。日產品壽命 2–3 天有相當部分可由機會解釋；逐時不小於 24 小時之樣本極少。壽命一律指「強核心結構（各站前 10%）」，不是全部 TRAP；null 僅 2 個種子。")
        P("三、檢出下限與近岸。核心無法位於距非海域格點約 2–3 公里內；近岸強核心與長壽命 track 多為靜止型，可能是地形鎖定之應變場或邊界效應。近岸與近岸外之歸類依 track 核心平均距岸判定，對 10 公里緩衝寬度敏感。")
        P("四、相對定義。強核心與熱點皆為各站自己之 10% 分位數，不可作跨站大小比較；熱點面積依定義約占站內面積之 10%。核心與曲線出現率皆為 100%，無區辨力。")
        P("五、參數與解析度。相對強度 0.30、曲線長度上限、追蹤門檻等未做完整敏感度分析；1 公里網格之近岸結果受網格與遮罩影響。日產品 25 小時平均不是理想低通。")
        P("六、傳播速度是結構位移，不是水體或漂浮物速度；逐時與日產品之最少觀測數不同（7 與 3）。")
        P(f"七、距岸距離場上限。兩次最終結果之 manifest 記錄距離場於 {num(hm['coast_distance']['cap_m'] / 1000)} 公里飽和，而目前程式預設上限為 200 公里；10 公里近岸分割不受影響，但距岸直方圖之末格為「不小於 30 公里」。")
        P("八、相干性比值之單位。本章之相干性（真實除以 null）一律採最終 site_statistics 之逐站值；方法章門檻掃描階段之記錄值以流場區域為單位計算，與逐站值單位不同，不可逐項對照。")
        P("九、季節分層只涵蓋兩個年度，不可推論氣候季節性。")
        P("十、使用說明。引用本章數值時，請同時註明「僅模型資料（MODEL_ONLY）、無獨立觀測驗證」，以及程式版本與兩次執行之 param_id（r1_hourly、d1_daily25h）。"
          "完整結果檔（site_statistics.csv／.json、manifest.json、逐站空間陣列與全部 60 × 2 張成果圖）見兩次執行之 aggregate_report 目錄。")

    build.prefixes = ("表乙", "圖乙")
    return build
