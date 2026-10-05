"""Content of 三、瞬態吸引剖面（TRAPs）實作方法 (LBT dialect Markdown)."""
from __future__ import annotations

import math

import numpy as np

from word_md_common import (INDENT, MINUS, SITES, Doc, ci, g, num, pct, ratio, sci, tex_sci, two_pass)


def m(x, n=4):
    return g(x, n)


def sc(x, n=4):
    """TeX scientific notation (use inside $...$)."""
    return tex_sci(x, n)


def scu(x, n=4):
    """Unicode scientific notation (plain text)."""
    return sci(x, n)


def build_method(stats, man, figs):
    hm, dm = man["hourly"], man["daily"]
    det = hm["detection_config"]

    # ------------------------------------------------------------------ worked examples (script-computed)
    dx = 2000.0
    uE, uW, uN, uS = 0.12, 0.20, 0.18, 0.14
    vE, vW, vN, vS = 0.05, 0.01, 0.00, 0.04
    dudx, dudy = (uE - uW) / dx, (uN - uS) / dx
    dvdx, dvdy = (vE - vW) / dx, (vN - vS) / dx
    a, d = dudx, dvdy
    b = 0.5 * (dvdx + dudy)
    rad = math.hypot((a - d) / 2, b)
    s1 = (a + d) / 2 - rad
    s2 = (a + d) / 2 + rad
    gap = s2 - s1
    vec = np.array([b, s2 - a])
    e2 = vec / np.linalg.norm(vec)
    e1 = np.array([-e2[1], e2[0]])
    Se2 = np.array([[a, b], [b, d]]) @ e2
    # step 3 quadratic fit
    PATCH = np.array([[-5.1, -6.3, -5.8], [-6.6, -8.0, -7.1], [-5.5, -6.9, -6.0]])
    A_rows, yv = [], []
    for eta, row in zip((1, 0, -1), PATCH):
        for xi, val in zip((-1, 0, 1), row):
            A_rows.append([1, xi, eta, xi * xi, xi * eta, eta * eta])
            yv.append(val)
    c = np.linalg.lstsq(np.array(A_rows), np.array(yv), rcond=None)[0]
    Hm = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    gv = c[1:3]
    pstar = np.linalg.solve(Hm, -gv)
    detH = float(np.linalg.det(Hm))
    # step 4 bilinear + RK2
    th0, thm = math.radians(30.0), math.radians(36.0)
    k1 = np.array([math.cos(th0), math.sin(th0)])
    h_step = 500.0
    x0 = np.array([12000.0, 20000.0])
    xm = x0 + 0.5 * h_step * k1
    k2 = np.array([math.cos(thm), math.sin(thm)])
    x1 = x0 + h_step * k2
    corners_e2 = np.array([[0.80, 0.60], [0.79, 0.61], [-0.78, -0.62], [0.81, 0.59]])
    ref = corners_e2[0]
    aligned = np.array([cn if cn @ ref >= 0 else -cn for cn in corners_e2])
    wts = np.array([0.4, 0.3, 0.2, 0.1])
    mixed = (wts[:, None] * aligned).sum(axis=0)
    mixed_n = mixed / np.linalg.norm(mixed)
    naive = (wts[:, None] * corners_e2).sum(axis=0)
    s1_core_demo = -20.0
    s1_seq = [-20.0, -17.0, -12.0, -7.0, -5.8]
    rel_thr = 0.30 * abs(s1_core_demo)
    # step 6 filter
    def Hresp(T, N=25):
        return abs(math.sin(math.pi * N / T) / (N * math.sin(math.pi / T)))

    tides = [("M2", 12.4206), ("S2", 12.0), ("N2", 12.6583), ("K1", 23.9345), ("O1", 25.8193), ("P1", 24.0659),
             ("3 日週期", 72.0), ("7 日週期", 168.0)]
    tt = np.arange(-12, 13)
    sig = 0.05 + 0.10 * np.cos(2 * np.pi * tt / 12.4206)
    mean25 = float(sig.mean())
    resid = mean25 - 0.05
    # step 7 Hausdorff toy
    GA = np.array([[0, 0], [500, 0], [1000, 100], [1500, 300]], float)
    GB = np.array([[200, 300], [700, 350], [1200, 500], [1700, 700]], float)
    dAB = max(min(np.linalg.norm(p - q) for q in GB) for p in GA)
    dBA = max(min(np.linalg.norm(q - p) for p in GA) for q in GB)
    dH_toy = max(dAB, dBA)
    dc_toy = 1200.0
    Dc, Dh = 2000.0, 4000.0
    cost_toy = dc_toy / Dc + dH_toy / Dh
    pairs = [("A1–B1", 600.0, 1500.0), ("A1–B2", 1700.0, 3600.0), ("A2–B1", 900.0, 2800.0), ("A2–B2", 700.0, 1800.0),
             ("A2–B3", 2400.0, 1000.0)]
    pair_rows = []
    for nm, dcv, dhv in pairs:
        if dcv <= Dc and dhv <= Dh:
            pair_rows.append((nm, dcv, dhv, dcv / Dc + dhv / Dh))
        else:
            pair_rows.append((nm, dcv, dhv, None))
    # step 10 toy
    s1_vals = np.array([-0.4, -0.6, -0.9, -1.1, -1.4, -1.8, -2.2, -2.9, -3.5, -4.1, -4.8, -5.6, -6.7, -7.9, -9.4,
                        -11.0, -13.2, -15.8, -19.6, -24.0])
    p10 = float(np.quantile(s1_vals, 0.10))
    strong_n = int((s1_vals <= p10).sum())
    Fvals = np.array([0.02, 0.05, 0.07, 0.09, 0.11, 0.12, 0.14, 0.16, 0.18, 0.20, 0.21, 0.23, 0.25, 0.27, 0.30,
                      0.33, 0.36, 0.40, 0.40, 0.52])
    thrF = float(np.quantile(Fvals, 0.9))
    hot = Fvals >= thrF
    # step 12 propagation toy
    th = np.arange(7.0)
    px = np.array([0, 420, 880, 1250, 1700, 2110, 2540.0])
    py = np.array([0, -150, -290, -460, -590, -760, -900.0])
    tb = th.mean()
    cx = float(((th - tb) * (px - px.mean())).sum() / ((th - tb) ** 2).sum())
    cy = float(((th - tb) * (py - py.mean())).sum() / ((th - tb) ** 2).sum())
    resx = px - (px.mean() + cx * (th - tb))
    resy = py - (py.mean() + cy * (th - tb))
    rms = float(np.sqrt(np.mean(resx ** 2 + resy ** 2)))
    dur = float(th[-1] - th[0])
    disp = math.hypot(cx, cy) * dur
    speed_kmday = math.hypot(cx, cy) * 24 / 1000
    # strong-core thresholds (actual)
    thr_h = hm["strong_core_thresholds"]
    thr_d = dm["strong_core_thresholds"]
    tcf_h = hm["track_core_filter"]["thresholds"]
    tcf_d = dm["track_core_filter"]["thresholds"]

    G = stats["hourly"]["gongliao"]["all"]
    Gd = stats["daily"]["gongliao"]["all"]
    ex_h = hm["site_track_figures"]["gongliao"]["example"]
    ex_hc = hm["site_track_figures"]["gongliao"]["example_coastal"]
    ex_d = dm["site_track_figures"]["gongliao"]["example"]

    def build(doc: Doc):
        P, H = doc.p, doc.h
        doc.h(1, "三、瞬態吸引剖面（TRAPs）實作方法")
        doc.note("本章為工項 2「瞬態吸引剖面（TRAPs）」之實作方法，併入全文時置於工項 1 之後；式號暫自 (1) 起編，"
                 "表號暫定「表甲-」，圖號暫定「圖甲-」，併入時請重編；各項數值取自本計畫兩次正式執行之 manifest 與 "
                 "site_statistics，標明「示範數值」者為說明用。")
        P("本章說明瞬態吸引剖面（Transient Attracting Profiles, TRAPs，以下簡稱 TRAP）之完整計算程序，依計算順序分為十二個步驟。"
          "每個步驟依「目的、輸入、計算步驟、輸出、算例」五項說明：先列出控制方程式與各符號之定義與單位，再逐步代入數字計算。"
          "讀者可依序照做，得到與本計畫相同之結果。")
        P("本計畫之 TRAP 分析以表層二維流速為唯一輸入，不使用波浪、風與水深資料。TRAP 屬歐拉（Eulerian）幾何結構，"
          "描述某一瞬間流場中「壓縮最強」之位置與其骨架曲線，並非粒子、水團或廢棄物之軌跡；所有結果只能解讀為模型內之"
          "「表層短期吸引潛勢」，不是海廢濃度、滯留時間、海底廢聚集或污染來源。")
        P("流場以兩種產品獨立分析：逐時產品（param_id 為 r1_hourly）保留潮汐；日產品（param_id 為 d1_daily25h）為 25 小時"
          "置中平均之去潮汐流場。兩個產品共用步驟 1 至 5 與步驟 8 至 12 之全部公式與檢出參數，差別只在步驟 6 之流場來源、"
          "步驟 7 之追蹤門檻與取樣間隔。每個研究站獨立統計，不做跨站比較。")
        P("本章用語如下。「結構軌跡」指同一 TRAP 結構在連續時刻被追蹤所形成的序列；「隨機時間置換對照」（簡稱「隨機對照」）指保留追蹤設定、僅把各可用時間槽之檢出隨機置換後重新追蹤之結果，用以估計「沒有真實時間連續性」時之壽命基準；"
          "「互補累積分布」（complementary cumulative distribution function, CCDF）指壽命不小於某值之結構軌跡比例隨該值之變化；「均方根」（root mean square, RMS）為殘差平方之平均再開方；"
          "「逐日區塊重抽樣」（block bootstrap）指以日為單位重抽樣以估計信賴區間。成果表中，每日（day⁻¹）為應變率之單位，其後僅以 day⁻¹ 表示。")
        P("算例分兩類。標明「實際數字」者，取自本計畫之設定、輸入資料或計算結果；標明「示範數值」者，為說明計算流程而自行設定之數字，"
          "數值落在合理範圍，但不代表任何一個實際站點或時刻。所有算例之算術均以程式逐項計算後填入，數字以四捨五入方式取到所列位數，"
          "連續運算以未四捨五入之中間值進行。")
        doc.table("overview", "TRAP 計算步驟總覽",
                  ["步驟", "目的", "主要式號"],
                  [["1、速度梯度與應變率張量", "由表層流速求應變率矩陣", f"{doc.E('gradu')}–{doc.E('conv')}"],
                   ["2、特徵值與特徵向量", "求最強壓縮率與拉伸方向", f"{doc.E('eig')}–{doc.E('e2')}"],
                   ["3、TRAP 核心定位與次網格修正", "找出壓縮最強之局部極小點", f"{doc.E('corecond')}–{doc.E('stat')}"],
                   ["4、TRAP 曲線積分與停止條件", "沿拉伸方向積分出骨架曲線", f"{doc.E('ode')}–{doc.E('stopturn')}"],
                   ["5、品質門檻", "決定快照、核心、曲線與強核心是否採用", f"{doc.E('cover')}–{doc.E('strong')}"],
                   ["6、逐時／日平均流場產品", "建立去潮汐之 25 小時日產品", f"{doc.E('daily')}–{doc.E('Hresp')}"],
                   ["7、時序追蹤", "把相鄰時刻之 TRAP 配成同一條 track", f"{doc.E('dc')}–{doc.E('cost')}"],
                   ["8、隨機時間置換對照", "估計機會水準之壽命", f"{doc.E('excess')}"],
                   ["9、壽命與截尾", "計算壽命並處理截尾", f"{doc.E('life')}–{doc.E('frac')}"],
                   ["10、統計指標", "出現率、通過比例、熱點與信賴區間", f"{doc.E('rate')}–{doc.E('boot')}"],
                   ["11、近岸診斷", "區分近岸型與近岸外", f"{doc.E('coast')}"],
                   ["12、傳播速度", "估計 TRAP 核心位移速度", f"{doc.E('fit')}–{doc.E('speed')}"]],
                  "本研究整理。")

        # ---------------------------------------------------------------- （一）
        H(2, "（一）資料與前處理")
        H(3, "1、流場資料與研究站")
        P("輸入為中央氣象署海洋模式（CWA-OCM，採 SCHISM 非結構化網格）之表層流速，東向分量檔名為 u_surface_mps、北向分量為 "
          "v_surface_mps，單位為每秒公尺。上游已把模式輸出重採樣為 1 公里規則公尺網格（座標 x_m、y_m；schema 3.0.0）。"
          "本計畫涵蓋四個流場區域與五個研究站，期間為 2024 年 1 月至 2025 年 12 月，共 4 個區域 × 24 個月 = 96 個 domain-month，"
          "依專案記錄全數為標準等級且狀態為 ready。")
        doc.table("domains", "流場區域與研究站",
                  ["流場區域", "flow_domain_id", "研究站"],
                  [["A", "northeast_taiwan_common_cache_v3", "貢寮、龜山島西側"],
                   ["B", "hsinchu_cache_v3", "新竹外海"],
                   ["C", "houwan_nmmba_cache_v3（顯示名稱：南灣）", "南灣"],
                   ["D", "lienchiang_common_cache_v3", "連江"]],
                  "本計畫研究設計（configs/study_design.json）。")
        P("A 區之貢寮與龜山島西側共用同一流場，以兩個 anchor（站點中心點）之公尺距離做互斥最近 anchor 分割，距離相等者歸貢寮；"
          "分割只決定統計歸屬，不裁切流場微分或曲線積分。B、C、D 三區各只含一站，使用完整流場區域之分析海域。")
        H(3, "2、正式時間軸與缺口")
        n_rows, n_dup, n_uni, n_miss = 17196, 72, 17124, 420
        n_days = 731
        P(f"逐時正式時間軸串接各月份後共 {num(n_rows)} 列，月份交界有 {n_dup} 個重複之 UTC 時刻；以 sort_and_deduplicate_prefer_last "
          f"（依 UTC 穩定排序，重複者保留較晚月份）處理後為 {num(n_uni)} 個唯一時刻。檢核：2024 年（366 日）與 2025 年（365 日）共 "
          f"{n_days} 日，應有 {n_days} × 24 = {num(n_days * 24)} 個整點，減去唯一時刻 {num(n_uni)} 個，缺漏 "
          f"{num(n_days * 24 - n_uni)} 個小時，與專案記錄之缺漏小時數 {n_miss} 相符。缺漏中有 48 個缺口長於 1 小時，最長 50 小時。"
          "缺口一律不內插、不補值，只記錄；追蹤與統計以缺口截尾處理（見步驟 7 與步驟 9）。")
        H(3, "3、海岸遮罩")
        P("分析海域遮罩採 coastline_corrected_v2 版本（以開放街圖精確海岸線製作，與專案 GeoJSON 同一 SHA-256），"
          "由 scripts/export_ocean_masks.py 逐位元匯出為 analysis_ocean_mask_coastline_corrected_v2.npy 並附 metadata sidecar。"
          "mask_static.npy 只代表網格支撐範圍，不等於陸地遮罩；正式檢出程式拒絕缺少 sidecar 之輸入。"
          "專案內之 GeoJSON（© OpenStreetMap contributors，ODbL）只供圖面疊加海岸線，不影響數值。")
        H(3, "4、資料落地與唯讀原則")
        P("資料由 NAS 複製到 SERVER 本機 SSD（11 GB，SHA-256 逐檔核對），沿用 Lagrangian 逆向追蹤專案「本機優先、NAS 只當來源」之慣例。"
          "程式以記憶體映射唯讀讀取來源資料，從不修改來源；schema、來源與逐時遮罩之檢查採 fail-closed（不符即停止，不自行補救）。")

        # ---------------------------------------------------------------- （二）
        H(2, "（二）數值計算")
        # step 1
        H(3, "1、速度梯度與應變率張量")
        H(4, "（1）目的")
        P("由表層二維流速求每個格點之速度梯度，組成應變率張量（應變率矩陣）。應變率描述流體在該點被拉長或壓縮之快慢，"
          "是 TRAP 之判定依據。")
        H(4, "（2）輸入")
        P("某一時刻（逐時產品為一個整點，日產品為一個 UTC 日）之表層東向流速 $u$ 與北向流速 $v$（每秒公尺，下稱每秒 m），"
          "定義在 1 公里規則網格；格點之公尺座標 $x_j$（東向）與 $y_i$（北向），其中 $i$ 為北向列序、$j$ 為東向行序；"
          "分析海域遮罩。")
        H(4, "（3）計算步驟")
        P("第一步，以中心差分求流速對位置之偏導數：")
        doc.eq("gradu", r"\frac{\partial u}{\partial x}\approx\frac{u_{i,j+1}-u_{i,j-1}}{x_{j+1}-x_{j-1}},\qquad "
                        r"\frac{\partial u}{\partial y}\approx\frac{u_{i+1,j}-u_{i-1,j}}{y_{i+1}-y_{i-1}}")
        doc.eq("gradv", r"\frac{\partial v}{\partial x}\approx\frac{v_{i,j+1}-v_{i,j-1}}{x_{j+1}-x_{j-1}},\qquad "
                        r"\frac{\partial v}{\partial y}\approx\frac{v_{i+1,j}-v_{i-1,j}}{y_{i+1}-y_{i-1}}")
        P("其中：$u_{i,j}$、$v_{i,j}$ 為第 $i$ 列第 $j$ 行格點之東向與北向流速（每秒 m）；$x_j$、$y_i$ 為格點之東向與北向公尺座標（公尺），"
          "分母為該格點兩側座標之差，約 2,000 公尺（兩個格距）；$\\partial u/\\partial x$ 等四個偏導數之單位為每秒（s⁻¹）。")
        P("第二步，由四個偏導數組成應變率張量，即速度梯度之對稱部分：")
        doc.eq("S", r"\mathbf{S}=\tfrac12\left(\nabla\mathbf{v}+\nabla\mathbf{v}^{\mathsf T}\right)="
                    r"\begin{pmatrix}\dfrac{\partial u}{\partial x} & \tfrac12\left(\dfrac{\partial v}{\partial x}+\dfrac{\partial u}{\partial y}\right)\\[2mm]"
                    r"\tfrac12\left(\dfrac{\partial v}{\partial x}+\dfrac{\partial u}{\partial y}\right) & \dfrac{\partial v}{\partial y}\end{pmatrix}"
                    r"=\begin{pmatrix}a & b\\ b & d\end{pmatrix}")
        P("其中：$\\mathbf{S}$ 為應變率張量（每秒，s⁻¹），對稱矩陣；$\\nabla\\mathbf{v}$ 為速度梯度張量，$\\mathbf{v}=(u,v)$；"
          "$a=\\partial u/\\partial x$ 為東西向之伸縮率；$d=\\partial v/\\partial y$ 為南北向之伸縮率；"
          "$b$ 為剪切應變率，其值為 $\\partial v/\\partial x$ 與 $\\partial u/\\partial y$ 之平均。式(%s)即工項需求之式(5)。" % doc.E("S").strip("()"))
        P("第三步，單位換算。計算以每秒為單位，成果表以每日（day⁻¹）表示，換算為乘以 86,400：")
        doc.eq("conv", r"s_{\mathrm{day}}=86400\,s_{\mathrm{s}}")
        P("其中：$s_{\\mathrm{s}}$ 為以每秒（s⁻¹）表示之應變率；$s_{\\mathrm{day}}$ 為以每日（day⁻¹）表示之應變率；86,400 為一日之秒數。")
        P("遮罩安全規則：中心格點與其東、西、南、北四個鄰點，只要有一格無效（陸地、缺值或非有限值），該格點就不產生導數，"
          "不把海岸或缺值當成 0。這是保守之 fail-closed 設計，也避免曲線穿過遮罩缺口。本計畫不做空間平滑"
          "（smoothing_radius_cells 為 0；程式保留有遮罩保護之圖擴散選項，本次未啟用）。")
        H(4, "（4）輸出")
        P("每個格點之 $a$、$b$、$d$（每秒）；導數無效之格點記為無效。流場本身與應變率場不落地，只儲存步驟 3 與步驟 4 之核心與曲線。")
        H(4, "（5）算例")
        P(f"以下為示範數值：某格點（相鄰格距 1,000 公尺，兩側座標差 {m(dx, 4)} 公尺）之四個鄰點流速（每秒 m）如{doc.T('ex1')}。")
        doc.table("ex1", "應變率張量算例之四個鄰點流速（示範數值，單位：每秒 m）",
                  ["鄰點", "東向流速 $u$", "北向流速 $v$"],
                  [["東（$j+1$）", m(uE), m(vE)], ["西（$j-1$）", m(uW), m(vW)],
                   ["北（$i+1$）", m(uN), m(vN)], ["南（$i-1$）", m(uS), m(vS)]],
                  "本研究示範計算。")
        P(f"依式{doc.E('gradu')}與式{doc.E('gradv')}：$\\partial u/\\partial x=({m(uE)}-{m(uW)})/{m(dx,4)}={sc(dudx)}$ 每秒；"
          f"$\\partial u/\\partial y=({m(uN)}-{m(uS)})/{m(dx,4)}={sc(dudy)}$ 每秒；"
          f"$\\partial v/\\partial x=({m(vE)}-{m(vW)})/{m(dx,4)}={sc(dvdx)}$ 每秒；"
          f"$\\partial v/\\partial y=({m(vN)}-{m(vS)})/{m(dx,4)}={sc(dvdy)}$ 每秒。")
        P(f"依式{doc.E('S')}：$a={sc(a)}$ 每秒，$d={sc(d)}$ 每秒，$b=\\tfrac12({sc(dvdx)}+{sc(dudy)})={sc(b)}$ 每秒。"
          f"換算為每日：$a={m(a * 86400)}$、$b={m(b * 86400)}$、$d={m(d * 86400)}$ 每日。"
          "若四個鄰點中有任何一個位於陸地，該格點不計算導數，亦不參與步驟 2 以後之任何判定。")

        # step 2
        H(3, "2、特徵值與特徵向量")
        H(4, "（1）目的")
        P("求應變率張量之兩個特徵值（最強壓縮率與最強拉伸率）及其方向。最小特徵值為負，代表該點在某一方向上流體被壓縮。")
        H(4, "（2）輸入")
        P("步驟 1 之 $a$、$b$、$d$（每秒）。")
        H(4, "（3）計算步驟")
        P("對稱 $2\\times2$ 矩陣之特徵方程 $\\det(\\mathbf{S}-\\lambda\\mathbf{I})=0$（工項需求之特徵方程）有封閉解：")
        doc.eq("eig", r"s_{1,2}=\frac{a+d}{2}\mp\sqrt{\left(\frac{a-d}{2}\right)^{2}+b^{2}},\qquad s_1\le s_2")
        P("其中：$s_1$ 為最小特徵值，即最強壓縮率（工項需求之 $\\lambda_{\\min}$，每秒）；$s_2$ 為最大特徵值，即最強拉伸率（每秒）；"
          "$a$、$b$、$d$ 見式%s。" % doc.E("S"))
        P("兩個特徵值之間距為 $s_2-s_1=2\\sqrt{((a-d)/2)^2+b^2}$。間距低於 $10^{-12}$ 每秒之格點視為方向退化（兩個方向無法區分），不參與檢出：")
        doc.eq("gap", r"s_2-s_1<\varepsilon_{\mathrm{gap}}=10^{-12}\ \Rightarrow\ \text{方向退化，不檢出}")
        P("其中：$\\varepsilon_{\\mathrm{gap}}$ 為方向退化門檻（每秒），對應 min_eigenvalue_gap_sinv。")
        P("拉伸方向為 $s_2$ 對應之單位特徵向量：")
        doc.eq("e2", r"\mathbf{e}_2=\frac{(b,\ s_2-a)}{\left\|(b,\ s_2-a)\right\|},\qquad "
                     r"\text{若 }\left\|(b,\ s_2-a)\right\|\text{ 過小，改用 }\frac{(s_2-d,\ b)}{\left\|(s_2-d,\ b)\right\|}")
        P("其中：$\\mathbf{e}_2$ 為拉伸方向之單位向量（無因次，東向與北向兩個分量）；兩種寫法取範數較大者，以穩定取得軸向特徵向量。"
          "壓縮方向 $\\mathbf{e}_1$（對應 $s_1$）與 $\\mathbf{e}_2$ 處處正交。工項需求寫「沿代表拉伸方向之特徵向量 $\\mathbf{e}_{\\max}$ 積分」，"
          "與本計畫沿 $\\mathbf{e}_2$ 積分一致。特徵向量只定義到正負號，後續步驟 4 以符號連續性處理。")
        H(4, "（4）輸出")
        P("每個有效格點之 $s_1$、$s_2$（每秒）與 $\\mathbf{e}_2$；方向退化之格點標為不可用。")
        H(4, "（5）算例")
        P(f"以下為示範數值，沿用步驟 1 之張量 $a={sc(a)}$、$b={sc(b)}$、$d={sc(d)}$ 每秒。$(a+d)/2={sc((a + d) / 2)}$；"
          f"$(a-d)/2={sc((a - d) / 2)}$；$\\sqrt{{(a-d)^2/4+b^2}}={sc(rad)}$。依式{doc.E('eig')}：$s_1={sc(s1)}$ 每秒（{m(s1 * 86400)} 每日），"
          f"$s_2={sc(s2)}$ 每秒（{m(s2 * 86400)} 每日）；$s_1<0$，該點為壓縮，且間距 $s_2-s_1={sc(gap)}$ 每秒遠大於 $10^{{-12}}$，不退化。")
        P(f"拉伸方向：$(b,\\ s_2-a)=({sc(b)},\\ {sc(s2 - a)})$，範數為 {scu(float(np.linalg.norm(vec)))}，"
          f"故 $\\mathbf{{e}}_2=({m(e2[0])},\\ {m(e2[1])})$。壓縮方向 $\\mathbf{{e}}_1=({m(e1[0])},\\ {m(e1[1])})$。"
          f"驗算：$\\mathbf{{S}}\\mathbf{{e}}_2=({sc(Se2[0])},\\ {sc(Se2[1])})$，除以 $\\mathbf{{e}}_2$ 之對應分量得 {scu(Se2[0] / e2[0])} 與 "
          f"{scu(Se2[1] / e2[1])}，皆等於 $s_2$；內積 $\\mathbf{{e}}_1\\cdot\\mathbf{{e}}_2={m(0.0 if abs(float(e1 @ e2)) < 1e-12 else float(e1 @ e2), 3)}$，兩向量正交。")

        # step 3
        H(3, "3、TRAP 核心定位與次網格修正")
        H(4, "（1）目的")
        P("在 $s_1$ 場中找出「為負且為局部極小值」之格點作為 TRAP 核心，再以二次曲面擬合把核心位置修正到格點之間（次網格），"
          "避免核心恆落在 1 公里格點上而使位移被格距量化。")
        H(4, "（2）輸入")
        P("步驟 2 之 $s_1$ 場；核心最小量值 $s_{\\min}=-10^{-8}$ 每秒（min_core_s1_sinv）；局部極小值鄰域半徑 1 格"
          "（local_min_radius_cells，即 $3\\times3$ 鄰域）；並列判定容許 $10^{-12}$ 每秒。")
        H(4, "（3）計算步驟")
        P("第一步，核心候選。格點 $(i,j)$ 須同時滿足：")
        doc.eq("corecond", r"s_1(i,j)<s_{\min}=-10^{-8}\ \mathrm{s^{-1}},\qquad s_1(i,j)\le s_1(i',j')\ \ \forall\,(i',j')\in\mathcal{N}_{3\times3}")
        P("其中：$s_1(i,j)$ 為該格點之最強壓縮率（每秒）；$\\mathcal{N}_{3\\times3}$ 為以該格點為中心之 $3\\times3$ 鄰域，"
          "鄰域九格之導數必須全部有效；並至少有一個鄰點嚴格較高，避免整片常數區被當作核心。")
        P("第二步，次網格定位。對核心格點之 $3\\times3$ 有效鄰域（9 個樣本）以最小平方法擬合二次曲面：")
        doc.eq("quad", r"s_1(\xi,\eta)\approx c_0+c_1\xi+c_2\eta+c_3\xi^{2}+c_4\xi\eta+c_5\eta^{2}")
        P("其中：$\\xi$、$\\eta$ 為相對核心格點之東向與北向偏移，以格距無因次化（取 $-1$、0、$+1$）；$c_0$ 至 $c_5$ 為六個待定係數"
          "（9 個樣本、6 個參數，故為超定問題，取最小平方解）。駐點（曲面之極值點）由下式求得：")
        doc.eq("stat", r"\mathbf{H}\,\mathbf{p}=-\mathbf{g},\qquad "
                       r"\mathbf{H}=\begin{pmatrix}2c_3 & c_4\\ c_4 & 2c_5\end{pmatrix},\qquad \mathbf{g}=\begin{pmatrix}c_1\\ c_2\end{pmatrix}")
        P("其中：$\\mathbf{p}=(\\hat{\\xi},\\hat{\\eta})$ 為駐點相對核心格點之偏移（格距）；$\\mathbf{H}$ 為曲面之海森矩陣；$\\mathbf{g}$ 為梯度。"
          "只有 $\\mathbf{H}$ 正定（$2c_3>0$ 且 $\\det\\mathbf{H}>0$，確保駐點為極小值）且偏移每個分量之絕對值不超過 0.5 格時才接受；"
          "否則保留格點位置並標記 refined 為否。曲線積分自細化後之位置起算；核心強度 $s_1$ 仍取格點值。")
        H(4, "（4）輸出")
        P("每個核心之格點位置、次網格位置（公尺）、強度 $s_1$（每秒）與是否細化。核心不再另做核心間距去重（core_min_separation_m 為 0）。")
        H(4, "（5）算例")
        P(f"以下為示範數值：某核心格點與其 $3\\times3$ 鄰域之 $s_1$（每日）如{doc.T('ex3')}，中央格點 {m(PATCH[1, 1])} 每日為九格中最小，且小於核心門檻。")
        doc.table("ex3", "核心 $3\\times3$ 鄰域之 $s_1$（示範數值，單位：每日）",
                  ["", "$\\xi=-1$（西）", "$\\xi=0$", "$\\xi=+1$（東）"],
                  [["$\\eta=+1$（北）"] + [m(v, 3) for v in PATCH[0]],
                   ["$\\eta=0$"] + [m(v, 3) for v in PATCH[1]],
                   ["$\\eta=-1$（南）"] + [m(v, 3) for v in PATCH[2]]],
                  "本研究示範計算。")
        P(f"最小平方擬合得 $c_0={m(c[0], 4)}$、$c_1={m(c[1], 4)}$、$c_2={m(c[2], 4)}$、$c_3={m(c[3], 4)}$、$c_4={m(c[4], 4)}$、$c_5={m(c[5], 4)}$。"
          f"海森矩陣 $2c_3={m(2 * c[3], 4)}$、$c_4={m(c[4], 4)}$、$2c_5={m(2 * c[5], 4)}$，$\\det\\mathbf{{H}}={m(detH, 4)}>0$ 且 $2c_3>0$，為正定。"
          f"依式{doc.E('stat')}解得 $(\\xi^{{*}},\\eta^{{*}})=({m(pstar[0], 3)},\\ {m(pstar[1], 3)})$，兩分量絕對值皆小於 0.5，接受。"
          f"若格距為 1,000 公尺，核心由格點向東移 {m(pstar[0] * 1000, 3)} 公尺、向北移 {m(pstar[1] * 1000, 3)} 公尺。")

        # step 4
        H(3, "4、TRAP 曲線積分與停止條件")
        H(4, "（1）目的")
        P("自每個核心沿拉伸方向 $\\mathbf{e}_2$ 向兩側各積分一條分支，得到與最強壓縮方向處處正交之積分曲線，即 TRAP 骨架。")
        H(4, "（2）輸入")
        P("步驟 3 之核心位置 $\\mathbf{x}_0$；步驟 2 之 $s_1$ 與 $\\mathbf{e}_2$ 場；積分參數（見" + doc.T('params') + "）：步長取半個格距（約 500 公尺）、"
          "每分支最長 20 公里、最多 2,000 步、相鄰切向量最大轉角 90°、相對強度下限 0.30、曲線至少 3 個點。")
        H(4, "（3）計算步驟")
        P("第一步，積分方程。自核心向 $\\pm\\mathbf{e}_2$ 兩個方向各積一條分支：")
        doc.eq("ode", r"\frac{d\mathbf{x}}{d\sigma}=\pm\,\mathbf{e}_2\big(\mathbf{x}(\sigma)\big),\qquad \mathbf{x}(0)=\mathbf{x}_0")
        P("其中：$\\sigma$ 為沿曲線之弧長（公尺）；$\\mathbf{x}(\\sigma)$ 為曲線上之位置（公尺）；$\\mathbf{x}_0$ 為核心之次網格位置。"
          "曲線輸出時固定為「反向端 → 核心 → 正向端」。")
        P("第二步，場值之雙線性內插。目標位置所在格之四個角點必須全部導數有效，否則停止（invalid_mask）。因特徵向量之正負號任意，"
          "先以第一個角點為參考，把與其內積為負之角點向量反號，再加權平均並正規化：")
        doc.eq("bilin", r"\mathbf{e}_2(\mathbf{x})=\frac{\sum_{k=1}^{4}w_k\,\tilde{\mathbf{e}}_{k}}{\left\|\sum_{k=1}^{4}w_k\,\tilde{\mathbf{e}}_{k}\right\|},\qquad "
                        r"\tilde{\mathbf{e}}_{k}=\begin{cases}\mathbf{e}_{2,k}, & \mathbf{e}_{2,k}\cdot\mathbf{e}_{2,1}\ge 0\\ -\mathbf{e}_{2,k}, & \mathbf{e}_{2,k}\cdot\mathbf{e}_{2,1}<0\end{cases}")
        P("其中：$w_k$ 為第 $k$ 個角點之雙線性權重（無因次，四者相加為 1）；$\\mathbf{e}_{2,k}$ 為第 $k$ 個角點之拉伸方向；"
          "$\\tilde{\\mathbf{e}}_k$ 為反號校正後之方向。$s_1$ 以同一組權重直接內插。")
        P("第三步，二階中點法（RK2）推進一步：")
        doc.eq("rk2", r"\mathbf{x}_{n+1}=\mathbf{x}_n+h\,\mathbf{e}_2\!\left(\mathbf{x}_n+\tfrac{h}{2}\,\mathbf{e}_2(\mathbf{x}_n)\right)")
        P("其中：$\\mathbf{x}_n$ 為第 $n$ 步位置（公尺）；$h$ 為步長（公尺），取半個格距約 500 公尺，即使使用者指定較大步長也切成不超過半格之子步，"
          "避免跨過一列無效格點。沿曲線前進時，再把新切向量反號到與前一步切向量夾角不大於 90°：")
        doc.eq("orient", r"\boldsymbol{\tau}_{n+1}\leftarrow\operatorname{sgn}\!\left(\boldsymbol{\tau}_{n+1}\cdot\boldsymbol{\tau}_{n}\right)\boldsymbol{\tau}_{n+1}")
        P("其中：$\\boldsymbol{\\tau}_n$ 為第 $n$ 步之單位切向量；$\\operatorname{sgn}$ 為正負號函數。如此 $\\mathbf{e}_2$ 之任意正負號不影響曲線。")
        P("第四步，停止準則。任一條件成立即停止該分支，並把原因記錄於 STOP_REASON_CODES：")
        doc.eq("stoprel", r"\left|s_1(\mathbf{x}_n)\right|<f_{\mathrm{rel}}\,\left|s_{1,\mathrm{core}}\right|,\qquad f_{\mathrm{rel}}=0.30")
        doc.eq("stoplen", r"\ell_n=\sum h\ge 20{,}000\ \text{公尺},\qquad n\ge 2000")
        doc.eq("stopturn", r"\arccos\left(\boldsymbol{\tau}_{n+1}\cdot\boldsymbol{\tau}_{n}\right)>90^{\circ}")
        P("其中：$s_{1,\\mathrm{core}}$ 為該曲線核心之強度（每秒）；$f_{\\mathrm{rel}}$ 為相對強度下限（curve_min_relative_strength）；$\\ell_n$ 為該分支累計弧長（公尺）。"
          "完整之停止原因另有：invalid_mask（四個角點含無效導數）、outside_flow_domain（超出網格）、s1_threshold（$s_1\\ge0$，不再壓縮）、"
          "s1_not_monotonic（$s_1$ 較前一點更負，自核心向外應單調回升）、invalid_step 與 zero_step（步長無效或位移為零）。"
          "曲線合格條件為至少 3 個點；每條曲線同時記錄弧長、平均 $s_1$ 與兩端停止原因。"
          "曲線經過之格點集合，以半格距加密後取最近格點（curve_pass_cells），供步驟 10 統計。")
        H(4, "（4）輸出")
        P("每個核心之合格曲線點列（公尺）、弧長、平均 $s_1$、兩端停止原因與曲線通過之格點集合。")
        H(4, "（5）算例")
        P(f"以下為示範數值：四個角點之拉伸方向為 $(0.80,\\ 0.60)$、$(0.79,\\ 0.61)$、$({MINUS}0.78,\\ {MINUS}0.62)$、$(0.81,\\ 0.59)$，"
          f"權重為 0.4、0.3、0.2、0.1。第三個角點與第一個角點內積為 {m(float(corners_e2[2] @ ref), 3)}（為負），先反號為 $(0.78,\\ 0.62)$；"
          f"依式{doc.E('bilin')}加權得 $({m(mixed[0], 4)},\\ {m(mixed[1], 4)})$，正規化後 $\\mathbf{{e}}_2=({m(mixed_n[0], 4)},\\ {m(mixed_n[1], 4)})$。"
          f"若不做反號校正，直接加權為 $({m(naive[0], 4)},\\ {m(naive[1], 4)})$，方向明顯被壓短，這是反號校正之必要性。")
        P(f"推進一步：核心位置 $\\mathbf{{x}}_0=(12{{,}}000,\\ 20{{,}}000)$ 公尺，步長 $h={m(h_step, 3)}$ 公尺，起點之 $\\mathbf{{e}}_2=({m(k1[0], 4)},\\ {m(k1[1], 4)})$"
          f"（方向角 30°）。中點位置 $\\mathbf{{x}}_0+\\tfrac{{h}}{{2}}\\mathbf{{e}}_2=({m(xm[0], 6)},\\ {m(xm[1], 6)})$ 公尺，中點處內插得 $\\mathbf{{e}}_2=({m(k2[0], 4)},\\ {m(k2[1], 4)})$（方向角 36°）。"
          f"依式{doc.E('rk2')}，$\\mathbf{{x}}_1=({m(x1[0], 6)},\\ {m(x1[1], 6)})$ 公尺。")
        P(f"停止條件之判定：若核心 $s_1$ 為 {m(s1_core_demo)} 每日，相對強度下限為 $0.30\\times|{m(s1_core_demo)}|={m(rel_thr)}$ 每日。沿曲線各點之 $s_1$ 依序為 "
          f"{'、'.join(m(v) for v in s1_seq)} 每日；前四點之 $|s_1|$（20、17、12、7）皆不小於 {m(rel_thr)}，繼續積分；第五點 $|{m(s1_seq[-1])}|={m(abs(s1_seq[-1]))}<{m(rel_thr)}$，"
          "以 relative_strength_threshold 停止，該分支共 5 個點，其餘分支同理。若某分支之曲線合計少於 3 個點，整條曲線不合格，不進入統計與追蹤。")

        # step 5
        H(3, "5、品質門檻")
        H(4, "（1）目的")
        P("決定哪些快照、核心、曲線與強核心可進入統計與追蹤，使結果不受缺值、海岸遮罩缺口與弱訊號污染。")
        H(4, "（2）輸入")
        P("各站分析區內之有效格點數；快照之有效格點比例；全部合格核心之 $s_1$；追蹤分位數 0.10（strong_quantile）。")
        H(4, "（3）計算步驟")
        P("第一步，快照覆蓋率。快照 $k$ 在站內之有效格點比例須不小於 0.95，否則該快照不計入分母，也不內插補值：")
        doc.eq("cover", r"\mathrm{cov}_k=\frac{M_{\mathrm{valid},k}}{M}\ge 0.95")
        P("其中：$M_{\\mathrm{valid},k}$ 為快照 $k$ 在站內導數有效之格點數；$M$ 為站內格點總數；0.95 為 min_coverage。"
          "通過者為有效快照，其數量記為 $N$；缺口時刻不在分母內，不以 0 取代。")
        P("第二步，合格核心與合格曲線。核心須滿足式%s，曲線須至少 3 個點；只有合格曲線進入追蹤與通過格點統計。" % doc.E("corecond"))
        P("第三步，強核心。各站取該站、標準時間軸且通過覆蓋率之快照中，全部合格核心 $s_1$ 之 10% 分位數為強核心門檻，"
          "$s_1$ 不大於門檻者為強核心（$s_1$ 越負越強）：")
        doc.eq("strong", r"\tau_{\mathrm{site}}=Q_{0.10}\big(\{s_1\}_{\mathrm{site}}\big),\qquad \text{強核心：}\ s_1\le\tau_{\mathrm{site}}")
        P("其中：$\\tau_{\\mathrm{site}}$ 為該站之強核心門檻（每秒或每日）；$Q_{0.10}$ 為 10% 分位數；$\\{s_1\\}_{\\mathrm{site}}$ 為該站全部合格核心之強度集合。"
          "此門檻為各站自己之相對定義，與追蹤用門檻及統計用門檻位元相同，數值大小不作跨站比較。")
        H(4, "（4）輸出")
        P("有效快照集合、合格核心與曲線，以及各站強核心門檻與追蹤保留之核心數。")
        H(4, "（5）算例")
        P(f"實際數字（強核心門檻）：各站之強核心門檻、合格核心數與保留數如{doc.T('thr')}（取自兩次執行之 manifest）。"
          f"例如貢寮逐時產品之合格核心 {num(tcf_h['gongliao']['qualified_core_count'])} 個，取 10% 分位數 {scu(thr_h['gongliao']['strong_core_threshold_sinv'])} 每秒 "
          f"（乘以 86,400 為 {m(thr_h['gongliao']['strong_core_threshold_day'], 4)} 每日），保留 {num(tcf_h['gongliao']['kept_core_count'])} 個，"
          f"保留比例 {pct(tcf_h['gongliao']['kept_core_count'] / tcf_h['gongliao']['qualified_core_count'])}。")
        rows = []
        for sid, nm, _ in SITES:
            rows.append([nm,
                         m(thr_h[sid]['strong_core_threshold_day'], 4), scu(thr_h[sid]['strong_core_threshold_sinv']),
                         f"{num(tcf_h[sid]['qualified_core_count'])}／{num(tcf_h[sid]['kept_core_count'])}",
                         m(thr_d[sid]['strong_core_threshold_day'], 4), scu(thr_d[sid]['strong_core_threshold_sinv']),
                         f"{num(tcf_d[sid]['qualified_core_count'])}／{num(tcf_d[sid]['kept_core_count'])}"])
        doc.table("thr", "各站強核心門檻與保留數",
                  ["站點", "逐時門檻（每日）", "逐時門檻（每秒）", "逐時合格／保留核心數", "日產品門檻（每日）", "日產品門檻（每秒）", "日產品合格／保留核心數"],
                  rows, "本計畫兩次執行之 manifest（strong_core_thresholds 與 track_core_filter）；單位為每秒（括號內為每日）；各站相對定義，不作站間比較。")
        P(f"以下為示範數值（覆蓋率）：站內有 6,852 個格點，某快照導數有效者 6,700 個，覆蓋率 6,700／6,852 = {m(6700 / 6852, 4)}，不小於 0.95，採用；"
          f"另一快照有效者僅 6,500 個，覆蓋率 {m(6500 / 6852, 4)}，小於 0.95，不計入分母。實際數字：兩次執行之各站有效快照皆為全部標準快照，"
          f"逐時 {num(G['n_valid_snapshots'])} 個、日產品 {num(Gd['n_valid_snapshots'])} 個。")
        P(f"以下為示範數值（10% 分位數）：20 個合格核心之 $s_1$（每日）由弱至強為 {'、'.join(m(v) for v in s1_vals)}；"
          f"10% 分位數（線性內插）為 {m(p10)} 每日；不大於此值者有 {strong_n} 個，即最強之 {strong_n} 個核心為強核心。")

        # step 6
        H(3, "6、逐時／日平均流場產品")
        H(4, "（1）目的")
        P("逐時表層流以潮汐為主，而 Kunz 等（2024）使用逐日快照。為提供去潮汐之第二個分析時間尺度，由逐時資料建立同 schema 之日產品"
          "（cache_kind 為 derived_daily_mean），兩個產品獨立分析。")
        H(4, "（2）輸入")
        P("正式逐時時間軸（步驟 2 之一部分）上之表層流速 $u$、$v$（每秒 m）；每個 UTC 日之目標時刻 $T_d$（當日 12:00 UTC）。")
        H(4, "（3）計算步驟")
        P("第一步，以 25 小時置中視窗取平均。視窗為正式逐時時間軸上 $|t-T_d|\\le12$ 小時之樣本（即當日 00:00 至隔日 00:00，含兩端，可跨月讀取鄰月）：")
        doc.eq("daily", r"\bar{\mathbf{u}}(\mathbf{x},T_d)=\frac{1}{N_d}\sum_{t_k\in W_d}\mathbf{u}(\mathbf{x},t_k),\qquad W_d=\{t_k:\ |t_k-T_d|\le 12\ \text{小時}\}")
        P("其中：$\\bar{\\mathbf{u}}$ 為日平均流速（每秒 m）；$N_d$ 為視窗內實際存在之小時數；$t_k$ 為逐時時刻；$W_d$ 為視窗內之時刻集合。規則如下：")
        P("一、$N_d\\ge23$，否則該日整日省略，不補值；專案記錄中每個流場區域保留 715 日、另 16 日省略（715 + 16 = 731 日）。"
          "二、格點僅在視窗內所有實際存在之小時皆有效時才有效（嚴格規則，使微分不混用不同樣本集合），無效處為 NaN。"
          "三、grid 目錄（含海岸遮罩 sidecar）原樣複製，來源逐時資料從不修改；續跑時只在濾波參數、來源與陣列 SHA-256 皆相符時略過。")
        P("第二步，潮汐抑制之理由。25 點（$N=25$、取樣間隔 $\\Delta t=1$ 小時）等權滑動平均，對週期 $T$ 之正弦訊號，振幅響應（保留比例）之解析式為：")
        doc.eq("Hresp", r"H(T)=\left|\frac{\sin\left(\pi N\,\Delta t/T\right)}{N\,\sin\left(\pi\,\Delta t/T\right)}\right|")
        P("其中：$H(T)$ 為週期 $T$ 之訊號經濾波後保留之振幅比例（無因次，1 為完全保留、0 為完全消除）；$T$ 為訊號週期（小時）；"
          "$N$ 為視窗點數；$\\Delta t$ 為取樣間隔（小時）。25 小時平均仍不是理想低通：響應在分潮頻率附近不為零，殘留之日潮與非潮汐高頻訊號可能不為零，"
          "相關資訊寫入各月 metadata.json 之 derived_from.filter。")
        H(4, "（4）輸出")
        P("每個流場區域每月之日產品快取（同 schema，1 公里網格），每日一個時刻，與逐時產品共用步驟 1 至 5 之檢出程序。")
        H(4, "（5）算例")
        rows = [[nm, m(T, 6), m(Hresp(T), 3)] for nm, T in tides]
        P(f"實際數字（濾波響應）：六個主要分潮與 3 日、7 日週期之振幅響應如{doc.T('resp')}，為濾波器之解析響應，非對模型資料之實測，以程式直接計算；"
          f"{doc.F('resp')}為響應曲線。六個主要分潮之振幅被壓到約 0.6%–4.3%，3 日與 7 日週期仍保留約 81% 與 96%，所以日產品保留天尺度變動、抑制潮汐。")
        doc.table("resp", "25 小時等權平均之振幅響應（實際數字，解析計算）",
                  ["訊號", "週期 $T$（小時）", "振幅響應 $H(T)$"], rows,
                  "本研究以式%s計算（$N=25$，$\\Delta t=1$ 小時）。" % doc.E("Hresp"))
        doc.fig("resp", "25 小時等權平均之振幅響應曲線（紅點為表甲所列訊號）", figs["resp"], "6.0in")
        P(f"以下為示範數值（直接平均）：訊號 $u(t)=0.05+0.10\\cos(2\\pi t/12.4206)$ 每秒 m（平均流 0.05 加 M2 潮流振幅 0.10），在視窗 $t=-12,\\dots,+12$ 小時取 25 點平均得 {m(mean25, 4)} 每秒 m，"
          f"殘留之潮汐成分為 {m(mean25, 4)} − 0.05 = {m(resid, 3)} 每秒 m，約為原潮流振幅之 {m(abs(resid) / 0.10 * 100, 3)}%，與表甲中 M2 之 $H$ 為 {m(Hresp(12.4206), 3)} 一致。")

        # step 7
        H(3, "7、時序追蹤")
        H(4, "（1）目的")
        P("把相鄰時刻之 TRAP（核心加曲線）配成同一條 track，以得到結構之生命週期。追蹤對象為歐拉幾何結構，不是粒子、水團或廢棄物之軌跡。")
        H(4, "（2）輸入")
        P("各時刻之合格 TRAP（在追蹤強核心時，只含各站強核心，見步驟 5）；追蹤門檻：逐時產品核心距離 $D_c=2{,}000$ 公尺、曲線距離 $D_h=4{,}000$ 公尺、"
          "名目步長 3,600 秒、最大時間差 10,800 秒（3 小時）；日產品 $D_c=5{,}000$ 公尺、$D_h=10{,}000$ 公尺、名目步長 86,400 秒、最大時間差 129,600 秒（36 小時）。")
        H(4, "（3）計算步驟")
        P("第一步，對相鄰可用快照之候選配對 $(A,B)$，須同時通過兩個距離門檻：")
        doc.eq("dc", r"d_c=\left\|\mathbf{x}^{A}_{\mathrm{core}}-\mathbf{x}^{B}_{\mathrm{core}}\right\|\le D_c")
        doc.eq("dh", r"d_H=\max\left\{\max_{p\in\Gamma_A}\operatorname{dist}(p,\Gamma_B),\ \max_{q\in\Gamma_B}\operatorname{dist}(q,\Gamma_A)\right\}\le D_h")
        P("其中：$d_c$ 為兩核心之距離（公尺）；$\\mathbf{x}^{A}_{\\mathrm{core}}$、$\\mathbf{x}^{B}_{\\mathrm{core}}$ 為兩核心位置（公尺）；$d_H$ 為兩條曲線 $\\Gamma_A$、$\\Gamma_B$"
          "（以 500 公尺步長加密）之對稱 Hausdorff 距離（公尺），即一條曲線上任一點到另一條曲線最遠之最近距離，取兩個方向之較大者；$\\operatorname{dist}(p,\\Gamma)$ 為點 $p$ 到曲線 $\\Gamma$ 之最短距離；"
          "$D_c$、$D_h$ 為上述門檻（公尺）。")
        P("第二步，通過者之成本與貪婪配對。所有候選依（成本、track 編號、檢出序號）排序後，貪婪地一對一配對，結果確定且可重現：")
        doc.eq("cost", r"C=\frac{d_c}{D_c}+\frac{d_H}{D_h}")
        P("其中：$C$ 為配對成本（無因次，越小越近）。先以核心距離與時間差做 numpy 向量化預篩，只對通過者計算曲線距離。")
        P("第三步，缺口處理。最多允許 1 個缺失時刻之 gap closing（max_missing_frames 為 1）；缺失兩個以上時刻則截斷原 track，由後續檢出另建新 track。"
          "只追蹤合格曲線。")
        H(4, "（4）輸出")
        P("tracks.csv（每條 track 之起訖、觀測數、截尾旗標與壽命）與逐觀測之 track_observations.npz，以及 tracking_summary 與壽命直方資料。")
        H(4, "（5）算例")
        P(f"以下為示範數值（Hausdorff 距離）：曲線 $\\Gamma_A$ 之加密點為 (0, 0)、(500, 0)、(1000, 100)、(1500, 300) 公尺，$\\Gamma_B$ 為 (200, 300)、(700, 350)、(1200, 500)、(1700, 700) 公尺。"
          f"$\\Gamma_A$ 各點到 $\\Gamma_B$ 之最近距離最大者為 {m(dAB, 5)} 公尺，$\\Gamma_B$ 各點到 $\\Gamma_A$ 之最近距離最大者為 {m(dBA, 5)} 公尺，故 $d_H={m(dH_toy, 5)}$ 公尺。"
          f"若核心距離 $d_c={m(dc_toy, 4)}$ 公尺，逐時門檻下兩者皆通過（$d_c\\le2{{,}}000$、$d_H\\le4{{,}}000$），成本 $C={m(dc_toy, 4)}/2000+{m(dH_toy, 5)}/4000={m(cost_toy, 4)}$。")
        rows = []
        for nm, dcv, dhv, cv in pair_rows:
            rows.append([nm, m(dcv, 4), m(dhv, 4), "通過" if cv is not None else "未通過（%s）" % ("$d_c>D_c$" if dcv > Dc else "$d_H>D_h$"),
                         m(cv, 4) if cv is not None else "—"])
        doc.table("pair", "貪婪配對算例之候選配對（示範數值，逐時門檻）",
                  ["候選配對", "$d_c$（公尺）", "$d_H$（公尺）", "門檻", "成本 $C$"], rows, "本研究示範計算。")
        order = sorted([r for r in pair_rows if r[3] is not None], key=lambda r: r[3])
        P(f"{doc.T('pair')}中有 {len(order)} 個候選通過門檻，依成本由小到大為 {'、'.join(r[0] + '（' + m(r[3], 4) + '）' for r in order)}。"
          f"貪婪配對先取成本最小之 {order[0][0]}，其兩端（{order[0][0].split('–')[0]} 與 {order[0][0].split('–')[1]}）即被佔用；"
          f"其次取不與已配對衝突之 {order[1][0]}；其餘涉及已佔用端點之候選一律捨棄。")
        P("實際數字（門檻掃描）：追蹤全部核心時，每個快照約 15 個核心／1000 平方公里，核心最近鄰間距僅數公里，與追蹤門檻相近，壽命大多可由隨機串接解釋"
          "（專案記錄之日產品置換壽命 p90 約 10 天、真實約 13 天）；因此改為只追蹤各站最強 10% 核心，並對各產品掃描追蹤門檻。"
          "掃描指標為「壽命不小於 3 個時間步之未截尾 track 比例，真實除以兩個置換 seed 之平均」，稱為相干性比值；"
          "比值接近 1 表示壽命與隨機串接無異。比值以流場區域為單位計算（A 區未拆成貢寮與龜山島西側），結果如" + doc.T('sweep_h') + "與" + doc.T('sweep_d') + "。")

        # gate sweep tables (project record values)
        doc.table("sweep_h", "逐時產品之追蹤門檻掃描：相干性比值（實際數字）",
                  ["核心／曲線門檻（公里）", "A 區（東北台灣）", "B 區（新竹）", "C 區（南灣）", "D 區（連江）"],
                  [["1／2", "51.7", "72.3", "31.8", "36.5"], ["**2／4（採用）**", "15.1", "19.3", "8.8", "14.1"], ["5／10", "5.5", "6.0", "3.1", "4.7"]],
                  "本計畫門檻掃描記錄（scripts/null_test_tracking.py，專案記錄值）；比值＝壽命不小於 3 小時之未截尾 track 比例，真實除以置換平均；以流場區域為單位。")
        doc.table("sweep_d", "日產品之追蹤門檻掃描：相干性比值（實際數字，強核心）",
                  ["核心／曲線門檻（公里）", "A 區（東北台灣）", "B 區（新竹）", "C 區（南灣）", "D 區（連江）"],
                  [["3／6", "2.35", "3.17", "2.55", "2.90"], ["**5／10（採用）**", "1.84", "2.52", "2.05", "2.43"], ["10／20", "1.69", "2.36", "1.84", "2.09"]],
                  "本計畫門檻掃描記錄（專案記錄值）；比值＝壽命不小於 3 個時間步（3 日）之未截尾 track 比例，真實除以置換平均；以流場區域為單位。")
        P("先前之日產品「全部核心（無強核心篩選）」測試中，10／20 公里之比值為 1.12–1.15，5／10 公里為 1.22–1.34，3／6 公里為 1.80–2.10，"
          "皆明顯低於強核心追蹤，這是改為強核心追蹤之原因。選擇理由：逐時採核心 2 公里、曲線 4 公里（相干性高且樣本數充足）；"
          "日產品採核心 5 公里、曲線 10 公里（相干性與樣本數之平衡）。上述以流場區域為單位之比值，與最終逐站統計之比值（步驟 8）單位不同，數值不可逐項對照。")

        # step 8
        H(3, "8、隨機時間置換對照")
        H(4, "（1）目的")
        P("估計「近鄰、同等密度之不相干結構被追蹤門檻串接」所能達到之機會水準壽命，作為真實壽命之對照，避免把隨機串接誤認為結構持續。")
        H(4, "（2）輸入")
        P("與真實追蹤相同之標準時間軸、可用性、覆蓋率與缺口；相同之偵測與追蹤設定及強核心篩選；亂數種子 1 與 2。")
        H(4, "（3）計算步驟")
        P("第一步，置換。track-period 以 --shuffle-seed 為種子，保留標準時間軸、可用性、覆蓋率與缺口，但把每個可用時間槽之偵測換成「僅在可用快照之間置換」之另一個快照"
          "（detection_keys 仍指向真實來源）。置換打斷真實之時間連續性，但保留每個快照之核心數、空間分布與強度。manifest 記錄 null_model 為 time_shuffle 與種子；"
          "彙整程式拒絕以 null 作為真實輸入，並要求 null 與真實之偵測、追蹤設定與強核心篩選完全相同（僅種子不同）。")
        P("第二步，超額與比值。兩個種子各自追蹤後取平均：")
        doc.eq("excess", r"\Delta f_{h}=f^{\mathrm{real}}_{\ge h}-\overline{f}^{\mathrm{null}}_{\ge h},\qquad "
                         r"\rho_h=\frac{f^{\mathrm{real}}_{\ge h}}{\overline{f}^{\mathrm{null}}_{\ge h}}\quad\left(\overline{f}^{\mathrm{null}}_{\ge h}=0\text{ 時不計算}\right)")
        P("其中：$f_{\\ge h}$ 為未截尾 track 中壽命不小於 $h$（小時）者之比例（無因次，見步驟 9）；上標 real 為真實追蹤、null 為置換追蹤，上橫線為兩個種子之平均；"
          "$\\Delta f_h$ 為超額（比例之差）；$\\rho_h$ 為比值。")
        H(4, "（4）輸出")
        P("各站之 null 壽命中位數與 p90、各壽命門檻之 null 比例、超額與比值，另有近岸外與近岸型之分項（見步驟 11）。")
        H(4, "（5）算例")
        rng = np.random.default_rng(1)
        slots = list(range(1, 9))
        perm = [int(v) for v in rng.permutation(slots)]
        P(f"以下為示範數值（置換）：可用時間槽 1 至 8 之真實偵測為 D1、D2、…、D8，缺口槽不參與。以種子 1 置換後，時間槽 1 至 8 依序改取 "
          f"{'、'.join('D' + str(v) for v in perm)}。真實序列中 D3 與 D4 時間相鄰、位置接近而可連成 track；置換後相鄰槽之偵測彼此來自不相干時刻，只有碰巧距離夠近者才會被串接。")
        P(f"實際數字：貢寮逐時產品壽命不小於 3 小時之未截尾 track 比例，真實為 {pct(G['track_frac_uncensored_ge_3h'])}、null 為 {pct(G['null_track_frac_uncensored_ge_3h'])}，"
          f"超額 {pct(G['excess_track_frac_uncensored_ge_3h'])}，比值 {ratio(G['ratio_track_frac_uncensored_ge_3h'])}；"
          f"貢寮日產品壽命不小於 168 小時（7 天）者，真實為 {pct(Gd['track_frac_uncensored_ge_168h'])}、null 為 {pct(Gd['null_track_frac_uncensored_ge_168h'])}，"
          f"超額 {pct(Gd['excess_track_frac_uncensored_ge_168h'])}，比值 {ratio(Gd['ratio_track_frac_uncensored_ge_168h'])}。"
          f"若 null 為 0（例如貢寮逐時不小於 12 小時），比值無定義而留空，只報超額。")

        # step 9
        H(3, "9、壽命與截尾")
        H(4, "（1）目的")
        P("定義 track 之壽命，並辨別因研究期結束或資料缺口而無法判斷是否持續之 track（截尾），使壽命統計不被截尾偏誤污染。")
        H(4, "（2）輸入")
        P("每條 track 之首次與末次觀測時間 $t_{\\mathrm{first}}$、$t_{\\mathrm{last}}$；名目取樣間隔 $\\Delta t$（逐時 1 小時，日產品 24 小時）；缺口紀錄。")
        H(4, "（3）計算步驟")
        P("第一步，壽命慣例（沿用 Kunz 等 2024 之 eddy-census 慣例）：")
        doc.eq("life", r"\text{壽命}=t_{\mathrm{last}}-t_{\mathrm{first}}+\Delta t")
        P("其中：壽命之單位為小時；$\\Delta t$ 為名目取樣間隔（小時）。單一時刻之 track 壽命為一個取樣間隔，不是 0。首末觀測之跨度（duration_seconds）另行保留。")
        P("第二步，截尾。左截尾：第一個觀測落在研究序列第一個時刻，或緊接資料缺口之後（前一幀不可用，或與前一幀時間差超過最大時間差）。"
          "右截尾：最後觀測因研究期結束或尾端資料缺口（time_gap、data_gap）而無法判斷是否持續。"
          "可用快照上「無檢出」或「無配對」，則保留為非截尾之實際結束。壽命統計只用未截尾 track。")
        P("第三步，壽命分數：")
        doc.eq("frac", r"f_{\ge h}=\frac{N_{h}}{N_{\mathrm{unc}}}")
        P("其中：$h$ 為壽命門檻（小時），逐時產品取 3、6、12、24，日產品取 48、72、168；$N_h$ 為未截尾 track 中壽命不小於 $h$ 之條數；$N_{\\mathrm{unc}}$ 為未截尾 track 數（track_n_uncensored），即分母。"
          "track 依其首個落在該站分析區內之觀測時間決定所屬之季節與年份分層。")
        H(4, "（4）輸出")
        P("每站之 track 數、未截尾 track 數、壽命平均、中位數與 p90，以及各門檻之壽命分數。")
        H(4, "（5）算例")
        P(f"實際數字（壽命慣例）：貢寮逐時產品之代表性 track（編號 {ex_h['track_id']}，起始 {ex_h['birth_time_utc']} UTC）有 {ex_h['n_observations']} 個逐時觀測，"
          f"壽命 {m(ex_h['lifetime_hours'], 4)} 小時；首末觀測跨度為 {m(ex_h['lifetime_hours'] - 1, 4)} 小時，加一個取樣間隔 1 小時，即 {m(ex_h['lifetime_hours'], 4)} 小時。"
          f"貢寮日產品之代表性 track（編號 {ex_d['track_id']}，起始 {ex_d['birth_time_utc']} UTC）有 {ex_d['n_observations']} 個逐日觀測，壽命 {m(ex_d['lifetime_hours'], 4)} 小時"
          f"（{m(ex_d['lifetime_hours'] / 24, 4)} 日）；首末觀測跨度 {m(ex_d['lifetime_hours'] - 24, 4)} 小時，加一個取樣間隔 24 小時而得。")
        doc.table("cens", "四條 track 之截尾判定（示範數值，逐時產品，取樣間隔 1 小時）",
                  ["track", "首次觀測", "末次觀測", "前後狀況", "壽命（小時）", "是否納入壽命統計"],
                  [["甲", "2024-03-05 02:00", "2024-03-05 08:00", "前一時刻可用且無檢出；末次後一時刻可用且無配對", "7", "納入"],
                   ["乙", "2024-03-09 12:00", "2024-03-09 12:00", "單一時刻；前後皆可用", "1", "納入"],
                   ["丙", "2024-03-11 20:00", "2024-03-12 03:00", "首次觀測前一小時為資料缺口", "8", "不納入（左截尾）"],
                   ["丁", "2025-12-31 20:00", "2025-12-31 23:00", "研究期結束", "4", "不納入（右截尾）"]],
                  "本研究示範計算；壽命＝末次－首次＋1 小時。")
        P(f"{doc.T('cens')}中甲之壽命為 8 − 2 = 6 小時之跨度加 1 小時 = 7 小時；乙為單一觀測，壽命 1 小時而不是 0；丙與丁因無法判斷真實起訖，不納入分母。"
          f"實際數字：貢寮逐時 track 共 {num(G['track_n_tracks'])} 條、未截尾 {num(G['track_n_uncensored'])} 條，兩者之差 {num(G['track_n_tracks'] - G['track_n_uncensored'])} 條為截尾 track；"
          f"日產品 {num(Gd['track_n_tracks'])} 條、未截尾 {num(Gd['track_n_uncensored'])} 條。")

        # step 10
        H(3, "10、統計指標")
        H(4, "（1）目的")
        P("以「頻率」與「強度」兩類指標，定量描述各站形成表層短期吸引潛勢熱區之情形。所有指標逐站獨立計算，範圍為站點分析區，任何門檻、熱點與色階只使用該站自己之資料，不跨站合併。")
        H(4, "（2）輸入")
        P("步驟 5 之有效快照、合格核心與曲線通過格點集合、強核心門檻；步驟 9 之 track；分層：全部（all）、四季（DJF、MAM、JJA、SON）與實際出現之年份（2024、2025）。")
        H(4, "（3）計算步驟")
        P("第一步，出現率。每個快照至多計一次：")
        doc.eq("rate", r"r_{\mathrm{core}}=\frac1N\sum_{k}\mathbb{1}^{\mathrm{core}}_k,\qquad r_{\mathrm{strong}}=\frac1N\sum_{k}\mathbb{1}^{\mathrm{strong}}_k,\qquad r_{\mathrm{curve}}=\frac1N\sum_{k}\mathbb{1}^{\mathrm{curve}}_k")
        P("其中：$N$ 為有效快照數；$\\mathbb{1}^{\\mathrm{core}}_k$、$\\mathbb{1}^{\\mathrm{strong}}_k$、$\\mathbb{1}^{\\mathrm{curve}}_k$ 為指示函數，快照 $k$ 在站內至少有一個合格核心、強核心、曲線通過站內格點時為 1，否則為 0。")
        P("第二步，曲線通過格點比例。快照 $k$ 之曲線所經過之站內格點數除以站內格點數，再對 $N$ 個快照平均：")
        doc.eq("pass", r"\bar p=\frac1N\sum_k\frac{\left|\bigcup_{\Gamma\in\mathcal{G}_k}\operatorname{cells}(\Gamma)\cap\Omega_{\mathrm{site}}\right|}{M}")
        P("其中：$\\mathcal{G}_k$ 為快照 $k$ 之合格曲線集合；$\\operatorname{cells}(\\Gamma)$ 為曲線通過之格點；$\\Omega_{\\mathrm{site}}$ 為站內格點集合；$M$ 為站內格點數。")
        P("第三步，核心密度與強度：")
        doc.eq("dens", r"\rho_{\mathrm{core}}=\frac{\sum_k n_k}{N\,A_{\mathrm{site}}}\times 1000")
        P("其中：$n_k$ 為快照 $k$ 站內合格核心數；$A_{\\mathrm{site}}$ 為站內面積（平方公里）；$\\rho_{\\mathrm{core}}$ 之單位為個每 1,000 平方公里每快照（含全部合格核心，不限強核心）。"
          "核心強度另報站內全部合格核心 $s_1$ 之平均、中位數與 p10（每日）。")
        P("第四步，逐格曲線通過頻率與熱點。格點 $g$ 之曲線通過頻率為：")
        doc.eq("hotF", r"F_g=\frac{n^{\mathrm{pass}}_g}{n^{\mathrm{valid}}_g},\qquad \text{熱點：}\ F_g\ge Q_{0.9}(\{F_g\}),\ F_g>0")
        P("其中：$n^{\\mathrm{pass}}_g$ 為曲線通過格點 $g$ 之快照數；$n^{\\mathrm{valid}}_g$ 為該格點之有效快照數；站外與無有效分母之格點為 NaN，不假裝為 0。"
          "熱點為站內 $F_g$ 最高之 10% 格點（hotspot_quantile 為 0.9），並要求 $F_g>0$；並列於門檻者全數納入，故格數可略多於 10%。"
          "輸出熱點門檻、格數、面積（平方公里）、熱點內平均 $F_g$、熱點內核心平均 $s_1$（每日，依核心數加權）、以及該站強核心落在熱點格點之比例。"
          "熱點為相對於該站自己分布之前 10%，不是絕對標準，面積約為站內面積之 10%。圖面另將 4-連通格數小於 3 之小塊濾除（hotspot_min_cells 為 3），"
          "此項只作用於圖面，不影響統計欄位。")
        P("第五步，日區塊 bootstrap 信賴區間。三個出現率與曲線通過格點比例，以「以日為區塊」重抽樣（1,000 次、種子 20260928），取 2.5% 與 97.5% 分位數為 95% 信賴區間，以反映同日快照之自相關：")
        doc.eq("boot", r"\mathrm{CI}_{95}=\left[Q_{0.025}\big(\{\hat r^{(b)}\}_{b=1}^{1000}\big),\ Q_{0.975}\big(\{\hat r^{(b)}\}_{b=1}^{1000}\big)\right]")
        P("其中：$\\hat r^{(b)}$ 為第 $b$ 次重抽樣所得之統計量。壽命分數、核心強度統計與傳播統計不附信賴區間。")
        H(4, "（4）輸出")
        P("每站每個分層一列統計（5 站 × 7 個分層 = 35 列，site_statistics.csv／.json），以及逐格空間陣列（spatial/*.npz）與成果圖。")
        H(4, "（5）算例")
        P(f"實際數字（出現率）：貢寮逐時產品有效快照 {num(G['n_valid_snapshots'])} 個，其中含強核心者 {num(G['strong_core_occurrence_numerator'])} 個，"
          f"$r_{{\\mathrm{{strong}}}}={num(G['strong_core_occurrence_numerator'])}/{num(G['n_valid_snapshots'])}={pct(G['strong_core_occurrence_rate'])}$，"
          f"95% 信賴區間 {ci(G['strong_core_occurrence_rate_ci95'])}。核心與曲線出現率皆為 100%（每個有效快照都至少有一個核心與一條曲線），不具區辨力，"
          "因此有資訊之頻率指標為強核心出現率、曲線通過格點比例與熱點統計。")
        P(f"實際數字（核心密度驗算）：貢寮逐時合格核心共 {num(G['core_s1_count'])} 個，站內面積 {m(G['site_area_km2'], 5)} 平方公里，"
          f"依式{doc.E('dens')}，$\\rho_{{\\mathrm{{core}}}}={num(G['core_s1_count'])}/({num(G['n_valid_snapshots'])}\\times{m(G['site_area_km2'], 5)})\\times1000="
          f"{m(G['core_s1_count'] / (G['n_valid_snapshots'] * G['site_area_km2']) * 1000, 4)}$ 個每 1,000 平方公里每快照，與統計表之 {m(G['core_density_per_1000km2_per_snapshot'], 4)} 相符。")
        P(f"以下為示範數值（熱點分位數）：站內 20 個格點之通過頻率 $F_g$ 由小到大為 {'、'.join(m(v) for v in Fvals)}。第 0.9 分位數（線性內插）為 {m(thrF)}；"
          f"$F_g$ 不小於此值者有 {int(hot.sum())} 格（{'、'.join(m(v) for v in Fvals[hot])}），並列於 0.40 者全數納入，故為 {int(hot.sum())} 格而非 2 格，這些格點構成熱點。"
          f"實際數字：貢寮逐時熱點為 {num(G['hotspot_cell_count'])} 格、{m(G['hotspot_area_km2'], 5)} 平方公里，熱點門檻 {m(G['hotspot_threshold_curve_pass_frequency'], 4)}，熱點內平均通過頻率 {m(G['hotspot_mean_curve_pass_frequency'], 4)}。")

        # step 11
        H(3, "11、近岸診斷")
        H(4, "（1）目的")
        P("檢查強核心與長壽命 track 是否集中在岸邊（可能是地形鎖定之應變場或邊界效應），並把 track 分為近岸型與近岸外兩類分別對照 null。")
        H(4, "（2）輸入")
        P("海岸遮罩；近岸緩衝帶寬度 $B=10$ 公里（coastal_buffer_km）；每個核心之位置；步驟 9 之 track。")
        H(4, "（3）計算步驟")
        P("距岸距離定義為「點到最近之非分析海域格點中心」之歐氏距離（以距離變換場雙線性取樣）：")
        doc.eq("coast", r"d_{\mathrm{coast}}(\mathbf{x})=\min_{\mathbf{c}\in\mathcal{L}}\left\|\mathbf{x}-\mathbf{c}\right\|,\qquad "
                        r"\text{近岸核心：}\ d_{\mathrm{coast}}<B;\quad \text{近岸型 track：}\ \overline{d_{\mathrm{coast}}}<B")
        P("其中：$\\mathcal{L}$ 為非分析海域（陸地與遮罩）格點中心之集合；$\\mathbf{c}$ 為其中一個格點中心（公尺）；$\\overline{d_{\\mathrm{coast}}}$ 為該 track 各觀測核心距岸距離之平均；$B$ 為緩衝帶寬度（10 公里）。"
          "核心距岸小於 $B$ 為近岸核心；track 之核心平均距岸小於 $B$ 為近岸型，其餘為近岸外。")
        P("輸出之診斷包括：近岸帶內核心與強核心之比例；長壽命 track（站內出生、未截尾、壽命不小於逐時 24 小時或日產品 168 小時）位於近岸帶之比例；"
          "近岸外強核心出現率（強核心門檻不變）；近岸外熱點（只在站內緩衝帶外格點重取分位數）；近岸型與近岸外各自之壽命分數及其 null。"
          "依專案記錄，核心不可能落在距非海域格點約 2.2–3 公里以內（檢出下限，因導數模板與局部極小值鄰域需要連續有效格點；診斷圖以灰帶標示站內最小核心距岸距離），因此 10 公里緩衝才具檢驗力。")
        H(4, "（4）輸出")
        P("近岸帶比例、近岸外強核心出現率、近岸外熱點與近岸型／近岸外壽命分數，皆寫入 site_statistics（只在 all 分層有值）；另有 coastal_diagnostic 圖。")
        H(4, "（5）算例")
        dd = [5.8, 6.4, 7.1, 8.3, 9.0]
        P(f"以下為示範數值：某 track 有 5 個觀測，核心距岸距離為 {'、'.join(m(v) for v in dd)} 公里，平均 {m(sum(dd) / len(dd), 4)} 公里，小於 10 公里，為近岸型；"
          "若平均為 10.5 公里則為近岸外。")
        P(f"實際數字：貢寮逐時代表性 track（編號 {ex_h['track_id']}）核心平均距岸 {m(ex_h['mean_coast_distance_m'] / 1000, 4)} 公里，大於 10 公里，歸為近岸外；"
          f"其近岸型對照 track（編號 {ex_hc['track_id']}）核心平均距岸 {m(ex_hc['mean_coast_distance_m'] / 1000, 4)} 公里，歸為近岸型。"
          f"貢寮逐時產品之全部核心有 {pct(G['frac_cores_within_buffer'])} 位於 10 公里近岸帶，強核心則有 {pct(G['frac_strong_cores_within_buffer'])}；"
          f"近岸外之強核心出現率為 {pct(G['offshore_strong_core_occurrence_rate'])}。")

        # step 12
        H(3, "12、傳播速度")
        H(4, "（1）目的")
        P("估計 TRAP 核心（結構）在模型內之位移速度與方向。此速度描述結構之位移，不等於水體或漂浮物之速度。")
        H(4, "（2）輸入")
        P("每條觀測數不少於 $n_{\\min}$ 之 track 之次網格核心位置 $x_k$、$y_k$（公尺）與觀測時間 $t_k$（小時或日）。")
        H(4, "（3）計算步驟")
        P("第一步，最小觀測數：步長小於 24 小時時為 $\\max(3,\\lceil6\\ \\text{小時}/\\Delta t\\rceil)+1$（逐時為 7），日產品為 3（manifest 之 propagation.min_observations）。")
        P("第二步，對每條 track 以最小平方法對核心位置作時間線性擬合：")
        doc.eq("fit", r"x(t)=x_0+c_x\,(t-\bar t),\qquad y(t)=y_0+c_y\,(t-\bar t)")
        P("其中：$t$ 為時間；$\\bar t$ 為該 track 觀測時間之平均；$x_0$、$y_0$ 為 $\\bar t$ 時之擬合位置（公尺）；$c_x$、$c_y$ 為擬合之東向與北向速度（公尺每小時，成果表換算為公里每日）。"
          "擬合殘差之均方根（RMS，公尺）一併記錄。")
        P("第三步，擬合位移與靜止規則：")
        doc.eq("disp", r"D_{\mathrm{fit}}=\sqrt{c_x^{2}+c_y^{2}}\,\left(t_{\mathrm{last}}-t_{\mathrm{first}}\right)<\Delta_{\mathrm{cell}}\ \Rightarrow\ \text{靜止}")
        P("其中：$D_{\\mathrm{fit}}$ 為擬合位移（公尺）；$\\Delta_{\\mathrm{cell}}$ 為 1 個格距（manifest 記錄逐時各流場區域約 990.6–993.9 公尺）。擬合位移小於 1 個格距者判為靜止，不進入速度與方向統計，"
          "因此速度欄位只含移動 track，另報靜止比例（prop_stationary_frac）。")
        P("第四步，速度大小：")
        doc.eq("speed", r"U=\sqrt{c_x^{2}+c_y^{2}}\times\frac{24}{1000}\quad(\text{公里每日，}c\text{ 以公尺每小時計})")
        P("其中：$U$ 為傳播速度（公里每日）；24 為每日小時數，1,000 為每公里公尺數。")
        H(4, "（4）輸出")
        P("每站移動與靜止 track 數、靜止比例、速度中位數與 p90（公里每日）、東向與北向分量中位數、擬合殘差 RMS 中位數（公尺）；另有 propagation 圖（日產品收錄）。")
        H(4, "（5）算例")
        P(f"以下為示範數值：某逐時 track 有 7 個觀測（$t=0,1,\\dots,6$ 小時），核心位置東向為 {'、'.join(m(v, 4) for v in px)} 公尺、北向為 {'、'.join(m(v, 4) for v in py)} 公尺。"
          f"最小平方擬合得 $c_x={m(cx, 4)}$ 公尺每小時、$c_y={m(cy, 4)}$ 公尺每小時；擬合殘差 RMS 為 {m(rms, 3)} 公尺。"
          f"擬合位移 $D_{{\\mathrm{{fit}}}}=\\sqrt{{{m(cx, 4)}^2+{m(cy, 4)}^2}}\\times{m(dur, 2)}={m(disp, 4)}$ 公尺，大於 993 公尺，為移動 track；"
          f"速度 $U={m(speed_kmday, 4)}$ 公里每日。若另一條 track 之擬合位移只有 600 公尺，小於 1 個格距，則判為靜止。")
        P(f"實際數字：貢寮逐時產品已擬合之 track {num(G['prop_n_tracks'])} 條，其中移動 {num(G['prop_n_moving'])} 條、靜止 {num(G['prop_n_stationary'])} 條，靜止比例 {pct(G['prop_stationary_frac'])}；"
          f"移動 track 之速度中位數 {m(G['prop_speed_median_kmday'], 4)} 公里每日、p90 為 {m(G['prop_speed_p90_kmday'], 4)} 公里每日。"
          "逐時（至少 7 個觀測）與日產品（至少 3 個觀測）之樣本不可直接對照。")

        # ---------------------------------------------------------------- （三）
        H(2, "（三）參數總表")
        P("本節彙整兩次正式執行之全部計算設定。檢出參數（" + doc.T('params') + "）逐時與日產品相同，取自兩次執行之 manifest。")
        doc.table("params", "檢出與曲線積分參數",
                  ["參數", "值", "說明"],
                  [["min_core_s1_sinv", f"{scu(det['min_core_s1_sinv'], 3)} 每秒", "核心最小量值（步驟 3）"],
                   ["min_eigenvalue_gap_sinv", f"{scu(det['min_eigenvalue_gap_sinv'], 3)} 每秒", "方向退化門檻（步驟 2）"],
                   ["local_min_radius_cells", str(det["local_min_radius_cells"]), "$3\\times3$ 局部極小值鄰域"],
                   ["local_min_tolerance_sinv", f"{scu(det['local_min_tolerance_sinv'], 3)} 每秒", "並列判定容許"],
                   ["core_min_separation_m", m(det["core_min_separation_m"]), "不另做核心間距去重"],
                   ["core_subgrid", f"{det['core_subgrid']}（refine 為 {'是' if det['core_subgrid_refine'] else '否'}）", "次網格核心（步驟 3）"],
                   ["curve_s1_stop_sinv", m(det["curve_s1_stop_sinv"]), "$s_1\\ge0$ 即停"],
                   ["curve_min_relative_strength", m(det["curve_min_relative_strength"]), "相對核心強度下限（步驟 4）"],
                   ["curve_monotonic_tolerance_sinv", f"{scu(det['curve_monotonic_tolerance_sinv'], 3)} 每秒", "單調性容許"],
                   ["max_curve_length_m", f"{num(det['max_curve_length_m'])} 公尺（每分支）", "曲線最長約 40 公里"],
                   ["max_integration_steps", num(det["max_integration_steps"]), "每分支最多步數"],
                   ["max_turn_angle_deg", f"{m(det['max_turn_angle_deg'])}°", "相鄰切向量最大轉角"],
                   ["min_curve_points", str(det["min_curve_points"]), "合格曲線最少點數"],
                   ["integration_step_m", "預設半個格距（約 500 公尺）", "RK2 步長"],
                   ["smoothing_radius_cells", str(det["smoothing_radius_cells"]), "無平滑（四鄰接圖擴散選項未啟用）"],
                   ["min_coverage", m(hm["min_coverage"]), "有效快照之有效格點比例下限（步驟 5）"]],
                  "本計畫兩次執行之 manifest（detection_config 與 min_coverage）。")
        tch, tcd = hm["tracking_config"], dm["tracking_config"]
        doc.table("trk", "追蹤、統計與近岸診斷設定",
                  ["項目", "逐時產品 r1_hourly", "日產品 d1_daily25h"],
                  [["追蹤核心距離門檻 $D_c$", f"{num(tch['max_core_distance_m'])} 公尺", f"{num(tcd['max_core_distance_m'])} 公尺"],
                   ["追蹤曲線距離門檻 $D_h$", f"{num(tch['max_curve_distance_m'])} 公尺", f"{num(tcd['max_curve_distance_m'])} 公尺"],
                   ["名目步長", f"{num(tch['nominal_time_step_seconds'])} 秒", f"{num(tcd['nominal_time_step_seconds'])} 秒"],
                   ["追蹤最大時間差", f"{num(tch['max_time_gap_seconds'])} 秒（3 小時）", f"{num(tcd['max_time_gap_seconds'])} 秒（36 小時）"],
                   ["最多缺失時刻（gap closing）", str(tch["max_missing_frames"]), str(tcd["max_missing_frames"])],
                   ["曲線加密步長", f"{num(tch['curve_sample_step_m'])} 公尺", f"{num(tcd['curve_sample_step_m'])} 公尺"],
                   ["強核心篩選（追蹤與統計）", "各站前 10% 核心", "各站前 10% 核心"],
                   ["時間置換 null 種子", "1、2", "1、2"],
                   ["壽命門檻（小時）", "3、6、12、24", "48、72、168"],
                   ["長壽命 track 定義", "未截尾、站內出生、壽命不小於 24 小時", "未截尾、站內出生、壽命不小於 168 小時"],
                   ["近岸緩衝帶", f"{m(hm['coastal_buffer_km'])} 公里", f"{m(dm['coastal_buffer_km'])} 公里"],
                   ["傳播速度最少觀測數", f"{hm['propagation']['min_observations']}", f"{dm['propagation']['min_observations']}"],
                   ["熱點分位數／最小連通格數", f"{m(hm['hotspot_quantile'])}／{hm['hotspot_min_cells']}", f"{m(dm['hotspot_quantile'])}／{dm['hotspot_min_cells']}"],
                   ["bootstrap 次數／種子", f"{num(hm['bootstrap_reps'])}／{hm['bootstrap_seed']}", f"{num(dm['bootstrap_reps'])}／{dm['bootstrap_seed']}"],
                   ["距岸距離場上限", f"{num(hm['coast_distance']['cap_m'])} 公尺", f"{num(dm['coast_distance']['cap_m'])} 公尺"]],
                  "本計畫兩次執行之 manifest（tracking_config 等）；強核心前 10% 對應 strong_quantile。")

        # ---------------------------------------------------------------- （四）
        H(2, "（四）程式架構與執行環境")
        P(f"程式為三階段、可平行、可續跑、帶 SHA-256 溯源之管線：階段 1 detect-shards 檢出各 domain-month 之核心與曲線；階段 2 track-period 串接月份、篩選強核心並串流追蹤；"
          f"階段 3 aggregate 彙整逐站統計與成果圖；另有 build-daily-fields 建立日產品。{doc.F('flow')}為管線流程，{doc.T('stages')}列出各階段之主要模組。")
        doc.fig("flow", "TRAP 分析管線流程", figs["flow"], "5.2in")
        doc.table("stages", "管線階段與主要模組",
                  ["階段", "CLI／模式", "主要模組", "說明"],
                  [["資料讀取", "STAGE、MASKS、PREPARE", "ocm.py、ocean_mask.py、scripts/stage_ocm_surface_local.py、scripts/export_ocean_masks.py", "記憶體映射唯讀；schema、來源與逐時遮罩 fail-closed 檢查"],
                   ["日產品", "DAILY（build-daily-fields）", "daily_fields.py", "25 小時置中平均，同 schema"],
                   ["研究設計", "—", "sites.py、configs/study_design.json", "四區五站、A 區最近 anchor 分割"],
                   ["階段 1", "DETECT（detect-shards）", "detection.py、period.py、models.py", "單一 domain-month，逐快照流式檢出，寫精簡 shard"],
                   ["階段 2", "TRACK、TRACKNULL（track-period）", "tracking.py、period.py", "串接月份、去重、強核心篩選、串流追蹤、時間置換 null"],
                   ["階段 3", "AGGREGATE（aggregate）", "aggregate.py、aggregate_figures.py、example_figures.py、report_font.py、coastline.py、colorbar_layout.py", "逐站統計與成果圖（schema traps.period.aggregate.v2）"],
                   ["對照腳本", "—", "scripts/null_test_tracking.py", "真實與置換壽命比較、相干性比值"],
                   ["編排", "SETUP、STATUS 等", "scripts/run_traps_server.sh", "見 SERVER_RUNBOOK"]],
                  "本計畫程式庫（src/traps 與 scripts）。")
        P("shard 內容：snapshots.npz（逐時 coverage 與 packbits 有效遮罩）、detections.npz（核心、曲線 ragged array、curve_pass_cells、停止原因整數碼、次網格核心與距岸距離）、grid_reference.npz（含距岸距離場）。"
          "每個輸出目錄附 manifest.json 與 manifest.sha256，記錄所有 artifact 之 SHA-256、位元組數、輸入 shard 與 track manifest 之 SHA-256、git_commit、git_dirty、param_id、state_label、全部參數與字型。"
          "aggregate 重新核對每個 shard manifest 之 SHA-256，並檢查四個流場區域共用同一組 param_id、偵測與追蹤設定、min_coverage 與 state_label（拒絕 PILOT_ONLY 與正式資料混用），且拒絕以 null 為主要輸入。"
          "_SUCCESS 標記與 --resume 只在 manifest 內全部 artifact 之 SHA-256 核對通過時略過重算；範例圖會重開來源月份並重算起始快照，"
          "核對 SHA-256，重算核心與已存核心須相差不超過 1 格（兩次執行之 recomputed_core_offset_cells 皆為 0）。")
        doc.table("env", "執行環境與耗時",
                  ["項目", "值"],
                  [["主機", "SERVER：Ubuntu 20.04、56 核、125 GB 記憶體"],
                   ["Python", "3.12（uv 管理，uv.lock 鎖定）"],
                   ["程式版本", f"git commit {hm['git_commit']}（git_dirty 為 {'是' if hm['git_dirty'] else '否'}）"],
                   ["逐時檢出", "約 0.9–1.1 秒每快照每核；96 個 domain-month 於 32–40 個平行 worker 約 45 分鐘（專案記錄）"],
                   ["強核心追蹤", "約 1.5 分鐘（專案記錄）"],
                   ["aggregate 耗時", f"逐時 {m(hm['runtime_seconds'], 4)} 秒；日產品 {m(dm['runtime_seconds'], 4)} 秒（manifest runtime_seconds）"],
                   ["日產品全流程", "約 10 分鐘（專案記錄）"],
                   ["測試", "254 項（pytest --collect-only）"]],
                  "本計畫執行紀錄與兩次執行之 manifest。")

        # ---------------------------------------------------------------- （五）
        H(2, "（五）品質控管與限制")
        H(3, "1、品質控管")
        P("品質控管項目包括：解析與合成流場之單元測試（梯度、特徵值、方向、曲線）；遮罩安全微分；標準時間軸不內插；資料缺口與期間邊界之截尾；"
          "shard、track、aggregate 三階段之 SHA-256 溯源；時間置換 null 之設定一致性檢查；範例圖之重算核心核對；圖面字型涵蓋檢查（缺合格 CJK 字型即失敗，不輸出方框字）。")
        H(3, "2、限制")
        P("一、僅模型資料。本計畫沒有獨立之漂流器或海廢觀測，未做 Serra 等（2020）圖 4a–c 型式之驗證；SVD 與 TRAP 共用 OCM 強迫，也不構成獨立驗證。結果只能稱為「表層短期吸引潛勢」。")
        P("二、壽命之機會成分。日產品壽命與 null 之相干性只有約 2 倍，2–3 天之壽命有相當部分來自機會；以強核心加 null 對照降低但未消除此問題。")
        P("三、檢出下限與近岸鎖定。核心無法定位於距非海域格點約 2–3 公里以內；強核心與長壽命 track 集中於近岸（10 公里帶內），多為近岸靜止型，應視為地形鎖定之應變場或邊界效應，而非漂移之收斂帶。")
        P("四、參數敏感度。相對強度 0.30、曲線長度上限、平滑尺度與追蹤門檻僅做有限之門檻掃描，未做完整敏感度分析；且 1 公里網格與次網格擬合之結論，不可直接與 Kunz 等（2024）之 0.1° 逐日分析相比。")
        P("五、日產品濾波。25 小時平均不是理想低通。")
        P("六、相對定義。強核心與熱點皆以各站自己之 10% 分位數定義，數值不可跨站比較大小意義。")
        P("七、出現率飽和。逐時與日產品中，各站之核心出現率與曲線出現率皆為 100%，不具區辨力；有資訊之頻率指標為強核心出現率、曲線通過格點比例與熱點統計。")
        P("八、傳播速度為模型內結構位移，且受 1 個格距之靜止門檻與最少觀測數影響。")
        P(f"九、距岸距離場上限。兩次最終結果之 manifest 記錄距離場於 {num(hm['coast_distance']['cap_m'] / 1000)} 公里飽和（coast_distance.cap_m 為 {num(hm['coast_distance']['cap_m'])}），"
          "而目前程式預設上限為 200 公里；距岸直方圖末端為「不小於 30 公里」，10 公里近岸分割不受影響，但超過 30 公里之距離不可從這批結果解讀。")
        H(3, "3、參考文獻")
        P("Serra, M. and Haller, G. (2016). Objective Eulerian coherent structures. Chaos, 26, 053110. https://doi.org/10.1063/1.4951720")
        P("Serra, M., et al. (2020). Search and rescue at sea aided by hidden flow structures. Nature Communications, 11, 2525. https://doi.org/10.1038/s41467-020-16281-x")
        P("Kunz, L., Griesel, A., Eden, C., Duran, R., and Sainte-Rose, B. (2024). Transient Attracting Profiles in the Great Pacific Garbage Patch. Ocean Science, 20, 1611–1630. https://doi.org/10.5194/os-20-1611-2024")
        P("Ghosh, et al. (2021). Persistency of debris accumulation in tidal estuaries using Lagrangian coherent structures. https://doi.org/10.1016/j.scitotenv.2021.146808（僅參照圖面呈現）")

    build.prefixes = ("表甲", "圖甲")
    return build
