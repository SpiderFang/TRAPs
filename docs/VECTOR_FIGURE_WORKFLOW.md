# TRAPs 報告向量圖流程

## 用途

正式成果圖的數值、色階、海岸遮罩、缺值與統計母體沿用既有 `aggregate` 輸出；本流程只改變繪圖格式與報告版面。圖面由既有 Matplotlib Figure 直接輸出，沒有把 PNG 包入 SVG，也沒有用點陣圖描圖。

## 啟用方式

在本機執行繪圖或重繪工具前明確設定：

```bash
export TRAPS_VECTOR_REPORT=1
```

未設定或值不是 `1` 時，仍只輸出既有 PNG。啟用後，每個成果圖會同時輸出：

- `name.png`：相容預覽；報告模式固定至少 300 dpi（6.3 英吋基準寬度約 1890 px）；
- `name.svg`：報告主要向量格式，文字以 path glyph 保存；
- `name.pdf`：可嵌入字型的向量 PDF。

四季圖採 2×2 地圖並將兩項統計置於下方；三面板方法比較圖採兩列配置；所有報告 Figure 的實際寬度固定為 6.3 英吋，圖高依內容調整但不超過 8.0 英吋。這個固定寬度是 Word 置入後的列印契約，故版面紀錄同時檢查縮入 6.3 英吋後的最小可見字級至少 8 pt，標題約 10–12 pt。地圖 aspect、色階、NaN／海岸遮罩與資料值不在此流程中改動。

## 安全檢查

`traps.vector_figures.save_figure` 會：

1. 對 Figure 內的 collections、axes 與 colorbar 關閉 rasterization；
2. 以同一 Figure 直接輸出 PNG、SVG、PDF；
3. 拒絕含 `<image>`、`data:image` 或外部 href 的 SVG；
4. 在 aggregate 根目錄留下 `vector_figure_layout.json`，記錄每圖英吋尺寸、最小可見字級、文字／axes 越界數及向量格式。

`scripts/refresh_period_figures.py render` 會再驗證兩套產品共 120 張 PNG 是否各有同名 SVG/PDF、SVG 是否純向量，以及數值 artifact 是否仍由原 manifest 保護。`publish` 會備份並同步 PNG、SVG、PDF；非 PNG 的科學 CSV、JSON、NPZ 必須逐值相同才會進入發布階段。`figure_presentation` 會記錄向量圖數量與 `primary_format: "svg"`。

## A 區整體圖

`scripts/plot_domain_a_overview.py` 共用同一向量儲存層。直接本機執行會在 A 區 overview PNG 旁產生 SVG/PDF；使用 `--via-ssh` 時，若本機已設定 `TRAPS_VECTOR_REPORT=1`，環境旗標會傳到 SERVER，回傳的 PNG、SVG、PDF 會同步下載到本機與報告圖目錄。

## 驗證

使用專案指定的 bundled Python 執行：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/mustlab/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest -q tests/test_vector_figures.py
```

測試使用合成 Figure 與實際成果 renderer，驗證 SVG 零點陣、NaN 資料保留、SVG 實際 6.3 英吋寬度、置入後 8 pt 最小字級、文字／axes 不越界，以及未設定旗標時仍只輸出 PNG。

## 2026-10-13 圖面與統計修正

- `long_tracks` 色條標籤改為「單一結構連續存在天數／時數」；同一產品五站共用色階（下限＝長壽命門檻，上限＝五站最長壽命進位到 10 天／12 小時的倍數）。`aggregate` 於渲染前跨站計算並寫入 manifest 的 `long_track_colour_range`；`TrackFigureInputs.colour_max_hours` 為 `None` 時行為與舊版相同。
- 空間 2×2 圖的「平均未截尾壽命」面板：逐日產品以「天」（小時 ÷ 24）顯示，逐時產品維持「小時」；npz、CSV 與 `colour_limits` 仍為小時。報告模式的 2×2 版面同時調整：兩排間距依色條高度計算、右欄隱藏重複的緯度標籤、圖高不再固定 5.8 英吋。
- 新增 `site_<site>_cumulative_days`（僅逐日產品）：每格 TRAP 曲線通過的有效日數，等於 `round(curve_pass_frequency × n_valid)`（npz 未儲存通過次數，此還原需為整數，否則拒絕）。
- 新增 `site_<site>_hotspot_overlap` 與 `hotspot_overlap.csv/json`：逐時與逐日曲線通過頻率前 10% 熱點（沿用 `aggregate._top_quantile_mask`，未濾除小連通塊）的交集、Jaccard、1／2 格容許與隨機基準。
- `scripts/render_report_fixes.py` 只由正式 `aggregate_report` 產物（spatial npz、site_statistics、manifest）重繪 `spatial`、`hotspots`、`seasonal` 與上述新圖，不需要 SERVER 的 track 資料；其餘圖（lifetime、timeseries、coastal_diagnostic、所有 track 診斷圖）仍須由 SERVER 上的 `aggregate` 重繪。

### 2026-10-13 後續修正（壽命圖、A 區圖、版面 gate）

- `lifetime` 圖：左面板 CCDF 圖例改為 figure-level、兩欄、置於兩個面板下方（圖高增加 0.55 英吋），不再壓在曲線上；左標題在語意邊界換行（「壽命互補累積分布」／「壽命對象：強核心結構」／「（各站前 10%）」），右標題改為「吸引強度隨正規化年齡演化」／「未截尾且壽命 ≥ N 小時」／「n = … 條軌跡」。軌跡數不再畫在面板內，因為面板內任何固定位置都可能壓到曲線。
- `scripts/plot_domain_a_overview.py`：圖例移到 x 軸標籤下方（地圖外、三欄），標題與地圖間距加大，避免壓到「龜山島西側」標籤、陸地與熱點格點。
- 局部重繪：`aggregate --figure-kinds lifetime[,...]` 只重繪指定圖種（略過 WP-H／WP-I 的 snapshot 圖）；`scripts/rerender_figure_kinds.py` 以舊 manifest 組參數呼叫它並寫 `figure_layout_check_partial.json`。局部重繪的 manifest 帶 `figure_kinds_filter`，只供取圖與數值核對，**不可發布**。
- `refresh_period_figures.py` 版面 gate 與 `vector_figure_layout.json` 對齊：兩者共用 `traps.vector_figures.off_view_tick_text_ids_of`，只排除可由刻度位置證實落在 view 範圍外（不會被繪出）的 tick label；真正碰到 Figure 邊界的文字或 view 內刻度標籤仍計為越界。gate 條件抽成 `layout_failures(records)`。

## 2026-10-14 統一圖面標準（畫布與地圖框）

目的：報告圖不再隨各區域長寬比改變高度。所有版面常數集中在 `src/traps/figure_standard.py`，報告模式（`TRAPS_VECTOR_REPORT=1`）下各繪圖函式一律照此建立軸框。科學數值（色階、遮罩、近岸帶、錨點、統計表）不變，只改版面與文字。

### 畫布家族（寬皆 6.3 in）

| 家族 | 畫布 | 圖種 |
| --- | --- | --- |
| single | 6.3 × 4.4 in | hotspots、cumulative_days、hotspot_overlap、long_tracks、A 區整體圖（共用地圖框 `MAP1_AXES` 0.72/0.95/4.50×2.65 in，色條 `MAP1_CBAR`）；壽命雙面板 lifetime（左右各 2.15×2.50 in） |
| multi | 6.3 × 7.0 in | spatial、seasonal、example／example_coastal、method_comparison、propagation、coastal_diagnostic、timeseries、track_density（上下兩張地圖，各 4.50×2.30 in） |

multi 家族的圖面差異大，因此「家族內畫布一致」之外，另以版面類別（map1、lifetime、spatial、seasonal、example、method、propagation、coastal、timeseries、density）要求同類別的軸框一致；`KIND_LAYOUT` 登錄每個圖種的家族與類別。

### 地圖範圍

各站以原視窗中心為準，把經緯度範圍擴張到與固定軸框同比例（含 cos 緯度），不改軸框大小（`fit_limits_to_box`）。色階只取原視窗內的場；近岸帶、錨點與遮罩不變。離線重建的近岸距離在站外為 0，報告需用真實格點距離（`render_report_fixes.py --runs-root`）。

### 字級與文字

- 標題 11 pt、副標 10 pt、其餘（刻度、標籤、圖例、色條）至少 8 pt。
- 圖面文字以肯定句表述：「未截尾」一律改為「起訖皆在觀測期內」或「完整軌跡」；圖面的 track 一律為「軌跡」；避免以「不、無、未、非、少於、小於、低於」描述條件，改述為「為 0」「以內」「至少」等。
- 例外：決定統計內容的 `strong_core_structures`、`per_site_top`、`all_qualified_structures` 三個鍵不可改寫。

### Gate

- `refresh_period_figures.py render` 與 `rerender_figure_kinds.py` 皆呼叫 `standard_layout_failures`，結果寫入 `figure_layout_check.json` 的 `uniform_standard`。
- 獨立檢查：`python scripts/check_uniform_figures.py --root <輸出目錄> --expected-total <張數>`，遞迴讀取 `vector_figure_layout.json`，檢查字級 ≥ 8 pt、溢出 0、SVG 內 `<image>` 0、家族畫布（±0.01 in）、同圖種及同版面類別軸框一致，不通過時結束碼為 1。
- 測試：`tests/test_uniform_figure_standard_20261014.py`。

## 2026-10-16 簡要報告圖組（concise）

簡要報告使用 16 張圖：逐時 A 區總覽、五站逐時 `hotspots`、五站逐日 `long_tracks` 與 `example`。版面沿用 2026-10-14 的統一標準（6.3 in 寬；single 家族 6.3×4.4、multi 家族 6.3×7.0；同圖種軸框一致），另加下列只影響標示的設定，科學數值、遮罩、近岸帶與統計不變，正式圖組不受影響。

### 啟用方式

設定 `TRAPS_VECTOR_REPORT=1` 與 `TRAPS_CONCISE_OPTIONS=<選項檔 JSON>`（schema `traps.concise_options.v1`），再用 `scripts/rerender_figure_kinds.py --figure-kinds ...`（軌跡類圖）或 `scripts/plot_domain_a_overview.py`（A 區）繪圖。環境變數未設定時完全不啟用。`TRAPS_CONCISE_RECORD=<jsonl>` 會記錄每張圖「原本」自己的色階，供挑選共用色階（兩階段：先記錄、再以共用色階重繪）。

選項檔欄位：

- `display_names`：站名顯示覆寫（例如 `{"guishan": "龜山島"}`，site_id 不變）。
- `limits`：跨站共用色階，例如 `hotspots.curve_pass_frequency = [0, 0.35]`、`example.s1_day = [-12, 12]`、`domainA_overview.curve_pass_frequency = [0, 0.35]`。

### 版面規則

- 地圖座標軸：規則間距的經緯度刻度（`regular_ticks`，每軸至少 3 個、取「標示刻度數 ≥ 4」的最大間距），度數格式（`121.5°E`、`24.8°N`，小數 1 位，全部圖種一致），刻度記號畫在框線外側，框線 1.1 pt。
- 色條：兩端點 vmin／vmax 都標示，端點之間有規則的內部刻度（`colorbar_tick_values`）；資料超出共用色階時加延伸箭頭，箭頭畫在軸框外，色條本體軸框不變。
- 圖面文字不含否定詞，也不含舊站名「龜山島西側」（簡要圖組顯示「龜山島」）。

### Gate

`python scripts/check_concise_figures.py --root <輸出目錄> --expected-total 16`：統一標準、字級、越界、SVG 無 `<image>`、PNG ≥ 300 dpi，加上標示刻度、色條端點、同產品同圖種各站共用色階一致與用語檢查。版面紀錄 `vector_figure_layout.json` 的 `concise` 欄位保存每張圖的標示刻度與色條資訊。`scripts/assemble_concise_figures.py` 把 SERVER 原始輸出整理成 `<param_id>/<site>/…` 並寫 manifest（來源、選項、共用色階、sha256）。測試：`tests/test_concise_figures_20261016.py`。

### 2026-10-16 簡要報告：縮放代表 TRAP（example_zoom）

簡要報告圖組的逐日 `example` 改為 Kunz et al. (2024) Fig. 2 風格的單一 TRAP 放大圖（single 家族 6.3×4.4 英吋、MAP1 軸框；gate 以
`figure_standard.CONCISE_KIND_LAYOUT` 檢查）。啟用方式：concise 選項檔加入 `"example_zoom": {"enabled": true}`，並以
`limits.example_zoom.s1_day` 給五站共用的對稱 s1 色階；沒有 `example_zoom` 時沿用原本的全域代表圖。

* 選取規則（只影響呈現，不影響統計）：站內出生、起訖皆在觀測期內、壽命 ≥ 7 天；整條核心路徑距分析區矩形外緣（格網最外緣）
  ≥ 10 km 者中取壽命最長，同壽命取平均核心 s1 最強，再取較小 track_id；沒有符合者放寬到 5 km 並回報。
* 視窗：各站相同公里大小（預設 78 × 45.9 km，與固定軸框同長寬比，經度依 cos 緯度換算），以起始核心為中心；後續位置超出視窗時改以外接框
  中心；最後平移到分析區矩形之內（位移量記在 manifest）。
* 內容：視窗內 s1、15×10 抽稀流速箭頭（每 1 m/s 0.3 英吋，五站相同，右下角附 0.5 m/s 參考箭頭）、代表 TRAP 起始曲線（紅線）與核心（紅點）、
  壽命 1/3 與 2/3 處的後續曲線（鮭魚紅、透明度遞增、標示天數）與核心路徑；不畫其他 TRAP 曲線。
* 兩階段：第一階段以 `TRAPS_CONCISE_RECORD` 記錄各站視窗內 |s1| 的 98 百分位自然上限，再選共用的整齊數值（目前 10 day⁻¹）重繪；
  `scripts/assemble_concise_figures.py --example-zoom-root … --keep-example-v1` 會保留舊版為 `*_example_v1.*`，並把選取結果寫入 manifest。
* `scripts/check_concise_figures.py` 另檢查 manifest 的 `example_zoom`：五站視窗公里數、s1 色階、箭頭比例相同；後續位置在視窗內；
  邊界距離不小於所用門檻。

### 2026-10-16 簡要報告：色條平整端點與 1 m/s 參考箭頭

簡要報告圖組的全部色條（hotspots、A 區總覽、縮放代表圖、長壽命路徑）一律兩端平整，不畫延伸箭頭；超出共用色階的值以端點色飽和，
飽和格點比例記在 layout 紀錄與 manifest 的 `colourbars.clipped_fraction`。`concise_record_failures` 檢查每個色條 `extend == "neither"`。
縮放代表圖的流速參考箭頭為 1 m/s（0.3 英吋），不加底框，位於地圖下緣外、經度標籤列右側。
`assemble_concise_figures.py --keep-v2 hotspots example domainA_overview` 會把覆寫前的版本另存為 `*_v2.*`。

### 2026-10-16 簡要報告：移除圖內標題與副標題

簡要報告圖組的 16 張圖不再畫標題與副標題（說明由報告圖說承擔，與期中報告參考版相同）。作法是在 `assemble_concise_figures.py` 之後執行
`python scripts/strip_concise_titles.py --root outputs/concise_figures_20261016`：

* 先把含標題的 png／svg／pdf 與 layout、manifest、gate 另存為 `*_v3.*`（已存在則不覆寫；重跑時一律由 `*_v3` 出發，結果相同）。
* SVG 內 matplotlib 文字以路徑繪出，每個 `<g id="text_N">` 的第一個子節點是原字串註解；`figure_1` 直接子層的 `text_N` 即標題與副標題，字串先寫入
  manifest 的 `removed_titles`（圖 → `{title, subtitle, facts}`，`facts` 為熱點面積、軌跡數、起始日、壽命、門檻等數字欄位，供圖說逐項比對），再刪除群組。
* 字形路徑 `<path id=…>` 只定義在第一次用到的文字群組內，標題先出現，所以刪除前必須把仍被其他文字（例如圖例的「熱點」「起始」）參照的定義搬到文件最前面的
  `<defs>`；否則別處的字會消失。`dangling_references` 檢查移除後沒有找不到定義的參照，gate 也檢查。
* 16 張圖都以同一高度（`figure_standard.CONCISE_TITLE_CROP_INCH` = 0.60 英吋）從頂端裁掉：viewBox 起點下移、height 縮短；軸框以左下角為原點，座標軸、色條、圖例相對位置不變。
  畫布 6.3 × 4.4 → 6.3 × 3.8 英吋，最高內容（色條頂端刻度）距頂端 0.151 英吋。
* PDF 由 LibreOffice Draw 轉自編輯後的 SVG，PNG 由 `pdftoppm -r 300 -cropbox` 轉自該 PDF（1890 × 1140 px）；腳本驗證 PDF 頁面尺寸、PNG 像素、頂端留白全白，
  以及 PNG 與 `*_v3.png` 裁掉同高後以 0.1 英吋方塊平均灰階比對（缺字、缺線會超過門檻），結果記在 manifest 的 `title_removal.png_checks`。
* gate 新增：`no_title`（SVG 無圖層級文字、被移除的字串不在 SVG 與 layout、manifest 有 `removed_titles`）、`top_margin`（同版面類別頂端留白相同、畫布 = 家族畫布 − 裁切量、
  SVG viewBox 與 PDF 頁面尺寸一致）、`svg_references`（無懸空參照）；PNG 檢查加上長寬比。含標題的舊版圖組用 `--allow-titles`。

### 2026-10-16 簡要報告：指定色條色階／刻度與參考箭頭移入地圖

* 曲線通過頻率色條（hotspots 與 A 區總覽）：色階 0–0.4，刻度 0／0.1／0.2／0.3／0.4，兩端平整；選項檔 `limits.hotspots.curve_pass_frequency`、`limits.domainA_overview.curve_pass_frequency`。
* 長壽命路徑壽命色條：色階 7–120 天（下限 = 長壽命門檻 7 天，上限不得小於本站最長 track），刻度 7／20／40／60／80／100／120，兩端平整；
  選項檔 `limits.long_tracks.lifetime` 與 `ticks.long_tracks.lifetime`（`concise_style.shared_ticks`／`style_colorbar_shared(tick_values=…)`；自動刻度會選 25 為間距，所以這條色條要明確指定）。
* 縮放代表圖的 1 m/s 參考箭頭放在地圖內右下角（箭頭尖端位於軸英吋 (4.08, 0.16)），不加底框，文字與箭頭加細白色描邊。
  固定保留區 `ZOOM_KEY_PATCH_INCH = (3.52, 0, 4.50, 0.34)`（五站相同位置與大小）：外框（加 0.05 英吋）與保留區相交的流速箭頭不畫，其餘 15×10 取樣與每 1 m/s 0.3 英吋的長度比例不變。
  版面紀錄的 `concise.example_key` 保存參考箭頭外框、保留區、排除的箭頭數、重疊數（必須為 0）與最小間距；SVG 內群組 id 為 `reference_key_arrow`、`reference_key_label`（描邊文字沒有字串註解，所以用 gid）。
* gate 新增：`requested_colourbars`（色階與刻度等於指定值，`--no-expected-colourbars` 可略過，供合成資料測試）、`example_key_in_axes`（SVG 的兩個群組各 1 個且在 `axes_1` 內）、
  concise 檢查的參考箭頭項（在軸內、在保留區內、與已畫箭頭 0 重疊）。含標題的來源版本改用 `strip_concise_titles.py --backup-suffix _v5 --previous-suffix _v4`（上一版無標題檔為 `*_v4.*`，舊含標題檔 `*_v3.*` 保留）。

### 2026-10-16 簡要報告：只用逐時產品、熱點四季重現圖、逐時代表 TRAP、400 dpi PNG

簡要報告圖組改為只使用逐時產品 `r1_hourly`（逐日產品 `d1_daily25h` 的舊檔仍在交付目錄，但 manifest 的 `excluded_groups` 把它排除在清單與 gate 之外）。
16 張圖 = A 區總覽 1 ＋ 各站 `hotspots` 5 ＋ 各站 `seasonal_recurrence` 5 ＋ 各站逐時 `example` 5，皆在 `r1_hourly/<site>/`（總覽在 `domainA/r1_hourly/`）。

* 熱點四季重現圖（`src/traps/seasonal_recurrence.py`、`scripts/plot_seasonal_recurrence.py`）：每格在 DJF／MAM／JJA／SON 中有幾季屬於該季站內曲線通過頻率前 10% 熱點，
  各季用自己的 90 百分位門檻（規則同 `aggregate._top_quantile_mask`：`>= 門檻` 且 `> 0`）。程式由 `spatial_<季>.npz` 的 `curve_pass_frequency` 與 `site_mask` 重算，
  並與 npz 內存的 `hotspot_curve_mask` 及門檻逐格核對，不一致就中止。1–4 季以 4 級離散色條顯示（色條兩端平整，刻度 1／2／3／4，標籤「熱點出現季數」，
  `concise_style.style_colorbar_discrete`；gate 檢查 `discrete`／`classes`），0 季透明、露出淺灰分析海域。側車 `site_<site>_seasonal_recurrence.json` 記錄各季門檻與格數、
  全年熱點格點中在 ≥3 季與 4 季皆為熱點的比例、季熱點聯集依 1–4 季分級的格數與占比；gate 以本機 npz（`--spatial-root`）重算並逐項比對。
* 逐時代表 TRAP（`example_figures.hourly_candidate_features`、`select_hourly_track`；選項檔 `example_zoom.mode = "hourly"`）：候選是站內出生、未截尾的強核心 track；
  起始核心在熱點內（圖面所示的前 10% 熱點）或距最近熱點格點不超過 3 km；核心路徑距分析區外緣 ≥ 5 km 且離起點最遠 ≤ 12 km；起始與兩個後續時刻的已存曲線長度皆 ≥ 6 km（曲線清楚可見）。
  壽命先看 ≥ 24 h 層（沒有符合者才用 ≥ 12 h 層）；層內依「熱點內優先 → 三個時刻中最短曲線最長 → 平均核心 s1 最強 → 較小 track_id」決定，完全可重現。
  後續曲線取壽命 1/3 與 2/3 處最近的觀測（逐時資料即整數小時）。圖面副標題（之後移除）為 `起始 YYYY-MM-DD HH:MM UTC｜壽命 N h`；側車 `site_<site>_example.json` 記錄 track id、
  起始 UTC 與臺灣時間（UTC+8）、壽命、平均距岸、類別（近岸／近岸外）、位移、是否在熱點內等；gate 檢查側車自洽、與 manifest／標題 facts／縮放紀錄一致。
* PNG 後備檔 400 dpi（2520 × 1520 px），SVG 仍是主要向量檔：`strip_concise_titles.py --dpi 400`（`pdftoppm -r 400`）；驗證時把 400 dpi PNG 以 Lanczos 縮回來源的 300 dpi 再做方塊比對。
  `check_concise_figures.py` 的 `MIN_PNG_DPI` 為 400（`--min-png-dpi` 可降低，供合成資料測試）。
* `scripts/assemble_hourly_concise_figures.py`：把四組含標題的 SERVER 原始輸出整理成暫存根目錄、執行 strip（`_t` 備份）、覆寫前把交付目錄既有檔另存為 `*_v6.*`、寫新的 manifest
  （`seasonal_recurrence`、`example_hourly`、`example_zoom`、`removed_titles`、`excluded_groups`）；`scripts/build_concise_contact_sheet.py` 產生檢視用 contact sheet。

### 2026-10-20 簡要報告：座標軸兩端標示、同尺寸版面、600 dpi、逐時代表圖視窗含站點與海岸

* 座標軸兩端：地圖範圍向外貼齊刻度間距（`concise_style.snap_limits_to_ticks`，間距階梯 2／1／0.5／0.4／0.3／0.25／0.2／0.15／0.1／0.05／0.02°），框線兩端就是標示的刻度值；
  軸框不變，用 `set_aspect(exact_aspect(...), adjustable="datalim")` 吃掉殘餘長寬比誤差（≤ 2%）。範圍因此比原先大 19–173%（框固定，只能擴範圍）。
  標示 1 位小數；刻度間距小於 0.1° 或用 0.25° 階梯時整張圖 2 位小數。色條：連續色條兩端標 vmin／vmax（熱點 0 與 0.4、s1 −15 與 15）；
  離散「熱點出現季數」色條為類別軸，1–4 標在各級中心。
* 同尺寸：16 張圖畫布 6.3 × 3.8 英吋、地圖軸框 [0.72, 0.2, 4.5, 2.65]、色條框 [5.32, 0.2, 0.16, 2.65]、圖例區位置相同（gate `same_size_boxes`，容差 0.01 英吋）。
* PNG 後備檔 600 dpi，確切 3780 × 2280 px（`pdftoppm -scale-to-x/-y`，避免 `-r 600` 浮點進位多 1 px）；SVG 仍是主要向量檔；gate `png_resolution` 檢查 dpi ≥ 600 與確切像素數。
* 逐時代表圖視窗：刻度格線上的矩形（預設經度 0.25° × 3＝0.75°、緯度 0.1° × 4＝0.4°，約 75 × 44 km；低緯度長寬比超過 2% 時改用 0.3° × 3 × 0.1° × 5 或 0.25° × 4 × 0.1° × 5），
  必須含站點錨點（標示站名）與 ≥ 12 km 海岸線，並含整條核心路徑與三個時刻的曲線（`find_example_window`；`require_window=true` 時候選須有可行視窗）。
  貢寮另設 `per_site.gongliao.max_mean_coast_km=12`（近岸軌跡）。gate `hourly_example` 檢查 `anchor_in_window`、視窗與圖面軸範圍一致、海岸線長度、起點在站統計範圍內。

### 2026-10-20（第二輪）空白面積最小的地圖範圍、分析範圍輪廓、站名位置

* 範圍搜尋（`concise_style.snap_limits_to_ticks`）：覆蓋資料外接框（站內分析區含 2 格邊界、錨點 ± 0.03°；A 區含兩個錨點）、兩端為標示刻度、軸框長寬比誤差 ≤ 2%、每軸 3–7 個標示。
  起點取 0.05° 格線，刻度從下限起算（不必是間距的整數倍）；間距先用經度 0.05／0.1／0.2／0.25／0.3／0.4／0.5、緯度 0.05／0.1／0.2／0.25／0.5，
  搜尋同時納入 0.05° 的所有倍數（權重 1.03）與 0.55–2° 粗間距（權重 1.25）。
* 目標是「空白」最小：`extent_coverage.Coverage`（0.01° 點陣＋積分影像）把範圍內的資料格（站內分析區遮罩）與陸地（海岸線 Polygon 以 XOR 填色）算作「有東西可看」，
  空白 ＝ 分析範圍外的海；成本 ＝ 空白面積 ＋ 0.1 × 範圍面積。版面紀錄（`concise.map_axes[].extent_info`）與 manifest（`map_extent[].fractions`）保存資料／陸地／空白比例；
  `scripts/measure_extent_blank.py` 以本機 npz 遮罩重算新舊 manifest 的比例。
* 分析範圍：熱點、四季重現、A 區總覽在資料格邊界畫細深灰輪廓（`draw_analysis_outline`，0.7 pt），圖例加「分析範圍」（四季重現原「分析海域」填色項改為同一項、附輪廓）；
  熱點圖例 4 項改單列（`ncol=4`），圖例底緣位置不變。gate `analysis_range` 檢查圖例有「分析範圍」並輸出 `map_extent_fractions`。
* 逐時代表圖站名（`_place_site_label`）：八個方位中取標示框（含白描邊）在軸內、與曲線／核心／路徑／日數標籤距離 ≥ 0.04 英吋、不碰 1 m/s 參考箭頭保留區者，
  再取落在陸地（無場值）側比例高、離軌跡遠者；結果記入側車 `site_label`，gate 檢查 `valid`。視窗內場值／陸地／空白比例記入 `window_fractions`。

### 2026-10-20（第三輪）地圖範圍＝flow domain 範圍、統一框高、純數字刻度、陸地填色、A 區兩站同範圍

* 範圍：站點圖（熱點、四季重現、A 區總覽）的地圖範圍＝該站 flow domain 網格範圍各自四捨五入到最近的 0.05°（`concise_style.domain_limits`），與 OCM-SVD-Analysis、Lagrangian-Ensemble-Backtracking 的流場範圍圖一致：
  A 121.30–122.80 × 24.60–25.50、B（新竹）119.70–121.20 × 24.30–25.20、C（南灣）120.15–121.60 × 21.55–22.45、D（連江）119.20–120.70 × 25.75–26.65。
  不再搜尋範圍；資料填滿整個 domain。標準地圖框改為 4.50 × 2.98 英吋（長寬比 1.510，約等於 domain 的公里長寬比 1.51），畫布 6.3 × 4.73（移除標題後 6.3 × 4.13），PNG 確切 3780 × 2478 px；所有 16 張圖仍用同一個框（±0.01 英吋）。長寬比誤差須 ≤ 3%（`frame_aspect_error`，gate 檢查）。
* 刻度（`concise_style.axis_ticks`）：兩端必標；先找從下限起的等間距刻度（0.5／0.3／0.25／0.2／0.15／0.1°）剛好在上限結束（例如經度 121.3、121.6 … 122.8），排不出時才用兩端加整數倍內部刻度；每軸 4–7 個標示；標示為純數字（座標軸標題 「經度 (°E)」「緯度 (°N)」 帶單位）。
* 陸地：海岸線多邊形以 #d9d9d9 填色（zorder 0.5），海岸線描邊在上（zorder 6）；16 張圖（含代表 TRAP）都有。分析範圍外的海仍為白色並以「分析範圍」輪廓標示。
* A 區（貢寮、龜山島）兩站圖都用整個 domain A：自己站的熱點與重現以全彩顯示，另一站的統計範圍以白色半透明面紗淡化（alpha 0.62），並畫橘色（#f4a300）「統計歸屬分界」（兩 anchor 的中垂大圓，`traps.domain_boundary`；海岸邊的單格缺口自動接起），圖例加「統計歸屬分界」。四季重現圖同樣處理，分析範圍填色改淡藍 #e1ecf5。
* 代表 TRAP 視窗：與新框同長寬比（預設經度 0.25° × 3＝0.75°、緯度 0.15° × 3＝0.45°，刻度 0.15°），含錨點、站名與海岸線；備援視窗 `[[0.25,3,0.05,9],[0.25,4,0.2,3],[0.5,2,0.3,2],[0.25,2,0.1,3]]`。新竹外海因視窗改變，所選 track 由 155507 改為 75154（其餘四站 track 不變）。
* gate：地圖範圍必須等於站所屬 flow domain 範圍（`configs/study_design.json` 的 bbox 四捨五入到 0.05°）、長寬比誤差 ≤ 3%、兩端標示、每軸 4–7 標示、純數字、陸地填色 `land_fill`、「統計歸屬分界」圖例；PNG 3780 × 2478。

### 2026-10-20（第四輪）南灣範圍改 120.15–121.65、等間距刻度 gate、用詞更名、圖內不印地點名稱

* 南灣（C）：四捨五入的 flow domain 範圍 120.15–121.60 會切掉右緣 2 欄資料，經度刻度也不等間距，改用 `concise_style.DOMAIN_LIMIT_OVERRIDES`：x 120.15–121.65（刻度 120.15、120.45 … 121.65，每 0.3°）、y 21.55–22.45（每 0.15°）；長寬比誤差 +2.3%（仍 ≤ 3%，gate 容差不必放寬）。
  原因記為「keep the full flow domain and even ticks」（版面紀錄 `extent_info.override`、manifest `map_extent[].extent_override`）；gate 的預期範圍同樣套用此覆寫表。
* gate 新增 `even_tick_failures`：每軸的標示刻度必須等間距（含兩端）。
* 用詞（`report_font.FIGURE_TEXT`，與正式圖組共用）：曲線通過頻率→「TRAP 曲線出現頻率」、核心頻率→「TRAP 核心出現頻率」、熱點格點圖例→「聚集熱區（TRAP 曲線出現頻率前 10%）」、核心圖例→「TRAP 核心最常出現的位置（前 10%）」、「研究站錨點」→「地點位置」；
  A 區總覽圖例「聚集熱區（各地點前 10%）」、四季重現色條「聚集熱區出現季數」（`concise_style.CONCISE_TERMS`）。熱點圖例固定兩列（`ncol` 2 或 3），圖例區與框不變。
* 簡要報告圖內不印地點名稱（`concise_style.show_site_names()`）：代表圖、A 區總覽、貢寮／龜山島圖只留錨點「＋」；代表圖側車 `site_label` 記為 `{"omitted": true}`。
* gate `terminology`：圖內文字不得含舊用詞（曲線通過頻率、熱點、核心頻率、研究站）與地點名稱，並要求各圖種的色條標籤與圖例字串等於指定用詞；manifest 每張圖記 `figure_texts`（移除標題後的圖內文字）。
* 分界線在海岸邊的單格缺口自動接起（`domain_boundary._bridge_short_gaps`，熱點圖、四季重現與總覽同一規則）。

### 2026-10-21 簡要報告：分析範圍外框只畫與站外有效海域相鄰的邊

* 原本以 `contour(level=0.5)` 沿站內遮罩（1 km 格網）描邊，靠岸格點採 fail-closed、遮罩邊緣離岸約 1 km，外框因此沿真實海岸線外側再描一圈，小島被框成八角形。
* `aggregate_figures.analysis_outline_paths`：以格點邊為單位，只保留「站內格點」與「站外但仍是有效海域的格點」之間的邊（另一側是陸地、遮罩、無效格點或格網外者不畫；與圖框重合或落在圖框外的邊不畫），串成折線，以 `LineCollection` 畫（0.7 pt、#4d4d4d、zorder 6.5）。
  有效海域＝`SiteFigureData.valid_sea_mask`（server 繪圖取 `grid_reference.npz` 的 `analysis_ocean_mask`；離線重繪取同 domain 各站遮罩聯集，兩者在四個 domain 逐格相同）；`None` 時取站內與搭檔站遮罩聯集。
* 結果：只有貢寮／龜山島的熱點與四季重現圖仍有線段（兩站遮罩之間的階梯狀分界，94 段，位於橘色統計歸屬分界線下方）；新竹外海、南灣、連江的兩種圖與 A 區總覽沒有線段。
* 圖例：`draw_analysis_outline` 沒有線段時回傳 `None`，熱點與 A 區總覽不放「分析範圍」線條圖例；四季重現的「分析範圍」是淡藍底色色塊，圖例項目保留，色塊的深灰框線只在有線段時加。
  版面紀錄 `concise.map_axes[0].analysis_outline_segments` 與 manifest 的 `analysis_outline_segments` 記錄畫出的邊數；gate `analysis_range`（`analysis_outline_legend_failures`）與報告檢查 `check_concise_results_report.py` 檢查「圖例／SVG 深灰色 #4d4d4d」與邊數一致。
* 測試：`tests/test_analysis_outline_20261021.py`（合成遮罩：貼陸地、包圍島嶼、圖框重合的邊不畫；與站外有效海域相鄰的邊畫；折線串接；圖例與 gate 規則）。

### 2026-10-21（第二輪）外框不描搭檔站分界、座標軸刻度統一、地點錨點改星號、百分位數改白話

* 分析範圍外框：`aggregate_figures.valid_sea_of` 把同 domain 搭檔站（貢寮／龜山島互為搭檔）的格點從「站外有效海域」扣掉，兩站分界已有橘色統計歸屬分界線，外框不再重複描。
  結果：11 張有外框的圖（熱點 5、四季重現 5、A 區總覽 1）的外框線段都是 0（原貢寮／龜山島各 94 段）；熱點與總覽圖例不放「分析範圍」線條，四季重現保留淡藍色塊圖例但不加深灰框。
  gate `analysis_outline_legend_failures` 預設預期邊數 `EXPECTED_OUTLINE_SEGMENTS = 0`（`expected_segments=None` 時只檢查圖例／SVG 對應規則，供單元測試）；報告檢查同樣要求 manifest 的 `analysis_outline_segments` 為 0。
* 刻度（共用處 `concise_style.axis_ticks`／`style_map_axes`）：標示一律 2 位小數（`AXIS_LABEL_DECIMALS`）；挑選等間距刻度時取標示數最接近 5.5 者、同樣接近取標示較少（間距較大）者，不再給「整 0.1 度」加分。
  結果：熱點、四季重現、A 區總覽 x 軸 6 個、y 軸 4 個，間距 0.3°，從下／左緣起算（連江 y 25.75、26.05、26.35、26.65；南灣 21.55、21.85、22.15、22.45）；代表性 TRAP 圖仍是 0.75° × 0.45°、間距 0.15°。地圖範圍（xlim、ylim）不變。
  gate 新增 `map_axis_standard`（`map_axis_standard_failures`）。
* 地點錨點：`concise_style.draw_site_star`／`site_star_handle`，畫法逐項取自 Lagrangian-Ensemble-Backtracking `native_runner/figures.py`（`marker="*"`、`markersize=13`、`markerfacecolor="#ffd92f"`、`markeredgecolor="black"`、`markeredgewidth=0.7`、`zorder=8`；圖例 `markersize=12`），`clip_on=False`，同位置不重複畫。
  只在 concise 模式啟用（`_prepare_map_axis`、`draw_partner_overlay`、`plot_domain_a_overview`），正式圖組（無 concise 選項檔）仍用十字。版面紀錄 `concise.markers`（`site_stars`／`cross_markers`／`legend_stars`）與 gate `site_marker` 檢查星號畫法、圖例大小、無十字、SVG 有 `#ffd92f`。
* 文字：生成器 INTRO「置於章末」→「置於本節末」（`concise_user_edits.json` 的 definition-paragraph-20261020，使用者 2026-10-21 同意）；代表性 TRAP 說明段與結論表的「第 90 百分位數」改為「本地點 90% 之結構壽命為 N 小時以內，此結構存在 M 小時，屬壽命較長之案例；…」與「90% 之結構壽命為 N 小時以內」；
  `check_concise_results_report.py` 新增 `check_wording_20261021`：精簡版（MD、DOCX 本文／儲存格／替代文字／頁尾／文件屬性）不得出現「百分位」「章末」。
* 重繪：本機以 `$HOME/scratch/outline/` 的 `render11.py`（11 張）→ `stage1.py`（去標題、600 dpi）→ `deliver_v34.py`（寫入交付目錄、更新版面紀錄與 manifest）完成；5 張代表性 TRAP 圖需要起始時刻 OCM 流場與重算 s1，本機沒有，仍為舊版（十字），待在 .203 以新版程式重繪。
* 測試：`tests/test_analysis_outline_20261021.py`（更新）、`tests/test_axes_markers_wording_20261021.py`（新增）。備份：`outputs/_archive/concise_v33_20261021/`。

### 2026-10-22 簡要報告：標準圖框（同類圖五個地點跨度與刻度一致、標示 1 位小數）

* 規格表只在 `concise_style.FRAME_SPECS`（取框規則 `standard_frame`、刻度 `axis_ticks`、gate 的 `map_axis_standard` 都讀這張表，各圖不自行寫死數字）：
  * `map`（A 區總覽、聚集熱區、四季重現共 11 張）：經度 1.6°（每 0.4°，5 個刻度）、緯度 1.0°（每 0.2°，6 個刻度）；
  * `zoom`（代表性 TRAP 放大圖 5 張）：經度 0.8°（每 0.2°，5 個刻度）、緯度 0.5°（每 0.1°，6 個刻度）。
* 取框規則：新框四邊落在 0.1° 的整數倍上並完整包住原圖框（大圖：flow domain 四捨五入到 0.05°［含南灣覆寫］的範圍；放大圖：原 0.75° × 0.45° 視窗）。有兩種取法時
  （原圖框的邊已在 0.1° 格線上），取「流場資料外框（格點範圍）距新框四邊的最小留白」較大者；差在 0.001° 內視為平手，往東、往北擴。
  結果：A 區／貢寮／龜山島 121.3–122.9 × 24.6–25.6；新竹外海 119.7–121.3 × 24.3–25.3；南灣 120.1–121.7 × 21.5–22.5；連江 119.2–120.8 × 25.7–26.7。
  放大圖：貢寮 121.5–122.3 × 24.9–25.4；龜山島 121.7–122.5 × 24.7–25.2；新竹外海 120.0–120.8 × 24.3–24.8；南灣 120.2–121.0 × 21.7–22.2；連江 119.7–120.5 × 25.9–26.4。
* 標示一律 1 位小數（`AXIS_LABEL_DECIMALS = 1`），首末刻度＝框邊並標出數值；`axis_ticks` 只接受規格表內的跨度（其他跨度 `ValueError`）。
* 放大圖：`find_example_window` 照舊選取視窗（錨點、海岸線、軌跡與曲線放得進、保留區、長寬比），入選後由 `standardize_example_window` 擴成標準圖框（選項檔 `example_zoom.window_lattice.standard_frame = "zoom"`），
  原視窗存於 `window.lattice_window`；選取在擴框之前，所以哪個 track 被選上不變。
* 圖檔尺寸（6.3 × 4.4 英吋）、軸框位置、字型、色階、色條、星號、圖例都不變，軸框固定為 4.5 × 2.98 英吋，因此長寬比誤差由緯度決定（經度 × cos 緯度 ／ 緯度 相對於軸框）：
  南灣 −1.8%、新竹外海 −3.8%、貢寮／龜山島 −4.1%、連江 −4.9%（圖面略為南北拉長，不改變尺寸與軸框）。`STANDARD_FRAME_ASPECT_TOLERANCE = 0.05` 取代原 3%，gate 另要求紀錄的誤差等於由圖框算出的值。
* gate：`map_axis_standard`（跨度、刻度數、間距、首末刻度、1 位小數、同類圖各地點一致）、`analysis_range`（大圖的圖框＝規則算出的標準圖框）、`hourly_example`（放大圖視窗＝0.8° × 0.5°、在 0.1° 格線上且包住 `window.lattice_window`）。
* 重繪：11 張大圖本機（`$HOME/scratch/lay12/render11b.py` → `stage1.py` → `deliver_maps.py`）；5 張放大圖在 .203（`/home/mustlab/Workspace/TRAPs_render_20261022/`，`nice` 執行，只讀 aggregate）→ `stage_ex.py` → `deliver_ex22.py`。
  備份：`outputs/_archive/concise_v41_20261022/`。測試：`tests/test_concise_figures_20261016.py`、`tests/test_axes_markers_wording_20261021.py`、`tests/test_hourly_concise_20261016.py`（更新）。

### 2026-10-22 第二輪 簡要報告：11 張大圖改採 LBT run09 axes_unified 規格並加「流場範圍邊界」

* 範圍（取代上一輪「向外擴到 0.1° 格線」的 `map` 規則）：`concise_style.FRAME_SPECS["map"]` 改為經度 1.8°、緯度 1.2°，間距都是 0.2°（經度 10 個、緯度 7 個刻度），各地點範圍直接指定在 `FRAME_SPECS["map"]["extents"]`
  （來源註記 `MAP_EXTENTS_SOURCE`：Lagrangian-Ensemble-Backtracking `成果報告_run09/_verify/axes_unified_run09.txt`）：貢寮／龜山島／圖1 121.2–123.0 × 24.4–25.6；新竹外海 119.6–121.4 × 24.2–25.4；南灣 120.0–121.8 × 21.4–22.6；連江 119.0–120.8 × 25.6–26.8。
  `map_frame(lon, lat)` 取得指定範圍並記錄是否完整包住格點範圍；`standard_frame` 不再適用 `map`（`ValueError`），仍用於放大圖 `zoom`（規格、檔案、manifest 條目都不變，未重繪）。
* 主圖框：`figure_standard.CONCISE_MAP_AXES`＝寬 2.98 × 1.8 × cos 24.1° ／ 1.2 ≈ 4.080 英吋、高 2.98 英吋（五個地點相同，長寬比誤差 ±2%：貢寮／龜山島／A 區 −0.71%、新竹外海 −0.55%、連江 −1.70%、南灣 +1.58%）；
  色條 0.12 英吋寬、間距 0.093 英吋（`CONCISE_MAP_CBAR`）。版面類別 `map_lbt`（熱點、四季重現、A 區總覽）；`example` 仍為 `map1`（4.5 × 2.98）。`MAP_FRAME_ASPECT_TOLERANCE = 0.02`（大圖）、`STANDARD_FRAME_ASPECT_TOLERANCE = 0.05`（放大圖）。
  畫布 6.3 × 4.73 英吋，去標題後裁掉 0.6 英吋成 6.3 × 4.13（3780 × 2478 px、600 dpi）；其他仍用 `concise_map_boxes()` 在 concise 模式選軸框。
* 流場範圍邊界：`aggregate_figures.draw_flow_domain_boundary`（`flow_domain_extent`＝規則經緯度網格的格緣矩形）；深藍色 `#1f3a5f`、線寬 1.0 pt、長虛線（6.5／2.2 pt），zorder 6.6，圖例「流場範圍邊界」排在最後；版面紀錄 `concise.map_axes[].flow_boundary`。
  流場範圍外的海維持白色；陸地填色與海岸線鋪滿整個圖框。
* LBT 底圖樣式（只用旗標套在 11 張大圖，放大圖不受影響）：無經緯度格線、海岸線黑色 0.35 pt（`LBT_COAST_COLOUR`／`LBT_COAST_WIDTH_PT`）、圖例有淺灰外框（`legend_below(..., framed=True)`，以軸框中心置中）。對照表見交付報告。
* gate（`scripts/check_concise_figures.py`）：`map_axis_standard`（跨度 1.8／1.2、刻度數 10／7、指定範圍）、`analysis_range`（範圍＝指定範圍並完整包住流場範圍、主圖框＝LBT 比例、誤差 ±2%）、`same_size_boxes`（大圖 11 張一組、放大圖 5 張一組，畫布 16 張相同）、
  新增 `flow_boundary`（圖例、版面紀錄、深藍長虛線 1.0 pt、四邊在圖框內且等於網格外框、與近岸帶／統計歸屬分界／海岸線顏色不同、SVG 有 #1f3a5f 與 stroke-dasharray；放大圖不要求）。`check_concise_results_report.py` 的圖例檢查同步（大圖必須有、放大圖不得有）。
* 重繪：本機（沿用 lay12 的 `render11b.py` 繪圖，輸出 `$HOME/scratch/lay13/raw11` → `stage1.py` 去標題與 600 dpi 到 `lay13/st` → `deliver_maps13.py` → `scripts/copy_concise_figures_20261016.py` → `generate_concise_results_md.py` → `build_concise_results_report.py` → `check_concise_results_report.py`）。
  備份：`outputs/_archive/concise_v42_20261022/`。測試：`tests/test_lbt_map_style_20261022.py`（新增）、`tests/test_concise_figures_20261016.py`、`tests/test_axes_markers_wording_20261021.py`、`tests/test_hourly_concise_20261016.py`（更新）。

### 2026-10-22 第三輪 簡要報告：5 張代表性 TRAP 放大圖的底圖樣式改為與 11 張大圖（LBT）一致

* 底圖樣式（`example_figures.render_example_zoom`，concise 模式以 `concise_style.concise_enabled()` 開關；正式圖組與非 concise 呼叫不變）：`_prepare_map_axis(..., lbt_style=True)`（黑色海岸線 `LBT_COAST_WIDTH_PT`＝0.35 pt、不畫經緯度格線）、
  色條寬 0.12 英吋且與主圖框間距 0.093 英吋（`CONCISE_ZOOM_CBAR`）、圖例 `legend_below(..., framed=True, center_x_inch=主圖框中心)`（淺灰外框、白底、不透明度 0.8、置中於主圖框下方）。
  不變：放大圖範圍 0.8° × 0.5°（各案例視窗沿用 manifest）、刻度（經度 0.2°、緯度 0.1°，1 位小數）、軸標題、s1 色階 −15～15、流速箭頭與 1 m/s 參考箭頭、TRAP 曲線與核心、+N h 曲線、核心路徑、近岸 10 km 帶、星號；放大圖不畫「流場範圍邊界」。
* 主圖框：評估結果是**不能與大圖用同一個 4.08 × 2.98 英吋**（0.8° × 0.5° 的視窗在 4.08 英吋框的長寬比誤差 +4.9%（連江）至 +8.4%（南灣），超過 ±2%），改用與 LBT 相同的 24.1° 參考緯度規則：
  `figure_standard.CONCISE_ZOOM_AXES`＝寬 2.98 × 0.8 × cos 24.1° ／ 0.5 ≈ 4.352 英吋、高 2.98 英吋；左緣（0.91）、下緣、框高、色條寬、間距、圖例規則都與大圖相同，只有寬度依各自的視窗跨度不同（大圖 4.08、放大圖 4.352）。
  長寬比誤差：貢寮 −0.83%、龜山島 −0.67%、新竹外海 −0.35%、連江 −1.66%、南灣 +1.62%（原軸框 4.5 × 2.98：−4.09%、−3.93%、−3.62%、−4.89%、−1.73%）；放大圖誤差上限由 5% 收緊為 ±2%
  （`concise_style.ZOOM_FRAME_ASPECT_TOLERANCE`；規格表以外的圖框仍 5%）。版面類別 `map_lbt_zoom`（`CONCISE_KIND_LAYOUT["example"]`）。1 m/s 參考箭頭的尖端與保留區對齊右緣（距右緣的英吋數不變，`example_figures.zoom_key_geometry`）。
* 視窗選取不變（`EXAMPLE_WINDOW_DEFAULTS["axes_aspect"]`＝4.5 / 2.98 仍只用於 `find_example_window` 的候選篩選，所以不會改變入選的 track）；`standardize_example_window` 與 sidecar `window.aspect_error` 的長寬比誤差改為相對 `CONCISE_ZOOM_AXES`。
* gate（`scripts/check_concise_figures.py`）：新增 `base_style`（16 張圖都：SVG 有黑色 0.35 pt 海岸線描邊且沒有舊的 #263238 0.6 pt、沒有格線描邊 #b0b0b0、色條寬 0.12／間距 0.093 英吋且與主圖框等高、圖例有淺灰外框〔白底、不透明度 0.8〕且中心對齊主圖框中心並在畫布內）；
  `same_size_boxes` 另要求大圖組與放大圖組的主圖框左緣、上緣、框高與色條上緣、寬、高、間距相同（`_map_vs_zoom_box_failures`；寬度不同是預期的）；`hourly_example` 的視窗長寬比誤差等於由 `CONCISE_ZOOM_AXES` 推得的值且在 ±2% 內。其他檢查沒有放寬。
* 重繪：在 .203 的新目錄 `/home/mustlab/Workspace/TRAPs_render_20261022b/`（`nice`、只讀既有 aggregate，約 6 分鐘）→ 本機 `stage_ex.py`（去標題、600 dpi）→ `deliver_ex43.py`（先驗證案例不變再寫入；sidecar 只有 `window.aspect_error` 改變）→
  `scripts/copy_concise_figures_20261016.py` → `generate_concise_results_md.py`（文字沒變）→ `build_concise_results_report.py` → `check_concise_results_report.py`（problems 0）→ `check_concise_figures.py --expected-total 16 --spatial-root`（通過）。
  11 張大圖的檔案與 manifest 條目逐位元不變。備份：`outputs/_archive/concise_v43_20261022/`。測試：`tests/test_lbt_map_style_20261022.py`（新增放大圖底圖樣式與 gate 測試）、`tests/test_concise_figures_20261016.py`、`tests/test_hourly_concise_20261016.py`（更新）。

### 2026-10-27 簡要報告：地點星號縮小為 LBT 列印大小；新增「地點位置周邊之聚集熱區」說明與表7

* 星號：LBT 的星號（`native_runner/figures.py` 3205–3265 行：地圖 `markersize=13`、`markeredgewidth=0.7`，圖例 `markersize=12`）在 LBT 報告中已按列印比例縮小（成果報告 docx 的圖以 6.27 英吋寬貼入、比例 1.0，
  縮小發生在繪圖時：6.27／8.9267＝0.7024），所以 TRAPs（圖檔以 6.3 英吋畫布原尺寸貼入）直接採用列印後的點數：`concise_style.LBT_STAR_PRINT_SCALE`＝0.70239，
  `SITE_STAR_STYLE` 的 `markersize`＝9.131、`markeredgewidth`＝0.4917，`SITE_STAR_LEGEND_MARKERSIZE`＝8.429（舊值 13／0.7／12）。集中在 `draw_site_star` 與 `site_star_handle`，16 張圖全部套用；
  `site_marker_failures` 另檢查圖例星號邊線寬。
* 重繪：11 張大圖在本機重繪；5 張放大圖在 .203 的新目錄 `/home/mustlab/Workspace/TRAPs_render_20261027/`（`nice`、只讀既有 aggregate）。寫入前先驗證案例與範圍不變（sidecar、視窗、note、manifest 的 `example_*` 與 `removed_titles` 相同；
  版面紀錄只差星號 `markersize`／`edge_width`；像素差異只在星號周圍）。備份：`outputs/_archive/concise_v44_20261027/`。
* 表7：`generate_concise_results_md.py` 由 `outputs/server/r1_hourly/aggregate_report/spatial/<site>/spatial_all.npz` 與 `configs/study_design.json` 的 `anchor_lonlat` 計算（球面近似、R＝6371 km、格點中心）。
  最近聚集熱區與最近 TRAP 核心距離用圖上顯示的濾除後遮罩（`hotspot_curve_mask_filtered`／`hotspot_core_mask_filtered`，1 位小數）；10／20 km 內聚集熱區格點比例用 `hotspot_curve_mask`，
  分母為範圍內有效格點（`site_mask` 且 `curve_pass_frequency` 有限），四捨五入到整數百分比。五段聚集熱區說明各加一句距離，綜合說明與結論開頭加說明段落與表7。
  `check_concise_results_report.py` 新增 `check_vicinity`（以另一套向量算法獨立重算，核對表7與說明段落的分組與範圍）；表號 1–7 連號。測試：`tests/test_vicinity_20261027.py`（新增）。

### 2026-10-27（v45）簡要報告：聚集熱區與核心黑點改畫未濾除遮罩（所見即所得），表7 距離與說明句同步

* 圖面：簡要報告（concise 模式）的聚集熱區圖（5 張）與圖1 總覽，紅色熱區格與黑色核心點都畫統計定義本身的未濾除遮罩（`hotspot_curve_mask`、`hotspot_core_mask`，各地點前 10%），
  不再濾除 4-連通格數少於 `hotspot_min_cells`（3）的小塊與孤立黑點。旗標：`concise_style.concise_enabled()`（`aggregate_figures.render_hotspots`、`scripts/plot_domain_a_overview.py`）；
  正式圖組（未啟用 concise）仍畫 `*_filtered`，行為不變。選擇函式與版面紀錄欄位在 `concise_style.hotspot_masks_for_figure`／`hotspot_mask_record`；
  版面紀錄 `concise.map_axes[0].hotspot_masks`（`basis: "unfiltered"`、`curve_cells_drawn`、`core_cells_drawn`）與 manifest 圖條目 `hotspot_masks`。
  其他圖層、色階、範圍、尺寸、星號、底圖樣式不變（交付時逐項驗證：版面紀錄只差 `hotspot_masks`，manifest 條目只差檔案 sha256／大小與 `hotspot_masks`）。
* 四季重現圖（5 張）：本來就是未濾除遮罩（`seasonal_recurrence.load_site_seasons` 由 `spatial_<季>.npz` 重算各季前 10% 並與存檔 `hotspot_curve_mask` 逐格核對），沒有濾除，未重畫。
  代表性 TRAP 放大圖不畫聚集熱區，未重畫。
* 表7 與各地點說明段的距離句：「最近聚集熱區距離」「最近 TRAP 核心最常出現的位置距離」改用未濾除遮罩（`generate_concise_results_md.DIST_MASKS`、`check_concise_results_report.VIC_DIST_MASKS`）；
  10／20 公里比例本來就用 `hotspot_curve_mask`。結論段「皆在 N 公里以內」的 N 由資料決定（現為 4）。
* 代表性 TRAP 說明段的「起始核心位於聚集熱區內／外、距聚集熱區約 X 公里」：舊值來自選案時的濾除後遮罩（sidecar `in_hotspot_area`、`start_dist_to_hotspot_km`），
  改依未濾除遮罩判定（`hourly_facts_20261016._start_unfiltered`，欄位 `start_in_hotspot_unfiltered`、`start_dist_to_hotspot_unfiltered_km`；同一算法在濾除遮罩上必須重現 sidecar 值）。
  案例選取（track id、時刻、視窗）不變；選案邏輯本身仍用濾除後遮罩（`aggregate.py` 的 `hotspot_source`），沒有更動。
* gate：`check_concise_figures.py` 新增 `hotspot_masks`（版面紀錄基準為 unfiltered、記錄格數等於 SVG 實際畫出的紅格與黑點數；給定 `--spatial-root` 時把 SVG 畫出的格點還原成經緯度，與 npz 未濾除遮罩逐格一致，圖1 為貢寮＋龜山島各取自己站內遮罩）。
  `check_concise_results_report.py` 的距離重算改用未濾除遮罩，另以獨立算法核對代表性 TRAP 的起始核心判定。測試：`tests/test_unfiltered_hotspots_20261027.py`（新增）、`tests/test_vicinity_20261027.py`（更新）。
* 重繪：本機（`render11b.py` 只畫 `hot,ov` → 去標題與 600 dpi → 交付腳本驗證只差 `hotspot_masks`）→ `copy_concise_figures_20261016.py` → `generate_concise_results_md.py` → `build_concise_results_report.py` → `check_concise_results_report.py`（problems 0）→ `check_concise_figures.py --expected-total 16 --spatial-root`（通過）。
  備份：`outputs/_archive/concise_v45_20261027/`。

### 2026-10-27（v46）簡要報告：聚集熱區圖與圖1 的核心黑點縮小，紅色熱區格不再被蓋住

* 圖面：核心黑點（`hotspot_core_mask`）直徑由 2.0 pt 加 0.4 pt 白邊（外徑 2.4 pt，大於約 1.6 pt 的格寬，面積約為一格的 175%）改為 1.2 pt、無邊線（約一格寬的 3/4、佔格面積約 44%）；
  圖例黑點由 5.0 pt（含 1.0 pt 白邊）依同比例縮為 3.0 pt、無邊線，與圖上同樣式。常數與樣式函式集中在 `concise_style.HOTSPOT_CORE_DOT_PT`／`HOTSPOT_CORE_DOT_EDGE_PT`／`HOTSPOT_CORE_LEGEND_PT`、
  `hotspot_core_dot_kwargs()`／`hotspot_core_legend_kwargs()`，`aggregate_figures.render_hotspots` 與 `scripts/plot_domain_a_overview.py` 的 concise 分支共用；正式圖組（未啟用 concise）仍是舊算法與舊大小。
  選擇依據：與 0.8 pt（約 40%）比較，0.8 pt 在列印尺寸下已近似細點；1.2 pt 仍清楚可見、紅格大多露出。資料、統計、文字、版面紀錄皆不變（版面紀錄無差異，manifest 只差檔案 sha256／大小）。
* 重繪：本機 `render` 只畫 `hot,ov` → 去標題與 600 dpi → 交付腳本驗證 → `copy_concise_figures_20261016.py` → `generate_concise_results_md.py`（MD 逐字不變）→ `build_concise_results_report.py` → `check_concise_results_report.py`（problems 0）→
  `check_concise_figures.py --expected-total 16 --spatial-root`（通過）。測試：`tests/test_core_dot_size_20261027.py`（新增，鎖定新大小與正式圖組不變）。備份：`outputs/_archive/concise_v46_20261027/`。

### 2026-10-28（v47）簡要報告：代表性 TRAP 圖一律不用白邊；各地點結論的難懂說法改為白話

* 圖面（`example_figures.render_example_zoom`，只動簡要報告的放大圖；正式 `render_example` 的白色描邊不變）：拿掉起始曲線的白色 halo（`path_effects.withStroke`）、起始核心與後續核心小點的白邊（`markeredgewidth=0`，不再有 `markeredgecolor="white"`）、
  1 m/s 參考箭頭與文字的白色描邊；圖例的起始曲線與核心記號與圖上一致（核心 6.5 pt、無邊線；圖例原為 5.5 pt 加 matplotlib 預設的 1 pt 同色邊線，看到的直徑同為 6.5 pt，所以圖例看起來大小不變）。
  線寬（1.8／3.2 pt）、`markersize`（6.5／3.5 pt）、顏色、流速箭頭、版面與軸都不變；`ZOOM_KEY_HALO_PT` 只留作參考箭頭文字外框量測的餘裕（保留區判定與版面紀錄不變），不再對應任何繪出的描邊。
  日數標籤（`annotate` 的半透明白底，`edgecolor="none"`）是文字底色，不是白邊，未動。
* gate：`check_concise_figures.py` 新增 `example_no_white_edge`（`example_white_edge_failures`）：代表性 TRAP 的 SVG 不得有任何白色 `stroke`（matplotlib 的 halo、白邊標記、白色描邊都以白色 stroke 輸出；圖例外框為 `#cccccc`、白色只作填色，不受影響）。
* 重繪：5 張放大圖在 .203 的新目錄 `/home/mustlab/Workspace/TRAPs_render_20261028/`（`nice`、只讀既有 aggregate；程式、設定、海岸線資料同步到該目錄，沒有動 server 上任何既有檔案）→ 本機 `stage_ex.py`（去標題、600 dpi）→ `deliver_ex47.py`
  （先驗證案例、視窗、sidecar、版面紀錄、參考箭頭紀錄皆不變，只差檔案 sha256／大小與 PNG 像素檢查）→ `copy_concise_figures_20261016.py` → `generate_concise_results_md.py` → `build_concise_results_report.py` → `check_concise_results_report.py`（problems 0）→
  `check_concise_figures.py --root outputs/concise_figures_20261016 --expected-total 16 --spatial-root outputs/server/r1_hourly/aggregate_report/spatial`（通過）。
* 用語（`generate_concise_results_md.py`、`check_concise_results_report.py` 同步）：
  連江「結構位移較快，位移速度中位數為每日 11.1 公里，為五個地點最高」→「與其他地點相比，壽命較長之吸引結構較少停留於原處，匯聚位置隨時間變動較大」（表6「潮汐尺度吸引」列與整體結論第 5 點；不寫速度數字，全文不得出現「位移速度」）；
  南灣「部分潮汐尺度吸引可維持一日以上」→「部分吸引結構可維持一日以上」（表5 與整體結論第 4 點）；「潮汐尺度吸引」「壽命 3 小時以上之比例」於貢寮圖4 說明段第一次使用前定義（`EX_FIRST`）。
  檢查腳本新增：全文不得有「位移速度」「部分潮汐尺度」；連江、南灣之白話句只出現在各自表格與整體結論；定義句恰出現一次且在第一次使用前。
* 測試：`tests/test_example_no_white_edge_20261028.py`（新增：合成資料畫放大圖後讀 SVG，確認起始曲線無 halo、核心與後續核心無白邊、參考箭頭無白色描邊、圖例與圖上一致、gate 函式行為、已交付 5 張 SVG 無白色 stroke、正式代表圖函式仍保留白色描邊；用語三項）。
  備份：`outputs/_archive/concise_v47_20261028/`。

### 2026-11-02（v64）簡要報告：聚集熱區圖與圖1 的黑點只標示聚集熱區內與地點位置 10 公里內者，並畫 10 km 圓圈（v65 已撤除，見下節；本節保留作歷史紀錄）

* 規則（`concise_style.hotspot_core_dots_mask`）：圖上黑點 ＝（`hotspot_core_mask` ∩ `hotspot_curve_mask`）∪（`hotspot_core_mask` 中距地點位置〔星號〕≤ 10 km 的格點）。距離以格點中心計，球面 R ＝ 6371 km（haversine，與 `generate_concise_results_md.great_circle_km` 相同；gate 以向量／atan2 另算）。
  黑點本身的定義（核心出現頻率前 10%）、大小（1.2 pt、無邊）、顏色、紅色聚集熱區（完整的未濾除遮罩）都不變；正式圖組不套用。
* 圓圈：以星號為圓心、球面半徑 10 km（`site_radius_circle_lonlat`，361 點），深灰 `#555555`、0.6 pt、虛線 `(0, (3, 2))`、`zorder` 4.6（高於熱區色塊、低於黑點與星號）、無白邊、無圈內文字；SVG 群組 id ＝ `site_radius_circle`。
  顏色不用 `#4d4d4d`：該色專用於分析範圍外框，gate 與報告檢查以它判斷外框有無。圖例新增「地點 10 km 範圍」（`concise_style.CONCISE_TERMS["site_radius_legend"]`），樣本線與圖上圓圈同色同寬同虛線。
* 兩個星號：各站圖只對本站星號套用（黑點保留與圓圈）。A 區的龜山島、貢寮圖上另一個站的星號落在該站的統計範圍（白色淡化層）內，那裡本來就不畫黑點，所以不加圓圈，以免讀者誤以為該處也保留了黑點；
  圖1 總覽兩個星號都套用（圓圈 2 個）。版面紀錄 `concise.map_axes[0].hotspot_masks` 新增 `core_dots`（規則與半徑）與 `near_site`（圓心清單、半徑）。
* 圖例：圓圈項緊接在星號（圖1）或核心黑點（各站圖）之後；欄數由項目數決定（< 5 項 2 欄、5–6 項 3 欄、≥ 7 項 4 欄；圖1 現為 7 項 4 欄），皆維持 2 列，所以圖例框高度不變，v50 的圖例間距 gate（`lifetime_legend_spacing`）仍通過，圖1 圖例框仍在畫布內。
* gate（`scripts/check_concise_figures.py`）：`hotspot_masks` 改為驗證 SVG 黑點等於新規則的遮罩（與完整核心遮罩相同時會提示「沒有依 v64 規則」）；新增 `site_radius_circles`：圓圈群組個數＝圓心數、群組內只有 1 條路徑（無白邊）、無文字、樣式正確，
  每個頂點都在以星號（設計檔錨點＝版面紀錄圓心＝圖上星號位置）為圓心、10 km（±0.15 km）的圓上，圖例有「地點 10 km 範圍」。`EXPECTED_TERMS` 的圖例清單加上這一項。
* 正文：定義段在「黑點與聚集熱區重疊之處…」之後補「為使圖面清楚，圖中黑點僅標示位於聚集熱區內者，以及位於地點位置 10 公里範圍內者（該範圍以圓圈標示）。」（`generate_concise_results_md.INTRO`、`concise_user_edits.json` 的 `definition-paragraph-20261020`、`check_concise_results_report.USER_DEFINITION_20261020`）。
  表7 與各地點距離句仍用完整核心遮罩，數值不變（3.8／2.9／2.5／4.0／2.5）；生成器與檢查腳本各自驗證最近的核心格 ≤ 10 km 且被畫出（`nearest_core_drawn`）。
* 重繪：本機 `render.py`（只畫 `hot,ov`）→ `stage.py`（去標題、600 dpi）→ `deliver.py`（先驗證版面紀錄只差 `core_cells_drawn`／`core_dots`／`near_site`／圖例文字，manifest 條目只差檔案 sha256／大小與同樣欄位）→ `copy_concise_figures_20261016.py` → `generate_concise_results_md.py` →
  `build_concise_results_report.py` → `check_concise_results_report.py`（problems 0）→ `check_concise_figures.py --root outputs/concise_figures_20261016 --expected-total 16 --spatial-root outputs/server/r1_hourly/aggregate_report/spatial`（通過）。
  黑點數（舊→新）：龜山島 439→225、貢寮 698→343、新竹 1,190→539、南灣 1,385→632、連江 959→477、圖1 1,137→568。備份：`outputs/_archive/concise_v64_20261102/`。測試：`tests/test_dot_rule_20261102.py`（新增），`test_unfiltered_hotspots_20261027.py`、`test_core_dot_size_20261027.py`、`test_concise_figures_20261016.py` 更新。

### 2026-11-03（v65）簡要報告：核心點改雙色呈現（撤除 v64 的 10 km 規則），圖例重新設計

* 規則（`concise_style.hotspot_core_dot_split(curve_mask, core_mask)`）：畫出全部核心格點（`hotspot_core_mask`，前 10%），單一規則、不依地點位置篩選。核心 ∩ 聚集熱區（`hotspot_curve_mask`）為黑點 `#111111`，其餘（核心 − 聚集熱區）為中灰 `#7f7f7f`（`HOTSPOT_CORE_OUTSIDE_COLOUR`，zorder 4.9，略低於黑點）；
  大小 1.2 pt、無邊線，圖例符號 3.0 pt。紅色聚集熱區（完整的未濾除遮罩）不變；正式圖組不套用。黑／灰點數：龜山島 186／253、貢寮 328／370、新竹 523／667、南灣 628／757、連江 453／506、圖1 514／623（＝龜山島＋貢寮）。
* 撤除 v64：10 km 圓圈、「地點 10 km 範圍」圖例、近距離保留規則、`HOTSPOT_NEAR_SITE_KM`、`hotspot_core_dots_mask`、`draw_site_radius_circle`、`site_radius_handle`、`site_radius_circle_lonlat`，生成器與檢查腳本的 `NEAR_SITE_KM` 與其 assert，gate 的 `site_radius_circles`。
  版面紀錄 `hotspot_masks.core_dots` 改記 `{rule: core_black_inside_hotspot_grey_outside, black_colour, grey_colour, black_cells_drawn, grey_cells_drawn}`，`core_cells_drawn` ＝ 黑點數 ＋ 灰點數。
* 圖例（聚集熱區圖 5 張與圖1 一致）：順序為聚集熱區、「TRAP 核心最常出現的位置（前 10%）」（黑點）、「同上，位於聚集熱區外」（灰點）、（統計歸屬分界、地點位置）、近岸 10 km 帶、流場範圍邊界；字級 8 pt、外框樣式與其他圖相同。
  圖例框左右兩端限制在「主圖框左緣 ～ 色條（含刻度與標籤）右緣」之內並置中於該範圍（`figure_standard.legend_span_inch`、`legend_below(span_inch=...)`；量測用 100 dpi 字形寬度、存成 SVG（72 dpi）每側約差 0.025 in，所以兩側預留 `LEGEND_SPAN_SAFETY_INCH` ＝ 0.04 in，超出即 `ValueError`）。
  欄數 `ceil(項目數 / 3)`（`aggregate_figures.concise_legend_columns`），各站圖 5–6 項（3 列 2 欄）、圖1 7 項（3 列 3 欄），三列。
* 畫布：圖例由兩列變三列，熱區圖與圖1 的畫布高度加 `CONCISE_LEG3_EXTRA_INCH` ＝ 0.1667 in（一列的列距）：6.3 × 4.2967 in（裁掉標題帶後；其他圖仍 4.13），主圖框與色條整體上移 0.1667 in，頂端留白不變；圖例框底緣到畫布底緣、x 軸標籤到圖例框頂緣與代表性 TRAP 圖一致（誤差 ≤ 0.02 in）。
  新版面類別 `map_lbt_leg3`（`figure_standard.CONCISE_KIND_LAYOUT` 的 `hotspots`、`domainA_overview`；四季重現仍 `map_lbt`）。6 張熱區圖互相同尺寸、同軸框位置。
* gate（`scripts/check_concise_figures.py`）：`hotspot_masks` 驗證 SVG 黑點 ＝ 核心 ∩ 聚集熱區、灰點 ＝ 核心 − 聚集熱區、顏色正確、黑＋灰 ＝ 完整核心遮罩、沒有 10 km 圓圈；新增 `legend_range`（圖例框在主圖框左緣到色條右緣之內並置中、3 列、8 pt，並與版面紀錄 `concise.legend` 核對）、`legend_spacing`（三列圖例的留白與代表性 TRAP 圖一致）；
  `same_size_boxes` 改為 leg3 與標準兩組畫布；`EXPECTED_TERMS` 的圖例清單改為灰點圖例。
* 正文：定義段刪除 v64 的規則句，改為「位於聚集熱區內者以黑點表示，代表 TRAP 曲線經常經過且匯聚中心亦經常位於該處，位於聚集熱區外者以灰點表示。」（`generate_concise_results_md.INTRO`、`concise_user_edits.json` 的 `definition-paragraph-20261020`、`check_concise_results_report.USER_DEFINITION_20261020`）。
  表7 與各地點距離句仍用完整核心遮罩，數值不變（3.8／2.9／2.5／4.0／2.5、「皆在 4 公里以內」）；檢查腳本改驗證最近的核心格在圖上以黑點或灰點畫出（`nearest_core_drawn`、`nearest_colour`）。
* 重繪：本機 `render.py`（只畫 `hot,ov`）→ `stage.py`（去標題、600 dpi）→ `deliver.py`（驗證非圖例文字與 v64 相同、僅整體上移 16.67 px）→ `copy_concise_figures_20261016.py` → `generate_concise_results_md.py` → `build_concise_results_report.py` → `check_concise_results_report.py`（problems 0）→ gate（16 張通過）。
  備份：`outputs/_archive/concise_v65_20261103/`。測試：`tests/test_dot_rule_20261102.py` 改寫（鎖定雙色規則），`test_unfiltered_hotspots_20261027.py`、`test_core_dot_size_20261027.py`、`test_concise_figures_20261016.py`、`test_hourly_concise_20261016.py`、`test_lifetime_figures_20261028.py` 更新。
