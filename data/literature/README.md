# TRAPs 文獻備查與紅框導讀

> 註：本文以 `路徑` 純文字標示的檔案（如 outputs/、data/literature/original/、data/literature/annotated/、docs/assets/ 等）僅存於本機，未納入版本控制。

本資料夾保存 TRAP 規劃文獻與 25 點逐時平均的補充依據。`original/` 保存未改動的下載 PDF、官方 HTML 及明確標示擷取方式的官方目錄文字存檔；`annotated/` 保存另製的紅框導讀副本。每份註記版首頁為繁體中文導讀，後續頁面以紅色向量矩形及編號標出引用位置。**原始 PDF 的導讀版頁碼比原件多一頁**；網頁閱讀版另有自己的頁碼，不能當作期刊頁碼。`manifest.json` 保存來源 URL、文件類型、原件及註記版 SHA-256、原始 PDF 頁碼和紅框位置。原文圖表及數值屬各來源研究，不是臺灣五站的分析結果。

| 文獻及版本 | 原件／紅框版 | 紅框重點 | 使用界線 |
| --- | --- | --- | --- |
| [Serra 與 Haller（2016），*Objective Eulerian Coherent Structures*](https://doi.org/10.1063/1.4951720)，[arXiv v2](https://arxiv.org/abs/1512.02112) | 原件（`data/literature/original/serra_haller_2016_oecs.pdf`）／導讀（`data/literature/annotated/serra_haller_2016_oecs_marked.pdf`） | A1–A2：應變率張量及特徵向量；A3：圖 10 將骨架、流線與短期變形並列。 | 來源檔為作者預印本；應以正式論文 DOI 引用。 |
| [Serra 等（2020），*Search and rescue at sea aided by hidden flow structures*](https://doi.org/10.1038/s41467-020-16281-x) | 原件（`data/literature/original/serra_et_al_2020_traps_search_rescue.pdf`）／導讀（`data/literature/annotated/serra_et_al_2020_traps_search_rescue_marked.pdf`） | B1：圖 2 的 TRAP 物理示意；B2–B3：圖 4、表 1 的漂流器／流線／散度／距離比較；B4–B5：圖 5、表 2 的逐時現地核對。 | 搜救漂流器實驗不是本研究海廢濃度觀測。期刊 PDF 已納入[僅修正參考文獻 17 連結的更正](https://doi.org/10.1038/s41467-020-16955-6)；文章標示 CC BY 4.0。 |
| [Kunz 等（2024），*Transient Attracting Profiles in the Great Pacific Garbage Patch*](https://doi.org/10.5194/os-20-1611-2024) | 原件（`data/literature/original/kunz_et_al_2024_traps_gpgp.pdf`）／導讀（`data/literature/annotated/kunz_et_al_2024_traps_gpgp_marked.pdf`） | C1：圖 2 的單時刻標準圖；C2：圖 4 的空間分布四分圖；C3：圖 5 的結構壽命；C4–C5：圖 9–10 的漂流器停留與相對運動。 | 北太平洋中尺度、逐日分析的門檻與壽命不可直接搬到臺灣約 1 km 網格；文章標示 CC BY 4.0。 |
| [Ghosh 等（2021），*Persistency of debris accumulation in tidal estuaries using Lagrangian coherent structures*](https://doi.org/10.1016/j.scitotenv.2021.146808)，[QUT 作者稿](https://eprints.qut.edu.au/213199/) | 原件（`data/literature/original/ghosh_et_al_2021_debris_lcs.pdf`）／導讀（`data/literature/annotated/ghosh_et_al_2021_debris_lcs_marked.pdf`） | D1：圖 2 的出現日數／歷史海廢並列；D2：清理紀錄取樣限制；D3–D4：圖 3、5 的潮相與大／小潮分層。 | **使用 backward FTLE／LCS，不是 TRAP。**僅參照成果呈現和觀測核對；來源稿標示 CC BY-NC-ND 4.0，註記副本只供本專案內部研究，勿對外散布。 |

## 閱讀與驗證

1. 先開啟對應的「導讀」PDF 首頁，按紅框編號跳到註記版頁碼；同頁原件頁碼可在導讀及 `manifest.json` 查到。
2. 比較各篇的圖面設計時，區分「TRAP 結構檢出」、「漂流器運動」、「海廢觀測」與「他種 LCS 方法」。特徵壽命、物體停留時間與廢棄物濃度不可混稱。
3. 以 `shasum -a 256 data/literature/original/*.pdf data/literature/annotated/*.pdf` 核對本地檔案，再與 `manifest.json` 中各筆 SHA-256 比較。清冊的紅框座標使用頁面左上角為原點的 0–1 比例，便於確認框選位置。

文獻來源及權利資訊以原始出版者／典藏頁和 PDF 所載為準；保留原件能使註記與原文分開稽核。任何後續報告如採用論文圖像，須再次核對授權及圖說歸屬。

## 25 點逐時平均之補充文獻

2026 年 10 月 14 日保存七項補充來源並製作 E–K 編號紅框導讀。四份為完整 PDF 原件，兩份為完整官方 HTML，另有一份官方目錄描述的文字存檔。Parker 原件 SHA-256 與 NOAA 典藏所列值相符。以下導讀框出方法依據，也保留限制條件，便於引用時區分「方法定義」、「應用先例」與「本研究設定」。

| 來源 | 保存原件／紅框導讀 | 引用位置與支持內容 |
| --- | --- | --- |
| Parker（2007），NOAA 潮汐分析手冊 | 完整 PDF（`data/literature/original/parker_2007_tidal_analysis_prediction.pdf`）／紅框導讀（`data/literature/annotated/parker_2007_tidal_analysis_prediction_marked.pdf`） | E1–E4：印刷頁 124–125；日潮與半日潮抑制、對稱置中、1/25 等權及非潮訊號衰減。 |
| Ueki 等（2003） | 完整期刊 PDF（`data/literature/original/ueki_et_al_2003_current_variations.pdf`）／紅框導讀（`data/literature/annotated/ueki_et_al_2003_current_variations_marked.pdf`） | F1：第 2 節第 [8] 段，原件第 3 頁；逐時 ADCP 流速先作 25 小時平均，再每日取樣。 |
| Ohshima 等（2002） | 完整期刊 PDF（`data/literature/original/ohshima_et_al_2002_okhotsk_drifters.pdf`）／紅框導讀（`data/literature/annotated/ohshima_et_al_2002_okhotsk_drifters_marked.pdf`） | G1：第 2 節第 [15] 段，原件第 5 頁；近表層流作 25 小時平均後區分平均／中尺度與日週期／慣性成分，原作者每 4 小時取樣。 |
| Murakami 等（1982），港灣研究機構報告 | 完整掃描 PDF（`data/literature/original/murakami_et_al_1982_constant_flow.pdf`）／紅框導讀（`data/literature/annotated/murakami_et_al_1982_constant_flow_marked.pdf`） | H1：英文摘要最後一段，原件第 4 頁；25 小時移動平均流速向量的背景流研究先例。 |
| California DWR，VTools 技術文件 | 官方 HTML 存檔（`data/literature/original/cadwr_vtools_tidal_filters.html`）／引用段落紅框閱讀版（`data/literature/annotated/cadwr_vtools_tidal_filters_marked.pdf`） | I1–I4：Filters 與其前方段落；月潮週期、完整 Godin 的 24／24／25 組成、數日訊號保留與每日取樣混疊。 |
| Walters 與 Heston（1982），USGS 論文紀錄與摘要 | 官方 HTML 存檔（`data/literature/original/walters_heston_1982_usgs_record.html`）／摘要紅框閱讀版（`data/literature/annotated/walters_heston_1982_usgs_record_marked.pdf`） | J1：Abstract；Godin 濾波對 2–3 日變化的衰減取捨。此份是官方摘要紀錄，不是四頁期刊全文。 |
| Scripps／NOAA，美國西岸次潮表層流產品 | 官方目錄文字存檔（`data/literature/original/noaa_scripps_uswc_subtidal_catalog_snapshot.txt`）／描述紅框閱讀版（`data/literature/annotated/noaa_scripps_uswc_subtidal_catalog_snapshot_marked.pdf`） | K1：Description；25 小時移動平均應用於逐時 HF 雷達表層流圖。直接下載目錄於保存日期回傳 404，故保留查核工具取得的官方描述文字；不是原始 HTML 或資料產品本身。 |

紅框位置及全部檔案雜湊見 `data/literature/tidal_filter_annotations.json`，重建規格見 `data/literature/tidal_filter_annotation_spec.json`，書目清冊見 `data/literature/tidal_filter_references.json`。主 `data/literature/manifest.json` 整合四篇 TRAP 文獻與七項補充來源。

```sh
/Users/mustlab/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_tidal_filter_literature.py
```

重建程式只讀取保存的原件、另製導讀版，並檢查頁數、框選座標與原件雜湊；不下載或修改流場資料。程式中的繁體中文註解說明網頁段落擷取、PDF 座標換算、原頁保留與可追溯清冊的處理邏輯。可重建不等於已完成目視檢查；實際交付亦須核對所有導讀頁及框選頁。

完整查核、解析振幅響應及適用界線見 [25 點逐時置中平均之文獻依據](../../docs/TRAPS_25H_FILTER_LITERATURE_AUDIT.md)。此濾波設定屬本專案前處理，應與 Kunz（2024）的每日瞬時場取樣方式分別說明。
