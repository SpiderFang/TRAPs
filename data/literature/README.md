# TRAPs 文獻備查與紅框導讀

本資料夾保存規劃階段查核的四篇文獻。`original/` 是下載後未改動的 PDF；`annotated/` 是另存的紅框導讀副本。每份註記版首頁為繁體中文導讀，後續頁面以紅框及 A／B／C／D 編號標出本專案可借鏡的圖、表或方法段。**導讀版頁碼比原始 PDF 多一頁**。`manifest.json` 保存 DOI、來源 URL、版本、原件及註記版 SHA-256、原始 PDF 頁碼和每個紅框的位置。原文圖表及數值均屬各篇研究，不是臺灣五站的分析結果。

| 文獻及版本 | 原件／紅框版 | 紅框重點 | 使用界線 |
| --- | --- | --- | --- |
| [Serra 與 Haller（2016），*Objective Eulerian Coherent Structures*](https://doi.org/10.1063/1.4951720)，[arXiv v2](https://arxiv.org/abs/1512.02112) | [原件](original/serra_haller_2016_oecs.pdf)／[導讀](annotated/serra_haller_2016_oecs_marked.pdf) | A1–A2：應變率張量及特徵向量；A3：圖 10 將骨架、流線與短期變形並列。 | 來源檔為作者預印本；應以正式論文 DOI 引用。 |
| [Serra 等（2020），*Search and rescue at sea aided by hidden flow structures*](https://doi.org/10.1038/s41467-020-16281-x) | [原件](original/serra_et_al_2020_traps_search_rescue.pdf)／[導讀](annotated/serra_et_al_2020_traps_search_rescue_marked.pdf) | B1：圖 2 的 TRAP 物理示意；B2–B3：圖 4、表 1 的漂流器／流線／散度／距離比較；B4–B5：圖 5、表 2 的逐時現地核對。 | 搜救漂流器實驗不是本研究海廢濃度觀測。期刊 PDF 已納入[僅修正參考文獻 17 連結的更正](https://doi.org/10.1038/s41467-020-16955-6)；文章標示 CC BY 4.0。 |
| [Kunz 等（2024），*Transient Attracting Profiles in the Great Pacific Garbage Patch*](https://doi.org/10.5194/os-20-1611-2024) | [原件](original/kunz_et_al_2024_traps_gpgp.pdf)／[導讀](annotated/kunz_et_al_2024_traps_gpgp_marked.pdf) | C1：圖 2 的單時刻標準圖；C2：圖 4 的空間分布四分圖；C3：圖 5 的結構壽命；C4–C5：圖 9–10 的漂流器停留與相對運動。 | 北太平洋中尺度、逐日分析的門檻與壽命不可直接搬到臺灣約 1 km 網格；文章標示 CC BY 4.0。 |
| [Ghosh 等（2021），*Persistency of debris accumulation in tidal estuaries using Lagrangian coherent structures*](https://doi.org/10.1016/j.scitotenv.2021.146808)，[QUT 作者稿](https://eprints.qut.edu.au/213199/) | [原件](original/ghosh_et_al_2021_debris_lcs.pdf)／[導讀](annotated/ghosh_et_al_2021_debris_lcs_marked.pdf) | D1：圖 2 的出現日數／歷史海廢並列；D2：清理紀錄取樣限制；D3–D4：圖 3、5 的潮相與大／小潮分層。 | **使用 backward FTLE／LCS，不是 TRAP。**僅參照成果呈現和觀測核對；來源稿標示 CC BY-NC-ND 4.0，註記副本只供本專案內部研究，勿對外散布。 |

## 閱讀與驗證

1. 先開啟對應的「導讀」PDF 首頁，按紅框編號跳到註記版頁碼；同頁原件頁碼可在導讀及 `manifest.json` 查到。
2. 比較各篇的圖面設計時，區分「TRAP 結構檢出」、「漂流器運動」、「海廢觀測」與「他種 LCS 方法」。特徵壽命、物體停留時間與廢棄物濃度不可混稱。
3. 以 `shasum -a 256 data/literature/original/*.pdf data/literature/annotated/*.pdf` 核對本地檔案，再與 `manifest.json` 中各筆 SHA-256 比較。清冊的紅框座標使用頁面左上角為原點的 0–1 比例，便於確認框選位置。

文獻來源及權利資訊以原始出版者／典藏頁和 PDF 所載為準；保留原件能使註記與原文分開稽核。任何後續報告如採用論文圖像，須再次核對授權及圖說歸屬。
