# 文獻圖型實作與驗收計畫

> 註：本文以 `路徑` 純文字標示的檔案（如 outputs/、data/literature/original/、data/literature/annotated/、docs/assets/ 等）僅存於本機，未納入版本控制。

決策與實作日期：2026-09-23。使用者要求將已蒐集文獻的成果呈現方式納入本專案。此處的「納入」指採用圖型所表達的研究問題、分圖與定量比較，使用本專案 TRAP 變數重新製圖；不將 Ghosh 等人的 FTLE 當成 TRAP，也不複製原文圖像或數值。下列一次性佇列均已完成，正式資料門檻仍適用。

## 研究與資料邊界

- 四區域、五站點與 A 區最近 anchor 分割沿用 [`configs/study_design.json`](../../configs/study_design.json)；C 區顯示南灣，上游 domain ID 不變。
- 可立即由已存在的 OCM/TRAP 試算資料驗證：原創機制示意、單時刻方法比較、逐格空間分布與通過次數、完整／截尾分開的壽命分布和生命週期圖。所有試算圖標 `PILOT_ONLY`；少量時刻不推論季節、熱區或長壽命。
- 潮相、大／小潮、獨立漂流器與海廢清理紀錄先做明示資料契約、時間／站點／單位／來源與品質檢查。沒有通過檢查的資料時，輸出 `DATA_REQUIRED`／`QC_BLOCKED` 狀態和缺項，不輸出空白、補值或合成的科學結果圖。相鄰 SVD 專案雖保存 CWA 潮位原始檔，尚未確立它們與本 TRAP 五站的潮相指派契約，不能直接繪製條件分層結果。
- 與文獻的「漂流器停留時間」相關圖須使用真正獨立的表層觀測和可查核來源；共用 OCM 流場的 Lagrangian 軌跡只能標成模型內部機制比較。物體停留時間與 TRAP 結構壽命分開。
- 完整兩年空間圖需有效格點的逐時分母；同一時刻同一格點有多條曲線時，出現次數只計一次。跨月份 track 接續未完成前，壽命分布只限單一 catalog 已記錄且未截尾的 track，不宣稱兩年壽命。`duration_seconds` 是首次至末次檢出的**觀測跨度**；單時刻 track 的 0 小時代表時間解析度不足，不能當成精確物理壽命。

## 一次性實作佇列

主代理界定介面、整合與最終審核；最多兩席 `luna_implementer` 同時執行程式實作與對應測試。子代理不得再派發任務，不得覆蓋其他任務的檔案。

| 序號 | 所有權 | 明確交付 | 驗收 |
| --- | --- | --- | --- |
| 1 | 空間統計與圖 | `src/traps/spatial.py`、`src/traps/spatial_figures.py`、`tests/test_spatial_figures.py`。從單區 catalog、通過來源檢查的 `SurfaceMonth` 建立 `valid_snapshot_count`、核心強度、TRAP 曲線逐格通過快照數、未截尾 track 觀測跨度等 2D 欄位；輸出借鏡 Kunz 圖 4 與 Ghosh 圖 2 的四分圖。 | 每格分子不超過有效時刻分母；無資料格點與零分母留空；曲線通過有明確格網化規則；截尾 track 不當完整壽命；地理軸與色階／單位、試算標籤齊全。**完成。** |
| 2 | 方法與壽命圖 | `src/traps/literature_figures.py`、`tests/test_literature_figures.py`。原創 TRAP 法向吸引／沿線輸送示意，OCM 單時刻 TRAP／流線／散度方法對照，及完整／截尾區分的 track 觀測跨度分布與沿生命週期強度圖。 | 示意圖標「概念」，流線不稱觀測軌跡；不同結構、物體與資料缺口不混淆；只有未截尾 track 用於完整觀測跨度統計，單時刻 track 明示時間解析度不足；圖示標時間、座標、單位與狀態。**完成。** |
| 3 | 條件資料介面及圖 | `src/traps/conditional_figures.py`、`tests/test_conditional_figures.py`。接收附 provenance metadata 的對齊 UTC 潮相表、漂流器／短期粒子軌跡表與海廢紀錄；驗證來源、站點、時間、覆蓋、QC 與分母；通過才繪潮相／大潮小潮對照、同群模型粒子凸包形變代理與同時刻 TRAP 並列、漂流器距 TRAP 與停留／相對距離變化、TRAP 出現圖與海廢紀錄疊圖。 | 缺檔、時區含糊、重複或缺失標籤、零分母及不合格 QC 都不產圖，回報具體 gate；模型軌跡須標模型內部比較，獨立觀測驗證須有確認的獨立來源；不補齊空缺、不宣稱海廢來源。**介面與條件圖完成，真實附加資料未提供。** |
| 4 | CLI 與來源記錄 | `src/traps/cli.py`、`tests/test_cli_literature.py`。加入獨立 `literature-figures` 子命令，收同參數／同 UTC 窗的四區 catalog，重開並核對上游月快取和海岸 sidecar，呼叫 1–3 的圖型；寫 `figure_availability.json`、帶 SHA-256 的 manifest。 | 無可用附加資料時只輸出可由 OCM/TRAP 產生的圖及缺項狀態；四區資料不能混時窗／混參數／混 pilot 與正式狀態；全測試 45 項通過。**完成。** |

第 1、2 項可並行；第 3 項在第 1 項完成後交給該席，且不得修改第 1、2 項的檔案；第 4 項待前述介面定型後交由另一席整合。主代理於全部回報後只做一次整合審核，更新 [`MODULE_FLOW_AND_FIGURE_AUDIT.md`](MODULE_FLOW_AND_FIGURE_AUDIT.md) 與專案 README。

## 預定介面與圖組

- 空間統計公開 `build_spatial_summary(catalog, month)`；地理軸從 `SurfaceMonth` 取得，整個流域為統計母域，無效格點為 `NaN`，逐格通過率為「有合格 TRAP 曲線穿越的有效快照數／該格有效快照數」。輸出結構須保存來源 snapshot 指標、門檻、有效格點分母及圖用單位。
- 方法／壽命圖直接接收已檢查的 snapshot、`detect_snapshot()` 結果或 catalog；概念示意獨立於 OCM 試算結果。
- 附加表格以 UTF-8 CSV 與同名 JSON metadata 為輸入；UTC 時間明確、站點映射與來源類型／獨立性須可查核。潮相使用 `flood`／`ebb` 和 `spring`／`neap` 的顯式標籤；漂流器、模型粒子與海廢資料各保留個體／紀錄 ID、位置、時間、品質與資料來源。模型粒子只供 A3 類型的內部短期變形對照，不能使用 B2–B5 的「獨立驗證」標籤。正式欄位以第 3 項完成的 schema 文件為準。
- 每個輸出目錄皆須為新的或空目錄。現行檔名為 `mechanism.png`、各區 `method_comparison.png`、`spatial_summary.png`、`observed_span_distribution.png`、`lifecycle_intensity.png` 與四區合併 `five_site_summary.png`；條件圖僅在 gate 通過後產出。所有圖用原始 TRAPs 數值與地理座標，並保存資料期間、狀態、分子分母及參數。圖像本身不得顯示不存在的觀測或把單一時刻稱為長期結果。

## 最終驗收

使用合成資料驗證空間分子／分母、分層 gate、觀測距離、粒子群凸包面積比及截尾；本機 2025-01-01 四區四時刻圖組見 `outputs/_archive/pilot/literature_figures_four_area_4frames_v3/`（`outputs/_archive/pilot/literature_figures_four_area_4frames_v3/figure_availability.json`）。最終執行須目視檢查各圖、核對 manifest SHA-256 與缺觀測／潮相時的 `DATA_REQUIRED`；圖組仍需完整 2024–2025 資料及獨立表層觀測，才能形成正式研究結論。
