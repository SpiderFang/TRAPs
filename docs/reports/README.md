# TRAPs 成果報告索引

本目錄收錄 TRAPs（工項 2：瞬態吸引剖面）的兩份可獨立引用的報告，沿用姊妹專案 Lagrangian 將「方法與實作」和「執行成果」分開的做法：方法報告不隨結果更新，成果報告則綁定特定的執行（`param_id`）與程式版本。

| 文件 | 內容 | 何時引用 |
| --- | --- | --- |
| [01_TRAPs_實作方式報告.md](01_TRAPs_實作方式報告.md) | 工項要求對應、資料與前處理、TRAP 演算法（公式與參數）、追蹤與生命週期、時間置換 null、統計指標定義、成果圖說明、軟體架構與品質控管 | 說明「怎麼算」、驗證方法或複現流程 |
| [02_TRAPs_執行成果報告.md](02_TRAPs_執行成果報告.md) | 2024–2025 年五站（逐時與去潮汐日產品）的統計、圖、解讀與限制 | 引用具體數值或圖面 |

## Word 版（計畫書用）

[`word/`](word/) 內為依姊妹專案 LBT 報告的寫法與版面重寫的 Word 版本，供納入計畫書；上表兩份 Markdown 報告維持不變。

| 文件 | 內容 |
| --- | --- |
| [word/TRAPs_實作方法.docx](word/TRAPs_實作方法.docx)（原稿 [.md](word/TRAPs_實作方法.md)） | 三、瞬態吸引剖面（TRAPs）實作方法：資料與前處理、12 步數值計算（公式、算例）、參數總表、程式架構、品質控管與限制 |
| [word/TRAPs_執行成果.docx](word/TRAPs_執行成果.docx)（原稿 [.md](word/TRAPs_執行成果.md)） | 四、TRAPs 執行成果：執行規模、五站逐站成果（各站獨立、不排名）、綜合說明、限制與使用說明 |

重建：`python3 scripts/generate_word_report_md.py && python3 scripts/build_word_reports.py`（需 pandoc、python-docx、matplotlib；結果目錄預設 `outputs/server`，可用 `TRAPS_RESULTS` 指定）。數字核對：`python3 scripts/check_word_report_numbers.py`（逐站表格與 `site_statistics.csv` 比對）。圖檔為 JPG（寬度不超過 1600 px），位於 `word/figures/`。

引用用圖放在 [`figures/`](figures/)（`figures/<param_id>/<site_id>/`）：兩個產品、五站皆有 `spatial`、`hotspots`、`lifetime`、`long_tracks`、`example`、`coastal_diagnostic`；日產品另有 `seasonal`、`propagation`。其餘 120 張圖中未收錄者見 `outputs/server/<param_id>/aggregate_report/figures/`（`outputs/` 不納入 git）。

## 如何引用

兩份報告共同對應下列版本：

- 程式：`99c31518646b14948b81afdd7fd83d4674112feb`（簡記 `99c3151`，`git_dirty = false`）
- 結果：`param_id = r1_hourly`（aggregate 完成於 2026-09-29 10:40 UTC）與 `d1_daily25h`（2026-09-29 10:34 UTC）
- 報告撰寫日期：2026-10-05

建議引用格式：

> TRAPs 專案（2026）。TRAPs 實作方式報告：表層瞬態吸引剖面之檢出、追蹤與統計方法。`docs/reports/01_TRAPs_實作方式報告.md`，程式版本 `99c3151`，2026-10-05。

> TRAPs 專案（2026）。TRAPs 執行成果報告：2024–2025 年五站表層短期吸引潛勢。`docs/reports/02_TRAPs_執行成果報告.md`，結果 `r1_hourly`／`d1_daily25h`，程式版本 `99c3151`，2026-10-05。

引用時請同時註明「僅模型資料（MODEL_ONLY）、無獨立觀測驗證」。方法所依據的文獻：

- Serra, M. and Haller, G. (2016). *Chaos*, 26, 053110. doi:10.1063/1.4951720
- Serra, M. et al. (2020). *Nature Communications*, 11, 2525. doi:10.1038/s41467-020-16281-x
- Kunz, L., Griesel, A., Eden, C., Duran, R., Sainte-Rose, B. (2024). *Ocean Science*, 20, 1611–1630. doi:10.5194/os-20-1611-2024

文獻備查檔見 [`data/literature/README.md`](../../data/literature/README.md)。

## 注意

- 報告中的數值取自最終 `site_statistics.csv／.json` 與 `manifest.json`；若重跑資料或更換程式版本，須另產生新的成果報告與引用資訊。
- 各站獨立分析，不應從報告表格推導站間排序。
