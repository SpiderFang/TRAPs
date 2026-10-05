# TRAPs 文件入口

## 閱讀路線

| 目的 | 先讀 | 再讀 |
| --- | --- | --- |
| 想知道研究做了什麼、得到什麼 | [reports/02_TRAPs_執行成果報告.md](reports/02_TRAPs_執行成果報告.md) | [reports/01_TRAPs_實作方式報告.md](reports/01_TRAPs_實作方式報告.md) |
| 想知道方法、公式、參數與統計定義 | [reports/01_TRAPs_實作方式報告.md](reports/01_TRAPs_實作方式報告.md) | [`../README.md`](../README.md) 的模組表與流程章節 |
| 想在 SERVER 上重跑或續跑 | [operations/SERVER_RUNBOOK.md](operations/SERVER_RUNBOOK.md) | [`../scripts/run_traps_server.sh`](../scripts/run_traps_server.sh) |
| 想引用報告或圖 | [reports/README.md](reports/README.md) | |
| 想看文獻與需求原件 | [`../data/literature/README.md`](../data/literature/README.md)、[`../data/工項2.pdf`](../data/工項2.pdf) | |

## 文件分類

| 文件 | 狀態 | 說明 |
| --- | --- | --- |
| [reports/01_TRAPs_實作方式報告.md](reports/01_TRAPs_實作方式報告.md) | 現行 | 方法與實作報告 |
| [reports/02_TRAPs_執行成果報告.md](reports/02_TRAPs_執行成果報告.md) | 現行 | 2024–2025 執行成果（commit `99c3151`） |
| [reports/README.md](reports/README.md) | 現行 | 報告索引與引用方式 |
| [operations/SERVER_RUNBOOK.md](operations/SERVER_RUNBOOK.md) | 現行 | SERVER 部署與批次執行手冊 |
| [archive/IMPLEMENTATION_PLAN.md](archive/IMPLEMENTATION_PLAN.md) | 歷史（pilot 階段） | 早期實作佇列與驗收界線 |
| [archive/MODULE_FLOW_AND_FIGURE_AUDIT.md](archive/MODULE_FLOW_AND_FIGURE_AUDIT.md) | 歷史（pilot 階段） | 本機四時刻試算的模組流程與文獻圖型對照 |
| [archive/LITERATURE_FIGURE_IMPLEMENTATION_PLAN.md](archive/LITERATURE_FIGURE_IMPLEMENTATION_PLAN.md) | 歷史（pilot 階段） | 文獻圖組實作與驗收記錄 |
| [archive/COASTLINE_OVERLAY_PLAN.md](archive/COASTLINE_OVERLAY_PLAN.md) | 歷史（pilot 階段） | 海岸線疊圖計畫 |
| [archive/COLORBAR_ALIGNMENT_PLAN.md](archive/COLORBAR_ALIGNMENT_PLAN.md) | 歷史（pilot 階段） | 色條對齊計畫 |

歸檔文件反映 2026-09 以前的 `PILOT_ONLY` 本機試算，內容（尤其是「跨月份追蹤尚未實作」等敘述與 `outputs/pilot/…` 連結）已不代表目前正式流程；若與現行文件衝突，以現行文件為準。這些文件中指向 `outputs/` 的連結指向未納入 git 的本機試算輸出。
