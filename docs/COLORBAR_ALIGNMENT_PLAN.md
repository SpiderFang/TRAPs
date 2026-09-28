# TRAPs 主圖與色條等高規則

決策日期：2026-09-23。使用者要求後續圖表的色條與其對應主圖同高，參考相鄰 `OCM-SVD-Analysis` 繪圖法。該專案 `src/ocm_svd_analysis/water_column_multivariate_svd.py` 的 `align_colorbar_to_map_axis()` 先讓 Matplotlib 完成 `aspect='equal'` 的實際座標框配置，再把色條 axes 的 `y0`、`height` 對齊主圖；其獨立報告版亦以顯式 axes 分欄。TRAPs 採其「以實際 axes 而非 GridSpec 預留欄量測」原則，保留本專案現有版型、座標比例及色階語義。

## 範圍與驗收

- 所有現有垂直色條：單時刻 `s1` 圖 1 條；方法比較圖的 `s1`、散度各 1 條；空間四分圖各分圖 1 條；資料到齊才產的海廢／TRAP 共現圖 2 條。無色條的概念圖、壽命圖、五站摘要及漂流器軌跡圖不受影響。
- 每條色條與其**對應主圖實際 axes 框**的上、下邊界各相差不超過 0.5 point，以完成 Matplotlib layout 與 aspect 計算後的座標為準。海廢圖兩條色條都對齊同一主圖；方法圖只對齊有色階的兩個分圖。
- 共用 `align_colorbars_to_axes(figure, pairs)`；`pairs` 是 `(main_axis, colorbar_axis)` 序列。先觸發畫布排版，固定 layout engine，再只修改色條 axes 的垂直位置與高度，保留水平位置／寬度。輸出前再次 draw 與量測；不改 `vmin/vmax`、自訂 locator／formatter、資料、海岸線、flow domain、TRAP 計算或條件資料 gate。原有自動刻度可因色條高度改變而重選顯示刻度，避免文字重疊。
- 用實際 Matplotlib axes 量測測試涵蓋等比例圖與多色條圖，並檢查 PNG 能輸出。重產新的本機 `PILOT_ONLY` 圖組及單時刻圖，與既有 v4 比對科學摘要 JSON 一致、manifest 雜湊相符並目視核對標題／色條／圖框沒有碰撞。舊輸出保留。

## 一次性兩席實作佇列

| 項目 | 子代理所有權 | 交付 |
| --- | --- | --- |
| 1 | `src/traps/colorbar_layout.py`、`src/traps/figures.py`、`src/traps/spatial_figures.py`、`tests/test_colorbar_layout.py` | 共用對齊 helper、單時刻及四分圖接線；合成等比例軸與實際 renderer 高度量測測試。 |
| 2 | `src/traps/literature_figures.py`、`src/traps/conditional_figures.py`、`tests/test_colorbar_other_figures.py` | 方法圖與雙色條海廢圖接線；實際 axes 上下界測試，並維持條件資料 gate。使用項目 1 的已約定 helper API，不修改其檔案。 |

主代理負責現況核對、此一次性計畫、文件、整合後唯一一次最終審核與新圖組驗收。兩位 `luna_implementer` 同時工作，各自只修改指定檔案，不再派發子代理。所有程式與測試須補繁體中文設計註解；若測試證明對齊需更改科學色階或研究設計，先停止並回報，不自行擴張範圍。

## 2026-09-23 整合驗收

- 共用 helper 已接入單時刻圖、方法比較圖兩個色階、空間四分圖四個色階及海廢／TRAP 共現圖兩個色階。測試以實際 Matplotlib axes 與輸出 PNG 量測等比例圖、constrained layout 與雙色條圖，上下界誤差均在 0.5 point 內；另核對已完成 layout 後色條水平位置與寬度維持不變。全套 pytest 61 項與修改程式 Ruff 檢查通過。
- [四區四時刻 v5 圖組](../outputs/pilot/literature_figures_four_area_4frames_v5_colorbar/figure_availability.json)與[A 區單時刻 v5 圖](../outputs/pilot/snapshot_a_20250101_t0100_v5_colorbar/snapshot_000000.png)已重產，代表性 A、C、D 圖及合成雙色條海廢圖已目視核對。圖組 27 個 artifact SHA-256 全部相符；四區 `spatial_summary_metadata.json`、`lifetime_summary.json`、五站摘要與概念示意和 v4 位元組一致。
- 新色條高度可能使 Matplotlib 自動 locator 減少可見刻度，避免短色條文字相疊；此為顯示層改變，未改 `vmin/vmax` 或資料。缺少合格外部海廢、漂流器與潮相資料的正式條件圖仍受既有 `DATA_REQUIRED` gate 限制。
