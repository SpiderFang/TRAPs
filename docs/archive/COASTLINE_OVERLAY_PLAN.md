# GeoJSON 海岸線成果圖整合計畫

決策日期：2026-09-23。使用者將 `data/taiwan_exact_coastline/taiwan_exact_coastline.geojson` 放入本專案。此檔為 CRS84 `FeatureCollection`，含 1,905 個 `Polygon`／`MultiPolygon` feature、1,912 個閉合 ring，SHA-256 為 `9e2e0ac9bc527aca87d89332cd428fdcb776eefbf94a85dd70f887f729b95fdd`。同一雜湊已記錄於相鄰 OCM 前處理專案的 `data/coastline/README.md`，來源為 OSMData land polygons，正式引用需保留 OpenStreetMap／ODbL attribution。

## 邊界與驗收

- 使用 GeoJSON **僅繪製向量海岸線輪廓**。它沒有高程、水深或潮位欄位，不能標成地形或水深量測。`SurfaceMonth.analysis_ocean_mask`、既定四個 flow domain、A 區兩站分割與 TRAP 數值不因圖面疊線而變更。
- 地理座標圖直接使用 CRS84 經度／緯度；局部公尺圖須由同一已核對的 OCM snapshot 經緯度軸與 `x_m/y_m` 軸轉換。只繪與圖面範圍相交的 ring，軸範圍由 OCM／研究圖決定，不因域外 polygon 被放大。
- 圖面與輸出清冊應記錄 coastline 路徑、SHA-256、CRS、來源／授權與 `visual_overlay_only`。預設讀取本專案資料夾，並允許 CLI `--coastline-geojson` 覆寫；顯式路徑錯誤或 CRS／幾何不合格須拒絕，沒有預設檔時明示未疊圖。
- 單時刻檢出圖、文獻方法三分圖、空間四分圖，以及資料到齊後的漂流器與海廢地理圖應疊海岸線。概念示意、五站長條圖、track 時間分布不需地理岸線。C 區顯示仍為南灣，上游 ID 不改。
- 使用現有四區四時刻 `PILOT_ONLY` catalog 重新輸出新版本文獻圖組，目視核對 A–D 岸線、圖上核心與曲線、地理軸及註腳；測試和 manifest 的全部 SHA-256 必須通過。

## 一次性兩席實作佇列

主代理先完成資料與流程檢查、制定本計畫，最後整合與審核；最多兩席 `luna_implementer` 同時執行，子代理不得再派發。各項程式所有權不重疊；所有修改程式與測試須有繁體中文設計註解。

| 序號 | 所有權 | 交付與驗收 |
| --- | --- | --- |
| 1 | GeoJSON 讀取與地理圖：`src/traps/coastline.py`、`src/traps/spatial_figures.py`、`src/traps/conditional_figures.py`、`tests/test_coastline_geographic.py` | 提供 `load_coastline(path)`、`draw_geographic_coastline(axis, coastline, *, bbox, ...)`、`draw_metric_coastline(axis, coastline, *, lon_axis, lat_axis, x_axis, y_axis, ...)`。驗證 CRS84、Polygon/MultiPolygon、閉合 ring、有限經緯度與 SHA；空間四分圖及條件漂流器／海廢地理圖可選疊線。合成 polygon 測試域內／域外、拒絕錯誤 CRS 與 malformed ring；既有條件 gate 不受影響。 |
| 2 | 單時刻及方法圖：`src/traps/figures.py`、`src/traps/literature_figures.py`、`tests/test_coastline_metric.py` | 使用第 1 項預定的 `draw_metric_coastline` 介面，為 `render_snapshot_figure` 與 `render_method_comparison_figure` 加入可選 coastline 參數。局部公尺轉換須使用 snapshot 自身的經緯度、公尺軸；缺軸或不單調時有明確錯誤。岸線應在流場底圖上可見而不遮住 TRAP 核心或曲線。 |
| 3 | CLI 與來源追溯：`src/traps/cli.py`、`tests/test_cli_coastline.py` | 等 1–2 介面完成後加入預設／顯式來源處理，接到 `detect --render-snapshots`、`render --kind snapshot`、`literature-figures`。manifest 與 `figure_availability.json` 記錄圖面岸線來源、SHA 與是否疊圖；catalog 的科學統計與 ocean mask 不變。測試含缺檔、覆寫與 provenance。 |

第 1、2 項並行；第 3 項交給先完成的既有席位。主代理在全部交付後做一次最終整合審核，更新 README 與模組流程圖，核對試算圖與來源清冊。若視覺圖需改變 OCM/analysis mask 或流域邊界，應視為另一項科學決策，不納入此圖面整合。

## 2026-09-23 整合驗收

- 三項佇列均完成。`load_coastline()` 核對本專案真實檔案為 CRS84、1,905 feature、1,912 ring、151,106 點；CLI 預設載入該檔，並提供 `--coastline-geojson` 顯式覆寫。顯式錯誤來源會拒絕執行，缺少預設檔則於輸出記錄 `default_missing`。
- [四區四時刻海岸線版試算圖組](../../outputs/_archive/pilot/literature_figures_four_area_4frames_v4_coastline/figure_availability.json)的 A–D 空間四分圖及方法三分圖已逐圖目視核對，C 區顯示為南灣；另產生[A 區單時刻圖](../../outputs/_archive/pilot/snapshot_a_20250101_t0100_v4_coastline/snapshot_000000.png)。全部 27 個圖組 artifact SHA-256 均與 manifest 相符，五站摘要、概念示意及四區 `spatial_summary_metadata.json`、`lifetime_summary.json` 與前版位元組一致。
- 全套 pytest 57 項通過；最後的地理軸／公尺軸等長驗證加固後，相關 12 項測試通過；修改程式的 Ruff 檢查亦通過。這是 `PILOT_ONLY` 的本機工程驗證，海廢、漂流器與潮相條件圖因無合格輸入仍為 `DATA_REQUIRED`。
