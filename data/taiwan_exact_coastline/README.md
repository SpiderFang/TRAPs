# 臺灣周邊海岸線向量資料

`taiwan_exact_coastline.geojson` 是使用者提供、留存於本專案的 CRS84 經緯度 `FeatureCollection`。檢查結果：1,905 個 Polygon／MultiPolygon feature、1,912 個閉合 ring、151,106 個座標點，涵蓋本研究 A–D 四個 flow domain；GeoJSON SHA-256 為 `9e2e0ac9bc527aca87d89332cd428fdcb776eefbf94a85dd70f887f729b95fdd`。

相鄰 `OCM-Data-Preprocessing/data/coastline/osm_land_polygons_taiwan_v1.geojson` 的實際檔案雜湊相同，其來源說明指出資料來自 [OSMData land polygons](https://osmdata.openstreetmap.de/data/land-polygons.html)，由 OpenStreetMap `natural=coastline` ways 組成。正式圖說或資料再散布須保留 **© OpenStreetMap contributors／ODbL** 來源歸屬。`taiwan_exact_coastline.png` 是同資料夾原有的視覺預覽，非程式計算的輸入。

本專案使用此 GeoJSON 作**視覺海岸線疊圖**。資料包含陸域 polygon 與海岸輪廓，沒有高程、水深或潮位欄位，因此不得標成地形高度或海底地形量測。TRAP 的有效海域及微分仍依各 catalog 已核對的 `analysis_ocean_mask` sidecar；此向量檔不重新劃定 flow domain、不直接取代格點遮罩。圖面疊線的方法與來源記錄見 [`docs/COASTLINE_OVERLAY_PLAN.md`](../../docs/archive/COASTLINE_OVERLAY_PLAN.md)。
