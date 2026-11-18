# 25 點逐時置中平均之文獻依據與適用範圍

> 註：本文以 `路徑` 純文字標示的檔案（如 outputs/、data/literature/original/、data/literature/annotated/、docs/assets/ 等）僅存於本機，未納入版本控制。

查核日期：2026 年 10 月 14 日。

本紀錄區分文獻所載方法、本專案實作及解析頻率響應。此次僅查核與保存參考資料，未修改流場處理參數、TRAP 檢出、追蹤或既有科學結果。

## 文獻查核結果

1. **Parker（2007），《Tidal Analysis and Prediction》，NOAA Special Publication NOS CO-OPS 3。**印刷頁 124–125（PDF 第 134–135 頁）直接說明 25 小時移動平均：對逐時資料使用 25 個權重，各為 1/25，可減弱半日潮與日潮訊號；同時明示，對許多應用而言仍殘留過多潮汐能量。印刷頁 127（PDF 第 137 頁）進一步討論箱形濾波器的旁瓣與低頻衰減。來源：[NOAA 典藏](https://repository.library.noaa.gov/view/noaa/50576)；本地原始 PDF（`data/literature/original/parker_2007_tidal_analysis_prediction.pdf`）。下載檔 SHA-256 已核對 NOAA 典藏所列值，見 補充清冊（`data/literature/tidal_filter_references.json`）。
2. **California Department of Water Resources，VTools 濾波技術文件。**「Filters」段明確提出：逐時資料可使用 25 個值的箱形平均，使其頻率響應較接近約 24.8 小時的月潮週期；同頁區分單次箱形平均與完整 Godin 濾波，並說明殘留潮汐在每日取樣後可能造成混疊。此資料是政府機關之實作技術文件，不是 TRAP 專屬驗證。來源：[Averaging, Filtering and Low Passing Data](https://cadwrdeltamodeling.github.io/vtools3/html/notebooks/filters.html)。
3. **Murakami、Morikawa 與 Sakaguchi（1982），Wind Effect and Water Discharge Effect on Constant Flow – Discussion Using Observation Data at Off-Sennan (1978–1981)。**日本港灣研究機構的原始研究摘要將 25 小時移動平均流速向量用於大阪灣流場的背景流變化分析，提供海流應用先例。摘要未逐項列出與本專案相同的置中、時間戳及缺值規則，不宜宣稱完整實作相同。來源：[PARI Report 021-04-01](https://www.pari.go.jp/en/1982/12/1982120210401.html)。
4. **Walters 與 Heston（1982），Removing tidal-period variations from time-series data using low-pass digital filters，Journal of Physical Oceanography, 12, 112–115。**作者比較潮汐低通濾波器，並指出 Godin 濾波會顯著衰減 2–3 日天氣事件造成的水位變化。本篇支持「抑潮能力與保留短期非潮訊號之間存在取捨」，並不直接證明單次 25 點平均最適合臺灣 TRAP 分析。來源：[USGS 原始論文紀錄與摘要](https://pubs.usgs.gov/publication/70174408)；DOI：10.1175/1520-0485(1982)012<0112:RTPVFT>2.0.CO;2。

5. **Ueki、Kashino 與 Kuroda（2003），Observation of current variations off the New Guinea coast including the 1997–1998 El Niño period and their relationship with Sverdrup transport，Journal of Geophysical Research: Oceans, 108(C7)。**第 2 節第 [8] 段明確說明，逐時 ADCP 流速先經 25 小時移動平均低通處理，再擷取每日 0000 UT 資料，建立每日流速序列。這是「逐時流速平均後每日取樣」的直接研究先例；其 ADCP 深度、區域、每日取樣時刻及缺值條件並非本專案設定，亦非 TRAP 專屬驗證。來源：[出版者全文](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2002JC001611)，DOI：10.1029/2002JC001611。
6. **Ohshima、Wakatsuchi、Fukamachi 與 Mizuta（2002），Near-surface circulation and tidal currents of the Okhotsk Sea observed with satellite-tracked drifters，Journal of Geophysical Research: Oceans, 107(C11)。**第 [14]–[15] 段先將漂流器速度整理為逐時資料，再使用 25 小時移動平均區分平均／中尺度流與日週期／慣性成分；濾波後資料以每 4 小時重新取樣。這是近表層流的應用先例，但作者的輸入內插與取樣規則不同於本專案。來源：[出版者全文](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2001JC001005)，DOI：10.1029/2001JC001005。
7. **Scripps Institution of Oceanography／NOAA，美國西岸 2 公里近即時次潮表層流產品。**官方資料說明記載，對逐時 HF 雷達表層流圖施加 25 小時移動平均以估計次潮流。這是與本專案「逐時表層流格點」更接近的產品應用先例；資料說明並未完整交代與本專案相同的缺值或置中規則。來源：[政府資料目錄](https://catalog.data.gov/dataset/near-real-time-subtidal-surface-ocean-velocity-u-s-west-coast-2km-resolution)。

以上三項於 2026 年 10 月 14 日補充核對出版者全文或官方產品說明。可將 Parker（2007）用於濾波定義與限制，Ueki 等（2003）用於平均後每日取樣的研究先例，Ohshima 等（2002）及 Scripps／NOAA 產品用於近表層流應用先例；不應將不同來源的研究區域、取樣時刻及缺值處理合併描述為與本專案完全相同的方法。

## 保存原件與紅框引用位置

四份完整 PDF（Parker、Ueki、Ohshima、Murakami）、兩份完整官方 HTML（VTools、USGS 論文摘要紀錄）及一份 Scripps／NOAA 官方目錄描述文字存檔保存於 `data/literature/original/`，對應 E–K 編號導讀保存於 `data/literature/annotated/`。文件類型、原件／導讀頁碼差、引用位置及用途見 [文獻索引](../data/literature/README.md#25-點逐時平均之補充文獻)。

VTools 與 USGS 的 PDF 為官方網頁引用段落閱讀版；USGS 紀錄不是 Walters 與 Heston 的完整期刊 PDF。Scripps／NOAA 目錄於保存日無法直接 HTTP 下載，已明確標示保存的是查核工具取得之官方描述文字。紅框支持的論述與來源版本均保留於清冊，避免把摘要、網頁閱讀版或本研究設定混稱為期刊原文。

## 本專案的實作對應

`src/traps/daily_fields.py` 對每個格點的東西向與南北向流速分量分別取等權平均。完整視窗為代表時刻之前 12 筆、中心 1 筆及之後 12 筆，共 25 筆逐時資料；不是前後各取 25 筆。每日中心時刻設定為 UTC 12:00。

採用此方法可提出的科學理由包括：減弱主要日潮與半日潮振盪、以奇數點形成對稱置中視窗，以及在抑潮的同時保留較多數日尺度流場變化。以上是此次文獻與解析計算所支持的選用理由；不等同於已查得當初決策的完整紀錄，也不代表已完成臺灣五站的濾波最佳化驗證。

其中，25 點逐時視窗的濾波響應較接近約 24.8 小時的月潮週期；奇數點且權重對稱可使輸出時間落在中心樣本，避免引入固定時間位移。採用單次平均而非完整 Godin 濾波的選用理由，可表述為「在減弱潮汐與保留數日尺度變化之間取捨」，並以後述振幅響應支持。這不是對臺灣各站最佳視窗長度的實測結論，也不能將原始逐時流場的所有日內變化直接認定為潮汐。

Kunz 等（2024）第 2.2 節使用每 3 小時提供的瞬時地轉流與 Ekman 流合成產品，擷取每日 UTC 00:00 的瞬時場。其正文未採用 25 點平均。因此 25 點平均應歸屬本研究的前處理設定，不能以 Kunz 作為此濾波步驟的直接依據。來源：[Kunz 等（2024）原文](https://os.copernicus.org/articles/20/1611/2024/#section2)。

## 完整視窗的解析振幅響應

對等間隔且無缺值的逐時正弦訊號，25 點平均的振幅保留比例為：

\[
G(T)=\left|\frac{\sin(25\pi\Delta t/T)}{25\sin(\pi\Delta t/T)}\right|,\qquad\Delta t=1\text{ 小時}.
\]

| 訊號 | 週期（小時） | 平均後振幅保留比例 |
| --- | ---: | ---: |
| M2 | 12.4206 | 0.642% |
| S2 | 12.0000 | 4.000% |
| N2 | 12.6583 | 1.278% |
| K1 | 23.9345 | 4.260% |
| O1 | 25.8193 | 3.280% |
| P1 | 24.0659 | 3.738% |
| 2 日尺度 | 48 | 61.028% |
| 3 日尺度 | 72 | 81.341% |
| 7 日尺度 | 168 | 96.403% |

以上數值於本次依公式重新計算，為濾波器對理想正弦訊號的解析響應，並非 OCM 資料的實測抑潮率；「振幅比例」也不是「能量比例」。例如 S2 振幅保留 4%，能量比例為 0.16%。

完整 Godin 濾波對逐時資料依序施加 24、24、25 點平均，等效權重支撐為 71 點。單次 25 點平均僅是其一個組成部分。依相同解析計算，完整 Godin 對 3 日與 7 日週期的振幅保留比例約為 55.666% 與 90.112%；單次 25 點平均則為 81.341% 與 96.403%。此比較顯示保留數日尺度變化的取捨，並非宣稱其中一種濾波器對所有研究目的較佳。

## 報告用語與驗證邊界

建議名稱為「25 點逐時置中等權平均之日尺度流場」或「25 小時置中平均之潮汐抑制流場」。避免將單次平均直接稱為完整 Godin 濾波，或將其結果描述為潮汐已完全移除。

目前程式允許視窗實際存在 23 或 24 筆資料時，以實際筆數為分母；這些視窗的權重及時間位置已不同於完整 25 點濾波，不能直接套用上表。對所有納入時刻皆有效的格點才輸出平均，否則保留缺值。

如需支持正式科學結論，應另以各站 OCM 的濾波前後頻譜或分潮振幅，量化殘留潮汐及數日尺度訊號保留程度，並檢查殘留潮汐在每日取樣後的混疊。本次尚未執行此資料驗證，亦未變更既有處理設定。
