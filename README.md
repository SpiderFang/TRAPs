# TRAPs：表層瞬態吸引剖面分析

本專案延續 `OCM-SVD-Analysis` 的流場分析與 `Lagrangian-Ensemble-Backtracking` 的五站研究設計，研究表層二維流場中的瞬態吸引剖面（Transient Attracting Profiles, TRAPs）。原始需求保存於 [`data/工項2.pdf`](data/工項2.pdf)，方法與成果圖的文獻原件、紅框導讀副本及來源清冊保存於 [`data/literature/`](data/literature/README.md)。

## 研究邊界與來源權威

四個流場區域的網格、邊界及遮罩沿用 `OCM-Data-Preprocessing/configs/ocm_flow_domains.json` 的既定版本；五個研究站點沿用 `Lagrangian-Ensemble-Backtracking` 的五站契約。本專案不自行重畫或擴張流場邊界。A 區兩站各自保存研究站點、統計範圍和樣本分母，雖共用流場，仍是兩個研究站點。

| 流場區域 | 研究站點 | 上游 `flow_domain_id` | 範圍說明 |
| --- | --- | --- | --- |
| A | 貢寮、龜山島西側 | `northeast_taiwan_common_cache_v3` | 兩站共用東北臺灣完整流場區域，站點比較分開統計。 |
| B | 新竹外海 | `hsinchu_cache_v3` | 使用既定新竹流場區域。 |
| C | 南灣 | `houwan_nmmba_cache_v3` | **本專案顯示名稱為南灣**；歷史資料產品 ID 仍標示後灣／海生館，流場邊界與範圍不變。 |
| D | 連江 | `lienchiang_common_cache_v3` | 使用既定連江共用流場區域，站點統計為連江一站。 |

上表描述研究設計，不代表本目錄已有經驗收的 OCM 資料副本。實際執行前必須唯讀核對上游的 schema、時間覆蓋、投影座標、濕乾遮罩、缺值、QC、版本與雜湊。PDF 以 2024–2025 年流場為基礎；2026 年資料須通過同等來源與品質檢查後另行納入。不同區域或時期缺少有效資料時，不以零值、最近值或未登錄外插補齊。

## 已實作的科學與資料流程

1. 讀取上游已驗收的逐時刻 OCM 表層 `u_surface_mps`、`v_surface_mps`、公尺制 `x_m`／`y_m`、時間軸與有效遮罩，固定各站研究範圍及其有效快照分母。
2. 在有效海域上以公尺座標計算速度梯度的對稱部分 `S = (∇v + ∇vᵀ)/2`，求最小特徵值 `s1` 及其正交的最大特徵向量 `e2`。負的 `s1` 局部極小值為候選核心；骨架沿 `e2` 積分。海岸、資料缺口、流場邊界、弱吸引及最大弧長均需有明確停止與 QC 規則。
3. 以解析／合成流場測試梯度、特徵值、方向及圖面。跨快照追蹤區分有效時刻未檢出、時間缺口及研究期邊界截尾；只把未截尾 track 納入首末檢出觀測跨度平均。單時刻的 0 小時表示時間解析度不足，不能稱為精確物理壽命。
4. 五站各自報告「有合格 TRAP 穿越研究範圍的有效快照比例」、核心吸引強度、密度或長度的面積標準化統計與結構觀測跨度。季節及潮相比較應以已驗證的時間／潮相資料分組，逐圖列出分子與分母。
5. 以現地表層漂流器或海廢觀測進行獨立核對時，另列資料品質與取樣限制。SVD 主成分與 TRAP 的關聯因共用 OCM 強迫，屬機制探索而非獨立驗證；完全沉沒的三維 Lagrangian 回溯結果也不是表層 TRAP 的直接驗證。

TRAP 描述瞬時流場的短期吸引結構。只有模型流場時，成果應稱為「表層短期吸引潛勢」；不得將檢出曲線直接寫成實測海廢濃度、長期滯留、海底廢聚集或污染來源。`data/literature/README.md` 說明各篇研究可借鏡的圖型與科學限制。

## 程式模組與輸出

| 模組 | 主要介面與責任 |
| --- | --- |
| [`src/traps/ocm.py`](src/traps/ocm.py) | `open_month()`／`SurfaceMonth.snapshot()`：唯讀 memory map、schema 3.0.0、來源與逐時遮罩檢查；trial 需顯式允許。 |
| [`src/traps/sites.py`](src/traps/sites.py) | `site_masks()`：固定五站研究設計，A 區最近 anchor 互斥分割；B/C/D 使用完整流域。研究設計快照在 [`configs/study_design.json`](configs/study_design.json)，來源路徑以本專案根目錄為基準。 |
| [`src/traps/detection.py`](src/traps/detection.py) | `detect_snapshot()`：遮罩安全微分與 0–3 格四鄰接平滑、`s1/e2`、核心及曲線、停止原因與 QC。 |
| [`src/traps/tracking.py`](src/traps/tracking.py) | `track_structures()`：一對一 Eulerian 結構配對，區分 data gap、未檢出與截尾。 |
| [`src/traps/metrics.py`](src/traps/metrics.py) | `compute_site_metrics()`：五站逐時覆蓋率、出現率的分子／分母、吸引強度和面積／有效快照標準化。 |
| [`src/traps/cli.py`](src/traps/cli.py)、[`src/traps/figures.py`](src/traps/figures.py) | `preflight`、`detect`、`summary`、`render`；輸出 catalog、CSV、WGS84 GeoJSON、圖、manifest 與 SHA-256。 |
| [`src/traps/spatial.py`](src/traps/spatial.py)、[`src/traps/spatial_figures.py`](src/traps/spatial_figures.py) | 單區逐格有效快照分母、核心強度、曲線通過率及未截尾觀測跨度；地理四分圖。 |
| [`src/traps/literature_figures.py`](src/traps/literature_figures.py) | 原創機制示意、瞬時 TRAP／流線／散度對照、完整與截尾觀測跨度分布、沿 track 強度歷程。 |
| [`src/traps/conditional_figures.py`](src/traps/conditional_figures.py) | 潮相、漂流器／模型粒子、海廢 CSV＋metadata 的來源/QC gate；通過才產條件圖；同群模型粒子可計凸包面積形變代理。 |
| [`src/traps/coastline.py`](src/traps/coastline.py) | 驗證使用者提供的 CRS84 GeoJSON 陸地 polygon 與 SHA-256，供地理座標圖及局部公尺圖畫海岸線；**僅供視覺呈現**，不更動檢出遮罩。 |
| [`src/traps/colorbar_layout.py`](src/traps/colorbar_layout.py) | 在 Matplotlib 完成等比例主圖排版後，以主圖實際 axes 上下界對齊所屬色條；僅處理成果圖版面。 |

`catalog.json` 保留原始公尺制曲線及逐時／站點分母；`catalog.geojson` 將同一曲線轉成 WGS84 經緯度。每個成果目錄須是新的或空的，以免舊圖與新參數混在同一份 manifest。單區 catalog 的域外站點雖保留零面積 schema，摘要圖只顯示該區有效站點；`summary` 可把**相同 UTC 時刻、相同檢出參數**的四區 catalog 合併為五站比較。跨月份同站點聚合與跨月份 track 接續目前尚未實作，因此不同月份不可直接交給 `summary` 當作完整壽命分析。完整[模組流程圖與文獻圖型對照](docs/MODULE_FLOW_AND_FIGURE_AUDIT.md)包含條件資料的拒絕路徑。

### 本機安裝與驗證

```bash
cd /Users/mustlab/Workspace/TRAPs
UV_CACHE_DIR=/private/tmp/traps-uv-cache uv run --extra test python3 -m pytest -q
UV_CACHE_DIR=/private/tmp/traps-uv-cache uv run python -m traps --help
```

### 可配置的 SERVER 資料路徑

相鄰 Lagrangian 專案的 runbook 記錄候選 SERVER 路徑 `/data/OCM-Preprocessed-Data/preprocessed/ocm_surface`；本次無法以 SSH 核對 SERVER 現況，因此執行時應以實際掛載路徑指定，不在程式中寫死。`--surface-root` 需與 `--domain-id`、`--month` 同用，也可直接給 `--month-dir`。例如：

```bash
OCM_SURFACE_ROOT=/data/OCM-Preprocessed-Data/preprocessed/ocm_surface
TRAP_OUTPUT_ROOT=/data/TRAPs/outputs/202501_a_baseline
python -m traps preflight \
  --surface-root "$OCM_SURFACE_ROOT" \
  --domain-id northeast_taiwan_common_cache_v3 --month 202501 \
  --ocean-mask /path/to/analysis_ocean_mask_osm_coastline_v1.npy
python -m traps detect \
  --surface-root "$OCM_SURFACE_ROOT" \
  --domain-id northeast_taiwan_common_cache_v3 --month 202501 \
  --ocean-mask /path/to/analysis_ocean_mask_osm_coastline_v1.npy \
  --output-root "$TRAP_OUTPUT_ROOT" --render-snapshots --render-summary
```

`--ocean-mask` 為上游 coastline analysis sidecar；`mask_static.npy` 只代表 OCM 網格支撐，不等於陸地遮罩。正式 `detect` 會拒絕缺 sidecar 或缺同版本 metadata 的輸入。正式研究必須另核對速度單位、月份完整性及參數敏感度。試算資料只有加上 `--allow-trial-pilot` 才可讀取，且所有產物標為 `PILOT_ONLY`。波浪資料不是此版基本 TRAP 檢出的輸入；若日後研究 Stokes 漂移或漂浮物實際運動，應作獨立的流速定義與驗證分析。

### 海岸線圖面資料

使用者提供的 [`taiwan_exact_coastline.geojson`](data/taiwan_exact_coastline/taiwan_exact_coastline.geojson) 是 CRS84 的陸地 polygon；其[來源、授權與雜湊說明](data/taiwan_exact_coastline/README.md)已留存。圖面以它畫海岸輪廓，預設由專案 `data/taiwan_exact_coastline/` 讀取，亦可用 `--coastline-geojson` 指向其他已驗證的檔案。圖說須標示 © OpenStreetMap contributors／ODbL。此檔沒有高程或水深，且不取代 `--ocean-mask` 的格點 sidecar；數值檢出與分母仍依原有遮罩。資料讀取、繪圖及遮罩的關係見[模組流程圖](docs/MODULE_FLOW_AND_FIGURE_AUDIT.md)。

### 成果圖色條版面

垂直色條須與其對應主圖的**實際座標框同高**；等比例地理圖應在 Matplotlib 完成比例配置後，再對齊色條上下界。雙色條圖的兩條色條均對齊同一主圖。此版面規則參考相鄰 `OCM-SVD-Analysis` 的主圖／色條配置，只調整繪圖幾何，不改色階上下限或科學數值。原本採自動刻度的圖可能隨色條縮短而調整顯示刻度密度，以免標籤重疊；實作與驗收界線見[色條對齊計畫](docs/COLORBAR_ALIGNMENT_PLAN.md)。

### 文獻圖組的本機重現

下列命令讀取**已完成檢出**且同 UTC 四時刻、同參數與同 `PILOT_ONLY` 狀態的四區 catalog；命令會重開原月份與 coastline analysis sidecar 核對來源，另外載入本專案 GeoJSON 畫海岸輪廓。`--output-root` 必須為新目錄或空目錄，並產生每區方法、空間、觀測跨度、強度歷程圖，以及五站摘要、`figure_availability.json`、帶 SHA-256 的 manifest。概念示意為 `CONCEPT_ONLY`，其餘試算圖為 `PILOT_ONLY`。當三類附加資料缺席時，命令記錄 `DATA_REQUIRED`，不製造觀測或潮相圖。

```bash
cd /Users/mustlab/Workspace/TRAPs
.venv/bin/python -m traps literature-figures \
  --catalog outputs/pilot/local_mac_a_20250101_4frames_v1/catalog.json \
            outputs/pilot/local_mac_b_20250101_4frames_v1/catalog.json \
            outputs/pilot/local_mac_c_20250101_4frames_v1/catalog.json \
            outputs/pilot/local_mac_d_20250101_4frames_v1/catalog.json \
  --output-root outputs/pilot/literature_figures_four_area_4frames_v5_colorbar_repro
```

正式圖組應另指定正式 catalog 與全期 OCM 資料；四時刻試算只驗證介面與版面。A 區站點統計採最近 anchor 互斥分割，空間圖的母域仍是完整 A flow domain。圖組不跨月份接續 track；`duration_seconds` 是首末檢出觀測跨度。見[文獻實作與驗收記錄](docs/LITERATURE_FIGURE_IMPLEMENTATION_PLAN.md)。

### 條件資料介面與圖型啟用

可選參數 `--tide-table`、`--trajectory-table`、`--debris-table` 各指向 UTF-8 CSV，且同目錄必須有**同 stem** JSON（例如 `tide.csv` 與 `tide.json`）。共通 JSON 欄位：`schema_version`、`table_kind`、`timezone: "UTC"`、非空 `source`、`independence_status`、`units`、`csv_sha256`（CSV 實際 SHA-256）。時間欄位皆為 Unix epoch 奈秒 UTC；座標為 WGS84 經緯度。任何缺檔、SHA 不符、時區不明、QC 不合格或無法對齊 catalog 時間窗都阻止相應圖型。

| 資料 | CSV 必要欄位 | 額外品質條件與圖型 |
| --- | --- | --- |
| 潮相：`traps.conditional.tidal_phase.v1` | `site_id,time_utc_ns,phase,spring_neap,qc_pass` | `phase` 為 `flood`／`ebb`；`spring_neap` 為 `spring`／`neap`；需覆蓋該區有效站點 × 全部有效 UTC 時刻，四組分層各列出分子／分母；零分母不畫。 |
| 軌跡：`traps.conditional.trajectory.v1` | `trajectory_id,time_utc_ns,lon,lat,source_kind,qc_pass` | 單檔只允許一種來源：`observed_drifter` 須 `independence_status: "confirmed_independent"` 才能標獨立觀測；`model_particle` 須 `model_internal` 或 `not_independent`。距 TRAP 與門檻內停留圖只採 UTC 同步點；12.5 km 距離門檻及相鄰 6 h 上限只是可覆寫工程預設。 |
| 同群模型粒子形變代理 | 使用上列 `model_particle` CSV | JSON 另需非空 `patch_id`，明示全表同一釋放群；至少三粒子在兩個以上 catalog 共同 UTC 時刻持續存在且非共線，局部跨度不超過 100 km。產出凸包面積比圖，**不等於** FTLE 或變形梯度。 |
| 海廢紀錄：`traps.conditional.marine_debris.v1` | `record_id,start_time_utc_ns,end_time_utc_ns,lon,lat,count,effort,qc_pass` | `effort > 0`，單位由 JSON `units` 指明；記錄期間需與 catalog 相交，且至少一筆落在該 flow domain 地理格網內。圖只描述時空共現，不作來源或因果推論。 |

三種 CSV 的 JSON `table_kind` 分別為 `tidal_phase`、`trajectory`、`marine_debris`。各類別實際缺項與拒絕原因見圖組的 [`figure_availability.json`](outputs/pilot/literature_figures_four_area_4frames_v5_colorbar/figure_availability.json)。若提供四區圖組，潮相 CSV 要涵蓋四區全部有效站點，軌跡與海廢 CSV 則各區分別檢查時間與空間對齊。不可把共用 OCM 強迫的 Lagrangian 模型粒子稱為獨立漂流器驗證。

## 目前產物、驗證與限制

需求 PDF、四篇文獻原件與紅框副本及來源清冊存於 [`data/literature/`](data/literature/README.md)；程式佇列與驗收界線見 [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)，[模組流程圖與文獻圖型對照](docs/MODULE_FLOW_AND_FIGURE_AUDIT.md) 逐項標明已可試跑和有資料 gate 的圖型。本機四區各 24 時刻的上游 `trial_partial_month` 用於來源與海岸 sidecar 檢查；四區各取同一 2025-01-01 UTC 四時刻進行檢出，並以[海岸線與等高色條版圖組](outputs/pilot/literature_figures_four_area_4frames_v5_colorbar/figure_availability.json)及[單時刻圖](outputs/pilot/snapshot_a_20250101_t0100_v5_colorbar/snapshot_000000.png)核對 A–D 圖面。這些產物一律是 `PILOT_ONLY` 工程試算，不是 2024–2025 正式 TRAP 統計。全套測試 61 項通過，圖組 27 個 artifact 的 SHA-256 與來源 catalog、研究設計設定雜湊已核對；四區科學摘要 JSON、五站摘要及概念圖和前版位元組一致。正式資料全期計算、跨月份生命週期、跨參數 robust 分類、速度單位確認，以及獨立表層觀測與潮相對照仍未完成；不能據此宣稱海廢濃度或污染來源。
