# TRAPs：表層瞬態吸引剖面分析

> 註：本文以 `路徑` 純文字標示的檔案（如 outputs/、data/literature/original/、data/literature/annotated/、docs/assets/ 等）僅存於本機，未納入版本控制。

> **成果報告**：方法與實作見 `outputs/reports/01_TRAPs_實作方式報告.md`，2024–2025 年五站執行成果見 `outputs/reports/02_TRAPs_執行成果報告.md`；索引與引用方式見 `outputs/reports/`（`outputs/reports/README.md`），文件總覽見 [`docs/README.md`](docs/README.md)。
>
> **成果集中於 `outputs/`**：報告、彙整結果與分析表皆在 `outputs/`（被 git 忽略，不入版控），各資料夾說明與正式交付檔見 `outputs/README.md`；`docs/` 只保留方法與操作文件。

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

日產品的 25 點逐時置中等權平均屬本研究前處理設定，其濾波定義及應用先例依據 Parker（2007）、Ueki 等（2003）等來源，不能歸因於 Kunz（2024）。原件及七份 E–K 編號紅框導讀見 [補充文獻索引](data/literature/README.md#25-點逐時平均之補充文獻)，頻率響應及適用範圍見 [濾波文獻查核](docs/TRAPS_25H_FILTER_LITERATURE_AUDIT.md)。

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

`catalog.json` 保留原始公尺制曲線及逐時／站點分母；`catalog.geojson` 將同一曲線轉成 WGS84 經緯度。每個成果目錄須是新的或空的，以免舊圖與新參數混在同一份 manifest。單區 catalog 的域外站點雖保留零面積 schema，摘要圖只顯示該區有效站點；`summary` 可把**相同 UTC 時刻、相同檢出參數**的四區 catalog 合併為五站比較。跨月份同站點聚合與跨月份 track 接續目前尚未實作，因此不同月份不可直接交給 `summary` 當作完整壽命分析。完整[模組流程圖與文獻圖型對照](docs/archive/MODULE_FLOW_AND_FIGURE_AUDIT.md)包含條件資料的拒絕路徑。

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

使用者提供的 [`taiwan_exact_coastline.geojson`](data/taiwan_exact_coastline/taiwan_exact_coastline.geojson) 是 CRS84 的陸地 polygon；其[來源、授權與雜湊說明](data/taiwan_exact_coastline/README.md)已留存。圖面以它畫海岸輪廓，預設由專案 `data/taiwan_exact_coastline/` 讀取，亦可用 `--coastline-geojson` 指向其他已驗證的檔案。圖說須標示 © OpenStreetMap contributors／ODbL。此檔沒有高程或水深，且不取代 `--ocean-mask` 的格點 sidecar；數值檢出與分母仍依原有遮罩。資料讀取、繪圖及遮罩的關係見[模組流程圖](docs/archive/MODULE_FLOW_AND_FIGURE_AUDIT.md)。

### 成果圖色條版面

垂直色條須與其對應主圖的**實際座標框同高**；等比例地理圖應在 Matplotlib 完成比例配置後，再對齊色條上下界。雙色條圖的兩條色條均對齊同一主圖。此版面規則參考相鄰 `OCM-SVD-Analysis` 的主圖／色條配置，只調整繪圖幾何，不改色階上下限或科學數值。原本採自動刻度的圖可能隨色條縮短而調整顯示刻度密度，以免標籤重疊；實作與驗收界線見[色條對齊計畫](docs/archive/COLORBAR_ALIGNMENT_PLAN.md)。

### 文獻圖組的本機重現

下列命令讀取**已完成檢出**且同 UTC 四時刻、同參數與同 `PILOT_ONLY` 狀態的四區 catalog；命令會重開原月份與 coastline analysis sidecar 核對來源，另外載入本專案 GeoJSON 畫海岸輪廓。`--output-root` 必須為新目錄或空目錄，並產生每區方法、空間、觀測跨度、強度歷程圖，以及五站摘要、`figure_availability.json`、帶 SHA-256 的 manifest。概念示意為 `CONCEPT_ONLY`，其餘試算圖為 `PILOT_ONLY`。當三類附加資料缺席時，命令記錄 `DATA_REQUIRED`，不製造觀測或潮相圖。

```bash
cd /Users/mustlab/Workspace/TRAPs
.venv/bin/python -m traps literature-figures \
  --catalog outputs/_archive/pilot/local_mac_a_20250101_4frames_v1/catalog.json \
            outputs/_archive/pilot/local_mac_b_20250101_4frames_v1/catalog.json \
            outputs/_archive/pilot/local_mac_c_20250101_4frames_v1/catalog.json \
            outputs/_archive/pilot/local_mac_d_20250101_4frames_v1/catalog.json \
  --output-root outputs/_archive/pilot/literature_figures_four_area_4frames_v5_colorbar_repro
```

正式圖組應另指定正式 catalog 與全期 OCM 資料；四時刻試算只驗證介面與版面。A 區站點統計採最近 anchor 互斥分割，空間圖的母域仍是完整 A flow domain。圖組不跨月份接續 track；`duration_seconds` 是首末檢出觀測跨度。見[文獻實作與驗收記錄](docs/archive/LITERATURE_FIGURE_IMPLEMENTATION_PLAN.md)。

### 條件資料介面與圖型啟用

可選參數 `--tide-table`、`--trajectory-table`、`--debris-table` 各指向 UTF-8 CSV，且同目錄必須有**同 stem** JSON（例如 `tide.csv` 與 `tide.json`）。共通 JSON 欄位：`schema_version`、`table_kind`、`timezone: "UTC"`、非空 `source`、`independence_status`、`units`、`csv_sha256`（CSV 實際 SHA-256）。時間欄位皆為 Unix epoch 奈秒 UTC；座標為 WGS84 經緯度。任何缺檔、SHA 不符、時區不明、QC 不合格或無法對齊 catalog 時間窗都阻止相應圖型。

| 資料 | CSV 必要欄位 | 額外品質條件與圖型 |
| --- | --- | --- |
| 潮相：`traps.conditional.tidal_phase.v1` | `site_id,time_utc_ns,phase,spring_neap,qc_pass` | `phase` 為 `flood`／`ebb`；`spring_neap` 為 `spring`／`neap`；需覆蓋該區有效站點 × 全部有效 UTC 時刻，四組分層各列出分子／分母；零分母不畫。 |
| 軌跡：`traps.conditional.trajectory.v1` | `trajectory_id,time_utc_ns,lon,lat,source_kind,qc_pass` | 單檔只允許一種來源：`observed_drifter` 須 `independence_status: "confirmed_independent"` 才能標獨立觀測；`model_particle` 須 `model_internal` 或 `not_independent`。距 TRAP 與門檻內停留圖只採 UTC 同步點；12.5 km 距離門檻及相鄰 6 h 上限只是可覆寫工程預設。 |
| 同群模型粒子形變代理 | 使用上列 `model_particle` CSV | JSON 另需非空 `patch_id`，明示全表同一釋放群；至少三粒子在兩個以上 catalog 共同 UTC 時刻持續存在且非共線，局部跨度不超過 100 km。產出凸包面積比圖，**不等於** FTLE 或變形梯度。 |
| 海廢紀錄：`traps.conditional.marine_debris.v1` | `record_id,start_time_utc_ns,end_time_utc_ns,lon,lat,count,effort,qc_pass` | `effort > 0`，單位由 JSON `units` 指明；記錄期間需與 catalog 相交，且至少一筆落在該 flow domain 地理格網內。圖只描述時空共現，不作來源或因果推論。 |

三種 CSV 的 JSON `table_kind` 分別為 `tidal_phase`、`trajectory`、`marine_debris`。各類別實際缺項與拒絕原因見圖組的 `outputs/_archive/pilot/literature_figures_four_area_4frames_v5_colorbar/figure_availability.json`。若提供四區圖組，潮相 CSV 要涵蓋四區全部有效站點，軌跡與海廢 CSV 則各區分別檢查時間與空間對齊。不可把共用 OCM 強迫的 Lagrangian 模型粒子稱為獨立漂流器驗證。

### 全期流程（detect-shards → track-period → aggregate）

`detect` 一次只處理單一 domain-month 並攤成一份 JSON catalog，無法跨月份接續；全期
研究（4 流場區域 × 2024–2025 共 24 個月）改用三階段、可平行、可續跑的流程：

```bash
python -m traps detect-shards \
  --surface-root "$OCM_SURFACE_ROOT" --domain-id hsinchu_cache_v3 --month 202401 \
  --ocean-mask /path/to/analysis_ocean_mask_osm_coastline_v1.npy \
  --output-dir "$SHARDS_ROOT/hsinchu_cache_v3/202401" --resume

python -m traps track-period \
  --shards-root "$SHARDS_ROOT/hsinchu_cache_v3" --domain-id hsinchu_cache_v3 \
  --months 202401-202512 --output-dir "$TRACK_ROOT/hsinchu_cache_v3" --resume

python -m traps aggregate \
  --track-root "$TRACK_ROOT" --detect-root "$SHARDS_ROOT" \
  --output-dir "$AGGREGATE_ROOT" --resume
```

`aggregate`（`$TRACK_ROOT`／`$SHARDS_ROOT` 各自是四個 domain 子目錄的容器）是第三
階段，逐站獨立計算五站全期出現率、核心吸引強度、track 壽命與熱點統計，並輸出各站逐格空間分布與圖組：

`detect-shards` 對單一 domain-month 流式呼叫 `detect_snapshot`，寫出精簡的
`snapshots.npz`（逐時 coverage／detection 計數與 packbits 化 valid mask）、
`detections.npz`（逐 detection 的核心、曲線 ragged array、以 `curve_pass_cells` 算出的
格點交會集合、以及固定整數表的停止原因）與 `grid_reference.npz`；`manifest.json`
保留完整 provenance、`detection_config`、`min_coverage`、schema
`traps.period.shard.v2`（v1 shard 仍可被 `track-period` 讀取：核心維持在格點上，近岸距離場由 `analysis_ocean_mask` 現算；`aggregate` 會在 manifest `shard_schema_versions` 記錄）。輸出目錄必須是新的或空的；`--resume` 只在既有
`_SUCCESS` 通過 manifest artifact SHA-256 全部核對時略過重算，否則清掉本命令已知
會寫出的檔案後重做（例如任何 artifact 被竄改）。

`track-period` 讀入同一個 domain 的多個月份 shard，先依月份時序串接、以 UTC 時刻
做穩定排序，重複 UTC（月份交界常見）固定保留**較晚月份**的一列
（`sort_and_deduplicate_prefer_last`，與 `OCM-SVD-Analysis`、
`Lagrangian-Ensemble-Backtracking` 相同的正式研究期時間軸契約）；此步驟從不內插或
補缺，只記錄缺口。之後以 `StreamingTracker` 逐月串流建立跨月份 Eulerian track，寫出
`canonical_snapshots.npz`、`track_observations.npz`、`tracks.csv`、
`tracking_summary.json` 與 schema `traps.period.track.v1` 的 `manifest.json`
（含 canonical-axis 統計、缺口清單、逐份 shard manifest 的 SHA-256）。

#### Null model：時間置換（`--shuffle-seed`）

日平均去潮汐場上每個 snapshot 的核心密度高、最近鄰間距僅數 km，而追蹤門檻（核心
≤10 km、曲線 Hausdorff ≤20 km）相近，track 可能只是不相干核心的串接。
`track-period --shuffle-seed INT` 保留 canonical 時間軸、availability／coverage 與缺口，
但把每個可用槽位的偵測換成種子決定、僅在可用 snapshot 間置換的另一個 snapshot（
`detection_keys` 仍指向真實來源）。manifest 與 `tracking_summary.json` 記錄
`null_model: {"type": "time_shuffle", "seed": s}`；每次執行另輸出
`lifetime_hist.json`（未截尾壽命的名目步數計數與截尾數）。**`aggregate` 會拒絕**任何
`null_model` 非空的 track 輸入。比較用：

```bash
python scripts/null_test_tracking.py --output-dir "$OUT" \
  --runs real="$TRACK_REAL" shuf1="$TRACK_SEED1" shuf2="$TRACK_SEED2"
```

輸出 `comparison.csv`（含 `coherence_ratio` ＝ 真實 frac(壽命≥3 步) ÷ 置換平均）與逐
domain 壽命 CCDF 的 `comparison.png`；比值接近 1 代表 track 壽命與隨機串接無異。

#### 只追蹤各站最強核心（`--track-strong-quantile`）與 null 對照（`--null-track-root`）

時間置換檢驗顯示：追蹤**全部**合格核心（每 snapshot 約 15 個／1000 km²）並使用 10 km／20 km
門檻時，壽命多半可由隨機串接解釋（置換 p90 約 10 天、真實約 13 天）。因此壽命改為只追蹤各站最強核心，
並一律對照 null 報告：

* `track-period --track-strong-quantile Q`（預設不設＝追蹤全部合格核心，行為不變）：Pass 1 只讀欄位，
  以 `sites.site_masks` 依核心格點（`core_row`, `core_col`）分站，母體＝canonical 且 `coverage_pass` 快照的
  合格核心；各站門檻＝該站核心 s1 的 Q 分位數（越負越強）。Pass 2 只讓 `core_s1 <=` 所屬站門檻的合格
  detection 進入 tracker，站外核心一律丟棄。門檻取自**真實** canonical 集合，與 `--shuffle-seed` 併用時真實與
  null 相同。逐站門檻（s⁻¹、day⁻¹）、Q、各站保留數、丟棄數記於 manifest 與 `tracking_summary.json` 的
  `track_core_filter`，且與 `aggregate --strong-quantile Q` 的逐站強核心門檻同一母體、位元相同。
* `aggregate --null-track-root DIR [DIR ...]`：每個 DIR 是 `track-period --shuffle-seed` 的 track 根目錄
  （含 `<domain>/`），必須與真實 track-root 同 detection／tracking 設定與 `track_core_filter`（僅 seed 不同），
  否則拒絕；主要 `--track-root` 仍不可是 null。stratum `all` 的 `site_statistics` 增加 `null_n_seeds`、
  `null_n_uncensored`、`null_median_lifetime_hours`、`null_p90_lifetime_hours`、`null_track_frac_uncensored_ge_<N>h`
  （跨 seed 平均）、`excess_track_frac_uncensored_ge_<N>h`（真實比例 − null 比例）與
  `ratio_track_frac_uncensored_ge_<N>h`（真實 ÷ null，null 為 0 時空白）；其他 stratum 為空。壽命圖在真實未截尾
  CCDF 上疊虛線灰色 null CCDF（跨 seed 平均，最小–最大帶）並標註各門檻的超額。manifest 記錄
  `null_inputs`（路徑、seed、各 domain manifest 的 sha256）、`track_core_filter` 與 `lifetime_scope_zh`。
  track 經強核心篩選時，圖表與 `site_statistics.json` 註明壽命指「強核心結構（各站前 Q%）」。
* `scripts/null_test_tracking.py` 的 CSV 增加 `core_filter_q`，標籤含分位數，且只與同 (domain, 門檻, q) 配對。

前兩階段輸出的欄位命名與陣列 schema 是為 `aggregate` 保留的完整證據鏈；`aggregate`
讀入所有 4 個 domain 的 `track-period` 輸出（`--allow-partial-domains` 可放寬為部分
domain，缺席站點會明列於 manifest），核對四區共用同一組 `param_id`、
`detection_config`、`tracking_config`、`min_coverage`、`state_label`（拒絕
`PILOT_ONLY` 與正式資料混用），並重新核對每個 shard manifest 的 SHA-256（偵測
`track-period` 完成後 shard 被竄改的情況）。

五個研究站彼此**獨立**，`aggregate` 不做跨站比較，也不跨站 pooled 任何門檻或色階。
每個站的統計範圍是它自己的「站點分析區」（`scope = site_area`）：直接沿用
`sites.site_masks()` 對 `analysis_ocean_mask` 的 flow-domain 分割，B／C／D 為整個
flow domain 的分析海域，A 區由貢寮與龜山島依最近 anchor 互斥分割；不使用任何接收區
半徑。每個 `site_id × stratum`（`stratum` 為 `all`、四季 `DJF/MAM/JJA/SON`、以及實際
出現的年份）輸出 `site_statistics.csv`／`site_statistics.json`：有效快照分母（站點
分析區內有效格點比例達 `min_coverage` 的 canonical 快照）、核心／強核心／曲線出現率
與分子（**強核心門檻是該站自己**分析區內、canonical `coverage_pass` 快照的合格核心
`s1` 的 `--strong-quantile` 分位數，預設最強 10%，各站門檻以 s⁻¹ 與 day⁻¹ 記入
manifest 的 `strong_core_thresholds` 與統計表）、`mean_pass_cell_fraction`（三個出現率
與此欄位均以 day-block bootstrap 給 95% CI）、核心強度統計（count/mean/median/p10）、
面積標準化核心密度、未截尾 track 的壽命統計（mean/median/p90，
`lifetime_seconds = duration_seconds + Δt`，與 `track-period` 的 track 壽命慣例一致；
track 依其首個落在該站分析區內的觀測時間指派 stratum），以及熱點指標：熱點＝該站
分析區內 `curve_pass_frequency` 最高 `1 - --hotspot-quantile`（預設前 10%）的格點，
輸出 `hotspot_threshold_curve_pass_frequency`、`hotspot_cell_count`、`hotspot_area_km2`、
熱點內平均曲線通過頻率、熱點內平均核心 s1（day⁻¹）、以及該站強核心落在熱點格點的
比例（每個分層各自以該分層的場定義熱點）。

`spatial/<site_id>/spatial_<all|DJF|MAM|JJA|SON>.npz` 保留該站分析區內的逐格有效分母、
核心／強核心頻率、曲線通過頻率、平均核心強度與平均未截尾壽命（站外與沒有有效分母的
格點固定是 `NaN`，不假裝是零），並包含 `site_mask`、`hotspot_curve_mask`、
`hotspot_core_mask` 與各門檻。

`--no-figures` 可略過圖面只留統計表；預設每站輸出一組五張圖到
`figures/<site_id>/`：`site_<id>_spatial.png`（站點分析區內 2×2：曲線通過頻率、核心
頻率、平均核心 s1、平均未截尾壽命；只標 anchor，不畫圓）、`site_<id>_hotspots.png`
（曲線通過頻率場上標出熱點格點，疊加核心頻率前 10% 的輪廓，圖內統計標示熱點面積
km²）、`site_<id>_seasonal.png`（DJF/MAM/JJA/SON 曲線通過頻率小圖，以及季節強核心
出現率與平均通過格點比例含 95% CI）、`site_<id>_lifetime.png`（壽命 CCDF，未截尾實線、
含截尾虛線；以及壽命 ≥ 12 h 未截尾 track 的平均核心 s1（day⁻¹）對正規化年齡 0–1
的演化，Kunz et al. 2024 Fig. 5b 風格，標示 track 數）、`site_<id>_timeseries.png`
（逐日含強核心的有效快照比例與平均通過格點比例，30 日滑動平均，資料缺口以斷線呈現）。
各圖色階上下限是該站自己分析區的 2–98 百分位（記錄於 manifest `colour_limits`），
海岸線疊圖與色條對齊沿用既有 `coastline.py`／`colorbar_layout.py`。

**WP-H 逐站追加圖與統計**（預設輸出，皆逐站獨立、色階取自該站）：

* `site_<id>_example.png`（Kunz 2024 Fig. 2 風格）：代表 track ＝該站「首個觀測在站內、未截尾」的真實 track 中壽命最長者
  （同壽命取平均核心 s1 最強；優先限定壽命 ≥ `--lifetime-thresholds-hours` 最大值，無則取最長）。於其起始時刻經 shard manifest
  的來源 provenance 重開 OCM 月份與海岸遮罩（核對 SHA-256；可用 `--surface-root` 覆寫搬移後的路徑），以該 shard 的
  `detection_config` 重算該 snapshot，重算核心與已存核心須相差 ≤ 1 格，否則明確失敗。圖含 s1（day⁻¹）、抽稀速度箭頭、全部合格曲線（淡）、
  代表曲線與核心（紅粗線）、核心路徑，以及約 1/3、2/3 壽命與最後觀測的曲線（較淺色，標「+6 h」或「+3 天」）。站內無候選 track 時輸出標示原因的空白圖。
* `site_<id>_method_comparison.png`（Serra 2020 Fig. 4d 風格）：同一 snapshot 的 (a) s1＋TRAP、(b) 流線（streamplot）＋TRAP、(c) 水平散度（day⁻¹）＋TRAP；
  以圖內必要統計與分圖標題說明方法差異，流線不是粒子軌跡、散度不同於 TRAP 吸引的完整說明留在報告。
* `site_<id>_long_tracks.png`（Kunz Fig. 4b）：未截尾、站內出生、壽命 ≥ `--long-track-hours`（預設＝最大壽命門檻）的核心路徑，依壽命著色；超過 2000 條只畫最長者並以圖內統計標示；
  給 `--null-track-root` 時保留 null 各 seed 平均條數的必要圖內統計。
* `site_<id>_track_density.png`（Kunz Fig. 4c）：通過各格的不同 track 數（全部真實 track／長壽命 track；每 track 每格計一次）；同時寫入 spatial npz 的 `track_count`、`long_track_count`。
* `site_<id>_propagation.png`（Kunz Fig. 5c 類比）：≥3 觀測 track 的核心中心差分傳播速度（東向 cx、北向 cy，km/day，不含首末觀測）直方圖與 16 扇區去向玫瑰圖。
  `site_statistics` 新增 `prop_n`、`prop_cx_median_kmday`、`prop_cy_median_kmday`、`prop_speed_median_kmday`、`prop_speed_p90_kmday`（僅 stratum `all`）。
* 旗標：`--no-example-figures`（略過需重開 OCM 的前兩張）、`--surface-root`、`--long-track-hours`。manifest 的 `site_track_figures` 記錄各站代表 track、重算核心偏移與實際來源目錄。

**WP-I 子格點核心定位與近岸診斷**（結果圖不得引起疑義）：

* **子格點核心**：TRAP 核心原本恰落在 1 km 格點上，逐 track 傳播速度因此被格點量化。`detect_snapshot` 現以核心格點的 3×3 二次曲面最小平方擬合（偏移以格距換算為公尺）求極值；Hessian 須正定且偏移 ≤ 0.5 格，否則保留格點並標 `refined=False`。`DetectionConfig.core_subgrid_refine`（預設開，`--no-core-subgrid-refine` 關閉）；manifest `detection_config.core_subgrid = "quadratic_3x3"`（關閉時 `"none"`）。`detections.npz` 新增 `core_grid_x_m`／`core_grid_y_m`（格點位置）、`core_refined`、`core_coast_distance_m`，`core_x_m`／`core_y_m` 為子格點位置；`grid_reference.npz` 新增 `coast_distance_m`（精確歐氏距離變換，飽和於 200 km（僅為速度防護，涵蓋整個 domain；舊 shard 為 30 km），雙線性取樣）。`track_observations.npz` 新增 `core_coast_distance_m`。
* **傳播速度**：改為逐 track 對核心位置（子格點）做時間線性最小平方擬合（≥ `--prop-min-obs` 個觀測，預設 3）；擬合位移在壽命內小於 1 格（`grid_cell_size_m`）者判為**靜止**，不進入方向玫瑰與速度直方圖（只用移動 track；圖上標示移動／靜止條數、靜止比例與殘差 RMS，避免零速度樣本落入「北」向）。`site_statistics` 新欄位：`prop_n_tracks`、`prop_n_moving`、`prop_n_stationary`、`prop_stationary_frac`、`prop_resid_rms_median_m`（原 `prop_n` 由 `prop_n_tracks` 取代；`prop_cx/cy/speed_*` 僅含移動 track）。
* **近岸診斷**（`--coastal-buffer-km`，預設 10）：以核心平均距岸小於緩衝帶者為**近岸型** track。**檢出下限**：偵測需要完整有效的導數模板與局部極小鄰域（實質 2–3 格），TRAP 核心不可能落在距非海域格點約 2.5–3 km 以內，因此舊預設 3 km 只剔除核心本來就不存在的區域，並未檢驗近岸影響；實測強核心與長壽命 track 集中在距岸 3–10 km，故預設取 10 km（10 km 緩衝下，近岸型與近岸外長壽命 track 仍都超出隨機對照）。近岸帶以 track 的核心平均距岸判定，近岸型與近岸外皆與隨機對照比較。新增 `frac_cores_within_buffer`、`frac_strong_cores_within_buffer`、`frac_long_tracks_within_buffer`、`long_track_n_born_uncensored`、`offshore_strong_core_occurrence_rate`（強核心門檻仍為全站同一門檻）、`hotspot_frac_area_within_buffer`、`offshore_hotspot_cell_count`、`offshore_hotspot_mean_curve_pass_frequency`（近岸外熱點：只在站內緩衝帶外格點重取分位數），以及 `coastal_`／`offshore_track_n_uncensored`、`{coastal,offshore}_track_frac_uncensored_ge_<N>h`；給 `--null-track-root` 時另有 `null_{coastal,offshore}_*` 與 `excess_offshore_*`。spatial npz 新增 `coast_distance_m`、`coast_buffer_mask`、`coast_buffer_m`。
* **圖面**：所有站圖疊加虛線緩衝帶輪廓（圖例「近岸 10 km 帶」，實際值隨 `--coastal-buffer-km`）；`long_tracks` 以虛線畫近岸型、實線畫近岸外，並標示近岸型／近岸外的真實與 null 條數；代表 track 優先取近岸外且壽命高於 null p99 者（準則寫入 manifest；否則退回近岸型並在圖上註明「（近岸型）」），另輸出 `site_<id>_example_coastal.png`；後續核心位置過於接近時，時間標籤改列為圖例中的位移清單；新增 `site_<id>_coastal_diagnostic.png`（距岸直方圖與 CCDF：全部／強核心／長壽命，近岸外真實 vs 近岸型 null；直方圖涵蓋整個距離場，末端以「≥ 上限」長條明確標示，並以灰色帶標出檢出下限＝站內合格核心最小距岸距離）。
* **判讀**：長壽命 track 多為近岸靜止型時，代表其為地形鎖定的應變場或邊界效應，**不是**漂移中的收斂帶；請以近岸外指標（`offshore_*`）作為保守結論依據，近岸型結果僅作為診斷並須連同靜止比例一併報告。

圖面文字為繁體中文，字型由 `report_font.resolve_cjk_font()` 依固定順序
（Noto Sans CJK TC、Noto Sans TC、Source Han Sans TC、PingFang TC、Heiti TC、
Arial Unicode MS）解析，並以 FreeType charmap 檢查涵蓋圖面所有固定文字與五站名稱；
選定字型的家族、檔名 basename 與檔案 SHA-256 記入 manifest `font`；字型、來源與
track 編號等 provenance 不在圖面重複印出。
找不到合格 CJK 字型時 `aggregate` 在寫任何檔案前以 `ValueError` 停止（圖面絕不輸出
方框字）；改用 `--no-figures` 則不需要字型。輸出目錄同樣必須是新的或空的（或
`--resume`），manifest schema 為 `traps.period.aggregate.v2`。所有輸出仍保留輸入的
`state_label`／科學狀態，圖說與 JSON 一律標示「MODEL_ONLY 表層短期吸引潛勢」或
`PILOT_ONLY`，不是實測海廢濃度或污染來源。

### 成果圖版面與註腳規則

正式 aggregate 圖面只保留主標題、分圖標題、圖例、座標軸與單位、色條，以及解讀
所需的圖內統計標示；圖下方不再配置資料表、描述性註腳、追蹤編號或來源文字，因而
不為這些文字預留底部高度。代表 track、來源路徑、字型雜湊、科學狀態與 OSM attribution
留在 `manifest.json`、`figure_presentation` 與方法/成果報告。既有 aggregate 可用
`scripts/refresh_period_figures.py render` 在新目錄產生版面檢查紀錄，再以 `publish`
通過非 PNG 數值 artifact、輸入指紋、代表 track 與色階界限的逐項比對後只替換 PNG；
工具不會重新執行 detect、track 或 25 h 日平均。

### 去潮汐日平均產品（build-daily-fields）

逐時流場以潮汐為主；TRAP 文獻使用日快照。`build-daily-fields` 由逐時 `ocm_surface` 建立
同 schema 的 25 h 置中等權平均日產品（每日 12:00 UTC 為目標時刻、視窗內至少 23 小時、格點須
全視窗有效；`cache_kind = derived_daily_mean`），`detect-shards`／`track-period`／`aggregate`
可直接在其上執行（追蹤請用 `--nominal-time-step-hours 24 --tracking-max-time-gap-hours 36`）：

```bash
python -m traps build-daily-fields --surface-root "$OCM_SURFACE_ROOT" \
  --domain-id hsinchu_cache_v3 --output-root "$DAILY_ROOT" --months 202401-202512 --resume
```

25 h 平均強烈抑制 M2/S2/K1/O1，但不是理想低通；濾波參數與被省略日期記在各月 `metadata.json`。
SERVER 用法見 `docs/operations/SERVER_RUNBOOK.md` 的 DAILY 一節（`TRAPS_FIELD=daily25h`）。

## 目前產物、驗證與限制

2024-01 至 2025-12 的全期分析已完成：逐時產品（`r1_hourly`）與 25 h 去潮汐日產品（`d1_daily25h`）各自完成 4 個流場區域 × 24 個月的偵測、追蹤與五站彙整（程式版本 `0f980cb`（改寫前為 `99c3151`）），全套測試 254 項通過。方法與實作見 `outputs/reports/01_TRAPs_實作方式報告.md`，成果、圖與限制見 `outputs/reports/02_TRAPs_執行成果報告.md`；索引與引用方式見 `outputs/reports/README.md`。早期 `PILOT_ONLY` 試算文件已歸檔於 [`docs/archive/`](docs/archive/IMPLEMENTATION_PLAN.md)，不代表目前結果。

限制：結果僅來自模型流場（`MODEL_ONLY`），沒有獨立的漂流器或海廢觀測驗證，只能稱為「表層短期吸引潛勢」，不得寫成海廢濃度、長期滯留、海底廢聚集或污染來源；日產品壽命的 2–3 天部分可由隨機串接解釋，近岸 2–3 km 內存在檢出下限。
