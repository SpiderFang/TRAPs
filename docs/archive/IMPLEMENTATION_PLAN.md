# TRAPs 程式實作與驗收計畫

## 已確認的研究契約

- 輸入為 `OCM-Data-Preprocessing` schema 3.0.0 的 `ocm_surface` 逐時表層 `u/v`、公尺制 `x_m/y_m`、靜態及逐時有效遮罩；不以 SVD 模態或三維 Lagrangian 流場替代。
- 四個流場區域及範圍沿用上游 `configs/ocm_flow_domains.json`。五站 ID、名稱、anchor 沿用 Lagrangian 正式設定。C 區站名為「南灣」，上游產品 ID 仍是 `houwan_nmmba_cache_v3`。
- 統計母域為整個 flow domain。A 區每個靜態有效格點依公尺距離歸於較近的貢寮或龜山島 anchor，兩站互斥且合起來覆蓋 A 區；相等時依固定站點順序歸貢寮。B、C、D 各自使用完整流場區域。此分割只定義統計歸屬，不裁切流場微分或骨架積分。
- `2024–2025` 為需求指定的分析期；沒有完整且通過 QC 的月份時，只能輸出明示的工程試算，不能聲稱兩年結果。`trial_partial_month` 僅可在顯式 pilot 選項下讀取，產物標為 `PILOT_ONLY`。上游速度單位待供應者確認時，強度與壽命等物理結論維持暫定。

## 一次性工作佇列

最多兩席 `luna_implementer` 並行；每席只負責下列明確的程式實作與對應驗證。主代理負責此計畫、介面整合及一次最終審核。

| 序號 | 所有權 | 交付內容 | 主要驗收 |
| --- | --- | --- | --- |
| 1 | 數值核心 | `src/traps/detection.py`、`src/traps/models.py` 與數值測試。遮罩內的公尺制速度梯度、對稱應變率、`s1/e2`、負局部極小核心、去重、沿 `e2` 雙向骨架積分、停止原因與 QC。 | 平移與剛體旋轉不產生有效吸引核心；解析應變的特徵值／方向正確；局地合成吸引場可檢出核心與曲線；無效遮罩不被跨越。 |
| 2 | 輸入與研究設計 | `src/traps/ocm.py`、`src/traps/sites.py`、`configs/study_design.json`、`pyproject.toml` 及資料／幾何測試。只讀 memory map，核對月份 metadata、schema、shape、時間、網格、QC，建立五站遮罩與來源鎖定資訊。 | 錯誤 domain/schema/shape/時間或未授權 trial 即拒絕；A 兩站遮罩互斥且聯集等於 A 靜態有效格點；C 名稱是南灣而產品 ID 不改。 |
| 3 | 時序與站點統計 | `src/traps/tracking.py`、`src/traps/metrics.py` 及測試。逐時結構一對一連結，時間缺口與邊界截尾分別登錄；五站有效快照、曲線交會、核心、長度、面積標準化及壽命統計保留分子分母。 | 不將缺時誤作結構消失；A 區不重複計入同一格點；分母為達到顯式覆蓋門檻的有效快照，零分母不輸出虛假比例。 |
| 4 | 執行與成果輸出 | `src/traps/cli.py`、`src/traps/figures.py`、`src/traps/__main__.py`、端到端測試。提供 preflight、單時刻／多時刻檢出、敏感度、統計、快照與五站彙整圖；輸出參數、來源、QC、檔案雜湊及 `PILOT_ONLY` 等狀態。 | 合成資料可端到端產出可回溯 catalog、站點表、圖與 manifest；正式模式拒絕 trial；圖示明示 `s1` 單位、時間、有效分母及來源狀態。 |

佇列 1、2 可同時開始；3 依賴 1、2 的資料介面；4 依賴前述模組。各 worker 不改其他工作所有權的程式檔，也不得覆蓋別人的變更。

## 共用介面與科學判讀

- `SurfaceMonth` 提供單時刻 `time_utc_ns, u, v, valid, x_m, y_m, lon, lat, cell_area_m2`，大陣列維持唯讀 memory map；不得整月載入或自動補缺。
- 上游 `mask_static.npy` 僅表示內插支撐與浴深有效，不能單獨視為海岸遮罩。若使用前處理專案產生的 `analysis_ocean_mask_<tag>.npy` sidecar，須驗證其網格、來源與版本，與逐時有效遮罩取交集；缺少海岸 sidecar 時，輸出須明示海岸 QC 未完成，不得把靠岸曲線作為正式結果。
- `detect_snapshot(u, v, valid, x_m, y_m, config)` 回傳 `SnapshotResult`，包括 `s1_sinv`、導數有效遮罩及 `TrapDetection` 清單；每個檢出項具有核心格點、公尺座標、`core_s1_sinv`、公尺制曲線點、長度、沿線平均強度與停止／品質標籤。所有門檻在設定或 manifest 中明列。
- `site_masks(domain_id, lon, lat, static_mask)` 只定義統計歸屬；TRAP 的應變率仍在完整上游 domain 內計算。站點出現率定義為「有效快照中，至少一條合格 TRAP 曲線與該站統計格點交會的快照數／該站有效快照數」。有效快照須符合顯式的逐時覆蓋率門檻，並記錄門檻及實際覆蓋率。
- 時序連結僅是 Eulerian 結構匹配，不是單一水團或廢棄物軌跡。壽命遇資料缺口、研究期起迄、品質不合格時須標記截尾。沒有獨立表層漂流器或海廢觀測時，圖表稱「表層短期吸引潛勢」。
- 敏感度至少包含 0、1、2、3 格平滑與兩組明示門檻；平滑限有效海域，不跨靜態陸地或逐時缺值。`robust` 須以預先登錄的跨參數重現準則判定，未完成該判定時不得貼此標籤。

## 驗收與剩餘風險

1. 先跑解析／合成流場單元測試及端到端 smoke test，再用本機既有 `trial_partial_month` 的**單一時刻**做顯式 `PILOT_ONLY` 讀取與圖面檢查；不得把該時刻推稱正式研究成果。
2. 核對五站遮罩、C 區命名、輸入 metadata 與圖表分母；檢查新增或修改程式檔的繁體中文註解、README 使用說明及輸出 manifest。
3. 正式兩年執行仍取決於上游完整月份及來源單位確認。參數門檻、海岸一格微分誤差和 tracking 分歧須透過敏感度及現地表層資料另行評估；此程式驗收不等於觀測驗證。
