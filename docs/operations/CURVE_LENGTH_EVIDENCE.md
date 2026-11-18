# 曲線每側長度上限（20 km）的依據：唯讀統計腳本與 SERVER 執行步驟

回答審查可能的追問：「TRAP 曲線每側長度上限 20 km 的依據是什麼？」

腳本：`scripts/curve_length_evidence.py`（測試：`tests/test_curve_length_evidence.py`）。
**只讀**既有的 DETECT shard、TRACK 的 `canonical_snapshots.npz` 與正式 aggregate 輸出，不重算檢出、不改任何既有目錄，
結果只寫到 `--output-dir`（三個檔案：`curve_length_evidence.json`、`curve_length_evidence.csv`、`summary_zh.md`）。

## 1. 報告所用的正式 run／aggregate（判定依據）

| 項目 | 內容 |
| --- | --- |
| param_id | `r1_hourly`（逐時產品） |
| aggregate | **`aggregate_shared10`**（`aggregate --shared-strong-core-s1-day=-10 --no-figures`，2026-11-05 在 .203 跑完，`git_commit` 558940f，位於 `traps-repo-shared10`） |
| SERVER 路徑 | `/home/mustlab/Workspace/traps-runs/runs/r1_hourly/aggregate_shared10` |
| Mac 鏡像 | `outputs/shared_threshold_trial_20261105/aggregate_shared10` |
| DETECT shard | `/home/mustlab/Workspace/traps-runs/runs/r1_hourly/detect/<domain>/<YYYYMM>/` |
| TRACK | `/home/mustlab/Workspace/traps-runs/runs/r1_hourly/track/<domain>/` |
| 強核心 | 合格核心 s1 ≤ −10 day⁻¹（五地共用） |
| 聚集熱區 | 各地點統計範圍內「強核心曲線出現頻率」（`strong_curve_pass_frequency`）前 10 % 格點（`_top_quantile_mask`，分位數 0.9） |

判定依據：報告表 1 的強核心比例（龜山島 16.4 %、貢寮 17.0 %、新竹 0.65 %、南灣 8.1 %、連江 3.8 %）與 `aggregate_shared10/manifest.json`
的 `shared_strong_core.sites.*.strong_core_fraction` 逐一相符（龜山島 16.37 %、貢寮 17.04 %、新竹 0.649 %、南灣 8.12 %、連江 3.81 %）。
舊的 `outputs/server/r1_hourly/aggregate_report` 用的是各地點自己的分位數門檻（龜山島 −13.4、貢寮 −14.8、新竹 −3.0、南灣 −8.4、連江 −6.3 day⁻¹），
比例不同，**不是**報告所用。報告的熱區圖與統計腳本（`render_r10_final.py`、`render_abs2_formal.py`、`stats_abs5d2.py` 等）也都以 `aggregate_shared10` 為輸入。

`aggregate_shared10` 與舊 `aggregate_report` 讀的是同一份 `r1_hourly/detect` 與 `track`；它只多了 `--shared-strong-core-s1-day`（opt-in，不影響其他欄位）。
本腳本不依賴這個 patch，只讀它產生的 `spatial/<site>/spatial_all.npz`（需有 `strong_curve_pass_count`）與 `manifest.json`。

manifest 記錄的核對值（腳本會逐項比對）：

| 地點 | 合格核心數 | 強核心數 | 強核心比例 | 有效快照數 |
| --- | ---: | ---: | ---: | ---: |
| 龜山島（guishan） | 968,760 | 158,540 | 16.37 % | 17,124 |
| 貢寮（gongliao） | 1,469,434 | 250,366 | 17.04 % | 17,124 |
| 新竹（hsinchu） | 3,528,213 | 22,896 | 0.65 % | 17,124 |
| 南灣（nanwan） | 3,584,375 | 291,020 | 8.12 % | 17,124 |
| 連江（lienchiang） | 2,737,928 | 104,352 | 3.81 % | 17,124 |

## 2. 腳本做什麼

1. **停止原因與分支長度**（每地點；「全部 TRAP」與「強核心 TRAP」兩組）：曲線數、分支數、各停止原因的分支數與比例
   （特別是 `max_curve_length`）、核心兩側（backward／forward）分開的分支長度中位數／P90／P99（點列累計弧長，1 m 解析度、最近秩法）、
   至少一側因 `max_curve_length` 停止的曲線比例、長度 ≥ 10／15 km 的分支比例。
   母體與 aggregate 的 `core_count_total`／`strong_core_count_total` 相同（canonical 且 coverage_pass 的有效快照、通過地點涵蓋率門檻、核心格點在地點 site mask 內、合格曲線）。
2. **截短敏感度**（不重算檢出）：每條強核心曲線的兩側各自截短到 L（預設 10、15 km；基準＝原上限 20 km），
   以 `curve_pass_cells`（內含 `densify_polyline`）重算通過格點，再用與 aggregate 相同的方法算強核心曲線出現頻率與前 10 % 熱區。
   截短只作用在弧長超過 L（加 2 m 浮點容差）的分支，於恰好 L 處線性內插終點；沒被截短的曲線直接沿用 shard 內原有的 `pass_cells`。
   輸出：與基準熱區之 Jaccard、基準熱區格點保留比例、熱區格點數、熱區面積位於近岸 10 km 帶內的比例、被截短的曲線比例、總出現量相對基準。
3. **一致性檢查（必要）**：基準（L＝原上限）的重算結果必須與正式 `spatial_all.npz` 逐格相同
   （`n_valid`、`strong_curve_pass_count`、`core_count`、`strong_core_count`、頻率、熱區遮罩，以及 manifest 的核心數與有效快照數）。
   不一致時印出差異（格點座標、正式值、重算值），仍寫出結果檔但摘要首行標示「不可使用」，並以**狀態碼 2** 結束。
   只跑部分月份時（`--months`）退化為包含關係檢查（重算值不得大於正式全期值）。
4. **（選用）加長檢查**：`--extended-detect-root` 指向以 `--max-curve-length-m 40000` 重跑 DETECT 的樣本月份，
   與同月份原上限比較：強核心分支停止原因、加長版分支長於 20 km 的比例、兩者熱區 Jaccard，
   並附「樣本月份基準熱區 vs 全期基準熱區」的 Jaccard 作為抽樣雜訊參考。
   腳本會驗證加長版 shard 的 `detection_config` 除 `max_curve_length_m` 外與正式 shard 完全相同、快照集合與有效遮罩相同。

### 事先固定之判讀門檻（寫在腳本常數 `PRE_REGISTERED` 與輸出 JSON 的 `decision_rules`，不依結果調整）

* 強核心分支中因 20 km 上限而停止者 < 5 % →「多數曲線由 30 % 強度條件截止，長度上限僅影響少數曲線」。
* 截短至 15 km 時各地點熱區 Jaccard ≥ 0.8 →「聚集熱區位置對長度上限不敏感」。

腳本逐地點照實輸出是否達標（表 D），不挑地點、不改門檻。摘要文字只列表格與門檻，不下結論句。

## 3. 在 SERVER 上執行

SERVER：`mustlab@140.117.88.203`；工作區 `/home/mustlab/Workspace`。腳本是單一檔案，**不放進 `traps-repo`**
（`PREPARE`／`DETECT` 會檢查 repo 乾淨，放進去會變髒），放在 `traps-runs/tools/`，用 `traps-repo` 的 `.venv` 與 `src`（唯讀 import）執行。
所需的既有函式（`traps.period.curve_pass_cells`／`STOP_REASON_CODES`／`_canonical_valid_snapshot_indices_by_month`、
`traps.aggregate._load_domain_grid`／`_build_domain_masks`／`_top_quantile_mask`／`_ragged_gather`、`traps.sites.load_study_design`）
在 `traps-repo` 目前版本（cff6128，src 與 99c3151 相同；歷史改寫後分別為 d930398、0f980cb）皆已存在，**不需要先同步其他檔案**。

### 3.1 把腳本帶上 SERVER（在 Mac）

```bash
cd /Users/mustlab/Workspace/TRAPs
shasum -a 256 scripts/curve_length_evidence.py
rsync -e ssh scripts/curve_length_evidence.py \
  mustlab@140.117.88.203:/home/mustlab/Workspace/deploy-bundles/curve_length_evidence.py
```

（與 `shared_threshold_trial_20261105/RUN_ON_203.md` 同一種做法；之後若這個腳本被 commit 進 repo，改走 `SERVER_RUNBOOK.md` 的 git bundle 即可。）

### 3.2 設定環境（在 SERVER，建議 tmux）

```bash
ssh mustlab@140.117.88.203
tmux new -s clev        # 斷線後 tmux attach -t clev

mkdir -p /home/mustlab/Workspace/traps-runs/tools
cp /home/mustlab/Workspace/deploy-bundles/curve_length_evidence.py /home/mustlab/Workspace/traps-runs/tools/
sha256sum /home/mustlab/Workspace/traps-runs/tools/curve_length_evidence.py     # 與 Mac 的 shasum 相同

export R=/home/mustlab/Workspace/traps-runs/runs/r1_hourly
export PY=/home/mustlab/Workspace/traps-repo/.venv/bin/python
export PYTHONPATH=/home/mustlab/Workspace/traps-repo/src PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

### 3.3 先用單一 domain-month 試跑（約 1 分鐘）

```bash
$PY /home/mustlab/Workspace/traps-runs/tools/curve_length_evidence.py \
  --aggregate-dir $R/aggregate_shared10 --detect-root $R/detect --track-root $R/track \
  --output-dir $R/curve_length_evidence_trial \
  --domains hsinchu_cache_v3 --months 202401
echo "exit=$?"        # 0 = 正常；2 = 一致性檢查未通過；其他 = 輸入或參數錯誤
cat $R/curve_length_evidence_trial/summary_zh.md
```

試跑是「部分月份」模式（只檢查包含關係），摘要內會標示「數字僅供試跑」。確認 shard 驗證（SHA-256 與 aggregate manifest 記錄相符）、表格欄位與進度顯示正常即可。

### 3.4 全期執行（逐 shard 串流，記憶體只需數百 MB）

```bash
$PY /home/mustlab/Workspace/traps-runs/tools/curve_length_evidence.py \
  --aggregate-dir $R/aggregate_shared10 --detect-root $R/detect --track-root $R/track \
  --output-dir $R/curve_length_evidence_20261106 \
  --lengths-km 10 15 \
  2>&1 | tee $R/logs/curve_length_evidence_20261106.log
echo "exit=${PIPESTATUS[0]}"
```

進度會逐 domain-month 印到 stderr（本月秒數、累計分鐘、預估剩餘）。可用 `--domains`（domain ID 或地點 ID）與 `--months`
（`YYYYMM`、`YYYYMM-YYYYMM`、逗號分隔）縮小範圍。輸出目錄已有同名檔案時需加 `--overwrite`。

預估耗時（**未在正式資料上實測**）：正式 aggregate 全期約 4–5 分鐘（含全部彙整）；本腳本多做分支幾何（本機以 20 萬條 × 60 點測得約 0.4 秒，
推算 1,230 萬條合格曲線約 1 分鐘）與強核心曲線截短重算（本機每條約 0.1 毫秒，約 83 萬條強核心 × 2 個 L，最多約 3 分鐘），
預估單一程序全期約 10–25 分鐘；試跑（單一 domain-month）約 1 分鐘。實際以試跑的「本月秒數 × 96」為準。

### 3.5（選用）產生 40 km 樣本月份，做加長檢查

對每個 domain 每季一個月（2024-01、04、07、10）以 `--max-curve-length-m 40000` 重跑 DETECT，輸出到**新的 param_id**
（`runs/r1_hourly_len40k_sample/detect/<domain>/<YYYYMM>`），完全不碰 `runs/r1_hourly`。照 runbook 的 `detect-shards` 參數，只改曲線長度上限：

```bash
# 先確認正式 shard 沒有額外的 DETECT 參數（只應有預設值；max_curve_length_m = 20000.0）
python3 -c "import json;print(json.load(open('$R/detect/hsinchu_cache_v3/202401/manifest.json'))['detection_config'])"
df -h /home | tail -1; du -sh $R/detect/hsinchu_cache_v3/202401      # 40 km 版曲線點較多，shard 預期略大於 20 km 版

cd /home/mustlab/Workspace/traps-repo
TRAPS_PARAM_ID=r1_hourly_len40k_sample \
TRAPS_MONTHS=202401,202404,202407,202410 \
TRAPS_DETECT_ARGS="--max-curve-length-m 40000" \
TRAPS_JOBS=16 \
scripts/run_traps_server.sh DETECT
scripts/run_traps_server.sh STATUS    # 需用同樣的 TRAPS_PARAM_ID／TRAPS_MONTHS 環境變數
```

這會展開成 4 個 domain × 4 個月 = 16 個工作，每個工作等同 runbook 的
`python -m traps detect-shards --surface-root … --domain-id <domain> --month <YYYYMM> --ocean-mask … --output-dir $TRAPS_RUNS/runs/r1_hourly_len40k_sample/detect/<domain>/<YYYYMM> --param-id r1_hourly_len40k_sample --resume --max-curve-length-m 40000`。
單一 domain-month 單核約 12 分鐘，16 個平行約 15–25 分鐘（40 km 版略慢）。若正式 run 當時有額外的 `TRAPS_DETECT_ARGS`，要同樣帶入
（腳本會檢查 `detection_config`，不同就拒絕）。

完成後加上 `--extended-detect-root` 再跑一次（可與 3.4 合併成同一次執行）：

```bash
$PY /home/mustlab/Workspace/traps-runs/tools/curve_length_evidence.py \
  --aggregate-dir $R/aggregate_shared10 --detect-root $R/detect --track-root $R/track \
  --extended-detect-root /home/mustlab/Workspace/traps-runs/runs/r1_hourly_len40k_sample/detect \
  --output-dir $R/curve_length_evidence_20261106 --lengths-km 10 15 \
  2>&1 | tee $R/logs/curve_length_evidence_20261106.log
```

### 3.6 結果拉回 Mac（只加不刪，不用 `--delete`）

```bash
mkdir -p /Users/mustlab/Workspace/TRAPs/outputs/curve_length_evidence_20261106
rsync -av mustlab@140.117.88.203:/home/mustlab/Workspace/traps-runs/runs/r1_hourly/curve_length_evidence_20261106/ \
  /Users/mustlab/Workspace/TRAPs/outputs/curve_length_evidence_20261106/
rsync -av mustlab@140.117.88.203:/home/mustlab/Workspace/traps-runs/runs/r1_hourly/logs/curve_length_evidence_20261106.log \
  /Users/mustlab/Workspace/TRAPs/outputs/curve_length_evidence_20261106/
```

## 4. 輸出內容

* `summary_zh.md`：可直接貼上的中文摘要（表 A 停止原因、表 B 分支長度、表 C 截短敏感度、表 D 事先固定門檻與各地點達標與否、表 E 加長檢查）。
* `curve_length_evidence.csv`：每地點一列（欄位前綴 `all_`／`strong_`；`L10_`／`L15_`／`L20_` 為各 L 的熱區指標；`rule_*_met`；`consistency_ok`）。
* `curve_length_evidence.json`：完整數值（含兩側分開的停止原因、輸入路徑與 SHA-256、定義說明 `definitions`、判讀門檻 `decision_rules`、一致性檢查細節、診斷值 `diagnostics`、腳本自身 SHA-256）。

`diagnostics` 的意義：`core_locate_max_dist_m`（核心在點列中的定位誤差，應為 0.0x m 內）、`length_vs_stored_max_abs_diff_m`
（點列弧長與 shard 內 `curve_length_m` 的最大差，float32 座標造成，應 < 1 m）、`pass_cells_recompute_*`（每 50 條強核心曲線抽樣，
以 float32 點列重算 `pass_cells` 與 shard 存值比對；不為 0 時代表格點邊界取樣點差一格，是截短曲線重算的方法雜訊量級）、
`n_curves_branch_over_cap_plus_tol`（弧長超過原上限 + 2 m 的曲線數，應為 0）。

## 5. 限制與注意

* 截短是「對既有曲線取前 L」的近似：正式 DETECT 在 L 處停止時，曲線與截短後的點列在 L 之前相同（積分是單向、與長度上限無關），
  所以截短等同於以 L 為上限重跑，但**停止原因（例如 30 % 強度條件）不會因截短而改變**；真正的加長版行為以 3.5 的樣本檢查為準。
* 未被截短的曲線沿用 shard 內原有 `pass_cells`（float64 座標算出），被截短的曲線用 shard 內 float32 點列重算；兩者可能在格點邊界上差一格，
  `pass_cells_recompute_*` 診斷量化這個差異。
* 分支長度的分位數為 1 m 解析度（向下取整）；剛好停在上限的分支長度＝ 20000 m（浮點誤差 < 1 m）。
* 只跑部分月份時，熱區是該批月份的熱區（不是全期），一致性檢查只驗證包含關係，結果僅供試跑。
* 「全部 TRAP」指合格（`is_qualified`）曲線；與 aggregate 的 `core_count_total` 同一母體。

## 6. 測試

```bash
cd /Users/mustlab/Workspace/TRAPs
.venv/bin/python -m pytest tests/test_curve_length_evidence.py -q
```

單元測試涵蓋弧長、截短內插、L ≥ 分支長度＝原樣、容差；整合測試用 A 區合成流場實際跑 `detect-shards`（上限 1500 m）→
`track-period` → `aggregate --shared-strong-core-s1-day` → 本腳本，以暴力迴圈獨立重算停止原因計數、分支長度與截短後的逐格出現次數，
並驗證基準與正式輸出逐格相同、竄改正式輸出時以狀態 2 結束且印出差異、加長檢查、拒絕非共用門檻的 aggregate。
