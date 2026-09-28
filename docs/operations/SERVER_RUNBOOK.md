# TRAPs SERVER 部署與執行手冊

本手冊說明如何在 SERVER（Ubuntu 20.04、56 核、125 GB RAM，帳號 `mustlab`，無
`sudo`）上部署並執行 TRAPs 分析管線：從 Mac 端把程式碼帶上 SERVER、準備 Python
環境、把 OCM 表層流場資料從 NAS 同步到本機 SSD、匯出 ocean mask、跑
detect/track/aggregate，以及把結果發布出去。

所有步驟都透過 `scripts/run_traps_server.sh <MODE>` 執行；本文件只描述操作流
程與預期結果，實際的檢出/追蹤演算法邏輯在 `python -m traps`（`src/traps/cli.py`）。

## 目的與範圍

- 目的：讓 TRAPs（四流場區域 × 五研究站的 TRAP 短期吸引潛勢分析）可以在 SERVER
  上批次跑完 2024-01 到 2025-12、四個 domain 的 detect → track → aggregate，並
  留下可追溯的 log 與 manifest。
- 範圍只涵蓋「部署與批次執行工具」（work package WP-C）；detect-shards /
  track-period 兩個 CLI 子指令的演算法邏輯由另一個工作項目負責，本手冊只描述
  怎麼呼叫它們。
- 本手冊模仿同一團隊下 LBT 專案在 SERVER 上已經驗證過的部署慣例（同一個
  `/home/mustlab/Workspace` 之下），TRAPs 沿用相同的目錄骨架與「本機優先、NAS
  只當來源」的原則。

## 為什麼要把資料複製到本機 SSD

NAS（`/data`）是 NFS 掛載，實測頻寬只有約 100 Mb/s，且內容可能在任何時候被清
理，不能當作長期或高吞吐量計算時的直接讀取來源。SERVER 本機 SSD（`/home`）有
約 1.1 TB 可用空間。因此所有大量、重複讀取的 OCM 表層流場資料，在跑
detect/track 之前都必須先用 `STAGE` 模式同步一份到本機（`traps-local-data`），
之後的計算全部只讀本機複本；NAS 只在 `STAGE`（讀）與最後把成果寫回
（`/data/TRAPs`，寫）時被觸碰。

## 目錄配置

TRAPs 在 SERVER 上的所有程式碼、輸入、輸出都放在
`/home/mustlab/Workspace`（下稱 `$TRAPS_WORKSPACE`）之下，**不要**直接放在
`$HOME`。這個規劃完全比照同一個 Workspace 下既有的 LBT 專案（`lbt-repo` /
`lbt-inputs` / `lbt-local-data` / `lbt-runs`），只是把 `lbt-` 前綴換成
`traps-`；兩者互不干涉，本手冊所有指令都絕對不會讀寫 `lbt-*` 目錄。

| 目錄（預設路徑，可用同名環境變數覆寫） | 用途 |
| --- | --- |
| `$TRAPS_WORKSPACE`（`/home/mustlab/Workspace`） | 共用根目錄，與 LBT 專案共用 |
| `$TRAPS_REPO`（`.../traps-repo`） | git clone（`main` 分支）+ `.venv` |
| `$TRAPS_INPUTS`（`.../traps-inputs`） | `study_design` 快照、海岸線 GeoJSON、ocean mask sidecar（`ocean_masks/<domain>/`） |
| `$TRAPS_LOCAL_DATA`（`.../traps-local-data`） | 本機端 OCM 表層流場複本，`ocm_surface/<domain>/{grid, months/<YYYYMM>}`，加上 `.staged/<domain>.done` 標記檔 |
| `$TRAPS_RUNS`（`.../traps-runs`） | 所有跑批結果：`runs/<param_id>/{detect,track,aggregate,logs}`、共用快取（`cache/{uv,uv-python,mpl,xdg}`）、`bin/uv`、`.lock` |
| `$TRAPS_WORKSPACE/deploy-bundles` | 從 Mac 端帶上來的 git bundle 檔案 |

`SETUP` 模式會建立以上 `traps-*` 目錄並設成 `chmod 700`；`deploy-bundles` 與
`$TRAPS_WORKSPACE` 本身則沿用既有慣例，不由本腳本管理權限。

## 首次部署（把程式碼帶上 SERVER）

SERVER 不能直接連外部 git remote，因此用 git bundle 從 Mac 端帶上去：

```bash
# 在 Mac 端 repo 內
git bundle create traps-$(date +%Y%m%d).bundle main

# 傳到 SERVER（例如透過 scp）
scp traps-$(date +%Y%m%d).bundle mustlab@140.117.88.203:/home/mustlab/Workspace/deploy-bundles/
```

```bash
# 在 SERVER 上，第一次建立 traps-repo
mkdir -p /home/mustlab/Workspace/deploy-bundles
cd /home/mustlab/Workspace
git clone /home/mustlab/Workspace/deploy-bundles/traps-YYYYMMDD.bundle traps-repo
cd traps-repo
git checkout main
```

之後每次更新程式碼，一樣是「產生新 bundle → scp 上來 → 在 `traps-repo` 內
`git fetch` 該 bundle 再 `git merge`/`git reset --hard` 到目標 commit」，例如：

```bash
cd /home/mustlab/Workspace/traps-repo
git fetch /home/mustlab/Workspace/deploy-bundles/traps-YYYYMMDD-2.bundle main
git checkout <目標 commit 或 main>
```

**絕對不要**用 `rsync --delete` 或任何會清空目的地的方式同步 `traps-repo`、
`traps-inputs`、`traps-local-data`、`traps-runs`，也絕對不要用任何操作觸碰
`lbt-*` 目錄——那是另一個專案在用，兩者共用同一台機器與同一個 Workspace 根目
錄，但完全獨立。

## SETUP：準備 Python 環境

```bash
scripts/run_traps_server.sh SETUP
```

依序執行：

1. 建立 `traps-repo` / `traps-inputs` / `traps-local-data` / `traps-runs` 及其
   子目錄，並設 `chmod 700`。
2. 複製 `uv` 執行檔到 `$TRAPS_RUNS/bin/uv`（來源預設是 NAS 上的一份
   `uv-bin/uv`；NAS 路徑本身不穩定，只在本機沒有 `uv` 時複製一次，之後都用本機
   這一份，不再依賴 NAS）。
3. 用 `uv python install 3.12`（快取在 `$TRAPS_RUNS/cache/uv-python`）安裝一份
   SERVER 系統 Python（3.8）之外、滿足 `pyproject.toml` 所要求
   `requires-python = ">=3.11"` 的 Python。
4. 在 `traps-repo` 內 `uv sync --frozen --extra test` 建立 `.venv`（PyPI 在
   SERVER 上是通的，`--frozen` 確保用 `uv.lock` 鎖定的版本，不會意外升級）。
5. 執行 `pytest -q`，確認環境本身沒有問題。

前提：`traps-repo` 必須已經是一個 git repo（見上一節「首次部署」）。SETUP 不會
自己去 clone。

## STAGE：把 OCM 資料同步到本機

```bash
scripts/run_traps_server.sh STAGE
```

實際呼叫 `scripts/stage_ocm_surface_local.py`（SERVER 系統內建的 `python3`
即可執行，不需要等 `.venv` 建好），把 `$TRAPS_NAS_SURFACE_ROOT` 下四個 domain、
`$TRAPS_MONTHS` 指定月份範圍的 `grid/` 與逐月檔案（`metadata.json`、
`time_utc_ns.npy`、`u_surface_mps.npy`、`v_surface_mps.npy`、
`valid_mask_surface.npy`、`qc_flags.npy`，`quality_report.json` 存在才複製；
`eta_m.npy` 預設不複製）複製到 `$TRAPS_LOCAL_DATA/ocm_surface/<domain>/...`。

- 每個檔案複製後都會用 sha256 + size 驗證，任何不符都會讓該次執行以非零狀態碼
  結束（log 見 `$TRAPS_RUNS/logs/stage-<時間戳>.log`）。
- 已存在且 hash 相符的檔案會被跳過，所以中斷後重跑是安全的、也不會重覆下載已
  經確認過的內容。
- 每個 domain 全部複製完成後，會寫入
  `.staged/<domain>.manifest.json`（逐檔案 size/sha256/來源路徑）與
  `.staged/<domain>.done` 標記檔；`PREPARE` 會檢查這個標記檔存不存在。
- 預估大小：單一 domain-month 的 u/v/valid/qc/time 五個必要陣列合計約
  103 MB；`TRAPS_MONTHS=202401-202512`、四個 domain 共 96 個 domain-month，總
  計約 10 GB（不含 `eta_m.npy`；加上 `--include-eta` 或
  `TRAPS_DETECT_ARGS`/自行呼叫 stage 腳本加 `--include-eta` 會再增加）。

想只驗證本機既有檔案是否完整、不複製任何新內容，可以直接呼叫底層腳本：

```bash
python3 scripts/stage_ocm_surface_local.py \
  --dest-root /home/mustlab/Workspace/traps-local-data \
  --verify-only
```

## MASKS：匯出 ocean mask sidecar

```bash
scripts/run_traps_server.sh MASKS
```

用 `.venv` 的 Python 執行既有的 `scripts/export_ocean_masks.py`（讀 NAS 上的
`ocm_surface` 與 `ocm_surface_coastline_corrected_v2` 兩種 grid layout），輸出
到 `$TRAPS_INPUTS/ocean_masks/<domain>/`。如果四個 domain 的 sidecar
（`analysis_ocean_mask_coastline_corrected_v2.npy`）都已經存在，這個模式會直接
略過並印出訊息，不會覆寫——如果需要重新匯出，要先手動移除舊檔案。

## PREPARE：跑批次前的唯讀健檢

```bash
scripts/run_traps_server.sh PREPARE
```

逐項印出 `PASS: ...` 或 `FAIL: ... -- 原因`，任何一項 FAIL 都會讓整體以非零狀
態碼結束（不會做任何寫入動作）。檢查項目：

- `traps-repo` 是 git 頂層目錄，且 `git status --porcelain --untracked-files=all`
  完全乾淨（沒有未提交也沒有未追蹤的檔案）。
- 若有設定 `TRAPS_EXPECTED_GIT_COMMIT`，HEAD 必須等於它（用來確保跑批的是預期
  的那個 commit）。
- `.venv` 的 python 版本 ≥ 3.11，且能 `import traps`。
- `/home` 可用空間 ≥ `TRAPS_MIN_FREE_GB`（預設 50 GB）。
- 每個 domain 都有 `traps-local-data/.staged/<domain>.done`。
- 每個 domain 都有 ocean mask sidecar（`.npy` + `.json`）。
- 沒有另一個 `run_traps_server.sh` 執行程序持有 `$TRAPS_RUNS/.lock`（用
  `flock` 實作；`DETECT`/`TRACK`/`AGGREGATE` 開始前也會做同樣的鎖檢查）。

`DETECT` 在開始平行跑工作前，會自動先做一次同樣的健檢，任何一項沒過就不會啟
動任何 detect 工作。

## 執行 DETECT / TRACK / AGGREGATE

三個模式都建議在 `tmux` 裡跑，避免 SSH 斷線中斷長時間的批次工作：

```bash
tmux new -s traps
scripts/run_traps_server.sh DETECT
# Ctrl-b d 離開 tmux，之後用 tmux attach -t traps 接回去看進度
```

### DETECT

對 `$TRAPS_DOMAINS`（四個）× `$TRAPS_MONTHS` 展開的每個 domain-month，各呼叫
一次：

```
$VENV_PY -m traps detect-shards \
  --surface-root $TRAPS_LOCAL_DATA/ocm_surface \
  --domain-id <domain> --month <YYYYMM> \
  --ocean-mask $TRAPS_INPUTS/ocean_masks/<domain>/analysis_ocean_mask_coastline_corrected_v2.npy \
  --output-dir $TRAPS_RUNS/runs/$TRAPS_PARAM_ID/detect/<domain>/<YYYYMM> \
  --param-id $TRAPS_PARAM_ID --resume $TRAPS_DETECT_ARGS
```

用 `xargs -P $TRAPS_JOBS`（若機器上裝了 GNU `parallel` 則優先用它）平行跑，每
個 job 的 stdout/stderr 各自落一份 log 在
`$TRAPS_RUNS/runs/$TRAPS_PARAM_ID/logs/detect/<domain>-<month>.log`；
`MPLCONFIGDIR`/`XDG_CACHE_HOME` 指到 `$TRAPS_RUNS/cache`，並強制
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`，避免每個平行 job
自己又搶開多執行緒 BLAS、互相拖垮。結束後寫出摘要 TSV
（`$TRAPS_RUNS/runs/$TRAPS_PARAM_ID/detect_summary.tsv`：domain、month、
status、秒數），任何一個 job 失敗都讓整體以非零狀態碼結束。

預估耗時：單一 snapshot 檢出約 1.1 秒/核；單一 domain-month 單核大約 12 分
鐘。96 個 domain-month 工作、`TRAPS_JOBS=24` 時，粗估整體約 1 小時（實際隨
snapshot 數量、机器負載變動，僅供排程參考）。

### TRACK

DETECT 全部成功後才跑：

```bash
scripts/run_traps_server.sh TRACK
```

依 domain 平行跑（最多同時 4 個），每個呼叫：

```
$VENV_PY -m traps track-period \
  --shards-root $TRAPS_RUNS/runs/$TRAPS_PARAM_ID/detect/<domain> \
  --domain-id <domain> --months $TRAPS_MONTHS \
  --output-dir $TRAPS_RUNS/runs/$TRAPS_PARAM_ID/track/<domain> \
  --param-id $TRAPS_PARAM_ID --resume
```

log 在 `$TRAPS_RUNS/runs/$TRAPS_PARAM_ID/logs/track/<domain>.log`。預估耗時：
每個 domain 數十分鐘量級，實際隨資料量變動。

### AGGREGATE

```bash
scripts/run_traps_server.sh AGGREGATE
```

呼叫 `$VENV_PY -m traps aggregate --track-root ... --detect-root ... --output-dir ... --param-id ... --resume`。
**這個子指令目前尚未實作**（由另一個工作項目負責）；在那之前，`AGGREGATE`
模式偵測到 `python -m traps aggregate --help` 失敗時，會印出清楚訊息並以
exit code 2 結束，不會假裝成功。

## 如何在失敗後續跑（resume）

三個模式（`DETECT`/`TRACK`/`STAGE`）都用同一個原則：**重跑同一個 MODE**，已經
成功且驗證過的輸出會被自動跳過（`--resume` 給 `detect-shards`/`track-period`；
`STAGE` 靠 size+sha256 比對）。不需要手動清除任何輸出目錄，也不需要自己挑出哪
些 domain-month 失敗——腳本會自己重跑沒有成功標記的部分。

如果懷疑某個輸出已經損毀（而不是單純沒跑完），才需要手動刪除對應的
`$TRAPS_RUNS/runs/<param_id>/...` 子目錄後再重跑該模式。

## STATUS：檢查目前進度

```bash
scripts/run_traps_server.sh STATUS
```

輸出：

- `DETECT`：每個 domain 目前有幾個 `_SUCCESS` shard，對照預期的
  domain-month 數（依 `TRAPS_MONTHS` 展開）。
- `TRACK`：每個 domain 是 `done`（有 `_SUCCESS`）、`in-progress-or-failed`
  （目錄存在但沒有 `_SUCCESS`），還是 `not-started`。
- 任何被記成 `failed` 的 detect job，印出對應 log 檔的最後幾行；track log 內
  若出現 `error`/`traceback` 字樣，也一併印出最後幾行，方便快速定位。

log 檔全部在 `$TRAPS_RUNS/runs/<param_id>/logs/{detect,track}/`；
`$TRAPS_RUNS/runs/<param_id>/detect_summary.tsv` 是 DETECT 的逐工作摘要。

「成功」的定義：`DETECT` 每個 domain-month 的輸出目錄下有 `_SUCCESS` 標記；
`TRACK` 每個 domain 的輸出目錄下有 `_SUCCESS` 標記；`AGGREGATE`（實作完成後）
比照辦理。`--resume` 就是靠這些標記判斷哪些部分已經完成、不需要重跑。

## 發布結果

批次全部成功（`STATUS` 顯示 DETECT/TRACK 都完成）後，把
`$TRAPS_RUNS/runs/<param_id>/{detect,track,aggregate}` 需要保留的部分發布出
去，依需求擇一：

- **發回 NAS**（供團隊其他人或長期保存）：複製到 `/data/TRAPs/<param_id>/...`
  （目錄若不存在需自行建立）；同樣不要用 `rsync --delete` 之類會清空目的地的
  作法，一律用「只加不刪」的方式同步。
- **拉回 Mac**：在 Mac 端用 `rsync -av`（**不要加 `--delete`**）從 SERVER pull
  回本機，例如：

  ```bash
  rsync -av mustlab@<SERVER 主機>:/home/mustlab/Workspace/traps-runs/runs/<param_id>/ \
    ./local-copy/<param_id>/
  ```

  （主機位址不寫死在本文件中，依實際連線資訊帶入。）

## 清理政策

- `traps-local-data`（本機 OCM 複本）：這是可以隨時從 NAS 重新 `STAGE` 出來的
  快取，不是唯一副本；空間吃緊時可以刪除整個 `ocm_surface/` 重新來過，但不要
  用 `rm -rf` 誤刪到 `.staged/` 之外還在用的目錄，也絕對不要動到 `lbt-local-data`。
- `traps-runs/runs/<param_id>/`：跑批的中繼與最終輸出。已經發布（NAS 或 Mac）
  且確認完整的 run，可以視空間需求清掉本機這份；清掉前務必先確認 `STATUS`
  顯示全部完成、且已經成功複製出去。
- `traps-runs/cache/`（`uv`、`mpl`、`xdg`）：純快取，可隨時清空，下次執行會自
  動重建，只是會變慢一點。
- 不要清理 `traps-inputs`（`study_design` 快照、ocean mask、海岸線資料）——這
  些是小檔案，且是後續所有 domain 的必要輸入。

## 已知限制 / 待補事項

- `AGGREGATE` 的底層 `python -m traps aggregate` 子指令尚未實作；本手冊描述的
  是介面，實際可用時間依該工作項目進度而定。
- `TRAPS_UV_SOURCE`（NAS 上的 `uv` 執行檔）本身放在 NAS，理論上也可能被清
  理；`SETUP` 只在本機 `$TRAPS_RUNS/bin/uv` 不存在時才去讀它，一旦複製成功，
  之後的 `SETUP` 都不再依賴 NAS 上那份。如果兩邊都不見了，需要手動找一份新的
  `uv` 靜態執行檔放到 `$TRAPS_RUNS/bin/uv`（或改 `TRAPS_UV_SOURCE` 指到新位
  置）。
