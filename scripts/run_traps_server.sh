#!/usr/bin/env bash
# run_traps_server.sh — TRAPs SERVER 端部署與批次執行入口。
#
# 用法：
#   scripts/run_traps_server.sh <MODE>
# MODE 為下列其中之一（大小寫需相符）：
#   SETUP | PREPARE | STAGE | MASKS | DETECT | TRACK | AGGREGATE | STATUS
#
# 完整說明見 docs/operations/SERVER_RUNBOOK.md。本檔案只做部署/執行編排，實際的
# TRAP 檢出與追蹤演算法在 `python -m traps` CLI（src/traps/cli.py）。
#
# 所有路徑與參數都可用環境變數覆寫，見下方「可覆寫設定」區塊。

set -euo pipefail

# ---------------------------------------------------------------------------
# 可覆寫設定（環境變數優先，未設定則使用預設值）
# ---------------------------------------------------------------------------

: "${TRAPS_WORKSPACE:=/home/mustlab/Workspace}"
: "${TRAPS_REPO:=${TRAPS_WORKSPACE}/traps-repo}"
: "${TRAPS_INPUTS:=${TRAPS_WORKSPACE}/traps-inputs}"
: "${TRAPS_LOCAL_DATA:=${TRAPS_WORKSPACE}/traps-local-data}"
: "${TRAPS_RUNS:=${TRAPS_WORKSPACE}/traps-runs}"
: "${TRAPS_NAS_SURFACE_ROOT:=/data/OCM-Preprocessed-Data/preprocessed/ocm_surface}"
: "${TRAPS_NAS_CORRECTED_ROOT:=/data/OCM-Preprocessed-Data/preprocessed/ocm_surface_coastline_corrected_v2}"
: "${TRAPS_MONTHS:=202401-202512}"
: "${TRAPS_DOMAINS:=northeast_taiwan_common_cache_v3 hsinchu_cache_v3 houwan_nmmba_cache_v3 lienchiang_common_cache_v3}"
: "${TRAPS_JOBS:=24}"
: "${TRAPS_PARAM_ID:=r0_default}"
: "${TRAPS_DETECT_ARGS:=}"
: "${TRAPS_EXPECTED_GIT_COMMIT:=}"
: "${TRAPS_MIN_FREE_GB:=50}"
: "${TRAPS_UV_SOURCE:=/data/LBT/cache/formal/ded8463/first-v2-h30-fixed-calendar-m10-435a57a/uv-bin/uv}"

TRAPS_UV_BIN="${TRAPS_RUNS}/bin/uv"
TRAPS_VENV_PY="${TRAPS_REPO}/.venv/bin/python"
TRAPS_LOCK_FILE="${TRAPS_RUNS}/.lock"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

# ---------------------------------------------------------------------------
# 共用小工具
# ---------------------------------------------------------------------------

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

die() {
  log "FATAL: $*"
  exit 1
}

usage() {
  cat >&2 <<'EOF'
用法: run_traps_server.sh <MODE>
MODE: SETUP | PREPARE | STAGE | MASKS | DETECT | TRACK | AGGREGATE | STATUS
詳見 docs/operations/SERVER_RUNBOOK.md
EOF
}

# 把 TRAPS_DOMAINS（空白分隔字串）轉成 bash array，供迴圈與傳給其他指令使用。
read -r -a DOMAIN_ARRAY <<<"${TRAPS_DOMAINS}"

# 需要 flock 時才使用鎖檔；只有 PREPARE / DETECT / TRACK / AGGREGATE 會檢查/持有鎖。
with_lock() {
  mkdir -p "${TRAPS_RUNS}"
  exec 9>"${TRAPS_LOCK_FILE}"
  if ! flock -n 9; then
    die "另一個 run_traps_server.sh 執行程序持有鎖（${TRAPS_LOCK_FILE}），請稍後再試"
  fi
}

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

mode_setup() {
  log "SETUP: 建立 workspace 目錄"
  local dir
  for dir in "${TRAPS_WORKSPACE}" "${TRAPS_REPO}" "${TRAPS_INPUTS}" "${TRAPS_LOCAL_DATA}" "${TRAPS_RUNS}"; do
    mkdir -p "${dir}"
  done
  # 只對 traps-* 目錄（不含共用的 TRAPS_WORKSPACE 本身）設 700，絕不碰 lbt-* 目錄。
  for dir in "${TRAPS_REPO}" "${TRAPS_INPUTS}" "${TRAPS_LOCAL_DATA}" "${TRAPS_RUNS}"; do
    chmod 700 "${dir}"
  done
  mkdir -p "${TRAPS_RUNS}/bin" "${TRAPS_RUNS}/logs" "${TRAPS_RUNS}/cache/uv" "${TRAPS_RUNS}/cache/uv-python" \
    "${TRAPS_RUNS}/cache/mpl" "${TRAPS_RUNS}/cache/xdg" "${TRAPS_RUNS}/runs"

  if [[ ! -x "${TRAPS_UV_BIN}" ]]; then
    log "SETUP: 複製 uv 執行檔到 ${TRAPS_UV_BIN}（來源：${TRAPS_UV_SOURCE}，NAS 上較不穩定，只複製一次）"
    [[ -f "${TRAPS_UV_SOURCE}" ]] || die "找不到來源 uv 執行檔：${TRAPS_UV_SOURCE}"
    cp -- "${TRAPS_UV_SOURCE}" "${TRAPS_UV_BIN}.tmp$$"
    chmod +x "${TRAPS_UV_BIN}.tmp$$"
    mv -- "${TRAPS_UV_BIN}.tmp$$" "${TRAPS_UV_BIN}"
  else
    log "SETUP: uv 執行檔已存在（${TRAPS_UV_BIN}），略過複製"
  fi

  [[ -d "${TRAPS_REPO}/.git" ]] || die "找不到 ${TRAPS_REPO}/.git；請先依 SERVER_RUNBOOK.md 完成首次部署（git bundle clone）"

  log "SETUP: 安裝 Python 3.12（UV_PYTHON_INSTALL_DIR=${TRAPS_RUNS}/cache/uv-python）"
  UV_PYTHON_INSTALL_DIR="${TRAPS_RUNS}/cache/uv-python" \
    UV_CACHE_DIR="${TRAPS_RUNS}/cache/uv" \
    "${TRAPS_UV_BIN}" python install 3.12

  log "SETUP: uv sync --frozen --extra test（於 ${TRAPS_REPO}）"
  (
    cd "${TRAPS_REPO}"
    UV_PYTHON_INSTALL_DIR="${TRAPS_RUNS}/cache/uv-python" \
      UV_CACHE_DIR="${TRAPS_RUNS}/cache/uv" \
      "${TRAPS_UV_BIN}" sync --frozen --extra test
  )

  log "SETUP: 執行 pytest -q（於 ${TRAPS_REPO}）"
  (
    cd "${TRAPS_REPO}"
    UV_CACHE_DIR="${TRAPS_RUNS}/cache/uv" "${TRAPS_UV_BIN}" run pytest -q
  )

  log "SETUP: 完成"
}

# ---------------------------------------------------------------------------
# PREPARE（唯讀健檢；每項印 PASS/FAIL，任何 FAIL 都讓整體以非零結束）
# ---------------------------------------------------------------------------

mode_prepare() {
  with_lock
  prepare_checks
}

# 實際健檢邏輯，抽出成獨立函式，讓 DETECT 可以在自己已持有鎖的情況下重用，不必
# 再呼叫一次 with_lock（flock 用同一個檔案描述子，不需要、也不應該重覆鎖定）。
prepare_checks() {
  local overall_status=0
  local check_name

  check_pass() {
    printf 'PASS: %s\n' "$1"
  }
  check_fail() {
    printf 'FAIL: %s -- %s\n' "$1" "$2"
    overall_status=1
  }

  check_name="repo 是 git worktree 頂層"
  if [[ -d "${TRAPS_REPO}/.git" ]] && git -C "${TRAPS_REPO}" rev-parse --show-toplevel >/dev/null 2>&1 \
    && [[ "$(git -C "${TRAPS_REPO}" rev-parse --show-toplevel)" == "$(cd "${TRAPS_REPO}" && pwd -P)" ]]; then
    check_pass "${check_name}"
  else
    check_fail "${check_name}" "${TRAPS_REPO} 不是 git repo 頂層"
  fi

  check_name="repo 工作目錄乾淨（含未追蹤檔案）"
  if [[ -d "${TRAPS_REPO}/.git" ]]; then
    if [[ -z "$(git -C "${TRAPS_REPO}" status --porcelain --untracked-files=all 2>/dev/null)" ]]; then
      check_pass "${check_name}"
    else
      check_fail "${check_name}" "有未提交或未追蹤的變更"
    fi
  else
    check_fail "${check_name}" "找不到 .git"
  fi

  if [[ -n "${TRAPS_EXPECTED_GIT_COMMIT}" ]]; then
    check_name="HEAD 符合 TRAPS_EXPECTED_GIT_COMMIT"
    if [[ -d "${TRAPS_REPO}/.git" ]]; then
      local head_commit
      head_commit="$(git -C "${TRAPS_REPO}" rev-parse HEAD 2>/dev/null || true)"
      if [[ "${head_commit}" == "${TRAPS_EXPECTED_GIT_COMMIT}" ]]; then
        check_pass "${check_name}"
      else
        check_fail "${check_name}" "HEAD=${head_commit:-<none>} 預期=${TRAPS_EXPECTED_GIT_COMMIT}"
      fi
    else
      check_fail "${check_name}" "找不到 .git"
    fi
  fi

  check_name=".venv python >= 3.11 且可 import traps"
  if [[ -x "${TRAPS_VENV_PY}" ]]; then
    if "${TRAPS_VENV_PY}" - <<'PYEOF' >/dev/null 2>&1
import sys
assert sys.version_info >= (3, 11)
import traps  # noqa: F401
PYEOF
    then
      check_pass "${check_name}"
    else
      check_fail "${check_name}" "${TRAPS_VENV_PY} 版本不符或無法 import traps"
    fi
  else
    check_fail "${check_name}" "找不到 ${TRAPS_VENV_PY}（請先執行 SETUP）"
  fi

  check_name="/home 可用空間 >= ${TRAPS_MIN_FREE_GB} GB"
  local free_kb free_gb
  free_kb="$(df -Pk /home 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ -n "${free_kb}" ]]; then
    free_gb=$((free_kb / 1024 / 1024))
    if (( free_gb >= TRAPS_MIN_FREE_GB )); then
      check_pass "${check_name}（目前約 ${free_gb} GB）"
    else
      check_fail "${check_name}" "目前約 ${free_gb} GB"
    fi
  else
    check_fail "${check_name}" "無法讀取 /home 可用空間（df 失敗）"
  fi

  local domain
  for domain in "${DOMAIN_ARRAY[@]}"; do
    check_name="domain ${domain} 已完成本機 staging（.staged/${domain}.done）"
    if [[ -f "${TRAPS_LOCAL_DATA}/.staged/${domain}.done" ]]; then
      check_pass "${check_name}"
    else
      check_fail "${check_name}" "找不到標記檔"
    fi

    check_name="domain ${domain} 有 ocean mask sidecar"
    local mask_npy="${TRAPS_INPUTS}/ocean_masks/${domain}/analysis_ocean_mask_coastline_corrected_v2.npy"
    local mask_json="${TRAPS_INPUTS}/ocean_masks/${domain}/analysis_ocean_mask_coastline_corrected_v2.json"
    if [[ -f "${mask_npy}" && -f "${mask_json}" ]]; then
      check_pass "${check_name}"
    else
      check_fail "${check_name}" "缺少 ${mask_npy} 或 ${mask_json}"
    fi
  done

  check_name="沒有其他 run_traps_server.sh 執行程序持有鎖"
  # 能執行到這裡代表 with_lock 已成功取得鎖（否則已 die），視為通過。
  check_pass "${check_name}"

  if [[ "${overall_status}" -ne 0 ]]; then
    log "PREPARE: 有檢查未通過"
  else
    log "PREPARE: 全部通過"
  fi
  return "${overall_status}"
}

# ---------------------------------------------------------------------------
# STAGE
# ---------------------------------------------------------------------------

mode_stage() {
  mkdir -p "${TRAPS_RUNS}/logs" "${TRAPS_LOCAL_DATA}"
  local ts log_file
  ts="$(date -u '+%Y%m%dT%H%M%SZ')"
  log_file="${TRAPS_RUNS}/logs/stage-${ts}.log"

  local domain_args=()
  local domain
  for domain in "${DOMAIN_ARRAY[@]}"; do
    domain_args+=(--domain-id "${domain}")
  done

  log "STAGE: 開始同步 NAS -> ${TRAPS_LOCAL_DATA}（log: ${log_file}）"
  python3 "${SCRIPT_DIR}/stage_ocm_surface_local.py" \
    --source-root "${TRAPS_NAS_SURFACE_ROOT}" \
    --dest-root "${TRAPS_LOCAL_DATA}" \
    "${domain_args[@]}" \
    --months "${TRAPS_MONTHS}" \
    --jobs "${TRAPS_JOBS}" \
    2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"
  if [[ "${status}" -ne 0 ]]; then
    die "STAGE 失敗（詳見 ${log_file}）"
  fi
  log "STAGE: 完成"
}

# ---------------------------------------------------------------------------
# MASKS
# ---------------------------------------------------------------------------

mode_masks() {
  [[ -x "${TRAPS_VENV_PY}" ]] || die "找不到 ${TRAPS_VENV_PY}；請先執行 SETUP"

  local output_root="${TRAPS_INPUTS}/ocean_masks"
  local domain
  local any_missing=0
  for domain in "${DOMAIN_ARRAY[@]}"; do
    if [[ ! -f "${output_root}/${domain}/analysis_ocean_mask_coastline_corrected_v2.npy" ]]; then
      any_missing=1
    fi
  done
  if [[ "${any_missing}" -eq 0 ]]; then
    log "MASKS: 所有 domain 的 ocean mask sidecar 皆已存在，略過（如需重建請先手動移除 ${output_root} 下對應檔案）"
    return 0
  fi

  local domain_args=()
  for domain in "${DOMAIN_ARRAY[@]}"; do
    domain_args+=(--domain-id "${domain}")
  done

  log "MASKS: 執行 export_ocean_masks.py -> ${output_root}"
  "${TRAPS_VENV_PY}" "${TRAPS_REPO}/scripts/export_ocean_masks.py" \
    --surface-root "${TRAPS_NAS_SURFACE_ROOT}" \
    --corrected-root "${TRAPS_NAS_CORRECTED_ROOT}" \
    --output-root "${output_root}" \
    "${domain_args[@]}"
  log "MASKS: 完成"
}

# ---------------------------------------------------------------------------
# DETECT
# ---------------------------------------------------------------------------

run_env_prefix() {
  # 供子行程使用的共用環境變數（cache 目錄、單執行緒 BLAS，避免和平行 job 互搶）。
  printf 'MPLCONFIGDIR=%q XDG_CACHE_HOME=%q OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1' \
    "${TRAPS_RUNS}/cache/mpl" "${TRAPS_RUNS}/cache/xdg"
}

mode_detect() {
  with_lock
  if ! prepare_checks; then
    die "DETECT 前置 PREPARE 檢查未通過，中止"
  fi

  [[ -x "${TRAPS_VENV_PY}" ]] || die "找不到 ${TRAPS_VENV_PY}"

  local run_dir="${TRAPS_RUNS}/runs/${TRAPS_PARAM_ID}"
  local detect_root="${run_dir}/detect"
  local log_dir="${run_dir}/logs/detect"
  mkdir -p "${detect_root}" "${log_dir}" "${TRAPS_RUNS}/cache/mpl" "${TRAPS_RUNS}/cache/xdg"

  local months
  read -r -a months <<<"$(python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from stage_ocm_surface_local import _parse_months
print(' '.join(_parse_months('${TRAPS_MONTHS}')))
")"

  local job_list_file="${run_dir}/detect_jobs.txt"
  : >"${job_list_file}"
  local domain month
  for domain in "${DOMAIN_ARRAY[@]}"; do
    for month in "${months[@]}"; do
      printf '%s\t%s\n' "${domain}" "${month}" >>"${job_list_file}"
    done
  done

  local summary_file="${run_dir}/detect_summary.tsv"
  printf 'domain\tmonth\tstatus\tseconds\n' >"${summary_file}"

  detect_one_job() {
    local domain="$1" month="$2"
    local job_output_dir="${detect_root}/${domain}/${month}"
    local job_log="${log_dir}/${domain}-${month}.log"
    mkdir -p "${job_output_dir}"
    local start_ts end_ts
    start_ts="$(date +%s)"
    if MPLCONFIGDIR="${TRAPS_RUNS}/cache/mpl" XDG_CACHE_HOME="${TRAPS_RUNS}/cache/xdg" \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "${TRAPS_VENV_PY}" -m traps detect-shards \
      --surface-root "${TRAPS_LOCAL_DATA}/ocm_surface" \
      --domain-id "${domain}" \
      --month "${month}" \
      --ocean-mask "${TRAPS_INPUTS}/ocean_masks/${domain}/analysis_ocean_mask_coastline_corrected_v2.npy" \
      --output-dir "${job_output_dir}" \
      --param-id "${TRAPS_PARAM_ID}" \
      --resume \
      ${TRAPS_DETECT_ARGS} \
      >"${job_log}" 2>&1
    then
      end_ts="$(date +%s)"
      printf '%s\t%s\t%s\t%s\n' "${domain}" "${month}" "ok" "$((end_ts - start_ts))" >>"${summary_file}"
    else
      end_ts="$(date +%s)"
      printf '%s\t%s\t%s\t%s\n' "${domain}" "${month}" "failed" "$((end_ts - start_ts))" >>"${summary_file}"
      return 1
    fi
  }
  export -f detect_one_job
  export TRAPS_VENV_PY TRAPS_LOCAL_DATA TRAPS_INPUTS TRAPS_PARAM_ID TRAPS_DETECT_ARGS TRAPS_RUNS
  export detect_root log_dir summary_file

  log "DETECT: ${#DOMAIN_ARRAY[@]} domain x ${#months[@]} 月份，平行度 ${TRAPS_JOBS}"
  local xargs_status=0
  if command -v parallel >/dev/null 2>&1; then
    # shellcheck disable=SC1083  # {1} {2} 是 GNU parallel 的欄位佔位符，不是 shell 展開。
    parallel --colsep '\t' -j "${TRAPS_JOBS}" detect_one_job {1} {2} <"${job_list_file}" || xargs_status=1
  else
    xargs -a "${job_list_file}" -P "${TRAPS_JOBS}" -I{} bash -c '
      line="{}"
      domain="${line%%$'"'"'\t'"'"'*}"
      month="${line##*$'"'"'\t'"'"'}"
      detect_one_job "${domain}" "${month}"
    ' || xargs_status=1
  fi

  log "DETECT: 摘要寫入 ${summary_file}"
  if [[ "${xargs_status}" -ne 0 ]] || grep -q "$(printf '\tfailed\t')" "${summary_file}"; then
    die "DETECT 有工作失敗，詳見 ${summary_file} 與 ${log_dir}/"
  fi
  log "DETECT: 完成"
}

# ---------------------------------------------------------------------------
# TRACK
# ---------------------------------------------------------------------------

mode_track() {
  with_lock
  [[ -x "${TRAPS_VENV_PY}" ]] || die "找不到 ${TRAPS_VENV_PY}"

  local run_dir="${TRAPS_RUNS}/runs/${TRAPS_PARAM_ID}"
  local detect_root="${run_dir}/detect"
  local track_root="${run_dir}/track"
  local log_dir="${run_dir}/logs/track"
  mkdir -p "${track_root}" "${log_dir}"

  track_one_domain() {
    local domain="$1"
    local job_log="${log_dir}/${domain}.log"
    "${TRAPS_VENV_PY}" -m traps track-period \
      --shards-root "${detect_root}/${domain}" \
      --domain-id "${domain}" \
      --months "${TRAPS_MONTHS}" \
      --output-dir "${track_root}/${domain}" \
      --param-id "${TRAPS_PARAM_ID}" \
      --resume \
      >"${job_log}" 2>&1
  }
  export -f track_one_domain
  export TRAPS_VENV_PY TRAPS_MONTHS TRAPS_PARAM_ID detect_root track_root log_dir

  log "TRACK: ${#DOMAIN_ARRAY[@]} domain，平行度 4"
  local pids=()
  local domain
  local failed=0
  local running=0
  for domain in "${DOMAIN_ARRAY[@]}"; do
    track_one_domain "${domain}" &
    pids+=("$!")
    running=$((running + 1))
    if (( running >= 4 )); then
      wait -n || failed=1
      running=$((running - 1))
    fi
  done
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done

  if [[ "${failed}" -ne 0 ]]; then
    die "TRACK 有 domain 失敗，詳見 ${log_dir}/"
  fi
  log "TRACK: 完成"
}

# ---------------------------------------------------------------------------
# AGGREGATE
# ---------------------------------------------------------------------------

mode_aggregate() {
  with_lock
  [[ -x "${TRAPS_VENV_PY}" ]] || die "找不到 ${TRAPS_VENV_PY}"

  local run_dir="${TRAPS_RUNS}/runs/${TRAPS_PARAM_ID}"
  local aggregate_dir="${run_dir}/aggregate"
  mkdir -p "${aggregate_dir}"

  if ! "${TRAPS_VENV_PY}" -m traps aggregate --help >/dev/null 2>&1; then
    log "AGGREGATE: 'python -m traps aggregate' 尚未實作，此模式目前是佔位符（placeholder）"
    exit 2
  fi

  "${TRAPS_VENV_PY}" -m traps aggregate \
    --track-root "${run_dir}/track" \
    --detect-root "${run_dir}/detect" \
    --output-dir "${aggregate_dir}" \
    --param-id "${TRAPS_PARAM_ID}" \
    --resume
  log "AGGREGATE: 完成"
}

# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------

mode_status() {
  local run_dir="${TRAPS_RUNS}/runs/${TRAPS_PARAM_ID}"
  local detect_root="${run_dir}/detect"
  local track_root="${run_dir}/track"

  echo "== TRAPs STATUS (param_id=${TRAPS_PARAM_ID}) =="

  local months
  read -r -a months <<<"$(python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from stage_ocm_surface_local import _parse_months
print(' '.join(_parse_months('${TRAPS_MONTHS}')))
" 2>/dev/null || true)"
  local expected_per_domain="${#months[@]}"

  echo "-- DETECT --"
  local domain
  for domain in "${DOMAIN_ARRAY[@]}"; do
    local success_count=0
    if [[ -d "${detect_root}/${domain}" ]]; then
      success_count="$(find "${detect_root}/${domain}" -name '_SUCCESS' 2>/dev/null | wc -l | tr -d ' ')"
    fi
    printf '%-40s %s/%s _SUCCESS shards\n' "${domain}" "${success_count}" "${expected_per_domain}"
  done

  echo "-- TRACK --"
  for domain in "${DOMAIN_ARRAY[@]}"; do
    if [[ -f "${track_root}/${domain}/_SUCCESS" ]]; then
      printf '%-40s done\n' "${domain}"
    elif [[ -d "${track_root}/${domain}" ]]; then
      printf '%-40s in-progress-or-failed\n' "${domain}"
    else
      printf '%-40s not-started\n' "${domain}"
    fi
  done

  echo "-- LAST LINES OF FAILED LOGS --"
  local log_dir="${run_dir}/logs"
  local summary_file="${run_dir}/detect_summary.tsv"
  if [[ -f "${summary_file}" ]]; then
    local domain_col month_col status_col rest
    while IFS=$'\t' read -r domain_col month_col status_col rest; do
      if [[ "${status_col}" == "failed" ]]; then
        local failed_log="${log_dir}/detect/${domain_col}-${month_col}.log"
        echo "### ${failed_log}"
        tail -n 5 "${failed_log}" 2>/dev/null || echo "(log 不存在)"
      fi
    done <"${summary_file}"
  fi
  if [[ -d "${log_dir}/track" ]]; then
    local track_log
    for track_log in "${log_dir}/track"/*.log; do
      [[ -e "${track_log}" ]] || continue
      if grep -qi 'traceback\|error' "${track_log}" 2>/dev/null; then
        echo "### ${track_log}"
        tail -n 5 "${track_log}"
      fi
    done
  fi
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

main() {
  local mode="${1:-}"
  case "${mode}" in
    SETUP) mode_setup ;;
    PREPARE) mode_prepare ;;
    STAGE) mode_stage ;;
    MASKS) mode_masks ;;
    DETECT) mode_detect ;;
    TRACK) mode_track ;;
    AGGREGATE) mode_aggregate ;;
    STATUS) mode_status ;;
    ""|-h|--help) usage; exit 2 ;;
    *) usage; die "未知 MODE: ${mode}" ;;
  esac
}

main "$@"
