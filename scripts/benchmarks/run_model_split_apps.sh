#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
APPS_DIR="$ROOT_DIR/ara/apps"
HW_DIR="$ROOT_DIR/ara/hardware"
TMP_DIR="$ROOT_DIR/tmp"
CONDA_SH="/data2/zzx/data/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="bitfly"

MODE="all"
BUILD_JOBS=16
PARALLEL=6
BATCH_SIZE=6
RUN_VERILATE=1
REBUILD_APPS=1
TRACE=0
HEARTBEAT_SEC=60
POLL_SEC=5
LOG_ROOT=""
MODELS_CSV=""
PRECS_CSV=""
IMPLS_CSV=""
APPS_CSV=""
EXTRA_MAKE_ARGS=""
APP_ENV_DEFINES="-DBMPMM_LOWP_DEFAULT_MODE=BMPMM_LOWP_EXEC_FAST -DRVV_BINARY_VECTOR_DEFAULT_MODE=RVV_BINARY_VECTOR_EXEC_FAST -DRVV_INT2_VECTOR_DEFAULT_MODE=RVV_INT2_VECTOR_EXEC_FAST -DRVV_INT4_VECTOR_DEFAULT_MODE=RVV_INT4_VECTOR_EXEC_FAST"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --mode <all|build|run>       Default: all
  --build-jobs <N>             make -j for app/hardware build, default: 16
  --parallel <N>               concurrent simv jobs, default: 6
  --batch-size <N>             apps per batch, default: 6
  --models <csv>               default: gemma3_270m,smollm2_360m,qwen25_05b,replit_code_v1_3b,tinyllama_11b,opt_13b,qwen25_15b,stablelm2_16b,smollm2_17b,gemma2_2b
  --precisions <csv>           default: binary,INT2,INT4
  --impls <csv>                default: bmpmm,rvv
  --apps <csv>                 explicit app list, overrides model/precision/impl filters
  --log-root <dir>             default: tmp/model_app_runs/<timestamp>
  --no-verilate                skip 'make -C ara/hardware verilate'
  --no-rebuild-apps            skip rebuilding app binaries
  --trace                      pass trace=1 to hardware make
  --heartbeat-sec <N>          runner heartbeat interval, default: 60
  --poll-sec <N>               runner poll interval, default: 5
  --extra-make-args <string>   extra args appended to hardware make
  --app-env-defines <string>   app build ENV_DEFINES, default enables fast estimate for bmpmm/rvv
  -h, --help                   show help

Examples:
  $(basename "$0") --mode all --parallel 6 --batch-size 6 --build-jobs 16
  $(basename "$0") --mode run --apps bmpmm_INT2_gemma3_270m,rvv_INT2_gemma3_270m
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --models) MODELS_CSV="$2"; shift 2 ;;
    --precisions) PRECS_CSV="$2"; shift 2 ;;
    --impls) IMPLS_CSV="$2"; shift 2 ;;
    --apps) APPS_CSV="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --no-verilate) RUN_VERILATE=0; shift ;;
    --no-rebuild-apps) REBUILD_APPS=0; shift ;;
    --trace) TRACE=1; shift ;;
    --heartbeat-sec) HEARTBEAT_SEC="$2"; shift 2 ;;
    --poll-sec) POLL_SEC="$2"; shift 2 ;;
    --extra-make-args) EXTRA_MAKE_ARGS="$2"; shift 2 ;;
    --app-env-defines) APP_ENV_DEFINES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$MODE" != "all" && "$MODE" != "build" && "$MODE" != "run" ]]; then
  echo "Invalid --mode: $MODE" >&2
  exit 1
fi

mkdir -p "$TMP_DIR"
if [[ -z "$LOG_ROOT" ]]; then
  LOG_ROOT="$TMP_DIR/model_app_runs/$(date +%Y%m%d_%H%M%S)_60apps"
fi
mkdir -p "$LOG_ROOT"
RUNNER_LOG="$LOG_ROOT/runner.log"
SUMMARY_CSV="$LOG_ROOT/summary.csv"
APPS_TXT="$LOG_ROOT/apps.txt"
MAIN_BASHPID="${BASHPID:-$$}"
IN_BATCH=0
CURRENT_BATCH_NAME=""
ACTIVE_PIDS=()

cleanup_active_batch() {
  local self_pid="${BASHPID:-$$}"
  local pid

  if [[ "$self_pid" != "$MAIN_BASHPID" ]]; then
    return 0
  fi
  if [[ "$IN_BATCH" -ne 1 || ${#ACTIVE_PIDS[@]} -eq 0 ]]; then
    return 0
  fi

  log "Cleaning up $CURRENT_BATCH_NAME with ${#ACTIVE_PIDS[@]} active app runners"
  for pid in "${ACTIVE_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}

trap cleanup_active_batch EXIT HUP INT TERM

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUNNER_LOG"
}

activate_env() {
  source "$CONDA_SH"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  conda activate "$CONDA_ENV_NAME"
  export VERILATOR_ROOT="${VERILATOR_ROOT:-$ROOT_DIR/ara/install/verilator/share/verilator}"
}

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  out_ref=()
  if [[ -n "$csv" ]]; then
    IFS=',' read -r -a out_ref <<< "$csv"
  fi
}

join_by() {
  local delimiter="$1"
  shift
  local first=1
  for item in "$@"; do
    if [[ $first -eq 1 ]]; then
      printf '%s' "$item"
      first=0
    else
      printf '%s%s' "$delimiter" "$item"
    fi
  done
}

list_target_apps() {
  local -a filtered models precs impls explicit

  if [[ -n "$APPS_CSV" ]]; then
    split_csv "$APPS_CSV" explicit
    printf '%s\n' "${explicit[@]}"
    return 0
  fi

  split_csv "$MODELS_CSV" models
  split_csv "$PRECS_CSV" precs
  split_csv "$IMPLS_CSV" impls

  if [[ ${#models[@]} -eq 0 ]]; then
    models=(gemma3_270m smollm2_360m qwen25_05b replit_code_v1_3b tinyllama_11b opt_13b qwen25_15b stablelm2_16b smollm2_17b gemma2_2b)
  fi
  if [[ ${#precs[@]} -eq 0 ]]; then
    precs=(binary INT2 INT4)
  fi
  if [[ ${#impls[@]} -eq 0 ]]; then
    impls=(bmpmm rvv)
  fi

  for impl in "${impls[@]}"; do
    for prec in "${precs[@]}"; do
      for model in "${models[@]}"; do
        filtered+=("${impl}_${prec}_${model}")
      done
    done
  done

  printf '%s\n' "${filtered[@]}" | awk 'NF' | while read -r app; do
    [[ -d "$APPS_DIR/$app" ]] && echo "$app"
  done
}

clean_app_artifacts() {
  local app="$1"
  find "$APPS_DIR/$app" -type f \( -name '*.o' -o -name '*.o.spike' \) -delete || true
  rm -f "$APPS_DIR/bin/$app" "$APPS_DIR/bin/$app.dump" "$APPS_DIR/bin/$app.spike" "$APPS_DIR/bin/$app.spike.dump"
}

build_apps() {
  local -a apps=("$@")
  (( ${#apps[@]} > 0 )) || return 0
  local -a bins=()
  for app in "${apps[@]}"; do
    clean_app_artifacts "$app"
    bins+=("bin/$app")
  done
  log "Building ${#apps[@]} apps with -j$BUILD_JOBS"
  log "App ENV_DEFINES: $APP_ENV_DEFINES"
  activate_env
  make -j"$BUILD_JOBS" -C "$APPS_DIR" -B ENV_DEFINES="$APP_ENV_DEFINES" "${bins[@]}"
}

build_verilator() {
  if [[ "$RUN_VERILATE" -eq 0 ]]; then
    log "Skipping verilate by request"
    return 0
  fi
  log "Building Verilator model with -j$BUILD_JOBS"
  activate_env
  if [[ "$TRACE" -eq 1 ]]; then
    make -j"$BUILD_JOBS" -C "$HW_DIR" verilate trace=1 $EXTRA_MAKE_ARGS
  else
    make -j"$BUILD_JOBS" -C "$HW_DIR" verilate $EXTRA_MAKE_ARGS
  fi
}

run_one_app() {
  local app="$1"
  local logfile="$2"
  local start_epoch end_epoch duration rc status
  start_epoch=$(date +%s)
  log "START $app logfile=$logfile"
  activate_env
  set +e
  if [[ "$TRACE" -eq 1 ]]; then
    stdbuf -oL -eL make -C "$HW_DIR" simv app="$app" trace=1 $EXTRA_MAKE_ARGS >"$logfile" 2>&1
    rc=$?
  else
    stdbuf -oL -eL make -C "$HW_DIR" simv app="$app" $EXTRA_MAKE_ARGS >"$logfile" 2>&1
    rc=$?
  fi
  set -e
  end_epoch=$(date +%s)
  duration=$((end_epoch - start_epoch))

  case "$rc" in
    0) status="PASS" ;;
    *) status="FAIL" ;;
  esac

  printf '%s,%s,%s,%s\n' "$app" "$status" "$duration" "$logfile" >> "$SUMMARY_CSV"
  echo "$status $app ${duration}s $logfile" | tee -a "$RUNNER_LOG"
  if [[ "$rc" -ne 0 ]]; then
    {
      echo "----- FAIL tail: $app -----"
      tail -n 40 "$logfile" 2>/dev/null || true
      echo "----- end FAIL tail: $app -----"
    } >> "$RUNNER_LOG"
  fi
  return 0
}

run_batch() {
  local batch_name="$1"
  shift
  local -a apps=("$@")
  local batch_dir="$LOG_ROOT/$batch_name"
  local -a active_pids=() active_apps=() active_starts=()
  local last_heartbeat now pid app idx age msg
  mkdir -p "$batch_dir"
  log "Running $batch_name with ${#apps[@]} apps, parallel=$PARALLEL"
  last_heartbeat=$(date +%s)
  IN_BATCH=1
  CURRENT_BATCH_NAME="$batch_name"
  ACTIVE_PIDS=()

  reap_finished() {
    local -a keep_pids=() keep_apps=() keep_starts=()
    for idx in "${!active_pids[@]}"; do
      pid="${active_pids[$idx]}"
      if kill -0 "$pid" 2>/dev/null; then
        keep_pids+=("$pid")
        keep_apps+=("${active_apps[$idx]}")
        keep_starts+=("${active_starts[$idx]}")
      else
        wait "$pid" || true
      fi
    done
    active_pids=("${keep_pids[@]}")
    active_apps=("${keep_apps[@]}")
    active_starts=("${keep_starts[@]}")
    ACTIVE_PIDS=("${active_pids[@]}")
  }

  emit_heartbeat() {
    if (( ${#active_pids[@]} == 0 )); then
      return 0
    fi
    now=$(date +%s)
    if (( now - last_heartbeat < HEARTBEAT_SEC )); then
      return 0
    fi
    msg="Heartbeat $batch_name: ${#active_pids[@]} running"
    for idx in "${!active_pids[@]}"; do
      age=$((now - ${active_starts[$idx]}))
      msg+=" | ${active_apps[$idx]}(${age}s)"
    done
    log "$msg"
    last_heartbeat=$now
  }

  for app in "${apps[@]}"; do
    run_one_app "$app" "$batch_dir/${app}.log" &
    pid=$!
    active_pids+=("$pid")
    active_apps+=("$app")
    active_starts+=("$(date +%s)")
    ACTIVE_PIDS=("${active_pids[@]}")
    log "LAUNCH $batch_name app=$app pid=$pid logfile=$batch_dir/${app}.log"
    while (( ${#active_pids[@]} >= PARALLEL )); do
      sleep "$POLL_SEC"
      reap_finished
      emit_heartbeat
    done
  done

  while (( ${#active_pids[@]} > 0 )); do
    sleep "$POLL_SEC"
    reap_finished
    emit_heartbeat
  done

  log "Completed $batch_name"
  ACTIVE_PIDS=()
  CURRENT_BATCH_NAME=""
  IN_BATCH=0
}

main() {
  local -a apps batch_apps
  local total idx batch_idx

  mapfile -t apps < <(list_target_apps)
  if (( ${#apps[@]} == 0 )); then
    echo "No apps matched the requested filters." >&2
    exit 1
  fi

  printf '%s\n' "${apps[@]}" > "$APPS_TXT"
  printf 'app,status,duration_sec,logfile\n' > "$SUMMARY_CSV"

  log "Log root: $LOG_ROOT"
  log "Selected ${#apps[@]} apps"
  log "Apps: $(join_by ', ' "${apps[@]}")"

  case "$MODE" in
    all|build)
      if [[ "$REBUILD_APPS" -eq 1 ]]; then
        build_apps "${apps[@]}"
      else
        log "Skipping app rebuild by request"
      fi
      build_verilator
      ;;
  esac

  if [[ "$MODE" == "build" ]]; then
    log "Build-only mode completed"
    exit 0
  fi

  total=${#apps[@]}
  idx=0
  batch_idx=0
  while (( idx < total )); do
    batch_apps=("${apps[@]:idx:BATCH_SIZE}")
    run_batch "$(printf 'batch_%02d' "$batch_idx")" "${batch_apps[@]}"
    idx=$((idx + BATCH_SIZE))
    batch_idx=$((batch_idx + 1))
  done

  log "All batches completed"
}

main "$@"
