#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data2/zzx/data/workspace/QLoMA"
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
LOG_ROOT=""
MODELS_CSV=""
PRECS_CSV=""
IMPLS_CSV=""
APPS_CSV=""
EXTRA_MAKE_ARGS=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --mode <all|build|run>       Default: all
  --build-jobs <N>             make -j for app/hardware build, default: 16
  --parallel <N>               concurrent simv jobs, default: 6
  --batch-size <N>             apps per batch, default: 6
  --models <csv>               default: gemma3_270m,qwen25_05b,opt_13b,qwen25_15b,gemma2_2b
  --precisions <csv>           default: binary,INT2,INT4
  --impls <csv>                default: bmpmm,rvv
  --apps <csv>                 explicit app list, overrides model/precision/impl filters
  --log-root <dir>             default: tmp/model_app_runs/<timestamp>
  --no-verilate                skip 'make -C ara/hardware verilate'
  --no-rebuild-apps            skip rebuilding app binaries
  --trace                      pass trace=1 to hardware make
  --extra-make-args <string>   extra args appended to hardware make
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
    --extra-make-args) EXTRA_MAKE_ARGS="$2"; shift 2 ;;
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
  LOG_ROOT="$TMP_DIR/model_app_runs/$(date +%Y%m%d_%H%M%S)_30apps"
fi
mkdir -p "$LOG_ROOT"
RUNNER_LOG="$LOG_ROOT/runner.log"
SUMMARY_CSV="$LOG_ROOT/summary.csv"
APPS_TXT="$LOG_ROOT/apps.txt"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUNNER_LOG"
}

activate_env() {
  source "$CONDA_SH"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  conda activate "$CONDA_ENV_NAME"
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
  local -a all_apps filtered models precs impls explicit
  mapfile -t all_apps < <(find "$APPS_DIR" -maxdepth 1 -mindepth 1 -type d | sed 's#^.*/##' | sort)

  if [[ -n "$APPS_CSV" ]]; then
    split_csv "$APPS_CSV" explicit
    printf '%s\n' "${explicit[@]}"
    return 0
  fi

  split_csv "$MODELS_CSV" models
  split_csv "$PRECS_CSV" precs
  split_csv "$IMPLS_CSV" impls

  if [[ ${#models[@]} -eq 0 ]]; then
    models=(gemma3_270m qwen25_05b opt_13b qwen25_15b gemma2_2b)
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
  activate_env
  make -j"$BUILD_JOBS" -C "$APPS_DIR" -B "${bins[@]}"
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
  activate_env
  if [[ "$TRACE" -eq 1 ]]; then
    stdbuf -oL -eL make -C "$HW_DIR" simv app="$app" trace=1 $EXTRA_MAKE_ARGS >"$logfile" 2>&1
    rc=$?
  else
    stdbuf -oL -eL make -C "$HW_DIR" simv app="$app" $EXTRA_MAKE_ARGS >"$logfile" 2>&1
    rc=$?
  fi
  end_epoch=$(date +%s)
  duration=$((end_epoch - start_epoch))

  case "$rc" in
    0) status="PASS" ;;
    *) status="FAIL" ;;
  esac

  printf '%s,%s,%s,%s\n' "$app" "$status" "$duration" "$logfile" >> "$SUMMARY_CSV"
  echo "$status $app ${duration}s $logfile" | tee -a "$RUNNER_LOG"
  return 0
}

run_batch() {
  local batch_name="$1"
  shift
  local -a apps=("$@")
  local batch_dir="$LOG_ROOT/$batch_name"
  mkdir -p "$batch_dir"
  log "Running $batch_name with ${#apps[@]} apps, parallel=$PARALLEL"

  local running=0
  for app in "${apps[@]}"; do
    run_one_app "$app" "$batch_dir/${app}.log" &
    running=$((running + 1))
    if (( running >= PARALLEL )); then
      wait -n || true
      running=$((running - 1))
    fi
  done
  wait || true
}

main() {
  local -a apps
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
    log "Build-only mode finished"
    exit 0
  fi

  local total=${#apps[@]}
  local start=0
  local batch_idx=0
  if (( BATCH_SIZE <= 0 || BATCH_SIZE >= total )); then
    run_batch batch_00 "${apps[@]}"
  else
    while (( start < total )); do
      local -a chunk=("${apps[@]:start:BATCH_SIZE}")
      run_batch "$(printf 'batch_%02d' "$batch_idx")" "${chunk[@]}"
      start=$((start + BATCH_SIZE))
      batch_idx=$((batch_idx + 1))
    done
  fi

  log "All requested runs finished"
  log "Summary CSV: $SUMMARY_CSV"
}

main "$@"
