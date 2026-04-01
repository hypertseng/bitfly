#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_SRC="$ROOT_DIR/src/apps/llama2/main.c"
ARA_APP_DIR="$ROOT_DIR/ara/apps/llama2"
ARA_APPS_DIR="$ROOT_DIR/ara/apps"
HW_DIR="$ROOT_DIR/ara/hardware"
TMP_DIR="$ROOT_DIR/tmp"

BUILD_JOBS=8
BUILD_HW=0
MODELS_CSV="15M,42M,110M,1B,3B"
PRECS_CSV="W1A8,W2A8,W4A8"
SEQS_CSV="32,64,128,256"
LOG_ROOT=""
DEFAULT_SEQS_CSV="32,64,128,256"

usage() {
  cat <<'EOF'
Usage: run_llama2_verilator_prefill_sweep.sh [options]

Options:
  --models <csv>        Model subset. Default: 15M,42M,110M,1B,3B
  --precisions <csv>    Precision subset. Default: W1A8,W2A8,W4A8
  --seqs <csv>          Sequence-length subset. Default: 32,64,128,256
  --build-jobs <N>      make -j value. Default: 8
  --build-hw            Rebuild Verilator hardware before the sweep
  --log-root <dir>      Output directory. Default: tmp/llama2_verilator_prefill/<timestamp>
  -h, --help            Show this help

Example:
  scripts/benchmarks/run_llama2_verilator_prefill_sweep.sh \
    --models 15M,42M --precisions W1A8,W2A8 --seqs 32,64
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS_CSV="$2"; shift 2 ;;
    --precisions) PRECS_CSV="$2"; shift 2 ;;
    --seqs) SEQS_CSV="$2"; shift 2 ;;
    --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
    --build-hw) BUILD_HW=1; shift ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$LOG_ROOT" ]]; then
  LOG_ROOT="$TMP_DIR/llama2_verilator_prefill/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_ROOT"
SUMMARY_CSV="$LOG_ROOT/summary.csv"
RUNNER_LOG="$LOG_ROOT/runner.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUNNER_LOG"
}

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  out_ref=()
  IFS=',' read -r -a out_ref <<< "$csv"
}

seqs_are_edge_subset() {
  local -a seqs
  split_csv "$SEQS_CSV" seqs
  for seq in "${seqs[@]}"; do
    [[ ",$DEFAULT_SEQS_CSV," == *",$seq,"* ]] || return 1
  done
  return 0
}

model_id() {
  case "$1" in
    15M) echo 1 ;;
    42M) echo 2 ;;
    110M) echo 3 ;;
    1B) echo 4 ;;
    3B) echo 5 ;;
    *) echo "Unknown model: $1" >&2; exit 1 ;;
  esac
}

prec_id() {
  case "$1" in
    W1A8) echo 1 ;;
    W2A8) echo 2 ;;
    W4A8) echo 3 ;;
    *) echo "Unknown precision: $1" >&2; exit 1 ;;
  esac
}

model_matches() {
  local requested="$1"
  local actual="$2"
  case "$requested" in
    1B)
      [[ "$actual" == "1B" || "$actual" == *"1B"* ]]
      ;;
    3B)
      [[ "$actual" == "3B" || "$actual" == *"3B"* ]]
      ;;
    *)
      [[ "$actual" == "$requested" ]]
      ;;
  esac
}

sync_llama2_app() {
  rsync -a --delete \
    --exclude '*.o' \
    --exclude '*.o.spike' \
    --exclude 'data.S' \
    "$ROOT_DIR/src/apps/llama2/" "$ARA_APP_DIR/"
}

build_case() {
  local model="$1"
  local prec="$2"
  local seq="$3"
  local model_define
  local prec_define
  model_define="$(model_id "$model")"
  prec_define="$(prec_id "$prec")"

  log "Building case model=$model prec=$prec seq=$seq"
  sync_llama2_app
  make -C "$ARA_APPS_DIR" -j"$BUILD_JOBS" \
    ENV_DEFINES="-DLLAMA2_FILTER_MODEL=$model_define -DLLAMA2_FILTER_PREC=$prec_define -DLLAMA2_FILTER_SEQ_LEN=$seq" \
    bin/llama2
}

build_hw_if_needed() {
  if [[ "$BUILD_HW" -eq 0 ]]; then
    return 0
  fi
  log "Rebuilding Verilator hardware"
  make -C "$HW_DIR" -j"$BUILD_JOBS" verilate
}

sanitize_name() {
  echo "$1" | tr '/ ' '__'
}

extract_prefill_line() {
  local logfile="$1"
  grep -E '^LLAMA2_RESULT,|^\[llama2\]   prefill_total_cycles:' "$logfile" | tail -n 1 || true
}

append_summary() {
  local model="$1"
  local prec="$2"
  local seq="$3"
  local logfile="$4"
  local line
  local bmpmm
  local rvv
  local speedup
  line="$(extract_prefill_line "$logfile")"
  if [[ -z "$line" ]]; then
    echo "$model,$prec,$seq,FAIL,FAIL,FAIL,$logfile" >> "$SUMMARY_CSV"
    return 1
  fi

  if [[ "$line" == LLAMA2_RESULT,* ]]; then
    bmpmm="$(echo "$line" | cut -d, -f5)"
    rvv="$(echo "$line" | cut -d, -f6)"
    speedup="$(echo "$line" | cut -d, -f7)"
  else
    bmpmm="$(echo "$line" | sed -n 's/.*bmpmm=\([0-9][0-9]*\).*/\1/p')"
    rvv="$(echo "$line" | sed -n 's/.*rvv=\([0-9][0-9]*\).*/\1/p')"
    speedup="$(echo "$line" | sed -n 's/.*speedup=\([0-9.][0-9.]*x\).*/\1/p')"
  fi
  echo "$model,$prec,$seq,$bmpmm,$rvv,$speedup,$logfile" >> "$SUMMARY_CSV"
}

append_summary_from_result_lines() {
  local logfile="$1"
  local count=0
  while IFS= read -r line; do
    [[ "$line" == LLAMA2_RESULT,* ]] || continue
    echo "$line" | awk -F',' -v logfile="$logfile" '{
      printf "%s,%s,%s,%s,%s,%s,%s\n", $2, $3, $4, $5, $6, $7, logfile
    }' >> "$SUMMARY_CSV"
    count=$((count + 1))
  done < "$logfile"

  if [[ "$count" -eq 0 ]]; then
    return 1
  fi
  return 0
}

append_pair_summary_from_results() {
  local model="$1"
  local prec="$2"
  local seqs_csv="$3"
  local logfile="$4"
  local count=0
  while IFS= read -r line; do
    local line_model
    local line_prec
    local line_seq
    local line_bmpmm
    local line_rvv
    local line_speedup
    [[ "$line" == LLAMA2_RESULT,* ]] || continue
    line_model="$(echo "$line" | cut -d, -f2)"
    line_prec="$(echo "$line" | cut -d, -f3)"
    line_seq="$(echo "$line" | cut -d, -f4)"
    model_matches "$model" "$line_model" || continue
    [[ "$line_prec" == "$prec" ]] || continue
    [[ ",$seqs_csv," == *",$line_seq,"* ]] || continue
    line_bmpmm="$(echo "$line" | cut -d, -f5)"
    line_rvv="$(echo "$line" | cut -d, -f6)"
    line_speedup="$(echo "$line" | cut -d, -f7)"
    echo "$model,$line_prec,$line_seq,$line_bmpmm,$line_rvv,$line_speedup,$logfile" >> "$SUMMARY_CSV"
    count=$((count + 1))
  done < "$logfile"

  if [[ "$count" -eq 0 ]]; then
    return 1
  fi
  return 0
}

append_model_summary_from_results() {
  local model="$1"
  local precs_csv="$2"
  local seqs_csv="$3"
  local logfile="$4"
  local count=0
  while IFS= read -r line; do
    local line_model
    local line_prec
    local line_seq
    local line_bmpmm
    local line_rvv
    local line_speedup
    [[ "$line" == LLAMA2_RESULT,* ]] || continue
    line_model="$(echo "$line" | cut -d, -f2)"
    line_prec="$(echo "$line" | cut -d, -f3)"
    line_seq="$(echo "$line" | cut -d, -f4)"
    model_matches "$model" "$line_model" || continue
    [[ ",$precs_csv," == *",$line_prec,"* ]] || continue
    [[ ",$seqs_csv," == *",$line_seq,"* ]] || continue
    line_bmpmm="$(echo "$line" | cut -d, -f5)"
    line_rvv="$(echo "$line" | cut -d, -f6)"
    line_speedup="$(echo "$line" | cut -d, -f7)"
    echo "$model,$line_prec,$line_seq,$line_bmpmm,$line_rvv,$line_speedup,$logfile" >> "$SUMMARY_CSV"
    count=$((count + 1))
  done < "$logfile"

  if [[ "$count" -eq 0 ]]; then
    return 1
  fi
  return 0
}

run_case() {
  local model="$1"
  local prec="$2"
  local seq="$3"
  local log_name
  local logfile

  log_name="$(sanitize_name "model_${model}__prec_${prec}__seq_${seq}.log")"
  logfile="$LOG_ROOT/$log_name"
  log "Running case model=$model prec=$prec seq=$seq"
  (
    cd "$HW_DIR"
    ./build/verilator/Vara_tb_verilator -l ram,../apps/bin/llama2,elf
  ) | tee "$logfile"
  append_summary "$model" "$prec" "$seq" "$logfile"
}

build_pair_app() {
  local model="$1"
  local prec="$2"
  local model_define
  local prec_define
  model_define="$(model_id "$model")"
  prec_define="$(prec_id "$prec")"

  log "Building grouped case model=$model prec=$prec seqs=$SEQS_CSV"
  sync_llama2_app
  make -C "$ARA_APPS_DIR" -j"$BUILD_JOBS" \
    ENV_DEFINES="-DLLAMA2_FILTER_MODEL=$model_define -DLLAMA2_FILTER_PREC=$prec_define -DLLAMA2_FILTER_SEQ_LEN=0" \
    bin/llama2
}

build_model_app() {
  local model="$1"
  local model_define
  model_define="$(model_id "$model")"

  log "Building grouped model=$model precs=$PRECS_CSV seqs=$SEQS_CSV"
  sync_llama2_app
  make -C "$ARA_APPS_DIR" -j"$BUILD_JOBS" \
    ENV_DEFINES="-DLLAMA2_FILTER_MODEL=$model_define -DLLAMA2_FILTER_PREC=0 -DLLAMA2_FILTER_SEQ_LEN=0" \
    bin/llama2
}

run_pair_case() {
  local model="$1"
  local prec="$2"
  local logfile="$LOG_ROOT/$(sanitize_name "model_${model}__prec_${prec}__seqs_group.log")"
  log "Running grouped case model=$model prec=$prec seqs=$SEQS_CSV"
  (
    cd "$HW_DIR"
    ./build/verilator/Vara_tb_verilator -l ram,../apps/bin/llama2,elf
  ) | tee "$logfile"
  append_pair_summary_from_results "$model" "$prec" "$SEQS_CSV" "$logfile"
}

run_model_case() {
  local model="$1"
  local logfile="$LOG_ROOT/$(sanitize_name "model_${model}__group.log")"
  log "Running grouped model=$model precs=$PRECS_CSV seqs=$SEQS_CSV"
  (
    cd "$HW_DIR"
    ./build/verilator/Vara_tb_verilator -l ram,../apps/bin/llama2,elf
  ) | tee "$logfile"
  append_model_summary_from_results "$model" "$PRECS_CSV" "$SEQS_CSV" "$logfile"
}

main() {
  local -a models precs seqs
  split_csv "$MODELS_CSV" models
  split_csv "$PRECS_CSV" precs
  split_csv "$SEQS_CSV" seqs

  {
    echo "model,precision,seq_len,bmpmm_cycles,rvv_cycles,speedup,logfile"
  } > "$SUMMARY_CSV"

  build_hw_if_needed

  if seqs_are_edge_subset; then
    for model in "${models[@]}"; do
      for prec in "${precs[@]}"; do
        build_pair_app "$model" "$prec"
        run_pair_case "$model" "$prec"
      done
    done
  else
    for model in "${models[@]}"; do
      for prec in "${precs[@]}"; do
        for seq in "${seqs[@]}"; do
          build_case "$model" "$prec" "$seq"
          run_case "$model" "$prec" "$seq"
        done
      done
    done
  fi

  log "Sweep finished. Summary: $SUMMARY_CSV"
}

main "$@"
