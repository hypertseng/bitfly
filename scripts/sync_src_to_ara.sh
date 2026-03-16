#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
ARA_DIR="$ROOT_DIR/ara"
PATCH_DIR_DEFAULT="$ROOT_DIR/patches/local"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PATCH_FILE_DEFAULT="$PATCH_DIR_DEFAULT/qloma_sync_${TIMESTAMP}.patch"

DO_SYNC=1
DO_PATCH=1
DO_STATUS=1
PATCH_FILE="$PATCH_FILE_DEFAULT"
LLVM_DST=""

usage() {
  cat <<'EOF'
Usage: scripts/sync_src_to_ara.sh [options]

Options:
  --no-sync            Skip rsync step
  --no-patch           Skip patch generation
  --no-status          Skip final git status output
  --patch-file <path>  Output patch path (default: patches/local/qloma_sync_<ts>.patch)
  --llvm-dst <path>    Optional destination root for src/llvm_instr sync
  -h, --help           Show this help

What this script does:
  1) Syncs selected QLoMA src modifications into ara workspace
  2) Generates a patch from ara repo changes (includes untracked files)
  3) Prints concise status summary
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sync)
      DO_SYNC=0
      shift
      ;;
    --no-patch)
      DO_PATCH=0
      shift
      ;;
    --no-status)
      DO_STATUS=0
      shift
      ;;
    --patch-file)
      PATCH_FILE="$2"
      shift 2
      ;;
    --llvm-dst)
      LLVM_DST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$SRC_DIR" ]]; then
  echo "[ERROR] Missing src directory: $SRC_DIR" >&2
  exit 1
fi

if ! git -C "$ARA_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ERROR] Missing ara git repository: $ARA_DIR" >&2
  exit 1
fi

sync_one() {
  local from="$1"
  local to="$2"
  if [[ ! -d "$from" ]]; then
    echo "[WARN] Skip missing source: $from"
    return 0
  fi
  mkdir -p "$to"
  rsync -a "$from/" "$to/"
  echo "[SYNC] $from -> $to"
}

if [[ "$DO_SYNC" -eq 1 ]]; then
  sync_one "$SRC_DIR/hardware/rtl/bmpu" "$ARA_DIR/hardware/src/bmpu"
  sync_one "$SRC_DIR/hardware/rtl/extended/src" "$ARA_DIR/hardware/src"
  sync_one "$SRC_DIR/hardware/rtl/extended/include" "$ARA_DIR/hardware/include"
  sync_one "$SRC_DIR/apps" "$ARA_DIR/apps"

  if [[ -n "$LLVM_DST" ]]; then
    sync_one "$SRC_DIR/llvm_instr" "$LLVM_DST"
  fi
fi

if [[ "$DO_PATCH" -eq 1 ]]; then
  mkdir -p "$(dirname "$PATCH_FILE")"
  : > "$PATCH_FILE"

  # Tracked modifications under synced trees
  git -C "$ARA_DIR" --no-pager diff --binary -- \
    hardware/src/bmpu \
    hardware/src \
    hardware/include \
    apps \
    >> "$PATCH_FILE"

  # Untracked files under synced trees
  while IFS= read -r rel; do
    (
      cd "$ARA_DIR"
      git --no-pager diff --binary --no-index /dev/null "$rel"
    ) >> "$PATCH_FILE" || true
  done < <(
    git -C "$ARA_DIR" ls-files --others --exclude-standard -- \
      hardware/src/bmpu \
      hardware/src \
      hardware/include \
      apps
  )

  echo "[PATCH] Generated: $PATCH_FILE"
  echo "[PATCH] Size: $(wc -c < "$PATCH_FILE") bytes"
fi

if [[ "$DO_STATUS" -eq 1 ]]; then
  echo "[STATUS] ara repo changes (short):"
  git -C "$ARA_DIR" status --short -- \
    hardware/src/bmpu \
    hardware/src \
    hardware/include \
    apps || true
fi
