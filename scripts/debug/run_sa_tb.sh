#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/sa_tb"
SRC_BMPU="$ROOT_DIR/src/hardware/rtl/bmpu"
TB_PKG="$SRC_BMPU/tb/sa_stub_pkgs.sv"
TB_CPP="$SRC_BMPU/tb/sa_tb_main.cpp"

VERILATOR="${VERILATOR:-}"
if [[ -z "$VERILATOR" ]]; then
  if [[ -x "$ROOT_DIR/ara/install/verilator/bin/verilator" ]]; then
    VERILATOR="$ROOT_DIR/ara/install/verilator/bin/verilator"
  elif [[ -x "$ROOT_DIR/ara/toolchain/verilator/bin/verilator" ]]; then
    VERILATOR="$ROOT_DIR/ara/toolchain/verilator/bin/verilator"
  else
    VERILATOR="verilator"
  fi
fi

COMPILER="${SA_TB_COMPILER:-}"
if [[ -z "$COMPILER" ]]; then
  COMPILER="clang"
  if ! command -v clang++ >/dev/null 2>&1; then
    COMPILER="gcc"
  fi
fi

mkdir -p "$BUILD_DIR"

"$VERILATOR" -Wall -Wno-fatal \
  --cc --exe --build \
  --compiler "$COMPILER" \
  --Mdir "$BUILD_DIR" \
  --top-module sa \
  "$TB_PKG" \
  "$SRC_BMPU/lbmac.sv" \
  "$SRC_BMPU/pe.sv" \
  "$SRC_BMPU/sa.sv" \
  "$TB_CPP"

"$BUILD_DIR/Vsa"
