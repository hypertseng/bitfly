# BMPU Testbenches

`src/hardware/rtl/bmpu/tb/` contains focused testbench sources for BMPU-local debug.

## Files

- `sa_tb.sv`: systolic-array testbench
- `sa_tb_main.cpp`: C++ harness for the SA testbench
- `sa_stub_pkgs.sv`: package stubs used by the SA testbench flow
- `lbmac_tb_main.cpp`: C++ harness for the LBMAC testbench

## Intended Use

Use these testbenches for the shortest debug loop when the issue is local to BMPU building blocks and does not require the full Ara integration stack.

They are useful for:

- SA functional bring-up
- LBMAC primitive debugging
- quick waveform-oriented checks before rerunning full-system validation

These files are the source-side counterpart of the scripts documented in [`../../../../../scripts/debug/README.md`](../../../../../scripts/debug/README.md).
