# BMPU Testbenches

`src/hardware/rtl/bmpu/tb/` contains focused testbench sources for BMPU-local debug.

## Files

- `sa_tb.sv`: systolic-array testbench
- `sa_tb_main.cpp`: C++ harness for the SA testbench
- `sa_stub_pkgs.sv`: package stubs used by the SA testbench flow
- `lbmac_tb_main.cpp`: C++ harness for the LBMAC testbench

These files are the source-side counterpart of the scripts documented in [`../../../../../scripts/debug/README.md`](../../../../../scripts/debug/README.md).
