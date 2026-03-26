# BMPU RTL

`src/hardware/rtl/bmpu/` contains the BMPU-local RTL and the smallest unit-test style testbenches in the repository.

## Main Files

- `bmpu.sv`: top-level BMPU logic
- `sa.sv`: systolic-array logic
- `lbmac.sv`: LBMAC implementation
- `pe.sv`: processing-element logic
- `tb/`: standalone testbench support for `sa` and `lbmac`

## When To Edit Here

Use this directory for changes that are local to the BMPU block or its direct unit tests.

For full-system Ara integration changes, move up to [`../extended/README.md`](../extended/README.md).
