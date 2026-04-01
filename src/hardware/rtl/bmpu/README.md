# BMPU RTL

`src/hardware/rtl/bmpu/` contains the BMPU-local RTL and the smallest unit-test style testbenches in the repository.

## Main Files

- `bmpu.sv`: top-level BMPU logic
- `sa.sv`: systolic-array logic
- `lbmac.sv`: LBMAC implementation
- `pe.sv`: processing-element logic
- `tb/`: standalone testbench support for `sa` and `lbmac`

## File Responsibilities

| File | Responsibility |
| --- | --- |
| `bmpu.sv` | top-level BMPU control, queueing, and result-store flow |
| `sa.sv` | systolic-array orchestration and PE array wiring |
| `pe.sv` | per-cell accumulation state and context handling |
| `lbmac.sv` | low-bit MAC primitive behavior |
| `tb/` | shortest-loop debug collateral for BMPU-local blocks |

## When To Edit Here

Use this directory for changes that are local to the BMPU block or its direct unit tests.

Typical examples:

- changing low-bit accumulation behavior
- updating BMPU-local issue / store sequencing
- modifying SA context management
- debugging PE or LBMAC behavior without involving the full Ara lane

For full-system Ara integration changes, move up to [`../extended/README.md`](../extended/README.md).
