# Debug Scripts

`scripts/debug/` contains focused launchers for module-level RTL debug, separate from full benchmark campaigns.

## Main Files

| File | Purpose |
| --- | --- |
| `run_sa_tb.sh` | launch or rebuild the systolic-array testbench flow |
| `run_lbmac_tb.sh` | launch or rebuild the LBMAC testbench flow |

## When To Use These Scripts

Use this directory when:

- a full `ara/hardware simv` run is too large for the bug you are chasing
- the issue is isolated to `bmpu`, `sa`, or `lbmac`
- you want a shorter reproduce-debug-edit loop

## Scope Boundary

These scripts are for block-level debug, not for paper-style benchmark evaluation and not for full-system correctness signoff. If the change affects integrated BMPU behavior, rerun the maintained full-flow check through `bmpu_verify` after block-level debugging.

For the corresponding source files, see [`../../src/hardware/rtl/bmpu/README.md`](../../src/hardware/rtl/bmpu/README.md).
