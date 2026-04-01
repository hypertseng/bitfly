# Lane RTL

`src/hardware/rtl/extended/src/lane/` contains lane-local overlays for scheduling, operand handling, and vector register behavior.

## Files

- `lane.sv`: lane-level integration
- `lane_sequencer.sv`: lane scheduling logic
- `operand_queues_stage.sv`: operand queue staging
- `operand_requester.sv`: operand request path
- `vector_regfile.sv`: vector register file overlay

## Typical Problems Routed Here

This directory is the right place for bugs that show up as:

- lane imbalance or starvation
- operand request / response mismatches
- issue-side scheduling stalls
- vector register interaction issues near BMPU or VFU execution

## Edit Boundary

If the bug is purely inside BMPU arithmetic or SA behavior, start from [`../../../bmpu/README.md`](../../../bmpu/README.md) instead of here.
