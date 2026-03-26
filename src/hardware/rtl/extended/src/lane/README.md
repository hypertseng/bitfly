# Lane RTL

`src/hardware/rtl/extended/src/lane/` contains lane-local overlays for scheduling, operand handling, and vector register behavior.

## Files

- `lane.sv`: lane-level integration
- `lane_sequencer.sv`: lane scheduling logic
- `operand_queues_stage.sv`: operand queue staging
- `operand_requester.sv`: operand request path
- `vector_regfile.sv`: vector register file overlay

This directory is the right place for bugs that show up as lane imbalance, operand starvation, or register-file interaction issues.
