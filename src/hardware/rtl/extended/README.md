# Extended RTL

`src/hardware/rtl/extended/` contains the Ara-side integration overlays needed to expose and support bitfly functionality in the broader hardware stack.

## Layout

- `include/`: package and type overlays synced into `ara/hardware/include/`
- `src/`: Ara module overlays synced into `ara/hardware/src/`
- `scripts/`: waveform helpers such as `wave_lane.tcl`
- `tb/`: integration testbench files such as the Verilator top

## `src/` Navigation

Within `extended/src/`, the most common areas are:

| Path | Use It For |
| --- | --- |
| `ara_dispatcher.sv` | request formation, decode-side control, BMPU issue plumbing |
| `ara_sequencer.sv` | global vector instruction sequencing |
| `lane/` | lane-local scheduling, operand movement, and VFU issue |
| `vlsu/` | vector load/store support and address generation |

## Typical Change Areas

- dispatcher and sequencer behavior
- lane scheduling and operand movement
- VLSU-side support logic
- package definitions shared across Ara modules

## When Not To Edit Here

Do not start here when the change is purely local to BMPU arithmetic or SA behavior. In that case, use [`../bmpu/README.md`](../bmpu/README.md) first and only come back here if the change crosses the block boundary into lane or Ara integration.

This directory is the bridge between BMPU-local RTL and the full Ara execution flow.
