# Extended RTL

`src/hardware/rtl/extended/` contains the Ara-side integration overlays needed to expose and support bitfly functionality in the broader hardware stack.

## Layout

- `include/`: package and type overlays synced into `ara/hardware/include/`
- `src/`: Ara module overlays synced into `ara/hardware/src/`
- `scripts/`: waveform helpers such as `wave_lane.tcl`
- `tb/`: integration testbench files such as the Verilator top

## Typical Change Areas

- dispatcher and sequencer behavior
- lane scheduling and operand movement
- VLSU-side support logic
- package definitions shared across Ara modules

This directory is the bridge between BMPU-local RTL and the full Ara execution flow.
