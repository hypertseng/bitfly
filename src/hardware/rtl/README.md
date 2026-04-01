# RTL Overlays

`src/hardware/rtl/` is intentionally split into two concerns:

- [`bmpu/README.md`](bmpu/README.md): BMPU-local modules and focused testbenches
- [`extended/README.md`](extended/README.md): Ara-integration overlays and broader RTL changes

## Directory Map

| Path | Role | Typical Files |
| --- | --- | --- |
| `bmpu/` | BMPU-local compute and storage logic | `bmpu.sv`, `sa.sv`, `pe.sv`, `lbmac.sv` |
| `bmpu/tb/` | focused unit-style testbenches | block-level bring-up collateral |
| `extended/include/` | Ara package/include overlays | typedef and package updates |
| `extended/src/` | Ara integration RTL overlays | dispatcher, sequencer, lane, VLSU |
| `extended/scripts/` | waveform and local debug helpers | Tcl / waveform helpers |
| `extended/tb/` | integration testbench overlays | top-level or simulation support files |

## Edit Rule

- Change `bmpu/` when the logic belongs to the BMPU block itself.
- Change `extended/` when the logic affects Ara integration, lane scheduling, dispatcher/sequencer behavior, or memory-side interaction.

## Common Navigation

If you are looking for:

- BMPU datapath execution state: start in `bmpu/bmpu.sv`
- systolic-array behavior: start in `bmpu/sa.sv`
- lane-side BMPU issue logic: start in `extended/src/lane/`
- Ara request formation or custom instruction handling: start in `extended/src/ara_dispatcher.sv`
- global instruction sequencing: start in `extended/src/ara_sequencer.sv`
- vector load/store support logic: start in `extended/src/vlsu/`

This split keeps local accelerator logic separate from the broader vector-core integration work.
