# RTL Overlays

`src/hardware/rtl/` is intentionally split into two concerns:

- [`bmpu/README.md`](bmpu/README.md): BMPU-local modules and focused testbenches
- [`extended/README.md`](extended/README.md): Ara-integration overlays and broader RTL changes

## Edit Rule

- Change `bmpu/` when the logic belongs to the BMPU block itself.
- Change `extended/` when the logic affects Ara integration, lane scheduling, dispatcher/sequencer behavior, or memory-side interaction.

This split keeps local accelerator logic separate from the broader vector-core integration work.
