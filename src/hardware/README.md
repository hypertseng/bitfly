# Hardware Overlays

`src/hardware/` contains the maintained RTL overlays that BitFly syncs into the Ara hardware tree.

## Start Here

Use this tree when the RTL change should remain part of BitFly rather than only a local Ara experiment.

Typical examples:

- BMPU-specific modules
- Ara integration changes required by custom instructions or scheduling logic
- local testbench collateral used by focused debug scripts

## Layout

- [`rtl/README.md`](rtl/README.md): top-level RTL organization and the split between BMPU-local and Ara-integration overlays

## Ownership Rule

- Preserve maintainable BitFly RTL under `src/hardware/`
- Sync it into `ara/hardware/` for build and simulation
- Avoid making `ara/` the accidental source of truth

The maintained sync path is documented in [`../../scripts/dev/README.md`](../../scripts/dev/README.md).
