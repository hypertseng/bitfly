# Hardware Overlays

`src/hardware/` contains bitfly-managed RTL overlays that are synced into the Ara hardware tree.

## Layout

- [`rtl/README.md`](rtl/README.md): top-level RTL organization

## Ownership Model

Use this tree for RTL that bitfly wants to preserve independently from the working Ara checkout, especially:

- BMPU-specific modules
- Ara integration changes needed by custom instructions or datapaths
- local testbench files used by focused debug scripts

The sync flow is documented in [`../../scripts/dev/README.md`](../../scripts/dev/README.md).
