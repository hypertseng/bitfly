# RTL Overlays

`src/hardware/rtl/` is split into two concerns:

- [`bmpu/README.md`](bmpu/README.md): BMPU-local modules and focused testbenches
- [`extended/README.md`](extended/README.md): Ara-side integration overlays and broader RTL changes

In practice:

- change `bmpu/` when the logic belongs to the BMPU block itself
- change `extended/` when the logic spans Ara integration, lane scheduling, or memory-side behavior
