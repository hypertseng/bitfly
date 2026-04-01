# VLSU RTL

`src/hardware/rtl/extended/src/vlsu/` contains vector load/store support overlays.

## Files

- `addrgen.sv`: address generation
- `vldu.sv`: vector load unit support
- `vlsu.sv`: top-level VLSU overlay
- `vstu.sv`: vector store unit support

## Typical Problems Routed Here

Start here when the issue is tied to:

- address generation
- load/store side state handling
- vector memory request timing
- BMPU-related behavior that only fails once data reaches the VLSU path

Edit this directory when the issue or feature is tied to vector memory movement rather than compute-side sequencing.
