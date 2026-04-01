# TCL Scripts

`tcl/` stores synthesis-oriented TCL collateral that is separate from the software build and benchmark flows.

## Layout

- [`synthesis/README.md`](synthesis/README.md): synthesis entry scripts and constraints

Use this directory when you are working on implementation or timing flows rather than simulation or app execution.

## Boundary

Keep synthesis assumptions here instead of scattering them into benchmark, debug, or application scripts. This separation keeps implementation collateral reviewable for hardware signoff.
