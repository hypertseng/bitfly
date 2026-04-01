# Synthesis TCL

`tcl/synthesis/` contains the current synthesis script set.

## Files

- `run.tcl`: main synthesis entry script
- `ara.tcl`: design-specific setup for the Ara-based target
- `constraint.sdc`: timing and constraint file

## Intended Use

This directory is the handoff point for hardware implementation flows. Keep synthesis-only assumptions here instead of mixing them into simulation scripts.

## Documentation Expectation

If synthesis inputs, constraints, or handoff assumptions change, update this README together with the Tcl files so the implementation flow remains auditable.
