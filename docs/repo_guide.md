# Repository Guide

## Main Directories

- `ara/`: executable research platform based on Ara
- `src/`: bitfly-side source of truth for overlays and custom RTL/app sources
- `scripts/`: automation, analysis, and debug tools
- `docs/`: workflow notes and experiment-facing documentation
- `tmp/`: generated logs and run outputs
- `build/`: local unit-test / debug build outputs

## Source Ownership

When developing new bitfly features:

- edit persistent project logic under `src/` when the source is meant to be synced into Ara
- edit `ara/` directly when you intentionally want to change the working Ara tree
- use `scripts/dev/sync_src_to_ara.sh` to propagate managed overlays from `src/` to `ara/`

## Experiment Artifacts

Large logs, generated assembly, Spike objects, and temporary build products are intentionally excluded from Git via `.gitignore` rules.
