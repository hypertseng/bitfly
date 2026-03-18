# Scripts Guide

The `scripts/` directory is organized by workflow so that experiment automation, analysis, and RTL debug utilities are easier to find.

## Layout

- `scripts/benchmarks/`
  - benchmark orchestration and shape extraction
  - main entry: `scripts/benchmarks/run_model_split_apps.sh`
- `scripts/analysis/`
  - plotting and design-space analysis helpers
- `scripts/debug/`
  - focused RTL unit-test or module-level debug launchers
- `scripts/dev/`
  - source sync / maintenance helpers

## Compatibility Wrappers

For convenience, the original entry paths are still kept as thin wrappers:

- `scripts/run_model_split_apps.sh`
- `scripts/run_lbmac_tb.sh`
- `scripts/run_sa_tb.sh`
- `scripts/sync_src_to_ara.sh`

This means existing command history remains valid while new work can use the categorized layout.

## Recommended Entry Points

- Full benchmark batches: `scripts/benchmarks/run_model_split_apps.sh`
- Roofline plotting: `scripts/analysis/roofline.py`
- SA debug: `scripts/debug/run_sa_tb.sh`
- LBMAC debug: `scripts/debug/run_lbmac_tb.sh`
- Sync `src/` overlays into `ara/`: `scripts/dev/sync_src_to_ara.sh`
