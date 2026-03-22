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

## Background Benchmark Run

For long 30-app Verilator campaigns, prefer launching the runner through `/bin/bash -lc` together with `nohup`, so the Conda environment and working directory are both explicit.

```bash
LOG_ROOT=tmp/model_app_runs/formal_30apps_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
nohup /bin/bash -lc '
  source /data2/zzx/data/miniconda3/etc/profile.d/conda.sh &&
  conda activate bitfly &&
  cd /data2/zzx/data/workspace/bitfly &&
  scripts/benchmarks/run_model_split_apps.sh --mode run --no-rebuild-apps --no-verilate --parallel 5 --batch-size 5 --log-root $LOG_ROOT
' > "$LOG_ROOT/launch.log" 2>&1 &
```
