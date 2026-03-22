# Benchmark Workflow

This document describes the recommended workflow for the model-split LLM GEMM experiments used in formal evaluation.

## Goal

Run one app per:

- model
- precision
- implementation

and compare total runtime between:

- BMPMM-based implementation
- RVV implementation

across the selected model linear-layer GEMM shapes.

## App Matrix

For each model, prepare the following six apps:

- `bmpmm_binary_<model>`
- `bmpmm_INT2_<model>`
- `bmpmm_INT4_<model>`
- `rvv_binary_<model>`
- `rvv_INT2_<model>`
- `rvv_INT4_<model>`

Recommended five-model set:

- `gemma3_270m`
- `qwen25_05b`
- `opt_13b`
- `qwen25_15b`
- `gemma2_2b`

## Standard Execution Flow

Activate the `bitfly` environment, then run the benchmark launcher.

Build and run all selected apps:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode all --build-jobs 16 --parallel 5 --batch-size 5
```

Reuse existing builds and only rerun simulations:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode run --no-rebuild-apps --no-verilate --parallel 5 --batch-size 5
```

For long 30-app runs, the recommended background launch is:

```bash
LOG_ROOT=tmp/model_app_runs/formal_30apps_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
nohup /bin/bash -lc '
  source /data2/zzx/data/miniconda3/etc/profile.d/conda.sh &&
  conda activate bitfly &&
  cd /data2/zzx/data/workspace/bitfly &&
  scripts/benchmarks/run_model_split_apps.sh     --mode run     --no-rebuild-apps     --no-verilate     --parallel 5     --batch-size 5     --log-root '"$LOG_ROOT"'
' > "$LOG_ROOT/launch.log" 2>&1 &
```

Run a single smoke-check app:

```bash
scripts/benchmarks/run_model_split_apps.sh \
  --mode run \
  --apps bmpmm_INT2_gemma3_270m \
  --parallel 1 \
  --batch-size 1 \
  --log-root tmp/model_app_runs/smoke_bmpmm_INT2_gemma3_270m
```

## Outputs

Each run generates:

- `apps.txt`: selected app list
- `runner.log`: batch-level execution log
- `summary.csv`: pass/fail and runtime summary
- `batch_XX/<app>.log`: detailed per-app simulator output

## Plotting for Paper

A practical paper-friendly presentation is:

- one subplot per precision
- x-axis: models
- y-axis: total runtime over all extracted linear GEMM shapes of one layer
- two bars/lines per model: BMPMM vs RVV

This keeps the comparison directly aligned with the deployment target and avoids exploding the number of figures.

## Log Interpretation

When using `simv`, each per-app log usually begins with ELF loading and `Program header...` messages before the benchmark app itself starts printing. This startup phase is normal.

Recommended checks:

- `tail -f tmp/model_app_runs/<run>/runner.log`
- `tail -f tmp/model_app_runs/<run>/batch_00/<app>.log`

If only loader messages are visible, the simulation is usually still in the startup phase rather than hung.
