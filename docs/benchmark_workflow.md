# Benchmark Workflow

This document defines the recommended methodology for the model-split LLM GEMM experiments used for formal evaluation.

## Research Question

The benchmark flow is designed to compare:

- the proposed BMPMM-based execution path
- the RVV baseline path

for the same model-derived GEMM workload slices under a shared Ara-based software and simulation stack.

## Experimental Contract

Each benchmark app corresponds to one point in the workload matrix:

- one implementation
- one precision
- one model

For each selected model, the standard six-app matrix is:

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

## Fair-Comparison Rules

When preparing paper-quality results, keep these controls fixed unless the experiment explicitly studies them:

- use the same model-derived shape set for `bmpmm_*` and `rvv_*`
- use the same simulator flow and hardware build configuration
- compare the same precision on both sides
- record outputs from the same run tree under `tmp/model_app_runs/<run>/`

In other words, the implementation should be the main variable; the rest of the stack should remain stable.

## Reproducibility Checklist

Before launching a campaign:

1. Sync persistent overlays from `src/` into `ara/`.
2. Confirm the intended app set and model list.
3. Decide whether hardware must be rebuilt or whether an existing Verilator build is valid.
4. Pick a fresh `log-root` so the run is self-contained.
5. Keep the exact command line used for the final reported run.

## Standard Execution Flow

Build and run the selected matrix:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode all --build-jobs 16 --parallel 5 --batch-size 5
```

Reuse existing builds and rerun simulations only:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode run --no-rebuild-apps --no-verilate --parallel 5 --batch-size 5
```

Run one smoke-check app:

```bash
scripts/benchmarks/run_model_split_apps.sh \
  --mode run \
  --apps bmpmm_INT2_gemma3_270m \
  --parallel 1 \
  --batch-size 1 \
  --log-root tmp/model_app_runs/smoke_bmpmm_INT2_gemma3_270m
```

For a long background campaign:

```bash
LOG_ROOT=tmp/model_app_runs/formal_30apps_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
nohup /bin/bash -lc '
  source /data2/zzx/data/miniconda3/etc/profile.d/conda.sh &&
  conda activate bitfly &&
  cd /data2/zzx/data/workspace/bitfly &&
  scripts/benchmarks/run_model_split_apps.sh \
    --mode run \
    --no-rebuild-apps \
    --no-verilate \
    --parallel 5 \
    --batch-size 5 \
    --log-root '"$LOG_ROOT"'
' > "$LOG_ROOT/launch.log" 2>&1 &
```

## Important Runner Knobs

The most important options are:

| Option | Meaning |
| --- | --- |
| `--mode <all|build|run>` | full flow, build only, or run only |
| `--build-jobs <N>` | parallelism for app and hardware build |
| `--parallel <N>` | concurrent simulator jobs |
| `--batch-size <N>` | number of apps grouped into one batch directory |
| `--models <csv>` | selected model subset |
| `--precisions <csv>` | selected precision subset |
| `--impls <csv>` | selected implementation subset |
| `--apps <csv>` | explicit app selection, bypassing matrix filters |
| `--log-root <dir>` | output run directory |
| `--no-verilate` | reuse an existing Verilator build |
| `--no-rebuild-apps` | reuse existing app binaries |

## Output Contract

Each run directory contains:

- `apps.txt`: exact app list for the run
- `runner.log`: batch-level orchestration log
- `summary.csv`: per-app status and runtime summary
- `batch_XX/<app>.log`: detailed simulator log for one app

`summary.csv` has the following columns:

- `app`: app name
- `status`: `PASS` or `FAIL`
- `duration`: wall-clock runtime in seconds as recorded by the runner
- `logfile`: path to the detailed per-app log

These files should be preserved together when producing final plots or tables.

## Log Interpretation

When using `simv`, each per-app log usually begins with ELF loading and `Program header...` messages before the benchmark app itself starts printing. This startup phase is normal.

Recommended live checks:

- `tail -f tmp/model_app_runs/<run>/runner.log`
- `tail -f tmp/model_app_runs/<run>/batch_00/<app>.log`

Interpretation rules:

- only loader messages early in the log usually means the simulator is still starting
- `FAIL` in `summary.csv` means the underlying make / simulator command returned a non-zero code
- `runner.log` is the first place to inspect global progress
- per-app logs are the authoritative source for app-specific failure details

## Failure Triage

Use this triage sequence:

1. check `summary.csv` to identify the failing app
2. inspect `runner.log` for the batch context and failure tail
3. open the corresponding `batch_XX/<app>.log`
4. decide whether the failure is:
   - build/configuration related
   - simulator/runtime related
   - application-logic related

If the issue looks like correctness rather than performance, rerun `bmpu_verify` before debugging the full benchmark matrix.

## Plotting For Paper

A paper-friendly presentation is:

- one subplot per precision
- x-axis: model
- y-axis: total runtime aggregated across the selected linear-layer GEMM shapes
- two series per model: BMPMM vs RVV

This presentation aligns with the experimental contract above:

- same workload slice
- same precision
- different implementation

and avoids mixing unrelated dimensions into one figure.

## What To Report

For final artifact-quality reporting, keep:

- the exact launcher command
- the exact app list in `apps.txt`
- the full `summary.csv`
- the run directory timestamp or unique name

This makes the experiment auditable and rerunnable, which is the standard expected for architecture artifacts.

For a final pre-submission sanity pass, use [`artifact_checklist.md`](artifact_checklist.md).
