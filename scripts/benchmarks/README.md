# Benchmark Scripts

`scripts/benchmarks/` is the main automation entry point for model-split benchmark campaigns.

## Primary Entry Point

```bash
scripts/benchmarks/run_model_split_apps.sh --help
```

This is the command surface used for most paper-style benchmark runs.

## Key Files

| File | Purpose |
| --- | --- |
| `run_model_split_apps.sh` | batch build-and-run launcher for the benchmark app matrix |
| `extract_gemm_shapes.py` | helper to derive GEMM shapes from model descriptions or traces |
| `models.txt` | default model list used by the workflow |

## Common Examples

Build and run the selected matrix:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode all --build-jobs 16 --parallel 5 --batch-size 5
```

Reuse existing builds and run simulations only:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode run --no-rebuild-apps --no-verilate --parallel 5 --batch-size 5
```

Run one app as a smoke test:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode run --apps bmpmm_INT2_gemma3_270m --parallel 1 --batch-size 1
```

## Output Contract

Each run writes into `tmp/model_app_runs/<run_name>/`:

- `apps.txt`
- `runner.log`
- `summary.csv`
- `batch_XX/<app>.log`

For the benchmark contract and output interpretation, see:

- [`../../docs/benchmark_workflow.md`](../../docs/benchmark_workflow.md)
- [`../../src/apps/README.md`](../../src/apps/README.md)
