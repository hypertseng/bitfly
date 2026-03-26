# Benchmark Scripts

`scripts/benchmarks/` is the main automation entry for model-split benchmark campaigns.

## Files

- `run_model_split_apps.sh`: batch builder / runner for the app matrix
- `extract_gemm_shapes.py`: extracts GEMM shapes from model descriptions or traces
- `models.txt`: default model list used by the workflow

## Main Command

```bash
scripts/benchmarks/run_model_split_apps.sh --help
```

## Common Examples

Build and run the full selected matrix:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode all --build-jobs 16 --parallel 5 --batch-size 5
```

Reuse existing builds and only run simulations:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode run --no-rebuild-apps --no-verilate --parallel 5 --batch-size 5
```

Run one app as a smoke test:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode run --apps bmpmm_INT2_gemma3_270m --parallel 1 --batch-size 1
```

## Outputs

Each run writes into `tmp/model_app_runs/<run_name>/`:

- `apps.txt`
- `runner.log`
- `summary.csv`
- `batch_XX/<app>.log`

For app naming rules, see [`../../src/apps/README.md`](../../src/apps/README.md).
