# Scripts

The `scripts/` tree is the command surface of BitFly. It is organized by workflow rather than by language or implementation detail.

## Start Here By Task

| Task | Primary Entry Point | Main Output |
| --- | --- | --- |
| sync maintained overlays into Ara | `scripts/dev/sync_src_to_ara.sh` | updated `ara/` tree and optional patch artifact |
| run one smoke-check benchmark app | `scripts/benchmarks/run_model_split_apps.sh --mode run --apps <app> --parallel 1 --batch-size 1` | one benchmark run directory |
| run the main model-split matrix | `scripts/benchmarks/run_model_split_apps.sh --mode all --build-jobs 16 --parallel 5 --batch-size 5` | benchmark campaign under `tmp/model_app_runs/` |
| regenerate search or roofline outputs | `python3 scripts/analysis/roofline.py` | plots and derived metrics |
| debug SA behavior | `scripts/debug/run_sa_tb.sh` | focused SA testbench output |
| debug LBMAC behavior | `scripts/debug/run_lbmac_tb.sh` | focused LBMAC testbench output |

## Directory Map

| Directory | Purpose |
| --- | --- |
| [`analysis/`](analysis/README.md) | tiling search, roofline analysis, and post-processing |
| [`benchmarks/`](benchmarks/README.md) | benchmark campaign launchers |
| [`debug/`](debug/README.md) | focused debug launchers for hardware blocks |
| [`dev/`](dev/README.md) | sync and repository maintenance helpers |

## Workflow Policy

Use `scripts/` as the repository command surface, not as a dumping ground for unrelated helpers.

Prefer:

- `scripts/dev/` for sync and maintenance operations
- `scripts/benchmarks/` for experiment launchers
- `scripts/analysis/` for post-processing and figure generation
- `scripts/debug/` for shortest-loop hardware debug

If a script does not clearly fit one of those workflows, it usually needs either a better home or better documentation.

## Command Notes

### Sync

```bash
scripts/dev/sync_src_to_ara.sh
```

Use this after changing maintained overlays under `src/`.

### Benchmark campaign

```bash
scripts/benchmarks/run_model_split_apps.sh --help
```

This is the main entry point for paper-style benchmark campaigns.

### Analysis

```bash
python3 scripts/analysis/roofline.py
```

Use this to regenerate roofline-style figures from the current search and configuration outputs.

## Compatibility Wrappers

Historical wrapper paths remain available so older shell history continues to work:

- `scripts/run_model_split_apps.sh`
- `scripts/run_lbmac_tb.sh`
- `scripts/run_sa_tb.sh`
- `scripts/sync_src_to_ara.sh`

For new documentation, new automation, and paper references, prefer the categorized paths under `scripts/`.

## Output Discipline

Runner and analysis scripts should write outputs into explicit run or output directories rather than beside maintained source files. The normal homes are:

- `tmp/model_app_runs/` for benchmark campaigns
- dedicated figure/output directories for analysis products
- `patches/local/` for local exported sync patches
