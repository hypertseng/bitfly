# Scripts

The `scripts/` tree is organized by workflow rather than by language or file type. Treat it as the command surface of the repository.

## Command Index

| Command / Entry | Use When | Main Output / Effect | First Thing To Inspect |
| --- | --- | --- | --- |
| `scripts/dev/sync_src_to_ara.sh` | you changed persistent overlays under `src/` | updated `ara/` tree and optional patch under `patches/local/` | Ara-side `git status` or generated patch |
| `scripts/benchmarks/run_model_split_apps.sh` | you want the main benchmark campaign | logs under `tmp/model_app_runs/<run>/` | `runner.log`, then per-app logs |
| `scripts/analysis/roofline.py` | you want post-processing or plots | analysis figures or derived metrics | input dataset and plotting arguments |
| `scripts/debug/run_sa_tb.sh` | you are debugging systolic-array behavior | focused SA testbench run | SA testbench output |
| `scripts/debug/run_lbmac_tb.sh` | you are debugging LBMAC behavior | focused LBMAC testbench run | LBMAC testbench output |

## Directory Map

- [`analysis/README.md`](analysis/README.md): post-processing, search, and plotting helpers
- [`benchmarks/README.md`](benchmarks/README.md): app-matrix execution
- [`debug/README.md`](debug/README.md): focused RTL debug launchers
- [`dev/README.md`](dev/README.md): sync and maintenance helpers

## Recommended Entry Points By Task

| Task | Command |
| --- | --- |
| sync persistent changes into Ara | `scripts/dev/sync_src_to_ara.sh` |
| run one smoke-check benchmark app | `scripts/benchmarks/run_model_split_apps.sh --mode run --apps <app> --parallel 1 --batch-size 1` |
| run the main model-split matrix | `scripts/benchmarks/run_model_split_apps.sh --mode all --build-jobs 16 --parallel 5 --batch-size 5` |
| inspect roofline-style behavior | `scripts/analysis/roofline.py` |
| isolate an SA or LBMAC bug | `scripts/debug/run_sa_tb.sh` or `scripts/debug/run_lbmac_tb.sh` |

## Compatibility Wrappers

Historical wrapper paths remain available so existing shell history continues to work:

- `scripts/run_model_split_apps.sh`
- `scripts/run_lbmac_tb.sh`
- `scripts/run_sa_tb.sh`
- `scripts/sync_src_to_ara.sh`

Use the categorized paths for new documentation, automation, and papers.
