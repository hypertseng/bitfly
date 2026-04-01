# BitFly

BitFly is a research codebase for studying low-bit LLM GEMM execution on top of the Ara RISC-V vector architecture. The repository contains:

- a proposed BMPU/BMPMM execution path for low-bit mixed-precision GEMM
- an RVV baseline path under the same Ara-based software and simulation stack
- application overlays derived from model-layer GEMM workloads
- analysis and automation scripts for correctness, benchmarking, tiling search, and roofline-style post-processing

The central evaluation question is:

> Under the same model-derived workload slices and the same Ara-based execution stack, how does the proposed BMPMM path compare against an RVV baseline?

## Highlights

- Persistent project-owned sources live under `src/`; the synced Ara working tree lives under `ara/`.
- Benchmark apps are organized as a model-split matrix: `bmpmm_*` versus `rvv_*`, across `binary`, `INT2`, and `INT4`.
- The repository includes both fast correctness checks such as `bmpu_verify` and paper-oriented benchmark runners.
- Tiling search and roofline analysis are implemented in `scripts/analysis/` and operate on the same execution assumptions used by the software template and RTL.

## Repository Status

This repository is best understood as a research artifact workspace rather than a polished end-user package manager distribution. The maintained workflow is:

```text
src/  ->  sync into ara/  ->  build and simulate  ->  logs and plots under tmp/ or repository outputs
```

If you want to preserve a change as part of BitFly, edit `src/` first and treat `ara/` as the working build tree.

## Quick Start

### 1. Clone the repository

```bash
git clone --recursive git@github.com:hypertseng/bitfly.git
cd bitfly
git submodule update --init --recursive
```

### 2. Check host dependencies

Recommended host tools:

- `git`
- `make`
- `python3`
- `gcc` / `g++`
- `rsync`
- `Verilator`
- optional: `QuestaSim`, `gtkwave`

Ara-specific prerequisites are documented in [`ara/DEPENDENCIES.md`](ara/DEPENDENCIES.md).

### 3. Sync BitFly overlays into Ara

```bash
scripts/dev/sync_src_to_ara.sh
```

### 4. Run the fastest correctness check

```bash
make -C ara/hardware verilate -j8
make -C ara/apps bin/bmpu_verify -j8
make -C ara/hardware simv app=bmpu_verify
```

A healthy run ends with `ALL CASES PASSED`.

### 5. Run a benchmark app or campaign

One smoke-check benchmark app:

```bash
scripts/benchmarks/run_model_split_apps.sh \
  --mode run \
  --apps bmpmm_INT2_gemma3_270m \
  --parallel 1 \
  --batch-size 1
```

Paper-style benchmark matrix:

```bash
scripts/benchmarks/run_model_split_apps.sh \
  --mode all \
  --build-jobs 16 \
  --parallel 5 \
  --batch-size 5
```

### 6. Regenerate roofline analysis

```bash
python3 scripts/analysis/roofline.py
```

This produces:

- `roofline_search_results.png`
- `roofline_search_results.pdf`

## How To Read The Repository

### Top-level layout

| Path | Role | Notes |
| --- | --- | --- |
| `src/` | BitFly source of truth | Project-owned overlays for apps, RTL, and LLVM-side instruction support |
| `ara/` | Working Ara tree | Main build and simulation workspace |
| `scripts/` | Automation entry points | Benchmark runners, sync scripts, analysis, and debug helpers |
| `docs/` | Reproducibility and workflow notes | Start here for artifact-style documentation |
| `tmp/` | Generated outputs | Logs, CSV summaries, and intermediate artifacts |
| `build/` | Local build outputs | Disposable products |
| `patches/` | Sync-side review artifacts | Optional exported diffs from sync workflows |

### Recommended reading order

1. [`docs/artifact_quickstart.md`](docs/artifact_quickstart.md)
2. [`docs/artifact_checklist.md`](docs/artifact_checklist.md)
3. [`docs/repo_structure.md`](docs/repo_structure.md)
4. [`src/README.md`](src/README.md)
5. [`src/apps/README.md`](src/apps/README.md)
6. [`scripts/README.md`](scripts/README.md)
7. [`docs/benchmark_workflow.md`](docs/benchmark_workflow.md)

## Experimental Contract

The main benchmark matrix treats each app as one workload slice:

```text
<implementation>_<precision>_<model>
```

Examples:

- `bmpmm_binary_gemma3_270m`
- `bmpmm_INT2_qwen25_15b`
- `rvv_INT4_opt_13b`

The intended comparison is always:

- proposed path: `bmpmm_*`
- baseline path: `rvv_*`

under:

- the same model-derived GEMM shape set
- the same precision
- the same Ara-based build and simulation stack unless intentionally changed

`bmpu_verify` is a correctness regression app, not a paper-performance data point.

## Reproducing Main Results

For artifact-quality runs, keep the following together:

- the exact launcher command
- the run directory under `tmp/model_app_runs/<run>/`
- `apps.txt`
- `runner.log`
- `summary.csv`
- per-app logs under `batch_XX/`

For a complete workflow, including runner options and log interpretation, see [`docs/benchmark_workflow.md`](docs/benchmark_workflow.md).

## Development Conventions

### `src/` versus `ara/`

- Edit `src/` when the change should be maintained by BitFly.
- Edit `ara/` only for local working-tree experiments or upstream Ara concerns.
- Use `scripts/dev/sync_src_to_ara.sh` to propagate maintained overlays into `ara/`.

### Generated files

Treat these as generated or runtime artifacts, not primary source:

- `tmp/`
- `build/`
- simulator logs
- large local model binaries
- generated figures and CSV summaries unless explicitly versioned for a reason

### Bench and analysis outputs

The repository intentionally separates source from outputs. Measurements should live in dedicated run directories rather than being mixed with maintained source overlays.

## Documentation Map

- [`docs/README.md`](docs/README.md): documentation index
- [`docs/artifact_quickstart.md`](docs/artifact_quickstart.md): fastest path from clone to benchmark and plot
- [`docs/artifact_checklist.md`](docs/artifact_checklist.md): artifact and repository release checklist
- [`docs/benchmark_workflow.md`](docs/benchmark_workflow.md): benchmark methodology and output contract
- [`docs/repo_structure.md`](docs/repo_structure.md): directory map and file placement guide
- [`docs/repo_guide.md`](docs/repo_guide.md): ownership boundaries and repository mental model
- [`src/README.md`](src/README.md): source-of-truth policy and overlay map
- [`src/apps/README.md`](src/apps/README.md): benchmark app taxonomy
- [`src/hardware/README.md`](src/hardware/README.md): RTL overlay organization
- [`src/llvm_instr/README.md`](src/llvm_instr/README.md): custom instruction support
- [`scripts/README.md`](scripts/README.md): command-oriented entry points

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for repository conventions, patch boundaries, and pre-PR checks.

## Citation

If BitFly is used in academic work, cite the repository and the associated paper when available. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

This repository is released under the [`MIT License`](LICENSE). The `ara/` submodule and other vendored dependencies keep their own upstream licenses.
