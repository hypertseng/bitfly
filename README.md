# bitfly

`bitfly` is a research workspace built on top of Ara for studying low-bit GEMM execution. It combines:

- a proposed BMPU / BMPMM-oriented hardware and software path
- an RVV baseline path for comparison
- custom LLVM instruction support
- app overlays, generators, and batch scripts for reproducible evaluation

At a high level, the repository answers one architecture question:

> For the same model-derived GEMM workload, how does the proposed BMPMM path compare with the RVV baseline under a shared Ara-based execution stack?

## System View

The repository is intentionally split into a persistent overlay tree and a working build tree:

```text
src/                    bitfly source of truth
  -> scripts/dev/sync_src_to_ara.sh
ara/                    working Ara tree used for build and simulation
  -> ara/apps + ara/hardware
tmp/model_app_runs/     generated experiment logs and summaries
```

Interpret the main directories as follows:

| Path | Type | Purpose |
| --- | --- | --- |
| `src/` | source of truth | Project-owned overlays for apps, RTL, and LLVM changes |
| `ara/` | working tree | Primary build and simulation workspace |
| `scripts/` | tooling | Benchmark, analysis, debug, and sync automation |
| `docs/` | methodology | Workflow notes and experiment interpretation |
| `patches/` | review artifact | Exported diffs for synced Ara-side changes |
| `tmp/` | generated output | Run logs, summaries, and temporary experiment artifacts |
| `build/` | generated output | Local temporary testbench or compile products |

If you are new to the repository, read in this order:

1. [`src/README.md`](src/README.md)
2. [`src/apps/README.md`](src/apps/README.md)
3. [`scripts/README.md`](scripts/README.md)
4. [`docs/benchmark_workflow.md`](docs/benchmark_workflow.md)

## Task-Oriented Entry Points

Use the repository by task, not by directory guessing:

| Task | Start Here | Why |
| --- | --- | --- |
| Run the fastest correctness check | `make -C ara/hardware simv app=bmpu_verify` | Validates BMPU packing and mixed-precision correctness quickly |
| Run the paper-style benchmark matrix | `scripts/benchmarks/run_model_split_apps.sh` | Builds and runs one app per model / precision / implementation |
| Change persistent benchmark logic | [`src/apps/README.md`](src/apps/README.md) | App overlays live under `src/apps/` |
| Change persistent RTL | [`src/hardware/README.md`](src/hardware/README.md) | RTL overlays live under `src/hardware/` |
| Change custom instruction support | [`src/llvm_instr/README.md`](src/llvm_instr/README.md) | LLVM-side custom instruction support is isolated there |
| Sync overlays into the build tree | `scripts/dev/sync_src_to_ara.sh` | Keeps `src/` and `ara/` roles clean |
| Interpret run outputs | [`docs/benchmark_workflow.md`](docs/benchmark_workflow.md) | Defines output files and log-reading rules |

## Research Artifact Contract

For architecture-style evaluation, the repository treats the following as the core experimental unit:

- one app
- one implementation: `bmpmm` or `rvv`
- one precision: `binary`, `INT2`, or `INT4`
- one model-derived workload slice

This contract is implemented as the model-split app matrix under `src/apps/` and executed by `scripts/benchmarks/run_model_split_apps.sh`.

The intended comparison is:

- proposed path: `bmpmm_*`
- baseline path: `rvv_*`

under:

- the same model-derived shape set
- the same Ara-based simulator flow
- the same hardware build configuration unless intentionally changed

## Reproducibility Quick Start

Clone with submodules:

```bash
git clone --recursive <repo-url> bitfly
cd bitfly
git submodule update --init --recursive
```

Recommended host tools:

- `git`
- `make`
- `gcc` / `g++`
- `python3`
- `rsync`
- `Verilator`
- optional: `QuestaSim`, `gtkwave`

Ara-specific prerequisites are documented in [`ara/DEPENDENCIES.md`](ara/DEPENDENCIES.md).

Sync the bitfly overlays into Ara:

```bash
scripts/dev/sync_src_to_ara.sh
```

Build hardware and one correctness app:

```bash
make -C ara/hardware verilate -j8
make -C ara/apps bin/bmpu_verify -j8
make -C ara/hardware simv app=bmpu_verify
```

A healthy correctness run ends with `ALL CASES PASSED`.

## Benchmark Workflow Summary

The main batch runner is:

```bash
scripts/benchmarks/run_model_split_apps.sh --help
```

Typical commands:

```bash
scripts/benchmarks/run_model_split_apps.sh --mode all --build-jobs 16 --parallel 5 --batch-size 5
scripts/benchmarks/run_model_split_apps.sh --mode run --no-rebuild-apps --no-verilate --parallel 5 --batch-size 5
scripts/benchmarks/run_model_split_apps.sh --mode run --apps bmpmm_INT2_gemma3_270m --parallel 1 --batch-size 1
```

Each run writes:

- `apps.txt`: selected app list
- `runner.log`: batch-level progress log
- `summary.csv`: app-level pass/fail and runtime summary
- `batch_XX/<app>.log`: per-app simulator log

For the full methodology and log interpretation rules, see [`docs/benchmark_workflow.md`](docs/benchmark_workflow.md).

## Directory Semantics

Use these rules consistently:

- edit `src/` when the change should be preserved as bitfly-owned logic
- edit `ara/` when the change is a local working-tree experiment or an upstream Ara concern
- treat `tmp/` and `build/` as disposable outputs
- treat large local model binaries as runtime assets, not source files

The maintained sync path is:

```bash
scripts/dev/sync_src_to_ara.sh
```

That script:

- syncs app overlays from `src/apps/` into `ara/apps/`
- syncs RTL overlays from `src/hardware/` into `ara/hardware/`
- optionally syncs `src/llvm_instr/` into an LLVM checkout
- emits a patch under `patches/local/` when patch generation is enabled

## Documentation Index

- [`src/README.md`](src/README.md): source-of-truth policy and edit boundaries
- [`src/apps/README.md`](src/apps/README.md): app taxonomy and experiment units
- [`src/hardware/README.md`](src/hardware/README.md): RTL overlay organization
- [`src/llvm_instr/README.md`](src/llvm_instr/README.md): LLVM custom instruction support
- [`scripts/README.md`](scripts/README.md): command index and tool entry points
- [`docs/README.md`](docs/README.md): higher-level workflow notes
- [`docs/benchmark_workflow.md`](docs/benchmark_workflow.md): benchmark methodology and result interpretation
- [`tcl/README.md`](tcl/README.md): synthesis collateral

## Hygiene

The repository intentionally excludes or de-emphasizes:

- generated logs under `tmp/`
- temporary local builds under `build/`
- transient Ara-side build products
- editor caches and Python cache directories

That separation is deliberate: the tracked repository should describe the system and the workflow, while measurements and generated outputs live in dedicated run directories.
