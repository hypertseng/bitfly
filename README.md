# QLoMA

QLoMA extends the Ara vector coprocessor stack with custom BMPMM-style instructions, RTL support, and benchmarking flows for LLM-oriented mixed-precision GEMM evaluation.

## Repository Layout

- `ara/`: primary Ara-based hardware, apps, toolchain, and simulation flow
- `src/`: QLoMA-specific source overlays that are synced into `ara/`
- `scripts/`: categorized automation, debug, and analysis utilities
- `docs/`: experiment workflow and repository guides
- `tmp/`: generated logs and benchmark outputs, ignored by Git
- `build/`: local temporary build directories, ignored by Git

Detailed guides:

- `docs/repo_guide.md`
- `docs/benchmark_workflow.md`
- `scripts/README.md`
- `ara/README.md`
- `ara/apps/README.md`
- `ara/config/README.md`

## Environment

Recommended host setup:

- `git` with submodules
- `make`, `gcc/g++`, `python3`
- `Verilator`
- optional: `QuestaSim`, `gtkwave`
- Conda environment `bitfly`

Ara-specific prerequisites are documented in `ara/DEPENDENCIES.md`.

## Initial Setup

Clone and initialize submodules:

```bash
git clone --recursive <repo-url> QLoMA
cd QLoMA
git submodule update --init --recursive
```

Sync the QLoMA LLVM/custom-instruction overlays into Ara:

```bash
cp -f src/instr/RISCVAsmParser.cpp ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/AsmParser/
cp -f src/instr/RISCVDisassembler.cpp ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/Disassembler/
cp -f src/instr/RISCVInstPrinter.cpp ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/MCTargetDesc/
cp -f src/instr/RISCVInstPrinter.h ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/MCTargetDesc/
cp -f src/instr/RISCVInstrInfoCustom.td ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/
cp -f src/instr/RISCVInstrInfo.td ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/
cp -f src/instr/RISCVMCCodeEmitter.cpp ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/MCTargetDesc/
```

Build the software side if needed:

```bash
cd src
sh compile.sh
```

## Ara Build and Simulation

Typical Ara hardware setup:

```bash
cd ara/hardware
make checkout
make apply-patches
make verilate
make simv app=hello_world
```

Console / Questa-style flow:

```bash
cd ara/hardware
make compile
make sim app=hello_world
make simc app=hello_world
```

## Script Layout

The top-level script tree is now organized by use case:

- `scripts/benchmarks/`: batch benchmark runners and shape utilities
- `scripts/analysis/`: roofline and design-space analysis
- `scripts/debug/`: module-level debug launchers
- `scripts/dev/`: source sync and maintenance helpers

Backward-compatible wrapper paths are still available under `scripts/`.

## Model-Split Benchmark Apps

The benchmark flow is organized as one app per:

- model
- precision (`binary`, `INT2`, `INT4`)
- implementation (`bmpmm`, `rvv`)

Current model tags:

- `gemma3_270m`
- `qwen25_05b`
- `opt_13b`
- `qwen25_15b`
- `gemma2_2b`

Example app names:

- `bmpmm_binary_gemma3_270m`
- `bmpmm_INT2_opt_13b`
- `rvv_INT4_gemma2_2b`

This split keeps each generated `data.S` small enough to compile and simulate efficiently.

## Batch Runner

Primary entry:

```bash
scripts/benchmarks/run_model_split_apps.sh --help
```

Legacy-compatible entry:

```bash
scripts/run_model_split_apps.sh --help
```

Common examples:

```bash
# Build and run all selected apps
scripts/benchmarks/run_model_split_apps.sh --mode all --build-jobs 16 --parallel 5 --batch-size 5

# Reuse existing binaries and Verilator build, only run apps
scripts/benchmarks/run_model_split_apps.sh --mode run --no-rebuild-apps --no-verilate --parallel 5 --batch-size 5

# Run only one app
scripts/benchmarks/run_model_split_apps.sh \
  --mode run \
  --apps bmpmm_INT2_gemma3_270m \
  --parallel 1 \
  --batch-size 1 \
  --log-root tmp/model_app_runs/single_check_int2

# Run only one precision across all models and both implementations
scripts/benchmarks/run_model_split_apps.sh --mode all --precisions INT4 --parallel 5 --batch-size 5
```

Key options:

- `--mode <all|build|run>`
- `--build-jobs <N>`
- `--parallel <N>`
- `--batch-size <N>`
- `--models <csv>`
- `--precisions <csv>`
- `--impls <csv>`
- `--apps <csv>`
- `--log-root <dir>`
- `--no-verilate`
- `--no-rebuild-apps`
- `--trace`
- `--extra-make-args <string>`

Outputs are written under `tmp/model_app_runs/<run_name>/`:

- `apps.txt`
- `runner.log`
- `summary.csv`
- `batch_XX/<app>.log`

## Repo Hygiene

This repository intentionally ignores generated files such as:

- benchmark logs under `tmp/`
- local build products under `build/`
- generated `data.S` and Spike objects under `ara/apps/`
- Python cache directories
- local editor settings

For experiment presentation guidance, see `docs/benchmark_workflow.md`.
