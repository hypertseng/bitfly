# QLoMA

**QLoMA** is an academic research project built on top of the Ara vector coprocessor for the CVA6 core. It extends the open‑source Ara infrastructure with custom RISC‑V instructions, a modified LLVM backend, and an RTL design tuned for large language model accelerators. The repository contains both the software toolchain and hardware descriptions necessary to build, simulate and deploy the system.

> 📌 This README is intended to provide a comprehensive walkthrough for newcomers and reviewers of top‑tier conferences. For more details about Ara itself consult `ara/README.md` and the sub‑directory documentation.

---

## ✅ Dependencies

Before you begin, make sure the following tools are installed on your host machine:

- **GNU make**, **gcc/g++ (≥7.2)** – used to build the toolchain and simulators.
- **Git** (with submodule support)
- **Python 3** (some auxiliary scripts in `scripts/`)
- **Verilator** (≥4.0) for RTL simulation
- **Spike** RISC‑V ISA simulator (built from sources below)
- **Questasim** (optional, for waveform‑accurate simulation and verification)
- **gtkwave** (optional, for inspecting `.fst` waveforms)

Additional dependencies required by Ara are listed in `ara/DEPENDENCIES.md`.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone --recursive <repo-url> QLoMA
cd QLoMA
```

If you already have a clone without submodules:

```bash
git submodule update --init --recursive
```

To sync submodule URLs after a remote change:

```bash
git submodule sync --recursive
```

### 2. Copy custom instruction sources

The project modifies the Ara LLVM backend; before building, the custom files in `src/instr/` must overwrite the stock files in the Ara toolchain:

```bash
cp -f src/instr/RISCVAsmParser.cpp ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/AsmParser/
cp -f src/instr/RISCVDisassembler.cpp ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/Disassembler/
cp -f src/instr/RISCVInstPrinter.cpp ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/MCTargetDesc/
cp -f src/instr/RISCVInstPrinter.h ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/MCTargetDesc/
cp -f src/instr/RISCVInstrInfoCustom.td ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/
cp -f src/instr/RISCVInstrInfo.td ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/
cp -f src/instr/RISCVMCCodeEmitter.cpp ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/MCTargetDesc/
```

### 3. Build the QLoMA software components

```bash
cd src
sh compile.sh
```

This script invokes the modified LLVM/Clang, assembles the custom ISA support, and produces the user binaries located under `ara/apps/bin`.

---

## 🔧 Toolchain Setup

The Ara project comes with helper Makefile targets for building a compatible RISC‑V toolchain and simulator.

From the top level of the workspace run:

```bash
# Build LLVM with vector‑extension support
make toolchain-llvm

# Build the Spike ISA simulator (required for software emulation)
# use static linking if you encounter library issues
make riscv-isa-sim LDFLAGS="-static-libstdc++"

# Build Verilator (used by the hardware flow)
make verilator
```

> ⚠️ Older GCC versions (7.2.0) are known to work reliably when compiling Spike.

---

## 🧩 Configuration

Ara’s parameters live in the `ara/config/` tree. See `ara/config/README.md` for a complete explanation of available configurations (number of lanes, memory sizes, etc.).

You can select a configuration by prefixing your `make` command with:

```bash
export ARA_CONFIGURATION=chosen_config
# or
make config=chosen_config <target>
```

---

## 🛠 Building and Running Applications

All example programs and benchmarks reside in `ara/apps`.

```bash
cd ara/apps
# compile the hello_world example
make bin/hello_world
```

---

## 🖥 RTL Simulation & Verification

### Hardware dependencies

Ara uses [Bender](https://github.com/lowRISC/bender) to manage third‑party IPs. To install dependencies:

```bash
cd ara/hardware
make checkout          # fetch all IPs
```

If IPs are re‑checked out, re‑apply the following patches once:

```bash
make apply-patches     # only needed once per checkout
```


The typical simulation command from the hardware directory:

```bash
cd ara/hardware
make compile            # synthesize design for ModelSim
make sim app=hello_world # run simulation with hello_world loaded
make simc                # run in console (no GUI)
```

### Verilator Flow

```bash
cd ara/hardware
make apply-patches       # once
make verilate            # build C++ model
make simv app=hello_world  # run Verilator model
make riscv_tests_simv     # run the unit-testbench
```

Trace files in `fst` format are produced by adding `trace=1` to any of the above targets. Open them with `gtkwave`.

### Model-Split Benchmark Runner

For LLM GEMM benchmarking, this repository now supports a **model-split app layout**: each app contains only the linear-layer GEMM shapes of **one model** and **one backend**. This keeps generated `data.S` files much smaller and reduces compile time significantly.

Current benchmark app naming follows:

- `bmpmm_binary_<model>`
- `rvv_binary_<model>`
- `bmpmm_INT2_<model>`
- `rvv_INT2_<model>`
- `bmpmm_INT4_<model>`
- `rvv_INT4_<model>`

where `<model>` is one of:

- `gemma3_270m`
- `qwen25_05b`
- `opt_13b`
- `qwen25_15b`
- `gemma2_2b`

A helper script is provided to build and run these apps concurrently or in batches:

```bash
scripts/run_model_split_apps.sh --help
```

Typical usage:

```bash
# Build all selected apps, build the Verilator model, then run them
scripts/run_model_split_apps.sh --mode all --build-jobs 16 --parallel 8 --batch-size 8

# Only run a subset of models
scripts/run_model_split_apps.sh --mode run --models gemma3_270m,opt_13b --parallel 6

# Only run one precision across all models
scripts/run_model_split_apps.sh --mode all --precisions INT2 --parallel 10

# Only run BMPMM implementations
scripts/run_model_split_apps.sh --mode run --impls bmpmm --parallel 10

# Run explicit app names
scripts/run_model_split_apps.sh \
  --apps bmpmm_INT2_gemma3_270m,rvv_INT2_gemma3_270m \
  --parallel 2
```

Supported options:

- `--mode <all|build|run>`: build only, run only, or do both
- `--build-jobs <N>`: parallelism for `make -C ara/apps` and `make -C ara/hardware verilate`
- `--parallel <N>`: number of concurrent `make simv` jobs per batch
- `--batch-size <N>`: split the selected app list into batches
- `--models <csv>`: choose a subset of model tags
- `--precisions <csv>`: choose from `binary,INT2,INT4`
- `--impls <csv>`: choose from `bmpmm,rvv`
- `--apps <csv>`: explicit app list, overriding model/precision/impl filters
- `--no-verilate`: skip rebuilding the Verilator model
- `--trace`: run `simv` with `trace=1`

Logs and summaries are written to:

- `tmp/model_app_runs/<timestamp>/apps.txt`: selected app list
- `tmp/model_app_runs/<timestamp>/runner.log`: runner progress log
- `tmp/model_app_runs/<timestamp>/summary.csv`: pass/fail summary with timestamps
- `tmp/model_app_runs/<timestamp>/batch_XX/<app>.log`: per-app simulation log

Recommended settings on a multi-core server:

- Use `--build-jobs 16` for compilation
- Start with `--parallel 8` or `--parallel 12` for `simv`
- For all 30 benchmark apps, use `--batch-size 8` or `--batch-size 12` to keep resource usage stable

If the Verilator model has already been built, you can skip that step with:

```bash
scripts/run_model_split_apps.sh --mode run --no-verilate --parallel 8
```

### Ideal Dispatcher Mode

For performance experiments where only Ara and memory are modeled, enable the ideal dispatcher:

```bash
cd ara/apps
make bin/<program>.ideal      # generate vector trace
cd ara/hardware
make sim app=<program> ideal_dispatcher=1
```

---

## 💾 Deployment Notes

### DRAM Configuration

To increase the L2 memory size change `L2NumWords` in `ara/hardware/src/ara_soc.sv` and rebuild.

---

## 📄 License & Contact

This project is released under the [Apache‑2.0 license](LICENSE). Please cite the accompanying paper when using QLoMA in your research.

For questions or contributions, open an issue or pull request on the repository.

---

*Last updated: March 1, 2026*



进行多精度tiling搜索，分别针对1、2、4位精度进行搜索：
for b in 1 2 4; do
  (
    python scripts/tilling_search.py \
      --shapes-csv tmp/llm_gemm_shapes.csv \
      --buffer-bits 16384 \
      --prec-bits "$b" \
      --out-best-csv "tmp/best_config_per_shape_b${b}.csv" \
      --out-anchor-csv "tmp/best_config_b${b}.csv" \
      --progress-every-shapes 50 \
      --jobs 16 \
      --chunksize 8
  ) &
done
wait