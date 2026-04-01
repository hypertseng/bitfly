# Repository Structure

This document is the maintainer-facing map of the BitFly repository. Use it when you need to answer:

- where a change should be made
- which directories are maintained source versus generated outputs
- how BitFly-specific overlays relate to the synced `ara/` working tree

## One-Screen Mental Model

The repository is organized around a clear ownership split:

```text
src/   = maintained BitFly source of truth
ara/   = synced Ara working tree for build and simulation
scripts/ = command entry points
docs/  = workflow and repository notes
tmp/   = generated run outputs
build/ = disposable local build products
```

If a change should survive local experiments and remain part of BitFly, it should normally start in `src/`.

## Top-Level Directory Map

| Path | Purpose | Typical Contents | Edit Policy |
| --- | --- | --- | --- |
| `src/` | Maintained BitFly overlays | apps, RTL overlays, custom instruction support | Edit here first for persistent project changes |
| `ara/` | Synced Ara workspace | build tree, apps, hardware, toolchain | Treat as working tree, not primary source of truth |
| `scripts/` | Operational command surface | sync helpers, benchmark launchers, analysis, debug scripts | Edit when changing workflow or automation |
| `docs/` | Reader and maintainer documentation | quick start, workflow notes, repo guides | Edit when improving navigation or reproducibility |
| `tmp/` | Generated run outputs | benchmark runs, CSV summaries, local logs | Usually not tracked |
| `build/` | Local build outputs | testbench binaries and temporary products | Usually not tracked |
| `patches/` | Sync-side artifacts | exported local patch snapshots | `patches/local/` is local-only by default |
| `tcl/` | Synthesis-side collateral | synthesis Tcl scripts | Edit only for flow changes |
| `.gitignore` | Local artifact policy | ignore rules for logs, models, traces | Keep aligned with actual workflow outputs |

## `src/` Structure

`src/` is the most important tree for maintained development.

| Path | Role | Sync Target |
| --- | --- | --- |
| `src/apps/` | App overlays, verification apps, shared software logic | `ara/apps/` |
| `src/hardware/rtl/bmpu/` | BMPU-local RTL and focused test logic | `ara/hardware/src/bmpu/` |
| `src/hardware/rtl/extended/src/` | Ara integration RTL overlays | `ara/hardware/src/` |
| `src/hardware/rtl/extended/include/` | Ara include/package overlays | `ara/hardware/include/` |
| `src/llvm_instr/` | LLVM-side instruction support | external LLVM checkout via sync tooling |

### `src/apps/`

This tree is organized by workload role:

| Path Pattern | Meaning |
| --- | --- |
| `bmpmm_*` | Proposed BMPMM benchmark path |
| `rvv_*` | RVV baseline benchmark path |
| `bmpu_verify/` | BMPU correctness regression app |
| `common/` | Shared generators, helpers, and templates |
| `llama2/` | Separate inference-oriented experiment flow |

Common app-local files:

| File / Dir | Purpose |
| --- | --- |
| `main.c` | app entry point and run-level logging |
| `kernel/` | generated tensors, metadata, and kernel code |
| `script/gen_data.py` | app-local data generation |

### `src/hardware/`

This tree carries maintained RTL overlays:

| Path | Meaning |
| --- | --- |
| `src/hardware/rtl/bmpu/` | BMPU datapath, SA, PE, and BMPU-local logic |
| `src/hardware/rtl/extended/` | Ara-side sequencer, lane, VLSU, dispatcher, and include overlays |

### `src/llvm_instr/`

This tree is only relevant when BitFly changes require custom instruction support on the toolchain side.

## `scripts/` Structure

The `scripts/` tree is organized by workflow, not by language:

| Path | Purpose |
| --- | --- |
| `scripts/dev/` | sync helpers and repository maintenance commands |
| `scripts/benchmarks/` | benchmark campaign launchers and post-processing helpers |
| `scripts/analysis/` | roofline, tiling search, and result analysis |
| `scripts/debug/` | focused hardware block bring-up and debug runners |

If you are not sure where to start:

| Task | Entry Point |
| --- | --- |
| sync BitFly overlays into Ara | `scripts/dev/sync_src_to_ara.sh` |
| run benchmark matrix | `scripts/benchmarks/run_model_split_apps.sh` |
| regenerate analysis plots | `scripts/analysis/roofline.py` |
| run focused BMPU block debug | `scripts/debug/` helpers |

## `docs/` Structure

The `docs/` tree is for longer-lived explanations and workflow notes:

| File | Use It For |
| --- | --- |
| `artifact_quickstart.md` | fastest path from clone to running something useful |
| `artifact_checklist.md` | release / artifact hygiene checklist |
| `benchmark_workflow.md` | benchmark methodology and output contract |
| `repo_guide.md` | ownership model and `src/` versus `ara/` split |
| `repo_structure.md` | directory structure and file placement map |

## Source vs Generated Outputs

These directories are normally maintained source:

- `src/`
- `scripts/`
- `docs/`
- `tcl/`

These are usually generated, local, or disposable:

- `tmp/`
- `build/`
- simulator logs
- generated plots and PDFs unless intentionally versioned
- local model binaries
- local traces and debug dumps
- `patches/local/`

## Where To Put A Change

| If you are changing... | Put it in... | Then do... |
| --- | --- | --- |
| benchmark app logic | `src/apps/` | sync and rebuild apps |
| BMPU or Ara overlay RTL | `src/hardware/` | sync and rebuild hardware |
| runner / analysis workflow | `scripts/` | rerun the relevant command path |
| repository navigation or explanation | `docs/` or top-level README | keep links consistent |
| temporary local experiment | `ara/` or `tmp/` | promote only if it becomes maintained |

## Sync Flow

The normal maintained workflow is:

```text
edit src/ -> sync into ara/ -> build and simulate -> inspect tmp/ outputs
```

Maintained sync entry point:

```bash
scripts/dev/sync_src_to_ara.sh
```

## Practical Navigation

Recommended reading order for new maintainers:

1. `README.md`
2. `docs/repo_structure.md`
3. `docs/repo_guide.md`
4. `src/README.md`
5. `scripts/README.md`

This order moves from repository layout, to ownership model, to source tree details, to command entry points.
