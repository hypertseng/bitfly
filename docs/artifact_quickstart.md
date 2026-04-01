# Artifact Quick Start

This document is the shortest path from cloning BitFly to running a correctness check, launching a benchmark app, and regenerating the roofline figure.

## What You Should Learn From This Guide

After following this page once, you should know:

- where maintained BitFly source lives
- how to sync that source into the buildable Ara tree
- how to run one correctness regression and one benchmark flow
- where generated outputs are supposed to land

## 1. Clone the repository

```bash
git clone --recursive git@github.com:hypertseng/bitfly.git
cd bitfly
git submodule update --init --recursive
```

## 2. Install core dependencies

Recommended host tools:

- `git`
- `make`
- `python3`
- `gcc` / `g++`
- `rsync`
- `Verilator`

Ara-specific setup is documented in [`../ara/DEPENDENCIES.md`](../ara/DEPENDENCIES.md).

## 3. Sync BitFly overlays into Ara

```bash
scripts/dev/sync_src_to_ara.sh
```

This copies maintained overlays from `src/` into the buildable Ara working tree under `ara/`.

If you plan to preserve a change in the repository, treat this sync step as mandatory rather than optional.

## 4. Run the fastest correctness regression

```bash
make -C ara/hardware verilate -j8
make -C ara/apps bin/bmpu_verify -j8
make -C ara/hardware simv app=bmpu_verify
```

Expected success signature:

```text
ALL CASES PASSED
```

Use this regression before long benchmark campaigns and after any maintained BMPU-related software or RTL change.

## 5. Run one benchmark app

```bash
scripts/benchmarks/run_model_split_apps.sh \
  --mode run \
  --apps bmpmm_INT2_gemma3_270m \
  --parallel 1 \
  --batch-size 1
```

Typical outputs appear under `tmp/model_app_runs/<run>/`:

- `apps.txt`
- `runner.log`
- `summary.csv`
- `batch_XX/<app>.log`

Treat that run directory as the minimal reproducibility unit for a benchmark result.

## 6. Run a full benchmark campaign

```bash
scripts/benchmarks/run_model_split_apps.sh \
  --mode all \
  --build-jobs 16 \
  --parallel 5 \
  --batch-size 5
```

For methodology and output interpretation, see [`benchmark_workflow.md`](benchmark_workflow.md).

## 7. Regenerate the roofline figure

```bash
python3 scripts/analysis/roofline.py
```

Outputs:

- `roofline_search_results.png`
- `roofline_search_results.pdf`

## 8. Mental model for the repository

Use this rule throughout development and reproduction:

```text
src/  = maintained BitFly logic
ara/  = synced build and simulation tree
tmp/  = generated outputs
```

## What Not To Commit As Maintained Source

These are typically local or generated artifacts rather than maintained project logic:

- simulator logs
- large model binaries
- traces and debug dumps
- temporary patch exports
- benchmark run directories under `tmp/`

If you need a more detailed repository explanation, continue with [`repo_structure.md`](repo_structure.md), [`repo_guide.md`](repo_guide.md), [`../src/README.md`](../src/README.md), and [`../scripts/README.md`](../scripts/README.md).
