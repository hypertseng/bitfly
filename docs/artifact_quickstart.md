# Artifact Quick Start

This document is the shortest path from cloning BitFly to running a correctness check, launching a benchmark app, and regenerating the roofline figure.

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

If you need a more detailed repository explanation, continue with [`repo_guide.md`](repo_guide.md), [`../src/README.md`](../src/README.md), and [`../scripts/README.md`](../scripts/README.md).
