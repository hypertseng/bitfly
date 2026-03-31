# Analysis Scripts

`scripts/analysis/` contains post-processing and design-space exploration helpers. These scripts help interpret benchmark behavior after runs have completed; they are not the primary launch point for simulation campaigns.

## Start Here

Most commonly used entry points:

- `python3 scripts/analysis/roofline.py`
- `python3 scripts/analysis/tiling_search.py --help`

## Main Files

| File | Purpose |
| --- | --- |
| `roofline.py` | roofline-style plotting for searched configurations |
| `tiling_search.py` | tiling and grouping search under current software-template and RTL assumptions |
| `search.py` | additional design-space search helper |
| `kernel_profiling.py` | kernel-level profiling support |
| `inference_time.py` | inference-time analysis helper |

## Use This Directory For

- converting benchmark outputs into plots or summaries
- exploring tiling or scheduling choices
- preparing figures for reports or papers
- checking whether search assumptions match the current software template and RTL

Simulation and benchmark launch still start from [`../benchmarks/README.md`](../benchmarks/README.md).
