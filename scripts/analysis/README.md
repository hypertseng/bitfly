# Analysis Scripts

`scripts/analysis/` contains post-processing helpers for understanding benchmark behavior rather than launching simulations.

## Files

- `roofline.py`: roofline-style performance plotting
- `search.py`: parameter or design-space search helper
- `tiling_search.py`: tiling exploration helper
- `kernel_profiling.py`: kernel-level profiling support
- `inference_time.py`: inference-time analysis helper
- `latency_breakdown.pdf` / `latency_breakdown.png`: generated figures checked into the workspace

## Use This Directory For

- converting raw benchmark outputs into plots
- exploring tiling or scheduling choices
- preparing figures for reports or papers

Raw execution should still start from [`../benchmarks/README.md`](../benchmarks/README.md).
