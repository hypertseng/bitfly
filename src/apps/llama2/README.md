# llama2

`src/apps/llama2/` contains the bitfly-side LLM inference experiment area. It is separate from the benchmark matrix in `bmpmm_*` / `rvv_*`.

## What This Directory Is For

Use this directory for inference-oriented experiments that do not fit the one-app-per-workload-slice benchmark model used by `run_model_split_apps.sh`.

Typical use cases:

- bring-up of end-to-end LLM inference flows on the Ara-based software stack
- comparing BMPMM-backed kernels against RVV-backed kernels inside one inference program
- generating embedded shape configurations for controlled prefill experiments
- local model conversion and tokenizer preparation

## Structure

| Path | Purpose |
| --- | --- |
| `main.c` | Ara-side inference entry point and experiment control flow |
| `kernel/` | BMPMM / RVV comparison kernels used by the inference flow |
| `scripts/` | model, tokenizer, and embedded-config generation helpers |
| `embedded_shape_cfgs.inc` | generated embedded shape table consumed by `main.c` |
| local `*.bin` assets | local model and tokenizer binaries used for experiments |
| `llama3.2.c/` | imported standalone upstream-style experiment area |

## Important Notes

- This directory intentionally mixes maintained source with local experiment assets.
- Not every file here is meant to be committed as project source.
- Large model binaries and local story/model artifacts should normally stay local.
- `llama3.2.c/` is a separate imported project area, not a normal BitFly-maintained app subtree.

## Tracked Source vs Local Assets

Usually maintained:

- `main.c`
- `kernel/`
- `scripts/`
- `embedded_shape_cfgs.inc` when it is intentionally versioned with the current experiment flow

Usually local-only:

- large model binaries
- local tokenizer/model dumps
- third-party imported experiment trees unless intentionally vendored

## Relationship To The Main Benchmark Matrix

This directory is not the same thing as the model-split benchmark app matrix under `bmpmm_*` and `rvv_*`.

Use `llama2/` when you want:

- end-to-end inference experiments
- prompt/prefill-oriented sweeps
- integrated kernel comparison inside one application

Use the model-split benchmark apps when you want:

- one workload slice per app
- batch benchmark campaigns
- paper-style app matrix runs under `scripts/benchmarks/run_model_split_apps.sh`

## Related Files

- [`llama3.2.c/README.md`](llama3.2.c/README.md)
- [`../common/README.md`](../common/README.md)
- [`../../../scripts/benchmarks/README.md`](../../../scripts/benchmarks/README.md)
