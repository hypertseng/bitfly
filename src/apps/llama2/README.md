# llama2

`src/apps/llama2/` contains the bitfly-side LLM inference experiment area. It is separate from the benchmark matrix in `bmpmm_*` / `rvv_*`.

## Contents

- `main.c`: standalone C inference program adapted for the Ara environment
- `kernel/`: matmul kernels used by the inference flow
- `scripts/`: model and tokenizer conversion helpers
- local model and tokenizer binaries used for experiments
- `llama3.2.c/`: imported standalone project for additional Llama experiments

## Important Notes

- This directory may contain large runtime assets and local experiment files.
- Not every file here is meant to be synced upstream as a clean benchmark app.
- Treat it as an experiment area for inference bring-up, model conversion, and kernel integration.

## Related Files

- [`llama3.2.c/README.md`](llama3.2.c/README.md)
- [`../common/README.md`](../common/README.md)
