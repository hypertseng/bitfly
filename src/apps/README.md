# Apps

`src/apps/` contains bitfly-managed application overlays that are synced into `ara/apps/`.

This tree is the software-side definition of the experimental workload set used by the repository.

## Taxonomy

| Category | Directories | Role |
| --- | --- | --- |
| proposed benchmark path | `bmpmm_*` | BMPMM-based implementation under evaluation |
| baseline benchmark path | `rvv_*` | RVV implementation used for comparison |
| correctness regression | `bmpu_verify` | fast validation of BMPU packing and mixed-precision behavior |
| shared infrastructure | `common` | shared generators, case definitions, and helper code |
| separate inference experiment | `llama2` | exploratory LLM inference flow, not the main paper-style benchmark matrix |

## Experimental Unit

For model-split evaluation, each benchmark app represents exactly one point in the workload matrix:

```text
<implementation>_<precision>_<model>
```

Examples:

- `bmpmm_binary_gemma3_270m`
- `bmpmm_INT2_opt_13b`
- `rvv_INT4_qwen25_15b`

This structure matters because it keeps:

- generated tensors scoped to one workload slice
- simulator logs scoped to one app
- runtime summaries easy to aggregate into paper figures

The intended comparison is always:

- `bmpmm_*` as the proposed path
- `rvv_*` as the baseline path

under the same model-derived shape set.

## Generic Vs Model-Split Apps

Benchmark apps come in two forms:

| Form | Example | Intended Use |
| --- | --- | --- |
| generic | `bmpmm_INT2`, `rvv_binary` | compact regression or bring-up |
| model-split | `bmpmm_INT2_gemma3_270m` | formal benchmark campaigns and paper plots |

The batch runner primarily consumes the model-split apps.

## Common Structure

Most benchmark app directories follow the same internal pattern:

- `main.c`: app entry and per-case logging
- `kernel/`: implementation files plus generated tensors and case metadata
- `tests.c` / `tests.h`: local helper or validation code when needed
- `script/gen_data.py`: app-specific generator

## Where To Edit What

Use this quick guide:

| Change Type | Edit Location |
| --- | --- |
| shared case-selection or generator policy | `common/` |
| one app's workload tensor generation | that app's `script/gen_data.py` |
| one app's kernel implementation | that app's `kernel/` |
| per-app logging or execution flow | that app's `main.c` |
| BMPU correctness-oriented checks | `bmpu_verify/` |
| local inference experiments outside the benchmark matrix | `llama2/` |

## Important Boundaries

- `bmpu_verify` is for correctness, not performance reporting.
- `bmpmm_*` vs `rvv_*` is the core performance comparison.
- `llama2/` is an experiment area and should not be confused with the benchmark matrix used by `run_model_split_apps.sh`.
- `common/` is the right place to factor out behavior shared across many benchmark apps.

## Documentation Strategy

Only directories with distinct behavior carry their own README:

- [`common/README.md`](common/README.md)
- [`bmpu_verify/README.md`](bmpu_verify/README.md)
- [`llama2/README.md`](llama2/README.md)

The many repetitive benchmark app directories are documented by the shared naming and structure rules in this file rather than by per-directory boilerplate.
