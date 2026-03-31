# Application Overlays

`src/apps/` contains BitFly-managed application overlays that are synced into `ara/apps/`. This tree is the software-side definition of the workload matrix used for correctness checks and paper-style benchmarking.

## Start Here

Use this directory to answer three questions:

- which apps represent the proposed BMPMM path versus the RVV baseline
- how one app maps to one workload slice
- where shared generator and template logic should live

## Taxonomy

| Category | Directories | Role |
| --- | --- | --- |
| proposed benchmark path | `bmpmm_*` | BMPMM-based implementation under evaluation |
| baseline benchmark path | `rvv_*` | RVV implementation used for comparison |
| correctness regression | `bmpu_verify` | focused validation of BMPU packing and low-bit execution behavior |
| shared infrastructure | `common` | generators, case definitions, templates, and common helpers |
| separate inference experiment | `llama2` | exploratory inference flow outside the main benchmark matrix |

## Experimental Unit

For model-split evaluation, one benchmark app corresponds to one workload slice:

```text
<implementation>_<precision>_<model>
```

Examples:

- `bmpmm_binary_gemma3_270m`
- `bmpmm_INT2_opt_13b`
- `rvv_INT4_qwen25_15b`

This structure keeps:

- generated tensors scoped to one workload slice
- simulator logs scoped to one app
- runtime summaries straightforward to aggregate into paper figures

The main comparison is always:

- `bmpmm_*` as the proposed path
- `rvv_*` as the baseline path

under the same model-derived shape set.

## Generic Versus Model-Split Apps

| Form | Example | Intended Use |
| --- | --- | --- |
| generic | `bmpmm_INT2`, `rvv_binary` | short regression, bring-up, or local debugging |
| model-split | `bmpmm_INT2_gemma3_270m` | formal benchmark campaigns and reported comparisons |

The batch runner primarily targets the model-split apps.

## Common App Structure

Most benchmark app directories contain:

- `main.c`: app entry point and case-level logging
- `kernel/`: implementation code plus generated tensors and case metadata
- `tests.c` / `tests.h`: local helpers or validation logic where needed
- `script/gen_data.py`: app-specific data generator

## Where To Edit

| Change Type | Edit Location |
| --- | --- |
| shared case-selection or generator policy | `common/` |
| one app's workload tensor generation | that app's `script/gen_data.py` |
| one app's kernel implementation | that app's `kernel/` |
| one app's logging or control flow | that app's `main.c` |
| correctness-oriented checks for BMPU behavior | `bmpu_verify/` |
| local inference experiments outside the benchmark matrix | `llama2/` |

## Boundaries

- `bmpu_verify` is for correctness, not paper-performance reporting.
- `bmpmm_*` versus `rvv_*` is the main reported comparison.
- `llama2/` is not the same thing as the benchmark matrix used by `run_model_split_apps.sh`.
- `common/` is the right place to factor out behavior shared across multiple apps.

## Related Documentation

- [`common/README.md`](common/README.md)
- [`bmpu_verify/README.md`](bmpu_verify/README.md)
- [`llama2/README.md`](llama2/README.md)
- [`../../docs/benchmark_workflow.md`](../../docs/benchmark_workflow.md)

The many repetitive benchmark app directories are intentionally documented by shared rules here rather than by duplicating boilerplate README files per app.
