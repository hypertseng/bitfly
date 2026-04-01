# Common App Support

`src/apps/common/` contains shared helpers used across multiple benchmark and verification apps.

## Use This Directory For

- benchmark case description and selection helpers
- BMP configuration dispatch support
- shared low-precision mixed-matmul helpers
- common operator templates and utility headers

## Important Files

| File | Purpose |
| --- | --- |
| `bmpmm_bench_common.h` | shared benchmark case types and runtime-cache helpers |
| `bmpmm_lowp_mixed_common.c` / `.h` | common mixed-precision execution helpers |
| `bmpcfg_dispatch.c` / `.h` | BMP configuration dispatch support |
| `bmpmm_case_selection.py` | shared case-selection logic for generators |
| `bmpmm_gen_common.py` | shared data-generation utilities |

## Typical Responsibilities

This directory is the right place for logic that should stay consistent across multiple apps, for example:

- benchmark case metadata shared by several app generators
- reference low-precision execution helpers reused by correctness and benchmark apps
- dispatch or configuration helpers that should behave the same in BMPMM and verification flows
- generator-side utilities that would otherwise be duplicated across many app-local `script/` directories

## When Not To Put Code Here

Do not move logic here when it is:

- specific to one model-split app
- specific to one inference experiment under `llama2/`
- tightly coupled to one app's local logging or control flow

If only one app needs the behavior, keep it local until duplication pressure is real.

If multiple apps need the same behavior, this directory is usually the right place to factor it out before duplicating logic across app directories.
