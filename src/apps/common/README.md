# Common App Support

`src/apps/common/` contains shared helpers used by multiple benchmark and verification apps.

## Main Responsibilities

- benchmark case description and selection helpers
- BMP configuration dispatch support
- shared low-precision mixed-matmul helpers
- common operator templates and utility headers

## Important Files

- `bmpmm_bench_common.h`: shared benchmark case types and runtime-cache helpers
- `bmpmm_lowp_mixed_common.c` / `.h`: common mixed-precision execution helpers
- `bmpcfg_dispatch.c` / `.h`: BMP configuration dispatch support
- `bmpmm_case_selection.py`: shared case-selection logic for generators
- `bmpmm_gen_common.py`: shared data-generation utilities

If multiple apps need the same behavior, this directory is usually the right place to factor it out.
