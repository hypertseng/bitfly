# bmpu_verify

`bmpu_verify` is the fastest correctness-oriented app in the repository for validating BMPU mixed-precision behavior.

## What It Checks

- generated test vectors match the declared benchmark cases
- BMPU execution completes successfully
- packed output can be unpacked back into column-major form
- unpacked output matches the reference result tensor

## Current Coverage

- 18 directed cases
- precisions: binary, INT2, INT4
- group shapes: `gm x gn = 2x2`, `4x1`, `1x2`, `1x4`, `8x1`, `1x8`
- tail coverage for non-full tiles and larger `ntile` edge cases

The `gm=8` / `gn=8` cases are legal even when the last tile group is smaller than 8.
Execution clips the tail group to the remaining tile count, so these cases explicitly
exercise the clipped-group path.

Additional current coverage includes small and tail-oriented binary cases such as:

- `mtile=8, ntile=128` minimal and tail cases
- `mtile=16, ntile=64` minimal and tail cases

## Data Layout Notes

- `result_torch` is stored as column-major reference data.
- `result_hp` is not a plain `M*N` matrix in this app. It is a packed BMPU store buffer laid out
  tile-by-tile and block-by-block.
- For non-full edge tiles, `result_hp` must be sized by packed tile capacity:
  `ceil(M/mtile) * ceil(N/ntile) * ceil(mtile/8) * ceil(ntile/16) * 8 * 16` int16 elements.

If `result_hp` is only allocated as `M*N`, tail-tile cases can overwrite following symbols in
the generated dataset and cause false mismatches or hangs.

## Why It Matters

Use this app before large benchmark campaigns when you changed:

- BMPU RTL
- low-precision packing logic
- benchmark data generation
- shared mixed-matmul helpers

This app should remain the quickest full-flow signal that a maintained BMPU-related change is still functionally safe.

## Local Structure

- `main.c`: executes the validation loop and compares results
- `kernel/`: generated tensors and benchmark case metadata
- `script/gen_data.py`: generator for the validation dataset

## Debug Controls

- app-side compare/debug prints are controlled by `BMPU_VERIFY_DEBUG`
- hardware-side debug prints are controlled from `src/hardware/rtl/extended/include/bitfly_debug.svh`
- keep both disabled by default for full regressions

## Typical Flow

```bash
source /data2/zzx/data/miniconda3/etc/profile.d/conda.sh
conda activate bitfly
./scripts/sync_src_to_ara.sh --no-patch
make -C ara/apps bin/bmpu_verify -j8
make -C ara/hardware simv app=bmpu_verify
```

A healthy run ends with `ALL CASES PASSED`.
