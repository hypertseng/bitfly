# bmpu_verify

`bmpu_verify` is the fastest correctness-oriented app in the repository for validating BMPU mixed-precision behavior.

## What It Checks

- generated test vectors match the declared benchmark cases
- BMPU execution completes successfully
- packed output can be unpacked back into column-major form
- unpacked output matches the reference result tensor

## Why It Matters

Use this app before large benchmark campaigns when you changed:

- BMPU RTL
- low-precision packing logic
- benchmark data generation
- shared mixed-matmul helpers

## Local Structure

- `main.c`: executes the validation loop and compares results
- `kernel/`: generated tensors and benchmark case metadata
- `script/gen_data.py`: generator for the validation dataset

## Typical Flow

```bash
make -C ara/apps bin/bmpu_verify -j8
make -C ara/hardware simv app=bmpu_verify
```

A healthy run ends with `ALL CASES PASSED`.
