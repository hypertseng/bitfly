# Dev Scripts

`scripts/dev/` contains maintenance helpers that keep the bitfly overlay tree and the Ara working tree in sync.

## Files

- `sync_src_to_ara.sh`: sync managed overlays from `src/` into `ara/` and optionally emit a patch

## Recommended Command

```bash
scripts/dev/sync_src_to_ara.sh
```

## What It Syncs

- `src/apps/` -> `ara/apps/`
- `src/hardware/rtl/bmpu/` -> `ara/hardware/src/bmpu/`
- `src/hardware/rtl/extended/src/` -> `ara/hardware/src/`
- `src/hardware/rtl/extended/include/` -> `ara/hardware/include/`

With `--llvm-dst`, it can also sync:

- `src/llvm_instr/` -> custom LLVM destination

## Why This Directory Matters

The repository uses `src/` as the bitfly source of truth. This script makes that policy practical and keeps Ara diffs reviewable.
