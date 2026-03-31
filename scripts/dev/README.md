# Development Scripts

`scripts/dev/` contains maintenance helpers that keep the BitFly overlay tree and the Ara working tree aligned.

## Primary Entry Point

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

## Why It Matters

BitFly uses `src/` as the maintained source of truth. This script makes that policy operational and keeps the synced Ara delta inspectable and reviewable.
