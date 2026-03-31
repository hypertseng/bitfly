# Source Overlays

`src/` is the maintained BitFly source-of-truth tree. If a change should survive across local Ara worktrees and remain part of the project, it belongs here first.

## Why `src/` Exists

BitFly separates:

- maintained project logic under `src/`
- the synced, buildable Ara workspace under `ara/`

The distinction is intentional:

```text
src/  = what BitFly means
ara/  = where BitFly is built and simulated
tmp/  = what BitFly produced
```

## Overlay Map

| Path | Scope | Sync Target |
| --- | --- | --- |
| `src/apps/` | benchmark apps, verification apps, shared software infrastructure | `ara/apps/` |
| `src/hardware/rtl/bmpu/` | BMPU-local RTL and focused testbenches | `ara/hardware/src/bmpu/` |
| `src/hardware/rtl/extended/src/` | Ara integration RTL overlays | `ara/hardware/src/` |
| `src/hardware/rtl/extended/include/` | package and typedef overlays | `ara/hardware/include/` |
| `src/llvm_instr/` | LLVM-side custom instruction support | external LLVM checkout via sync tooling |
| `src/Bender.yml` | top-level Bender overlay | `ara/Bender.yml` |

## Edit Decision Table

| If you need to change... | Edit here | Then do... |
| --- | --- | --- |
| benchmark case generation or app logic | `src/apps/` | sync to `ara/apps/`, rebuild apps |
| BMPU-local datapath or local testbench logic | `src/hardware/rtl/bmpu/` | sync to `ara/hardware/src/bmpu/`, rebuild hardware |
| Ara integration logic such as sequencer, lane, or VLSU overlays | `src/hardware/rtl/extended/` | sync to `ara/hardware/`, rebuild hardware |
| custom instruction parsing, printing, or encoding | `src/llvm_instr/` | sync to the LLVM checkout, rebuild the toolchain |
| a temporary one-off local experiment | `ara/` directly | promote back into `src/` only if it should be preserved |

## Promotion Rule

If a useful change started inside `ara/`, promote it back before treating it as a maintained project change:

1. move the logic into the matching location under `src/`
2. run `scripts/dev/sync_src_to_ara.sh`
3. review the synced result and any generated patch artifact

This keeps the repository reviewable and avoids making the Ara working tree the accidental source of truth.

## Sync Workflow

Maintained sync entry point:

```bash
scripts/dev/sync_src_to_ara.sh
```

Conceptually:

```text
src/ -> sync into ara/ -> build and simulate -> tmp/ outputs
```

## Local Navigation

- [`apps/README.md`](apps/README.md): benchmark app taxonomy and workload contract
- [`hardware/README.md`](hardware/README.md): RTL overlay organization
- [`llvm_instr/README.md`](llvm_instr/README.md): LLVM custom instruction support

There may be shorter local bring-up helpers in the tree, but `scripts/dev/` is the maintained synchronization path and the one the documentation assumes.
