# Source Overlays

`src/` is the persistent bitfly-owned overlay tree. In normal development, edit here first and treat `ara/` as the build target.

## Why This Directory Exists

The repository separates:

- stable project logic that should survive across worktrees
- transient Ara-side build state used to compile and simulate

`src/` holds the first category. `ara/` holds the second.

## Overlay Map

| Path | Scope | Sync Target |
| --- | --- | --- |
| `src/apps/` | benchmark apps, verification apps, shared app code | `ara/apps/` |
| `src/hardware/rtl/bmpu/` | BMPU-local RTL and focused testbenches | `ara/hardware/src/bmpu/` |
| `src/hardware/rtl/extended/src/` | Ara-side integration RTL | `ara/hardware/src/` |
| `src/hardware/rtl/extended/include/` | package / typedef overlays | `ara/hardware/include/` |
| `src/llvm_instr/` | LLVM assembler / disassembler / encoding changes | external LLVM checkout via `--llvm-dst` |
| `src/Bender.yml` | top-level Bender overlay | `ara/Bender.yml` |

## Edit Decision Table

Use this rule table before making changes:

| If you want to change... | Edit here | Then do... |
| --- | --- | --- |
| benchmark case generation or app logic | `src/apps/` | sync to `ara/apps/`, rebuild app |
| BMPU-local datapath or unit-testbench logic | `src/hardware/rtl/bmpu/` | sync to `ara/hardware/src/bmpu/`, rebuild hardware |
| Ara integration, sequencer, lane, or VLSU logic | `src/hardware/rtl/extended/` | sync to `ara/hardware/`, rebuild hardware |
| custom instruction parsing / printing / encoding | `src/llvm_instr/` | sync to LLVM checkout, rebuild toolchain |
| a one-off local experiment in the current Ara tree | `ara/` directly | decide later whether to promote back into `src/` |

## Promotion Rule

If you started from a quick edit inside `ara/` and later decide the change should become part of bitfly:

1. move the logic back into the matching location under `src/`
2. run `scripts/dev/sync_src_to_ara.sh`
3. review the generated patch under `patches/local/`

This keeps the source-of-truth policy intact.

## Sync Workflow

The maintained sync command is:

```bash
scripts/dev/sync_src_to_ara.sh
```

Conceptually:

```text
src/ -> sync script -> ara/ -> build/simulate -> tmp/ outputs
```

The script can also export an Ara-side patch so the synced delta remains reviewable.

## Local Files

- [`apps/README.md`](apps/README.md): app taxonomy and benchmark contract
- [`hardware/README.md`](hardware/README.md): RTL overlay organization
- [`llvm_instr/README.md`](llvm_instr/README.md): custom instruction support
- `compile.sh`: legacy helper that syncs and rebuilds Ara quickly

`compile.sh` is useful for short local bring-up loops, but the `scripts/dev/` path is the maintained workflow.
