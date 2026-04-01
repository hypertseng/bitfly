# Patches

The `patches/` directory stores exported diffs that capture Ara-side changes produced from bitfly-managed overlays.

## Layout

- `local/`: default output location for `scripts/dev/sync_src_to_ara.sh`

## Typical Use

Generate a sync patch after copying `src/` overlays into `ara/`:

```bash
scripts/dev/sync_src_to_ara.sh
```

Or generate only the patch:

```bash
scripts/dev/sync_src_to_ara.sh --no-sync
```

Use this directory when you want:

- a reviewable artifact of what changed inside `ara/`
- a portable diff to move bitfly-managed changes between worktrees
- a checkpoint before continuing local edits

## Tracking Policy

`patches/local/` is intended for local workflow support and is ignored by default. Promote a patch into tracked project history only if it serves as a deliberate reviewed artifact rather than a transient checkpoint.
