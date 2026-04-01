# Repository Guide

This note explains the ownership boundaries of the repository.

## Why This Matters

BitFly is easiest to maintain when source ownership stays explicit. Most confusion in this repository comes from mixing:

- maintained overlays under `src/`
- the synced Ara working tree under `ara/`
- generated outputs under `tmp/` and local build directories

## Directory Roles

| Path | Role |
| --- | --- |
| `src/` | persistent bitfly-owned overlays |
| `ara/` | active Ara working tree used for build and simulation |
| `scripts/` | operational entry points |
| `docs/` | methodology and workflow notes |
| `tmp/` | generated run outputs |
| `build/` | local temporary builds |

## Ownership Rule

When developing a change:

- edit `src/` if the change should become part of the maintained bitfly overlay set
- edit `ara/` directly if the change is a local experiment or an Ara-side concern not yet promoted into bitfly
- use `scripts/dev/sync_src_to_ara.sh` to propagate maintained overlays from `src/` into `ara/`

## Decision Table

| Situation | Correct Home |
| --- | --- |
| persistent benchmark-app or RTL improvement | `src/` |
| local build, simulation, or debug output | `tmp/` or `build/` |
| temporary Ara-only experiment not yet promoted | `ara/` |
| workflow explanation or repository navigation | `docs/` or a nearby `README.md` |

## Practical Reading

The repository is easiest to understand as:

```text
src/   = what bitfly means
ara/   = where bitfly is built and run
tmp/   = what bitfly produced
```

That distinction is more important than any individual file list.

## Experiment Artifacts

Large logs, generated assembly, simulator outputs, Spike objects, and other temporary build products are intentionally excluded from the tracked source tree. They belong under run directories such as `tmp/model_app_runs/`, not in the source overlays.

## Review Standard

Before treating a change as ready for a paper artifact, make sure a reviewer can answer all three questions quickly:

1. where the maintained source of the change lives
2. how that change reaches the buildable Ara tree
3. where the resulting outputs will be produced
