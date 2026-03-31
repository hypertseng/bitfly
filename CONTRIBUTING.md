# Contributing to BitFly

BitFly is a research codebase with a strict boundary between maintained source overlays and the synced Ara working tree. Contributions are much easier to review if they follow that boundary.

## Contribution Scope

Good contributions include:

- BMPU/BMPMM hardware or software improvements under `src/`
- benchmark or generator fixes
- documentation improvements
- reproducibility fixes
- analysis and plotting fixes

Before opening a change, make sure it belongs to BitFly rather than only to your local `ara/` worktree.

## Source-of-Truth Rule

- Edit `src/` if the change should be preserved as part of BitFly.
- Edit `ara/` directly only for temporary local experiments or Ara-upstream-only work.
- If a useful change started in `ara/`, promote it back into `src/` before preparing a reviewable patch.

The maintained sync path is:

```bash
scripts/dev/sync_src_to_ara.sh
```

## Directory Ownership

| Path | Use |
| --- | --- |
| `src/apps/` | benchmark apps, generators, shared app logic |
| `src/hardware/rtl/bmpu/` | BMPU-local RTL and focused testbenches |
| `src/hardware/rtl/extended/` | Ara integration RTL overlays |
| `src/llvm_instr/` | custom instruction support |
| `scripts/` | automation, debug, and analysis tooling |
| `docs/` | workflow, artifact, and repository documentation |

## Before Opening a PR

Please do the following:

1. Keep the diff scoped to one technical concern.
2. Sync from `src/` into `ara/` if the change needs Ara-side validation.
3. Run the smallest relevant validation for your change.
4. Update documentation if user-facing behavior or workflow changed.
5. Avoid committing local logs, generated plots, large binaries, or temporary run outputs unless the change explicitly requires a tracked artifact.

## Validation Expectations

Pick the narrowest useful check:

- BMPU correctness: `make -C ara/hardware simv app=bmpu_verify`
- one benchmark app smoke test:

```bash
scripts/benchmarks/run_model_split_apps.sh \
  --mode run \
  --apps <app> \
  --parallel 1 \
  --batch-size 1
```

- analysis-only change:

```bash
python3 scripts/analysis/roofline.py
```

If you could not run a relevant validation, state that clearly in the PR description.

## Commit and PR Guidance

- Prefer small, reviewable commits.
- Use descriptive commit messages that explain the technical change, not only the symptom.
- In PR descriptions, include:
  - what changed
  - why it changed
  - how it was validated
  - any remaining caveats

## Documentation Standard

BitFly is intended to be understandable to paper readers and artifact evaluators. If you add a new workflow, command surface, or directory role, update the relevant README or `docs/` entry at the same time.

## Questions

If the right ownership boundary is unclear, open an issue or explain the uncertainty in the PR description before expanding the patch.
