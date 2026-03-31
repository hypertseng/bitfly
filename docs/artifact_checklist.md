# Artifact Checklist

Use this checklist before publishing BitFly results, opening the repository to external readers, or preparing a paper artifact submission.

## Repository Presentation

- Root [`README.md`](../README.md) explains the research question, repository layout, and quick start.
- The `src/` versus `ara/` ownership model is documented clearly.
- Key entry-point commands are documented and tested at least once locally.
- Standard metadata files exist:
  - [`LICENSE`](../LICENSE)
  - [`CITATION.cff`](../CITATION.cff)
  - [`CONTRIBUTING.md`](../CONTRIBUTING.md)

## Reproducibility

- Clone and submodule instructions are correct.
- Required host tools are listed.
- Ara-specific dependencies point to [`../ara/DEPENDENCIES.md`](../ara/DEPENDENCIES.md).
- At least one correctness command is documented.
- At least one benchmark command is documented.
- Analysis / plotting regeneration commands are documented.

## Experimental Contract

- Proposed path and baseline path are named consistently.
- Precision naming is consistent across docs and scripts.
- Benchmark app naming rules are documented.
- The meaning of generated run directories is documented.

## Outputs And Logging

- Benchmark run outputs are described consistently:
  - `apps.txt`
  - `runner.log`
  - `summary.csv`
  - per-app logs
- Generated outputs are not confused with maintained source.
- Large logs, local binaries, and temporary artifacts are excluded from the maintained source narrative.

## Code And Documentation Consistency

- README links resolve correctly.
- Script names in docs match the current repository.
- Analysis docs match the actual generated figure names.
- If search or benchmark inputs changed, the corresponding docs were updated.

## Before Release Or Submission

- Remove or de-emphasize machine-local paths from user-facing documentation where possible.
- Confirm that no accidental local-only assets are required for the documented quick start.
- Record the exact commands used to generate the main reported results.
- Verify that the documentation still makes sense to a new reader who has not seen the internal workflow before.
