# Extended RTL Sources

`src/hardware/rtl/extended/src/` contains the main Ara-side module overlays that integrate bitfly behavior into the broader hardware stack.

## Main Areas

- top-level Ara integration modules such as `ara.sv` and `ara_soc.sv`
- dispatch and sequencing modules
- lane-local execution support
- VLSU-side support

## Directory Map

| Path | Responsibility |
| --- | --- |
| `ara.sv` / `ara_soc.sv` | top-level Ara integration overlays |
| `ara_dispatcher.sv` | request formation and decode-side control plumbing |
| `ara_sequencer.sv` | global instruction sequencing |
| `lane/` | lane-local scheduling and operand movement |
| `vlsu/` | vector memory-side support |

## Edit Rule

Start here when the change crosses BMPU-local logic and interacts with the broader Ara execution flow.

## Read Next

- [`lane/README.md`](lane/README.md)
- [`vlsu/README.md`](vlsu/README.md)
