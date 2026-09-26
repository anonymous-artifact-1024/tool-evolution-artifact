# Tool Evolution Study Artifact

This anonymous artifact contains the implementation, frozen protocol, and
analysis-ready data for a controlled study of tool-interface evolution in
repository-level fault localization agents.

The repository is intentionally curated. It contains the final authoritative
files used for the reported analysis, not engineering snapshots, candidate
configurations, caches, or intermediate review packages.

Some frozen configuration files retain a pre-execution status such as
`pending_validation_gates`. Those values record the point at which the design
was frozen; `data/provenance/main_experiment_activation.json` records the later
gate completion, and `data/results/main_results.json` records the verified
completed execution.

## Artifact at a glance

- 230 main-study maintenance tasks and 20 pre-experiment tasks.
- 2 models, 7 tool conditions, and 3 repetitions.
- 1,380 blocks reached a terminal protocol state, covering 9,660 scheduled episodes.
- 9,657 observed score cells and 3 protocol-defined missing-evidence cells.
- 227 tasks in the common-complete primary analysis.
- 20 primary contrasts fixed before formal execution.

The compact artifact is self-contained for inspecting and verifying the
reported results. Re-running model inference additionally requires the pinned
LinuxFLBench input and Linux source snapshots described in
[`docs/DATASET_SETUP.md`](docs/DATASET_SETUP.md).

## Quick verification

Python 3.11 or newer is required. The integrity checks use only the standard
library:

```bash
python scripts/verify_artifact.py
python scripts/summarize_results.py
```

To run the implementation test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Repository map

| Path | Contents |
|---|---|
| `src/` | Agent, tools, scheduling, scoring, and statistical analysis code |
| `scripts/` | Final execution, scoring, analysis, and verification entry points |
| `configs/` | Final experimental conditions and model/protocol configuration |
| `data/tasks/` | Label-free task catalogs and the frozen task split |
| `data/results/` | Complete analysis-ready episode, block, and contrast results |
| `data/provenance/` | Frozen schedule, contracts, checksums, and evidence manifests |
| `data/sample_logs/` | One complete seven-condition block for each model |
| `docs/` | Dataset preparation, data dictionary, and reproduction instructions |

## Data availability boundary

The analysis-ready data used to support the paper's tables and conclusions are
included directly in `data/results/`. The upstream benchmark, Linux source
trees, generated function indexes, and the 892 MiB full call-log archive are
not committed to Git. Their pinned sources, expected hashes, and verification
instructions are documented in `docs/` and `data/provenance/`.

The included task catalogs expose only the label-free task information made
available to the agent. Ground-truth patches, files, methods, and commits are
not included in agent inputs.

## Results status

All 1,380 scheduled blocks reached a terminal protocol state. Three service
model episodes ended as `missing_evidence` after a second infrastructure
failure, as specified by the frozen policy. They were not converted to zero
scores or rerun based on observed results. The primary analysis therefore uses
the same 227 common-complete tasks for both models and reports zero-to-one
missingness bounds over all 230 scheduled tasks.

## Licensing

The artifact code is released under the MIT License. LinuxFLBench retains its
own MIT copyright notice in `licenses/LinuxFLBench-LICENSE`. Linux kernel
source code is not redistributed by this repository.

The core implementation and configuration files named in the activation record
are preserved byte for byte, and their SHA-256 values match the frozen
provenance record. Portable
release-only helpers (`verify_artifact.py`, `verify_dataset.py`, and
`summarize_results.py`) are separate from the frozen execution implementation.
