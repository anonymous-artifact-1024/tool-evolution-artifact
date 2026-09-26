# Reproducibility guide

## Level 1: inspect and verify the released results

No model endpoint or benchmark download is required:

```bash
python scripts/verify_artifact.py
python scripts/summarize_results.py
```

This validates row counts, factorial coverage, missing-evidence handling,
reported analysis metadata, and file checksums.

## Level 2: run implementation tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Level 3: reproduce model execution

Full execution requires the pinned LinuxFLBench checkout, prepared Linux
source snapshots and function indexes, a container runtime, the pinned local
model or compatible service endpoint, and environment-provided credentials.

Credentials are never stored in configuration, logs, or command-line
arguments. Full execution is computationally expensive: the released main run
contains 9,660 episodes.

The frozen configuration in `configs/` and `data/provenance/main_schedule.json`
defines the factorial design. The main runner is `scripts/run_main_block.py`,
trusted-host scoring is in `scripts/score_main_run.py`, and full evidence
analysis is in `scripts/analyze_main.py`.

These core runtime files are preserved byte-for-byte to keep the hashes in the
activation record valid. Their historical default paths use the original
workspace layout; when invoking them from this compact artifact, pass the
released schedule, task catalog, dataset manifest, and output paths explicitly
through their command-line options.

`data/results/primary_contrasts.csv` is the flat table for the 20 primary
comparisons. `data/results/main_results.json` retains the complete analysis,
including task-level differences behind each reported contrast.
