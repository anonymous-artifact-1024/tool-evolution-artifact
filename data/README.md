# Data guide

This directory contains the final data products needed to inspect and verify
the reported experiment. CSV and JSON are the canonical formats; spreadsheets
and internal review workbooks are intentionally excluded.

## `tasks/`

- `main_tasks.jsonl`: 230 label-free main-study tasks.
- `preexperiment_tasks.jsonl`: 20 label-free protocol-validation tasks.
- `task_split.json`: deterministic split metadata and seed.

Each JSONL task contains `id`, `title`, `description`, and `kernel_version`.
The catalogs do not expose ground-truth patches, target files, methods, or
bisected commits to the agent.

## `results/`

### `episode_results.csv`

One row per task x model x repetition x condition: 9,660 rows. Important
columns include identifiers, terminal status, reciprocal rank, Recall@K,
generation/tool-call counts, submitted files, and log provenance.

The three `missing_evidence` rows have missing metrics by design.

### `block_results.csv`

One row per task x model x repetition: 1,380 rows. It records the randomized
seven-condition order, completeness, analysis eligibility, and hashes of the
block summary and score records.

### `primary_contrasts.csv`

The 20 preregistered primary comparisons. Estimates are task-level mean
differences after averaging the three repetitions within each task, model, and
condition. The confidence limits are simultaneous 95% max-t bootstrap
intervals over the complete common task set.

### `main_results.json`

The complete machine-readable analysis record, including task-level contrast
values, secondary metrics, process summaries, missingness bounds, and
provenance. This is the authoritative detailed result object.

## `provenance/`

This directory provides the frozen schedule, condition and scoring contracts,
execution activation record, raw-log manifest, verification index, and
curated-file checksums.

## `sample_logs/`

The sample contains task 10042, repetition 1, for both models and all seven
conditions, together with each block's `summary.json` and `scores.json`.

The complete raw archive contains 9,660 JSONL files:

```text
filename: raw_call_logs_9660.tar.gz
bytes:    935610414
sha256:   d5fa92bf0b4ccca045acd2e21ac370178d806b49cf7a87f3dd0c286772fb0f8f
```

It should be distributed as a release asset or through the anonymous artifact
host, rather than as a Git object.
