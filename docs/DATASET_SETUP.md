# Upstream dataset setup

The large upstream dataset and Linux source trees are not vendored in this
repository. This keeps the Git checkout small while retaining a precise,
verifiable input definition.

## LinuxFLBench

```text
repository: https://github.com/FudanSELab/LinuxFLBench.git
commit:     de65c3958f264137835eb23cacf3591fb89fc09b
license:    MIT
```

```bash
git clone https://github.com/FudanSELab/LinuxFLBench.git
cd LinuxFLBench
git checkout --detach de65c3958f264137835eb23cacf3591fb89fc09b
```

Expected benchmark file:

```text
path:    dataset/LINUXFLBENCH_dataset.jsonl
rows:    250
bytes:   1587676
sha256:  cc60e91b37a78d1cfb85f12c0bf884e41362afe798ac7ca2fa83e4b27bde974c
```

Verify it from the artifact root:

```bash
python scripts/verify_dataset.py /path/to/LinuxFLBench
```

## Linux source snapshots

The study uses source versions referenced by the frozen task catalogs. The
preparation scripts download or check upstream versions and construct
deterministic function indexes. Generated source trees and indexes can require
tens of gigabytes and must be stored outside the Git checkout.

The released `scripts/build_function_index.py` constructs the deterministic
per-version function index after source preparation. The full source-download
and storage orchestration is intentionally not included because it is tied to
the original large-data environment; the pinned upstream inputs and hashes
above are the portable reproduction contract.

LinuxFLBench's MIT notice is included in `licenses/LinuxFLBench-LICENSE`.
Linux kernel sources remain under their upstream licenses and are not
redistributed in this artifact.
