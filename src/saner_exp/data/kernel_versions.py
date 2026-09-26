"""Kernel version inventory helpers; never substitute a nearby release."""

from __future__ import annotations

from dataclasses import dataclass
import re


STABLE_REPOSITORY = "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git"
ARCHIVE_ROOT = "https://www.kernel.org/pub/linux/kernel"
VERSION_PATTERN = re.compile(r"(2\.[456]\.\d+(?:\.\d+)?|[3-9]\.\d+(?:\.\d+)?)(?:-rc([1-9]\d*))?")


@dataclass(frozen=True)
class KernelVersion:
    original: str
    numbers: tuple[int, ...]
    rc: int | None

    @property
    def tag(self) -> str:
        return "v" + self.original

    @property
    def historical_archive(self) -> bool:
        return self.numbers < (2, 6, 12)

    @property
    def archive_series(self) -> str:
        if self.numbers[0] == 2:
            return f"v2.{self.numbers[1]}"
        return f"v{self.numbers[0]}.x"


def parse_version(value: str) -> KernelVersion:
    match = VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported exact kernel version: {value!r}")
    return KernelVersion(value, tuple(map(int, match[1].split("."))), int(match[2]) if match[2] else None)


def version_order(value: str) -> tuple:
    version = parse_version(value)
    return version.numbers, version.rc is None, version.rc or 0


def parse_tag_refs(text: str) -> dict[str, dict[str, str]]:
    """Keep advertised tag and peeled object IDs distinct; types are unverified."""
    refs = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{40})\s+refs/tags/(\S+)", line)
        if match is None:
            raise ValueError("malformed git tag advertisement")
        oid, tag = match.groups()
        key = "peeled_object_id" if tag.endswith("^{}") else "ref_object_id"
        tag = tag.removesuffix("^{}")
        entry = refs.setdefault(tag, {})
        if key in entry and entry[key] != oid:
            raise ValueError(f"conflicting advertised object for {tag}")
        entry[key] = oid
    if not refs or any("ref_object_id" not in entry for entry in refs.values()):
        raise ValueError("empty tag listing or peeled object without a tag")
    return refs


def parse_checksums(text: str) -> dict[str, str]:
    """Parse published hashes only; this does not authenticate the PGP signature."""
    checksums = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+\*?(\S+)", line)
        if match:
            digest, filename = match.groups()
            if filename in checksums and checksums[filename] != digest:
                raise ValueError(f"conflicting checksum for {filename}")
            checksums[filename] = digest
    if not checksums:
        raise ValueError("no SHA-256 entries found")
    return checksums
