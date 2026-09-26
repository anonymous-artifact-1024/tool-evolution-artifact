"""Read upstream JSONL without forwarding raw records to an agent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .task import AgentTask, GroundTruth


class DatasetValidationError(ValueError):
    """Malformed input or a mismatch with the recorded dataset identity."""


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _string(record: dict, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{key}: expected a nonempty string")
    return value  # Preserve official text, including whitespace.


def _strings(record: dict, key: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise DatasetValidationError(f"{key}: expected a nonempty list of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise DatasetValidationError(f"{key}: expected nonempty string elements")
    return tuple(value)


class LinuxFLBenchDataset:
    """Orchestrator-owned container. Pass only agent_task(id) to the agent.

    The container itself holds labels and must never be passed to an agent.
    This separation does not replace filesystem/process isolation.
    """

    def __init__(
        self,
        tasks: Mapping[str, AgentTask],
        truths: Mapping[str, GroundTruth],
        sha256: str,
    ) -> None:
        self._tasks = MappingProxyType(dict(tasks))
        self._truths = MappingProxyType(dict(truths))
        self.sha256 = sha256

    @classmethod
    def load(
        cls,
        dataset_path: str | Path,
        *,
        manifest_path: str | Path | None = None,
    ) -> LinuxFLBenchDataset:
        """Validate input; optionally enforce recorded byte hash and task count.

        File order is retained. No sampling, ID conversion, source-version
        resolution, or rewriting of upstream text is performed here.
        """
        raw = Path(dataset_path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        manifest = None
        if manifest_path is not None:
            try:
                manifest = json.loads(
                    Path(manifest_path).read_text(encoding="utf-8"),
                    object_pairs_hook=_object_without_duplicates,
                )
            except (UnicodeDecodeError, ValueError) as exc:
                raise DatasetValidationError("invalid dataset manifest JSON") from exc
            if not isinstance(manifest, dict):
                raise DatasetValidationError("dataset manifest must be an object")
            if manifest.get("dataset_name") != "LinuxFLBench":
                raise DatasetValidationError("manifest dataset_name must be LinuxFLBench")
            if type(manifest.get("task_count")) is not int or manifest["task_count"] <= 0:
                raise DatasetValidationError("manifest task_count must be a positive integer")
            if manifest.get("sha256") != digest:
                raise DatasetValidationError("dataset SHA-256 does not match manifest")

        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise DatasetValidationError("dataset must be UTF-8") from exc
        tasks = {}
        truths = {}
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, object_pairs_hook=_object_without_duplicates)
                if not isinstance(record, dict):
                    raise DatasetValidationError("task record must be an object")
                task_id = _string(record, "id")
                if task_id in tasks:
                    raise DatasetValidationError(f"duplicate task id: {task_id}")
                task = AgentTask(
                    task_id=task_id,
                    title=_string(record, "title"),
                    description=_string(record, "description"),
                    kernel_version=_string(record, "Kernel Version"),
                )
                truth = GroundTruth(
                    task_id=task_id,
                    patch=_strings(record, "patch"),
                    paths=_strings(record, "paths"),
                    methods=_strings(record, "methods"),
                )
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"line {line_number}: invalid JSON") from exc
            except DatasetValidationError as exc:
                raise DatasetValidationError(f"line {line_number}: {exc}") from exc
            tasks[task_id] = task
            truths[task_id] = truth

        if not tasks:
            raise DatasetValidationError("dataset contains no tasks")
        if manifest is not None and len(tasks) != manifest["task_count"]:
            raise DatasetValidationError("task count does not match manifest")
        return cls(tasks, truths, digest)

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def agent_task(self, task_id: str) -> AgentTask:
        return self._tasks[task_id]

    def ground_truth(self, task_id: str) -> GroundTruth:
        return self._truths[task_id]
