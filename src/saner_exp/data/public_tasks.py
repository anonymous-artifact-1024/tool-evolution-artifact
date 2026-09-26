"""Load the label-free task projection allowed inside an agent container."""

from __future__ import annotations

import json
from pathlib import Path

from .linuxflbench import DatasetValidationError, _object_without_duplicates, _string
from .task import AgentTask


ALLOWED_FIELDS = frozenset({"id", "title", "description", "kernel_version"})


def load_public_tasks(path: Path | str) -> dict[str, AgentTask]:
    tasks = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_object_without_duplicates)
            if type(record) is not dict or set(record) != ALLOWED_FIELDS:
                raise DatasetValidationError("public task record has unexpected fields")
            task = AgentTask(
                task_id=_string(record, "id"),
                title=_string(record, "title"),
                description=_string(record, "description"),
                kernel_version=_string(record, "kernel_version"),
            )
        except (json.JSONDecodeError, DatasetValidationError) as error:
            raise DatasetValidationError(f"line {line_number}: {error}") from error
        if task.task_id in tasks:
            raise DatasetValidationError(f"line {line_number}: duplicate task id: {task.task_id}")
        tasks[task.task_id] = task
    if not tasks:
        raise DatasetValidationError("public task catalog is empty")
    return tasks
