"""Immutable experiment inputs. Task IDs retain their upstream string type."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AgentTask:
    """Allowlisted task data; task_id is an orchestration identifier."""

    task_id: str
    title: str
    description: str
    kernel_version: str


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Evaluator-only labels; excluded from the default object representation."""

    task_id: str
    patch: tuple[str, ...] = field(repr=False)
    paths: tuple[str, ...] = field(repr=False)
    methods: tuple[str, ...] = field(repr=False)
