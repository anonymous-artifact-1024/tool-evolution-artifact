"""Deterministic task partitioning, independent of labels and global RNG state."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


SPLIT_ALGORITHM_VERSION = "1"
ID_ORDER = "ascending (integer value, original string); IDs retained as strings"
RANDOM_ALGORITHM = "Python random.Random(seed).shuffle; MT19937; isolated RNG"


@dataclass(frozen=True, slots=True)
class TaskSplit:
    preexperiment: tuple[str, ...]
    main: tuple[str, ...]


def split_task_ids(
    task_ids: Iterable[str], *, seed: int = 2027, preexperiment_size: int = 20,
) -> TaskSplit:
    """Sort numeric IDs, shuffle once, then take a prefix for pre-experiment.

    A string tie-breaker makes ordering unambiguous even with zero-padded IDs.
    Saved list order is partition permutation order, not episode execution order.
    """
    ids = list(task_ids)
    if any(not isinstance(x, str) or not x.isascii() or not x.isdigit() for x in ids):
        raise ValueError("task IDs must be nonempty ASCII digit strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task IDs are not allowed")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(preexperiment_size) is not int or not 0 < preexperiment_size < len(ids):
        raise ValueError("preexperiment_size must leave both partitions nonempty")
    ordered = sorted(ids, key=lambda task_id: (int(task_id), task_id))
    random.Random(seed).shuffle(ordered)
    return TaskSplit(tuple(ordered[:preexperiment_size]), tuple(ordered[preexperiment_size:]))
