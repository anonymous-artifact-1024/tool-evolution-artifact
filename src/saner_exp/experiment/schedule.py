"""Stable condition ordering that does not depend on global random state."""

from __future__ import annotations

import hashlib
import random


CONDITION_NAMES = ("A", "B1", "C1", "D1", "B2", "C2", "D2")


def condition_order(task_id: str, model: str, repetition: int, *, seed: int = 2027) -> tuple[str, ...]:
    if type(task_id) is not str or not task_id or type(model) is not str or not model:
        raise ValueError("task_id and model must be non-empty strings")
    if type(repetition) is not int or repetition < 1:
        raise ValueError("repetition must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    material = f"saner-condition-order-v1\0{seed}\0{task_id}\0{model}\0{repetition}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    order = list(CONDITION_NAMES)
    random.Random(derived_seed).shuffle(order)
    return tuple(order)
