"""Task-level factorial contrasts and joint max-t bootstrap intervals."""

from __future__ import annotations

import math
import random
from statistics import median
from typing import Mapping, Sequence


CONTRASTS = (
    ("extension_original", "RQ1", "C", "A"),
    ("extension_refactored", "RQ1", "D", "B"),
    ("refactoring_base", "RQ2", "B", "A"),
    ("refactoring_extended", "RQ2", "D", "C"),
)


def arithmetic_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = arithmetic_mean(values)
    return math.sqrt(math.fsum((value - center) ** 2 for value in values) / (len(values) - 1))


def describe(values: Sequence[float]) -> dict:
    if not values:
        raise ValueError("cannot describe an empty sequence")
    return {
        "count": len(values),
        "mean": arithmetic_mean(values),
        "median": median(values),
        "sample_sd": sample_sd(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def factorial_contrast_vectors(
    task_ids: Sequence[str],
    models: Sequence[str],
    task_condition_means: Mapping[tuple[str, str, str], float],
) -> dict[str, dict]:
    """Return the 20 predeclared paired task-level contrast vectors."""
    records: dict[str, dict] = {}
    for model in models:
        for instance in ("1", "2"):
            names = {letter: f"{letter}{instance}" for letter in ("B", "C", "D")}
            names["A"] = "A"
            component_vectors: dict[str, list[float]] = {}
            for contrast, rq, left, right in CONTRASTS:
                left_condition = names[left]
                right_condition = names[right]
                values = [
                    task_condition_means[(task_id, model, left_condition)]
                    - task_condition_means[(task_id, model, right_condition)]
                    for task_id in task_ids
                ]
                key = f"{model}|instance-{instance}|{contrast}"
                records[key] = {
                    "model": model,
                    "instance": int(instance),
                    "rq": rq,
                    "contrast": contrast,
                    "left_condition": left_condition,
                    "right_condition": right_condition,
                    "task_differences": values,
                }
                component_vectors[contrast] = values
            interaction = [
                extended - base
                for extended, base in zip(
                    component_vectors["refactoring_extended"],
                    component_vectors["refactoring_base"],
                )
            ]
            key = f"{model}|instance-{instance}|interaction"
            records[key] = {
                "model": model,
                "instance": int(instance),
                "rq": "RQ3",
                "contrast": "interaction",
                "left_condition": f"(D{instance}-C{instance})",
                "right_condition": f"(B{instance}-A)",
                "task_differences": interaction,
            }
    return records


def joint_max_t_intervals(
    vectors: Mapping[str, Sequence[float]],
    *,
    resamples: int = 10_000,
    seed: int = 2027,
    confidence: float = 0.95,
) -> dict:
    """Compute the paper's joint task-level bootstrap simultaneous intervals."""
    if not vectors:
        raise ValueError("at least one contrast vector is required")
    lengths = {len(values) for values in vectors.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("all contrast vectors must share at least two aligned tasks")
    if resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    task_count = next(iter(lengths))
    estimates = {key: arithmetic_mean(values) for key, values in vectors.items()}
    standard_errors = {
        key: sample_sd(values) / math.sqrt(task_count) for key, values in vectors.items()
    }
    active = [key for key, standard_error in standard_errors.items() if standard_error > 0.0]
    maxima: list[float] = []
    if active:
        rng = random.Random(seed)
        for _ in range(resamples):
            indexes = [rng.randrange(task_count) for _ in range(task_count)]
            maxima.append(max(
                abs(
                    arithmetic_mean([vectors[key][index] for index in indexes]) - estimates[key]
                ) / standard_errors[key]
                for key in active
            ))
        maxima.sort()
        quantile_index = max(0, math.ceil(confidence * resamples) - 1)
        critical_value: float | None = maxima[quantile_index]
    else:
        critical_value = None

    intervals = {}
    for key in vectors:
        standard_error = standard_errors[key]
        interval = None if standard_error == 0.0 or critical_value is None else [
            estimates[key] - critical_value * standard_error,
            estimates[key] + critical_value * standard_error,
        ]
        intervals[key] = {
            "estimate": estimates[key],
            "standard_error": standard_error,
            "simultaneous_interval": interval,
        }
    return {
        "method": "joint_task_level_max_t_bootstrap",
        "resamples": resamples,
        "seed": seed,
        "confidence": confidence,
        "task_count": task_count,
        "critical_value": critical_value,
        "zero_standard_error_contrasts": sorted(set(vectors) - set(active)),
        "intervals": intervals,
    }


def practical_effect_class(interval: Sequence[float] | None, *, delta: float = 0.02) -> str:
    if interval is None:
        return "no_inferential_interval_zero_task_variance"
    lower, upper = interval
    if lower > delta:
        return "positive_exceeding_threshold"
    if upper < -delta:
        return "negative_exceeding_threshold"
    if lower >= -delta and upper <= delta:
        return "practical_equivalence"
    return "inconclusive"
