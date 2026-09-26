"""Frozen main-experiment complete-case and missingness sensitivity analysis."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Mapping, Sequence

from .factorial import (
    arithmetic_mean, factorial_contrast_vectors, joint_max_t_intervals,
    practical_effect_class, sample_sd,
)


CONDITIONS = ("A", "B1", "C1", "D1", "B2", "C2", "D2")
MODELS = ("Qwen3-14B", "qwen-plus-2025-12-01")


def _validate_cell(value: float | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError("main-analysis scores must be missing or finite values in [0,1]")


def condition_interval(
    cells: Mapping[tuple[str, str, str, int], float | None],
    task: str, model: str, condition: str,
) -> tuple[float, float]:
    values = [cells[(task, model, condition, repetition)] for repetition in (1, 2, 3)]
    for value in values:
        _validate_cell(value)
    observed = sum(float(value) for value in values if value is not None)
    missing = sum(value is None for value in values)
    return observed / 3.0, (observed + missing) / 3.0


def contrast_bounds(
    task_ids: Sequence[str], models: Sequence[str],
    cells: Mapping[tuple[str, str, str, int], float | None],
) -> dict[str, dict]:
    result = {}
    formulas = (
        ("extension_original", "RQ1", (1, "C"), (-1, "A")),
        ("extension_refactored", "RQ1", (1, "D"), (-1, "B")),
        ("refactoring_base", "RQ2", (1, "B"), (-1, "A")),
        ("refactoring_extended", "RQ2", (1, "D"), (-1, "C")),
        ("interaction", "RQ3", (1, "D"), (-1, "C"), (-1, "B"), (1, "A")),
    )
    for model in models:
        for instance in (1, 2):
            names = {letter: f"{letter}{instance}" for letter in "BCD"}
            names["A"] = "A"
            for name, rq, *terms in formulas:
                task_ranges = []
                for task in task_ids:
                    lower = upper = 0.0
                    for sign, letter in terms:
                        low, high = condition_interval(cells, task, model, names[letter])
                        if sign > 0:
                            lower += low
                            upper += high
                        else:
                            lower -= high
                            upper -= low
                    task_ranges.append((lower, upper))
                key = f"{model}|instance-{instance}|{name}"
                result[key] = {
                    "model": model,
                    "instance": instance,
                    "rq": rq,
                    "contrast": name,
                    "lower": arithmetic_mean([item[0] for item in task_ranges]),
                    "upper": arithmetic_mean([item[1] for item in task_ranges]),
                    "tasks": len(task_ranges),
                }
    return result


def grouped_joint_max_t_intervals(
    vectors: Mapping[str, Sequence[float]], task_ids: Sequence[str],
    components: Mapping[str, str], *, resamples: int = 10_000, seed: int = 2027,
    confidence: float = 0.95,
) -> dict:
    if set(task_ids) - set(components):
        raise ValueError("component metadata is missing for analyzed tasks")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(task_ids):
        value = components[task_id]
        if not isinstance(value, str) or not value:
            raise ValueError("component labels must be non-empty strings")
        groups[value].append(index)
    if len(groups) < 2:
        return {"status": "inadequate_group_count", "group_count": len(groups), "intervals": {}}
    estimates = {key: arithmetic_mean(values) for key, values in vectors.items()}
    labels = sorted(groups)
    rng = random.Random(seed)
    bootstrap_estimates = {key: [] for key in vectors}
    for _ in range(resamples):
        indexes = [index for _ in range(len(labels))
                   for index in groups[labels[rng.randrange(len(labels))]]]
        for key in vectors:
            bootstrap_estimates[key].append(
                arithmetic_mean([vectors[key][index] for index in indexes])
            )
    # Unequal component sizes make the number of sampled tasks variable.  Estimate
    # the scale from the grouped replicates themselves instead of applying the
    # independent-task standard error to a cluster bootstrap.
    standard_errors = {
        key: sample_sd(values) for key, values in bootstrap_estimates.items()
    }
    active = [key for key, value in standard_errors.items() if value > 0]
    maxima = [
        max(
            abs(bootstrap_estimates[key][index] - estimates[key]) / standard_errors[key]
            for key in active
        ) if active else 0.0
        for index in range(resamples)
    ]
    maxima.sort()
    critical = maxima[max(0, math.ceil(confidence * resamples) - 1)] if active else None
    intervals = {
        key: None if key not in active else [
            estimates[key] - critical * standard_errors[key],
            estimates[key] + critical * standard_errors[key],
        ]
        for key in vectors
    }
    sizes = [len(groups[label]) for label in labels]
    return {
        "status": "complete",
        "method": "component_grouped_max_t_bootstrap",
        "standard_error_method": "sd_of_component_grouped_bootstrap_estimates",
        "resamples": resamples,
        "seed": seed,
        "confidence": confidence,
        "group_count": len(labels),
        "minimum_group_size": min(sizes),
        "maximum_group_size": max(sizes),
        "critical_value": critical,
        "intervals": intervals,
    }


def analyze_primary(
    task_ids: Sequence[str], models: Sequence[str],
    cells: Mapping[tuple[str, str, str, int], float | None],
    components: Mapping[str, str], *, resamples: int = 10_000,
    grouped_resamples: int = 10_000,
) -> dict:
    expected = {
        (task, model, condition, repetition)
        for task in task_ids for model in models for condition in CONDITIONS
        for repetition in (1, 2, 3)
    }
    if set(cells) != expected:
        raise ValueError("main score cells do not exactly match the scheduled factorial design")
    for value in cells.values():
        _validate_cell(value)
    complete = [task for task in task_ids if all(
        cells[(task, model, condition, repetition)] is not None
        for model in models for condition in CONDITIONS for repetition in (1, 2, 3)
    )]
    excluded = [task for task in task_ids if task not in set(complete)]
    if len(complete) < 2:
        raise ValueError("fewer than two common complete tasks remain")
    means = {
        (task, model, condition): arithmetic_mean([
            float(cells[(task, model, condition, repetition)]) for repetition in (1, 2, 3)
        ])
        for task in complete for model in models for condition in CONDITIONS
    }
    records = factorial_contrast_vectors(complete, models, means)
    vectors = {key: row["task_differences"] for key, row in records.items()}
    primary = joint_max_t_intervals(vectors, resamples=resamples, seed=2027, confidence=0.95)
    bounds = contrast_bounds(task_ids, models, cells)
    contrasts = []
    for key, row in records.items():
        interval = primary["intervals"][key]
        bound = bounds[key]
        instance = row["instance"]
        if row["contrast"] == "interaction":
            reported_conditions = ("A", f"B{instance}", f"C{instance}", f"D{instance}")
        else:
            reported_conditions = (row["left_condition"], row["right_condition"])
        condition_means = {
            condition: arithmetic_mean([
                means[(task, row["model"], condition)] for task in complete
            ])
            for condition in reported_conditions
        }
        repetition_variability = {
            condition: {
                "measure": "mean_within_task_sample_sd_across_three_repetitions",
                "value": arithmetic_mean([
                    sample_sd([
                        float(cells[(task, row["model"], condition, repetition)])
                        for repetition in (1, 2, 3)
                    ])
                    for task in complete
                ]),
                "task_count": len(complete),
            }
            for condition in reported_conditions
        }
        differences = row["task_differences"]
        primary_class = practical_effect_class(interval["simultaneous_interval"], delta=0.02)
        bound_interval = [bound["lower"], bound["upper"]]
        bound_class = practical_effect_class(bound_interval, delta=0.02)
        has_missing = not math.isclose(bound["lower"], bound["upper"], abs_tol=1e-15)
        contrasts.append({
            "key": key,
            "model": row["model"],
            "instance": row["instance"],
            "rq": row["rq"],
            "contrast": row["contrast"],
            "left_condition": row["left_condition"],
            "right_condition": row["right_condition"],
            **interval,
            "primary_delta_0_02_class": primary_class,
            "delta_sensitivity": {
                "0.01": practical_effect_class(interval["simultaneous_interval"], delta=0.01),
                "0.05": practical_effect_class(interval["simultaneous_interval"], delta=0.05),
            },
            "missing_score_bound": bound_interval,
            "missing_score_bound_delta_0_02_class": bound_class,
            "missingness_robustness": (
                "no_missing_evidence" if not has_missing else
                "same_practical_class_under_zero_to_one_bounds"
                if bound_class == primary_class else
                "not_robust_do_not_claim_primary_class"
            ),
            "condition_means": condition_means,
            "task_difference_distribution": {
                "task_count": len(complete),
                "mean": arithmetic_mean(differences),
                "sample_sd": sample_sd(differences),
                "minimum": min(differences),
                "maximum": max(differences),
                "values": [
                    {"task_id": task, "difference": value}
                    for task, value in zip(complete, differences)
                ],
            },
            "repetition_variability": repetition_variability,
        })
    grouped = grouped_joint_max_t_intervals(
        vectors, complete, components, resamples=grouped_resamples, seed=2027, confidence=0.95,
    )
    exploratory = []
    for model in models:
        for instance in (1, 2):
            left = f"D{instance}"
            values = [means[(task, model, left)] - means[(task, model, "A")]
                      for task in complete]
            exploratory.append({
                "model": model,
                "instance": instance,
                "contrast": "overall_upgrade",
                "expression": f"{left}-A",
                "status": "exploratory_not_in_primary_family",
                "complete_common_task_count": len(complete),
                "condition_means": {
                    left: arithmetic_mean([means[(task, model, left)] for task in complete]),
                    "A": arithmetic_mean([means[(task, model, "A")] for task in complete]),
                },
                "mean_difference": arithmetic_mean(values),
                "task_differences": [
                    {"task_id": task, "difference": value}
                    for task, value in zip(complete, values)
                ],
            })
    return {
        "format": "saner-main-primary-analysis-v1",
        "scheduled_task_count": len(task_ids),
        "complete_common_task_count": len(complete),
        "complete_common_task_ids": complete,
        "excluded_task_ids": excluded,
        "excluded_task_count": len(excluded),
        "primary_contrast_count": len(contrasts),
        "primary_bootstrap": {key: value for key, value in primary.items() if key != "intervals"},
        "component_grouped_sensitivity": grouped,
        "contrasts": contrasts,
        "exploratory_overall_upgrades": exploratory,
    }
