import pytest

from saner_exp.evaluation import (
    factorial_contrast_vectors,
    joint_max_t_intervals,
    practical_effect_class,
)


def test_factorial_contrasts_preserve_task_pairing_and_interaction():
    tasks = ["t1", "t2"]
    model = "m"
    values = {}
    rows = {
        "t1": {"A": 0.1, "B1": 0.2, "C1": 0.4, "D1": 0.8,
               "B2": 0.1, "C2": 0.2, "D2": 0.3},
        "t2": {"A": 0.2, "B1": 0.3, "C1": 0.5, "D1": 0.9,
               "B2": 0.2, "C2": 0.4, "D2": 0.7},
    }
    for task, conditions in rows.items():
        for condition, value in conditions.items():
            values[(task, model, condition)] = value

    contrasts = factorial_contrast_vectors(tasks, [model], values)
    assert len(contrasts) == 10
    assert contrasts["m|instance-1|extension_original"]["task_differences"] == pytest.approx([0.3, 0.3])
    assert contrasts["m|instance-1|interaction"]["task_differences"] == pytest.approx([0.3, 0.3])
    assert contrasts["m|instance-2|interaction"]["task_differences"] == pytest.approx([0.1, 0.3])


def test_joint_bootstrap_is_deterministic_and_joint():
    vectors = {
        "a": [-0.1, 0.0, 0.1, 0.2],
        "b": [0.2, 0.1, 0.0, -0.1],
    }
    first = joint_max_t_intervals(vectors, resamples=500, seed=2027)
    second = joint_max_t_intervals(vectors, resamples=500, seed=2027)
    assert first == second
    assert first["critical_value"] is not None
    assert set(first["intervals"]) == {"a", "b"}


def test_zero_variance_and_practical_classification():
    result = joint_max_t_intervals({"constant": [0.03, 0.03, 0.03]})
    assert result["critical_value"] is None
    assert result["intervals"]["constant"]["simultaneous_interval"] is None
    assert practical_effect_class([0.021, 0.04]) == "positive_exceeding_threshold"
    assert practical_effect_class([-0.01, 0.01]) == "practical_equivalence"
    assert practical_effect_class([-0.03, 0.01]) == "inconclusive"
