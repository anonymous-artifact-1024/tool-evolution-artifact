import pytest

from saner_exp.evaluation.main_analysis import CONDITIONS, MODELS, analyze_primary


def make_cells(tasks):
    cells = {}
    for task_index, task in enumerate(tasks):
        for model_index, model in enumerate(MODELS):
            for condition_index, condition in enumerate(CONDITIONS):
                for repetition in (1, 2, 3):
                    cells[(task, model, condition, repetition)] = (
                        0.1 + task_index * 0.01 + model_index * 0.02
                        + condition_index * 0.005 + repetition * 0.001
                    )
    return cells


def test_main_analysis_uses_common_complete_tasks_and_exact_missing_bounds():
    tasks = ["1", "2", "3"]
    cells = make_cells(tasks)
    cells[("1", "qwen-plus-2025-12-01", "C1", 3)] = None
    components = {"1": "driver", "2": "fs", "3": "fs"}

    result = analyze_primary(
        tasks, MODELS, cells, components, resamples=100, grouped_resamples=100,
    )

    assert result["complete_common_task_ids"] == ["2", "3"]
    assert result["excluded_task_ids"] == ["1"]
    assert result["primary_contrast_count"] == 20
    affected = next(
        row for row in result["contrasts"]
        if row["model"] == "qwen-plus-2025-12-01"
        and row["instance"] == 1 and row["contrast"] == "extension_original"
    )
    assert affected["missing_score_bound"][1] - affected["missing_score_bound"][0] \
        == pytest.approx(1 / 9)
    assert set(affected["condition_means"]) == {"A", "C1"}
    assert len(affected["task_difference_distribution"]["values"]) == 2
    assert set(affected["repetition_variability"]) == {"A", "C1"}
    unaffected = next(
        row for row in result["contrasts"]
        if row["model"] == "Qwen3-14B"
        and row["instance"] == 1 and row["contrast"] == "extension_original"
    )
    assert unaffected["missing_score_bound"][0] == pytest.approx(
        unaffected["missing_score_bound"][1]
    )
    assert len(result["exploratory_overall_upgrades"]) == 4


def test_component_grouped_sensitivity_is_seeded_and_discloses_imbalance():
    tasks = ["1", "2", "3", "4"]
    cells = make_cells(tasks)
    components = {"1": "large", "2": "large", "3": "large", "4": "small"}
    first = analyze_primary(
        tasks, MODELS, cells, components, resamples=50, grouped_resamples=50,
    )["component_grouped_sensitivity"]
    second = analyze_primary(
        tasks, MODELS, cells, components, resamples=50, grouped_resamples=50,
    )["component_grouped_sensitivity"]
    assert first == second
    assert first["group_count"] == 2
    assert first["minimum_group_size"] == 1
    assert first["maximum_group_size"] == 3
    assert first["standard_error_method"] == "sd_of_component_grouped_bootstrap_estimates"
