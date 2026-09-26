import pytest

from saner_exp.evaluation import SubmissionFormatError, score_submission


def test_single_target_rank_and_recall_metrics():
    score = score_submission(["drivers/fault.c"], ["a.c", "drivers/fault.c", "z.c"])
    assert score.reciprocal_rank == 0.5
    assert score.recall_at_1 == 0.0
    assert score.recall_at_5 == score.recall_at_10 == 1.0
    assert score.first_relevant_rank == 2


def test_official_fractional_recall_semantics_for_multiple_targets():
    score = score_submission(["a.c", "b.c"], ["b.c"])
    assert score.reciprocal_rank == 1.0
    assert score.recall_at_1 == score.recall_at_5 == score.recall_at_10 == 0.5


def test_leading_dot_slash_is_removed_and_duplicates_keep_first_position():
    score = score_submission(["a.c"], ["x.c", "./x.c", "./a.c"])
    assert score.submitted_count == 2
    assert score.first_relevant_rank == 2
    assert score.reciprocal_rank == 0.5


def test_missing_submission_scores_zero():
    score = score_submission(["a.c"], None)
    assert score.reciprocal_rank == 0.0
    assert score.recall_at_1 == score.recall_at_5 == score.recall_at_10 == 0.0


@pytest.mark.parametrize("paths", [[""], ["../a.c"], ["/a.c"], ["a\\b.c"], ["a.c"] * 11])
def test_invalid_submission_is_rejected(paths):
    with pytest.raises(SubmissionFormatError):
        score_submission(["a.c"], paths)
