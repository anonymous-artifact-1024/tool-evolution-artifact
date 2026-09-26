import random

import pytest

from saner_exp.experiment import CONDITION_NAMES, condition_order


def test_condition_order_is_stable_complete_and_does_not_touch_global_rng():
    random.seed(99)
    before = random.getstate()
    first = condition_order("8848", "Qwen3-14B", 1)
    assert random.getstate() == before
    assert set(first) == set(CONDITION_NAMES)
    assert len(first) == 7
    assert first == condition_order("8848", "Qwen3-14B", 1)
    assert first == ("A", "D1", "C1", "D2", "C2", "B2", "B1")


def test_condition_order_is_keyed_by_block_identity():
    orders = {
        condition_order("8848", "Qwen3-14B", repetition)
        for repetition in (1, 2, 3)
    }
    assert len(orders) > 1


@pytest.mark.parametrize("args", [("", "model", 1), ("1", "", 1), ("1", "model", 0)])
def test_condition_order_rejects_invalid_identity(args):
    with pytest.raises(ValueError):
        condition_order(*args)
