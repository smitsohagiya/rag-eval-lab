import pytest

from rag_eval_lab.retrieval import reciprocal_rank_fusion


def test_items_ranked_well_by_both_lists_come_first() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert [item for item, _ in fused] == ["b", "c", "a", "d"]


def test_scores_follow_the_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_ties_keep_first_seen_order() -> None:
    fused = reciprocal_rank_fusion([["x"], ["y"]], k=60)
    assert [item for item, _ in fused] == ["x", "y"]


def test_empty_rankings() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
