import pytest

from rag_eval_lab.evals.metrics import (
    QuestionResult,
    first_relevant_rank,
    hit_rate,
    is_relevant,
    mean_reciprocal_rank,
    summarize,
)
from rag_eval_lab.retrieval.base import Hit


def hit(doc_id: str, start: int, end: int) -> Hit:
    return Hit(
        chunk_id=f"{doc_id}#{start}",
        doc_id=doc_id,
        heading="",
        start=start,
        end=end,
        text="",
        score=0.0,
    )


def result(rank: int | None, latency: float = 10.0) -> QuestionResult:
    return QuestionResult(id="q001", area="tutorial", first_relevant_rank=rank, latency_ms=latency)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0, 100, True),
        (150, 160, True),
        (90, 101, True),
        (0, 100 - 1, False),
        (200, 300, False),
        (199, 250, True),
    ],
)
def test_relevance_is_overlap_with_the_section(start: int, end: int, expected: bool) -> None:
    assert is_relevant(hit("a.md", start, end), "a.md", (99, 200)) is expected


def test_a_hit_from_another_page_is_never_relevant() -> None:
    assert not is_relevant(hit("b.md", 0, 1000), "a.md", (0, 1000))


def test_first_relevant_rank() -> None:
    hits = [hit("b.md", 0, 50), hit("a.md", 0, 10), hit("a.md", 120, 130)]
    assert first_relevant_rank(hits, "a.md", (100, 200)) == 3
    assert first_relevant_rank(hits, "c.md", (0, 10)) is None


def test_hit_rate_and_mrr() -> None:
    results = [result(1), result(3), result(None), result(12)]
    assert hit_rate(results, 1) == 0.25
    assert hit_rate(results, 3) == 0.5
    assert hit_rate(results, 10) == 0.5
    assert mean_reciprocal_rank(results, 10) == pytest.approx((1 + 1 / 3) / 4)


def test_empty_results() -> None:
    assert hit_rate([], 5) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


def test_summarize_reports_cutoffs_mrr_and_latency() -> None:
    metrics = summarize([result(1, 10.0), result(2, 30.0)])
    assert metrics["hit@1"] == 0.5
    assert metrics["hit@3"] == 1.0
    assert metrics["mrr@10"] == pytest.approx(0.75)
    assert metrics["latency_p50_ms"] == pytest.approx(20.0)
