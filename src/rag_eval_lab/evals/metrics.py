"""Retrieval metrics computed from where the first relevant chunk appears in the results."""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from rag_eval_lab.retrieval.base import Hit

CUTOFFS = (1, 3, 5, 10)


class QuestionResult(BaseModel):
    """What happened for one question: the rank of the first relevant chunk, and the time taken."""

    id: str
    area: str
    first_relevant_rank: int | None
    latency_ms: float


def is_relevant(hit: Hit, doc_id: str, span: tuple[int, int]) -> bool:
    """A hit is relevant when it comes from the right page and overlaps the answering section."""
    return hit.doc_id == doc_id and hit.start < span[1] and hit.end > span[0]


def first_relevant_rank(hits: Sequence[Hit], doc_id: str, span: tuple[int, int]) -> int | None:
    """1-based rank of the first relevant hit, or None if no hit is relevant."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, doc_id, span):
            return rank
    return None


def hit_rate(results: Sequence[QuestionResult], k: int) -> float:
    """Share of questions with a relevant chunk in the top ``k``."""
    if not results:
        return 0.0
    found = sum(
        1 for r in results if r.first_relevant_rank is not None and r.first_relevant_rank <= k
    )
    return found / len(results)


def mean_reciprocal_rank(results: Sequence[QuestionResult], k: int = 10) -> float:
    """Average of 1 / rank of the first relevant chunk, counting 0 beyond rank ``k``."""
    if not results:
        return 0.0
    total = sum(
        1.0 / r.first_relevant_rank
        for r in results
        if r.first_relevant_rank is not None and r.first_relevant_rank <= k
    )
    return total / len(results)


def summarize(results: Sequence[QuestionResult]) -> dict[str, float]:
    """Hit rate at each cutoff, MRR@10 and latency percentiles."""
    latencies = [r.latency_ms for r in results] or [0.0]
    metrics = {f"hit@{k}": hit_rate(results, k) for k in CUTOFFS}
    metrics["mrr@10"] = mean_reciprocal_rank(results, 10)
    metrics["latency_p50_ms"] = float(np.percentile(latencies, 50))
    metrics["latency_p95_ms"] = float(np.percentile(latencies, 95))
    return metrics
