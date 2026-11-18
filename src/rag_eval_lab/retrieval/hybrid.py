"""Hybrid search: fuse the vector and keyword rankings with reciprocal rank fusion (ADR-0003)."""

from collections.abc import Sequence

from rag_eval_lab.retrieval.base import Hit, Retriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids. Each list adds ``1 / (k + rank)`` to an id, with rank from 1.

    Only positions matter, so scores from different retrievers never need to be calibrated
    against each other. Ties keep the order in which ids were first seen.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    """Takes the top ``candidates`` from each retriever and keeps the best ``k`` after fusion."""

    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], candidates: int = 50, rrf_k: int = 60):
        self._retrievers = retrievers
        self._candidates = candidates
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[Hit]:
        by_id: dict[str, Hit] = {}
        rankings: list[list[str]] = []
        for retriever in self._retrievers:
            hits = retriever.retrieve(query, self._candidates)
            rankings.append([hit.chunk_id for hit in hits])
            for hit in hits:
                by_id.setdefault(hit.chunk_id, hit)
        fused = reciprocal_rank_fusion(rankings, self._rrf_k)[:k]
        return [by_id[chunk_id].model_copy(update={"score": score}) for chunk_id, score in fused]
