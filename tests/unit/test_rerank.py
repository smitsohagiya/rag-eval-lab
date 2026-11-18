from collections.abc import Sequence

from rag_eval_lab.retrieval.base import Hit
from rag_eval_lab.retrieval.rerank import RerankingRetriever


def hit(chunk_id: str, text: str) -> Hit:
    return Hit(
        chunk_id=chunk_id, doc_id="d.md", heading="", start=0, end=len(text), text=text, score=0.0
    )


class ListRetriever:
    name = "list"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.asked_for: list[int] = []

    def retrieve(self, query: str, k: int) -> list[Hit]:
        self.asked_for.append(k)
        return self.hits[:k]


class WordOverlapReranker:
    """Scores a text by how many query words it contains."""

    name = "word-overlap"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        words = set(query.lower().split())
        return [float(len(words & set(text.lower().split()))) for text in texts]


HITS = [
    hit("a", "nothing relevant"),
    hit("b", "docker image"),
    hit("c", "build a docker image fast"),
]


def test_candidates_are_reordered_by_the_reranker() -> None:
    retriever = RerankingRetriever(ListRetriever(HITS), WordOverlapReranker(), candidates=3)
    hits = retriever.retrieve("build docker image", k=2)
    assert [h.chunk_id for h in hits] == ["c", "b"]
    assert hits[0].score == 3.0


def test_it_fetches_at_least_the_candidate_count() -> None:
    base = ListRetriever(HITS)
    RerankingRetriever(base, WordOverlapReranker(), candidates=30).retrieve("docker", k=5)
    RerankingRetriever(base, WordOverlapReranker(), candidates=2).retrieve("docker", k=5)
    assert base.asked_for == [30, 5]


def test_ties_keep_the_base_order_and_names_compose() -> None:
    retriever = RerankingRetriever(ListRetriever(HITS), WordOverlapReranker(), candidates=3)
    assert [h.chunk_id for h in retriever.retrieve("unrelated words", k=3)] == ["a", "b", "c"]
    assert retriever.name == "list+rerank"


def test_no_candidates_means_no_results() -> None:
    assert RerankingRetriever(ListRetriever([]), WordOverlapReranker()).retrieve("x", k=3) == []
