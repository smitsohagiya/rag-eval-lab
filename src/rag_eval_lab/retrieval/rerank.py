"""Re-rank a retriever's candidates with a cross-encoder."""

from collections.abc import Sequence
from typing import Protocol

from rag_eval_lab.embeddings import embedding_text
from rag_eval_lab.retrieval.base import Hit, Retriever


class Reranker(Protocol):
    """Scores how well each text answers the query. Higher is better."""

    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A local ONNX cross-encoder run with fastembed.

    It reads the question and each chunk together, which is slower than comparing two
    precomputed embeddings but usually more precise. The model is downloaded on first use.
    """

    def __init__(
        self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2", batch_size: int = 32
    ) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # loaded only when used

        self.name = model_name
        self._batch_size = batch_size
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self._model.rerank(query, list(texts), batch_size=self._batch_size)
        return [float(value) for value in scores]


class RerankingRetriever:
    """Fetches ``candidates`` chunks from a base retriever and re-orders them with a reranker."""

    def __init__(self, base: Retriever, reranker: Reranker, candidates: int = 30) -> None:
        self._base = base
        self._reranker = reranker
        self._candidates = candidates
        self.name = f"{base.name}+rerank"

    def retrieve(self, query: str, k: int) -> list[Hit]:
        hits = self._base.retrieve(query, max(k, self._candidates))
        if not hits:
            return []
        scores = self._reranker.score(query, [embedding_text(h.heading, h.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)[:k]
        return [hits[i].model_copy(update={"score": scores[i]}) for i in order]
