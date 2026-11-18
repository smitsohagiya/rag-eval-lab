"""Build a retriever by name for one indexed collection."""

from rag_eval_lab.embeddings import Embedder
from rag_eval_lab.retrieval.base import Retriever, RetrieverName
from rag_eval_lab.retrieval.hybrid import HybridRetriever
from rag_eval_lab.retrieval.keyword import KeywordRetriever
from rag_eval_lab.retrieval.vector import VectorRetriever
from rag_eval_lab.store import Connection


def build_retriever(
    name: RetrieverName, conn: Connection, collection: str, embedder: Embedder
) -> Retriever:
    """Return the retriever called ``name``, reading from ``collection``."""
    if name is RetrieverName.VECTOR:
        return VectorRetriever(conn, collection, embedder)
    if name is RetrieverName.KEYWORD:
        return KeywordRetriever(conn, collection)
    return HybridRetriever(
        [VectorRetriever(conn, collection, embedder), KeywordRetriever(conn, collection)]
    )
