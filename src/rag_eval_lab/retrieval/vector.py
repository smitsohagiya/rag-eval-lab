"""Nearest-neighbour search over chunk embeddings with pgvector."""

from rag_eval_lab.embeddings import Embedder
from rag_eval_lab.retrieval.base import HIT_COLUMNS, Hit, hits_from_rows
from rag_eval_lab.store import Connection

_QUERY = f"""
SELECT {HIT_COLUMNS}, 1 - (embedding <=> %(vector)s) AS score
FROM chunks
WHERE collection = %(collection)s
ORDER BY embedding <=> %(vector)s
LIMIT %(k)s
"""


class VectorRetriever:
    """Cosine similarity between the question's embedding and each chunk's embedding."""

    name = "vector"

    def __init__(
        self, conn: Connection, collection: str, embedder: Embedder, ef_search: int = 100
    ) -> None:
        self._conn = conn
        self._collection = collection
        self._embedder = embedder
        # A larger HNSW candidate list trades a little speed for recall.
        conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")

    def retrieve(self, query: str, k: int) -> list[Hit]:
        vector = self._embedder.embed_query(query)
        rows = self._conn.execute(
            _QUERY, {"vector": vector, "collection": self._collection, "k": k}
        ).fetchall()
        return hits_from_rows(rows)
