"""Keyword search with Postgres full-text search."""

from rag_eval_lab.retrieval.base import HIT_COLUMNS, Hit, hits_from_rows
from rag_eval_lab.store import Connection

# plainto_tsquery joins the question's words with AND, which misses chunks that do not contain
# every word. Swapping the operators for OR lets ts_rank_cd reward chunks that match more of them.
_QUERY = f"""
WITH q AS (
    SELECT replace(plainto_tsquery('english', %(query)s)::text, '&', '|')::tsquery AS query
)
SELECT {HIT_COLUMNS}, ts_rank_cd(tsv, q.query, 32) AS score
FROM chunks, q
WHERE collection = %(collection)s AND tsv @@ q.query
ORDER BY score DESC, chunk_id
LIMIT %(k)s
"""


class KeywordRetriever:
    """Ranks chunks by how well their words cover the question's words (``ts_rank_cd``)."""

    name = "keyword"

    def __init__(self, conn: Connection, collection: str) -> None:
        self._conn = conn
        self._collection = collection

    def retrieve(self, query: str, k: int) -> list[Hit]:
        rows = self._conn.execute(
            _QUERY, {"query": query, "collection": self._collection, "k": k}
        ).fetchall()
        return hits_from_rows(rows)
