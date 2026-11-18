"""Postgres storage. One table holds every chunk with its embedding (pgvector) and a generated
full-text column, so vector, keyword and hybrid search all read the same rows (ADR-0001)."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import TupleRow

from rag_eval_lab.embeddings import Vector
from rag_eval_lab.ingest.chunking import Chunk

Connection = psycopg.Connection[TupleRow]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    collection   text    NOT NULL,
    chunk_id     text    NOT NULL,
    doc_id       text    NOT NULL,
    heading      text    NOT NULL,
    start_offset integer NOT NULL,
    end_offset   integer NOT NULL,
    text         text    NOT NULL,
    embedding    vector({dim}) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || text)) STORED,
    PRIMARY KEY (collection, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
"""

_INSERT = """
INSERT INTO chunks
    (collection, chunk_id, doc_id, heading, start_offset, end_offset, text, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


@contextmanager
def connect(url: str) -> Iterator[Connection]:
    """Open an autocommit connection with the pgvector extension and type adapters ready.

    Writes that must be atomic use ``conn.transaction()`` explicitly.
    """
    with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        yield conn


def ensure_schema(conn: Connection, dim: int) -> None:
    """Create the table and indexes if needed. Refuses to mix vectors of different sizes."""
    row = conn.execute(
        # Look only in the current schema, which is where CREATE TABLE would put the table.
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = to_regclass(quote_ident(current_schema()) || '.chunks') "
        "AND attname = 'embedding'"
    ).fetchone()
    if row is not None and row[0] != dim:
        raise ValueError(
            f"chunks.embedding holds {row[0]}-dimension vectors but the embedder makes {dim}; "
            "drop the chunks table before indexing with a different model"
        )
    conn.execute(_SCHEMA.format(dim=int(dim)))


def replace_collection(
    conn: Connection, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]
) -> int:
    """Swap one collection's rows for ``chunks`` in a single transaction.

    Re-running with the same input leaves the table in the same state, so indexing is
    idempotent. Returns the number of rows now in the collection.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
    rows = [
        (collection, c.chunk_id, c.doc_id, c.heading, c.start, c.end, c.text, vector)
        for c, vector in zip(chunks, vectors, strict=True)
    ]
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute("DELETE FROM chunks WHERE collection = %s", (collection,))
        cursor.executemany(_INSERT, rows)
    return count(conn, collection)


def count(conn: Connection, collection: str) -> int:
    """Number of chunks stored for ``collection``."""
    row = conn.execute(
        "SELECT count(*) FROM chunks WHERE collection = %s", (collection,)
    ).fetchone()
    return int(row[0]) if row else 0
