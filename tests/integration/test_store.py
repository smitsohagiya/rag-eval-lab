from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_eval_lab import cli, store
from rag_eval_lab.config import get_settings
from rag_eval_lab.embeddings import Embedder, HashingEmbedder, Vector
from rag_eval_lab.ingest.chunking import HeadingChunker, write_chunks
from rag_eval_lab.ingest.loaders import Document

pytestmark = pytest.mark.integration

DOC = Document(
    doc_id="tutorial/query-params.md",
    title="Query Parameters",
    text="# Query Parameters\n\nIntro.\n\n## Defaults\n\nUse a default value.\n\n"
    "## Optional parameters\n\nSet the default to None.\n",
)
CHUNKS = HeadingChunker().chunk(DOC)
EMBEDDER = HashingEmbedder(dim=64)


def vectors() -> list[Vector]:
    return EMBEDDER.embed_documents([chunk.text for chunk in CHUNKS])


def test_indexing_twice_does_not_duplicate_rows(conn: store.Connection) -> None:
    store.ensure_schema(conn, EMBEDDER.dim)
    assert store.replace_collection(conn, "heading", CHUNKS, vectors()) == len(CHUNKS)
    assert store.replace_collection(conn, "heading", CHUNKS, vectors()) == len(CHUNKS)


def test_collections_are_kept_apart(conn: store.Connection) -> None:
    store.ensure_schema(conn, EMBEDDER.dim)
    store.replace_collection(conn, "heading", CHUNKS, vectors())
    store.replace_collection(conn, "fixed", CHUNKS[:1], vectors()[:1])
    assert store.count(conn, "heading") == len(CHUNKS)
    assert store.count(conn, "fixed") == 1


def test_full_text_column_is_generated(conn: store.Connection) -> None:
    store.ensure_schema(conn, EMBEDDER.dim)
    store.replace_collection(conn, "heading", CHUNKS, vectors())
    row = conn.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('english', 'optional')"
    ).fetchone()
    assert row is not None and row[0] == 1


def test_a_different_vector_size_is_rejected(conn: store.Connection) -> None:
    store.ensure_schema(conn, 64)
    with pytest.raises(ValueError, match="64-dimension"):
        store.ensure_schema(conn, 32)


def test_mismatched_inputs_are_rejected(conn: store.Connection) -> None:
    store.ensure_schema(conn, EMBEDDER.dim)
    with pytest.raises(ValueError, match="chunks but"):
        store.replace_collection(conn, "heading", CHUNKS, vectors()[:1])


def test_index_command(database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chunks_file = tmp_path / "heading.jsonl"
    write_chunks(CHUNKS, chunks_file)
    monkeypatch.setenv("APP_DATABASE_URL", database_url)
    get_settings.cache_clear()

    def fake_embedder(model_name: str) -> Embedder:
        return EMBEDDER

    monkeypatch.setattr(cli, "load_embedder", fake_embedder)
    result = CliRunner().invoke(
        cli.app, ["index", "--strategy", "heading", "--chunks-file", str(chunks_file)]
    )
    assert result.exit_code == 0, result.output
    assert f"Indexed {len(CHUNKS)} chunks into 'heading'" in result.output
