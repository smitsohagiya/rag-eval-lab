import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_eval_lab import cli, store
from rag_eval_lab.config import get_settings
from rag_eval_lab.embeddings import Embedder, HashingEmbedder, embedding_text
from rag_eval_lab.ingest.chunking import HeadingChunker, chunk_documents
from rag_eval_lab.ingest.loaders import load_markdown_dir

pytestmark = pytest.mark.integration

EMBEDDER = HashingEmbedder(dim=256)
PAGES = {
    "docker.md": "# Docker\n\n## Build an image\n\nWrite a Dockerfile and build the image.\n",
    "query.md": "# Query Parameters\n\n## Required\n\n"
    "Leave out the default value to make it required.\n",
}
QUESTIONS = [
    {
        "id": "q001",
        "question": "how do I build a docker image",
        "answer": "Write a Dockerfile.",
        "doc_id": "docker.md",
        "section": "Docker > Build an image",
        "area": "deployment",
    },
    {
        "id": "q002",
        "question": "make a query parameter required",
        "answer": "No default value.",
        "doc_id": "query.md",
        "section": "Query Parameters > Required",
        "area": "tutorial",
    },
]


def test_eval_grid_writes_runs_and_summary(
    conn: store.Connection, database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    corpus = data / "corpus" / "fastapi"
    corpus.mkdir(parents=True)
    for name, text in PAGES.items():
        (corpus / name).write_text(text)
    (data / "eval").mkdir()
    (data / "eval" / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))

    for strategy in ("heading", "fixed"):
        chunks = chunk_documents(load_markdown_dir(corpus), HeadingChunker())
        vectors = EMBEDDER.embed_documents([embedding_text(c.heading, c.text) for c in chunks])
        store.ensure_schema(conn, EMBEDDER.dim)
        store.replace_collection(conn, strategy, chunks, vectors)

    monkeypatch.setenv("APP_DATABASE_URL", database_url)
    monkeypatch.setenv("APP_DATA_DIR", str(data))
    monkeypatch.setenv("APP_RESULTS_DIR", str(tmp_path / "results"))
    get_settings.cache_clear()

    def fake_embedder(model_name: str) -> Embedder:
        return EMBEDDER

    class FirstWordReranker:
        name = "first-word"

        def score(self, query: str, texts: Sequence[str]) -> list[float]:
            return [float(query.split()[0] in text.lower()) for text in texts]

    def fake_reranker(model_name: str) -> FirstWordReranker:
        return FirstWordReranker()

    monkeypatch.setattr(cli, "load_embedder", fake_embedder)
    monkeypatch.setattr(cli, "load_reranker", fake_reranker)
    result = CliRunner().invoke(cli.app, ["eval-grid"])

    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "results" / "runs").glob("*.json"))) == 12
    summary = (tmp_path / "results" / "summary.md").read_text()
    assert "| Chunking | Retrieval |" in summary
    assert "heading-keyword" in result.output
    assert "fixed-hybrid-rerank" in result.output
