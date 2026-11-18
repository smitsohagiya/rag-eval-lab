from collections.abc import Iterator

import pytest
from typer.testing import CliRunner

from rag_eval_lab import cli, store
from rag_eval_lab.config import get_settings
from rag_eval_lab.embeddings import Embedder, HashingEmbedder, embedding_text
from rag_eval_lab.ingest.chunking import HeadingChunker, chunk_documents
from rag_eval_lab.ingest.loaders import Document
from rag_eval_lab.retrieval import RetrieverName, build_retriever

pytestmark = pytest.mark.integration

EMBEDDER = HashingEmbedder(dim=256)
DOCS = [
    Document(
        doc_id="tutorial/query-params.md",
        title="Query Parameters",
        text="# Query Parameters\n\nFunction parameters that are not part of the path are "
        "interpreted as query parameters.\n",
    ),
    Document(
        doc_id="deployment/docker.md",
        title="FastAPI in Containers - Docker",
        text="# FastAPI in Containers - Docker\n\nBuild a Docker image for your app and run "
        "it in a container.\n",
    ),
    Document(
        doc_id="tutorial/path-params.md",
        title="Path Parameters",
        text="# Path Parameters\n\nDeclare path parameters with the same syntax as Python "
        "format strings.\n",
    ),
]


@pytest.fixture
def indexed(conn: store.Connection) -> Iterator[store.Connection]:
    chunks = chunk_documents(DOCS, HeadingChunker())
    vectors = EMBEDDER.embed_documents([embedding_text(c.heading, c.text) for c in chunks])
    store.ensure_schema(conn, EMBEDDER.dim)
    store.replace_collection(conn, "heading", chunks, vectors)
    yield conn


@pytest.mark.parametrize("name", list(RetrieverName))
def test_each_retriever_finds_the_docker_page(
    indexed: store.Connection, name: RetrieverName
) -> None:
    retriever = build_retriever(name, indexed, "heading", EMBEDDER)
    hits = retriever.retrieve("run my app in a docker container", k=2)
    assert hits[0].doc_id == "deployment/docker.md"
    assert retriever.name == name.value


def test_keyword_search_matches_any_word_not_every_word(indexed: store.Connection) -> None:
    hits = build_retriever(RetrieverName.KEYWORD, indexed, "heading", EMBEDDER).retrieve(
        "query parameters spaceship", k=5
    )
    assert hits and hits[0].doc_id == "tutorial/query-params.md"


def test_a_question_of_only_stopwords_returns_nothing(indexed: store.Connection) -> None:
    retriever = build_retriever(RetrieverName.KEYWORD, indexed, "heading", EMBEDDER)
    assert retriever.retrieve("the of and", k=5) == []


def test_hybrid_scores_are_fused_ranks(indexed: store.Connection) -> None:
    hits = build_retriever(RetrieverName.HYBRID, indexed, "heading", EMBEDDER).retrieve(
        "docker container", k=3
    )
    assert hits[0].score == pytest.approx(2 / 61)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_command(
    indexed: store.Connection, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_DATABASE_URL", database_url)
    get_settings.cache_clear()

    def fake_embedder(model_name: str) -> Embedder:
        return EMBEDDER

    monkeypatch.setattr(cli, "load_embedder", fake_embedder)
    result = CliRunner().invoke(cli.app, ["search", "docker container", "-k", "2"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0].endswith(
        "deployment/docker.md :: FastAPI in Containers - Docker"
    )
