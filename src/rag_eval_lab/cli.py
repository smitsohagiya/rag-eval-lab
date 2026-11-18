"""Command-line entry point."""

import time
from pathlib import Path
from statistics import median
from typing import Annotated

import typer

from rag_eval_lab import __version__, store
from rag_eval_lab.config import get_settings
from rag_eval_lab.embeddings import Embedder, FastEmbedEmbedder, embedding_text
from rag_eval_lab.evals.dataset import load_questions, missing_sections
from rag_eval_lab.ingest.chunking import (
    Strategy,
    chunk_documents,
    make_chunker,
    read_chunks,
    write_chunks,
)
from rag_eval_lab.ingest.corpus import FASTAPI_DOCS, fetch_corpus
from rag_eval_lab.ingest.loaders import load_markdown_dir
from rag_eval_lab.log import configure_logging
from rag_eval_lab.retrieval import RetrieverName, build_retriever

app = typer.Typer(help="Run RAG retrieval experiments and evaluations.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Set up logging before any command runs."""
    configure_logging(get_settings().log_level)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command("fetch-corpus")
def fetch_corpus_command(
    dest: Annotated[Path | None, typer.Option(help="Where to put the pages.")] = None,
    force: Annotated[bool, typer.Option(help="Replace an existing copy.")] = False,
) -> None:
    """Download the pinned FastAPI docs (English pages only) and verify the checksum."""
    target = dest or get_settings().corpus_dir
    count = fetch_corpus(FASTAPI_DOCS, target, force=force)
    typer.echo(f"Fetched {count} pages of FastAPI {FASTAPI_DOCS.version} into {target}")


@app.command()
def ingest(
    strategy: Annotated[Strategy, typer.Option(help="How to cut pages into chunks.")] = (
        Strategy.HEADING
    ),
    corpus: Annotated[Path | None, typer.Option(help="Directory of markdown pages.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output JSONL file.")] = None,
    max_chars: Annotated[int, typer.Option(help="Fixed strategy: largest chunk size.")] = 1200,
    overlap: Annotated[int, typer.Option(help="Fixed strategy: overlap between chunks.")] = 200,
) -> None:
    """Cut the corpus into chunks and write them as JSON lines."""
    settings = get_settings()
    documents = load_markdown_dir(corpus or settings.corpus_dir)
    chunker = make_chunker(strategy, max_chars=max_chars, overlap=overlap)
    chunks = chunk_documents(documents, chunker)
    target = out or settings.chunks_dir / f"{strategy.value}.jsonl"
    write_chunks(chunks, target)
    sizes = [len(chunk.text) for chunk in chunks]
    typical = median(sizes) if sizes else 0
    typer.echo(
        f"{len(documents)} pages -> {len(chunks)} chunks ({strategy.value}, "
        f"median {typical:.0f} chars) written to {target}"
    )


def load_embedder(model_name: str) -> Embedder:
    """Build the embedder used for indexing and queries. Tests replace this function."""
    return FastEmbedEmbedder(model_name)


@app.command()
def index(
    strategy: Annotated[Strategy, typer.Option(help="Which chunk file to index.")] = (
        Strategy.HEADING
    ),
    chunks_file: Annotated[Path | None, typer.Option(help="Chunk JSONL file to read.")] = None,
) -> None:
    """Embed one strategy's chunks and store them in Postgres, replacing that collection."""
    settings = get_settings()
    chunks = read_chunks(chunks_file or settings.chunks_dir / f"{strategy.value}.jsonl")
    started = time.perf_counter()
    embedder = load_embedder(settings.embedding_model)
    vectors = embedder.embed_documents([embedding_text(c.heading, c.text) for c in chunks])
    with store.connect(settings.database_url) as conn:
        store.ensure_schema(conn, embedder.dim)
        stored = store.replace_collection(conn, strategy.value, chunks, vectors)
    typer.echo(
        f"Indexed {stored} chunks into '{strategy.value}' with {embedder.name} "
        f"({embedder.dim} dims) in {time.perf_counter() - started:.0f}s"
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="The question to search for.")],
    strategy: Annotated[Strategy, typer.Option(help="Which indexed collection to search.")] = (
        Strategy.HEADING
    ),
    retriever: Annotated[RetrieverName, typer.Option(help="Retrieval method.")] = (
        RetrieverName.HYBRID
    ),
    k: Annotated[int, typer.Option("-k", help="How many chunks to show.")] = 5,
) -> None:
    """Show the top chunks for a question, with their pages and headings."""
    settings = get_settings()
    embedder = load_embedder(settings.embedding_model)
    with store.connect(settings.database_url) as conn:
        hits = build_retriever(retriever, conn, strategy.value, embedder).retrieve(query, k)
    if not hits:
        typer.echo("No matching chunks.")
    for rank, hit in enumerate(hits, start=1):
        typer.echo(f"{rank}. [{hit.score:.3f}] {hit.doc_id} :: {hit.heading}")


@app.command("check-evalset")
def check_evalset(
    questions_file: Annotated[Path | None, typer.Option(help="Eval set to check.")] = None,
    corpus: Annotated[Path | None, typer.Option(help="Directory of markdown pages.")] = None,
) -> None:
    """Check that every eval question points at a page and section that exist in the corpus."""
    settings = get_settings()
    questions = load_questions(questions_file or settings.eval_file)
    pages = {doc.doc_id: doc.text for doc in load_markdown_dir(corpus or settings.corpus_dir)}
    problems = missing_sections(questions, pages)
    for problem in problems:
        typer.echo(problem)
    if problems:
        raise typer.Exit(code=1)
    typer.echo(f"{len(questions)} questions, every page and section found")
