"""Command-line entry point."""

from pathlib import Path
from statistics import median
from typing import Annotated

import typer

from rag_eval_lab import __version__
from rag_eval_lab.config import get_settings
from rag_eval_lab.ingest.chunking import Strategy, chunk_documents, make_chunker, write_chunks
from rag_eval_lab.ingest.corpus import FASTAPI_DOCS, fetch_corpus
from rag_eval_lab.ingest.loaders import load_markdown_dir
from rag_eval_lab.log import configure_logging

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
