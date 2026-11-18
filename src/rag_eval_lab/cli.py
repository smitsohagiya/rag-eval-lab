"""Command-line entry point."""

import time
from pathlib import Path
from statistics import median
from typing import Annotated

import typer

from rag_eval_lab import __version__, store
from rag_eval_lab.config import Settings, get_settings
from rag_eval_lab.embeddings import Embedder, FastEmbedEmbedder, embedding_text
from rag_eval_lab.evals.dataset import EvalQuestion, load_questions, missing_sections
from rag_eval_lab.evals.report import render_summary
from rag_eval_lab.evals.runner import RunConfig, RunResult, load_runs, run_eval, save_run
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
from rag_eval_lab.retrieval import CrossEncoderReranker, Reranker, RetrieverName, build_retriever

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


def load_reranker(model_name: str) -> Reranker:
    """Build the cross-encoder used for re-ranking. Tests replace this function."""
    return CrossEncoderReranker(model_name)


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
    rerank: Annotated[bool, typer.Option(help="Re-rank candidates with a cross-encoder.")] = False,
) -> None:
    """Show the top chunks for a question, with their pages and headings."""
    settings = get_settings()
    embedder = load_embedder(settings.embedding_model)
    reranker = load_reranker(settings.rerank_model) if rerank else None
    with store.connect(settings.database_url) as conn:
        found = build_retriever(
            retriever, conn, strategy.value, embedder, reranker, settings.rerank_candidates
        )
        hits = found.retrieve(query, k)
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


def _pages(settings: Settings) -> dict[str, str]:
    return {doc.doc_id: doc.text for doc in load_markdown_dir(settings.corpus_dir)}


def _evaluate(
    config: RunConfig,
    conn: store.Connection,
    embedder: Embedder,
    questions: list[EvalQuestion],
    pages: dict[str, str],
    reranker: Reranker | None = None,
) -> RunResult:
    if config.rerank and reranker is None:
        raise ValueError(f"{config.name} needs a reranker")
    retriever = build_retriever(
        config.retriever,
        conn,
        config.strategy.value,
        embedder,
        reranker if config.rerank else None,
        get_settings().rerank_candidates,
    )
    return run_eval(
        questions,
        pages,
        retriever,
        config,
        corpus=f"FastAPI {FASTAPI_DOCS.version} docs",
        embedding_model=embedder.name,
    )


def _describe(result: RunResult) -> str:
    m = result.metrics
    return (
        f"{result.config.name:22} hit@1 {m['hit@1']:4.0%}  hit@5 {m['hit@5']:4.0%}  "
        f"hit@10 {m['hit@10']:4.0%}  mrr@10 {m['mrr@10']:.3f}  p50 {m['latency_p50_ms']:.0f} ms"
    )


@app.command("eval")
def eval_command(
    strategy: Annotated[Strategy, typer.Option(help="Indexed collection to search.")] = (
        Strategy.HEADING
    ),
    retriever: Annotated[RetrieverName, typer.Option(help="Retrieval method.")] = (
        RetrieverName.HYBRID
    ),
    k: Annotated[int, typer.Option("-k", help="How many chunks to retrieve.")] = 10,
    rerank: Annotated[bool, typer.Option(help="Re-rank candidates with a cross-encoder.")] = False,
) -> None:
    """Score one configuration on the eval set and save the run."""
    settings = get_settings()
    questions, pages = load_questions(settings.eval_file), _pages(settings)
    embedder = load_embedder(settings.embedding_model)
    reranker = load_reranker(settings.rerank_model) if rerank else None
    config = RunConfig(strategy=strategy, retriever=retriever, rerank=rerank, k=k)
    with store.connect(settings.database_url) as conn:
        result = _evaluate(config, conn, embedder, questions, pages, reranker)
    path = save_run(result, settings.results_dir / "runs")
    typer.echo(f"{_describe(result)}  -> {path}")


@app.command("eval-grid")
def eval_grid(
    rerank: Annotated[bool, typer.Option(help="Also run every pair with re-ranking.")] = True,
) -> None:
    """Score every chunking strategy with every retrieval method, then write the summary."""
    settings = get_settings()
    questions, pages = load_questions(settings.eval_file), _pages(settings)
    embedder = load_embedder(settings.embedding_model)
    reranker = load_reranker(settings.rerank_model) if rerank else None
    variants = [False, True] if rerank else [False]
    with store.connect(settings.database_url) as conn:
        for strategy in Strategy:
            for name in RetrieverName:
                for with_rerank in variants:
                    config = RunConfig(strategy=strategy, retriever=name, rerank=with_rerank)
                    result = _evaluate(config, conn, embedder, questions, pages, reranker)
                    save_run(result, settings.results_dir / "runs")
                    typer.echo(_describe(result))
    report()


@app.command()
def report() -> None:
    """Write results/summary.md from the saved runs and print it."""
    settings = get_settings()
    text = render_summary(load_runs(settings.results_dir / "runs"))
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "summary.md").write_text(text, encoding="utf-8")
    typer.echo(text)
