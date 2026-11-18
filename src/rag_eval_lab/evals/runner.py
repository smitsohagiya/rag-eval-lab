"""Run every eval question through one retrieval configuration and record the results."""

import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from rag_eval_lab.evals.dataset import EvalQuestion, section_span
from rag_eval_lab.evals.metrics import QuestionResult, first_relevant_rank, summarize
from rag_eval_lab.ingest.chunking import Strategy
from rag_eval_lab.retrieval.base import Retriever, RetrieverName


class RunConfig(BaseModel):
    """One point in the experiment grid."""

    strategy: Strategy
    retriever: RetrieverName
    rerank: bool = False
    k: int = 10

    @property
    def name(self) -> str:
        suffix = "-rerank" if self.rerank else ""
        return f"{self.strategy.value}-{self.retriever.value}{suffix}"


class RunResult(BaseModel):
    """Everything needed to report on, compare or audit one run."""

    config: RunConfig
    corpus: str
    embedding_model: str
    started_at: str
    questions: int
    metrics: dict[str, float]
    per_question: list[QuestionResult]


def run_eval(
    questions: Sequence[EvalQuestion],
    pages: Mapping[str, str],
    retriever: Retriever,
    config: RunConfig,
    *,
    corpus: str,
    embedding_model: str,
) -> RunResult:
    """Ask every question, find the rank of the first relevant chunk and time each search."""
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    results: list[QuestionResult] = []
    for question in questions:
        span = section_span(pages[question.doc_id], question.section)
        if span is None:
            raise ValueError(f"{question.id}: section not found; run check-evalset")
        began = time.perf_counter()
        hits = retriever.retrieve(question.question, config.k)
        latency_ms = (time.perf_counter() - began) * 1000
        results.append(
            QuestionResult(
                id=question.id,
                area=question.area,
                first_relevant_rank=first_relevant_rank(hits, question.doc_id, span),
                latency_ms=round(latency_ms, 2),
            )
        )
    return RunResult(
        config=config,
        corpus=corpus,
        embedding_model=embedding_model,
        started_at=started_at,
        questions=len(results),
        metrics=summarize(results),
        per_question=results,
    )


def save_run(result: RunResult, runs_dir: Path) -> Path:
    """Write ``result`` to ``<runs_dir>/<config name>.json`` and return the path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{result.config.name}.json"
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_runs(runs_dir: Path) -> list[RunResult]:
    """Read every saved run, sorted by name."""
    return [
        RunResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(runs_dir.glob("*.json"))
    ]
