"""Turn saved runs into the markdown tables published in results/summary.md and the README."""

from collections.abc import Sequence

from rag_eval_lab.evals.metrics import hit_rate
from rag_eval_lab.evals.runner import RunResult

_HEADER = (
    "Chunking",
    "Retrieval",
    "Re-rank",
    "Hit@1",
    "Hit@3",
    "Hit@5",
    "Hit@10",
    "MRR@10",
    "p50 latency",
)


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def summary_table(runs: Sequence[RunResult]) -> str:
    """One row per configuration, best MRR@10 first."""
    lines = ["| " + " | ".join(_HEADER) + " |", "|" + " --- |" * len(_HEADER)]
    for run in sorted(runs, key=lambda r: (-r.metrics["mrr@10"], r.config.name)):
        m = run.metrics
        lines.append(
            f"| {run.config.strategy.value} | {run.config.retriever.value} "
            f"| {'yes' if run.config.rerank else 'no'} "
            f"| {_pct(m['hit@1'])} | {_pct(m['hit@3'])} | {_pct(m['hit@5'])} "
            f"| {_pct(m['hit@10'])} | {m['mrr@10']:.3f} | {m['latency_p50_ms']:.0f} ms |"
        )
    return "\n".join(lines)


def area_table(run: RunResult) -> str:
    """Hit@5 by area of the docs for one run."""
    areas = sorted({r.area for r in run.per_question})
    lines = ["| Area | Questions | Hit@5 |", "| --- | --- | --- |"]
    for area in areas:
        subset = [r for r in run.per_question if r.area == area]
        lines.append(f"| {area} | {len(subset)} | {_pct(hit_rate(subset, 5))} |")
    return "\n".join(lines)


def render_summary(runs: Sequence[RunResult]) -> str:
    """The full results/summary.md text."""
    if not runs:
        return "# Results\n\nNo runs yet. Run `rag-eval-lab eval-grid`.\n"
    best = max(runs, key=lambda r: r.metrics["mrr@10"])
    first = runs[0]
    return (
        "# Results\n\n"
        f"Corpus: {first.corpus}. Questions: {first.questions}. "
        f"Embedding model: `{first.embedding_model}`.\n\n"
        "Hit@k is the share of questions with a chunk from the answering section in the top k. "
        "MRR@10 is the mean of 1 / rank of the first such chunk (0 if it is not in the top 10).\n\n"
        f"{summary_table(runs)}\n\n"
        f"## Best configuration by area: {best.config.name}\n\n"
        f"{area_table(best)}\n"
    )
