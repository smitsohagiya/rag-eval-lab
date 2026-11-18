from pathlib import Path

import pytest

from rag_eval_lab.evals.dataset import EvalQuestion
from rag_eval_lab.evals.report import render_summary
from rag_eval_lab.evals.runner import RunConfig, load_runs, run_eval, save_run
from rag_eval_lab.ingest.chunking import Strategy
from rag_eval_lab.retrieval.base import Hit, RetrieverName

PAGE = "# Guide\n\nIntro.\n\n## Install\n\npip install x\n\n## Use\n\nRun it.\n"
INSTALL_START = PAGE.index("## Install")


class FakeRetriever:
    name = "fake"

    def __init__(self, hits_by_question: dict[str, list[Hit]]) -> None:
        self._hits = hits_by_question

    def retrieve(self, query: str, k: int) -> list[Hit]:
        return self._hits.get(query, [])[:k]


def hit(start: int, end: int, doc_id: str = "guide.md") -> Hit:
    return Hit(
        chunk_id=f"{doc_id}#{start}",
        doc_id=doc_id,
        heading="",
        start=start,
        end=end,
        text="",
        score=1.0,
    )


def question(qid: str, text: str, section: str) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        question=text,
        answer="A reference answer.",
        doc_id="guide.md",
        section=section,
        area="tutorial",
    )


QUESTIONS = [
    question("q001", "how to install it", "Guide > Install"),
    question("q002", "how to use it", "Guide > Use"),
]
RETRIEVER = FakeRetriever(
    {
        "how to install it": [hit(INSTALL_START, INSTALL_START + 10)],
        "how to use it": [hit(0, 5), hit(INSTALL_START, INSTALL_START + 10)],
    }
)
CONFIG = RunConfig(strategy=Strategy.HEADING, retriever=RetrieverName.HYBRID)


def test_run_eval_scores_each_question() -> None:
    result = run_eval(
        QUESTIONS, {"guide.md": PAGE}, RETRIEVER, CONFIG, corpus="test", embedding_model="fake"
    )
    ranks = {r.id: r.first_relevant_rank for r in result.per_question}
    assert ranks == {"q001": 1, "q002": None}
    assert result.metrics["hit@1"] == 0.5
    assert result.config.name == "heading-hybrid"


def test_unknown_section_stops_the_run() -> None:
    bad = [question("q001", "where is the missing part?", "Guide > Missing")]
    with pytest.raises(ValueError, match="check-evalset"):
        run_eval(bad, {"guide.md": PAGE}, RETRIEVER, CONFIG, corpus="test", embedding_model="fake")


def test_runs_round_trip_and_render(tmp_path: Path) -> None:
    result = run_eval(
        QUESTIONS,
        {"guide.md": PAGE},
        RETRIEVER,
        CONFIG,
        corpus="test corpus",
        embedding_model="fake",
    )
    path = save_run(result, tmp_path / "runs")
    assert path.name == "heading-hybrid.json"
    runs = load_runs(tmp_path / "runs")
    assert runs == [result]
    summary = render_summary(runs)
    assert "| heading | hybrid | no | 50% |" in summary
    assert "Best configuration by area: heading-hybrid" in summary
    assert "Corpus: test corpus." in summary


def test_empty_summary() -> None:
    assert "No runs yet" in render_summary([])


def test_rerank_configs_have_their_own_name() -> None:
    assert (
        RunConfig(strategy=Strategy.FIXED, retriever=RetrieverName.VECTOR, rerank=True).name
        == "fixed-vector-rerank"
    )
