import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_eval_lab.cli import app
from rag_eval_lab.evals.dataset import (
    EvalQuestion,
    load_questions,
    missing_sections,
    section_span,
)

REPO_EVAL_SET = Path(__file__).resolve().parents[2] / "data" / "eval" / "questions.jsonl"

PAGE = (
    "# Query Parameters\n\nIntro.\n\n## Defaults\n\nDefault values.\n\n### Detail\n\nMore.\n\n"
    "## Required\n\nNo default.\n"
)


def question(**overrides: str) -> EvalQuestion:
    fields = {
        "id": "q001",
        "question": "How do I make a parameter required?",
        "answer": "Do not give it a default value.",
        "doc_id": "query.md",
        "section": "Query Parameters > Required",
        "area": "tutorial",
    }
    return EvalQuestion(**{**fields, **overrides})


def test_the_repo_eval_set_is_valid() -> None:
    questions = load_questions(REPO_EVAL_SET)
    assert len(questions) == 100
    assert {q.area for q in questions} <= {"tutorial", "advanced", "deployment", "how-to", "core"}
    assert len({(q.doc_id, q.section) for q in questions}) == 100


def test_section_span_includes_subsections_and_stops_at_a_sibling() -> None:
    span = section_span(PAGE, "Query Parameters > Defaults")
    assert span is not None
    text = PAGE[span[0] : span[1]]
    assert text.startswith("## Defaults")
    assert "### Detail" in text
    assert "## Required" not in text


def test_last_section_runs_to_the_end_of_the_page() -> None:
    span = section_span(PAGE, "Query Parameters > Required")
    assert span is not None and span[1] == len(PAGE)


def test_unknown_section_is_none() -> None:
    assert section_span(PAGE, "Query Parameters > Missing") is None


def test_missing_sections_are_reported() -> None:
    problems = missing_sections(
        [
            question(),
            question(id="q002", section="Query Parameters > Nope"),
            question(id="q003", doc_id="x.md"),
        ],
        {"query.md": PAGE},
    )
    assert problems == [
        "q002: no section 'Query Parameters > Nope' in query.md",
        "q003: page x.md is not in the corpus",
    ]


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(question().model_dump_json() for _ in range(2)) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)


def test_bad_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        question(id="question-1")


def test_check_evalset_command(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "query.md").write_text(PAGE)
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps(question().model_dump()) + "\n")
    runner = CliRunner()

    ok = runner.invoke(
        app, ["check-evalset", "--questions-file", str(questions), "--corpus", str(corpus)]
    )
    assert ok.exit_code == 0, ok.output
    assert "1 questions, every page and section found" in ok.output

    questions.write_text(
        json.dumps(question(section="Query Parameters > Nope").model_dump()) + "\n"
    )
    bad = runner.invoke(
        app, ["check-evalset", "--questions-file", str(questions), "--corpus", str(corpus)]
    )
    assert bad.exit_code == 1
    assert "no section" in bad.output
