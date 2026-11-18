"""The eval set: questions with a reference answer and the section of the docs that answers them."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from rag_eval_lab.ingest.chunking import heading_paths
from rag_eval_lab.ingest.markdown import BlockKind, parse_blocks


class EvalQuestion(BaseModel):
    """One question, with the page and heading path of the section that answers it."""

    id: str = Field(pattern=r"^q\d{3}$")
    question: str = Field(min_length=10)
    answer: str = Field(min_length=10)
    doc_id: str
    section: str
    area: str


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read the eval set from JSON lines. Duplicate ids are an error."""
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [EvalQuestion.model_validate_json(line) for line in lines if line.strip()]
    ids = [question.id for question in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} has duplicate question ids")
    return questions


def section_span(text: str, section: str) -> tuple[int, int] | None:
    """Character span of the section whose heading path is ``section``, subsections included.

    The span runs from the heading line to the next heading of the same or a higher level, or
    to the end of the page. Returns None when the page has no such heading path.
    """
    blocks = parse_blocks(text)
    for index, (block, path) in enumerate(zip(blocks, heading_paths(blocks), strict=True)):
        if block.kind is BlockKind.HEADING and path == section:
            end = len(text)
            for later in blocks[index + 1 :]:
                if later.kind is BlockKind.HEADING and later.level <= block.level:
                    end = later.start
                    break
            return block.start, end
    return None


def missing_sections(questions: Sequence[EvalQuestion], pages: Mapping[str, str]) -> list[str]:
    """Describe every question whose page or section is not in ``pages`` (doc_id -> text)."""
    problems: list[str] = []
    for question in questions:
        text = pages.get(question.doc_id)
        if text is None:
            problems.append(f"{question.id}: page {question.doc_id} is not in the corpus")
        elif section_span(text, question.section) is None:
            problems.append(f"{question.id}: no section '{question.section}' in {question.doc_id}")
    return problems
