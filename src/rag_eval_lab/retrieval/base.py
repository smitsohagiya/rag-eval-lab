"""What every retriever returns, and the interface they share."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel


class RetrieverName(StrEnum):
    """Retrieval methods compared in the experiments."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class Hit(BaseModel):
    """One retrieved chunk. ``score`` is only comparable within a single retriever's results."""

    chunk_id: str
    doc_id: str
    heading: str
    start: int
    end: int
    text: str
    score: float


class Retriever(Protocol):
    """Returns the ``k`` best chunks for a question, best first."""

    name: str

    def retrieve(self, query: str, k: int) -> list[Hit]: ...


HIT_COLUMNS = "chunk_id, doc_id, heading, start_offset, end_offset, text"


def hits_from_rows(rows: Sequence[Sequence[Any]]) -> list[Hit]:
    """Build hits from rows of ``HIT_COLUMNS`` followed by a score column."""
    return [
        Hit(
            chunk_id=row[0],
            doc_id=row[1],
            heading=row[2],
            start=row[3],
            end=row[4],
            text=row[5],
            score=float(row[6]),
        )
        for row in rows
    ]
