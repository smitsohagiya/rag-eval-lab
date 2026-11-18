"""Cut documents into chunks: one per heading section, or fixed-size windows.

Both strategies pack whole markdown blocks, so fenced code is never split, and every chunk is an
exact slice of its page: ``document.text[chunk.start:chunk.end] == chunk.text``.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel

from rag_eval_lab.ingest.loaders import Document
from rag_eval_lab.ingest.markdown import Block, BlockKind, parse_blocks

HEADING_SEPARATOR = " > "


class Strategy(StrEnum):
    """Chunking strategies compared in the experiments."""

    HEADING = "heading"
    FIXED = "fixed"


class Chunk(BaseModel):
    """A retrievable piece of a page, with the heading path in effect where it starts."""

    chunk_id: str
    doc_id: str
    strategy: Strategy
    heading: str
    start: int
    end: int
    text: str


class Chunker(Protocol):
    """Anything that can cut one document into chunks."""

    def chunk(self, document: Document) -> list[Chunk]: ...


def heading_paths(blocks: list[Block]) -> list[str]:
    """For each block, the heading path in effect at that block, e.g. "Tutorial > Query"."""
    stack: list[tuple[int, str]] = []
    paths: list[str] = []
    for block in blocks:
        if block.kind is BlockKind.HEADING:
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.title))
        paths.append(HEADING_SEPARATOR.join(title for _, title in stack))
    return paths


def _make_chunk(
    document: Document, strategy: Strategy, index: int, start: int, end: int, heading: str
) -> Chunk:
    return Chunk(
        chunk_id=f"{document.doc_id}#{strategy.value}-{index}",
        doc_id=document.doc_id,
        strategy=strategy,
        heading=heading,
        start=start,
        end=end,
        text=document.text[start:end],
    )


class HeadingChunker:
    """One chunk per heading section. Sections that hold nothing but a heading are skipped."""

    strategy: ClassVar[Strategy] = Strategy.HEADING

    def chunk(self, document: Document) -> list[Chunk]:
        blocks = parse_blocks(document.text)
        paths = heading_paths(blocks)
        sections: list[list[int]] = []
        for index, block in enumerate(blocks):
            if block.kind is BlockKind.HEADING or not sections:
                sections.append([])
            sections[-1].append(index)

        chunks: list[Chunk] = []
        for members in sections:
            if all(blocks[i].kind is BlockKind.HEADING for i in members):
                continue
            first, last = blocks[members[0]], blocks[members[-1]]
            chunks.append(
                _make_chunk(
                    document, self.strategy, len(chunks), first.start, last.end, paths[members[0]]
                )
            )
        return chunks


@dataclass(frozen=True)
class FixedSizeChunker:
    """Pack consecutive blocks into windows of at most ``max_chars``, overlapping by up to
    ``overlap`` characters.

    Prose blocks longer than ``max_chars`` are split at whitespace. Code blocks are never split,
    so a code block longer than ``max_chars`` becomes one oversized chunk.
    """

    max_chars: int = 1200
    overlap: int = 200
    strategy: ClassVar[Strategy] = Strategy.FIXED

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if not 0 <= self.overlap < self.max_chars:
            raise ValueError("overlap must be at least 0 and smaller than max_chars")

    def chunk(self, document: Document) -> list[Chunk]:
        blocks = parse_blocks(document.text)
        pieces = [
            (piece, path)
            for block, path in zip(blocks, heading_paths(blocks), strict=True)
            for piece in self._split(document.text, block)
        ]

        chunks: list[Chunk] = []
        window: list[tuple[Block, str]] = []
        for piece in pieces:
            if window and _span([*window, piece]) > self.max_chars:
                chunks.append(self._emit(document, len(chunks), window))
                window = self._tail(window)
                if window and _span([*window, piece]) > self.max_chars:
                    window = []
            window.append(piece)
        if window:
            chunks.append(self._emit(document, len(chunks), window))
        return chunks

    def _split(self, text: str, block: Block) -> list[Block]:
        """Split a long prose block at whitespace. Code and short blocks come back unchanged."""
        if block.kind is BlockKind.CODE or block.end - block.start <= self.max_chars:
            return [block]
        pieces: list[Block] = []
        start = block.start
        while block.end - start > self.max_chars:
            window = text[start : start + self.max_chars]
            cut = max(window.rfind(" "), window.rfind("\n"))
            if cut <= 0:  # one very long word: cut it hard
                cut = len(window)
            pieces.append(Block(block.kind, start, start + cut))
            start += cut
            while start < block.end and text[start].isspace():
                start += 1
        if start < block.end:
            pieces.append(Block(block.kind, start, block.end))
        return pieces

    def _tail(self, window: list[tuple[Block, str]]) -> list[tuple[Block, str]]:
        """The trailing blocks of ``window`` that fit inside the overlap budget."""
        tail: list[tuple[Block, str]] = []
        for item in reversed(window):
            candidate = [item, *tail]
            if _span(candidate) > self.overlap:
                break
            tail = candidate
        return tail

    def _emit(self, document: Document, index: int, window: list[tuple[Block, str]]) -> Chunk:
        first, heading = window[0]
        return _make_chunk(document, self.strategy, index, first.start, window[-1][0].end, heading)


def _span(window: list[tuple[Block, str]]) -> int:
    return window[-1][0].end - window[0][0].start


def make_chunker(strategy: Strategy, *, max_chars: int = 1200, overlap: int = 200) -> Chunker:
    """Build the chunker for ``strategy``. The size options only apply to fixed-size chunks."""
    if strategy is Strategy.HEADING:
        return HeadingChunker()
    return FixedSizeChunker(max_chars=max_chars, overlap=overlap)


def chunk_documents(documents: Iterable[Document], chunker: Chunker) -> list[Chunk]:
    """Chunk every document, keeping document order."""
    return [chunk for document in documents for chunk in chunker.chunk(document)]


def write_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    """Write chunks as JSON lines, creating the parent directory. Returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
            count += 1
    return count


def read_chunks(path: Path) -> list[Chunk]:
    """Read chunks written by :func:`write_chunks`."""
    with path.open(encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]
