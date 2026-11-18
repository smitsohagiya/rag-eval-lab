"""Split markdown into blocks without breaking fenced code.

A block is a heading line, a fenced code block (kept whole, blank lines included) or a run of
consecutive non-blank lines. Offsets point into the original text, so
``text[block.start:block.end]`` is exactly the block.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}\s*$")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")


class BlockKind(StrEnum):
    """What a block contains."""

    HEADING = "heading"
    CODE = "code"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class Block:
    """A span of the source text. ``level`` and ``title`` are only set for headings."""

    kind: BlockKind
    start: int
    end: int
    level: int = 0
    title: str = ""


def _closes_fence(line: str, marker: str) -> bool:
    body = line.rstrip("\r\n")
    indent = len(body) - len(body.lstrip(" "))
    fence = body.strip()
    return indent <= 3 and len(fence) >= len(marker) and fence == marker[0] * len(fence)


def _is_paragraph_line(line: str) -> bool:
    body = line.rstrip("\r\n")
    return bool(body.strip()) and not _FENCE_OPEN.match(body) and not _HEADING.match(body)


def _clean_title(raw: str) -> str:
    """Drop the ``{ #anchor }`` suffix that the FastAPI docs add to headings."""
    return _ANCHOR.sub("", raw).strip()


def parse_blocks(text: str) -> list[Block]:
    """Return the blocks of ``text`` in document order."""
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    def end_of(index: int) -> int:
        return starts[index] + len(lines[index].rstrip("\r\n"))

    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        if not line.strip():
            i += 1
            continue
        if fence := _FENCE_OPEN.match(line):
            marker = fence.group(1)
            j = i + 1
            while j < len(lines) and not _closes_fence(lines[j], marker):
                j += 1
            last = min(j, len(lines) - 1)  # an unclosed fence runs to the end of the text
            blocks.append(Block(BlockKind.CODE, starts[i], end_of(last)))
            i = last + 1
            continue
        if heading := _HEADING.match(line):
            blocks.append(
                Block(
                    BlockKind.HEADING,
                    starts[i],
                    end_of(i),
                    level=len(heading.group(1)),
                    title=_clean_title(heading.group(2)),
                )
            )
            i += 1
            continue
        j = i
        while j + 1 < len(lines) and _is_paragraph_line(lines[j + 1]):
            j += 1
        blocks.append(Block(BlockKind.TEXT, starts[i], end_of(j)))
        i = j + 1
    return blocks


def first_heading(text: str) -> str | None:
    """Return the title of the first level-1 heading, if there is one."""
    for block in parse_blocks(text):
        if block.kind is BlockKind.HEADING and block.level == 1:
            return block.title
    return None
