from itertools import pairwise
from pathlib import Path

import pytest

from rag_eval_lab.ingest.chunking import (
    FixedSizeChunker,
    HeadingChunker,
    Strategy,
    chunk_documents,
    make_chunker,
    read_chunks,
    write_chunks,
)
from rag_eval_lab.ingest.loaders import Document

FENCE = "`" * 3


def doc(text: str, doc_id: str = "guide.md") -> Document:
    return Document(doc_id=doc_id, title=doc_id, text=text)


GUIDE = doc(
    "# Guide\n\nIntro.\n\n## Install\n\npip install x\n\n### Linux\n\napt install x\n\n"
    "## Use\n\nRun it.\n"
)


def test_heading_chunker_makes_one_chunk_per_section_with_heading_paths() -> None:
    chunks = HeadingChunker().chunk(GUIDE)
    assert [chunk.heading for chunk in chunks] == [
        "Guide",
        "Guide > Install",
        "Guide > Install > Linux",
        "Guide > Use",
    ]
    assert chunks[1].text == "## Install\n\npip install x"


def test_every_chunk_is_an_exact_slice_of_its_document() -> None:
    for chunker in (HeadingChunker(), FixedSizeChunker(max_chars=40, overlap=10)):
        for chunk in chunker.chunk(GUIDE):
            assert GUIDE.text[chunk.start : chunk.end] == chunk.text


def test_heading_only_sections_are_skipped_and_preamble_is_kept() -> None:
    chunks = HeadingChunker().chunk(doc("Preamble.\n\n# Title\n\n## Sub\n\nBody.\n"))
    assert [(chunk.heading, chunk.text) for chunk in chunks] == [
        ("", "Preamble."),
        ("Title > Sub", "## Sub\n\nBody."),
    ]


def test_fixed_chunks_respect_the_size_limit_and_overlap() -> None:
    paragraphs = [f"Paragraph {i} " + "word " * 15 for i in range(12)]
    document = doc("\n\n".join(p.strip() for p in paragraphs) + "\n")
    chunks = FixedSizeChunker(max_chars=350, overlap=120).chunk(document)

    assert len(chunks) > 2
    assert all(len(chunk.text) <= 350 for chunk in chunks)
    for previous, current in pairwise(chunks):
        assert current.start < previous.end, "neighbouring chunks should overlap"
    assert chunks[0].start == 0
    assert chunks[-1].end == len(document.text.rstrip("\n"))


def test_fixed_chunker_never_splits_a_code_block() -> None:
    code = f"{FENCE}python\n" + "print('line')\n" * 40 + FENCE
    document = doc(f"# Code\n\nShort intro.\n\n{code}\n\nOutro text.\n")
    chunks = FixedSizeChunker(max_chars=200, overlap=50).chunk(document)

    holding = [chunk for chunk in chunks if f"{FENCE}python" in chunk.text]
    assert len(holding) == 1
    assert code in holding[0].text
    for chunk in chunks:
        assert chunk.text.count(FENCE) % 2 == 0, "a chunk must not hold half a code block"


def test_long_prose_is_split_at_whitespace_without_breaking_words() -> None:
    words = [f"word{i}" for i in range(200)]
    document = doc(" ".join(words) + "\n")
    chunks = FixedSizeChunker(max_chars=300, overlap=0).chunk(document)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert [w for chunk in chunks for w in chunk.text.split()] == words


def test_overlap_must_be_smaller_than_the_window() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker(max_chars=100, overlap=100)
    with pytest.raises(ValueError, match="max_chars"):
        FixedSizeChunker(max_chars=0, overlap=0)


def test_chunk_ids_are_stable_and_unique() -> None:
    first = chunk_documents([GUIDE], make_chunker(Strategy.FIXED, max_chars=40, overlap=10))
    second = chunk_documents([GUIDE], make_chunker(Strategy.FIXED, max_chars=40, overlap=10))
    ids = [chunk.chunk_id for chunk in first]
    assert ids == [chunk.chunk_id for chunk in second]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "guide.md#fixed-0"


def test_make_chunker_picks_the_strategy() -> None:
    assert isinstance(make_chunker(Strategy.HEADING), HeadingChunker)
    fixed = make_chunker(Strategy.FIXED, max_chars=500, overlap=50)
    assert isinstance(fixed, FixedSizeChunker)
    assert fixed.max_chars == 500


def test_chunks_round_trip_through_jsonl(tmp_path: Path) -> None:
    chunks = HeadingChunker().chunk(GUIDE)
    path = tmp_path / "out" / "heading.jsonl"
    assert write_chunks(chunks, path) == len(chunks)
    assert read_chunks(path) == chunks
