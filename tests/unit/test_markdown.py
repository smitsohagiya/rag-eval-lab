from rag_eval_lab.ingest.markdown import BlockKind, first_heading, parse_blocks

FENCE = "`" * 3

DOC = f"""# First Steps {{ #first-steps }}

Intro paragraph
continues here.

{FENCE}python
# this is a comment, not a heading
x = 1

y = 2
{FENCE}

## Check it {{ #check-it }}
"""


def test_blocks_follow_the_document_structure() -> None:
    kinds = [block.kind for block in parse_blocks(DOC)]
    assert kinds == [BlockKind.HEADING, BlockKind.TEXT, BlockKind.CODE, BlockKind.HEADING]


def test_code_fence_is_one_block_with_blank_lines_and_hash_lines() -> None:
    code = next(block for block in parse_blocks(DOC) if block.kind is BlockKind.CODE)
    text = DOC[code.start : code.end]
    assert text.startswith(f"{FENCE}python")
    assert text.endswith(FENCE)
    assert "# this is a comment" in text
    assert "y = 2" in text


def test_heading_anchor_suffix_is_removed() -> None:
    titles = [block.title for block in parse_blocks(DOC) if block.kind is BlockKind.HEADING]
    assert titles == ["First Steps", "Check it"]


def test_blocks_are_exact_slices_without_trailing_newlines() -> None:
    for block in parse_blocks(DOC):
        assert not DOC[block.start : block.end].endswith("\n")
        assert DOC[block.start : block.end].strip()


def test_unclosed_fence_runs_to_the_end() -> None:
    text = f"Before\n\n{FENCE}\nopen code\n# still code\n"
    blocks = parse_blocks(text)
    assert [block.kind for block in blocks] == [BlockKind.TEXT, BlockKind.CODE]
    assert text[blocks[1].start : blocks[1].end].endswith("# still code")


def test_tilde_fence_needs_a_long_enough_closing_fence() -> None:
    text = "~~~~\ncode\n~~~\nstill code\n~~~~~\n\nafter\n"
    blocks = parse_blocks(text)
    assert [block.kind for block in blocks] == [BlockKind.CODE, BlockKind.TEXT]
    assert "still code" in text[blocks[0].start : blocks[0].end]


def test_closing_hashes_are_not_part_of_the_title() -> None:
    blocks = parse_blocks("## Install ##\n")
    assert blocks[0].level == 2
    assert blocks[0].title == "Install"


def test_first_heading() -> None:
    assert first_heading(DOC) == "First Steps"
    assert first_heading("## Only a subheading\n\ntext\n") is None
