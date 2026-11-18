from pathlib import Path

import pytest

from rag_eval_lab.ingest.loaders import load_markdown_dir


def test_loads_markdown_recursively_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.md").write_text("# Bee\n\ntext\n")
    (tmp_path / "sub" / "a.md").write_text("no heading here\n")
    (tmp_path / "notes.txt").write_text("not markdown\n")

    documents = load_markdown_dir(tmp_path)

    assert [doc.doc_id for doc in documents] == ["b.md", "sub/a.md"]
    assert documents[0].title == "Bee"
    assert documents[1].title == "sub/a.md"


def test_missing_directory_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="corpus directory not found"):
        load_markdown_dir(tmp_path / "missing")
