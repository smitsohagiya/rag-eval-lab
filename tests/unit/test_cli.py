from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_eval_lab import __version__, cli
from rag_eval_lab.cli import app
from rag_eval_lab.ingest.chunking import read_chunks

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_ingest_writes_chunks(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nFirst page.\n\n## Part\n\nMore text.\n")
    (corpus / "b.md").write_text("# B\n\nSecond page.\n")
    out = tmp_path / "chunks.jsonl"

    result = runner.invoke(
        app, ["ingest", "--strategy", "heading", "--corpus", str(corpus), "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert "2 pages -> 3 chunks (heading" in result.output
    assert [chunk.heading for chunk in read_chunks(out)] == ["A", "A > Part", "B"]


def test_fetch_corpus_command_reports_the_page_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(source: object, dest: Path, *, force: bool = False) -> int:
        return 159

    monkeypatch.setattr(cli, "fetch_corpus", fake_fetch)
    result = runner.invoke(app, ["fetch-corpus", "--dest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Fetched 159 pages" in result.output
