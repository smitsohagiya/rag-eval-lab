import hashlib
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from rag_eval_lab.ingest.corpus import CorpusSource, download, extract_docs, fetch_corpus


def make_archive() -> bytes:
    files = {
        "fastapi-1.0/docs/en/docs/index.md": "# Home\n",
        "fastapi-1.0/docs/en/docs/tutorial/first.md": "# First\n",
        "fastapi-1.0/docs/en/docs/img/logo.png": "not markdown",
        "fastapi-1.0/docs/en/docs/release-notes.md": "# Release Notes\n",
        "fastapi-1.0/docs/en/docs/tutorial/_internal.md": "# Internal\n",
        "fastapi-1.0/docs/es/docs/index.md": "# Inicio\n",
        "fastapi-1.0/README.md": "# Readme\n",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


ARCHIVE = make_archive()


def source(sha256: str = hashlib.sha256(ARCHIVE).hexdigest()) -> CorpusSource:
    return CorpusSource(
        name="fastapi",
        version="1.0",
        url="https://example.test/fastapi.tar.gz",
        sha256=sha256,
        docs_prefix="docs/en/docs",
        exclude=("release-notes.md", "_*.md"),
    )


def serving(content: bytes) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    )


def test_extract_keeps_only_english_markdown(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(ARCHIVE)
    dest = tmp_path / "out"

    assert extract_docs(archive, dest, "docs/en/docs") == 4
    assert (dest / "index.md").read_text() == "# Home\n"
    assert (dest / "tutorial" / "first.md").exists()
    assert not (dest / "img").exists()


def test_extract_skips_excluded_pages(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(ARCHIVE)
    dest = tmp_path / "out"

    assert extract_docs(archive, dest, "docs/en/docs", ("release-notes.md", "_*.md")) == 2
    assert not (dest / "release-notes.md").exists()
    assert not (dest / "tutorial" / "_internal.md").exists()


def test_fetch_verifies_the_checksum_and_records_the_source(tmp_path: Path) -> None:
    dest = tmp_path / "corpus"
    assert fetch_corpus(source(), dest, client=serving(ARCHIVE)) == 2
    recorded = json.loads((dest / ".source.json").read_text())
    assert recorded["version"] == "1.0"


def test_fetch_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_corpus(source(sha256="0" * 64), tmp_path / "corpus", client=serving(ARCHIVE))


def test_fetch_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    dest = tmp_path / "corpus"
    dest.mkdir()
    (dest / "old.md").write_text("old")
    with pytest.raises(FileExistsError):
        fetch_corpus(source(), dest, client=serving(ARCHIVE))
    assert fetch_corpus(source(), dest, force=True, client=serving(ARCHIVE)) == 2
    assert not (dest / "old.md").exists()


def test_download_retries_transient_errors(tmp_path: Path) -> None:
    calls = {"count": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("connection reset", request=request)
        if calls["count"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, content=b"payload")

    dest = tmp_path / "file.bin"
    client = httpx.Client(transport=httpx.MockTransport(flaky))
    download("https://example.test/f", dest, client=client, backoff_seconds=0)
    assert dest.read_bytes() == b"payload"
    assert calls["count"] == 3


def test_download_does_not_retry_client_errors(tmp_path: Path) -> None:
    calls = {"count": 0}

    def missing(_: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(missing))
    with pytest.raises(httpx.HTTPStatusError):
        download("https://example.test/f", tmp_path / "f", client=client, backoff_seconds=0)
    assert calls["count"] == 1
