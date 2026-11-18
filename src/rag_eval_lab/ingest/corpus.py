"""Download the pinned FastAPI docs and keep only the English markdown pages."""

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SOURCE_FILE = ".source.json"


class CorpusSource(BaseModel):
    """Where a corpus comes from and how to check that the download is the expected one."""

    name: str
    version: str
    url: str
    sha256: str
    docs_prefix: str
    exclude: tuple[str, ...] = ()


FASTAPI_DOCS = CorpusSource(
    name="fastapi",
    version="0.165.0",
    url="https://codeload.github.com/fastapi/fastapi/tar.gz/refs/tags/0.165.0",
    sha256="2704788113681424548544a939b6f753d8848f371dce9b0cd7d511fb23fec6d1",
    docs_prefix="docs/en/docs",
    # The changelog (742 KB) and internal pages such as _llm-test.md are not documentation
    # anyone asks questions about, and the changelog alone would dominate the chunk count.
    exclude=("release-notes.md", "_*.md"),
)


def _is_retryable(error: httpx.HTTPError) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code >= 500
    return isinstance(error, httpx.TransportError)


def download(
    url: str,
    dest: Path,
    *,
    client: httpx.Client | None = None,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> None:
    """Stream ``url`` into ``dest``. Network errors and 5xx responses are retried with
    exponential backoff; 4xx responses fail at once."""
    http = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        for attempt in range(1, attempts + 1):
            try:
                with http.stream("GET", url) as response:
                    response.raise_for_status()
                    with dest.open("wb") as handle:
                        for part in response.iter_bytes():
                            handle.write(part)
                return
            except httpx.HTTPError as error:
                if attempt == attempts or not _is_retryable(error):
                    raise
                wait = backoff_seconds * 2 ** (attempt - 1)
                logger.warning(
                    "download attempt %d failed (%s); retrying in %.1fs", attempt, error, wait
                )
                time.sleep(wait)
    finally:
        if client is None:
            http.close()


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB pieces."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_docs(archive: Path, dest: Path, docs_prefix: str, exclude: tuple[str, ...] = ()) -> int:
    """Copy the markdown files under ``<top folder>/<docs_prefix>/`` into ``dest``.

    Returns the number of pages written. Entries outside the prefix, non-markdown files, pages
    matching an ``exclude`` pattern and any path that tries to climb out of ``dest`` are ignored.
    """
    prefix = PurePosixPath(docs_prefix)
    count = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".md")):
                continue
            inside = PurePosixPath(*PurePosixPath(member.name).parts[1:])
            if not inside.is_relative_to(prefix):
                continue
            relative = inside.relative_to(prefix)
            if ".." in relative.parts or any(relative.match(pattern) for pattern in exclude):
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            target = dest.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            count += 1
    return count


def fetch_corpus(
    source: CorpusSource, dest: Path, *, force: bool = False, client: httpx.Client | None = None
) -> int:
    """Download ``source``, verify its checksum and extract its pages into ``dest``.

    Refuses to touch a non-empty ``dest`` unless ``force`` is set, so a local copy is never
    replaced by accident. Returns the number of pages.
    """
    if dest.exists() and any(dest.iterdir()):
        if not force:
            raise FileExistsError(f"{dest} is not empty; pass force=True to replace it")
        shutil.rmtree(dest)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{source.name}-{source.version}.tar.gz"
        download(source.url, archive, client=client)
        digest = sha256_of(archive)
        if digest != source.sha256:
            raise ValueError(
                f"checksum mismatch for {source.name} {source.version}: "
                f"expected {source.sha256}, got {digest}"
            )
        dest.mkdir(parents=True, exist_ok=True)
        count = extract_docs(archive, dest, source.docs_prefix, source.exclude)
    (dest / SOURCE_FILE).write_text(json.dumps(source.model_dump(), indent=2) + "\n")
    logger.info("extracted %d pages of %s %s into %s", count, source.name, source.version, dest)
    return count
