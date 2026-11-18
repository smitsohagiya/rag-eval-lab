"""Load markdown pages from a directory tree."""

from pathlib import Path

from pydantic import BaseModel

from rag_eval_lab.ingest.markdown import first_heading


class Document(BaseModel):
    """One markdown page. ``doc_id`` is its path relative to the corpus root."""

    doc_id: str
    title: str
    text: str


def load_markdown_dir(root: Path) -> list[Document]:
    """Load every ``.md`` file under ``root``, sorted by path so runs are reproducible."""
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    documents: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_id = path.relative_to(root).as_posix()
        documents.append(Document(doc_id=doc_id, title=first_heading(text) or doc_id, text=text))
    return documents
