"""Text embeddings: a local ONNX model for real runs and a hashing embedder for tests."""

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Turns text into fixed-size vectors. Documents and queries may be embedded differently."""

    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class FastEmbedEmbedder:
    """A local ONNX embedding model run with fastembed. The model is downloaded on first use."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> None:
        from fastembed import TextEmbedding  # imported here so unit tests never load a model

        self.name = model_name
        self._batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        vectors = self._model.passage_embed(list(texts), batch_size=self._batch_size)
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]

    def embed_query(self, text: str) -> Vector:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words vectors: each word adds weight to a hashed dimension.

    There is no model to download, so tests stay fast and offline. Texts that share words get
    similar vectors, which is all the tests need.
    """

    def __init__(self, dim: int = 64) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _embed(self, text: str) -> Vector:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return (vector / norm).astype(np.float32) if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)


def embedding_text(heading: str, text: str) -> str:
    """What gets embedded for a chunk: its heading path, then the chunk itself.

    The heading path gives short chunks their context, e.g. "Query Parameters > Defaults".
    """
    return f"{heading}\n\n{text}" if heading else text
