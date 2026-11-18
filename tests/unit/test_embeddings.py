import numpy as np

from rag_eval_lab.embeddings import HashingEmbedder, embedding_text


def test_hashing_vectors_are_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dim=32)
    first = embedder.embed_query("path parameters with types")
    second = embedder.embed_documents(["path parameters with types"])[0]
    assert first.shape == (32,)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_texts_that_share_words_are_closer() -> None:
    embedder = HashingEmbedder(dim=256)
    query = embedder.embed_query("declare query parameters")
    near = embedder.embed_query("query parameters are declared as function arguments")
    far = embedder.embed_query("deploy with docker containers")
    assert float(query @ near) > float(query @ far)


def test_empty_text_gives_a_zero_vector() -> None:
    assert not HashingEmbedder(dim=8).embed_query("!!!").any()


def test_embedding_text_puts_the_heading_first() -> None:
    assert embedding_text("Tutorial > Query", "Body") == "Tutorial > Query\n\nBody"
    assert embedding_text("", "Body") == "Body"
