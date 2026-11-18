"""Retrievers: vector, keyword and hybrid search over one indexed collection."""

from rag_eval_lab.retrieval.base import Hit, Retriever, RetrieverName
from rag_eval_lab.retrieval.factory import build_retriever
from rag_eval_lab.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_eval_lab.retrieval.keyword import KeywordRetriever
from rag_eval_lab.retrieval.vector import VectorRetriever

__all__ = [
    "Hit",
    "HybridRetriever",
    "KeywordRetriever",
    "Retriever",
    "RetrieverName",
    "VectorRetriever",
    "build_retriever",
    "reciprocal_rank_fusion",
]
