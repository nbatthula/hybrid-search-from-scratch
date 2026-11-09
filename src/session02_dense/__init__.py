"""Session 2: dense search. Embeddings, cosine similarity, exact KNN, and a toy ANN graph."""

from .ann_toy import QUERY_B, TOY_DOCS, TOY_POINTS, ToyGraph, toy_ann_search
from .knn import brute_force_knn
from .similarity import cosine, dot, euclidean, norm

__all__ = [
    "QUERY_B",
    "TOY_DOCS",
    "TOY_POINTS",
    "ToyGraph",
    "brute_force_knn",
    "cosine",
    "dot",
    "euclidean",
    "norm",
    "toy_ann_search",
]
