"""Exact brute-force KNN for Session 2: dense search.

Brute force means: score the query against every doc, sort, keep
the top k. It is exact, because nothing is skipped, and slow at
scale, because the work grows with the number of docs.
"""

from .similarity import cosine


def brute_force_knn(query, docs, k=2):
    """Return the top k docs by cosine similarity to the query.

    ``docs`` is a sequence of ``(label, vector)`` pairs. The result
    is a list of ``(label, score)`` pairs, best first.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    scored = [(label, cosine(query, vector)) for label, vector in docs]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
