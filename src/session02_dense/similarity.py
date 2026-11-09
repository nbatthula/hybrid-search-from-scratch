"""Similarity measures for Session 2: dense search.

Stdlib only. Vectors are plain lists of floats.
"""

import math


def dot(a, b):
    """Dot product of two vectors of the same length."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    """Euclidean length (L2 norm) of a vector."""
    if len(a) == 0:
        raise ValueError("vector must not be empty")
    return math.sqrt(sum(x * x for x in a))


def cosine(a, b):
    """Cosine similarity: dot(a, b) / (norm(a) * norm(b)).

    Returns a number from -1 to 1. A value of 1 means the vectors
    point in exactly the same direction. Raises ValueError for
    zero-length vectors, because the angle is then undefined.
    """
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine is undefined for a zero vector")
    return dot(a, b) / (na * nb)


def euclidean(a, b):
    """Euclidean distance: the straight-line distance between vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
