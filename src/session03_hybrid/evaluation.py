"""Evaluation metrics for ranked retrieval.

Precision at K, Recall at K, MRR, NDCG at K, Average Precision at K,
and MAP at K. Standard library only (math for log2).

Conventions used here, matching the session article:
- retrieved is a list of doc ids in rank order, best first.
- relevant is a set of doc ids that count as relevant.
- relevance grades for ndcg_at_k are a dict of doc id to grade
  (0 = not relevant, higher = better). Gain is 2 ** grade - 1 and
  the discount at 1-based position i is log2(i + 1).
- average_precision_at_k divides by the total number of relevant
  docs, not by the number returned.
"""

import math


def precision_at_k(retrieved, relevant, k):
    """Share of the top k results that are relevant."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for doc in top if doc in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k):
    """Share of all relevant docs found in the top k."""
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for doc in top if doc in relevant)
    return hits / len(relevant)


def average_precision_at_k(retrieved, relevant, k):
    """Mean of Precision at each rank where a relevant doc appears.

    The sum runs over relevant docs in the top k, and the divisor is
    the total number of relevant docs.
    """
    if not relevant:
        return 0.0
    total = 0.0
    hits = 0
    for position, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant)


def map_at_k(ranked_lists, relevant_sets, k):
    """Mean of average_precision_at_k across queries."""
    if not ranked_lists:
        return 0.0
    scores = [
        average_precision_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(ranked_lists, relevant_sets)
    ]
    return sum(scores) / len(scores)


def _first_relevant_rank(retrieved, relevant):
    """1-based rank of the first relevant doc, or None."""
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return position
    return None


def mrr(ranked_lists, relevant_sets):
    """Mean Reciprocal Rank across queries.

    Each query scores 1 / rank of its first relevant doc, or 0 if
    no relevant doc appears in the list.
    """
    if not ranked_lists:
        return 0.0
    total = 0.0
    for retrieved, relevant in zip(ranked_lists, relevant_sets):
        rank = _first_relevant_rank(retrieved, relevant)
        if rank is not None:
            total += 1.0 / rank
    return total / len(ranked_lists)


def ndcg_at_k(retrieved, relevance_grades, k):
    """NDCG at k with gain 2 ** grade - 1 and log2 discount.

    relevance_grades maps doc id to grade. Docs missing from the
    dict count as grade 0. Returns 0.0 when the ideal DCG is 0.
    """
    top = retrieved[:k]

    def dcg(grades_in_order):
        total = 0.0
        for position, grade in enumerate(grades_in_order, start=1):
            total += (2 ** grade - 1) / math.log2(position + 1)
        return total

    actual = dcg([relevance_grades.get(doc, 0) for doc in top])
    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    ideal = dcg(ideal_grades)
    if ideal == 0.0:
        return 0.0
    return actual / ideal
