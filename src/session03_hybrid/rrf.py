"""Reciprocal Rank Fusion (RRF).

Cormack, Clarke, and Buettcher (2009). A doc scores the sum over each
ranked list of 1 / (k + rank), with k = 60 by default. Ranks are
1-based. A doc missing from a list gets default_rank (1000), so its
term is 1 / (60 + 1000) = 1 / 1060 instead of zero.

Standard library only.
"""

DEFAULT_K = 60
DEFAULT_RANK = 1000


def rrf_score(rank_lists, k=DEFAULT_K, default_rank=DEFAULT_RANK):
    """Score every doc in the union of the ranked lists.

    rank_lists: a list of lists. Each inner list holds doc ids in
    rank order, best first. Returns a dict of doc id to RRF score.
    """
    scores = {}
    for ranking in rank_lists:
        for position, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + position)
    # Add the missing-list term for docs absent from a list.
    all_docs = set(scores)
    for ranking in rank_lists:
        present = set(ranking)
        for doc in all_docs - present:
            scores[doc] += 1.0 / (k + default_rank)
    return scores


def fuse(rank_lists, k=DEFAULT_K, default_rank=DEFAULT_RANK):
    """Fuse ranked lists with RRF.

    Returns a list of (doc, score) pairs, best score first. Ties
    break by doc id so the order is stable.
    """
    scores = rrf_score(rank_lists, k=k, default_rank=default_rank)
    return sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
