"""Score fusion and rank fusion for hybrid search.

Score fusion: min-max normalize each score dict to the range 0 to 1,
then take a weighted sum, 0.7 dense and 0.3 sparse by default.
Rank fusion: a thin wrapper around rrf.fuse.

Standard library only.
"""

from session03_hybrid import rrf

DEFAULT_DENSE_WEIGHT = 0.7
DEFAULT_SPARSE_WEIGHT = 0.3


def min_max_normalize(scores):
    """Rescale a dict of doc id to score into the range 0 to 1.

    The best score becomes 1 and the worst becomes 0. If all scores
    are equal, every doc gets 0.5, which keeps the order unchanged.
    """
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {doc: 0.5 for doc in scores}
    span = high - low
    return {doc: (value - low) / span for doc, value in scores.items()}


def weighted_sum(dense_norm, sparse_norm, dense_weight=DEFAULT_DENSE_WEIGHT,
                 sparse_weight=DEFAULT_SPARSE_WEIGHT):
    """Weighted sum of two normalized scores.

    Example from the session article: dense 0.9, sparse 0.8, weights
    0.7 and 0.3 gives 0.7 * 0.9 + 0.3 * 0.8 = 0.87.
    """
    return dense_weight * dense_norm + sparse_weight * sparse_norm


def score_fusion(dense_scores, sparse_scores,
                 dense_weight=DEFAULT_DENSE_WEIGHT,
                 sparse_weight=DEFAULT_SPARSE_WEIGHT):
    """Fuse two score dicts by min-max normalization plus weighted sum.

    Each dict maps doc id to a raw score. Returns a dict of doc id to
    fused score for the union of both dicts. A doc missing from one
    side uses the normalized floor of that side, 0.0, for its term.
    """
    dense_norm = min_max_normalize(dense_scores)
    sparse_norm = min_max_normalize(sparse_scores)
    fused = {}
    for doc in set(dense_norm) | set(sparse_norm):
        fused[doc] = weighted_sum(
            dense_norm.get(doc, 0.0),
            sparse_norm.get(doc, 0.0),
            dense_weight,
            sparse_weight,
        )
    return fused


def score_fusion_ranking(dense_scores, sparse_scores,
                          dense_weight=DEFAULT_DENSE_WEIGHT,
                          sparse_weight=DEFAULT_SPARSE_WEIGHT):
    """Score fusion as a ranked list of (doc, score), best first."""
    fused = score_fusion(dense_scores, sparse_scores,
                         dense_weight, sparse_weight)
    return sorted(fused.items(), key=lambda item: (-item[1], str(item[0])))


def rank_fusion(rank_lists, k=rrf.DEFAULT_K, default_rank=rrf.DEFAULT_RANK):
    """Rank fusion wrapper around rrf.fuse.

    Takes ranked lists of doc ids and returns (doc, score) pairs,
    best first.
    """
    return rrf.fuse(rank_lists, k=k, default_rank=default_rank)
