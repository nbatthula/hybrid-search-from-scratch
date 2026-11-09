"""Session 3: hybrid search.

Dense search plus sparse search, fused by rank or by score.
Standard library only.
"""

from session03_hybrid.evaluation import (
    average_precision_at_k,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from session03_hybrid.fusion import (
    min_max_normalize,
    rank_fusion,
    score_fusion,
    weighted_sum,
)
from session03_hybrid.rrf import fuse, rrf_score
from session03_hybrid.splade_toy import (
    DOCS,
    QUERY_B_VECTOR,
    SPLADE_VECTORS,
    dot_product,
    score_query_b,
)

__all__ = [
    "DOCS",
    "QUERY_B_VECTOR",
    "SPLADE_VECTORS",
    "average_precision_at_k",
    "dot_product",
    "fuse",
    "map_at_k",
    "min_max_normalize",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "rank_fusion",
    "recall_at_k",
    "rrf_score",
    "score_fusion",
    "score_query_b",
    "weighted_sum",
]
