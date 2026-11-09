"""Tests for Session 3: hybrid search.

Every test reproduces a worked number from
docs/session-03-hybrid-search.md.
"""

import math

import pytest

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
    bm25_score_query_b,
    score_query_b,
)

# Dense list: Doc 0, Doc 1, Doc 2. Sparse list: Doc 2, Doc 3, Doc 0.
# Doc 0 is rank 1 dense and rank 3 sparse. Doc 1 is rank 2 dense
# and missing from the sparse list.
DENSE_RANKING = ["Doc 0", "Doc 1", "Doc 2"]
SPARSE_RANKING = ["Doc 2", "Doc 3", "Doc 0"]


def test_rrf_rank1_dense_rank3_sparse():
    scores = rrf_score([DENSE_RANKING, SPARSE_RANKING])
    expected = 1 / 61 + 1 / 63
    assert scores["Doc 0"] == pytest.approx(expected, abs=1e-9)
    assert scores["Doc 0"] == pytest.approx(0.0323, abs=0.0001)


def test_rrf_rank2_dense_missing_sparse():
    scores = rrf_score([DENSE_RANKING, SPARSE_RANKING])
    expected = 1 / 62 + 1 / 1060
    assert scores["Doc 1"] == pytest.approx(expected, abs=1e-9)
    assert scores["Doc 1"] == pytest.approx(0.0170, abs=0.0001)


def test_rrf_both_lists_beats_one_list():
    scores = rrf_score([DENSE_RANKING, SPARSE_RANKING])
    assert scores["Doc 0"] > scores["Doc 1"]
    assert scores["Doc 2"] == pytest.approx(1 / 63 + 1 / 61, abs=1e-9)


def test_rrf_fuse_order():
    fused = fuse([DENSE_RANKING, SPARSE_RANKING])
    docs = [doc for doc, _ in fused]
    assert docs[0] in ("Doc 0", "Doc 2")
    assert docs[1] in ("Doc 0", "Doc 2")
    assert set(docs) == {"Doc 0", "Doc 1", "Doc 2", "Doc 3"}


def test_rrf_default_k_and_missing_rank():
    scores = rrf_score([["A"], ["B"]], k=60, default_rank=1000)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 1060, abs=1e-9)
    assert scores["B"] == pytest.approx(1 / 1060 + 1 / 61, abs=1e-9)


def test_min_max_normalize():
    assert min_max_normalize({"a": 10.0, "b": 20.0, "c": 30.0}) == pytest.approx(
        {"a": 0.0, "b": 0.5, "c": 1.0}
    )
    assert min_max_normalize({"a": 5.0, "b": 5.0}) == {"a": 0.5, "b": 0.5}
    assert min_max_normalize({}) == {}


def test_score_fusion_worked_example():
    # Article example, already normalized scores, 0.7 dense + 0.3 sparse.
    assert weighted_sum(0.9, 0.8, 0.7, 0.3) == pytest.approx(0.87, abs=1e-9)
    assert weighted_sum(0.6, 0.9, 0.7, 0.3) == pytest.approx(0.69, abs=1e-9)
    assert weighted_sum(0.7, 0.2, 0.7, 0.3) == pytest.approx(0.55, abs=1e-9)


def test_score_fusion_normalizes_then_weights():
    dense = {"Doc 0": 0.92, "Doc 1": 0.71, "Doc 2": 0.65}
    sparse = {"Doc 0": 2.4, "Doc 1": 0.3, "Doc 2": 3.1}
    fused = score_fusion(dense, sparse, 0.7, 0.3)
    # Dense normalized: Doc 0 = 1.0, Doc 2 = 0.0.
    # Sparse normalized: Doc 0 = 0.75, Doc 2 = 1.0.
    assert fused["Doc 0"] == pytest.approx(0.7 * 1.0 + 0.3 * 0.75, abs=1e-9)
    assert fused["Doc 2"] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0, abs=1e-9)
    assert fused["Doc 0"] > fused["Doc 2"] > fused["Doc 1"]


def test_rank_fusion_wrapper_matches_rrf():
    assert rank_fusion([DENSE_RANKING, SPARSE_RANKING]) == fuse(
        [DENSE_RANKING, SPARSE_RANKING]
    )


def test_precision_at_k_query_a():
    retrieved = ["Doc 0", "Doc 1", "Doc 2"]
    relevant = {"Doc 0", "Doc 2"}
    assert precision_at_k(retrieved, relevant, 1) == pytest.approx(1.00)
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(0.67, abs=0.01)


def test_average_precision_and_map_query_a():
    retrieved = ["Doc 0", "Doc 1", "Doc 2"]
    relevant = {"Doc 0", "Doc 2"}
    assert average_precision_at_k(retrieved, relevant, 3) == pytest.approx(
        0.83, abs=0.01
    )
    # Perfect order Doc 0, Doc 2, Doc 1 gives AP 1.00.
    perfect = ["Doc 0", "Doc 2", "Doc 1"]
    assert average_precision_at_k(perfect, relevant, 3) == pytest.approx(1.00)
    assert map_at_k([retrieved, perfect], [relevant, relevant], 3) == pytest.approx(
        (0.8333333333 + 1.0) / 2, abs=0.01
    )


def test_recall_at_k():
    # Ten relevant docs in total. Top 5 holds 4, top 10 holds 8.
    retrieved = ["R1", "R2", "X1", "R3", "R4", "X2", "R5", "R6", "R7", "R8"]
    relevant = {"R%d" % i for i in range(1, 11)}
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(0.40)
    assert recall_at_k(retrieved, relevant, 10) == pytest.approx(0.80)


def test_mrr():
    # First relevant doc at ranks 1, 2, and 4.
    ranked = [
        ["A", "B", "C"],
        ["X", "A", "B"],
        ["X", "Y", "Z", "A"],
    ]
    relevant = [{"A"}, {"A"}, {"A"}]
    assert mrr(ranked, relevant) == pytest.approx(0.583, abs=0.001)
    assert mrr(ranked, relevant) == pytest.approx(
        (1 + 0.5 + 0.25) / 3, abs=1e-9
    )


def test_ndcg():
    # Returned grades 1, 2, 0. Ideal order 2, 1, 0. NDCG is 0.80.
    retrieved = ["A", "B", "C"]
    grades = {"A": 1, "B": 2, "C": 0}
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert ndcg_at_k(retrieved, grades, 3) == pytest.approx(expected, abs=1e-9)
    assert ndcg_at_k(retrieved, grades, 3) == pytest.approx(0.80, abs=0.01)
    # Perfect order scores 1.00.
    assert ndcg_at_k(["B", "A", "C"], grades, 3) == pytest.approx(1.00)


def test_splade_toy_query_b():
    # Notebook example: Query B scores about 6.74 with SPLADE.
    assert score_query_b("Doc 0") == pytest.approx(6.74, abs=0.01)
    assert score_query_b("Doc 2") == pytest.approx(5.10, abs=0.01)
    assert score_query_b("Doc 1") == pytest.approx(0.0, abs=1e-9)


def test_bm25_query_b_is_zero():
    # "forgot" and "login" appear in no doc, so BM25 scores 0.
    assert bm25_score_query_b("Doc 0") == pytest.approx(0.0, abs=1e-9)
    assert bm25_score_query_b("Doc 2") == pytest.approx(0.0, abs=1e-9)
