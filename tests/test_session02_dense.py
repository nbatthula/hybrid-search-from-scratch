"""Tests for Session 2: dense search.

Every number here comes from the worked example in
docs/session-02-dense-search.md. The vectors are toy vectors for
teaching, not outputs of a real embedding model. Tests use
pytest.approx because the article rounds to two decimals.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session02_dense.ann_toy import TOY_DOCS, QUERY_B, toy_ann_search
from session02_dense.knn import brute_force_knn
from session02_dense.similarity import cosine, dot, euclidean, norm

DOC_0 = [0.90, 0.10, 0.20]
DOC_1 = [0.10, 0.90, 0.10]
DOC_2 = [0.85, 0.15, 0.25]


def test_dot_products():
    assert dot(QUERY_B, DOC_0) == pytest.approx(0.844, abs=0.001)
    assert dot(QUERY_B, DOC_2) == pytest.approx(0.816, abs=0.001)
    assert dot(QUERY_B, DOC_1) == pytest.approx(0.216, abs=0.001)


def test_norms():
    assert norm(QUERY_B) == pytest.approx(0.9104, abs=0.001)
    assert norm(DOC_0) == pytest.approx(0.9274, abs=0.001)
    assert norm(DOC_2) == pytest.approx(0.8986, abs=0.001)
    assert norm(DOC_1) == pytest.approx(0.9110, abs=0.001)


def test_cosine_query_b():
    # Article values, rounded to two decimals: Doc 0 about 1.00,
    # Doc 2 about 0.99, Doc 1 about 0.26 (the worked steps give
    # 0.9997, 0.9975, and 0.2604).
    assert cosine(QUERY_B, DOC_0) == pytest.approx(1.00, abs=0.01)
    assert cosine(QUERY_B, DOC_2) == pytest.approx(0.99, abs=0.01)
    assert cosine(QUERY_B, DOC_1) == pytest.approx(0.26, abs=0.01)


def test_cosine_ranking():
    assert cosine(QUERY_B, DOC_0) > cosine(QUERY_B, DOC_2) > cosine(QUERY_B, DOC_1)


def test_cosine_same_vector_is_one():
    assert cosine(DOC_0, DOC_0) == pytest.approx(1.0, abs=1e-9)


def test_cosine_zero_vector_raises():
    with pytest.raises(ValueError):
        cosine(QUERY_B, [0.0, 0.0, 0.0])


def test_euclidean_distances():
    assert euclidean(QUERY_B, DOC_0) == pytest.approx(0.0283, abs=0.001)
    assert euclidean(QUERY_B, DOC_2) == pytest.approx(0.0656, abs=0.001)
    assert euclidean(QUERY_B, DOC_1) == pytest.approx(1.1076, abs=0.001)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        dot([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        euclidean([1.0, 2.0], [1.0])


def test_brute_force_knn_returns_doc0_then_doc2():
    results = brute_force_knn(QUERY_B, TOY_DOCS, k=2)
    assert [label for label, _ in results] == ["Doc 0", "Doc 2"]
    assert results[0][1] == pytest.approx(1.00, abs=0.01)
    assert results[1][1] == pytest.approx(0.99, abs=0.01)


def test_brute_force_knn_k1():
    assert brute_force_knn(QUERY_B, TOY_DOCS, k=1)[0][0] == "Doc 0"


def test_toy_ann_returns_same_top2():
    results = toy_ann_search(QUERY_B, k=2)
    assert [label for label, _ in results] == ["Doc 0", "Doc 2"]


def test_toy_ann_scores_match_exact():
    exact = dict(brute_force_knn(QUERY_B, TOY_DOCS, k=2))
    approx = dict(toy_ann_search(QUERY_B, k=2))
    for label in exact:
        assert approx[label] == pytest.approx(exact[label], abs=1e-9)
