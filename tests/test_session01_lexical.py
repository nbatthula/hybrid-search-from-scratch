"""Tests for session 1: tokenizer, inverted index, TF-IDF, BM25.

Every number here is checked by hand in docs/session-01-lexical-search.md.
pytest.approx with rel=0.02 allows the rounding used in the article.
"""

import pytest

from session01_lexical import tokenize
from session01_lexical.inverted_index import build_index, lookup
from session01_lexical import tfidf as tfidf_mod
from session01_lexical import bm25 as bm25_mod

DOCS = [
    "How to reset your password",
    "VPN setup for remote work",
    "How to change your password",
]

VOCAB = [
    "how",
    "to",
    "reset",
    "your",
    "password",
    "vpn",
    "setup",
    "for",
    "remote",
    "work",
    "change",
]


def test_tokenizer_output():
    assert tokenize(DOCS[0]) == ["how", "to", "reset", "your", "password"]
    assert tokenize(DOCS[1]) == ["vpn", "setup", "for", "remote", "work"]
    assert tokenize(DOCS[2]) == ["how", "to", "change", "your", "password"]
    assert tokenize("Reset PASSWORD!") == ["reset", "password"]


def test_vocab_is_11_words():
    vocab = sorted({term for doc in DOCS for term in tokenize(doc)})
    assert vocab == sorted(VOCAB)
    assert len(vocab) == 11


def test_inverted_index_posting_lists():
    index = build_index(DOCS)
    assert lookup("reset", index) == [(0, 1)]
    assert lookup("password", index) == [(0, 1), (2, 1)]
    assert lookup("how", index) == [(0, 1), (2, 1)]
    assert lookup("vpn", index) == [(1, 1)]
    assert lookup("change", index) == [(2, 1)]
    assert lookup("forgot", index) == []


def test_tfidf_weights():
    docs_tokens = [tokenize(doc) for doc in DOCS]
    assert tfidf_mod.term_frequency("reset", docs_tokens[0]) == pytest.approx(0.20)
    assert tfidf_mod.inverse_document_frequency("reset", docs_tokens) == pytest.approx(
        0.477, rel=0.02
    )
    assert tfidf_mod.inverse_document_frequency("password", docs_tokens) == pytest.approx(
        0.176, rel=0.02
    )
    assert tfidf_mod.tfidf("reset", docs_tokens[0], docs_tokens) == pytest.approx(
        0.096, rel=0.02
    )
    assert tfidf_mod.tfidf("password", docs_tokens[0], docs_tokens) == pytest.approx(
        0.035, rel=0.02
    )
    assert tfidf_mod.tfidf("how", docs_tokens[0], docs_tokens) == pytest.approx(
        0.035, rel=0.02
    )


def test_tfidf_query_a_scores():
    scores = tfidf_mod.score("reset password", DOCS)
    assert scores[0] == pytest.approx(0.131, rel=0.02)
    assert scores[2] == pytest.approx(0.035, rel=0.02)
    assert scores[1] == 0.0


def test_bm25_idf():
    docs_tokens = [tokenize(doc) for doc in DOCS]
    assert bm25_mod.inverse_document_frequency("reset", docs_tokens) == pytest.approx(
        0.98, rel=0.02
    )
    assert bm25_mod.inverse_document_frequency("password", docs_tokens) == pytest.approx(
        0.47, rel=0.02
    )


def test_bm25_query_a_scores():
    scores = bm25_mod.score("reset password", DOCS)
    assert scores[0] == pytest.approx(1.45, rel=0.02)
    assert scores[2] == pytest.approx(0.47, rel=0.02)
    assert scores[1] == 0.0


def test_bm25_saturation_cap():
    assert bm25_mod.saturation_factor(1) == pytest.approx(1.00, rel=0.02)
    assert bm25_mod.saturation_factor(2) == pytest.approx(1.37, rel=0.02)
    assert bm25_mod.saturation_factor(10) == pytest.approx(1.96, rel=0.02)
    assert bm25_mod.saturation_factor(100) == pytest.approx(2.17, rel=0.02)
    assert bm25_mod.saturation_factor(100) < 2.20
    assert bm25_mod.saturation_factor(1000000) == pytest.approx(2.20, rel=0.02)


def test_bm25_length_normalization():
    # Average length 10, one mention. Short doc (5 words) scores
    # higher than long doc (20 words) for the same mention.
    short = bm25_mod.term_frequency_component(1, 5, 10)
    long = bm25_mod.term_frequency_component(1, 20, 10)
    assert short == pytest.approx(1.26, rel=0.02)
    assert long == pytest.approx(0.71, rel=0.02)


def test_query_b_scores_zero():
    # "forgot my login" shares no word with any doc.
    assert tfidf_mod.score("forgot my login", DOCS) == {0: 0.0, 1: 0.0, 2: 0.0}
    assert bm25_mod.score("forgot my login", DOCS) == {0: 0.0, 1: 0.0, 2: 0.0}
