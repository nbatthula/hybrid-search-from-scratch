"""BM25 scoring, with k1 = 1.2 and b = 0.75.

IDF uses the Robertson form with natural log:
ln((N - DF + 0.5) / (DF + 0.5) + 1).
The term frequency part saturates: it approaches k1 + 1 = 2.20
and never passes it. Length normalization scales a match by
doc length against average doc length.

Running example, Query A "reset password":
Doc 0 scores about 1.45, Doc 2 scores about 0.47, Doc 1 scores 0.
Values in the article are rounded to two places. Tests allow
a 2 percent relative tolerance.
"""

import math
from collections import Counter

from .tokenizer import tokenize

K1 = 1.2
B = 0.75


def inverse_document_frequency(term, docs_tokens):
    """Robertson IDF: ln((N - DF + 0.5) / (DF + 0.5) + 1)."""
    total = len(docs_tokens)
    df = sum(1 for tokens in docs_tokens if term in tokens)
    if total == 0 or df == 0:
        return 0.0
    return math.log((total - df + 0.5) / (df + 0.5) + 1)


def term_frequency_component(term_freq, doc_len, avg_len, k1=K1, b=B):
    """Saturating TF part for one term in one doc.

    Equals f * (k1 + 1) / (f + k1 * (1 - b + b * len / avg_len)).
    When doc_len equals avg_len, this is f * (k1 + 1) / (f + k1).
    """
    if term_freq == 0 or avg_len == 0:
        return 0.0
    norm = 1 - b + b * (doc_len / avg_len)
    return term_freq * (k1 + 1) / (term_freq + k1 * norm)


def saturation_factor(term_freq, k1=K1):
    """TF part with doc length equal to average length.

    Values: f=1 gives 1.00, f=2 gives 1.37, f=10 gives 1.96,
    f=100 gives 2.17. The cap is k1 + 1 = 2.20.
    """
    if term_freq == 0:
        return 0.0
    return term_freq * (k1 + 1) / (term_freq + k1)


def score(query, docs, k1=K1, b=B):
    """Score each doc for query. Returns a dict of doc_id to score.

    docs is a dict of doc_id to text, or a list of texts. Query
    terms are counted once each, even if repeated in the query.
    """
    if isinstance(docs, dict):
        items = sorted(docs.items())
    else:
        items = list(enumerate(docs))
    docs_tokens = [tokenize(text) for _, text in items]
    lengths = [len(tokens) for tokens in docs_tokens]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    query_terms = sorted(set(tokenize(query)))
    scores = {}
    for (doc_id, _), doc_tokens in zip(items, docs_tokens):
        counts = Counter(doc_tokens)
        total = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if freq:
                idf = inverse_document_frequency(term, docs_tokens)
                tf_part = term_frequency_component(
                    freq, len(doc_tokens), avg_len, k1=k1, b=b
                )
                total += idf * tf_part
        scores[doc_id] = total
    return scores
