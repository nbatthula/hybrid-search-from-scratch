"""TF-IDF scoring, with log base 10 IDF.

TF is the count of a term in a doc divided by the doc length.
IDF is log10(N / DF), where N is the number of docs and DF is the
number of docs that contain the term. TF-IDF is TF times IDF.
A doc score for a query is the sum of the TF-IDF weights of the
query terms in that doc.

Running example, Query A "reset password":
Doc 0 scores 0.131, Doc 2 scores 0.035, Doc 1 scores 0.
Values in the article are rounded to three places. Tests allow
a 2 percent relative tolerance, so 0.0954 counts as 0.096.
"""

import math
from collections import Counter

from .tokenizer import tokenize


def term_frequency(term, doc_tokens):
    """Count of term in doc_tokens divided by doc length."""
    if not doc_tokens:
        return 0.0
    return doc_tokens.count(term) / len(doc_tokens)


def document_frequency(term, docs_tokens):
    """Number of docs in docs_tokens that contain term."""
    return sum(1 for tokens in docs_tokens if term in tokens)


def inverse_document_frequency(term, docs_tokens):
    """log10(N / DF). Returns 0.0 when the term is in no doc."""
    total = len(docs_tokens)
    df = document_frequency(term, docs_tokens)
    if total == 0 or df == 0:
        return 0.0
    return math.log10(total / df)


def tfidf(term, doc_tokens, docs_tokens):
    """TF-IDF weight of term in one doc, given the whole set."""
    return term_frequency(term, doc_tokens) * inverse_document_frequency(
        term, docs_tokens
    )


def score(query, docs):
    """Score each doc for query. Returns a dict of doc_id to score.

    docs is a dict of doc_id to text, or a list of texts. Query
    terms are counted once each, even if repeated in the query.
    """
    if isinstance(docs, dict):
        items = sorted(docs.items())
    else:
        items = list(enumerate(docs))
    docs_tokens = [tokenize(text) for _, text in items]
    query_terms = sorted(set(tokenize(query)))
    scores = {}
    for (doc_id, _), doc_tokens in zip(items, docs_tokens):
        counts = Counter(doc_tokens)
        total = 0.0
        for term in query_terms:
            if term in counts:
                total += tfidf(term, doc_tokens, docs_tokens)
        scores[doc_id] = total
    return scores
