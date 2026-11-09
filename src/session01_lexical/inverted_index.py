"""Build and look up an inverted index.

An inverted index maps each term to a posting list. A posting list
holds one (doc_id, count) pair per doc that contains the term,
sorted by doc_id. To answer a query, look up each query term and
score only the docs in its posting list.
"""

from collections import Counter

from .tokenizer import tokenize


def build_index(docs):
    """Build an inverted index from docs.

    docs is a dict of doc_id to text, or a list of texts. With a
    list, doc ids are 0, 1, 2, in order. Returns a dict of term to
    a list of (doc_id, count) pairs, sorted by doc_id.
    """
    if isinstance(docs, dict):
        items = sorted(docs.items())
    else:
        items = list(enumerate(docs))
    index = {}
    for doc_id, text in items:
        counts = Counter(tokenize(text))
        for term, count in counts.items():
            index.setdefault(term, []).append((doc_id, count))
    for term in index:
        index[term].sort()
    return index


def lookup(term, index):
    """Return the posting list for term, or an empty list."""
    return list(index.get(term, []))
