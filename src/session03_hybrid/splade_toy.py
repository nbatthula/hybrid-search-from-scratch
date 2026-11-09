"""SPLADE-style expansion as a small teaching toy.

This is NOT a real model. The weights below are fixed numbers from
the Colab notebook example, written out by hand so the arithmetic is
easy to follow. A real SPLADE model learns its weights from a trained
BERT checkpoint. Nothing here trains, loads, or runs a model.

The running example docs:
- Doc 0: "How to reset your password"
- Doc 1: "VPN setup for remote work"
- Doc 2: "How to change your password"

Query B is "forgot my login". Plain BM25 scores Doc 0 as 0, because
neither word appears in the doc. The SPLADE toy expands Doc 0 to
terms it never contained (forgot, login, change, credentials), so the
same query scores about 6.74.
"""

DOCS = {
    "Doc 0": "How to reset your password",
    "Doc 1": "VPN setup for remote work",
    "Doc 2": "How to change your password",
}

# Plain BM25-style vectors from the notebook example. Doc 0 has only
# its two literal terms, with small weights.
BM25_VECTORS = {
    "Doc 0": {"reset": 0.096, "password": 0.035},
    "Doc 1": {"vpn": 0.096, "setup": 0.035, "remote": 0.035, "work": 0.035},
    "Doc 2": {"change": 0.096, "password": 0.035},
}

# Fixed SPLADE-style vectors. Doc 0 keeps its literal terms with
# large weights and gains four expanded terms. The Doc 1 and Doc 2
# vectors use the same pattern with fixed toy weights.
SPLADE_VECTORS = {
    "Doc 0": {
        "reset": 2.1,
        "password": 1.8,
        "forgot": 1.2,
        "login": 1.1,
        "change": 0.9,
        "credentials": 0.7,
    },
    "Doc 1": {
        "vpn": 2.0,
        "setup": 1.6,
        "remote": 1.5,
        "work": 1.2,
        "network": 0.9,
        "access": 0.8,
    },
    "Doc 2": {
        "change": 2.0,
        "password": 1.9,
        "reset": 1.1,
        "forgot": 0.8,
        "login": 0.7,
        "account": 0.6,
    },
}

# Fixed expanded query vector for Query B ("forgot my login").
# Toy weights, chosen so the dot product with Doc 0 comes out at
# about 6.74, the notebook example value. The products are:
#   reset 0.9 * 2.1 = 1.89, password 0.6 * 1.8 = 1.08,
#   forgot 1.5 * 1.2 = 1.80, login 1.1 * 1.1 = 1.21,
#   change 0.5 * 0.9 = 0.45, credentials 0.44 * 0.7 = 0.308.
# Sum = 6.738, which rounds to 6.74.
QUERY_B_VECTOR = {
    "reset": 0.9,
    "password": 0.6,
    "forgot": 1.5,
    "login": 1.1,
    "change": 0.5,
    "credentials": 0.44,
}


def dot_product(vec_a, vec_b):
    """Dot product of two sparse term-to-weight dicts."""
    return sum(weight * vec_b.get(term, 0.0) for term, weight in vec_a.items())


def score_query_b(doc="Doc 0"):
    """SPLADE toy score of Query B against one doc.

    Query B against Doc 0 returns about 6.74. Against Doc 2 it
    returns 5.10, and against Doc 1 it returns 0.0.
    """
    return dot_product(QUERY_B_VECTOR, SPLADE_VECTORS[doc])


def bm25_score_query_b(doc="Doc 0"):
    """Plain BM25 toy score of Query B against one doc.

    Always 0.0 here: "forgot" and "login" appear in no doc, so no
    BM25 vector has a weight for them.
    """
    query_terms = {"forgot", "login"}
    return sum(
        weight for term, weight in BM25_VECTORS[doc].items() if term in query_terms
    )
