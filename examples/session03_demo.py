"""Demo for Session 3: dense vs sparse vs RRF ranking.

Run from the repo root:

    python examples/session03_demo.py

The dense scores are fixed toy values. The sparse scores are the
BM25 toy vectors for Query A and the SPLADE toy vectors for Query B.
Rankings come from sorting those scores, and RRF fuses the two
rankings with k = 60 and rank 1000 for a missing doc.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session03_hybrid.fusion import rank_fusion, score_fusion_ranking
from session03_hybrid.splade_toy import (
    BM25_VECTORS,
    QUERY_B_VECTOR,
    SPLADE_VECTORS,
    dot_product,
)

DOCS = ["Doc 0", "Doc 1", "Doc 2"]

# Fixed toy dense scores (cosine-style, 0 to 1).
DENSE_QUERY_A = {"Doc 0": 0.92, "Doc 2": 0.85, "Doc 1": 0.40}
DENSE_QUERY_B = {"Doc 0": 0.88, "Doc 2": 0.81, "Doc 1": 0.45}


def ranking_from_scores(scores):
    return [doc for doc, _ in sorted(scores.items(), key=lambda item: -item[1])]


def show(title, dense_scores, sparse_scores, sparse_name):
    print("=" * 60)
    print(title)
    print("=" * 60)
    dense_ranking = ranking_from_scores(dense_scores)
    sparse_ranking = ranking_from_scores(sparse_scores)
    print("Dense ranking:  %s" % dense_ranking)
    print("%s ranking: %s" % (sparse_name, sparse_ranking))
    print("RRF ranking:    %s" % [doc for doc, _ in rank_fusion([dense_ranking, sparse_ranking])])
    print("Score fusion ranking (0.7 dense + 0.3 sparse):")
    for doc, score in score_fusion_ranking(dense_scores, sparse_scores):
        print("  %s: %.3f" % (doc, score))
    print()


def main():
    # Query A: "reset password". BM25 matches the exact words.
    bm25_a = {
        doc: sum(BM25_VECTORS[doc].get(term, 0.0) for term in ("reset", "password"))
        for doc in DOCS
    }
    show('Query A: "reset password"', DENSE_QUERY_A, bm25_a, "BM25")

    # Query B: "forgot my login". BM25 scores 0 for every doc, so
    # its ranking is a tie at zero. SPLADE expansion fixes it.
    bm25_b = {
        doc: sum(BM25_VECTORS[doc].get(term, 0.0) for term in ("forgot", "login"))
        for doc in DOCS
    }
    print("Query B BM25 scores (all zero, no shared words): %s" % bm25_b)
    splade_b = {
        doc: dot_product(QUERY_B_VECTOR, SPLADE_VECTORS[doc]) for doc in DOCS
    }
    print("Query B SPLADE scores: %s" % {d: round(s, 2) for d, s in splade_b.items()})
    print()
    show('Query B: "forgot my login"', DENSE_QUERY_B, splade_b, "SPLADE")


if __name__ == "__main__":
    main()
