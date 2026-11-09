"""Session 2 demo: rank the 3 toy docs for Query B by cosine similarity.

Run from the repo root: python examples/session02_demo.py
Stdlib only. No network, no model download.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session02_dense.ann_toy import QUERY_B, TOY_DOCS, toy_ann_search
from session02_dense.knn import brute_force_knn

DOCS = {
    "Doc 0": "How to reset your password",
    "Doc 1": "VPN setup for remote work",
    "Doc 2": "How to change your password",
}


def main():
    print('Query B: "forgot my login"')
    print("Exact brute-force KNN (cosine):")
    for label, score in brute_force_knn(QUERY_B, TOY_DOCS, k=2):
        print(f"  {label}: {DOCS[label]} (cosine {score:.4f})")
    print("Toy ANN graph search:")
    for label, score in toy_ann_search(QUERY_B, k=2):
        print(f"  {label}: {DOCS[label]} (cosine {score:.4f})")


if __name__ == "__main__":
    main()
