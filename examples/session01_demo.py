"""Demo for session 1. Run from the repo root:

    python examples/session01_demo.py

Prints Query A scores under TF-IDF and BM25.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session01_lexical import bm25, tfidf

DOCS = [
    "How to reset your password",
    "VPN setup for remote work",
    "How to change your password",
]


def main():
    query = "reset password"
    print("Query A:", query)
    print("TF-IDF scores:", tfidf.score(query, DOCS))
    print("BM25 scores:", bm25.score(query, DOCS))


if __name__ == "__main__":
    main()
