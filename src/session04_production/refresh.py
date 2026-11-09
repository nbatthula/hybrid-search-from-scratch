"""Incremental refresh and stale reads, on a tiny in-memory index.

The point of this file is one idea: an index is a copy of the source
data, and a copy can be out of date. Refresh only the docs that
changed. Keep a version and an updated_at stamp on each doc, so a
search can show which copy it returned.
"""

from __future__ import annotations


def _tokens(text):
    return [t.strip(".,!?;:") for t in text.lower().split() if t.strip(".,!?;:")]


class InMemoryIndex:
    """A small index that holds one current version of each doc."""

    def __init__(self):
        self.docs = {}

    def upsert(self, doc):
        """Store one doc, replacing any older version with the same id."""
        stored = dict(doc)
        stored.setdefault("version", 1)
        stored.setdefault("updated_at", "")
        self.docs[stored["id"]] = stored
        return stored

    def refresh(self, changed_docs):
        """Upsert only the docs that changed. Return the ids refreshed.

        Docs that are not in changed_docs are left alone. That is the
        whole difference between an incremental refresh and a full
        rebuild: a full rebuild re-indexes every doc, this one touches
        only the changed ones.
        """
        refreshed = []
        for doc in changed_docs:
            self.upsert(doc)
            refreshed.append(doc["id"])
        return refreshed

    def get(self, doc_id):
        """Return the current indexed version of one doc, or None."""
        return self.docs.get(doc_id)

    def search(self, query, top_k=3):
        """Return current indexed docs that share words with the query.

        Ranked by shared word count, then by doc id for a stable order.
        This is a toy ranker. It exists so the demo can show which
        version is in the index, not to teach ranking again.
        """
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return []
        scored = []
        for doc in self.docs.values():
            text = doc.get("title", "") + " " + doc.get("text", "")
            shared = len(query_tokens & set(_tokens(text)))
            if shared > 0:
                scored.append((-shared, doc["id"], doc))
        scored.sort()
        return [doc for _, _, doc in scored[:top_k]]


def stale_read_demo():
    """Show a stale read, then a fresh read, with the same query.

    Timeline, from the article:
    - 10:00: Doc 0 is edited in the source DB, version 2.
    - 10:02: a user searches. The refresher has not run yet, so the
      index still holds version 1. The user sees the old text.
    - 10:05: the refresher runs and upserts only Doc 0.
    - 10:06: the same search now returns version 2.
    """
    index = InMemoryIndex()
    index.upsert(
        {
            "id": 0,
            "title": "How to reset your password",
            "text": "Go to Settings, then click Reset, then follow the old steps.",
            "version": 1,
            "updated_at": "09:00",
        }
    )
    before = index.search("reset password", top_k=1)[0]
    index.refresh(
        [
            {
                "id": 0,
                "title": "How to reset your password",
                "text": "Go to Settings, then click Reset, then check your email for a code.",
                "version": 2,
                "updated_at": "10:00",
            }
        ]
    )
    after = index.search("reset password", top_k=1)[0]
    return {
        "before_version": before["version"],
        "before_text": before["text"],
        "before_updated_at": before["updated_at"],
        "after_version": after["version"],
        "after_text": after["text"],
        "after_updated_at": after["updated_at"],
    }
