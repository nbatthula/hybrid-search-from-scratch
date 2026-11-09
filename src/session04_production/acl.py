"""ACL filter before search, and the post-filter pattern to avoid.

Rule: filter by permission before ranking, never rank then hide.
Post-filtering leaks in two ways. First, a forbidden doc can take a
top-k slot, so an allowed doc that should have been returned is lost.
Second, ranking scores for forbidden docs can leak through timing or
counts if the code is not careful. Filter first, and the ranker never
sees a doc the user may not see.
"""

from __future__ import annotations


def _tokens(text):
    return [t.strip(".,!?;:") for t in text.lower().split() if t.strip(".,!?;:")]


def _score(query, doc):
    query_tokens = set(_tokens(query))
    text = doc.get("title", "") + " " + doc.get("text", "")
    return len(query_tokens & set(_tokens(text)))


def _rank(docs, query):
    scored = [(doc, _score(query, doc)) for doc in docs]
    scored = [(doc, s) for doc, s in scored if s > 0]
    scored.sort(key=lambda pair: (-pair[1], pair[0]["id"]))
    return [doc for doc, _ in scored]


def filter_before_search(user, docs):
    """Return only the docs this user is allowed to see.

    user is a dict with a "principals" list, for example groups or
    roles such as ["eng"] or ["support"]. A doc is allowed when its
    "acl" list shares at least one principal with the user.
    """
    principals = set(user.get("principals", []))
    return [doc for doc in docs if set(doc.get("acl", [])) & principals]


def search_after_filter(user, docs, query, top_k=3):
    """The correct order: filter by ACL, then rank, then take top-k."""
    allowed = filter_before_search(user, docs)
    return _rank(allowed, query)[:top_k]


def post_filter_wrong(user, docs, query, top_k=3):
    """The wrong order, kept here so tests can show the loss.

    Rank all docs first, take top-k, then remove forbidden docs.
    When a forbidden doc takes a top-k slot, an allowed doc below it
    never gets returned, even though it matched the query.
    """
    ranked = _rank(docs, query)[:top_k]
    principals = set(user.get("principals", []))
    return [doc for doc in ranked if set(doc.get("acl", [])) & principals]


def acl_example():
    """Two users, same query, different allowed docs.

    Doc 0 is for eng only, Doc 2 is for support only. Ana is in eng,
    Ben is in support. Both search "reset password". Ana sees Doc 0,
    Ben sees Doc 2. Neither sees the other's doc.
    """
    docs = [
        {
            "id": 0,
            "title": "How to reset your password",
            "text": "Go to Settings, then click Reset.",
            "acl": ["eng"],
        },
        {
            "id": 1,
            "title": "VPN setup for remote work",
            "text": "Install the VPN client, then sign in.",
            "acl": ["eng"],
        },
        {
            "id": 2,
            "title": "How to change your password",
            "text": "Go to Settings, then click Change password.",
            "acl": ["support"],
        },
    ]
    ana = {"name": "Ana", "principals": ["eng"]}
    ben = {"name": "Ben", "principals": ["support"]}
    query = "reset password"
    return {
        "docs": docs,
        "ana": search_after_filter(ana, docs, query, top_k=3),
        "ben": search_after_filter(ben, docs, query, top_k=3),
        "ana_post_filter_top1": post_filter_wrong(ana, docs, query, top_k=1),
        "ben_post_filter_top1": post_filter_wrong(ben, docs, query, top_k=1),
    }
