"""Hybrid search as a tool, and a ReAct-style loop sketch.

SKETCH, not a live LangChain agent. There is no LLM here, no network,
and no API key. The tool is a plain function over the same 3 toy docs
used in every session. It shows the shape an agent needs: a search
function it can call, results it can read, and an answer that cites
doc titles or says honestly when nothing was found.

Toy vectors: Sessions 1 and 2 do not ship in this repo yet, so the
tiny vectors are written out here with a comment, instead of being
imported. Each vector has 3 dims: password, reset, vpn. Doc 0 is
password plus reset, Doc 1 is vpn, Doc 2 is password only. Real
embeddings have hundreds of dims and come from a model. These 3 dims
exist only to make the fusion visible by hand.
"""

from __future__ import annotations

import math

DOCS = [
    {
        "id": 0,
        "title": "How to reset your password",
        "text": "Go to Settings, then click Reset, then check your email for a code.",
        "vector": [1.0, 1.0, 0.0],
    },
    {
        "id": 1,
        "title": "VPN setup for remote work",
        "text": "Install the VPN client, then sign in with your work account.",
        "vector": [0.0, 0.0, 1.0],
    },
    {
        "id": 2,
        "title": "How to change your password",
        "text": "Go to Settings, then click Change password.",
        "vector": [1.0, 0.0, 0.0],
    },
]

# Word to toy query vector, same 3 dims as above. Words not listed
# here add nothing, which is how a nonsense query scores zero.
_QUERY_VECTORS = {
    "password": [1.0, 0.0, 0.0],
    "reset": [0.0, 1.0, 0.0],
    "change": [0.0, 0.2, 0.0],
    "forgot": [0.7, 0.3, 0.0],
    "login": [0.9, 0.1, 0.0],
    "vpn": [0.0, 0.0, 1.0],
    "remote": [0.0, 0.0, 0.7],
    "work": [0.0, 0.0, 0.2],
    "setup": [0.0, 0.0, 0.3],
}


def _tokens(text):
    return [t.strip(".,!?;:") for t in text.lower().split() if t.strip(".,!?;:")]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _query_vector(query):
    vec = [0.0, 0.0, 0.0]
    for token in _tokens(query):
        word_vec = _QUERY_VECTORS.get(token)
        if word_vec:
            for i in range(3):
                vec[i] += word_vec[i]
    return vec


def _lexical_scores(query):
    query_tokens = set(_tokens(query))
    scores = {}
    for doc in DOCS:
        text = doc["title"] + " " + doc["text"]
        shared = len(query_tokens & set(_tokens(text)))
        if shared > 0:
            scores[doc["id"]] = shared / len(query_tokens)
    return scores


def _dense_scores(query):
    query_vec = _query_vector(query)
    if all(v == 0 for v in query_vec):
        return {}
    scores = {}
    for doc in DOCS:
        score = _cosine(query_vec, doc["vector"])
        if score > 0:
            scores[doc["id"]] = score
    return scores


def _ranked(scores):
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def hybrid_search_tool(query, top_k=2):
    """Search the 3 toy docs and fuse lexical and dense ranks with RRF.

    RRF score for a doc is the sum of 1 / (60 + rank) over each list
    it appears in, with rank starting at 1. Ties break toward the
    better lexical score, then the lower doc id. A query with no
    lexical match and no dense match returns an empty list.
    """
    lexical = _lexical_scores(query)
    dense = _dense_scores(query)
    if not lexical and not dense:
        return []
    fused = {}
    for ranked in (_ranked(lexical), _ranked(dense)):
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (60 + rank)
    ordered = sorted(
        fused.items(), key=lambda item: (-item[1], -lexical.get(item[0], 0.0), item[0])
    )
    by_id = {doc["id"]: doc for doc in DOCS}
    return [
        {
            "id": doc_id,
            "title": by_id[doc_id]["title"],
            "text": by_id[doc_id]["text"],
            "score": round(score, 4),
        }
        for doc_id, score in ordered[:top_k]
    ]


def answer_query(query, top_k=2):
    """Call the tool, then format an answer that cites doc titles.

    If the tool finds nothing, say so honestly. Never invent a doc,
    a step, or a product name that the tool did not return.
    """
    results = hybrid_search_tool(query, top_k=top_k)
    if not results:
        return (
            f'I could not find anything in the docs for "{query}". '
            "Try different words, or check that the doc exists."
        )
    lines = [f'Based on the docs for "{query}":']
    for i, result in enumerate(results, start=1):
        lines.append(f'{i}. "{result["title"]}": {result["text"]}')
    return "\n".join(lines)


def react_loop_sketch(query):
    """A ReAct-style loop, as a fixed sketch with no LLM.

    ReAct, from Yao et al. 2022, is a loop: Thought, Action,
    Observation, then Thought again. The sketch below runs that loop
    once with the tool above, in place of a model choosing the steps.
    In a live LangChain agent, the model writes the Thought lines and
    picks the Action. Here the steps are fixed, so you can read the
    shape of the loop without an API key.
    """
    results = hybrid_search_tool(query, top_k=2)
    titles = [result["title"] for result in results]
    steps = [
        "Thought: I need to search the docs before answering.",
        f"Action: hybrid_search_tool({query!r})",
        f"Observation: found {titles}." if titles else "Observation: found nothing.",
    ]
    if results:
        steps.append("Thought: The results match the query, so I can answer from them.")
    else:
        steps.append("Thought: Nothing matched, so I must say so and stop.")
    steps.append(f"Final answer: {answer_query(query, top_k=2)}")
    return steps
