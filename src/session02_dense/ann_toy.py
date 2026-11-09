"""A teaching toy for ANN search, Session 2: dense search.

This is NOT production HNSW. It is a small graph over a few toy
points, searched with a plain greedy walk. It shows the core idea
of graph search: start somewhere, step to a neighbor that is
closer to the query, stop when no neighbor is closer, then rank
what you visited.

Real HNSW adds layers, a better neighbor rule, and a candidate
list (ef_search). See docs/session-02-dense-search.md for how.

Stdlib only. No network, no model download, no hnswlib needed.
"""

from .similarity import cosine

# The running example docs, with the toy vectors from the article.
# These are toy vectors for teaching, not outputs of a real model.
TOY_DOCS = [
    ("Doc 0", [0.90, 0.10, 0.20]),
    ("Doc 1", [0.10, 0.90, 0.10]),
    ("Doc 2", [0.85, 0.15, 0.25]),
]

QUERY_B = [0.88, 0.12, 0.20]

# Extra points, only so the toy graph has more than 3 nodes. They
# are plain made-up points, not docs, and none of them beats Doc 0
# or Doc 2 for Query B.
SYNTHETIC_POINTS = [
    ("Synthetic A", [0.10, 0.20, 0.90]),
    ("Synthetic B", [0.20, 0.85, 0.10]),
    ("Synthetic C", [0.90, 0.85, 0.10]),
]

TOY_POINTS = TOY_DOCS + SYNTHETIC_POINTS


class ToyGraph:
    """A tiny undirected graph where each node links to its nearest neighbors."""

    def __init__(self, nodes, degree=2):
        """Build the graph. ``nodes`` is a list of (label, vector) pairs."""
        if len(nodes) < 2:
            raise ValueError("need at least 2 nodes for a graph")
        self.nodes = list(nodes)
        self.edges = {i: set() for i in range(len(nodes))}
        for i, (_, vec_i) in enumerate(self.nodes):
            scored = []
            for j, (_, vec_j) in enumerate(self.nodes):
                if i != j:
                    scored.append((cosine(vec_i, vec_j), j))
            scored.sort(reverse=True)
            for _, j in scored[:degree]:
                self.edges[i].add(j)
                self.edges[j].add(i)

    def greedy_search(self, query, k=2, entry=0):
        """Greedy walk from ``entry``. Return top k (label, score) visited."""
        if k < 1:
            raise ValueError("k must be at least 1")
        current = entry
        best_score = cosine(query, self.nodes[current][1])
        visited = {current}
        while True:
            nxt = None
            nxt_score = best_score
            for nb in sorted(self.edges[current]):
                if nb in visited:
                    continue
                score = cosine(query, self.nodes[nb][1])
                visited.add(nb)
                if score > nxt_score:
                    nxt = nb
                    nxt_score = score
            if nxt is None:
                break
            current = nxt
            best_score = nxt_score
        ranked = [
            (self.nodes[i][0], cosine(query, self.nodes[i][1]))
            for i in visited
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]


def toy_ann_search(query, k=2):
    """Greedy graph search over the toy points. Return top k (label, score)."""
    graph = ToyGraph(TOY_POINTS, degree=2)
    return graph.greedy_search(query, k=k, entry=0)
