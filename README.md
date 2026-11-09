# Hybrid Search from Scratch

Hybrid search, taught from scratch, in plain English, with working code.

This repo is a 4 session course. Each session is one article plus code you can run. No prior search knowledge is assumed. Short sentences, concrete examples, and the math shown step by step.

## The running example

Every session uses the same 3 docs, so nothing resets:

- Doc 0: "How to reset your password"
- Doc 1: "VPN setup for remote work"
- Doc 2: "How to change your password"

Query A: "reset password". It shares exact words with Doc 0 and Doc 2.
Query B: "forgot my login". It means the same as Doc 0 and Doc 2, and shares no useful words.

A second example appears where it helps: bathroom exhaust fans, from a live class notebook. Query "quiet bathroom exhaust fan" and query "revent 80 cfm", on the Amazon ESCI product dataset.

## The 4 sessions

| Session | Topic | Article | Code |
| --- | --- | --- | --- |
| 1 | Lexical search: tokenization, inverted index, TF-IDF, BM25 | [docs/session-01-lexical-search.md](docs/session-01-lexical-search.md) | `src/session01_lexical/` |
| 2 | Dense search: embeddings, cosine, exact KNN vs ANN, HNSW, IVF, PQ | [docs/session-02-dense-search.md](docs/session-02-dense-search.md) | `src/session02_dense/` |
| 3 | Hybrid search: SPLADE, score fusion vs rank fusion, RRF, evaluation | [docs/session-03-hybrid-search.md](docs/session-03-hybrid-search.md) | `src/session03_hybrid/` |
| 4 | Production and agentic RAG: refresh, eventual consistency, ACL filter, A/B testing, agent tool | [docs/session-04-production-agentic-rag.md](docs/session-04-production-agentic-rag.md) | `src/session04_production/` |

Session 1 is fully built, with code and tests that reproduce every number in the article. Sessions 2 to 4 are complete article drafts with working starter code and tests. The starter code is labelled as starter code in each session, and in the status note below.

## How to run

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

Run one session demo at a time:

```bash
python examples/session01_demo.py
python examples/session02_demo.py
python examples/session03_demo.py
python examples/session04_demo.py
```

Tests use only toy docs and toy vectors. They need no network, no model download, and no API key.

## The big picture

```
query
  |
  +--> dense search (meaning) --+
  |                              |
  +--> sparse search (words) ---+--> RRF fusion --> ranked results
```

Lexical search finds exact words, codes, and names. Dense search finds meaning, synonyms, and paraphrase. Hybrid search runs both in parallel and merges the two ranked lists. That is the whole course in one picture.

## Status

- Session 1, lexical search: complete, code and tests pass.
- Session 2, dense search: complete article, working starter code and tests.
- Session 3, hybrid search: complete article, working starter code and tests.
- Session 4, production and agentic RAG: complete article, working starter code and tests.

Starter code means the code teaches the idea on toy data. It is not a production search engine. Real models, real indexes, and real scale are named in each article, with paper references, so you know what to swap in next.

## Papers

Each session article lists its papers with links. The papers used across the course include TF-IDF and IDF, BM25, HNSW, FAISS, Sentence-BERT, SPLADE, RRF, DPR, RAG, and ReAct. See each article for the exact reference.

## License

MIT. See [LICENSE](LICENSE).
