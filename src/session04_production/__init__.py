"""Session 4: production and agentic RAG.

Starter code. It teaches the ideas on toy data in memory. It is not
a production search engine. No network, no model download, no API key.
"""

from session04_production.ab_test import assign, ctr, ctr_percent, srm_check
from session04_production.acl import (
    filter_before_search,
    post_filter_wrong,
    search_after_filter,
)
from session04_production.agent_tool import (
    DOCS,
    answer_query,
    hybrid_search_tool,
    react_loop_sketch,
)
from session04_production.refresh import InMemoryIndex, stale_read_demo

__all__ = [
    "DOCS",
    "InMemoryIndex",
    "answer_query",
    "assign",
    "ctr",
    "ctr_percent",
    "filter_before_search",
    "hybrid_search_tool",
    "post_filter_wrong",
    "react_loop_sketch",
    "search_after_filter",
    "srm_check",
    "stale_read_demo",
]
