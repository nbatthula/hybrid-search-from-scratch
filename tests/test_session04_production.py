"""Tests for Session 4: refresh, ACL, A/B testing, and the agent tool.

Toy docs and toy vectors only. No network, no model download, no key.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session04_production.ab_test import assign, ctr, srm_check
from session04_production.acl import (
    acl_example,
    filter_before_search,
    post_filter_wrong,
    search_after_filter,
)
from session04_production.agent_tool import answer_query, hybrid_search_tool
from session04_production.refresh import InMemoryIndex, stale_read_demo

DOCS = [
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


def test_refresh_shows_stale_then_fresh_version():
    demo = stale_read_demo()
    assert demo["before_version"] == 1
    assert demo["after_version"] == 2
    assert "old steps" in demo["before_text"]
    assert "check your email for a code" in demo["after_text"]
    assert demo["before_updated_at"] == "09:00"
    assert demo["after_updated_at"] == "10:00"


def test_refresh_only_touches_changed_docs():
    index = InMemoryIndex()
    index.upsert(
        {"id": 0, "title": "How to reset your password", "text": "Old text.", "version": 1, "updated_at": "09:00"}
    )
    index.upsert(
        {"id": 1, "title": "VPN setup for remote work", "text": "VPN text.", "version": 1, "updated_at": "09:00"}
    )
    refreshed = index.refresh(
        [{"id": 0, "title": "How to reset your password", "text": "New text.", "version": 2, "updated_at": "10:00"}]
    )
    assert refreshed == [0]
    assert index.get(0)["version"] == 2
    assert index.get(0)["text"] == "New text."
    assert index.get(1)["version"] == 1
    assert index.get(1)["text"] == "VPN text."
    assert index.search("reset password")[0]["version"] == 2


def test_acl_filter_hides_forbidden_doc_before_ranking():
    ben = {"name": "Ben", "principals": ["support"]}
    allowed = filter_before_search(ben, DOCS)
    assert [doc["id"] for doc in allowed] == [2]
    results = search_after_filter(ben, DOCS, "reset password", top_k=3)
    assert [doc["id"] for doc in results] == [2]
    assert all("eng" not in doc.get("acl", []) or "support" in doc["acl"] for doc in results)


def test_acl_two_users_same_query_different_docs():
    example = acl_example()
    assert [doc["id"] for doc in example["ana"]] == [0]
    assert [doc["id"] for doc in example["ben"]] == [2]
    assert example["ana"][0]["id"] != example["ben"][0]["id"]


def test_post_filter_loses_a_slot():
    ben = {"name": "Ben", "principals": ["support"]}
    # Doc 0 ranks first for "reset password" but Ben may not see it.
    # Post-filter takes top 1, then hides it, so Ben gets nothing.
    assert post_filter_wrong(ben, DOCS, "reset password", top_k=1) == []
    # Filter first, and the allowed Doc 2 fills the slot instead.
    assert [doc["id"] for doc in search_after_filter(ben, DOCS, "reset password", top_k=1)] == [2]


def test_ab_assignment_is_stable_for_same_user():
    assert assign("user-123") == assign("user-123")
    assert assign("user-123", seed="session04") == assign("user-123", seed="session04")
    assert assign("user-123") in ("A", "B")
    # Different users can land in different arms; at least the
    # function is deterministic, which is what this test checks.
    arms = {assign(f"user-{i}") for i in range(20)}
    assert arms == {"A", "B"}


def test_ctr_120_of_1000_is_0_12():
    assert ctr(120, 1000) == 0.12
    assert ctr(0, 0) == 0.0


def test_srm_check_expected_is_half_of_total():
    result = srm_check(520, 480)
    assert result["total"] == 1000
    assert result["expected_a"] == 500
    assert result["expected_b"] == 500
    assert result["difference_a"] == 20
    assert result["difference_b"] == -20
    assert result["ratio_a"] == 0.52
    assert result["ratio_b"] == 0.48


def test_agent_tool_returns_cited_titles_for_query_a():
    results = hybrid_search_tool("reset password", top_k=2)
    assert results[0]["id"] == 0
    assert results[0]["title"] == "How to reset your password"
    answer = answer_query("reset password", top_k=2)
    assert "How to reset your password" in answer
    assert "How to change your password" in answer


def test_agent_tool_honest_no_result_for_nonsense_query():
    assert hybrid_search_tool("zxqv blorp wumpus", top_k=2) == []
    answer = answer_query("zxqv blorp wumpus", top_k=2)
    assert "could not find anything" in answer
    assert "How to reset your password" not in answer
