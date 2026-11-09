"""A/B testing helpers: stable assignment, CTR, and SRM check.

Stdlib hashlib only. No network, no random state. The same user id
and the same seed always land in the same arm, so a user does not
flip between A and B on each search.
"""

from __future__ import annotations

import hashlib


def assign(user_id, seed="session04"):
    """Assign a user to arm "A" or "B", deterministically.

    Hash the seed and the user id, take the first 8 hex chars, and
    use even or odd. Change the seed to start a fresh test, so old
    assignments do not carry over.
    """
    key = f"{seed}:{user_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return "A" if int(digest[:8], 16) % 2 == 0 else "B"


def ctr(clicks, impressions):
    """Click-through rate as a fraction: clicks / impressions.

    Example: 120 clicks from 1000 searches gives 0.12.
    Zero impressions gives 0.0, not a crash.
    """
    if impressions <= 0:
        return 0.0
    return clicks / impressions


def ctr_percent(clicks, impressions):
    """Click-through rate as a percent. 120 / 1000 is 12.0 percent."""
    return ctr(clicks, impressions) * 100


def srm_check(count_a, count_b):
    """Sample ratio mismatch check, in plain numbers.

    A clean 50/50 test expects half the users in each arm. This
    function returns the expected count per arm, the observed counts,
    the difference per arm, and the observed share per arm. It does
    not run a stats test. If the observed share is far from 0.5, stop
    and find the bug in assignment or logging before trusting the
    result. A small gap on a small sample is normal noise.
    """
    total = count_a + count_b
    expected = total / 2
    return {
        "total": total,
        "count_a": count_a,
        "count_b": count_b,
        "expected_a": expected,
        "expected_b": expected,
        "difference_a": count_a - expected,
        "difference_b": count_b - expected,
        "ratio_a": (count_a / total) if total else 0.0,
        "ratio_b": (count_b / total) if total else 0.0,
    }
