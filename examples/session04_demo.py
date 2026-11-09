"""Session 4 demo: stale to fresh refresh, then an ACL example.

Run from the repo root:

    python examples/session04_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session04_production.acl import acl_example
from session04_production.agent_tool import answer_query, react_loop_sketch
from session04_production.refresh import stale_read_demo


def main():
    print("Refresh: stale read, then fresh read")
    demo = stale_read_demo()
    print(f'  At 10:02, search "reset password" returns version {demo["before_version"]}:')
    print(f'    {demo["before_text"]}')
    print(f'  Refresher runs at 10:05. At 10:06, the same search returns version {demo["after_version"]}:')
    print(f'    {demo["after_text"]}')
    print()
    print("ACL: two users, same query")
    example = acl_example()
    print('  Ana (eng) searches "reset password":')
    for doc in example["ana"]:
        print(f'    {doc["title"]}')
    print('  Ben (support) searches "reset password":')
    for doc in example["ben"]:
        print(f'    {doc["title"]}')
    print(f'  Ben with post-filter at top 1 gets: {example["ben_post_filter_top1"]}')
    print("  That empty list is the lost slot. Filter first, then rank.")
    print()
    print("Agent tool sketch: ReAct loop for Query A")
    for step in react_loop_sketch("reset password"):
        print(f"  {step}")
    print()
    print("Agent answer:")
    print(answer_query("reset password", top_k=2))


if __name__ == "__main__":
    main()
