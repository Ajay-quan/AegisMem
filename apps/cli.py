"""stateful.ai CLI — talk to a running stateful.ai service from the terminal.

Zero extra dependencies (uses the stdlib-only SDK). Examples:

    python -m apps.cli health
    python -m apps.cli ingest "Alice prefers Python and FAISS." --user alice --type fact
    python -m apps.cli recall "what does alice like?" --user alice
    python -m apps.cli feedback <query_id> <memory_id> --useful
    python -m apps.cli stats --user alice
"""
from __future__ import annotations

import argparse
import json
import sys

from sdk import StatefulClient, StatefulError


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stateful_ai", description="stateful.ai memory CLI")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--api-key", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="store a memory")
    sp.add_argument("text")
    sp.add_argument("--user", required=True)
    sp.add_argument("--type", default="observation")

    sp = sub.add_parser("recall", help="retrieve memories")
    sp.add_argument("query")
    sp.add_argument("--user", required=True)
    sp.add_argument("--top-k", type=int, default=5)

    sp = sub.add_parser("update", help="smart update/supersede a memory")
    sp.add_argument("content")
    sp.add_argument("--user", required=True)

    sp = sub.add_parser("feedback", help="report usefulness of a retrieved memory")
    sp.add_argument("query_id")
    sp.add_argument("memory_id")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--useful", action="store_true")
    g.add_argument("--not-useful", action="store_true")
    sp.add_argument("--score", type=float, default=None)
    sp.add_argument("--outcome", default="")

    sp = sub.add_parser("stats", help="memory stats for a user")
    sp.add_argument("--user", required=True)

    sub.add_parser("learning", help="continual-learning stats")
    sub.add_parser("health", help="service health")

    args = p.parse_args(argv)
    client = StatefulClient(base_url=args.base_url, api_key=args.api_key)

    try:
        if args.cmd == "ingest":
            _print(client.ingest(args.text, args.user, memory_type=args.type))
        elif args.cmd == "recall":
            _print(client.retrieve(args.query, args.user, top_k=args.top_k))
        elif args.cmd == "update":
            _print(client.update(args.user, args.content))
        elif args.cmd == "feedback":
            useful = True if args.useful else (False if args.not_useful else None)
            _print(client.feedback(args.query_id, args.memory_id, useful=useful,
                                   score=args.score, outcome=args.outcome))
        elif args.cmd == "stats":
            _print(client.stats(args.user))
        elif args.cmd == "learning":
            _print(client.learning_stats())
        elif args.cmd == "health":
            _print(client.health())
    except StatefulError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
