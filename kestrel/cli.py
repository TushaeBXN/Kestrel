"""Kestrel CLI. Commands: serve (browser UI), run (terminal, current directory)."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

from .loop import run as run_loop
from . import profiles
from .workspace import Workspace


def _run(goal: str, directory: str, yes: bool, profile: str | None):
    cfg = profiles.load()
    model, provider = profiles.get(profile, cfg)
    expert = profiles.expert_fn(cfg) if (profile or cfg.get("default")) != cfg.get("expert") else None
    ws = Workspace(Path(directory))

    def emit(ev: dict):
        t = ev.get("type")
        if t == "text":
            sys.stdout.write(ev["text"]); sys.stdout.flush()
        elif t == "tool_call":
            print(f"\n→ {ev['name']}({json.dumps(ev['input'])[:100]})")
        elif t == "tool_result":
            print(f"← {json.dumps(ev['result'])[:120]}")
        elif t == "error":
            print(f"\nERROR: {ev['message']}")
        elif t == "done":
            print("\nDone.")

    def approve(tc, detail: str) -> bool:
        if yes:
            return True
        return input(f"\nAllow {tc.name}: {detail} ? [y/N] ").strip().lower() == "y"

    run_loop(goal, model, provider, emit, ws, approve=approve, expert=expert)


def main():
    ap = argparse.ArgumentParser(prog="kestrel")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the browser UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=3001)
    s.add_argument("--no-open", action="store_true", help="don't open a browser tab")

    r = sub.add_parser("run", help="run one goal in the terminal")
    r.add_argument("goal")
    r.add_argument("--dir", default=".", help="project directory (default: current)")
    r.add_argument("-y", "--yes", action="store_true", help="auto-approve shell and git actions")

    r.add_argument("-p", "--profile", help="model profile from ~/.kestrel/models.json")
    sub.add_parser("init-models", help="write an example ~/.kestrel/models.json")

    args = ap.parse_args()
    if args.cmd == "serve":
        from .server import serve
        serve(args.host, args.port, open_browser=not args.no_open)
    elif args.cmd == "run":
        _run(args.goal, args.dir, args.yes, args.profile)
    elif args.cmd == "init-models":
        print(profiles.write_example())


if __name__ == "__main__":
    main()
