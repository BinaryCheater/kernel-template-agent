"""Frozen source attention replay campaign coordinator and submission interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kta_template_replay.manifest import load, write_new

from .evaluate import attempt_window, confirm, evaluate, initialize, public, select


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("data", type=Path)
    init.add_argument("--campaign", type=Path, required=True)
    session = sub.add_parser("session")
    session.add_argument("data", type=Path)
    session.add_argument("--slot", required=True)
    session.add_argument("--resume", action="store_true")
    for name in ("preflight", "evaluate", "select", "confirm"):
        command = sub.add_parser(name)
        command.add_argument("data", type=Path)
        command.add_argument("--slot", default="run-01")
        if name in {"evaluate", "select"}:
            command.add_argument("--candidate", type=Path, required=name == "evaluate")
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.data, args.campaign)
        print(json.dumps({"initialized": str(args.data)}))
        return 0
    if args.command == "session":
        from .session import run

        result = run(args.data.resolve(), args.slot, args.resume)
        print(json.dumps(public(result), indent=2))
        return 0
    if args.command == "preflight":
        campaign = load(args.data / "campaign.json")
        windows = [
            attempt_window(
                args.data,
                campaign,
                args.slot,
                "preflight",
                1,
                index,
                None,
                1800,
                args.data / "preparation" / f"window-{index}",
            )
            for index in (1, 2)
        ]
        result = {
            "passed": all(w.get("window_valid") for w in windows),
            "windows": windows,
            "purpose": "original versus original, swapped measurement labels/order",
        }
        write_new(args.data / "preparation" / "evaluator-preflight.json", result)
    elif args.command == "select":
        result = select(args.data, args.slot, args.candidate)
        result["passed"] = True
    elif args.command == "confirm":
        result = confirm(args.data, args.slot)
        result["passed"] = result["confirmed"]
    else:
        result = evaluate(args.data, args.slot, args.candidate)
    print(json.dumps(public(result), indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
