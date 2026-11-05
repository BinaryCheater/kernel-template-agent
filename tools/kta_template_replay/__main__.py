"""Prepare and run evaluator-owned replay tasks; GPU work always uses Executor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .manifest import freeze, load, write_new
from .package import build


def public_feedback(value):
    """Private input seeds and reference-side checks stay in retained evaluator artifacts."""
    if isinstance(value, dict):
        return {
            key: public_feedback(item)
            for key, item in value.items()
            if key not in {"seed", "production_checks", "manifest_identity", "candidate_identity"}
        }
    if isinstance(value, list):
        return [public_feedback(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("case", choices=("gemm", "attention"))
    prepare.add_argument("run", type=Path)
    prepare.add_argument("--environment", type=Path, required=True)
    prepare.add_argument("--seeds", type=int, nargs="+", required=True)
    package = sub.add_parser("package")
    package.add_argument("run", type=Path)
    package.add_argument("destination", type=Path)
    package.add_argument("--library", type=Path, required=True)
    package.add_argument("--template", action="append", default=[])
    smoke = sub.add_parser("smoke")
    smoke.add_argument("run", type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("candidate", type=Path)
    evaluate.add_argument("--agent-active-s", type=float, required=True)
    seal = sub.add_parser("seal-preflight")
    seal.add_argument("run", type=Path)
    seal.add_argument("result", type=Path)
    stage = sub.add_parser("stage")
    stage.add_argument("run", type=Path)
    stage.add_argument("candidate", type=Path)
    stage.add_argument("--agent-active-s", type=float, required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("attempt", type=Path)
    collect.add_argument("result", type=Path)
    collect.add_argument("--wall-s", type=float, required=True)
    collect.add_argument("--queue-s", type=float, required=True)
    remote = sub.add_parser("remote-evaluate")
    remote.add_argument("run", type=Path)
    remote.add_argument("candidate", type=Path)
    remote.add_argument(
        "--agent-active-s",
        type=float,
        help="override automatic active time from session-started.json",
    )
    remote.add_argument("--host", default="spark-hk")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = freeze(args.case, load(args.environment), args.seeds)
        args.run.mkdir(parents=True, exist_ok=True)
        write_new(args.run / "manifest.json", manifest)
        print(json.dumps({"manifest": str(args.run / "manifest.json")}))
    elif args.command == "package":
        result = build(args.run, args.library, args.destination, args.template)
        print(json.dumps({"workspace": str(args.destination), "files": len(result)}))
    elif args.command == "remote-evaluate":
        from .remote import remote_evaluate

        agent_active_s = args.agent_active_s
        if agent_active_s is None:
            started = load(args.run / "session-started.json")["epoch_s"]
            previous_s = sum(
                item["wall_s"] + item.get("queue_s", 0)
                for item in (
                    load(path) for path in (args.run / "attempts").glob("*/execution.json")
                )
            )
            agent_active_s = max(0, time.time() - started - previous_s)
        result = remote_evaluate(
            Path(__file__).resolve().parents[2],
            args.run,
            args.candidate,
            agent_active_s,
            args.host,
        )
        print(json.dumps(public_feedback(result), indent=2))
        return 0 if result["passed"] else 1
    elif args.command in ("seal-preflight", "stage", "collect"):
        from . import direct

        if args.command == "stage":
            attempt = direct.stage(args.run, args.candidate, args.agent_active_s)
            print(str(attempt.resolve()))
            return 0
        if args.command == "seal-preflight":
            result = direct.seal_preflight(args.run, args.result)
        else:
            result = direct.collect(args.attempt, args.result, args.wall_s, args.queue_s)
        print(json.dumps(public_feedback(result), indent=2))
        return 0 if result["passed"] else 1
    else:
        from kta.cli.exec_cmds import command
        from kta.cli.main import context_from_environ
        from kta.orchestrator.pipeline import Services

        from . import coordinator

        with command(context_from_environ, "template-replay", args.command) as ctx:
            if ctx.env != "dev":
                raise ValueError("Template replay requires KTA_ENV=dev")
            svc = Services.open(ctx)
            if args.command == "smoke":
                result = coordinator.smoke(svc.executor, args.run)
            else:
                result = coordinator.evaluate(
                    svc.executor, args.run, args.candidate, args.agent_active_s
                )
        print(json.dumps(public_feedback(result), indent=2))
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
