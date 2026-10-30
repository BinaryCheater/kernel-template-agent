"""``kta agents ...``: profile export and batch environments (D322).

The cli module (WP6) owns ``src/kta/cli``; this hook adds the subcommands so that the
logic stays in ``kta.agents``. ``register(sub, make_ctx, emit)`` is called from the
parser builder; ``emit(args, data, text)`` is the cli's output function (only the cli
may write to stdout, conventions §4).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kta.agents import batches
from kta.core.context import Context
from kta.core.registry import resolve
from kta.store import code_state

ContextFactory = Callable[[str], Context]
Emit = Callable[[argparse.Namespace, dict[str, Any], str], None]
EXIT_VERIFY_FAILED = 1


def cmd_export_profile(args: argparse.Namespace, make_ctx: ContextFactory, emit: Emit) -> int:
    ctx = make_ctx("agents-export-profile")
    files = batches.export_profile(resolve("store", ctx), args.profile, Path(args.dir))
    emit(
        args,
        {"profile": args.profile, "dir": args.dir, "files": files},
        f"{len(files)} files in {args.dir}",
    )
    return 0


def cmd_env_build(args: argparse.Namespace, make_ctx: ContextFactory, emit: Emit) -> int:
    ctx = make_ctx("agents-env-build")
    state = code_state()
    root = batches.build(
        resolve("store", ctx), ctx.data_dir, args.batch, args.profile, state.commit
    )
    lock = batches.read_lock(root)
    n = len(lock["files"])
    text = f"batch {args.batch} at {root} ({n} template files, commit {state.commit[:12]})"
    if state.dirty:
        text += "\nwarning: the code has uncommitted changes; this batch is not reproducible"
    emit(args, {"root": str(root), "dirty": state.dirty, **lock}, text)
    return 0


def cmd_env_verify(args: argparse.Namespace, make_ctx: ContextFactory, emit: Emit) -> int:
    ctx = make_ctx("agents-env-verify")
    root = batches.batches_root(ctx.data_dir) / args.batch
    result = batches.verify(resolve("store", ctx), root)
    text = "ok" if result.ok else "\n".join(["FAILED", *result.problems])
    emit(args, {"batch": args.batch, "ok": result.ok, "problems": result.problems}, text)
    return 0 if result.ok else EXIT_VERIFY_FAILED


def register(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory, emit: Emit
) -> None:
    agents = sub.add_parser("agents", help="agent profiles and batch environments")
    cmds = agents.add_subparsers(dest="agents_command", required=True)

    ex = cmds.add_parser("export-profile", help="write a profile and all its blobs as files")
    ex.add_argument("profile")
    ex.add_argument("dir")
    ex.add_argument("--json", action="store_true")
    ex.set_defaults(func=lambda a: cmd_export_profile(a, make_ctx, emit))

    env = cmds.add_parser("env", help="batch environments ($KTA_DATA/<env>/batches/<batch>/)")
    envs = env.add_subparsers(dest="env_command", required=True)
    b = envs.add_parser("build", help="generate a new batch root from one profile")
    b.add_argument("batch")
    b.add_argument("--profile", required=True)
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=lambda a: cmd_env_build(a, make_ctx, emit))
    v = envs.add_parser("verify", help="regenerate a batch's template and compare")
    v.add_argument("batch")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=lambda a: cmd_env_verify(a, make_ctx, emit))
