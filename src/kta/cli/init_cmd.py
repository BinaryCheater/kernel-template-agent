"""``kta init``: create the environment's store and import the seed knowledge (WP4, D212)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from kta.cli.output import emit_json, emit_text
from kta.core.context import Context
from kta.core.registry import resolve
from kta.knowledge import (
    ImportResult,
    SeedError,
    import_hw_catalog,
    import_references,
    import_seed,
    seed_dir,
)
from kta.targets.gb10 import catalog

ContextFactory = Callable[[str], Context]


def _summary(result: ImportResult) -> dict[str, Any]:
    return {
        "batch": result.batch,
        "written": len(result.written),
        "reused": len(result.reused),
        "counts": result.counts,
        "record_ids": result.written,
        "warnings": [str(w) for w in result.warnings],
    }


def cmd_init(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    ctx = make_ctx("init")
    store = resolve("store", ctx)  # opening the store creates and migrates the database
    root = seed_dir(ctx)
    out: dict[str, Any] = {"data_dir": str(ctx.data_dir), "seed_dir": str(root), "batches": []}
    code = 0
    try:
        out["batches"].append(_summary(import_hw_catalog(ctx, store, catalog())))
        out["batches"].append(_summary(import_seed(ctx, store, root)))
        if args.references:
            out["batches"].append(_summary(import_references(ctx, store, root / "references")))
    except SeedError as exc:
        out["error"] = str(exc)
        code = 1
    if args.json:
        emit_json(out)
    else:
        lines = [f"data dir {out['data_dir']}", f"seed dir {out['seed_dir']}"]
        for b in out["batches"]:
            lines.append(f"{b['batch']}: wrote {b['written']}, reused {b['reused']} {b['counts']}")
            lines += [f"  WARN {w}" for w in b["warnings"]]
        if "error" in out:
            lines.append(f"FAIL {out['error']}")
        emit_text("\n".join(lines))
    return code


def add_init_command(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    p = sub.add_parser("init", help="create the store and import the seed knowledge")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--references",
        action="store_true",
        help="also import the reference decompositions under <seed>/references",
    )
    p.set_defaults(func=lambda a: cmd_init(a, make_ctx))
