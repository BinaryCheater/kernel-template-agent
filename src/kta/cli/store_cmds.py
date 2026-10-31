"""``kta store ...``: store benchmark and record bundles (WP1, decision D153)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kta.cli.output import emit_json, emit_text
from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.doctor import cuda_home
from kta.core.envmanifest import collect_manifest
from kta.core.registry import resolve
from kta.core.schemas import Record
from kta.store import backup, bench, transfer

# Synthetic records get fixed timestamps so a re-run produces the same records (and IDs).
SYNTHETIC_START = datetime(2026, 1, 1, tzinfo=UTC)

ContextFactory = Callable[[str], Context]


def cmd_bench(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    ctx = make_ctx("store-bench")
    store = resolve("store", ctx)
    manifest = collect_manifest(args.role, repo_root(), cuda_home(ctx))
    env_id = store.put(Record.new("EnvManifest", manifest.to_dict(), bench.ACTOR, ctx.clock.now()))
    result = bench.run(store, args.records, args.queries, args.seed, SYNTHETIC_START)
    report_id = store.put(bench.report_record(store, result, env_id, ctx.clock.now()))
    out: dict[str, Any] = {"report_id": report_id, "env_manifest_id": env_id, **result.data}
    if args.bundle:
        out["bundle"] = transfer.export_bundle(store, [report_id], Path(args.bundle))
    if args.json:
        emit_json(out)
    else:
        lines = [f"report {report_id}", f"env manifest {env_id}"]
        lines += [
            f"  {name:22} p50 {s['p50_ms']:8.3f} ms  p95 {s['p95_ms']:8.3f} ms"
            for name, s in result.data["queries"].items()
        ]
        lines.append(f"passed: {result.data['passed']}")
        emit_text("\n".join(lines))
    return 0 if result.data["passed"] else 1


def cmd_export(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    store = resolve("store", make_ctx("store-export"))
    emit_json(transfer.export_bundle(store, list(args.ids), Path(args.out)))
    return 0


def cmd_import(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    store = resolve("store", make_ctx("store-import"))
    emit_json(transfer.import_bundle(store, Path(args.bundle)))
    return 0


def cmd_backup(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    ctx = make_ctx("store-backup")
    out = backup.backup(ctx.data_dir, Path(args.dest), ctx.clock.now(), args.handover)
    if args.json:
        emit_json(out)
    else:
        handed = f"; handed over to {args.handover}" if args.handover else ""
        emit_text(
            f"backup of {ctx.data_dir} in {args.dest}: {out['blobs']} blobs "
            f"({out['blobs_copied']} copied), db {out['db_sha256'][:12]}{handed}"
        )
    return 0


def cmd_restore(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    ctx = make_ctx("store-restore")
    out = backup.restore(Path(args.src), ctx.data_dir)
    if args.json:
        emit_json(out)
    else:
        emit_text(f"restored {args.src} into {ctx.data_dir}")
    return 0


def add_store_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    store = sub.add_parser("store", help="store benchmark and record bundles")
    cmds = store.add_subparsers(dest="store_command", required=True)

    b = cmds.add_parser("bench", help="synthetic-store query and full-text latency")
    b.add_argument("--records", type=int, default=100_000)
    b.add_argument("--queries", type=int, default=500)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--role", choices=["control", "worker"], default="worker")
    b.add_argument("--bundle", help="also write the Report and its refs to this bundle file")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=lambda a: cmd_bench(a, make_ctx))

    e = cmds.add_parser("export", help="write records (and everything they reference) to a bundle")
    e.add_argument("ids", nargs="+")
    e.add_argument("--out", required=True)
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=lambda a: cmd_export(a, make_ctx))

    i = cmds.add_parser("import", help="verify and write a bundle into this store")
    i.add_argument("bundle")
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=lambda a: cmd_import(a, make_ctx))

    bk = cmds.add_parser("backup", help="consistent copy of this store (WP23); --handover to move")
    bk.add_argument("dest")
    bk.add_argument("--handover", help="mark this store as handed over to that control plane")
    bk.add_argument("--json", action="store_true")
    bk.set_defaults(func=lambda a: cmd_backup(a, make_ctx))

    rs = cmds.add_parser("restore", help="restore a backup into this (empty) data directory")
    rs.add_argument("src")
    rs.add_argument("--json", action="store_true")
    rs.set_defaults(func=lambda a: cmd_restore(a, make_ctx))
