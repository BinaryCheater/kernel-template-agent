"""``kta worker``, ``kta target probe``, ``kta backend``: WP2 commands (thin wrappers)."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from kta.backends.smoke import Smoke
from kta.cli.output import emit_json, emit_text
from kta.core.context import Context
from kta.core.errors import ConfigError, NotSupported
from kta.core.ids import UlidFactory
from kta.core.logging import setup
from kta.core.registry import resolve
from kta.core.schemas.hardware import Target
from kta.core.trace import new_tracer, span, tracing
from kta.execution import JobExecutor
from kta.execution.registry import register, worker_configs
from kta.store import code_state
from kta.targets import Gb10Target

ContextFactory = Callable[[str], Context]


@contextmanager
def command(make_ctx: ContextFactory, process: str, name: str) -> Generator[Context]:
    """Context with logging and a span for one ``kta`` invocation (conventions §4.1)."""
    ctx = make_ctx(process)
    log_file = setup(ctx)
    ops = resolve("ops", ctx)
    tracer = new_tracer(ops, ctx.clock, UlidFactory(ctx.clock), os.environ)
    with tracing(tracer), span("cli", name, f"process:{process}", log=str(log_file)):
        yield ctx


def _out(args: argparse.Namespace, data: dict[str, Any], text: str) -> None:
    if args.json:
        emit_json(data)
    else:
        emit_text(text)


def cmd_register(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "worker-register", f"kta worker register {args.name}") as ctx:
        configs = worker_configs(ctx)
        if args.name not in configs:
            raise ConfigError(f"no [workers.{args.name}] in the configuration")
        reg = register(
            ctx, resolve("store", ctx), resolve("ops", ctx), configs[args.name], code_state().commit
        )
    labels = " ".join(f"{k}={v}" for k, v in sorted(reg.worker.labels.items()))
    data = {"worker": reg.worker_id, "env_manifest": reg.env_id, **reg.worker.model_dump()}
    _out(args, data, f"worker {reg.worker_id}\nenv manifest {reg.env_id}\nlabels {labels}")
    return 0


def cmd_list(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    ctx = make_ctx("worker-list")
    ex: JobExecutor = resolve("executor", ctx)
    rows = [
        {
            "name": w.name,
            "worker": w.reg.worker_id,
            "healthy": w.healthy,
            "labels": w.reg.worker.labels,
            "devices": w.reg.worker.devices,
        }
        for w in ex.workers
    ]
    text = "\n".join(
        f"{r['name']:8} {r['worker'][:12]} healthy={r['healthy']} devices={r['devices']}"
        for r in rows
    )
    _out(args, {"workers": rows}, text or "(no registered workers for this commit)")
    return 0


def cmd_serve(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    raise NotSupported("the pull-transport worker daemon (kta/execution/transport/pull.md)")


def cmd_probe(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "target-probe", "kta target probe") as ctx:
        plugin: Gb10Target = resolve("target", ctx)
        target = plugin.probe(resolve("executor", ctx))
    report = plugin.last_probe
    if report is None:
        raise ConfigError("probe finished without a report")
    counts = {s: sum(v == s for v in report.support.values()) for s in ("yes", "no", "unknown")}
    data = {
        "target": report.target_id,
        "body": target.model_dump(),
        "facts": report.facts,
        "features": report.features,
        "support": report.support,
    }
    lines = [f"target {report.target_id}", f"  {target.model_dump()}", f"features {counts}"]
    lines += [f"  {n:28} {s}" for n, s in sorted(report.support.items())]
    _out(args, data, "\n".join(lines))
    return 0


def _smoke(ctx: Context) -> tuple[Smoke, JobExecutor, Gb10Target]:
    plugin: Gb10Target = resolve("target", ctx)
    ex: JobExecutor = resolve("executor", ctx)
    store = resolve("store", ctx)
    target_rec = store.get(plugin.latest_target_id())
    target = Target.model_validate(target_rec.body)
    return Smoke(store, resolve("ops", ctx), ctx.clock, ex, target), ex, plugin


def cmd_smoke(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    results: list[dict[str, Any]] = []
    with command(make_ctx, "backend-smoke", f"kta backend smoke {args.backend}") as ctx:
        smoke, _, _ = _smoke(ctx)
        names = ["cuda", "tilelang"] if args.backend == "all" else [args.backend]
        for name in names:
            r = smoke.vector_add(resolve(f"backend.{name}", ctx), args.seed)
            results.append(
                {
                    "backend": name,
                    "build": r.build_id,
                    "build_ok": r.build_ok,
                    "check": r.check_id,
                    "passed": r.passed,
                    **r.detail,
                }
            )
    text = "\n".join(
        f"{r['backend']:9} build {r['build'][:12]} ok={r['build_ok']} check "
        f"{(r['check'] or '-')[:12]} passed={r['passed']}"
        for r in results
    )
    _out(args, {"results": results}, text)
    return 0 if all(r["passed"] for r in results) else 1


def cmd_selftest(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "backend-selftest", f"kta backend selftest {args.test}") as ctx:
        smoke, ex, _ = _smoke(ctx)
        cuda = resolve("backend.cuda", ctx)
        if args.test == "lease":
            gpu = [w.name for w in ex.workers if w.reg.worker.labels.get("gpu") == smoke.target.sm]
            if not gpu:
                raise ConfigError(f"no registered worker with gpu={smoke.target.sm}")
            report_id = smoke.lease_deferral(cuda, lambda: ex.lease_held(gpu[0]))
        else:
            report_id = smoke.timeout_recovery(cuda, args.seed, args.timeout)
        report = resolve("store", ctx).get(report_id)
    _out(
        args, {"report": report_id, **report.body}, f"report {report_id}\n{report.body['summary']}"
    )
    return 0


def add_exec_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    worker = sub.add_parser("worker", help="register and list compute workers")
    wc = worker.add_subparsers(dest="worker_command", required=True)
    r = wc.add_parser("register", help="probe a configured worker; write Worker and EnvManifest")
    r.add_argument("name")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=lambda a: cmd_register(a, make_ctx))
    ls = wc.add_parser("list", help="registered workers usable by this commit")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=lambda a: cmd_list(a, make_ctx))
    sv = wc.add_parser("serve", help="pull-transport worker daemon (not implemented)")
    sv.add_argument("--json", action="store_true")
    sv.set_defaults(func=lambda a: cmd_serve(a, make_ctx))

    target = sub.add_parser("target", help="hardware target")
    tc = target.add_subparsers(dest="target_command", required=True)
    p = tc.add_parser("probe", help="probe and write Target, HwFact and HwFeature records")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=lambda a: cmd_probe(a, make_ctx))

    backend = sub.add_parser("backend", help="backend smoke tests on real workers")
    bc = backend.add_subparsers(dest="backend_command", required=True)
    s = bc.add_parser("smoke", help="build and run vector add, check the output")
    s.add_argument("--backend", choices=["cuda", "tilelang", "all"], default="all")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=lambda a: cmd_smoke(a, make_ctx))
    t = bc.add_parser("selftest", help="executor behaviour on the GPU worker")
    t.add_argument("test", choices=["lease", "timeout"])
    t.add_argument("--seed", type=int, default=2)
    t.add_argument("--timeout", type=int, default=10)
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=lambda a: cmd_selftest(a, make_ctx))
