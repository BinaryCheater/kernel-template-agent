"""Commands of the WP11 processes: learning backlog, improvement batches, experiments,
hardware calibration and process quotas (``kta process start <name>`` runs them)."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from kta.agents import profile
from kta.cli.exec_cmds import ContextFactory, _out, command
from kta.orchestrator import improve, proc_calibrate, proc_learn, proc_summarize, quotas
from kta.orchestrator.pipeline import Services
from kta.orchestrator.proc_common import ProcReport, get_state
from kta.orchestrator.proc_experiment import approved
from kta.orchestrator.runner import Runner


def cmd_learn_add(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "learn-add", f"kta learn add {args.reference}") as ctx:
        svc = Services.open(ctx)
        items = proc_learn.add_to_backlog(svc.store, svc.ops, args.reference)
    _out(args, {"backlog": items}, f"backlog: {len(items)} references")
    return 0


def cmd_learn_backlog(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    items = proc_learn.backlog(Services.open(make_ctx("learn-backlog")).ops)
    _out(args, {"backlog": items}, "\n".join(items) or "(empty)")
    return 0


def cmd_improve_run(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """Without ``--now`` the summarize process picks the request up at its next tick."""
    label = args.label or "manual"
    with command(make_ctx, "improve-run", f"kta improve run {label}") as ctx:
        svc = Services.open(ctx)
        now = ctx.clock.now()
        if not args.now:
            proc_summarize.request(svc.ops, label, now)
            data: dict[str, Any] = {"requested": label}
            text = f"requested batch {label}; the summarize process writes it"
        else:
            raw = svc.ops.get_pointer(proc_summarize.LAST)
            since = None if raw is None else datetime.fromisoformat(raw)
            rid = improve.summarize(Runner(svc, "cli"), label, since, now)
            data = {"improvement": rid}
            text = f"improvement Report {rid[:12]}"
    _out(args, data, text)
    return 0


def cmd_experiment_status(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    svc = Services.open(make_ctx("experiment-status"))
    rows = []
    for pid in approved(svc.store):
        st = get_state(svc.ops, f"experiment:{pid}")
        slots: dict[str, Any] = st.get("slots", {})
        rows.append(
            {
                "protocol": pid,
                "name": svc.store.get(pid).body["name"],
                "status": st.get("status", "approved"),
                "runs": sum(1 for s in slots.values() if s.get("run")),
                "started": len(slots),
                "error": get_state(svc.ops, f"experiment:{pid}:error").get("error"),
            }
        )
    text = "\n".join(
        f"{r['protocol'][:12]} {r['name']} {r['status']} runs={r['runs']}" for r in rows
    )
    _out(args, {"experiments": rows}, text or "(no approved experiment)")
    return 0


def cmd_calibrate_hw(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """Peaks (when the HwModel is stale) and the microbenchmark suite, now (GPU worker)."""
    with command(make_ctx, "calibrate-hw", "kta calibrate hw") as ctx:
        rep = ProcReport("calibrate")
        proc_calibrate.hardware(Runner(Services.open(ctx), "cli"), rep)
    _out(args, {"done": rep.done}, str(rep))
    return 0


def cmd_quotas(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    ctx = make_ctx("quotas")
    ops = Services.open(ctx).ops
    base = quotas.shares(ctx.config)
    active = quotas.active_experiments(ops)
    in_force = quotas.effective(base, bool(active))
    text = "\n".join(
        f"{k}: {in_force[k]:.3f} (configured {v:.3f})" for k, v in sorted(base.items())
    )
    _out(args, {"configured": base, "in_force": in_force, "experiments": active}, text)
    return 0


def cmd_profile_write(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """Record a profile version (idempotent); approving it is a separate step (D366)."""
    spec = {"codex-v1": profile.CODEX_V1, "codex-v2": profile.CODEX_V2}[args.name]
    with command(make_ctx, "profile-write", f"kta profile write {args.name}") as ctx:
        pid = profile.write_profile(Services.open(ctx).store, ctx.clock.now(), spec)
    _out(args, {"profile": pid, "name": spec.name}, f"AgentProfile {spec.name}: {pid}")
    return 0


def add_process_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    def parser(
        parent: argparse._SubParsersAction[argparse.ArgumentParser], name: str, fn: Any, h: str
    ) -> argparse.ArgumentParser:
        p = parent.add_parser(name, help=h)
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=lambda a: fn(a, make_ctx))
        return p

    learn = sub.add_parser("learn", help="the learning backlog").add_subparsers(
        dest="learn_command", required=True
    )
    parser(learn, "add", cmd_learn_add, "put a Reference on the backlog").add_argument("reference")
    parser(learn, "backlog", cmd_learn_backlog, "References waiting for an expand loop")
    imp = sub.add_parser("improve", help="improvement batches").add_subparsers(
        dest="improve_command", required=True
    )
    run = parser(imp, "run", cmd_improve_run, "close an improvement batch")
    run.add_argument("--label")
    run.add_argument("--now", action="store_true", help="write the Report in this command")
    exp = sub.add_parser("experiment", help="approved experiment protocols").add_subparsers(
        dest="experiment_command", required=True
    )
    parser(exp, "status", cmd_experiment_status, "runs of every approved protocol")
    cal = sub.add_parser("calibrate", help="hardware calibration").add_subparsers(
        dest="calibrate_command", required=True
    )
    parser(cal, "hw", cmd_calibrate_hw, "measure peaks and microbenchmarks now")
    parser(sub, "quotas", cmd_quotas, "process quotas configured and in force")
    prof = sub.add_parser("profile", help="agent profile versions").add_subparsers(
        dest="profile_command", required=True
    )
    parser(prof, "write", cmd_profile_write, "record a profile version").add_argument(
        "name", choices=["codex-v1", "codex-v2"]
    )
