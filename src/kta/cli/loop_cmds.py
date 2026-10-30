"""Management commands of WP6: cases, references, loops, processes, inbox, budgets,
coverage and export (spec/modules/orchestrator-cli.md "cli")."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml

from kta.cli.exec_cmds import ContextFactory, _out, command
from kta.core.errors import ConfigError, NotSupported
from kta.core.registry import resolve
from kta.core.schemas import Actor, Record, RefValue
from kta.core.schemas.hardware import Target
from kta.knowledge import coverage_report, render_text_knowledge, seed_dir
from kta.measure.baseline import ensure_baseline
from kta.orchestrator import budget, processes
from kta.orchestrator.inbox import inbox
from kta.orchestrator.machines import machine
from kta.orchestrator.pipeline import Services
from kta.orchestrator.runner import Runner

HUMAN = Actor(type="human", name="kta case add")


def _yaml(path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a YAML mapping")
    return cast(dict[str, Any], data)


def _target(ctx: Any) -> Target | None:
    """The newest Target record (``kta target probe``), or None before any probe."""
    try:
        plugin = resolve("target", ctx)
        return Target.model_validate(resolve("store", ctx).get(plugin.latest_target_id()).body)
    except NotSupported:
        return None


def case_file(arg: str, ctx: Any) -> Path:
    """A path, or the name of a seed case (``knowledge/seed/cases/<name>.yaml``)."""
    path = Path(arg)
    if path.is_file():
        return path
    seed_case = seed_dir(ctx) / "cases" / f"{arg}.yaml"
    if seed_case.is_file():
        return seed_case
    raise ConfigError(f"{arg} is neither a file nor a seed case under {seed_case.parent}")


def cmd_case_add(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """The seed case format (D363): ``problem``, ``contract``, ``case`` bodies and an
    optional ``baseline`` {library, call}. ``Case.target`` is the Target name (D287),
    the newest Target's unless the file names one; the baseline Report needs that Target's
    worker image. ``Case.problem`` is filled in."""
    with command(make_ctx, "case-add", f"kta case add {args.file}") as ctx:
        doc = _yaml(str(case_file(args.file, ctx)))
        svc = Services.open(ctx)
        now = ctx.clock.now()
        put = svc.store.put
        target = _target(ctx)
        problem = put(Record.new("Problem", doc["problem"], HUMAN, now))
        contract = put(
            Record.new("Contract", doc["contract"], HUMAN, now, refs={"problem": problem})
        )
        refs: dict[str, RefValue] = {"contract": contract, "problem": problem}
        if doc.get("baseline"):
            if target is None:
                raise NotSupported("a baseline needs the worker image: run `kta target probe`")
            b = doc["baseline"]
            refs["baseline"] = ensure_baseline(
                svc.store, ctx.clock, problem, str(b["call"]), str(b["library"]), target.image
            )
        name = doc["case"].get("target") or (target.name if target else None)
        if not name:
            raise ConfigError("the case names no target and no Target record exists")
        body = {**doc["case"], "problem": problem, "target": name}
        case = put(Record.new("Case", body, HUMAN, now, refs=refs))
    data = {"case": case, "problem": problem, "contract": contract, **refs}
    _out(args, data, f"case {case}\nproblem {problem}\ncontract {contract}")
    return 0


def cmd_reference_add(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """YAML keys: ``source`` (Source body), ``reference`` (Reference body) and optional
    ``problem`` (a Problem body): written and linked as ``refs.problem``, with
    ``reference.problem`` set to its ID (D376). Otherwise ``reference.problem`` may name an
    existing Problem record."""
    doc = _yaml(args.file)
    with command(make_ctx, "reference-add", f"kta reference add {args.file}") as ctx:
        svc = Services.open(ctx)
        now = ctx.clock.now()
        actor = HUMAN.model_copy(update={"name": "kta reference add"})
        src = svc.store.put(Record.new("Source", doc["source"], actor, now))
        refs: dict[str, RefValue] = {"source": src}
        body = dict(doc["reference"])
        if doc.get("problem"):
            body["problem"] = svc.store.put(Record.new("Problem", doc["problem"], actor, now))
        problem = body.get("problem")
        if isinstance(problem, str) and svc.store.exists(problem):
            refs["problem"] = problem
        ref = svc.store.put(Record.new("Reference", body, actor, now, refs=refs))
    data = {"reference": ref, "source": src, "problem": refs.get("problem")}
    _out(args, data, f"reference {ref}\nsource {src}\nproblem {refs.get('problem') or '-'}")
    return 0


def _analysis_only(svc: Services, row: dict[str, Any], reason: str) -> None:
    """Your decision that this expand loop skips build, measurement and reproduction (D385)."""
    data = {**row["data"], "analysis_only": f"set by you: {reason}"}
    svc.ops.put_loop(row["id"], row["kind"], row["subject"], row["state"], data)
    body = {"action": f"expand analysis-only {row['id']}", "reason": reason}
    actor = HUMAN.model_copy(update={"name": "kta expand --analysis-only"})
    refs: dict[str, RefValue] = {"reference": str(row["subject"])}
    svc.store.put(Record.new("HumanAction", body, actor, svc.ctx.clock.now(), refs=refs))


def _start(args: argparse.Namespace, make_ctx: ContextFactory, kind: str) -> int:
    with command(make_ctx, f"{kind}-start", f"kta {kind} {args.subject}") as ctx:
        svc = Services.open(ctx)
        rec = svc.store.get(args.subject)
        want = {"solve": "Case", "expand": "Reference"}[kind]
        if rec.kind != want:
            raise ConfigError(f"{args.subject} is a {rec.kind}, not a {want}")
        name = args.name or str(rec.body.get("name", args.subject[:12]))
        loop_id = Runner(svc, f"cli:{kind}").start(kind, args.subject, name)
        row = svc.ops.get_loop(loop_id)
        if getattr(args, "analysis_only", None) and row["state"] == "frozen":
            _analysis_only(svc, row, args.analysis_only)
        steps = []
        for _ in range(args.steps):
            t = Runner(svc, f"cli:{kind}").advance(loop_id)
            if t is None:
                break
            steps.append({"from": t.src, "to": t.to, "kind": t.kind, "note": t.note})
        row = svc.ops.get_loop(loop_id)
    data = {"loop": loop_id, "state": row["state"], "steps": steps, "data": row["data"]}
    lines = [f"loop {loop_id} ({kind}) state {row['state']}"]
    lines += [f"  {s['from']} -> {s['to'] or '(hold)'} {s['note']}" for s in steps]
    if row["data"].get("blocked"):
        b = row["data"]["blocked"]
        lines.append(f"  BLOCKED in {b['state']}: {b['error']}: {b['message']}")
    _out(args, data, "\n".join(lines))
    return 0


def cmd_loop_list(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    svc = Services.open(make_ctx("loop-list"))
    rows = svc.ops.list_loops(args.kind)
    out = [
        {
            "id": r["id"],
            "kind": r["kind"],
            "subject": r["subject"],
            "state": r["state"],
            "round": r["data"].get("round"),
            "name": r["data"].get("name"),
            "blocked": bool(r["data"].get("blocked")),
            "escalated": bool(r["data"].get("escalated")),
            "waiting_on": r["data"].get("waiting_on"),
        }
        for r in rows
    ]
    text = "\n".join(
        f"{r['id']} {r['kind']:6} {r['state']:12} round {r['round']} {r['name']}"
        + (" BLOCKED" if r["blocked"] else "")
        + (" ESCALATED" if r["escalated"] else "")
        for r in out
    )
    _out(args, {"loops": out}, text or "(no loops)")
    return 0


def cmd_loop_show(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    svc = Services.open(make_ctx("loop-show"))
    row = svc.ops.get_loop(args.id)
    m = machine(row["kind"])
    data = {**row, "updated_at": str(row["updated_at"]), "terminal": m.terminal(row["state"])}
    _out(args, data, yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    return 0


def cmd_loop_step(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "loop-step", f"kta loop step {args.id}") as ctx:
        svc = Services.open(ctx)
        t = Runner(svc, "cli:step").advance(args.id)
        row = svc.ops.get_loop(args.id)
    data: dict[str, Any] = {"loop": args.id, "state": row["state"]}
    if t is not None:
        data.update({"from": t.src, "to": t.to, "kind": t.kind, "rule": t.rule, "note": t.note})
    text = f"loop {args.id}: " + (
        f"{t.src} -> {t.to or '(hold)'} {t.note}" if t else f"no step ({row['state']})"
    )
    if row["data"].get("blocked"):
        text += f"\nBLOCKED: {row['data']['blocked']['message']}"
    _out(args, data, text)
    return 0


def cmd_process_start(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, args.name, f"kta process start {args.name}") as ctx:
        ticks = processes.run(Services.open(ctx), args.name, args.interval, args.ticks)
    _out(args, {"process": args.name, "ticks": ticks}, f"process {args.name}: {ticks} ticks")
    return 0


def cmd_process_stop(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    processes.request_stop(Services.open(make_ctx("process-stop")), args.name)
    _out(args, {"process": args.name, "stop": True}, f"stop requested for {args.name}")
    return 0


def cmd_inbox(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    items = inbox(Services.open(make_ctx("inbox")))
    text = "\n".join(f"- {i.trigger}: {i.subject}\n    {i.suggestion}" for i in items)
    _out(args, {"items": [i.to_dict() for i in items]}, text or "(inbox empty)")
    return 0


def cmd_budget(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    svc = Services.open(make_ctx("budget"))
    if args.set:
        key, _, value = args.set.partition("=")
        current = (
            budget.view(svc.ops, args.scope)["limit"]
            if budget.has_scope(svc.ops, args.scope)
            else {}
        )
        svc.ops.set_budget(args.scope, {**current, key: float(value)})
    if not budget.has_scope(svc.ops, args.scope):
        _out(args, {"scope": args.scope, "limit": None}, f"{args.scope}: no limit")
        return 0
    v = budget.view(svc.ops, args.scope)
    text = "\n".join(f"{k:6} {key}: {val}" for k, d in v.items() for key, val in d.items())
    _out(args, {"scope": args.scope, **v}, text or f"{args.scope}: empty")
    return 0


def cmd_coverage(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    ctx = make_ctx("coverage")
    rep = coverage_report(Services.open(ctx).store, args.snapshot, ctx.rng("coverage"))
    data = rep.to_dict()
    _out(args, data, "\n".join(f"{k}: {_short(v)}" for k, v in data.items()))
    return 0


def _short(value: Any, limit: int = 160) -> str:
    """One line per report key (D377): scalars as is, containers by size and a prefix."""
    if isinstance(value, dict | list) and value:
        text = json.dumps(value, default=str, sort_keys=True)
        size = f"{len(cast(Any, value))} {'keys' if isinstance(value, dict) else 'items'}"
        return f"[{size}] {text[:limit]}{'...' if len(text) > limit else ''}"
    return str(value)


def cmd_export(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    ctx = make_ctx("export")
    render_text_knowledge(Services.open(ctx).store, args.snapshot, Path(args.dir))
    _out(args, {"snapshot": args.snapshot, "dir": args.dir}, f"wrote {args.dir}")
    return 0


def add_loop_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    def parser(
        parent: argparse._SubParsersAction[argparse.ArgumentParser], name: str, fn: Any, h: str
    ) -> argparse.ArgumentParser:
        p = parent.add_parser(name, help=h)
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=lambda a: fn(a, make_ctx))
        return p

    case = sub.add_parser("case", help="problems, contracts and cases")
    parser(
        case.add_subparsers(dest="case_command", required=True),
        "add",
        cmd_case_add,
        "new Problem/Contract/Case from YAML",
    ).add_argument("file")
    ref = sub.add_parser("reference", help="reference implementations")
    parser(
        ref.add_subparsers(dest="reference_command", required=True),
        "add",
        cmd_reference_add,
        "new Source/Reference from YAML",
    ).add_argument("file")
    for kind, verb in (("solve", "solve"), ("expand", "expand")):
        p = parser(sub, verb, lambda a, m, k=kind: _start(a, m, k), f"start the {kind} loop")
        p.add_argument("subject", help="Case ID" if kind == "solve" else "Reference ID")
        p.add_argument("--name")
        p.add_argument("--steps", type=int, default=0, help="advance this many steps now")
        if kind == "expand":
            p.add_argument(
                "--analysis-only",
                metavar="REASON",
                help="decompose without building, measuring or reproducing (D385)",
            )
    loop = sub.add_parser("loop", help="loop instances")
    lc = loop.add_subparsers(dest="loop_command", required=True)
    parser(lc, "list", cmd_loop_list, "loop instances").add_argument("--kind")
    parser(lc, "show", cmd_loop_show, "one loop instance").add_argument("id")
    parser(lc, "step", cmd_loop_step, "advance one instance by one step").add_argument("id")
    proc = sub.add_parser("process", help="control processes")
    pc = proc.add_subparsers(dest="process_command", required=True)
    ps = parser(pc, "start", cmd_process_start, "run a process in the foreground until stopped")
    ps.add_argument("name", choices=processes.ALL)
    ps.add_argument("--interval", type=float, default=10.0)
    ps.add_argument("--ticks", type=int, help="stop after this many ticks")
    parser(pc, "stop", cmd_process_stop, "ask a running process to stop").add_argument(
        "name", choices=processes.ALL
    )
    parser(sub, "inbox", cmd_inbox, "items waiting for you")
    b = parser(sub, "budget", cmd_budget, "budget, used, reserved and remaining of a scope")
    b.add_argument("scope", nargs="?", default="global")
    b.add_argument("--set", help="<key>=<limit>, e.g. agent_min=600")
    parser(sub, "coverage", cmd_coverage, "coverage report summary").add_argument("--snapshot")
    e = parser(sub, "export", cmd_export, "export a snapshot as a browsable directory")
    e.add_argument("snapshot")
    e.add_argument("dir")
