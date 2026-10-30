"""Agent-facing commands (docs/design/10 §5). No task-token authentication yet (WP8).

The task comes from ``KTA_TASK`` (set by the runner) or ``--task``. Output is short by
default; ``--full`` prints whole records. ``instantiate``, ``build`` and ``sass`` work on a
Family record on the control plane (D371); ``run`` and ``src`` raise NotSupported. Inside
an agent workspace (``KTA_TASK_TOKEN`` set) the CLI is an API client (client.py).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from kta.checks.sass import sass_check
from kta.cli.exec_cmds import ContextFactory, _out, command
from kta.cli.task_cmds import NOTE_FILE, _queue
from kta.core.errors import ConfigError, NotSupported
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Case
from kta.core.schemas.knowledge import Template
from kta.knowledge import (
    card,
    load_family,
    nearest_chains,
    retrieve_families,
    retrieve_templates,
    validate_family,
    validate_template,
)
from kta.orchestrator.pipeline import (
    Services,
    build,
    chain_sass,
    family_package,
    files_of,
    generate,
)

SHORT_BODY = 1200


def _task(args: argparse.Namespace) -> str | None:
    return args.task or os.environ.get("KTA_TASK")


def cmd_show(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "show", f"kta show {args.id}") as ctx:
        rec = Services.open(ctx).store.get(args.id)
    data = rec.to_json_dict()
    text = json.dumps(data["body"], indent=1, default=str)
    if not args.full and len(text) > SHORT_BODY:
        text = text[:SHORT_BODY] + f"\n... ({len(text)} chars; --full for all)"
    _out(args, data, f"{rec.kind} {args.id}\nrefs {rec.refs}\n{text}")
    return 0


def cmd_query(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "query", f"kta query {args.what} {args.term or ''}") as ctx:
        store = Services.open(ctx).store
        if args.what == "tmpl":
            cards = retrieve_templates(store, text=args.term, k=args.k)
        else:
            subject = args.case or args.near or args.term
            if not subject:
                raise ConfigError(f"kta query {args.what} needs --case or --near")
            rec = store.get(subject)
            case = Case.model_validate(rec.body) if rec.kind == "Case" else None
            if case is None:
                raise NotSupported("query near a family: give the Case ID (--case)")
            cards = (
                retrieve_families(store, case, k=args.k)
                if args.what == "pool"
                else nearest_chains(store, case, k=args.k)
            )
    data = {
        "cards": [{"id": c.id, "kind": c.kind, "name": c.name, "lines": c.lines} for c in cards]
    }
    text = "\n\n".join(f"{c.kind} {c.id[:16]} {c.name}\n" + "\n".join(c.lines) for c in cards)
    _out(args, data, text or "(nothing found)")
    return 0


def cmd_why(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "why", f"kta why {args.id}") as ctx:
        g = Services.open(ctx).store.lineage(args.id, depth=args.depth)
    data = {"root": g.root, "nodes": g.nodes, "edges": g.edges}
    text = "\n".join(
        f"{a[:16]} --{rel}--> {b[:16]} ({g.nodes.get(b, '?')})" for a, rel, b in g.edges
    )
    _out(args, data, text or f"{args.id}: no evidence links")
    return 0


def cmd_card(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "card", f"kta card {args.id}") as ctx:
        c = card(Services.open(ctx).store, args.id)
    _out(args, {"id": c.id, "lines": c.lines}, "\n".join(c.lines))
    return 0


def cmd_note(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    task = _task(args)
    if task is None:
        raise ConfigError("kta note needs a task (KTA_TASK or --task)")
    with command(make_ctx, "note", "kta note") as ctx:
        q = _queue(ctx)
        t = q.get(task)
        body = {"title": "note", "subject": t.card_id, "summary": args.text, "data": {}}
        actor = Actor(
            type="agent" if os.environ.get("KTA_TASK") else "human", name="kta note", task=task
        )
        rid = q.store.put(
            Record.new("Report", body, actor, ctx.clock.now(), refs={"card": t.card_id})
        )
    _out(args, {"note": rid}, f"note {rid}")
    return 0


def cmd_submit(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """Agent path: submit ``out/`` (delivery note ``out/DELIVERY.md``) of a running task."""
    task = _task(args)
    if task is None:
        raise ConfigError("kta submit needs a task (KTA_TASK or --task)")
    root = Path(args.dir)
    note = root / NOTE_FILE
    if not note.is_file():
        raise ConfigError(f"the delivery note {note} is missing")
    with command(make_ctx, "submit", f"kta submit {task}") as ctx:
        q = _queue(ctx)
        owner = q.get(task).lease_owner or ""
        files = {k: v for k, v in files_of(root).items() if Path(k) != Path(NOTE_FILE)}
        sub = q.submit(
            task,
            owner,
            files,
            note.read_text(encoding="utf-8"),
            Actor(type="agent", name="kta submit"),
        )
    _out(args, {"task": task, "submission": sub}, f"submitted {sub}")
    return 0


def cmd_validate_template(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "validate-template", f"kta validate-template {args.file}") as ctx:
        doc = yaml.safe_load(Path(args.file).read_text(encoding="utf-8"))
        body = doc.get("template", doc) if isinstance(doc, dict) else doc
        issues = validate_template(Template.model_validate(body), Services.open(ctx).store)
    return _issues(args, issues)


def cmd_validate_family(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "validate-family", f"kta validate-family {args.dir}") as ctx:
        issues = validate_family(load_family(Path(args.dir)), Services.open(ctx).store)
    return _issues(args, issues)


def _issues(args: argparse.Namespace, issues: list[Any]) -> int:
    rows = [
        {"kind": i.kind, "message": i.message, "where": i.where, "severity": i.severity}
        for i in issues
    ]
    errors = [r for r in rows if r["severity"] == "error"]
    text = "\n".join(f"{r['severity'].upper():7} {r['where']}: {r['message']}" for r in rows)
    _out(args, {"issues": rows, "ok": not errors}, text or "OK: no issues")
    return 1 if errors else 0


def _generated(args: argparse.Namespace, svc: Services) -> tuple[Any, Any, dict[str, Any]]:
    fp = family_package(svc, args.family)
    assignment = json.loads(args.assignment)
    return fp, generate(fp, assignment, json.loads(args.shape)), assignment


def cmd_instantiate(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """Generate a family's source for one assignment and shape (no compilation)."""
    with command(make_ctx, "instantiate", f"kta instantiate {args.family}") as ctx:
        _, gk, _ = _generated(args, Services.open(ctx))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in gk.files.items():
        (out / name).write_text(text, encoding="utf-8")
    (out / "launch.json").write_text(json.dumps(gk.launch.to_dict(), indent=1), "utf-8")
    data = {"dir": str(out), "files": sorted(gk.files), "entry": gk.entry, "markers": gk.markers}
    _out(args, data, f"wrote {', '.join(sorted(gk.files))} and launch.json to {out}")
    return 0


def cmd_build(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """Generate and compile through the Executor; prints the Build feedback."""
    with command(make_ctx, "build", f"kta build {args.family}") as ctx:
        svc = Services.open(ctx)
        _, gk, _ = _generated(args, svc)
        b, module = build(svc, gk)
    data = {"build": module.build_id, **b.model_dump()}
    _out(
        args,
        data,
        f"build {module.build_id} ok={b.ok} regs={b.regs} spills={b.spills} "
        f"smem={b.smem}\n{b.log_summary[:600]}",
    )
    return 0 if b.ok else 1


def cmd_sass(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """Build, then compare the SASS with the chain's expectations (D361)."""
    with command(make_ctx, "sass", f"kta sass {args.family}") as ctx:
        svc = Services.open(ctx)
        fp, gk, assignment = _generated(args, svc)
        b, module = build(svc, gk)
        if not b.ok or not module.sass_blob:
            raise ConfigError(f"build {module.build_id[:12]} failed or has no SASS")
        expect = chain_sass(svc.store, fp, assignment)
        checked = sass_check(svc.checker(), module, expect, assignment)
    data = {"check": checked.id, "passed": checked.passed, "failures": checked.body.failures}
    _out(
        args,
        data,
        f"sass check {checked.id[:12]} passed={checked.passed}\n"
        + "\n".join(str(f) for f in checked.body.failures),
    )
    return 0 if checked.passed else 1


def cmd_not_yet(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    why = {
        "run": "a quick run needs input buffers per Problem; use the loop's checks (kta check)",
        "src": "indexed source search of references arrives with WP10b",
    }
    raise NotSupported(f"kta {args.command}: {why.get(args.command, 'not provided')}")


def add_agent_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    def parser(name: str, fn: Any, h: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=h)
        p.add_argument("--json", action="store_true")
        p.add_argument("--task", help="task ID (default: $KTA_TASK)")
        p.set_defaults(func=lambda a: fn(a, make_ctx))
        return p

    s = parser("show", cmd_show, "one record")
    s.add_argument("id")
    s.add_argument("--full", action="store_true")
    q = parser("query", cmd_query, "retrieve families (pool), Templates (tmpl) or chains")
    q.add_argument("what", choices=["pool", "tmpl", "chain"])
    q.add_argument("term", nargs="?")
    q.add_argument("--case")
    q.add_argument("--near")
    q.add_argument("-k", type=int, default=5)
    w = parser("why", cmd_why, "the evidence links of a record")
    w.add_argument("id")
    w.add_argument("--depth", type=int, default=4)
    parser("card", cmd_card, "the short card of a knowledge record").add_argument("id")
    parser("note", cmd_note, "attach a note to the current task").add_argument("text")
    parser("submit", cmd_submit, "submit out/ of the current task").add_argument(
        "--dir", default="out"
    )
    parser("validate-template", cmd_validate_template, "check a Template file").add_argument("file")
    parser("validate-family", cmd_validate_family, "check a family directory").add_argument("dir")
    for name, fn, h in (
        ("instantiate", cmd_instantiate, "generate a family's source for one assignment"),
        ("build", cmd_build, "generate and compile; print the compile feedback"),
        ("sass", cmd_sass, "build and check the SASS against the chain's expectations"),
    ):
        p = parser(name, fn, h)
        p.add_argument("family", help="Family record ID")
        p.add_argument("--assignment", required=True, help="JSON object of family variables")
        p.add_argument("--shape", required=True, help='JSON object, e.g. {"M": 256}')
        if name == "instantiate":
            p.add_argument("--out", default="kernel")
    for name in ("run", "src"):
        p = parser(name, cmd_not_yet, f"{name} (not provided yet)")
        p.add_argument("args", nargs="*")
