"""``kta task ...``: the task queue (WP6). Agents do not exist yet (WP8); ``kta task submit``
is the human/CLI submission path: it leases, starts and submits in one step (D269)."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path
from typing import Any

from kta.cli.exec_cmds import ContextFactory, _out, command
from kta.core.context import Context
from kta.core.errors import ConfigError
from kta.core.schemas import Actor
from kta.orchestrator.pipeline import Services, files_of
from kta.orchestrator.tasks import TaskQueue, Verdict
from kta.store.base import Task

NOTE_FILE = "DELIVERY.md"


def _queue(ctx: Context) -> TaskQueue:
    svc = Services.open(ctx)
    return TaskQueue(ctx, svc.store, svc.ops)


def task_row(q: TaskQueue, t: Task) -> dict[str, Any]:
    card = q.card(t)
    return {
        "id": t.id,
        "type": t.card_type,
        "state": t.state,
        "goal": card.goal,
        "loop": t.work_unit,
        "attempts": t.attempts,
        "owner": t.lease_owner,
        "card": t.card_id,
    }


def cmd_list(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    q = _queue(make_ctx("task-list"))
    states = args.state.split(",") if args.state else None
    rows = [task_row(q, t) for t in q.ops.list_tasks(args.loop, states)]
    text = "\n".join(
        f"{r['id']}  {r['type']}  {r['state']:10} {r['goal'][:60]}  (loop {r['loop'][:10]})"
        for r in rows
    )
    _out(args, {"tasks": rows}, text or "(no tasks)")
    return 0


def cmd_show(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    q = _queue(make_ctx("task-show"))
    t = q.get(args.id)
    row = {**task_row(q, t), "card_body": q.card(t).model_dump(), "info": q.info(t.id)}
    lines = [f"{k:10} {v}" for k, v in row.items() if k not in ("card_body",)]
    lines += [f"inputs     {' '.join(row['card_body']['inputs'])}"]
    lines += [f"checks     {', '.join(row['card_body']['done_checks'])}"]
    _out(args, row, "\n".join(lines))
    return 0


def _verdict(args: argparse.Namespace, v: Verdict) -> int:
    data = {
        "task": v.task_id,
        "state": v.state,
        "report": v.report_id,
        "checks": [o.to_dict() for o in v.outcomes],
    }
    lines = [f"task {v.task_id}: {v.state} (report {v.report_id})"]
    for o in v.outcomes:
        lines.append(f"  {'PASS' if o.passed else 'FAIL'} {o.name}")
        lines += [f"       {f}" for f in o.failures]
    _out(args, data, "\n".join(lines))
    return 0 if v.state == "accepted" else 1


def cmd_submit(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """Human/CLI path: lease (with budget), start, submit a directory, optionally check."""
    with command(make_ctx, "task-submit", f"kta task submit {args.id}") as ctx:
        q = _queue(ctx)
        root = Path(args.dir)
        note_path = Path(args.note) if args.note else root / NOTE_FILE
        if not root.is_dir() or not note_path.is_file():
            raise ConfigError(f"need a directory {root} and a delivery note {note_path}")
        files = {k: v for k, v in files_of(root).items() if Path(k) != Path(NOTE_FILE)}
        owner = f"human:{getpass.getuser()}"
        t = q.get(args.id)
        if t.state == "queued":
            q.lease(t.id, owner)
            t = q.get(args.id)
        if t.state == "leased":
            q.start(t.id, owner)
        actor = Actor(type="human", name="kta task submit")
        sub = q.submit(t.id, owner, files, note_path.read_text(encoding="utf-8"), actor)
        if not args.check:
            _out(args, {"task": t.id, "submission": sub}, f"task {t.id} submitted ({sub})")
            return 0
        return _verdict(args, q.check(t.id))


def cmd_check(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "task-check", f"kta task check {args.id}") as ctx:
        return _verdict(args, _queue(ctx).check(args.id))


def cmd_recheck(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "task-recheck", f"kta task recheck {args.id}") as ctx:
        return _verdict(args, _queue(ctx).recheck(args.id, args.reason))


def cmd_cancel(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "task-cancel", f"kta task cancel {args.id}") as ctx:
        _queue(ctx).cancel(args.id, args.reason)
    _out(args, {"task": args.id, "state": "cancelled"}, f"task {args.id} cancelled")
    return 0


def cmd_retry(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "task-retry", f"kta task retry {args.id}") as ctx:
        _queue(ctx).retry(args.id, args.reason)
    _out(args, {"task": args.id, "state": "queued"}, f"task {args.id} queued again")
    return 0


def add_task_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    task = sub.add_parser("task", help="the task queue (T1-T6)")
    tc = task.add_subparsers(dest="task_command", required=True)

    def add(name: str, fn: Any, help_: str, *, reason: bool = False) -> argparse.ArgumentParser:
        p = tc.add_parser(name, help=help_)
        if name != "list":
            p.add_argument("id")
        if reason:
            p.add_argument("--reason", required=True)
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=lambda a: fn(a, make_ctx))
        return p

    ls = add("list", cmd_list, "tasks, oldest first")
    ls.add_argument("--loop", help="only tasks of this loop instance")
    ls.add_argument("--state", help="comma-separated lifecycle states")
    add("show", cmd_show, "one task with its card")
    s = add("submit", cmd_submit, "submit a directory for a task (human/CLI path)")
    s.add_argument("--dir", required=True, help="the delivered files")
    s.add_argument("--note", help=f"delivery note (default <dir>/{NOTE_FILE})")
    s.add_argument("--check", action="store_true", help="run the done-checks now")
    add("check", cmd_check, "run the done-checks of a submitted task")
    add("recheck", cmd_recheck, "rerun checks of a suspended task", reason=True)
    add("cancel", cmd_cancel, "cancel a task", reason=True)
    add("retry", cmd_retry, "queue an escalated task again", reason=True)
