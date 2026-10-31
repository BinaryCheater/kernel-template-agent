"""Task header, pull log and plan view of the access discipline (WP19; 02 §5.4, D-T20).

- ``situation_of`` / ``task_header``: the one-screen header of a case-loop task (case;
  bound; last round's gap decomposition; default plan and score; budget; situation) as a
  ``TaskHeader`` for ``TaskRequest.header``. The gap is a mapping of the components of
  ``T_meas - T_SOL`` (eps, delta, dpre, l_rep, irreducible) in us, plus ``class`` (the
  attributed class) and, for an L_rep gap, ``idle`` (the idle resource): WP18's record
  provides them.
- ``record_pull``: one Pull record per knowledge fetch of a worker, classified against the
  table version in force (pulls outside the situation's row are allowed and logged).
- ``pull_log``: the pull-log view of a task (rows and counts); ``plan_view``: the plan a
  task's header names, with its diff and score against its default when both exist.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from kta.core.access import table
from kta.core.errors import ConfigError
from kta.core.schemas import Actor, Record
from kta.core.schemas.plan import FormulationPlan, Pull, TaskHeader
from kta.core.schemas.process import TaskCard
from kta.solve.plan import diff_plans
from kta.store.base import Store

HEADER_MAX_LINES = 24  # default of knob session.task_header_max_lines
HEADER_MAX_COLS = 100
GAP_ORDER = ("eps", "delta", "dpre", "l_rep", "irreducible")  # attribution order (D-T6)


def situation_of(gap: Mapping[str, Any] | None) -> str:
    """The access-discipline row of a case task from the last round's gap."""
    if not gap:
        return "start"
    cls = str(gap.get("class", ""))
    if cls in ("eps", "ε"):
        return "eps"
    if cls == "wall":
        return "wall"
    if cls == "l_rep" and gap.get("idle"):
        return "idle_resource"
    if cls == "stop":
        return "stop"
    return "diagnostic"


def _clip(line: str) -> str:
    return line if len(line) <= HEADER_MAX_COLS else line[: HEADER_MAX_COLS - 1] + "…"


def task_header(
    *,
    case: str,
    case_id: str | None,
    bound_us: float | None,
    bottleneck: str | None,
    gap: Mapping[str, Any] | None,
    plan_id: str | None,
    score: Mapping[str, Any] | None,
    budget: Mapping[str, Any],
    discipline: str,
    max_lines: int = HEADER_MAX_LINES,
    situation: str | None = None,
) -> TaskHeader:
    """The one-screen header; ConfigError when it does not fit ``max_lines``."""
    t = table(discipline)
    sit = situation or situation_of(gap)
    row = t.row(sit)
    lines = [f"## Case header (access discipline {discipline})", ""]
    lines.append(f"- case: {case}" + (f" (`{case_id}`)" if case_id else ""))
    if bound_us is not None:
        lines.append(
            f"- bound (SOL): {bound_us:.4g} us" + (f", on {bottleneck}" if bottleneck else "")
        )
    else:
        lines.append("- bound (SOL): not available")
    if gap:
        parts = ", ".join(f"{k} {float(gap[k]):.3g}" for k in GAP_ORDER if k in gap)
        extra = f"; idle: {gap['idle']}" if gap.get("idle") else ""
        lines.append(f"- last round's gap (us): {parts}; class {gap.get('class', '?')}{extra}")
    else:
        lines.append("- last round's gap: none (first round)")
    if plan_id:
        s = dict(score or {})
        keys = ("changes", "open_dims", "space", "bound_us")
        summary = ", ".join(f"{k} {s[k]:.4g}" if isinstance(s.get(k), float) else f"{k} {s.get(k)}"
                            for k in keys if k in s)  # fmt: skip
        lines.append(f"- default plan: `{plan_id}`" + (f" ({summary})" if summary else ""))
    else:
        lines.append("- default plan: none")
    if budget:
        lines.append("- budget: " + ", ".join(f"{k} {v}" for k, v in sorted(budget.items())))
    lines += ["", f"**Situation: {row.situation}** (`{row.id}`)", f"- fetch: {row.fetch}",
              f"- decide: {row.decide}"]  # fmt: skip
    lines.append("- pulls outside this row are allowed and logged against this task")
    lines = [_clip(x) for x in lines]
    if len(lines) > max_lines:
        raise ConfigError(f"task header has {len(lines)} lines, above {max_lines}")
    return TaskHeader(
        text="\n".join(lines), situation=sit, discipline=discipline, case=case_id, plan=plan_id
    )


def card_header(store: Store, card_id: str) -> TaskHeader | None:
    return TaskCard.model_validate(store.get(card_id).body).header


def record_pull(
    store: Store,
    task: str,
    header: TaskHeader | None,
    discipline: str,
    command: str,
    target: str,
    kind: str | None,
    at: datetime,
) -> str:
    """Write one Pull record; ``row`` is None when the pull is outside the table."""
    version = header.discipline if header else discipline
    situation = header.situation if header else ""
    row = table(version).classify(command, kind, situation) if situation else None
    body = Pull(task=task, command=command, target=target[:200], kind=kind,
                situation=situation, row=row, discipline=version)  # fmt: skip
    actor = Actor(type="system", name="kta.api (pull log)", task=task)
    return store.put(Record.new("Pull", body.model_dump(mode="json"), actor, at))


def pull_log(store: Store, task: str) -> dict[str, Any]:
    """The pull-log view of one task: rows in order and in-table / outside counts."""
    rows = [
        Pull.model_validate(r.body) for r in store.query("Pull", where={"task": task}, limit=10_000)
    ]
    by_row = Counter(p.row or "outside" for p in rows)
    return {
        "task": task,
        "pulls": [p.model_dump(mode="json") for p in rows],
        "in_table": sum(1 for p in rows if p.row),
        "outside": sum(1 for p in rows if not p.row),
        "by_row": dict(sorted(by_row.items())),
        "by_command": dict(sorted(Counter(p.command for p in rows).items())),
    }


def pull_stats(store: Store) -> dict[str, Any]:
    """Pull logs across tasks (the system loop's input to revise the table, 02 §6.2)."""
    rows = [Pull.model_validate(r.body) for r in store.query("Pull", limit=100_000)]
    outside = Counter(
        f"{p.situation or '-'}: {p.command} {p.kind or ''}".strip() for p in rows if not p.row
    )
    return {
        "pulls": len(rows),
        "in_table": sum(1 for p in rows if p.row),
        "by_version": dict(sorted(Counter(p.discipline for p in rows).items())),
        "outside_by_situation_and_pull": dict(outside.most_common()),
    }


def plan_view(store: Store, plan_id: str) -> dict[str, Any]:
    """A FormulationPlan record with its default's ID and the stored diff, if any."""
    rec = store.get(plan_id)
    plan = FormulationPlan.model_validate(rec.body)
    out: dict[str, Any] = {"id": plan_id, "plan": plan.model_dump(mode="json")}
    default = rec.refs.get("default")
    if isinstance(default, str):
        d = FormulationPlan.model_validate(store.get(default).body)
        out["default"] = default
        out["diff"] = [c.__dict__ for c in diff_plans(d, plan)]
    return out
