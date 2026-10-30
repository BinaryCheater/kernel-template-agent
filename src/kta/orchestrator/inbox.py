"""Pending items for you (docs/design/30 §2), derived from stored state (D270).

Items are not stored: each is computed from ops rows and records, so it disappears when
its cause is handled. The count is the "attention" resource of docs/design/11 §2: at the
limit, loops defer transitions that would create tasks.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from kta.orchestrator import budget
from kta.orchestrator.pipeline import Services
from kta.orchestrator.transition import Settings

REPEAT_FAILURES = 3
BUDGET_WARN = 0.8


@dataclass(frozen=True)
class Item:
    id: str  # stable key of the cause
    trigger: str
    subject: str
    suggestion: str
    record: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tasks(svc: Services) -> list[Item]:
    out: list[Item] = []
    for t in svc.ops.list_tasks(states=("escalated", "suspended")):
        goal = str(svc.store.get(t.card_id).body.get("goal", ""))
        if t.state == "escalated":
            raw = svc.ops.get_pointer(f"task:{t.id}") or "{}"
            last = json.loads(raw).get("last_rejection")
            out.append(
                Item(
                    f"task:{t.id}",
                    "Task over retry limit",
                    f"{t.card_type} {goal}",
                    "Retry with a note, modify the task, or drop it",
                    last,
                )
            )
        else:
            out.append(
                Item(
                    f"task:{t.id}",
                    "Checker crashed",
                    f"{t.card_type} {goal}",
                    "Fix the checker, then `kta task recheck`",
                    t.card_id,
                )
            )
    return out


def _loops(svc: Services) -> list[Item]:
    out: list[Item] = []
    for row in svc.ops.list_loops():
        d, name = row["data"], str(row["data"].get("name", row["subject"]))
        if d.get("escalated"):
            out.append(
                Item(
                    f"loop:{row['id']}",
                    "Loop needs you (round limit or stuck task)",
                    name,
                    "Look at the Gaps and the last judgement",
                    row["subject"],
                )
            )
        if d.get("blocked"):
            b = d["blocked"]
            out.append(
                Item(
                    f"blocked:{row['id']}",
                    f"Loop blocked in {b['state']}",
                    f"{name}: {b['error']}: {b['message'][:200]}",
                    "Fix the cause, then start the loop again",
                    row["subject"],
                )
            )
        if d.get("suspicious") and d.get("judgement"):
            out.append(
                Item(
                    f"suspicious:{row['id']}",
                    "Result suspiciously good",
                    name,
                    "Re-verify the measurement and the checks",
                    str(d["judgement"]),
                )
            )
    return out


def _repeated_failures(svc: Services) -> list[Item]:
    by_check: Counter[str] = Counter()
    for rec in svc.store.query("Report", {"title": "done_checks"}, limit=5000):
        for c in rec.body["data"].get("checks", []):
            if not c["passed"]:
                by_check[c["name"]] += 1
    return [
        Item(
            f"check:{name}",
            "Same done-check failing repeatedly",
            f"{name} x{n}",
            "Check the checker and the failing samples",
        )
        for name, n in sorted(by_check.items())
        if n >= REPEAT_FAILURES
    ]


def _model_error(svc: Services) -> list[Item]:
    recent = svc.store.query("Report", {"title": "judgement"}, order="-created_at", limit=3)
    if len(recent) == REPEAT_FAILURES and all(r.body["data"]["rule"] == 2 for r in recent):
        return [
            Item(
                "model:error",
                "Model error large repeatedly",
                "last 3 judgements rule 2",
                "Start a model diagnosis (T4)",
            )
        ]
    return []


def _budgets(svc: Services) -> list[Item]:
    out: list[Item] = []
    for scope in ("global", *(f"process:{p}" for p in ("solve", "learn", "maintain"))):
        if not budget.has_scope(svc.ops, scope):
            continue
        v = budget.view(svc.ops, scope)
        for key, limit in v["limit"].items():
            spent = v["used"].get(key, 0.0) + v["reserved"].get(key, 0.0)
            if limit > 0 and spent >= BUDGET_WARN * limit:
                out.append(
                    Item(
                        f"budget:{scope}:{key}",
                        "Budget at 80%",
                        f"{scope} {key} {spent:.1f}/{limit:.1f}",
                        "Raise it or stop",
                    )
                )
    return out


def inbox(svc: Services) -> list[Item]:
    return [
        *_tasks(svc),
        *_loops(svc),
        *_repeated_failures(svc),
        *_model_error(svc),
        *_budgets(svc),
    ]


def at_attention_limit(svc: Services) -> bool:
    return len(inbox(svc)) >= Settings.from_config(svc.ctx.config).attention_limit
