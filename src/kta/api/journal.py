"""GET /journal: the system's history as readable lines, newest first (WP25, D660).

Sources: records (every write), spans (loop steps, agent sessions, jobs, commands), task
state changes (ops_events of the tasks table) and loops stopped by an error. Each event is
phrased by ``journal.yaml``; kinds the table does not list use its ``other`` phrase and are
counted in ``unphrased``. Runs of the same kind, case and actor close in time are grouped
into one line with a count, so 198 candidates read as one event.

Every line keeps what produced it (record id, span id, task id) so the page can open the
evidence; the text never drives a state change (AGENTS.md: interpretation is not state).
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from importlib import resources
from typing import Any

import yaml
from fastapi import APIRouter, Query

from kta.api import opsread
from kta.api.common import Api, iso
from kta.api.operate import parse_action
from kta.core.errors import ConfigError
from kta.store.sqlite import RecordNotFound

LEVELS = ("milestone", "detail")


def _when(value: str) -> datetime:
    """A stored timestamp in either form the system writes ("...Z" or "...+00:00")."""
    t = datetime.fromisoformat(value)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def phrases() -> dict[str, Any]:
    text = resources.files("kta.api").joinpath("journal.yaml").read_text("utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError("journal.yaml must be a mapping")
    return data  # pyright: ignore[reportUnknownVariableType]


class _Fields(string.Formatter):
    """str.format that renders a missing or unformattable placeholder as "?"."""

    def get_value(self, key: int | str, args: Any, kwargs: Any) -> Any:
        return kwargs.get(key, "?") if isinstance(key, str) else "?"

    def format_field(self, value: Any, format_spec: str) -> str:
        try:
            return super().format_field(value, format_spec)
        except (TypeError, ValueError):
            return str(value)


_FMT = _Fields()


def render(template: str, values: dict[str, Any]) -> str:
    return _FMT.format(template, **values)


@dataclass
class Event:
    at: datetime
    source: str  # record | span | task | blocked
    kind: str
    level: str
    tone: str
    template: str
    many: str | None
    values: dict[str, Any]
    case: str | None = None
    case_name: str | None = None
    actor: str = ""
    refs: list[str] = field(default_factory=list)
    count: int = 1
    phrased: bool = True
    oldest: datetime | None = None

    def key(self) -> tuple[str, str, str | None, str, str]:
        return (self.source, self.kind, self.case, self.actor, str(self.values.get("lane", "")))


class _Names:
    """Case id -> display name, cached for one request."""

    def __init__(self, api: Api) -> None:
        self.api = api
        self.cache: dict[str, str] = {}
        self.loops = {lp.id: lp for lp in opsread.loops(api.ops)}

    def loop(self, loop_id: str) -> str:
        """A loop id -> the name of its subject (case or reference)."""
        lp = self.loops.get(loop_id)
        if lp is None:
            return loop_id[-8:]
        return self.case(lp.subject) or lp.subject[:12]

    def case(self, cid: str | None) -> str | None:
        if not cid:
            return None
        if cid not in self.cache:
            try:
                b = self.api.store.get(cid).body
                prob = b.get("problem")
                name = self.api.store.get(prob).body.get("name") if prob else b.get("name")
                self.cache[cid] = str(name or cid[:12])
            except RecordNotFound:
                self.cache[cid] = cid[:12]
        return self.cache[cid]


def _record_event(api: Api, rid: str, kind: str, table: dict[str, Any], names: _Names) -> Event:
    rec = api.store.get_json(rid)
    body = rec.get("body", {})
    refs = rec.get("refs", {})
    actor = str(rec.get("actor", {}).get("name") or rec.get("actor", {}).get("type") or "")
    case = refs.get("case") if isinstance(refs.get("case"), str) else None
    if kind == "Case":
        case = rid
    sub = body.get("data", {}).get("kind") if isinstance(body.get("data"), dict) else None
    p = table.get(f"{kind}:{sub}") or table.get(kind)
    phrased = p is not None
    p = p or table["other"]
    values: dict[str, Any] = {k: v for k, v in body.items() if isinstance(v, str | int | float)}
    if kind == "HumanAction":
        values["action_words"] = _action_words(str(body.get("action", "")), names)
        values["reason_words"] = f" — {body['reason']}" if body.get("reason") else ""
    values.update(
        kind=kind,
        actor=actor,
        case=names.case(case) or "?",
        ok_word="succeeded" if body.get("ok") else "failed",
        passed_word="passed" if body.get("passed") else "failed",
        change_count=len(body.get("changes", [])),
    )
    if isinstance(body.get("pct_of_bound"), int | float):
        values["pct_of_bound_pct"] = f"{body['pct_of_bound'] * 100:.0f}%"
    tone = p["tone"]
    if (kind == "Build" and body.get("ok") is False) or (
        kind == "Check" and body.get("passed") is False
    ):
        tone = "fail"
    return Event(
        at=_when(rec["created_at"]),
        source="record",
        kind=kind,
        level=p["level"],
        tone=tone,
        template=p["text"],
        many=p.get("many"),
        values=values,
        case=case,
        case_name=names.case(case),
        actor=actor,
        refs=[rid],
        phrased=phrased,
    )


def _action_words(action: str, names: _Names) -> str:
    """A HumanAction's ``action`` in words: loop ids become case names, JSON becomes text."""
    parsed = parse_action(action)
    verb, params, args = parsed["verb"], parsed["params"] or {}, parsed["args"]
    loop = params.get("loop") or (args[5:] if args.startswith("loop:") else None)
    if loop:
        target = names.loop(str(loop))
        mode = params.get("mode")
        return f"{mode} {target}" if verb == "advance" and mode else f"{verb} {target}"
    if verb == "set_controls":
        return f"new controls version {params.get('version', '')}".strip()
    return f"{verb} {args}".strip() if args else verb


def _span_event(
    s: Any, table: dict[str, Any], loops: dict[str, Any], names: _Names
) -> Event | None:
    if s.kind in table.get("skip", []):
        return None
    kind = s.kind
    if kind == "job_run" and s.status == "fail":
        kind = "job_run_fail"
    p = table.get(kind)
    phrased = p is not None
    p = p or table["other"]
    loop = loops.get(s.trace_id)
    case = loop.subject if loop is not None and loop.kind == "solve" else None
    values: dict[str, Any] = {k: v for k, v in s.attrs.items() if isinstance(v, str | int | float)}
    values.update(
        name=s.name,
        lane=s.lane,
        kind=s.kind,
        case=names.case(case) or (loop.subject[:12] if loop is not None else "?"),
        error=s.attrs.get("error_kind") or s.status,
    )
    return Event(
        at=s.start_ts,
        source="span",
        kind=kind,
        level=p["level"],
        tone="fail" if s.status == "fail" else p["tone"],
        template=p["text"],
        many=p.get("many"),
        values=values,
        case=case,
        case_name=names.case(case),
        actor=s.lane,
        refs=[s.span_id],
        phrased=phrased,
    )


def _group(events: list[Event], group_s: float) -> list[Event]:
    """Merge events into one line when they share a key (kinds with a ``many`` phrase) or
    render the same text, and each lies within ``group_s`` of the group's oldest member.
    Members need not be adjacent: 198 candidates interleaved with predictions still give
    one line each."""
    out: list[Event] = []
    open_: dict[tuple[Any, ...], Event] = {}
    for e in events:  # newest first
        gk: tuple[Any, ...] = e.key() if e.many else (*e.key(), render(e.template, e.values))
        g = open_.get(gk)
        if g is not None and g.oldest and (g.oldest - e.at).total_seconds() <= group_s:
            g.count += 1
            g.oldest = e.at
            if len(g.refs) < 50:
                g.refs.append(e.refs[0])
            continue
        e.oldest = e.at
        open_[gk] = e
        out.append(e)
    return out


def journal(
    api: Api,
    before: datetime | None,
    limit: int,
    level: str,
    case: str | None,
) -> dict[str, Any]:
    table = phrases()
    names = _Names(api)
    before = before or api.now() + timedelta(seconds=1)
    raw = max(400, limit * 40)
    loops = {lp.id: lp for lp in opsread.loops(api.ops)}

    events: list[Event] = []
    for rid, kind, _at in opsread.records_before(api.store, iso(before) or "", raw):
        events.append(_record_event(api, rid, kind, table["records"], names))
    for s in opsread.spans_before(api.ops, before, raw):
        e = _span_event(s, table["spans"], loops, names)
        if e is not None:
            events.append(e)
    tasks = {t.id: t for t in opsread.tasks(api.ops)}
    tp = table["tasks"]["state"]
    for ev in opsread.task_events_before(api.ops, before, raw):
        state = ev.change.get("state")
        if not state:
            continue
        t = tasks.get(ev.row_id)
        lp = loops.get(t.work_unit) if t else None
        cid = lp.subject if lp is not None and lp.kind == "solve" else None
        events.append(
            Event(
                at=ev.at,
                source="task",
                kind="task",
                level=tp["level"],
                tone="fail" if state in ("rejected", "expired") else tp["tone"],
                template=tp["text"],
                many=None,
                values={"type": t.card_type if t else "?", "state": state, "task": ev.row_id},
                case=cid,
                case_name=names.case(cid),
                actor="orchestrator",
                refs=[ev.row_id],
            )
        )
    bp = table["blocked"]
    for lp in loops.values():
        b = lp.data.get("blocked")
        if not b or not b.get("at"):
            continue
        at = _when(b["at"])
        if at >= before:
            continue
        cid = lp.subject if lp.kind == "solve" else None
        events.append(
            Event(
                at=at,
                source="blocked",
                kind="blocked",
                level=bp["level"],
                tone=bp["tone"],
                template=bp["text"],
                many=None,
                values={
                    "case": names.case(cid) or lp.subject[:12],
                    "state": b.get("state", lp.state),
                    "error": b.get("error", "?"),
                    "message": b.get("message", ""),
                },
                case=cid,
                case_name=names.case(cid),
                actor="orchestrator",
                refs=[lp.id],
            )
        )

    events.sort(key=lambda e: e.at, reverse=True)
    if case:
        events = [e for e in events if e.case == case]
    unphrased: dict[str, int] = {}
    for e in events:
        if not e.phrased:
            unphrased[e.kind] = unphrased.get(e.kind, 0) + 1
    grouped = _group(events, float(table.get("group_s", 120)))
    if level == "milestone":
        grouped = [e for e in grouped if e.level == "milestone"]
    page = grouped[:limit]
    complete = len(grouped) > limit or len(events) < raw
    return {
        "entries": [_json(e) for e in page],
        "next_before": iso(page[-1].at) if page and complete else None,
        "unphrased": unphrased,
        "scanned": len(events),
    }


def _json(e: Event) -> dict[str, Any]:
    if e.count > 1 and e.many:
        text = render(e.many, {**e.values, "n": e.count})
    else:
        text = render(e.template, e.values) + (f" (×{e.count})" if e.count > 1 else "")
    return {
        "at": iso(e.at),
        "source": e.source,
        "kind": e.kind,
        "level": e.level,
        "tone": e.tone,
        "text": text,
        "count": e.count,
        "case": e.case,
        "case_name": e.case_name,
        "actor": e.actor,
        "refs": e.refs,
    }


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/journal")
    def get_journal(  # pyright: ignore[reportUnusedFunction]
        before: str | None = None,
        limit: int = Query(60, ge=1, le=500),
        level: str = Query("milestone", pattern="^(milestone|all)$"),
        case: str | None = None,
    ) -> dict[str, Any]:
        return journal(api, _when(before) if before else None, limit, level, case)

    return r
