"""Check a Template once and write its proof records, reusing certificates whose premises
did not change (paper §4.2, §4.4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kta.core.errors import SchemaError
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import Template
from kta.core.schemas.proof import ProofOutline
from kta.store.sqlite import SqliteStore

from . import records as rc
from .encode import Sym
from .models import GROUPS
from .smt import DEFAULT_TIMEOUT_MS
from .template import TemplateCheck, check_template, fold
from .types import Draft

CHECKER = Actor(type="checker", name="kta prove")


@dataclass
class Proved:
    summary: str
    level: str | None
    results: list[str]
    reused: int
    queries: int
    gaps: list[str]
    check: TemplateCheck


def template_ref(t: Template | Mapping[str, Any]) -> str:
    b = t.model_dump() if isinstance(t, Template) else t
    return f"{b['name']}@{b['version']}"


def find_template(store: SqliteStore, ref: str) -> str:
    name, _, version = ref.partition("@")
    found = store.query("Template", where={"name": name, "version": int(version)}, limit=1)
    if not found:
        raise SchemaError(f"Template {ref} is not in the store")
    return record_id(found[0])


def outline_for(store: SqliteStore, ref: str) -> tuple[str, ProofOutline] | None:
    """The newest live ProofOutline of a Template reference."""
    best: tuple[int, Record] | None = None
    for o in store.query("ProofOutline", where={"template": ref}, limit=100):
        if rc.alive(store, record_id(o)) and (best is None or o.body["version"] > best[0]):
            best = (o.body["version"], o)
    if best is None:
        return None
    return record_id(best[1]), ProofOutline.model_validate(best[1].body)


def _expected_groups(
    t: Template, outline: ProofOutline | None
) -> dict[tuple[str, str], tuple[str, ...]]:
    out: dict[tuple[str, str], tuple[str, ...]] = {
        ("K1", p.id): ("gamma", "assumes", "X", "C", f"pre:{p.id}") for p in t.preconditions
    }
    out[("K3", "*")] = ("gamma", "assumes", "X", "C")
    for b in outline.models if outline else []:
        out[("K2", b.obligation)] = GROUPS.get(b.model, ("gamma", "assumes", "X", "pre", "program"))
    return out


def _from_record(rec: Record, groups: tuple[str, ...]) -> Draft:
    b = rec.body
    return Draft(
        b["condition"],
        b["obligation"],
        b["outcome"],
        b["domain"],
        reason=b["reason"],
        data=dict(b["data"]),
        groups=groups,
    )


def prove_template(
    store: SqliteStore,
    template_id: str,
    now: datetime,
    *,
    actor: Actor = CHECKER,
    subject: str | None = None,
    scope: str = "template",
    part: str = "",
    restrict: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Sym] | None = None,
    menus: Mapping[str, list[Any]] | None = None,
    extra: Any = None,
    ns: str = "",
    conditions: tuple[str, ...] = ("K1", "K2", "K3"),
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    reuse: bool = True,
) -> Proved:
    """K1–K3 of one Template, written as ProofResults and one ProofSummary."""
    t = Template.model_validate(store.get(template_id).body)
    found = outline_for(store, template_ref(t))
    outline_id, outline = found if found else (None, None)
    fixed = {k: (v.name if isinstance(v, Sym) else v) for k, v in (restrict or {}).items()}
    fixed.update({f"attr:{k}": str(v.term) for k, v in (attrs or {}).items()})
    fixed.update({f"menu:{k}": str(v) for k, v in (menus or {}).items()})
    if extra is not None:
        fixed["extra"] = str(extra)
    g = rc.groups(t, outline, fixed)
    v = rc.versions()
    binding = {b.obligation: b for b in (outline.models if outline else [])}
    reused: dict[tuple[str, str], Record] = {}
    expected = _expected_groups(t, outline)
    if reuse:
        for key, grp in expected.items():
            if key[0] not in conditions:
                continue
            probe = Draft(key[0], key[1], "proved", "", groups=grp)
            prem = rc.premises(probe, g, t, binding.get(key[1]), v)
            hit = rc.reusable(store, rc.h(prem))
            if hit is not None:
                reused[key] = hit
    chk = check_template(
        t, outline, restrict=restrict, attrs=attrs, menus=menus, extra=extra, ns=ns,
        timeout_ms=timeout_ms,
        conditions=conditions, skip=set(reused),
    )  # fmt: skip
    subject = subject or template_id
    recs: list[Record] = []
    drafts = list(chk.drafts)
    for d in chk.drafts:
        prem = rc.premises(d, g, t, binding.get(d.obligation) if d.condition == "K2" else None, v)
        recs.append(rc.result_record(d, subject, scope, prem, actor, now, outline_id))
    for key, old in reused.items():
        d = _from_record(old, expected[key])
        drafts.append(d)
        prem = rc.premises(d, g, t, binding.get(key[1]) if key[0] == "K2" else None, v)
        recs.append(
            rc.result_record(d, subject, scope, prem, actor, now, outline_id, reused=record_id(old))
        )
    pre_gaps = [x for x in chk.gaps if x.startswith("assumption ")]
    level, gaps = fold(drafts, chk.unchecked, pre_gaps)
    if chk.level is None and not chk.drafts and chk.gaps:
        level, gaps = None, chk.gaps
    ids = store.put_many(recs)
    summary = rc.summary_record(
        subject,
        scope,
        level,
        list(zip(ids, recs, strict=True)),
        chk.unchecked,
        gaps,
        actor,
        now,
        part=part,
    )
    sid = store.put(summary)
    return Proved(
        summary=sid,
        level=level,
        results=ids,
        reused=len(reused),
        queries=sum(len(d.queries) for d in chk.drafts),
        gaps=gaps,
        check=chk,
    )
