"""Deterministic retrieval of Templates, Families and Chains, rendered as cards (22 §5).

Ranking (knowledge.md "检索排序"): hard filters first (Pattern, Γ true or unknown,
every required Capability ``yes`` on the case's Target, D445), then assurance, then reuse
count; Γ-unknown entries go last and are marked on the card. Solutions do not exist
before WP7, so the %-of-bound term is not used yet (allowed by WP4, decision D210).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from kta.core.errors import SchemaError
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Case
from kta.core.schemas.knowledge import Chain, Template
from kta.knowledge import pool
from kta.knowledge.capabilities import template_availability
from kta.knowledge.findings import findings
from kta.knowledge.scope import item_params, problem_params, scope_holds
from kta.knowledge.types import Card
from kta.store.base import Store

ASSURANCE_ORDER = {"proved": 0, "bounded": 1, "tested": 2, "declared": 3}
CARD_MAX_LINES = 5
CARD_KINDS = frozenset(
    {
        "Template",
        "Chain",
        "Pattern",
        "Structure",
        "Gap",
        "Finding",
        "Coupling",
        "Negative",
        "Regime",
        "Family",
        "Reference",
        "Reconstruction",
        "Realisation",
        "Capability",
        "CapabilityDecl",
        "HwFeature",
        "HwFact",
    }
)
_WORD = re.compile(r"[a-z0-9]+")


def _first_sentence(text: str, limit: int = 160) -> str:
    s = " ".join(text.split())
    end = s.find(". ")
    s = s if end < 0 else s[: end + 1]
    return s if len(s) <= limit else s[: limit - 1] + "…"


def reuse_counts(store: Store) -> dict[str, int]:
    """Template ref -> number of Chains that use it."""
    counts: dict[str, int] = {}
    for rec in pool.records(store, "Chain"):
        for ref in set(pool.chain_template_refs(rec.body)):
            counts[ref] = counts.get(ref, 0) + 1
    return counts


def _negatives(store: Store, name: str) -> int:
    return len(findings(store, subject=name, kind="negative"))


def _template_card(store: Store, rid: str, t: Template, scope_unknown: bool = False) -> Card:
    status, assurance = pool.status_of(store, rid)
    ref = pool.template_ref(t.name, t.version)
    uses = reuse_counts(store).get(ref, 0)
    lines = [
        f"{ref} [{status}, {assurance}] — {t.refines} -> {', '.join(t.produces) or '-'}, "
        f"level {t.level_from}->{t.level_to}, object {t.object.type}",
        f"author: {_first_sentence(t.summary)}",
        f"vars: {', '.join(v.name for v in t.variables)}; "
        f"obligations: {', '.join(o.name for o in t.obligations)}",
        f"evidence: used in {uses} chain(s); negatives {_negatives(store, t.name)}; "
        f"requires: {', '.join(t.requires) or 'none'}; probed on: "
        f"{', '.join(t.probed_on) or 'none'}",
    ]
    if scope_unknown:
        lines.append(f"scope NOT EVALUATED for this case: {t.scope.predicate}")
    return Card(rid, "Template", ref, lines[:CARD_MAX_LINES], status, assurance, scope_unknown)


def _hw_ok(store: Store, t: Template, target: str | None) -> bool:
    return template_availability(store, t.requires, target).available


def unavailable_templates(store: Store, target: str) -> dict[str, list[str]]:
    """Usable Templates retrieval hides on ``target``: ``unavailable`` (a required
    Capability is ``no``: known, unavailable here) and ``unconfirmed`` (``unknown``)."""
    out: dict[str, list[str]] = {"unavailable": [], "unconfirmed": []}
    for rid, t in pool.templates(store):
        if not pool.usable(store, rid):
            continue
        a = template_availability(store, t.requires, target)
        ref = pool.template_ref(t.name, t.version)
        if a.missing:
            out["unavailable"].append(ref)
        elif a.unconfirmed:
            out["unconfirmed"].append(ref)
    return {k: sorted(v) for k, v in out.items()}


def retrieve_templates(
    store: Store,
    pattern: str | None = None,
    unit: str | None = None,
    text: str | None = None,
    case: Case | None = None,
    k: int = 10,
    hw: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    target: str | None = None,
) -> list[Card]:
    """Templates matching every given filter, best first, as cards.

    ``target`` (default: the case's) hides Templates whose required Capabilities are not
    all ``yes`` there (D445); ``unavailable_templates`` lists what was hidden.
    """
    target = target or (case.target if case is not None else None)
    text_ids = (
        {h.id for h in store.search(text, kinds=["Template"], limit=pool.ALL)} if text else None
    )
    p = {**(problem_params(store, case) if case else {}), **(params or {})}
    counts = reuse_counts(store)
    reserved = reserved_resources(store) if unit is not None else {}
    ranked: list[tuple[tuple[Any, ...], Card]] = []
    for rid, t in pool.templates(store):
        if not pool.usable(store, rid):
            continue
        if pattern is not None and t.refines != pattern:
            continue
        if unit is not None and unit not in _resources(t, reserved):
            continue
        if text_ids is not None and rid not in text_ids:
            continue
        unknown = False
        if target is not None and not _hw_ok(store, t, target):
            continue
        if case is not None:
            holds = scope_holds(t, case, hw or {}, p).holds
            if holds is False:
                continue
            unknown = holds is None
        _, assurance = pool.status_of(store, rid)
        ref = pool.template_ref(t.name, t.version)
        key = (unknown, ASSURANCE_ORDER[assurance], -counts.get(ref, 0), ref)
        ranked.append((key, _template_card(store, rid, t, unknown)))
    return [c for _, c in sorted(ranked, key=lambda x: x[0])[:k]]


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _case_words(store: Store, case: Case) -> set[str]:
    words = _words(case.problem)
    hit = pool.find_by_name(store, "Problem", case.problem)
    if hit is not None:
        body = hit[1].body
        words |= _words(
            " ".join(
                [
                    str(body.get("reference_impl", "")),
                    *(f"{t['dtype']} {t['name']}" for t in body.get("inputs", [])),
                ]
            )
        )
    return words


def _chain_similarity(case_words: set[str], chain: Chain) -> float:
    words = _words(chain.workload)
    return len(case_words & words) / len(case_words | words) if words else 0.0


def nearest_chains(store: Store, case: Case, k: int = 3) -> list[Card]:
    """Chains whose workload text is most similar to the case's Problem (token Jaccard)."""
    cw = _case_words(store, case)
    scored: list[tuple[float, str, Chain]] = []
    for rid, rec in pool.with_ids(pool.records(store, "Chain")):
        if pool.usable(store, rid):
            chain = Chain.model_validate(rec.body)
            scored.append((_chain_similarity(cw, chain), rid, chain))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [_chain_card(store, rid, ch, sim) for sim, rid, ch in scored[:k]]


def _chain_card(store: Store, rid: str, chain: Chain, similarity: float | None = None) -> Card:
    status, assurance = pool.status_of(store, rid)
    owners = [store.get(o).kind for o in store.neighbors(rid, direction="in")]
    sim = "" if similarity is None else f"; similarity {similarity:.2f}"
    lines = [
        f"chain {rid[:12]} [{status}] — {_first_sentence(chain.workload, 120)}",
        f"steps: {', '.join(i.template for i in chain.items)}"[:300],
        f"used by: {', '.join(sorted(set(owners))) or 'nothing'}{sim}",
    ]
    return Card(rid, "Chain", rid[:12], lines, status, assurance)


def _family_chain(store: Store, rec: Record) -> Chain | None:
    chain_id = rec.refs.get("chain")
    if not isinstance(chain_id, str):
        return None
    return Chain.model_validate(store.get(chain_id).body)


def retrieve_families(
    store: Store, case: Case, k: int = 5, hw: Mapping[str, Any] | None = None
) -> list[Card]:
    """Families whose every step is admissible on the case (Γ not false, hardware yes)."""
    p = problem_params(store, case)
    cw = _case_words(store, case)
    ranked: list[tuple[tuple[Any, ...], Card]] = []
    for rid, rec in pool.with_ids(pool.records(store, "Family")):
        chain = _family_chain(store, rec)
        if chain is None or not pool.usable(store, rid):
            continue
        unknown, ok = False, True
        for item in chain.items:
            hit = pool.find_template(store, item.template)
            if hit is None or not _hw_ok(store, hit[1], case.target):
                ok = False
                break
            holds = scope_holds(hit[1], case, hw or {}, item_params(p, item.bindings)).holds
            ok &= holds is not False
            unknown |= holds is None
        if not ok:
            continue
        status, assurance = pool.status_of(store, rid)
        name = f"{rec.body['name']}@{rec.body['version']}"
        lines = [
            f"family {name} [{status}, {assurance}] backend {rec.body['backend']}",
            f"steps: {', '.join(i.key for i in chain.items)}"[:300],
            f"variables: {', '.join(v['name'] for v in rec.body.get('variables', []))}"[:300],
        ]
        if unknown:
            lines.append("scope NOT EVALUATED for at least one step on this case")
        key = (unknown, ASSURANCE_ORDER[assurance], -_chain_similarity(cw, chain), name)
        ranked.append((key, Card(rid, "Family", name, lines, status, assurance, unknown)))
    return [c for _, c in sorted(ranked, key=lambda x: x[0])[:k]]


def diff_templates(store: Store, reference_chain: str, current_chain: str) -> list[Card]:
    """Templates in the reference chain that the current chain does not use (by name)."""
    ref = Chain.model_validate(store.get(reference_chain).body)
    cur = Chain.model_validate(store.get(current_chain).body)
    have = {pool.parse_template_ref(i.template)[0] for i in cur.items}
    cards: list[Card] = []
    for tref in dict.fromkeys(i.template for i in ref.items):
        if pool.parse_template_ref(tref)[0] in have:
            continue
        hit = pool.find_template(store, tref)
        if hit is not None:
            cards.append(_template_card(store, hit[0], hit[1]))
    return cards


def card(store: Store, record_id: str) -> Card:
    """A card for any knowledge record; at most five lines."""
    rec = store.get(record_id)
    if rec.kind not in CARD_KINDS:
        raise SchemaError(f"no card for kind {rec.kind}")
    if rec.kind == "Template":
        return _template_card(store, record_id, Template.model_validate(rec.body))
    if rec.kind == "Chain":
        return _chain_card(store, record_id, Chain.model_validate(rec.body))
    status, assurance = pool.status_of(store, record_id)
    body = rec.body
    title = str(
        body.get("name")
        or body.get("claim")
        or body.get("description")
        or body.get("subject")
        or record_id
    )
    lines = [f"{rec.kind} {_first_sentence(title, 120)} [{status}, {assurance}]"]
    for key in ("description", "condition", "effect", "summary", "evidence_needed", "kind"):
        if key in body and body[key] and len(lines) < CARD_MAX_LINES:
            lines.append(f"{key}: {_first_sentence(str(body[key]))}")
    return Card(record_id, rec.kind, title[:80], lines, status, assurance)


def reserved_resources(store: Store) -> dict[str, set[str]]:
    """Operation class -> resources its reservation table uses, over every HwArch in the
    store (D460: a Template names transaction kinds; the hardware model names resources)."""
    out: dict[str, set[str]] = {}
    for rec in store.query("HwArch", limit=200):
        for op in rec.body.get("operation_classes", []):
            out.setdefault(str(op["name"]), set()).update(
                str(row["resource"]) for row in op.get("reservation", [])
            )
    return out


def _resources(t: Template, reserved: Mapping[str, set[str]]) -> set[str]:
    units = {c.unit for c in t.cost} | set(t.signature.resources)
    for tx in t.effects.transactions:
        for kind in [tx.kind, *tx.kind_map.values()]:
            units |= reserved.get(kind, set())
    return units
