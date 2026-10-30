"""Realisations of a Template: how a move is written per backend and ISA range (D443).

Realisation records reference their Template (``refs.template``). Templates stored
before D443 (schema 1) carried code fragments and SASS expectations themselves; those
stay in the stored body and are read back here as one *legacy* realisation per Template
(backend ``any``, no record of its own), so nothing written earlier is lost or rewritten.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from kta.core.schemas.knowledge import IsaRange, Realisation, SassExpect
from kta.knowledge import pool
from kta.knowledge.templates import check_expr
from kta.knowledge.types import Issue
from kta.store.base import Store

ANY_BACKEND = "any"


@dataclass(frozen=True)
class RealisationRef:
    id: str | None  # None for a legacy realisation read from a v1 Template body
    template_id: str
    body: Realisation
    legacy: bool = False


def _legacy(template_id: str, ref: str, stored: dict[str, object]) -> RealisationRef | None:
    program = stored.get("program")
    fragments = program.get("fragments", {}) if isinstance(program, dict) else {}
    sass = stored.get("expected_sass") or []
    if not fragments and not sass:
        return None
    body = Realisation(
        template=ref,
        backend=ANY_BACKEND,
        isa=IsaRange(family="sm"),
        fragments={str(k): str(v) for k, v in dict(fragments).items()},  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        expected_sass=[SassExpect.model_validate(s) for s in sass],  # pyright: ignore[reportGeneralTypeIssues]
        summary="legacy: code and SASS expectations stored in the Template body (schema 1)",
    )
    return RealisationRef(None, template_id, body, legacy=True)


def realisations_of(
    store: Store, template_ref: str, backend: str | None = None
) -> list[RealisationRef]:
    """Usable realisations of ``name@version``; ``backend`` keeps that backend and ``any``."""
    hit = pool.find_template(store, template_ref)
    if hit is None:
        return []
    tid = hit[0]
    out: list[RealisationRef] = []
    stored = store.get(tid)
    if stored.upcast:
        legacy = _legacy(tid, template_ref, stored.stored_body)
        if legacy is not None:
            out.append(legacy)
    recs = pool.records(store, "Realisation", {"template": template_ref}, order="created_at")
    for rid, rec in pool.with_ids(recs):
        if pool.usable(store, rid):
            out.append(RealisationRef(rid, tid, Realisation.model_validate(rec.body)))
    if backend is not None:
        out = [r for r in out if r.body.backend in (backend, ANY_BACKEND)]
    return out


def expected_sass(
    store: Store, template_refs: Iterable[str], backend: str
) -> dict[str, list[SassExpect]]:
    """Template ref -> SASS expectations of its realisations on ``backend`` (and ``any``)."""
    return {
        ref: [e for r in realisations_of(store, ref, backend) for e in r.body.expected_sass]
        for ref in dict.fromkeys(template_refs)
    }


def probed_on(store: Store, template_ref: str) -> list[str]:
    """Targets on which the Template or any of its realisations was probed."""
    hit = pool.find_template(store, template_ref)
    if hit is None:
        return []
    names = list(hit[1].probed_on)
    for r in realisations_of(store, template_ref):
        names += r.body.probed_on
    return sorted(set(names))


def validate_realisation(r: Realisation, store: Store) -> list[Issue]:
    """Template exists; ISA constraints and SASS ``when`` parse over its variables;
    ``requires`` names known ISA features (HwFeature, warning otherwise)."""
    hit = pool.find_template(store, r.template)
    if hit is None:
        return [
            Issue("unknown_template", f"Template {r.template!r} is not in the store", "template")
        ]
    declared = {v.name for v in hit[1].variables}
    issues: list[Issue] = []
    if not r.summary.strip():
        issues.append(Issue("missing_field", "summary is empty", "summary"))
    if not r.backend.strip():
        issues.append(Issue("missing_field", "backend is empty", "backend"))
    for c in r.isa_constraints:
        issues += check_expr(c.expr, declared, f"isa_constraints.{c.id}", None)[1]
    for i, e in enumerate(r.expected_sass):
        issues += check_expr(e.when, declared, f"expected_sass[{i}].when", None)[1]
    features = pool.names(store, "HwFeature")
    issues += [
        Issue("unknown_isa_feature", f"HwFeature {f!r} is not in the store", "requires", "warning")
        for f in r.requires
        if f not in features
    ]
    return issues
