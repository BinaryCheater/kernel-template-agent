"""Learning-loop coverage (docs/design/02 §8.1, §8.2, §8.5; WP20, D567).

- SOTA coverage in four levels: expressible (match rate), reproducible (reproduction
  error within ε), reachable and surpassed (O12 Reports, not run yet);
- system-level features counted apart from scheduling moves;
- language gaps counted by the admit step;
- per-backend expressibility coverage (steps of reconstructed chains whose Template has
  code on the backend, from a Realisation with code markers or fragments, or a Family
  chain item with code markers) and intent fidelity (steps with SASS expectations on the
  backend; confirmation needs compiled SASS, so it is pending until real runs).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from kta.core.schemas import Record
from kta.knowledge import pool
from kta.knowledge.admission import latest_admissions
from kta.knowledge.learning import sota_level_rows, system_feature, tables
from kta.knowledge.realisations import ANY_BACKEND, realisations_of
from kta.store.base import Store

O12_TITLE = "o12"  # Report of one blind reconstruction: data {reference, level, ...}
PENDING = {
    "reproducible": "needs reference builds measured on GB10 and reproduced by a family (L1, L2)",
    "reachable": "needs O12 blind reconstruction runs (L4, defined, not run)",
    "surpassed": "needs O12 blind reconstruction runs (L4, defined, not run)",
    "intent_fidelity": "needs compiled SASS of families on the backend",
}


def _chains(store: Store, recons: list[tuple[str, Record]]) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    for _, rec in recons:
        cid = rec.refs.get("chain")
        out.append(list(store.get(cid).body.get("items", [])) if isinstance(cid, str) else [])
    return out


def sota_levels(
    store: Store, recons: list[tuple[str, Record]], refs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per level: references at that level or above, of all decomposed ones."""
    eps = float(tables()["performance_core"]["epsilon_rel"])
    n = len(refs)
    steps = sum(int(r["total"]) for r in refs)
    matched = sum(int(r["matched"]) for r in refs)
    reproduced = [
        str(r.refs.get("reference"))
        for _, r in recons
        if r.body.get("reproduction_error") is not None
        and float(r.body["reproduction_error"]) <= eps
    ]
    o12: dict[str, int] = {}
    for rec in pool.records(store, "Report", {"title": O12_TITLE}, order="created_at"):
        o12[str(rec.body["data"].get("reference"))] = int(rec.body["data"].get("level", 0))
    hits = {
        1: [str(r["reference_id"]) for r in refs if r["matched"] == r["total"]],
        2: reproduced,
        3: [k for k, v in o12.items() if v >= 3],
        4: [k for k, v in o12.items() if v >= 4],
    }
    out: list[dict[str, Any]] = []
    for row in sota_level_rows():
        level = int(row["level"])
        got = hits.get(level, [])
        item = {**row, "references": len(got), "of": n, "pending": None}
        if level == 1:
            item["step_rate"] = matched / steps if steps else None
        elif not got:
            item["pending"] = PENDING.get(str(row["name"]))
        out.append(item)
    return out


def system_features(recons: list[tuple[str, Record]]) -> dict[str, Any]:
    """Unmatched transactions that are system-level features, per feature (02 §8.2)."""
    counts: Counter[str] = Counter()
    for _, rec in recons:
        for tx in rec.body.get("unmatched_transactions", []):
            f = system_feature(tx)
            if f is not None:
                counts[f] += 1
    keys = [*tables()["system_features"], "other"]
    by = {k: counts.get(k, 0) for k in keys}
    return {"total": sum(by.values()), "by_feature": by}


def language_gaps(store: Store, refs: list[dict[str, Any]]) -> dict[str, Any]:
    """Expressibility gaps of the language counted at admission, per kind and reference."""
    data = latest_admissions(store, [str(r["reference_id"]) for r in refs])
    by_kind: Counter[str] = Counter()
    per: dict[str, int] = {}
    for r in refs:
        d = data.get(str(r["reference_id"]))
        if d is None:
            continue
        per[str(r["reference"])] = int(d["counts"]["language_gaps"])
        by_kind.update(d["counts"]["language_gaps_by_kind"])
    return {
        "total": sum(per.values()),
        "by_kind": dict(sorted(by_kind.items())),
        "by_reference": per,
        "admitted_references": len(per),
    }


def _family_marked(store: Store, members: set[str] | None) -> dict[str, set[str]]:
    """Backend -> Template refs some Family chain item realises with code markers."""
    out: dict[str, set[str]] = {}
    for rid, rec in pool.with_ids(pool.records(store, "Family")):
        cid = rec.refs.get("chain")
        if (members is not None and rid not in members) or not isinstance(cid, str):
            continue
        for item in store.get(cid).body.get("items", []):
            if item.get("code_markers"):
                out.setdefault(str(rec.body["backend"]), set()).add(str(item["template"]))
    return out


def backends(
    store: Store, recons: list[tuple[str, Record]], members: set[str] | None
) -> dict[str, dict[str, Any]]:
    """Per backend: expressibility coverage and declared intent (02 §8.5, D-T29)."""
    steps = [str(i["template"]) for items in _chains(store, recons) for i in items]
    refs = sorted(set(steps))
    real = {t: realisations_of(store, t) for t in refs}
    names = set(tables()["backends"]) | {
        r.body.backend for rs in real.values() for r in rs if r.body.backend != ANY_BACKEND
    }
    marked = _family_marked(store, members)
    out: dict[str, dict[str, Any]] = {}
    for b in sorted(names):
        code = {
            t
            for t, rs in real.items()
            if any(r.body.backend == b and (r.body.code_markers or r.body.fragments) for r in rs)
        } | marked.get(b, set())
        intent = {
            t
            for t, rs in real.items()
            if any(r.body.backend in (b, ANY_BACKEND) and r.body.expected_sass for r in rs)
        }
        n = len(steps)
        expressible = sum(1 for t in steps if t in code)
        out[b] = {
            "steps": n,
            "expressible": expressible,
            "rate": expressible / n if n else None,
            "intent_declared": sum(1 for t in steps if t in intent),
            "intent_confirmed": None,
            "note": str(tables()["backends"].get(b, "")),
        }
    return out


def learning_coverage(
    store: Store,
    recons: list[tuple[str, Record]],
    refs: list[dict[str, Any]],
    members: set[str] | None,
) -> dict[str, Any]:
    return {
        "sota_levels": sota_levels(store, recons, refs),
        "system_features": system_features(recons),
        "language_gaps": language_gaps(store, refs),
        "backends": backends(store, recons, members),
    }


def page_fields(learning: Mapping[str, Any]) -> dict[str, Any]:
    """The coverage page's learning section (spec/modules/api-web.md "Coverage")."""
    return {k: learning[k] for k in ("sota_levels", "system_features", "language_gaps", "backends")}
