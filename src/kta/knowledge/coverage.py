"""Coverage statistics over the knowledge pool (docs/design/21 §2).

Species estimates treat each Reconstruction as one sample and each Template name as a
species; incidence is "the Template appears in that reconstruction's chain". Only the
latest Reconstruction of each Reference counts: the expand loop's re-match writes a new
one per round (D386).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, get_args

from kta.core.errors import SchemaError
from kta.core.schemas import Record
from kta.core.schemas.common import Assurance, Level
from kta.core.schemas.knowledge import Gap
from kta.knowledge import pool
from kta.knowledge.types import CoverageReport
from kta.store.base import Store

PERMUTATIONS = 50
ASSURANCES: tuple[str, ...] = get_args(Assurance)
LEVELS: tuple[str, ...] = tuple(str(v) for v in get_args(Level))
GAP_MISSING: tuple[str, ...] = get_args(Gap.model_fields["missing"].annotation)
GAP_ATTRIBUTION: tuple[str, ...] = get_args(Gap.model_fields["attribution"].annotation)
# 21 §2 quantities this report cannot compute from records without real runs.
PENDING = {
    "instruction_coverage": "needs the SASS of reference builds and of our generated kernels",
    "gap_explained": "needs solve-loop judgements on measured cases",
    "input_space_coverage": "needs Solutions over the shape and dtype space",
}


def _members(store: Store, snapshot_id: str | None) -> set[str] | None:
    if snapshot_id is None:
        return None
    rec = store.get(snapshot_id)
    if rec.kind != "PoolSnapshot":
        raise SchemaError(f"{snapshot_id} is a {rec.kind}, not a PoolSnapshot")
    return set(rec.body.get("members", []))


def _select(store: Store, kind: str, members: set[str] | None) -> list[tuple[str, Record]]:
    """Records of ``kind`` in the snapshot, by creation time then insertion order."""
    return [
        (rid, r)
        for rid, r in pool.with_ids(pool.records(store, kind, order="created_at"))
        if members is None or rid in members
    ]


def _latest(recons: list[tuple[str, Record]]) -> list[tuple[str, Record]]:
    """The newest Reconstruction per Reference, kept in the order of first decomposition."""
    order: list[str] = []
    latest: dict[str, tuple[str, Record]] = {}
    for rid, rec in recons:
        key = str(rec.refs.get("reference") or rid)
        if key not in latest:
            order.append(key)
        latest[key] = (rid, rec)
    return [latest[k] for k in order]


def _zeros(keys: Sequence[str], counts: Mapping[str, int]) -> dict[str, int]:
    return {k: int(counts.get(k, 0)) for k in [*keys, *sorted(set(counts) - set(keys))]}


def _template_name(ref: str) -> str:
    return ref.partition("@")[0]


def _samples(store: Store, recons: list[tuple[str, Record]]) -> list[set[str]]:
    out: list[set[str]] = []
    for _, rec in recons:
        chain_id = rec.refs.get("chain")
        chain = store.get(chain_id).body if isinstance(chain_id, str) else {"items": []}
        out.append({_template_name(r) for r in pool.chain_template_refs(chain)})
    return out


def _curve(samples: list[set[str]]) -> list[int]:
    seen: set[str] = set()
    curve: list[int] = []
    for s in samples:
        seen |= s
        curve.append(len(seen))
    return curve


def discovery(
    samples: list[set[str]], rng: random.Random, n: int = PERMUTATIONS
) -> tuple[list[float], list[int], list[int]]:
    """Mean, min and max of the cumulative distinct-Template curve over ``n`` shuffles."""
    if not samples:
        return [], [], []
    curves = []
    for _ in range(n):
        order = list(samples)
        rng.shuffle(order)
        curves.append(_curve(order))
    cols = list(zip(*curves, strict=True))
    return [sum(c) / n for c in cols], [min(c) for c in cols], [max(c) for c in cols]


def chao1(observed: int, f1: int, f2: int) -> float:
    """S_obs + f1²/(2·f2); the bias-corrected f1(f1−1)/2 when f2 = 0 (Chao 1987)."""
    if f2 > 0:
        return observed + f1 * f1 / (2 * f2)
    return observed + f1 * (f1 - 1) / 2


def chao1_interval(observed: int, f1: int, f2: int) -> tuple[float, float]:
    """95% log-normal interval of Chao1 (Chao 1987); the lower end is at least ``observed``."""
    est = chao1(observed, f1, f2)
    unseen = est - observed
    if f2 > 0:
        r = f1 / f2
        var = f2 * (r**4 / 4 + r**3 + r**2 / 2)
    else:
        var = f1 * (f1 - 1) / 2 + f1 * (2 * f1 - 1) ** 2 / 4 - (f1**4 / (4 * est) if est else 0)
    if unseen <= 0 or var <= 0:
        return float(observed), est
    k = math.exp(1.96 * math.sqrt(math.log(1 + var / unseen**2)))
    return observed + unseen / k, observed + unseen * k


def _references(store: Store, recons: list[tuple[str, Record]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rid, rec in recons:
        ref_id = rec.refs.get("reference")
        ref = store.get(ref_id).body if isinstance(ref_id, str) else {}
        total = int(rec.body["total"])
        out.append(
            {
                "reference": ref.get("name"),
                "reference_id": ref_id,
                "reconstruction": rid,
                "problem": ref.get("problem"),
                "analysis_only": not ref.get("runnable_on"),
                "matched": rec.body["matched"],
                "total": total,
                "match_rate": rec.body["matched"] / total if total else None,
                "unmatched": len(rec.body.get("unmatched_transactions", [])),
                "reproduced": rec.body.get("reproduction_error") is not None,
            }
        )
    return sorted(out, key=lambda r: str(r["reference"]))


def _hw_grid(
    store: Store, templates: list[tuple[str, Record]]
) -> dict[str, dict[str, dict[str, int]]]:
    """HwFeature -> refined Pattern -> assurance -> Template count ("none" for no feature)."""
    grid: dict[str, dict[str, dict[str, int]]] = {}
    for rid, rec in templates:
        _, assurance = pool.status_of(store, rid)
        for feature in rec.body.get("requires_hw") or ["none"]:
            cell = grid.setdefault(feature, {}).setdefault(rec.body["refines"], {})
            cell[assurance] = cell.get(assurance, 0) + 1
    return grid


def _families_per_template(
    store: Store, members: set[str] | None, templates: list[tuple[str, Record]]
) -> dict[str, int]:
    """Every Template name in the pool (0 when no family uses it)."""
    counts: Counter[str] = Counter({str(r.body["name"]): 0 for _, r in templates})
    for _, rec in _select(store, "Family", members):
        chain_id = rec.refs.get("chain")
        if isinstance(chain_id, str):
            counts.update(
                {_template_name(r) for r in pool.chain_template_refs(store.get(chain_id).body)}
            )
    return dict(sorted(counts.items()))


def _gaps(store: Store, recons: list[tuple[str, Record]]) -> list[Gap]:
    out: list[Gap] = []
    for _, rec in recons:
        ids = rec.refs.get("gaps") or []
        out += [
            Gap.model_validate(store.get(g).body) for g in ([ids] if isinstance(ids, str) else ids)
        ]
    return out


def _missing_templates(store: Store, recons: list[tuple[str, Record]]) -> list[str]:
    refs: set[str] = set()
    for _, rec in recons:
        chain_id = rec.refs.get("chain")
        if isinstance(chain_id, str):
            refs.update(pool.chain_template_refs(store.get(chain_id).body))
    return sorted(r for r in refs if pool.find_template(store, r) is None)


def coverage_report(store: Store, snapshot_id: str | None, rng: random.Random) -> CoverageReport:
    """All coverage quantities of 21 §2 for a PoolSnapshot (``None``: the whole store).

    ``rng`` drives the discovery-curve shuffles; pass ``ctx.rng("coverage")`` and record
    the seed with the report.
    """
    members = _members(store, snapshot_id)
    recons = _latest(_select(store, "Reconstruction", members))
    samples = _samples(store, recons)
    incidence = Counter(name for s in samples for name in s)
    f1 = sum(1 for c in incidence.values() if c == 1)
    f2 = sum(1 for c in incidence.values() if c == 2)
    n = sum(incidence.values())
    mean, low, high = discovery(samples, rng)
    by_level: Counter[str] = Counter()
    by_pattern: Counter[str] = Counter()
    for _, rec in recons:
        for tx in rec.body.get("unmatched_transactions", []):
            by_level[str(tx["level"])] += 1
            by_pattern[str(tx["pattern_guess"])] += 1
    templates = _select(store, "Template", members)
    assurance = Counter(pool.status_of(store, rid)[1] for rid, _ in templates)
    patterns = _select(store, "Pattern", members)
    gaps = _gaps(store, recons)
    refs = _references(store, recons)
    return CoverageReport(
        snapshot_id=snapshot_id,
        reconstructions=len(recons),
        discovery_observed=_curve(samples),
        discovery_mean=mean,
        discovery_low=low,
        discovery_high=high,
        discovery_permutations=PERMUTATIONS,
        templates_observed=len(incidence),
        singletons=f1,
        doubletons=f2,
        chao1=chao1(len(incidence), f1, f2),
        good_turing=1 - f1 / n if n else None,
        occurrences=n,
        references=refs,
        unmatched_by_level=_zeros(LEVELS, by_level),
        unmatched_by_pattern=dict(sorted(by_pattern.items())),
        # The fallback Pattern "other" itself is not a parked transaction type.
        other_patterns=sum(
            1 for _, r in patterns if r.body["category"] == "other" and r.body["name"] != "other"
        ),
        hw_pattern_grid=_hw_grid(store, templates),
        assurance=_zeros(ASSURANCES, assurance),
        families_per_template=_families_per_template(store, members, templates),
        gaps_by_missing=_zeros(GAP_MISSING, Counter(g.missing for g in gaps)),
        gaps_by_attribution=_zeros(GAP_ATTRIBUTION, Counter(g.attribution for g in gaps)),
        templates_missing=_missing_templates(store, recons),
        direct_test={
            "references": len(refs),
            "all_steps_matched": sum(1 for r in refs if r["matched"] == r["total"]),
            "nothing_unexplained": sum(
                1 for r in refs if r["matched"] == r["total"] and not r["unmatched"]
            ),
        },
        pending=dict(PENDING),
    )


def coverage_page(
    rep: CoverageReport,
    loop_states: Mapping[str, str],
    trend: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The coverage page's fields (spec/modules/api-web.md "Coverage", web CoverageReport).

    ``loop_states`` maps a Reference ID to its expand-loop state; ``trend`` holds the
    ``{snapshot, good_turing, expressible}`` points of earlier coverage Reports, oldest first.
    Quantities that need real runs are empty lists and are named in ``pending``.
    """
    lo, hi = chao1_interval(rep.templates_observed, rep.singletons, rep.doubletons)
    n = rep.direct_test["references"]
    point = {
        "snapshot": rep.snapshot_id or "whole store",
        "good_turing": rep.good_turing if rep.good_turing is not None else 0.0,
        "expressible": rep.direct_test["all_steps_matched"] / n if n else 0.0,
    }
    problems = Counter(str(r["problem"]) for r in rep.references)
    features = sorted(rep.hw_pattern_grid)
    pats = sorted({p for row in rep.hw_pattern_grid.values() for p in row})
    cells = [
        {
            "f": fi,
            "p": pi,
            "n": sum(row.get(p, {}).values()),
            "best": max(row[p], key=ASSURANCES.index) if row.get(p) else "",
        }
        for fi, row in ((i, rep.hw_pattern_grid[f]) for i, f in enumerate(features))
        for pi, p in enumerate(pats)
    ]
    backlog = [(f"extract {t}", "named by a chain, not in the pool") for t in rep.templates_missing]
    backlog += [
        (
            f"{c} Gap(s) missing {k}",
            "recorded when the reference was decomposed; a Gap stays after it is filled",
        )
        for k, c in sorted(rep.gaps_by_missing.items(), key=lambda kv: -kv[1])
        if c
    ]
    return {
        "scope": {
            "description": "decomposed reference implementations (public GPU kernels)",
            "total": len(rep.references),
            "by_category": dict(sorted(problems.items())),
        },
        "discovery": {
            "sources": rep.discovery_observed,
            "mean": rep.discovery_mean,
            "lo": rep.discovery_low,
            "hi": rep.discovery_high,
            "chao1": {"estimate": rep.chao1, "lo": lo, "hi": hi},
        },
        "trend": [*trend, point],
        "instruction_families": [],
        "hw_patterns": {"features": features, "patterns": pats, "cells": cells},
        "references": [
            {
                "name": r["reference"],
                "state": loop_states.get(str(r["reference_id"]), "not started"),
                "record": r["reconstruction"],
                "gaps": r["unmatched"],
            }
            for r in rep.references
        ],
        "gap_explained": [],
        "backlog": [
            {"item": item, "why": why, "priority": i + 1} for i, (item, why) in enumerate(backlog)
        ],
        "structure": {
            "other_patterns": rep.other_patterns,
            "unmatched_by_kind": rep.unmatched_by_pattern,
            "assurance": rep.assurance,
        },
        "pending": rep.pending,
    }
