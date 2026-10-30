"""Coverage statistics over the knowledge pool (docs/design/21 §2).

Species estimates treat each Reconstruction as one sample and each Template name as a
species; incidence is "the Template appears in that reconstruction's chain".
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from kta.core.errors import SchemaError
from kta.core.schemas import Record
from kta.knowledge import pool
from kta.knowledge.types import CoverageReport
from kta.store.base import Store

PERMUTATIONS = 50


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


def _references(store: Store, recons: list[tuple[str, Record]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, rec in recons:
        ref_id = rec.refs.get("reference")
        name = store.get(ref_id).body.get("name") if isinstance(ref_id, str) else None
        total = int(rec.body["total"])
        out.append(
            {
                "reference": name,
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


def _families_per_template(store: Store, members: set[str] | None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _, rec in _select(store, "Family", members):
        chain_id = rec.refs.get("chain")
        if isinstance(chain_id, str):
            counts.update(
                {_template_name(r) for r in pool.chain_template_refs(store.get(chain_id).body)}
            )
    return dict(counts)


def coverage_report(store: Store, snapshot_id: str | None, rng: random.Random) -> CoverageReport:
    """All coverage quantities of 21 §2 for a PoolSnapshot (``None``: the whole store).

    ``rng`` drives the discovery-curve shuffles; pass ``ctx.rng("coverage")`` and record
    the seed with the report.
    """
    members = _members(store, snapshot_id)
    recons = _select(store, "Reconstruction", members)
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
        references=_references(store, recons),
        unmatched_by_level=dict(sorted(by_level.items())),
        unmatched_by_pattern=dict(sorted(by_pattern.items())),
        # The fallback Pattern "other" itself is not a parked transaction type.
        other_patterns=sum(
            1 for _, r in patterns if r.body["category"] == "other" and r.body["name"] != "other"
        ),
        hw_pattern_grid=_hw_grid(store, templates),
        assurance=dict(sorted(assurance.items())),
        families_per_template=_families_per_template(store, members),
    )
