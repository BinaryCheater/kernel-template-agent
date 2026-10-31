"""Ablation on our own families: suspected pairs, designs, coupling evidence (02 §8.4, D566).

A factor is a chain step (on = applied, off = removed) or a family variable (on = its
default/high value, off = the low one). Coupling is tested on suspected pairs only:
variables sharing a glue constraint, Templates related by a relation, Templates sharing a
resource. Small sets get a pairwise design (base, each single, each pair); larger ones a
two-level fractional factorial (base columns full, the rest aliased to products of three
or more base columns, so main effects are clear of two-factor interactions). The
criterion: the joint effect deviates from the product of single effects beyond the noise
band. Results become ``Coupling`` records (proposed, basis ``measured``) with the
measurements as evidence.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

from kta.core.controls import baseline
from kta.core.errors import ConfigError
from kta.core.expr import Expr, ExprError
from kta.core.knobs import Knobs
from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import (
    Chain,
    Coupling,
    CouplingMember,
    Family,
    RangeSpec,
    Scope,
    Template,
)
from kta.store.base import Store

Level = Literal[0, 1]  # 0 = toggled off, 1 = as in the family


@dataclass(frozen=True)
class Factor:
    name: str  # chain item key or variable name
    kind: Literal["template", "variable"]
    template: str | None = None  # name@version of a chain step
    on: Any = True
    off: Any = False


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    reason: str  # shared_variable | template_relation | shared_resource


@dataclass(frozen=True)
class Design:
    mode: Literal["pairwise", "fractional"]
    factors: list[Factor]
    pairs: list[Pair]
    runs: list[dict[str, Level]]
    aliases: dict[str, str] = field(default_factory=lambda: dict[str, str]())  # fractional

    def settings(self, run: Mapping[str, Level]) -> dict[str, Any]:
        """The family setting of one run: chain steps to drop and variable values."""
        out: dict[str, Any] = {"drop": [], "values": {}}
        for f in self.factors:
            level = run.get(f.name, 1)
            if f.kind == "template" and level == 0:
                out["drop"].append(f.name)
            elif f.kind == "variable":
                out["values"][f.name] = f.on if level else f.off
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def factors(family: Family, chain: Chain) -> list[Factor]:
    """Every chain step, and every family variable with at least two values."""
    out = [Factor(i.key, "template", i.template) for i in chain.items]
    for v in family.variables:
        d = v.domain
        lo, hi = (d.lo, d.hi) if isinstance(d, RangeSpec) else (d[0], d[-1]) if d else (0, 0)
        if lo != hi:
            out.append(Factor(v.name, "variable", on=hi, off=lo))
    return out


def _names(src: str, declared: set[str]) -> set[str]:
    try:
        return set(Expr.parse(src, declared).free_vars)
    except ExprError:
        return set()


def suspected_pairs(
    fs: Sequence[Factor], family: Family, chain: Chain, templates: Mapping[str, Template]
) -> list[Pair]:
    """Pairs worth a coupling test (table ``ablation.suspects``)."""
    by_name = {f.name: f for f in fs}
    found: dict[tuple[str, str], str] = {}

    def add(a: str, b: str, reason: str) -> None:
        if a != b and a in by_name and b in by_name:
            found.setdefault((min(a, b), max(a, b)), reason)

    declared = {v.name for v in family.variables}
    for g in family.glue_constraints:
        for a, b in itertools.combinations(sorted(_names(g.expr, declared)), 2):
            add(a, b, "shared_variable")
    steps = [(i.key, templates.get(i.template)) for i in chain.items]
    for (ka, ta), (kb, tb) in itertools.combinations(steps, 2):
        if ta is None or tb is None:
            continue
        targets_a = {r.target.partition("@")[0] for r in ta.relations}
        targets_b = {r.target.partition("@")[0] for r in tb.relations}
        if tb.name in targets_a or ta.name in targets_b:
            add(ka, kb, "template_relation")
        elif set(ta.signature.resources) & set(tb.signature.resources):
            add(ka, kb, "shared_resource")
    return [Pair(a, b, r) for (a, b), r in sorted(found.items())]


def _fractional(names: list[str], max_runs: int) -> tuple[list[dict[str, Level]], dict[str, str]]:
    k = len(names)
    for m in range(1, int(math.log2(max_runs)) + 1):
        words = [c for size in range(3, m + 1) for c in itertools.combinations(range(m), size)]
        if m + len(words) >= k:
            break
    else:
        raise ConfigError(f"{k} factors do not fit a fractional design of {max_runs} runs")
    base, extra = names[:m], names[m:]
    aliases = {n: "*".join(base[i] for i in w) for n, w in zip(extra, words, strict=False)}
    runs: list[dict[str, Level]] = []
    for bits in itertools.product((1, 0), repeat=m):
        run: dict[str, Level] = dict(zip(base, bits, strict=True))  # type: ignore[arg-type]
        for n, w in zip(extra, words, strict=False):
            sign = math.prod(1 if bits[i] else -1 for i in w)
            run[n] = 1 if sign > 0 else 0
        runs.append(run)
    return runs, aliases


def _off(names: Sequence[str]) -> dict[str, Level]:
    return {n: 0 for n in names}


def design(fs: Sequence[Factor], pairs: Sequence[Pair], controls: Knobs | None = None) -> Design:
    """Runs for the factors in suspected pairs (the others stay as in the family); run
    limits are the knobs ``learning.ablation_*`` (default: the registry baseline)."""
    c = controls or baseline()
    involved = [f for f in fs if any(f.name in (p.a, p.b) for p in pairs)]
    names = [f.name for f in involved]
    pairwise = 1 + len(names) + len(pairs)
    if pairwise <= c.ablation_pairwise_max_runs:
        runs = [_off([]), *(_off([n]) for n in names), *(_off([p.a, p.b]) for p in pairs)]
        return Design("pairwise", involved, list(pairs), runs)
    runs, aliases = _fractional(names, c.ablation_max_runs)
    return Design("fractional", involved, list(pairs), runs, aliases)


@dataclass(frozen=True)
class Estimate:
    pair: Pair
    joint_ratio: float  # joint effect / product of single effects (time ratios)
    coupled: bool


def estimate(
    d: Design,
    times: Sequence[float],
    noise_band: float | None = None,
    controls: Knobs | None = None,
) -> list[Estimate]:
    """Coupling estimates from one median time per run, in ``d.runs`` order."""
    if len(times) != len(d.runs) or any(t <= 0 for t in times):
        raise ConfigError("one positive time per design run is needed")
    band = (controls or baseline()).coupling_noise_band if noise_band is None else noise_band
    out: list[Estimate] = []
    for p in d.pairs:
        if d.mode == "pairwise":
            idx = {tuple(sorted(r)): i for i, r in enumerate(d.runs)}
            t0, ta, tb = times[0], times[idx[(p.a,)]], times[idx[(p.b,)]]
            tab = times[idx[tuple(sorted((p.a, p.b)))]]
            ratio = (tab / t0) / ((ta / t0) * (tb / t0))
        else:  # T(--)T(++) / (T(+-)T(-+)) = exp(4 b_ab) in ±1 coding of log time
            n = len(d.runs)
            b = sum(
                (1 if r[p.a] else -1) * (1 if r[p.b] else -1) * math.log(t)
                for r, t in zip(d.runs, times, strict=True)
            )
            ratio = math.exp(4 * b / n)
        out.append(Estimate(p, ratio, abs(ratio - 1.0) > band))
    return out


def propose_couplings(
    store: Store,
    d: Design,
    estimates: Sequence[Estimate],
    *,
    family_id: str,
    target: str,
    workload: str,
    evidence: list[str],
    actor: Actor,
    now: datetime,
    controls: Knobs | None = None,
) -> list[str]:
    """Write a proposed Coupling per coupled pair, with the run measurements as evidence."""
    kinds = {f.name: f for f in d.factors}
    band = (controls or baseline()).coupling_noise_band
    ids: list[str] = []
    for e in estimates:
        if not e.coupled:
            continue
        members = [
            CouplingMember(
                kind=kinds[n].kind,
                name=kinds[n].template or n if kinds[n].kind == "template" else n,
            )
            for n in (e.pair.a, e.pair.b)
        ]
        body = Coupling(
            members=members,
            scope=Scope(targets=[target], workload=workload),
            effect=(
                f"{e.pair.a} and {e.pair.b} ({e.pair.reason}): joint/product {e.joint_ratio:.3f}"
            ),
            joint_ratio=e.joint_ratio,
            noise_band=band,
            basis="measured",
        )
        refs: dict[str, Any] = {"family": family_id, "evidence": sorted(evidence)}
        ids.append(store.put(Record.new("Coupling", body, actor, now, refs=refs)))
    return ids
