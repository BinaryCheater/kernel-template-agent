"""The bounded chain enumerator and composition statistics (WP19; 02 §5.5, D-T19, E12).

Grammar (D542): a symbol is a transaction pattern at a level. A Template rewrites the
symbol ``(refines, level_from)`` into ``(p, level_to)`` for every ``p`` in ``produces``.
A chain is a derivation from the workload's start symbols in which every start symbol is
refined at least once; produced symbols may stay unrefined. The enumerator expands
symbols in canonical order (no permutations of one set), keeps the newest usable version
of each Template name, and per candidate step applies the filters: capability on the
target (WP14), Γ on the case (false drops the step; unknown keeps it, counted), and the
``excludes``/``requires`` relations. Each chain is bounded by the SOL of the case's
minimal work on the resources its steps reach (a resource of the minimal work that no
step reaches leaves the chain unbounded, counted); chains whose bound exceeds the
incumbent are pruned. Counts make the coverage visible: which start symbols have no
rule, how many chains each filter removed.

Composition statistics (co-occurrence, order, conditional frequency "pattern at level →
Template") are a view over reconstructed chains, never authored.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.core.schemas.evaluation import Case
from kta.core.schemas.knowledge import Chain, Template
from kta.knowledge import pool
from kta.knowledge.capabilities import template_availability
from kta.knowledge.scope import problem_params, scope_holds
from kta.models import MachineModel
from kta.store.base import Store

Symbol = tuple[str, int]  # (pattern, level)
MAX_STEPS = 12  # default of knob formulation.chain_max_steps
MAX_CHAINS = 2000  # default of knob formulation.chain_max_expansions


def ref(t: Template) -> str:
    return pool.template_ref(t.name, t.version)


def start_symbols(templates: Iterable[Template]) -> list[Symbol]:
    """Symbols a chain's steps consume that no earlier step produced (in chain order)."""
    produced: set[Symbol] = set()
    out: list[Symbol] = []
    for t in templates:
        s = (t.refines, t.level_from)
        if s not in produced and s not in out:
            out.append(s)
        produced |= {(p, t.level_to) for p in t.produces}
    return out


@dataclass(frozen=True)
class EnumeratedChain:
    steps: tuple[str, ...]  # Template refs in derivation order
    bound_us: float | None  # None: some resource of the minimal work is not reached
    gamma_unknown: tuple[str, ...] = ()  # steps whose Γ could not be evaluated here
    unrefined: tuple[Symbol, ...] = ()  # produced symbols left unrefined

    @property
    def key(self) -> frozenset[str]:
        return frozenset(self.steps)


@dataclass
class ChainEnumeration:
    start: list[Symbol]
    chains: list[EnumeratedChain] = field(default_factory=lambda: list[EnumeratedChain]())
    counts: Counter[str] = field(default_factory=lambda: Counter[str]())
    rules: dict[str, list[str]] = field(default_factory=lambda: dict[str, list[str]]())
    missing: list[Symbol] = field(default_factory=lambda: list[Symbol]())  # no rule at all
    filtered: dict[str, str] = field(default_factory=lambda: dict[str, str]())  # ref -> why
    exhaustive: bool = True

    @property
    def best(self) -> EnumeratedChain | None:
        """The best-bound chain: bounded first, smallest bound, more steps, then name."""
        ranked = sorted(
            self.chains,
            key=lambda c: (c.bound_us is None, c.bound_us or 0.0, -len(c.steps), c.steps),
        )
        return ranked[0] if ranked else None

    def find(self, steps: Iterable[str]) -> EnumeratedChain | None:
        want = frozenset(steps)
        return next((c for c in self.chains if c.key == want), None)

    def coverage(self) -> dict[str, Any]:
        uses = Counter(s for c in self.chains for s in set(c.steps))
        return {
            "start": [f"{p}@L{lv}" for p, lv in self.start],
            "missing_rules": [f"{p}@L{lv}" for p, lv in self.missing],
            "rules_per_symbol": {k: len(v) for k, v in self.rules.items()},
            "chains": len(self.chains),
            "chains_per_template": dict(sorted(uses.items())),
            "counts": dict(self.counts),
            "filtered": self.filtered,
            "exhaustive": self.exhaustive,
        }


def _newest(templates: Iterable[tuple[str, Template]], store: Store) -> list[Template]:
    best: dict[str, Template] = {}
    for rid, t in templates:
        if pool.usable(store, rid) and (t.name not in best or t.version > best[t.name].version):
            best[t.name] = t
    return [best[k] for k in sorted(best)]


def _reached(t: Template, mm: MachineModel) -> set[str]:
    out: set[str] = set()
    domains = {v.name: v.domain for v in t.variables if isinstance(v.domain, list)}
    for tx in t.effects.transactions:
        # The kind may name a Template variable whose values are (or map to) classes.
        for kind in {tx.kind, *tx.kind_map.values(), *map(str, domains.get(tx.kind, []))}:
            op = mm.op_class(kind)
            out |= {r.resource for r in op.rows} if op else set()
    return out


def _relations_ok(steps: list[Template]) -> bool:
    names = {t.name for t in steps}
    for t in steps:
        for r in t.relations:
            target = r.target.split("@", 1)[0]
            if r.type == "excludes" and target in names:
                return False
            if r.type == "requires" and target not in names:
                return False
    return True


def enumerate_chains(
    store: Store,
    case: Case,
    start: list[Symbol],
    mm: MachineModel,
    sol_us: Mapping[str, float],
    incumbent_us: float | None = None,
    max_steps: int = MAX_STEPS,
    max_chains: int = MAX_CHAINS,
    target: str | None = None,
) -> ChainEnumeration:
    """Legal chains for ``case`` from ``start``, bounded by SOL and pruned by ``incumbent_us``."""
    res = ChainEnumeration(list(start))
    target = target if target is not None else case.target
    params = problem_params(store, case)
    rules: dict[Symbol, list[Template]] = {}
    gamma_unknown: set[str] = set()
    for t in _newest(pool.templates(store), store):
        a = template_availability(store, t.requires, target)
        if not a.available:
            res.filtered[ref(t)] = f"capability: missing {a.missing}, unconfirmed {a.unconfirmed}"
            res.counts["capability_filtered"] += 1
            continue
        holds = scope_holds(t, case, mm.hw, params).holds
        if holds is False:
            res.filtered[ref(t)] = f"Γ false: {t.scope.predicate}"
            res.counts["gamma_false"] += 1
            continue
        if holds is None:
            gamma_unknown.add(ref(t))
        rules.setdefault((t.refines, t.level_from), []).append(t)
    res.rules = {f"{p}@L{lv}": [ref(t) for t in ts] for (p, lv), ts in sorted(rules.items())}
    res.missing = [s for s in start if s not in rules]
    if res.missing:
        res.counts["start_without_rule"] = len(res.missing)
        return res
    seen: set[frozenset[str]] = set()

    def finish(steps: list[Template], left: list[Symbol]) -> None:
        if not _relations_ok(steps):
            res.counts["relation"] += 1
            return
        refs = tuple(ref(t) for t in steps)
        if frozenset(refs) in seen:
            return
        seen.add(frozenset(refs))
        reached = set().union(*(_reached(t, mm) for t in steps))
        bound = max((v for r, v in sol_us.items() if r in reached), default=None)
        if any(r not in reached for r in sol_us):
            bound = None
            res.counts["unbounded"] += 1
        if bound is not None and incumbent_us is not None and bound > incumbent_us:
            res.counts["pruned_incumbent"] += 1
            return
        unknown = tuple(r for r in refs if r in gamma_unknown)
        res.chains.append(EnumeratedChain(refs, bound, unknown, tuple(left)))

    def expand(
        pending: list[Symbol], steps: list[Template], refined: set[Symbol], left: list[Symbol]
    ) -> None:
        res.counts["nodes"] += 1
        if res.counts["nodes"] > max_chains:
            res.exhaustive = False
            return
        if not pending:
            finish(steps, left)
            return
        sym, rest = pending[0], pending[1:]
        if sym not in start or sym in refined:
            expand(rest, steps, refined, [*left, sym])  # leave a produced symbol unrefined
        if len(steps) >= max_steps:
            res.counts["depth_cut"] += 1
            return
        for t in rules.get(sym, []):
            produced = [(p, t.level_to) for p in t.produces]
            todo = sorted(set(rest) | set(produced))
            expand(todo, [*steps, t], refined | {sym}, left)

    expand(sorted(start), [], set(), [])
    res.counts["legal"] = len(res.chains)
    return res


# ---- composition statistics (a derived view) -------------------------------------------


def composition_stats(chains: Iterable[Chain], templates: Mapping[str, Template]) -> dict[str, Any]:
    """Co-occurrence, order and conditional frequency over reconstructed chains (02 §5.5).

    ``templates`` maps a Template ref to its body (for the pattern and level it refines);
    a ref without a body is counted under ``unknown_templates``.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    order: Counter[tuple[str, str]] = Counter()
    given: Counter[tuple[str, str]] = Counter()  # ("pattern@L<level>", template name)
    per_symbol: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    n = 0
    for ch in chains:
        n += 1
        names = [i.template.split("@", 1)[0] for i in ch.items]
        for a in sorted(set(names)):
            for b in sorted(set(names)):
                if a < b:
                    pairs[(a, b)] += 1
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if a != b:
                    order[(a, b)] += 1
        for item in ch.items:
            t = templates.get(item.template)
            if t is None:
                unknown[item.template] += 1
                continue
            sym = f"{t.refines}@L{t.level_from}"
            per_symbol[sym] += 1
            given[(sym, t.name)] += 1
    conditional = {
        f"{sym} → {name}": round(c / per_symbol[sym], 4) for (sym, name), c in sorted(given.items())
    }
    return {
        "chains": n,
        "co_occurrence": {f"{a}+{b}": c for (a, b), c in sorted(pairs.items())},
        "order": {f"{a}<{b}": c for (a, b), c in sorted(order.items())},
        "conditional": conditional,
        "unknown_templates": dict(unknown),
    }


def composition_view(store: Store) -> dict[str, Any]:
    """``composition_stats`` over every usable Chain record in the store."""
    chains = [
        Chain.model_validate(rec.body)
        for rid, rec in pool.with_ids(pool.records(store, "Chain"))
        if pool.usable(store, rid)
    ]
    templates = {ref(t): t for _, t in pool.templates(store)}
    return composition_stats(chains, templates)
