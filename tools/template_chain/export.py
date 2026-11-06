"""Mechanical MIP export of a family's form: any `form.py`, no per-family code.

One one-hot per declared factor. Each term of the form is tabulated over the whole legal domain,
its true dependency set is found by testing, and it becomes a table over that factor subset, tied
to the factor one-hots. Each kernel's max is an epigraph. So the MIP has the structure the agent's
form really has: a form that reads every factor in one term exports as one joint one-hot.
"""

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass

import highspy
from tools.kta_bridge import forms
from tools.kta_bridge.mip import Model


def as_list(a) -> list[forms.Amounts]:
    return list(a) if isinstance(a, list | tuple) else [a]


@dataclass
class Domain:
    names: tuple
    factors: dict
    points: list  # tuples in factor order, legal only
    kernels: list  # per point, the Amounts of each kernel in launch order


def domain(form, shape: dict) -> Domain:
    names = tuple(form.FACTORS)
    points, kernels = [], []
    for values in itertools.product(*form.FACTORS.values()):
        x = dict(zip(names, values, strict=True))
        if form.legal(shape, x):
            points.append(values)
            kernels.append(as_list(form.amounts(shape, x, None)))
    return Domain(names, dict(form.FACTORS), points, kernels)


def time_at(amounts: list[forms.Amounts], c: dict) -> float:
    return sum(forms.time_us(a, c) for a in amounts)


def dependency(points: list, values: list[float], width: int) -> tuple:
    """The smallest factor subset found greedily on which `values` is a function."""
    keep = list(range(width))
    for f in range(width):
        trial = [k for k in keep if k != f]
        seen: dict = {}
        if all(
            abs(seen.setdefault(tuple(p[k] for k in trial), v) - v) < 1e-9
            for p, v in zip(points, values, strict=True)
        ):
            keep = trial
    return tuple(keep)


def build(d: Domain, c: dict) -> tuple[Model, dict]:
    m = Model()
    width = len(d.names)
    y = {f: m.one_hot(list(d.factors[n])) for f, n in enumerate(d.names)}
    tables: dict = {}

    def table(subset: tuple) -> dict:
        if subset not in tables:
            keys = list(itertools.product(*(d.factors[d.names[f]] for f in subset)))
            z = {k: m.binary() for k in keys}
            m.row(1, 1, dict.fromkeys(z.values(), 1.0))
            for pos, f in enumerate(subset):
                groups = defaultdict(dict)
                for k, v in z.items():
                    groups[k[pos]][v] = 1.0
                for value, row in groups.items():
                    m.row(0, 0, {**row, y[f][value]: -1.0})
            tables[subset] = z
        return tables[subset]

    z = table(tuple(range(width)))  # the legal points, so nothing outside the domain is reachable
    legal = set(d.points)
    for k, v in z.items():
        if k not in legal:
            m.h.changeColBounds(v, 0.0, 0.0)

    def term(values: list[float]) -> tuple[tuple, dict]:
        subset = dependency(d.points, values, width)
        out: dict = {}
        for p, v in zip(d.points, values, strict=True):
            out[tuple(p[f] for f in subset)] = v
        return subset, out

    for j in range(max(map(len, d.kernels))):
        add, branches = [], defaultdict(list)
        for kernels in d.kernels:  # a point with fewer kernels costs nothing in this slot
            additive, br = forms.terms(kernels[j]) if j < len(kernels) else ({}, {})
            add.append(forms.branch_us(additive, c))
            for name in ("compute", "copy", "recurrence", "dram"):
                branches[name].append(forms.branch_us(br.get(name, {}), c))
        subset, values = term(add)
        col = table(subset)
        for k, v in values.items():
            m.h.changeColCost(col[k], v)
        e = m.real(0.0, 1.0)
        for values_ in branches.values():
            if not any(values_):
                continue
            subset, values = term(values_)
            col = table(subset)
            m.row(0.0, highspy.kHighsInf, {e: 1.0, **{col[k]: -v for k, v in values.items()}})
    return m, y


def solve(d: Domain, c: dict, k: int, structure: tuple, exclude: tuple = (), limit=30.0) -> dict:
    """The k best points, by re-solving; `structure` factors are cut after each pick."""
    t0 = time.perf_counter()
    m, y = build(d, c)
    m.h.setOptionValue("time_limit", limit)
    for point in exclude:
        m.row(
            -highspy.kHighsInf,
            len(d.names) - 1.0,
            {y[f][v]: 1.0 for f, v in enumerate(point)},
        )
    picks, status = [], []
    for _ in range(k):
        m.h.run()
        st = m.h.modelStatusToString(m.h.getModelStatus())
        status.append(st)
        if st != "Optimal":
            break
        sol = m.h.getSolution().col_value
        point = tuple(next(v for v, col in y[f].items() if sol[col] > 0.5) for f in y)
        picks.append({"x": dict(zip(d.names, point, strict=True)), "objective": _objective(m)})
        cut = {y[f][point[f]]: 1.0 for f, n in enumerate(d.names) if n in structure}
        m.row(-highspy.kHighsInf, len(cut) - 1.0, cut)
    return {
        "picks": picks,
        "status": status,
        "columns": m.n,
        "rows": m.rows,
        "seconds": time.perf_counter() - t0,
    }


def _objective(m: Model) -> float:
    return m.h.getInfo().objective_function_value


def enumerate_all(d: Domain, c: dict) -> list[tuple[float, tuple]]:
    """Direct evaluation of the same form, the control for the translation."""
    return sorted(
        ((time_at(k, c), p) for p, k in zip(d.points, d.kernels, strict=True)),
        key=lambda r: r[0],
    )
