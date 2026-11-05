"""Exact MIP of the decision-level form (`forms.py`) over one family's menus.

Mechanical rule: one one-hot per decision factor; a term that reads several factors is a table over
their product, tied to the factors by linked indicators; the max is an epigraph. Occupancy is not a
free choice: the machine takes the minimum of its walls, so it is encoded as that minimum of two
table values. Nothing here is fitted or chosen per family beyond the declared factor supports.
"""

import dataclasses
import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import highspy
import numpy as np

from tools.kta_bridge import forms


@dataclass(frozen=True)
class Family:
    """Factors A, B, T and the functions that close the form over them."""

    a: tuple  # tile choices
    b: tuple  # the remaining choices
    t: tuple  # threads
    amounts: Callable  # (a, b, t) -> Amounts with occ from the walls
    smem: Callable  # (a, b) -> bytes per block
    regs: Callable  # (a, b, t) -> registers per thread
    assign: Callable  # (a, b, t) -> decision dict
    regs_support_b: bool = False  # the register estimate also reads B
    width: Callable = lambda b: b  # the part of B that tiles the K loop (or all of it)
    stage: Callable = lambda b: b  # the pipeline depth in B


def amounts_at(f: Family, a, b, t, occ: int) -> forms.Amounts:
    return dataclasses.replace(f.amounts(a, b, t), occ=occ)


def value(f: Family, a, b, t, c: dict) -> float:
    return forms.time_us(f.amounts(a, b, t), c)


def walls(f: Family, a, b, t) -> tuple[int, int]:
    smem_occ = 102400 // (f.smem(a, b) + 1024)
    warps = math.ceil(t / 32)
    rw = math.ceil(max(f.regs(a, b, t), 16) * 32 / 256) * 256
    return smem_occ, 65536 // (warps * rw)


class Model:
    """Thin column and row store over HiGHS."""

    def __init__(self) -> None:
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("mip_rel_gap", 0.0)
        self.h.setOptionValue("mip_abs_gap", 1e-6)
        self.n = 0
        self.rows = 0

    def binary(self, cost: float = 0.0) -> int:
        self.h.addVar(0.0, 1.0)
        self.h.changeColCost(self.n, cost)
        self.h.changeColIntegrality(self.n, highspy.HighsVarType.kInteger)
        self.n += 1
        return self.n - 1

    def real(self, lo: float, cost: float) -> int:
        self.h.addVar(lo, highspy.kHighsInf)
        self.h.changeColCost(self.n, cost)
        self.n += 1
        return self.n - 1

    def row(self, lo: float, hi: float, terms: dict[int, float]) -> None:
        idx = np.array(list(terms), dtype=np.int32)
        val = np.array(list(terms.values()), dtype=np.float64)
        self.h.addRow(lo, hi, len(idx), idx, val)
        self.rows += 1

    def one_hot(self, keys: list, cost: Callable = lambda _: 0.0) -> dict:
        v = {k: self.binary(cost(k)) for k in keys}
        self.row(1, 1, dict.fromkeys(v.values(), 1.0))
        return v

    def link(self, left: dict, right: dict, cost: Callable = lambda _: 0.0) -> dict:
        """One-hot over left × right, whose marginals are the two one-hots."""
        z = {(i, j): self.binary(cost((i, j))) for i in left for j in right}
        for i, x in left.items():
            self.row(0, 0, {**{z[i, j]: 1.0 for j in right}, x: -1.0})
        for j, y in right.items():
            self.row(0, 0, {**{z[i, j]: 1.0 for i in left}, y: -1.0})
        return z

    def group(self, source: dict, key: Callable) -> dict:
        """A one-hot over values of `key`, tied to the one-hot `source`."""
        out: dict = {}
        for k in sorted({key(s) for s in source}):
            out[k] = self.binary()
            self.row(0, 0, {**{source[s]: 1.0 for s in source if key(s) == k}, out[k]: -1.0})
        return out


def build(f: Family, c: dict, ceiling_us: float = 1e6) -> tuple[Model, dict]:
    m = Model()
    a = m.one_hot(list(f.a))
    b = m.one_hot(list(f.b))
    t = m.one_hot(list(f.t))
    ab = m.link(a, b)
    legal = {k for k in ab if f.smem(*k) + 1024 <= 102400}
    for k in ab:
        if k not in legal:
            m.h.changeColBounds(ab[k], 0.0, 0.0)
    rep_b = f.b[0]
    # the two wall values and their minimum with the fixed limits
    smem_v = m.group(ab, lambda k: max(1, min(24, 102400 // (f.smem(*k) + 1024))))
    if f.regs_support_b:
        abt = m.link(ab, t)
        reg_v = m.group(abt, lambda k: walls(f, k[0][0], k[0][1], k[1])[1])
    else:
        at = m.link(a, t)
        reg_v = m.group(at, lambda k: walls(f, k[0], rep_b, k[1])[1])
    pair = m.link(smem_v, reg_v)
    tval = m.link(pair, t)
    occ_of = lambda k: max(1, min(k[0][0], k[0][1], 1536 // k[1], 24))  # noqa: E731
    occ = m.group(tval, occ_of)
    # every remaining term reads at most (a, width of b, t, occ), except the stage count, which
    # enters linearly or only through the recurrence
    width = m.group(b, f.width)
    stage = m.group(b, f.stage)
    q = m.link(m.link(m.link(a, width), t), occ)
    y = m.real(0.0, 1.0)
    add0, _ = forms.terms(amounts_at(f, f.a[0], rep_b, f.t[0], 1))
    m.h.changeObjectiveOffset(sum(c.get(k, 0.0) * add0[k] for k in ("launch", "store")))
    for st, v in stage.items():
        m.h.changeColCost(v, c.get("stage", 0.0) * st)
    for (((ai, wi), ti), oi), v in q.items():
        if oi > max(1, min(walls_at(f, ai, ti), 1536 // ti, 24)):
            m.h.changeColBounds(v, 0.0, 0.0)
            continue
        bi = next(x for x in f.b if f.width(x) == wi)
        add, br = forms.terms(amounts_at(f, ai, bi, ti, oi))
        m.h.changeColCost(v, sum(c.get(k, 0.0) * add[k] for k in ("wave", "trip", "barrier")))
        floor = max(forms.branch_us(br[k], c) for k in br if k != "recurrence")
        m.row(0.0, highspy.kHighsInf, {y: 1.0, v: -floor})
        for st, w in stage.items():
            sb = next((x for x in f.b if f.width(x) == wi and f.stage(x) == st), None)
            if sb is None:
                continue
            rec = forms.branch_us(forms.terms(amounts_at(f, ai, sb, ti, oi))[1]["recurrence"], c)
            if rec > floor:  # both chosen: y >= rec
                m.row(-rec, highspy.kHighsInf, {y: 1.0, v: -rec, w: -rec})
    return m, {"a": a, "b": b, "t": t}


def walls_at(f: Family, a, t) -> int:
    """The register wall at this tile and thread count, the weakest over B."""
    return max(walls(f, a, b, t)[1] for b in f.b)


def solve(
    f: Family, c: dict, k: int = 16, time_limit: float = 120.0, diverse: bool = False
) -> dict:
    """The k best assignments in objective order, by re-solving with no-good cuts.

    `diverse` cuts the whole tile and thread choice after each pick: a shortlist of distinct
    structures, each with its best remaining choices, instead of near-ties of one structure.
    """
    t0 = time.perf_counter()
    m, x = build(f, c)
    m.h.setOptionValue("time_limit", time_limit)
    built = time.perf_counter() - t0
    out = []
    status = []
    for _ in range(k):
        m.h.run()
        st = m.h.modelStatusToString(m.h.getModelStatus())
        status.append(st)
        if st != "Optimal":
            break
        sol = m.h.getSolution().col_value
        pick = [next(key for key, v in x[n].items() if sol[v] > 0.5) for n in ("a", "b", "t")]
        out.append(
            {
                "x": f.assign(*pick),
                "key": pick,
                "objective": m.h.getInfo().objective_function_value,
                "direct": value(f, *pick, c),
            }
        )
        names = ("a", "t") if diverse else ("a", "b", "t")
        cut = {x[n][p]: 1.0 for n, p in zip(("a", "b", "t"), pick, strict=True) if n in names}
        m.row(-highspy.kHighsInf, len(names) - 1.0, cut)
    return {
        "picks": out,
        "status": status,
        "columns": m.n,
        "rows": m.rows,
        "build_s": built,
        "solve_s": time.perf_counter() - t0 - built,
    }


def enumerate_all(f: Family, c: dict) -> list[tuple[float, tuple]]:
    """Direct evaluation of the same form over the whole product, the control for translation."""
    rows = []
    for k in itertools.product(f.a, f.b, f.t):
        if f.smem(k[0], k[1]) + 1024 <= 102400:
            rows.append((value(f, *k, c), k))
    return sorted(rows, key=lambda r: r[0])
