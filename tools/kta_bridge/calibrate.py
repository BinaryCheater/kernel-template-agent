"""Deterministic constants for the decision-level form.

The form is linear in its constants once each row's active branch of the max is known, so the fit
alternates: assign branches, solve a relative least squares with non-negative constants, repeat.
No random search: the same rows give the same constants.
"""

import numpy as np
from tools.kta_bridge import forms


def nnls(a: np.ndarray, b: np.ndarray, sweeps: int = 4000) -> np.ndarray:
    """Non-negative least squares by coordinate descent."""
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1.0
    x_ = a / scale
    g, c = x_.T @ x_, x_.T @ b
    x = np.zeros(a.shape[1])
    for _ in range(sweeps):
        for j in range(len(x)):
            if g[j, j] > 0:
                x[j] = max(0.0, x[j] + (c[j] - g[j] @ x) / g[j, j])
    return x / scale


def fit(
    rows: list[dict],
    target: str,
    start: dict[str, float],
    frozen: tuple[str, ...] = (),
    rounds: int = 25,
) -> dict[str, float]:
    """`rows` carry `a` (forms.Amounts) and the target time; `frozen` constants keep their start."""
    rows = [r for r in rows if r.get(target)]
    c = dict(start)
    names = sorted({k for r in rows for k in _keys(r["a"])} - set(frozen))
    previous: list[str] | None = None
    for _ in range(rounds):
        active = [forms.active_branch(r["a"], c) for r in rows]
        if active == previous:
            break
        previous = active
        a = np.zeros((len(rows), len(names)))
        b = np.ones(len(rows))
        for i, (r, branch) in enumerate(zip(rows, active, strict=True)):
            additive, branches = forms.terms(r["a"])
            for k, v in {**additive, **branches[branch]}.items():
                if k in frozen:
                    b[i] -= c[k] * v / r[target]
                else:
                    a[i, names.index(k)] = v / r[target]
        x = nnls(a, b)
        c = {**{k: c[k] for k in frozen}, **dict(zip(names, map(float, x), strict=True))}
    return c


def _keys(a: forms.Amounts) -> set[str]:
    additive, branches = forms.terms(a)
    return set(additive) | {k for b in branches.values() for k in b}
