"""Independent interpreter for solver witnesses over contract expressions.

A witness the solver returns (a point p, x) is re-evaluated with the expression language's
own evaluator (``kta.core.expr``, plain Python semantics), not with the Z3 encoding that
produced it. Only a witness the interpreter confirms becomes a counterexample; one it
cannot evaluate or does not confirm leaves the condition unresolved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kta.core.expr import Expr, ExprError

from .encode import OTHER


def split(point: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """A flat witness ({"S": 2, "p.size_bytes": 4}) as (env, scopes)."""
    env: dict[str, Any] = {}
    scopes: dict[str, dict[str, Any]] = {"hw": {}, "p": {}, "case": {}}
    for k, v in point.items():
        root, dot, attr = k.partition(".")
        if dot and root in scopes:
            scopes[root][attr] = v
        else:
            env[k] = v
    return env, scopes


def holds(src: str, point: Mapping[str, Any]) -> bool | None:
    """The truth of ``src`` at ``point``; None when the interpreter cannot evaluate it."""
    env, scopes = split(point)
    if any(v == OTHER for v in point.values()):
        # "anything else" has no concrete string; pick one no expression names.
        env = {k: ("∅" if v == OTHER else v) for k, v in env.items()}
        scopes = {
            r: {k: ("∅" if v == OTHER else v) for k, v in s.items()} for r, s in scopes.items()
        }
    try:
        e = Expr.parse(src, set(env))
        return bool(e.eval(env, hw=scopes["hw"], p=scopes["p"], case=scopes["case"]))
    except ExprError:
        return None


def confirm(point: Mapping[str, Any], must: Mapping[str, bool]) -> tuple[bool, dict[str, Any]]:
    """Evaluate each expression of ``must`` at ``point``; agree when every one has the
    required truth value. Returns (agrees, per-expression values)."""
    got = {src: holds(src, point) for src in must}
    return all(got[s] is want for s, want in must.items()), {"values": got}
