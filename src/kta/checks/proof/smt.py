"""One SMT query: the formula is the *negation* of what should hold, so ``unsat`` proves it,
``sat`` gives a witness (to be replayed before anyone calls it a counterexample) and
``unknown`` (timeout, incomplete theory) is an open question, never a proof.

Pattern after the read-only reference
/Users/dingyi/Projects/agentic-codesign/backend/ventus_compiled/proof_templates/queries.py
(replayable queries with explicit unknown); reimplemented, not imported.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import z3

from kta.core.schemas.proof import Query

DEFAULT_TIMEOUT_MS = 5000


def solver_version() -> str:
    return f"z3-{z3.get_version_string()}"


@dataclass
class QueryRun:
    result: str  # unsat | sat | unknown
    seconds: float
    sha256: str
    model: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    reason: str = ""

    def record(self) -> Query:
        return Query(sha256=self.sha256, result=self.result, seconds=round(self.seconds, 6))


def _value(v: Any) -> Any:
    if z3.is_true(v):
        return True
    if z3.is_false(v):
        return False
    if z3.is_int_value(v):
        return v.as_long()
    if z3.is_rational_value(v):
        return float(v.numerator_as_long()) / float(v.denominator_as_long())
    return str(v)


def run(
    bad: Any, decls: Mapping[str, Any] | None = None, timeout_ms: int = DEFAULT_TIMEOUT_MS
) -> QueryRun:
    """Check ``bad``; on ``sat`` read ``decls`` (name -> z3 term) out of the model."""
    s = z3.Solver()
    s.set("timeout", int(timeout_ms))
    s.set("random_seed", 0)
    s.add(bad)
    text = s.to_smt2()
    digest = hashlib.sha256(text.encode()).hexdigest()
    t0 = time.perf_counter()
    r = s.check()
    dt = time.perf_counter() - t0
    if r == z3.unsat:
        return QueryRun("unsat", dt, digest)
    if r == z3.sat:
        m = s.model()
        values = {k: _value(m.eval(v, model_completion=True)) for k, v in (decls or {}).items()}
        return QueryRun("sat", dt, digest, values)
    return QueryRun("unknown", dt, digest, reason=str(s.reason_unknown()))


def enumerated(description: str, seconds: float) -> Query:
    """The Query row of a bounded enumeration (no solver involved)."""
    digest = hashlib.sha256(description.encode()).hexdigest()
    return Query(sha256=digest, result="enumerated", seconds=round(seconds, 6))
