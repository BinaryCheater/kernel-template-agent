"""Shortlist selection before compilation and re-ranking after it.

Method choice (formulation-solve-models.md): a space below EXHAUSTIVE_LIMIT points is
enumerated completely; a larger one goes to the MIP when its constraints and cost
terms are linear and exportable, else to pruned enumeration with a node budget and,
when an rng is given, shuffled value order. "Model-optimal" is claimed only for an
exhaustive enumeration, or a MIP with gap 0 under a model without interaction terms
(the MIP objective ignores them), and is scoped to family, formulation, model version,
hardware model and shape bucket.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from typing import Any

from kta.core.errors import NotSupported, SchemaError
from kta.core.schemas.effects import LEGACY_KIND_PREFIX
from kta.core.schemas.evaluation import Build, Case, Prediction
from kta.core.schemas.hardware import HwModel
from kta.core.schemas.hwmodel import HwArch
from kta.formulation import Formulation, per_unit_work, price_point, work_terms
from kta.knowledge.types import FamilyPackage
from kta.models import MachineModel, NoResidentCta, UnitModel, machine
from kta.solve.enumerate import enumerate_space, space_size
from kta.solve.mip import NotExportable, solve_mip
from kta.solve.result import Candidate, Method, SolveResult

EXHAUSTIVE_LIMIT = 10**6
DEFAULT_MAX_NODES = 2_000_000
# The family variable read as threads per CTA in compile feedback (decision D250).
THREADS_VAR = "threads"


def shape_buckets(case: Case, family: FamilyPackage | None = None) -> list[dict[str, Any]]:
    """One bucket per shape of the case, or the family's dispatch buckets when they name
    a shape each: ``{"name", "shape", "weight"}``."""
    dispatch = family.family.dispatch if family else None
    if dispatch and dispatch.buckets and all("shape" in b for b in dispatch.buckets):
        return [
            {"name": str(b.get("name", f"b{i}")), "shape": dict(b["shape"]), "weight": 1.0}
            for i, b in enumerate(dispatch.buckets)
        ]
    return [
        {"name": f"s{i}", "shape": dict(item["shape"]), "weight": float(item.get("weight", 1.0))}
        for i, item in enumerate(case.shape_dist)
    ]


def _bucket(shape_bucket: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    if "shape" in shape_bucket and isinstance(shape_bucket["shape"], Mapping):
        name = shape_bucket.get("name")
        return dict(shape_bucket["shape"]), str(name) if name is not None else None
    return dict(shape_bucket), None


def _check_inputs(f: Formulation, shape: Mapping[str, Any], hw: MachineModel) -> None:
    missing_hw = sorted(f.hw_refs - set(hw.hw))
    if missing_hw:
        raise NotSupported(
            f"HwModel {hw.target} v{hw.hw_version} lacks hw.{', hw.'.join(missing_hw)}"
        )
    missing = sorted(f.case_refs - set(shape))
    if missing:
        raise SchemaError(f"shape {dict(shape)} lacks {missing} used by {f.family}")


class _Shortlist:
    """The ``n`` best points by (total, assignment JSON); deterministic under ties."""

    def __init__(self, n: int):
        self.n = n
        self.items: list[tuple[float, str, dict[str, Any], Prediction]] = []

    def offer(self, assignment: dict[str, Any], p: Prediction) -> None:
        self.items.append((p.total, json.dumps(assignment, sort_keys=True), assignment, p))
        if len(self.items) > 4 * self.n + 64:
            self._trim()

    def _trim(self) -> None:
        self.items.sort(key=lambda i: (i[0], i[1]))
        del self.items[self.n :]

    def sorted(self) -> list[tuple[dict[str, Any], Prediction]]:
        self._trim()
        return [(a, p) for _, _, a, p in self.items]


def solve(
    f: Formulation,
    case: Case,
    shape_bucket: dict[str, Any],
    model: UnitModel,
    hw: HwModel | MachineModel,
    shortlist: int = 64,
    time_limit_s: float = 60,
    rng: random.Random | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    arch: HwArch | None = None,
) -> SolveResult:
    """Candidates for one shape bucket, best predicted first.

    ``time_limit_s`` bounds the MIP; enumeration is bounded by ``max_nodes`` so that its
    result does not depend on machine speed (decision D248). Points are priced by L1 on
    ``hw`` resolved on ``arch`` (D464); points that hit a no-launch wall are left out and
    counted in the notes.
    """
    shape, name = _bucket(shape_bucket)
    mm = machine(hw, arch)
    _check_inputs(f, shape, mm)
    size = space_size(f)
    scope = {
        "family": f.family,
        "formulation": f.fingerprint,
        "model": model.version,
        "hw_model": f"{mm.target}@{mm.hw_version}",
        "hw_arch": mm.arch,
        "case_problem": case.problem,
        "bucket": name,
        "shape": shape,
    }
    notes: list[str] = []
    best = _Shortlist(shortlist)

    priced = _priced(f)
    if priced and mm.arch is None:
        raise NotSupported(
            f"{f.family} declares effects; HwModel {mm.target}@{mm.hw_version} has no "
            "architecture class to price them (D463)"
        )
    walled: list[int] = [0]

    def predict(assignment: dict[str, Any]) -> Prediction | None:
        if not priced:
            over, serial = per_unit_work(f, assignment, shape, mm.hw)
            return model.predict(over, serial, mm)
        try:
            return model.predict_l1(price_point(f, assignment, shape, mm), mm)
        except NoResidentCta:
            walled[0] += 1
            return None

    def offer(assignment: dict[str, Any]) -> None:
        p = predict(assignment)
        if p is not None:
            best.offer(assignment, p)

    if size >= EXHAUSTIVE_LIMIT and f.linear and _terms_linear(f, mm):
        try:
            out = solve_mip(f, shape, mm, model, shortlist, time_limit_s)
        except NotExportable as exc:
            notes.append(f"MIP not used: {exc}")
        else:
            for a in out.assignments:
                offer(a)
            notes.append(f"MIP solves: {out.solves}, runtime {out.runtime_s:.3f} s")
            return _result(
                best, name, "mip", size, len(out.assignments), 0, {}, False,
                out.first_optimal and not model.interactions, scope, notes, out.status, out.gap,
            )  # fmt: skip
    exhaustive_mode = size < EXHAUSTIVE_LIMIT
    stats = enumerate_space(
        f,
        shape,
        mm.hw,
        offer,
        max_nodes=None if exhaustive_mode else max_nodes,
        rng=None if exhaustive_mode else rng,
    )
    if not stats.exhaustive:
        notes.append(f"node budget {max_nodes} spent; not exhaustive")
    if walled[0]:
        notes.append(f"{walled[0]} feasible points hit a no-launch wall of the hardware model")
    method: Method = "enumerate"
    return _result(
        best, name, method, size, stats.feasible, stats.pruned, stats.pruned_by,
        stats.exhaustive, stats.exhaustive, scope, notes, None, None,
    )  # fmt: skip


def _priced(f: Formulation) -> bool:
    """True when some transaction is priced by an operation class (not legacy cost)."""
    return any(t.kind_var or not t.kind.startswith(LEGACY_KIND_PREFIX) for t in f.transactions)


def _terms_linear(f: Formulation, mm: MachineModel) -> bool:
    try:
        return all(t.work.is_linear for t in work_terms(f, mm))
    except NotSupported:
        return False


def _result(
    best: _Shortlist,
    bucket: str | None,
    method: Method,
    size: int,
    feasible: int,
    pruned: int,
    pruned_by: dict[str, int],
    exhaustive: bool,
    optimal: bool,
    scope: dict[str, Any],
    notes: list[str],
    status: str | None,
    gap: float | None,
) -> SolveResult:
    cands = [Candidate(a, bucket, p) for a, p in best.sorted()]
    return SolveResult(
        candidates=cands,
        method=method,
        space_size=size,
        feasible=feasible,
        pruned=pruned,
        pruned_by=pruned_by,
        exhaustive=exhaustive,
        model_optimal=optimal and bool(cands),
        scope=scope,
        mip_status=status,
        mip_gap=gap,
        notes=notes,
    )


def feedback_of(build: Build, assignment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "regs": build.regs,
        "spills": build.spills,
        "smem": build.smem,
        "threads": assignment.get(THREADS_VAR),
    }


def rerank(
    candidates: list[Candidate],
    builds: dict[str, Build],
    model: UnitModel,
    hw: HwModel | MachineModel,
    k: int = 8,
) -> list[Candidate]:
    """Re-predict built candidates with compile feedback; the best ``k`` that built.

    ``builds`` maps ``Candidate.key`` to its Build. Candidates without a Build, with a
    failed Build, or whose resources leave no CTA resident are not returned.
    """
    out: list[Candidate] = []
    for c in candidates:
        build = builds.get(c.key)
        if build is None or not build.ok:
            continue
        try:
            p = model.repredict(c.prediction, hw, feedback_of(build, c.assignment))
        except NoResidentCta:
            continue  # cannot launch; the reason is in the Build's resources
        out.append(Candidate(c.assignment, c.bucket, p))
    out.sort(key=lambda c: (c.prediction.total, json.dumps(c.assignment, sort_keys=True)))
    return out[:k]
