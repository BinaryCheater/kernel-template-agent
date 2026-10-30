"""full_check and verify_space (spec/modules/measure-check.md, docs/design/50 §4).

A module is compiled for one shape (its LaunchSpec is fixed), so checking other shapes
needs a new build. Callers pass ``build_for(shape) -> (GeneratedKernel, BuiltModule)``;
generation and compilation stay with the orchestrator, and checks never imports
knowledge (decision D227). ``full_check`` runs, for the module's own shape, numeric,
sanitizer, anti-cheat and SASS checks; numeric checks for every Contract edge shape and
every hidden shape; and the Contract identity check against the Case record. It writes
one Check per sub-check and an aggregate Check referencing them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from kta.backends.base import BuiltModule, GeneratedKernel
from kta.checks import hidden
from kta.checks.anticheat import anticheat_check
from kta.checks.base import Checked, Checker, Evidence, actor, write_check
from kta.checks.numeric import NAME as NUMERIC
from kta.checks.numeric import dists_for, run_numeric
from kta.checks.sanitizer import sanitizer_check
from kta.checks.sass import sass_check
from kta.core.context import Context
from kta.core.errors import CheckerError, ExecutionError, NotSupported
from kta.core.ids import canonical_json, sha256_hex
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Case, Contract, Problem
from kta.core.schemas.knowledge import SassExpect
from kta.core.schemas.process import Report, StatusChange

NAME = "full_checker"
BuildFor = Callable[[dict[str, Any]], tuple[GeneratedKernel, BuiltModule]]


def _hash(body: Any) -> str:
    return sha256_hex(canonical_json(body))


def identity_failures(
    c: Checker, case: Case, case_id: str, contract: Contract
) -> list[dict[str, Any]]:
    """The Case record is the one checked against, and names this exact Contract."""
    rec = c.store.get(case_id)
    out: list[dict[str, Any]] = []
    if rec.body != case.model_dump(mode="json"):
        out.append({"check": "anticheat.case", "detail": "Case differs from the Case record"})
    cref = rec.refs.get("contract")
    if not isinstance(cref, str):
        out.append({"check": "anticheat.contract", "detail": "the Case references no Contract"})
    elif _hash(c.store.get(cref).body) != _hash(contract.model_dump(mode="json")):
        out.append(
            {"check": "anticheat.contract", "detail": "Contract hash differs from the Case's"}
        )
    bref = rec.refs.get("baseline")
    if bref is not None and not isinstance(bref, str):
        out.append({"check": "anticheat.baseline", "detail": "the Case names several baselines"})
    return out


def _numeric_one(
    c: Checker,
    module: BuiltModule,
    problem: Problem,
    contract: Contract,
    shape: dict[str, Any],
    shown: str | None,
    dists: list[str],
) -> Checked:
    ev = Evidence()
    run_numeric(c, ev, module, problem, contract, shape, shown, dists)
    if shown is None:
        ev.detail.update({"dists": dists, "ctx_seed": c.seed})
    else:
        ev.detail = {"shape_class": shown, "dists": dists}
    return write_check(c, NUMERIC, ev, module.build_id)


def run_full(
    c: Checker,
    module: BuiltModule,
    gk: GeneratedKernel,
    problem: Problem,
    contract: Contract,
    case: Case,
    *,
    case_id: str,
    shape: dict[str, Any],
    build_for: BuildFor | None,
    sass: list[SassExpect] | None = None,
    assignment: dict[str, Any] | None = None,
    extra_dists: list[str] | None = None,
) -> Checked:
    if build_for is None:
        raise NotSupported("full_check needs build_for to compile edge and hidden shapes")
    rebuild = build_for
    dists = dists_for(contract, extra_dists)
    agg = Evidence()
    subs: list[Checked] = []

    def guarded(name: str, fn: Callable[[], Checked]) -> None:
        try:
            subs.append(fn())
        except (CheckerError, ExecutionError) as exc:
            agg.error(name, str(exc))

    guarded("numeric", lambda: _numeric_one(c, module, problem, contract, shape, None, dists))
    guarded("sanitizer", lambda: sanitizer_check(c, module, problem, shape))
    guarded("anticheat", lambda: anticheat_check(c, module, gk, problem, shape))
    if sass:
        guarded("sass", lambda: sass_check(c, module, sass, assignment or {}))
    others = [(e, None) for e in contract.edge_shapes]
    seed = hidden.private_seed(c.private_dir)
    others += [(s, cat) for s, cat in hidden.hidden_shapes(seed, shape)]
    for other, category in others:
        label = "numeric" if category is None else f"numeric.hidden.{category}"

        def one(other: dict[str, Any] = other, category: str | None = category) -> Checked:
            _, m = rebuild(other)
            return _numeric_one(c, m, problem, contract, other, category, dists)

        guarded(label, one)
    agg.failures += identity_failures(c, case, case_id, contract)
    for s in subs:
        agg.failures += [{**f, "sub_check": s.id} for f in s.body.failures]
    first = c.store.get(subs[0].id) if subs else c.store.get(module.build_id)
    agg.refs.update({"env": first.refs["env"], "worker": first.refs["worker"]})
    agg.refs["checks"] = [s.id for s in subs]
    agg.refs["case"] = case_id
    agg.detail = {"sub_checks": {s.id: s.passed for s in subs}, "hidden_shapes": len(others)}
    return write_check(c, NAME, agg, module.build_id)


def full_check(
    ctx: Context,
    module: BuiltModule,
    gk: GeneratedKernel,
    problem: Problem,
    contract: Contract,
    case: Case,
    *,
    case_id: str,
    shape: dict[str, Any],
    build_for: BuildFor | None,
    sass: list[SassExpect] | None = None,
    assignment: dict[str, Any] | None = None,
) -> Checked:
    return run_full(
        Checker.from_ctx(ctx),
        module,
        gk,
        problem,
        contract,
        case,
        case_id=case_id,
        shape=shape,
        build_for=build_for,
        sass=sass,
        assignment=assignment,
    )


@dataclass(frozen=True)
class SpaceItem:
    path_id: str
    assignment: dict[str, Any]
    gk: GeneratedKernel
    module: BuiltModule
    problem: Problem
    contract: Contract
    kind: Literal["path", "sample"]
    case: Case
    case_id: str
    shape: dict[str, Any]
    build_for: BuildFor | None = None
    sass: list[SassExpect] = field(default_factory=lambda: list[SassExpect]())


@dataclass(frozen=True)
class SpaceReport:
    passed: bool
    items: list[dict[str, Any]]
    status_changes: list[str]
    report_id: str


def space_verify(c: Checker, items: Iterable[SpaceItem], subjects: list[str]) -> SpaceReport:
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for it in items:
        if it.kind == "path":
            chk = run_full(
                c,
                it.module,
                it.gk,
                it.problem,
                it.contract,
                it.case,
                case_id=it.case_id,
                shape=it.shape,
                build_for=it.build_for,
                sass=it.sass,
                assignment=it.assignment,
            )
        else:
            chk = _numeric_one(
                c, it.module, it.problem, it.contract, it.shape, None, dists_for(it.contract, None)
            )
        ids.append(chk.id)
        rows.append({"path_id": it.path_id, "kind": it.kind, "check": chk.id, "passed": chk.passed})
    paths = sum(r["kind"] == "path" for r in rows)
    passed = bool(rows) and paths > 0 and all(r["passed"] for r in rows)
    changes: list[str] = []
    if passed:
        reason = f"space verified: {paths} structural paths, {len(rows) - paths} samples passed"
        for s in subjects:
            body = StatusChange(subject=s, to_assurance="tested", reason=reason)
            rec = Record.new(
                "StatusChange", body, actor("space_verifier"), c.clock.now(), refs={"checks": ids}
            )
            changes.append(c.store.put(rec))
    summary = f"{len(rows)} items ({paths} paths): passed={passed}"
    report = Report(
        title="space verification",
        subject="verify_space",
        summary=summary,
        data={"items": rows, "subjects": subjects, "status_changes": changes},
    )
    rid = c.store.put(
        Record.new("Report", report, actor("space_verifier"), c.clock.now(), refs={"checks": ids})
    )
    return SpaceReport(passed, rows, changes, rid)


def verify_space(ctx: Context, items: Iterable[SpaceItem], subjects: list[str]) -> SpaceReport:
    return space_verify(Checker.from_ctx(ctx), items, subjects)
