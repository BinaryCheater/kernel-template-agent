"""full_check and verify_space (spec/modules/measure-check.md, docs/design/50 §4).

A module is compiled for one shape (its LaunchSpec is fixed), so checking other shapes
needs a new build. Callers pass ``build_for(shape) -> (GeneratedKernel, BuiltModule)``;
generation and compilation stay with the orchestrator, and checks never imports
knowledge (decision D227). ``full_check`` runs, for the module's own shape, numeric,
sanitizer, anti-cheat and SASS checks; numeric checks for every Contract edge shape and
every hidden shape; and the Contract identity check against the Case record. It writes
one Check per sub-check and an aggregate Check referencing them.

SYSDEBUG-2 (D694, D696): edge and hidden shapes outside the Contract's ``shape_scope`` are
skipped and listed; ``quick_check`` is the cheap gate before measuring (numeric on the
module's own shape with one input distribution at the loose level, anti-cheat, SASS and
identity; no extra builds), and the full check (strict level) is for a round's best.
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
QUICK = "quick_checker"
QUICK_DISTS = ("normal",)
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


def in_scope(
    shape: dict[str, Any], scope: dict[str, Any], case_shapes: list[dict[str, Any]]
) -> bool:
    """Whether ``shape`` is one the Contract holds the kernel accountable for (D696)."""
    kind = scope.get("kind")
    if not kind:
        return True
    dims = {k: v for k, v in shape.items() if isinstance(v, int) and not isinstance(v, bool)}
    if kind == "static":
        return any(all(s.get(k) == v for k, v in dims.items()) for s in case_shapes)
    if kind == "range":
        ranges: dict[str, Any] = scope.get("ranges") or {}
        return all(int(r[0]) <= dims[k] <= int(r[1]) for k, r in ranges.items() if k in dims)
    raise NotSupported(f"shape_scope kind {kind!r} (expected static or range)")


def case_shapes(case: Case) -> list[dict[str, Any]]:
    return [dict(s.get("shape", {})) for s in case.shape_dist]


def _numeric_one(
    c: Checker,
    module: BuiltModule,
    problem: Problem,
    contract: Contract,
    shape: dict[str, Any],
    shown: str | None,
    dists: list[str],
    level: str | None = None,
) -> Checked:
    ev = Evidence()
    run_numeric(c, ev, module, problem, contract, shape, shown, dists, level)
    if shown is None:
        ev.detail.update({"dists": dists, "ctx_seed": c.seed, "level": level})
    else:
        ev.detail = {"shape_class": shown, "dists": dists, "level": level}
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
    sanitizer_shape: dict[str, Any] | None = None,
    level: str | None = None,
) -> Checked:
    """``sanitizer_shape``: run compute-sanitizer on a module built for this (small) shape
    instead of the module's own shape (docs/design/50 §3, decision D285); None keeps the
    module's shape. ``level``: library-relative acceptance level (D695), None = fixed."""
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

    guarded(
        "numeric", lambda: _numeric_one(c, module, problem, contract, shape, None, dists, level)
    )
    if sanitizer_shape is None:
        guarded("sanitizer", lambda: sanitizer_check(c, module, problem, shape))
    else:
        small = sanitizer_shape
        guarded("sanitizer", lambda: sanitizer_check(c, rebuild(small)[1], problem, small))
    guarded("anticheat", lambda: anticheat_check(c, module, gk, problem, shape))
    if sass:
        guarded("sass", lambda: sass_check(c, module, sass, assignment or {}))
    others: list[tuple[dict[str, Any], str | None]] = [(e, None) for e in contract.edge_shapes]
    seed = hidden.private_seed(c.private_dir)
    others += [(s, cat) for s, cat in hidden.hidden_shapes(seed, shape)]
    scope, own = contract.shape_scope, case_shapes(case)
    skipped = [cat or "edge" for s, cat in others if not in_scope(s, scope, own)]
    others = [(s, cat) for s, cat in others if in_scope(s, scope, own)]
    for other, category in others:
        label = "numeric" if category is None else f"numeric.hidden.{category}"

        def one(other: dict[str, Any] = other, category: str | None = category) -> Checked:
            _, m = rebuild(other)
            return _numeric_one(c, m, problem, contract, other, category, dists, level)

        guarded(label, one)
    agg.failures += identity_failures(c, case, case_id, contract)
    for s in subs:
        agg.failures += [{**f, "sub_check": s.id} for f in s.body.failures]
    first = c.store.get(subs[0].id) if subs else c.store.get(module.build_id)
    agg.refs.update({"env": first.refs["env"], "worker": first.refs["worker"]})
    agg.refs["checks"] = [s.id for s in subs]
    agg.refs["case"] = case_id
    agg.detail = {
        "sub_checks": {s.id: s.passed for s in subs},
        "hidden_shapes": len(others),
        "out_of_scope": skipped,
        "shape_scope": scope,
        "level": level,
    }
    return write_check(c, NAME, agg, module.build_id)


def run_quick(
    c: Checker,
    module: BuiltModule,
    gk: GeneratedKernel,
    problem: Problem,
    contract: Contract,
    case: Case,
    *,
    case_id: str,
    shape: dict[str, Any],
    sass: list[SassExpect] | None = None,
    assignment: dict[str, Any] | None = None,
    level: str = "loose",
) -> Checked:
    """The gate before measuring (D694): cheap, on the module as built, no rebuilds."""
    wanted = [d for d in QUICK_DISTS if d in contract.input_dists]
    dists = wanted or dists_for(contract, None)[:1]
    agg = Evidence()
    subs: list[Checked] = []

    def guarded(name: str, fn: Callable[[], Checked]) -> None:
        try:
            subs.append(fn())
        except (CheckerError, ExecutionError) as exc:
            agg.error(name, str(exc))

    guarded(
        "numeric", lambda: _numeric_one(c, module, problem, contract, shape, None, dists, level)
    )
    guarded("anticheat", lambda: anticheat_check(c, module, gk, problem, shape))
    if sass:
        guarded("sass", lambda: sass_check(c, module, sass, assignment or {}))
    agg.failures += identity_failures(c, case, case_id, contract)
    for s in subs:
        agg.failures += [{**f, "sub_check": s.id} for f in s.body.failures]
    first = c.store.get(subs[0].id) if subs else c.store.get(module.build_id)
    agg.refs.update({"env": first.refs["env"], "worker": first.refs["worker"]})
    agg.refs["checks"] = [s.id for s in subs]
    agg.refs["case"] = case_id
    agg.detail = {"sub_checks": {s.id: s.passed for s in subs}, "dists": dists, "level": level}
    return write_check(c, QUICK, agg, module.build_id)


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
    extra_dists: list[str] | None = None,
    sanitizer_shape: dict[str, Any] | None = None,
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
        extra_dists=extra_dists,
        sanitizer_shape=sanitizer_shape,
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


REASON_KEYS = ("check", "dist", "rel_l2", "rel_l2_limit", "max_abs_err", "max_abs_limit",
               "mismatches", "elements", "reason", "error")  # fmt: skip


def failure_reason(failures: list[dict[str, Any]]) -> str | None:
    """The first failure of a check in one line (SYSDEBUG-2): what failed, with its numbers."""
    if not failures:
        return None
    f = failures[0]
    parts = [
        f"{k} {f[k]:.3g}" if isinstance(f[k], float) else f"{k} {f[k]}"
        for k in REASON_KEYS
        if f.get(k) is not None
    ]
    more = f" (+{len(failures) - 1} more)" if len(failures) > 1 else ""
    return (", ".join(parts) or str(f)[:200]) + more


def space_verify(c: Checker, items: Iterable[SpaceItem], subjects: list[str]) -> SpaceReport:
    """Every structural path gets the full check and every sample a numeric check, both at
    the loose level (SYSDEBUG-2 D711: obvious errors only, relative to the library, as the
    solve loop's quick check), so a delivery is not refused for fp16 rounding."""
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
                level="loose",
            )
        else:
            chk = _numeric_one(
                c,
                it.module,
                it.problem,
                it.contract,
                it.shape,
                None,
                dists_for(it.contract, None),
                "loose",
            )
        ids.append(chk.id)
        rows.append({"path_id": it.path_id, "kind": it.kind, "check": chk.id, "passed": chk.passed,
                     "reason": failure_reason(list(chk.body.failures))})  # fmt: skip
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
