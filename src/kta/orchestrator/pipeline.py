"""Shared services of a control-plane process and the generate -> build path.

One ``Services`` per process holds one JobExecutor (D184), created on first use, and
the Checker/Measurer built over it. Family packages are materialized from the Family
record's blobs (``files/<path>``, written when a T1/T6 is accepted) or, for seed
families that were never submitted, read from ``<seed_dir>/families/<name>`` (D272).
"""

from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from kta.backends.base import Backend, BuiltModule, GeneratedKernel, LaunchSpec
from kta.checks.base import Checker
from kta.core.context import Context
from kta.core.errors import NotSupported, SchemaError
from kta.core.expr import Expr
from kta.core.ids import record_id
from kta.core.registry import resolve
from kta.core.schemas import Actor
from kta.core.schemas.evaluation import Build, Case, Contract, Problem
from kta.core.schemas.hardware import HwModel, Target
from kta.core.schemas.knowledge import SassExpect, Template
from kta.execution import JobExecutor
from kta.knowledge import FamilyPackage, load_family, seed_dir
from kta.knowledge.pool import find_template
from kta.knowledge.realisations import expected_sass
from kta.measure.runner import Measurer
from kta.models import MODEL_SUBJECT, UnitModel, load_model, save_model
from kta.store.base import Store
from kta.store.sqlite_ops import SqliteOps

ACTOR = Actor(type="system", name="kta.orchestrator")


@dataclass
class Services:
    ctx: Context
    store: Store
    ops: SqliteOps
    _executor: JobExecutor | None = field(default=None, init=False)

    @classmethod
    def open(cls, ctx: Context) -> Services:
        return cls(ctx, resolve("store", ctx), resolve("ops", ctx))

    @property
    def executor(self) -> JobExecutor:
        if self._executor is None:
            self._executor = resolve("executor", self.ctx)
        ex: JobExecutor = self._executor  # pyright: ignore[reportAssignmentType]
        return ex

    def checker(self) -> Checker:
        c = self.ctx
        return Checker(self.store, self.executor, c.clock, c.rng, c.data_dir / "checker", c.seed)

    def measurer(self) -> Measurer:
        c = self.ctx
        return Measurer(self.store, self.executor, c.clock, c.rng, c.seed)

    def backend(self, name: str) -> Backend:
        return resolve(f"backend.{name}", self.ctx)

    def target(self) -> Target:
        plugin = resolve("target", self.ctx)
        return Target.model_validate(self.store.get(plugin.latest_target_id()).body)

    def hw_model(self, target: str) -> tuple[str, HwModel]:
        found = self.store.query("HwModel", {"target": target}, order="-created_at", limit=1)
        if not found:
            raise NotSupported(f"no HwModel record for target {target} (peaks are measured)")
        rec = found[0]
        return record_id(rec), HwModel.model_validate(rec.body)

    def unit_model(self, target: str) -> tuple[str, UnitModel]:
        """The newest model version; an uncalibrated default is saved when none exists."""
        found = self.store.query(
            "Report", {"subject": MODEL_SUBJECT}, order="-created_at", limit=50
        )
        for rec in found:
            if rec.body["data"].get("target") == target:
                rid = record_id(rec)
                return rid, load_model(self.store, rid)
        model = UnitModel(target=target)
        return save_model(self.ctx, self.store, model, ACTOR), model

    def protocol(self) -> tuple[str, float]:
        """The newest measurement Protocol and its significance threshold (3 x noise floor)."""
        found = self.store.query("Protocol", {"type": "measurement"}, order="-created_at", limit=50)
        if not found:
            raise NotSupported("no measurement Protocol record (kta measure protocol)")
        rec = max(found, key=lambda r: int(r.body["version"]))
        noise = float(rec.body["params"].get("noise_floor_rel", 0.0))
        return record_id(rec), 3.0 * noise


@dataclass(frozen=True)
class CaseView:
    id: str
    case: Case
    problem: Problem
    contract: Contract
    contract_id: str
    baseline: str | None


def case_view(store: Store, case_id: str) -> CaseView:
    rec = store.get(case_id)
    if rec.kind != "Case":
        raise SchemaError(f"{case_id} is a {rec.kind}, not a Case")
    case = Case.model_validate(rec.body)
    contract_id = rec.refs.get("contract")
    if not isinstance(contract_id, str):
        raise SchemaError(f"Case {case_id[:12]} references no Contract")
    baseline = rec.refs.get("baseline")
    return CaseView(
        case_id,
        case,
        Problem.model_validate(store.get(case.problem).body),
        Contract.model_validate(store.get(contract_id).body),
        contract_id,
        baseline if isinstance(baseline, str) else None,
    )


def family_package(svc: Services, family_id: str) -> FamilyPackage:
    rec = svc.store.get(family_id)
    if rec.kind != "Family":
        raise SchemaError(f"{family_id} is a {rec.kind}, not a Family")
    files = {k.removeprefix("files/"): v for k, v in rec.blobs.items() if k.startswith("files/")}
    if files:
        root = svc.ctx.data_dir / "families" / family_id[:16]
        if root.exists():
            shutil.rmtree(root)
        for name, sha in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(svc.store.blob_path(sha), root / name)
        return load_family(root)
    path = seed_dir(svc.ctx) / "families" / str(rec.body["name"])
    if not path.is_dir():
        raise NotSupported(f"Family {family_id[:12]} has no files and no seed directory {path}")
    return load_family(path)


def templates_of(store: Store, fp: FamilyPackage) -> dict[str, Template]:
    out: dict[str, Template] = {}
    for item in fp.chain.items:
        hit = find_template(store, item.template)
        if hit is None:
            raise SchemaError(f"family {fp.family.name}: Template {item.template} not in the store")
        out[item.template] = hit[1]
    return out


def chain_sass(store: Store, fp: FamilyPackage, assignment: Mapping[str, Any]) -> list[SassExpect]:
    """``sass_for`` with the chain's Templates and their Realisations' expectations."""
    refs = [item.template for item in fp.chain.items]
    expectations = expected_sass(store, refs, fp.family.backend)
    return sass_for(fp, templates_of(store, fp), assignment, expectations)


def _generator(fp: FamilyPackage) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location(f"kta_family_{fp.family.name}", fp.gen_path)
    if spec is None or spec.loader is None:
        raise SchemaError(f"cannot load {fp.gen_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, fp.gen_function, None)
    if not callable(func):
        raise SchemaError(f"{fp.gen_path} has no function {fp.gen_function}")
    return cast(Callable[..., Any], func)


def generate(
    fp: FamilyPackage, assignment: Mapping[str, Any], shape: Mapping[str, Any]
) -> GeneratedKernel:
    """Run the family generator in this process (D213): it only produces source text."""
    out = _generator(fp)(dict(assignment), dict(shape))
    if isinstance(out, GeneratedKernel):
        return out
    if not isinstance(out, Mapping):
        raise SchemaError(f"{fp.family.name}: generate() returned {type(out).__name__}")
    g = cast(Mapping[str, Any], out)
    if "launch" not in g or "entry" not in g:
        raise SchemaError(f"{fp.family.name}: generate() must return files, entry and launch")
    return GeneratedKernel(
        backend=str(g.get("backend", fp.family.backend)),
        files={str(k): str(v) for k, v in g["files"].items()},
        entry=str(g["entry"]),
        launch=LaunchSpec.from_dict(dict(g["launch"])),
        markers=[str(m) for m in g.get("markers", [])],
    )


def build(svc: Services, gk: GeneratedKernel) -> tuple[Build, BuiltModule]:
    return svc.backend(gk.backend).build(gk, svc.target(), svc.executor)


def builder(
    svc: Services, fp: FamilyPackage, assignment: Mapping[str, Any]
) -> Callable[[dict[str, Any]], tuple[GeneratedKernel, BuiltModule]]:
    """``build_for`` of checks.full_check: the same assignment at another shape."""

    def build_for(shape: dict[str, Any]) -> tuple[GeneratedKernel, BuiltModule]:
        gk = generate(fp, assignment, shape)
        return gk, build(svc, gk)[1]

    return build_for


def sass_for(
    fp: FamilyPackage,
    templates: Mapping[str, Template],
    assignment: Mapping[str, Any],
    expectations: Mapping[str, list[SassExpect]],
) -> list[SassExpect]:
    """The chain's SASS expectations whose ``when`` holds (D361).

    ``expectations`` maps each Template ref to the SASS expectations of its Realisations
    on the family's backend (D443). ``when`` names Template variables, so it is evaluated
    per chain item over that item's bindings: a binding naming a family variable takes
    the assignment's value, a constant binding stays. The kept expectations are passed
    on with ``when = "True"``.
    """
    out: list[SassExpect] = []
    for item in fp.chain.items:
        t = templates[item.template]
        env: dict[str, Any] = {
            v.name: assignment.get(str(item.bindings[v.name]), item.bindings[v.name])
            for v in t.variables
            if v.name in item.bindings
        }
        for e in expectations.get(item.template, []):
            if Expr.parse(e.when, set(env)).eval(env) is True:
                out.append(
                    SassExpect(
                        when="True",
                        must_contain=e.must_contain,
                        must_not_contain=e.must_not_contain,
                    )
                )
    return out


def sanitizer_shape(cv: CaseView) -> dict[str, Any] | None:
    """The small shape compute-sanitizer runs on (D285, D360): the Contract's first edge
    shape with every dimension above 1 (usually the non-tile-multiple one); None keeps
    the checked shape."""
    for shape in cv.contract.edge_shapes:
        if shape and all(isinstance(v, int) and v > 1 for v in shape.values()):
            return dict(shape)
    return None


def bucket_shape(bucket: Mapping[str, Any]) -> dict[str, Any]:
    return dict(bucket.get("shape", bucket))


def files_of(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
