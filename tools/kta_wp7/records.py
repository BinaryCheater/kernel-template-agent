"""WP7 records: Protocol v2, Case (Problem, Contract, baseline), Family and Chain, HwModel.

Every record is written by the tool actor ``kta wp7`` and reused when an identical one
exists, so re-running a step does not duplicate its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kta.core.config import repo_root
from kta.core.context import Clock, Context
from kta.core.errors import ConfigError, NotSupported
from kta.core.ids import record_id
from kta.core.registry import resolve
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Case, Contract, Problem
from kta.core.schemas.hardware import HwModel, Target
from kta.knowledge import FamilyPackage, load_family
from kta.knowledge.pool import find_template
from kta.measure import protocol
from kta.measure.baseline import ensure_baseline
from kta.store.base import Store

ACTOR = Actor(type="tool", name="kta wp7")
SEED = repo_root() / "knowledge" / "seed"
FAMILY_DIR = SEED / "families" / "gemm_fp16_tilelang"
CASE_FP16 = SEED / "cases" / "gemm_fp16.yaml"
CASE_BF16 = SEED / "cases" / "gemm_bf16.yaml"
FAMILY_FILES = ("family.yaml", "chain.yaml", "gen.py", "edge_cases.yaml", "README.md")
# WP3 measured a 0.4% noise floor on spark (deliveries/WP3.md, Report 1794b0b1b4b6 in the
# WP3 dev store); each WP's dev store is separate, so v2 is written here with that value.
V2_NOISE_FLOOR = 0.004
V2_DESCRIPTION = (
    "measure-v1 with the noise floor measured by WP3's noise probe on spark (0.4%: CV of 30 "
    "group medians, vector add 0.31%, FP16 GEMM 1024^3 0.35%, rounded up; deliveries/WP3.md)"
)


def ensure(
    store: Store,
    clock: Clock,
    kind: str,
    body: Any,
    refs: dict[str, Any] | None = None,
    blobs: dict[str, str] | None = None,
) -> str:
    data = body if isinstance(body, dict) else body.model_dump(mode="json")
    for rec in store.query(kind, {"actor.name": ACTOR.name}, order="created_at", limit=1000):
        if rec.body == data and rec.refs == (refs or {}) and rec.blobs == (blobs or {}):
            return record_id(rec)
    return store.put(Record.new(kind, data, ACTOR, clock.now(), refs=refs or {}, blobs=blobs or {}))


def protocol_v2(store: Store, clock: Clock) -> str:
    v1 = protocol.ensure_v1(store, clock)
    params = protocol.revise(protocol.V1, V2_NOISE_FLOOR)
    return protocol.ensure_protocol(store, clock, 2, params, V2_DESCRIPTION, {"previous": v1})


def target(ctx: Context) -> tuple[Target, str]:
    plugin = resolve("target", ctx)
    tid = plugin.latest_target_id()
    store: Store = resolve("store", ctx)
    return Target.model_validate(store.get(tid).body), tid


@dataclass(frozen=True)
class CaseIds:
    problem: str
    contract: str
    case: str
    baseline: str | None
    body: Case
    problem_body: Problem
    contract_body: Contract


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    return data


def write_case(store: Store, clock: Clock, path: Path, target: str, image: str | None) -> CaseIds:
    """Problem, Contract, baseline Report (when ``image`` is given) and Case records.

    ``Case.target`` is the Target's name (``gb10``): HwModel and unit-model records are
    looked up by it (decision D287)."""
    doc = read_yaml(path)
    problem = Problem.model_validate(doc["problem"])
    contract = Contract.model_validate(doc["contract"])
    pid = ensure(store, clock, "Problem", problem)
    cid = ensure(store, clock, "Contract", contract, {"problem": pid})
    refs: dict[str, Any] = {"problem": pid, "contract": cid}
    baseline: str | None = None
    if image is not None and "baseline" in doc:
        b = doc["baseline"]
        baseline = ensure_baseline(store, clock, pid, str(b["call"]), str(b["library"]), image)
        refs["baseline"] = baseline
    case = Case.model_validate({**doc["case"], "problem": pid, "target": target})
    case_id = ensure(store, clock, "Case", case, refs)
    return CaseIds(pid, cid, case_id, baseline, case, problem, contract)


def write_family(
    store: Store, clock: Clock, problem_id: str, path: Path = FAMILY_DIR
) -> tuple[str, str, FamilyPackage]:
    """Chain and Family records; the package files are the Family's ``files/*`` blobs."""
    fp = load_family(path)
    templates = []
    for item in fp.chain.items:
        hit = find_template(store, item.template)
        if hit is None:
            raise ConfigError(f"Template {item.template} is not in the store; run `kta init`")
        templates.append(hit[0])
    chain_id = ensure(store, clock, "Chain", fp.chain, {"templates": sorted(set(templates))})
    blobs = {
        f"files/{n}": store.put_blob((path / n).read_bytes(), "text/plain") for n in FAMILY_FILES
    }
    family_id = ensure(
        store, clock, "Family", fp.family, {"chain": chain_id, "problem": problem_id}, blobs
    )
    return family_id, chain_id, fp


def latest_hwmodel(store: Store, target_name: str) -> tuple[str, HwModel]:
    found = store.query("HwModel", {"target": target_name}, order="-created_at", limit=1)
    if not found:
        raise NotSupported(f"no HwModel for {target_name}; run `kta_wp7 peaks` first")
    return record_id(found[0]), HwModel.model_validate(found[0].body)
