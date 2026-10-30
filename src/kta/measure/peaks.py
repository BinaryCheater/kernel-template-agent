"""Measured per-unit peaks and the HwModel record (docs/design/50 §6, decision D283).

``measure_peaks`` builds each microbenchmark of ``kta.measure.peak_kernels`` with the CUDA
backend, times it under a measurement Protocol (one ``measure`` job each, on the GPU
worker) and writes, all with origin ``real``:

- one timing-only Problem, Contract and Case per kernel (tool actor, like the selftests);
- one HwFact per unit: ``quantity = "peak_<unit>"``, method ``measured``, value = work /
  median time, CI from the time CI, refs to the Measurement and the Target;
- one HwModel: those peaks, plus ``limits`` = the Target's ``hw.*`` facts and the model
  assumptions below; refs to the facts and measurements.

``MODEL_ASSUMPTIONS`` are ``hw.*`` values the seed Templates read that no device query or
documentation gives; each is stated with its reason in the HwModel's ``assumptions`` blob.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from kta.backends.base import Backend, BuiltModule, GeneratedKernel, LaunchSpec
from kta.core.context import Clock
from kta.core.errors import ExecutionError
from kta.core.ids import record_id
from kta.core.jobs import JobRunner
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Case, Contract, Problem, TensorSpec
from kta.core.schemas.hardware import HwFact, HwModel, Target
from kta.measure.peak_kernels import PeakKernel, kernels
from kta.measure.runner import Measurer
from kta.store.base import Store

log = logging.getLogger("kta.measure.peaks")
ACTOR = Actor(type="tool", name="kta measure peaks")
TIMING_ONLY = "none: timing-only microbenchmark of one hardware unit's peak"
MODEL_ASSUMPTIONS: dict[str, tuple[int, str]] = {
    "smem_queue_depth": (
        1,
        "shared_resource_binding@1 fixes the smem request queue depth as a chip variable; "
        "GB10 documents no value, so the model uses 1 (one outstanding request per bank pass)",
    ),
    "smem_banks": (32, "CUDA C++ Programming Guide: shared memory has 32 banks"),
    "smem_bank_bytes": (4, "CUDA C++ Programming Guide: successive 32-bit words map to banks"),
}


@dataclass(frozen=True)
class PeakResult:
    hwmodel_id: str
    peaks: dict[str, float]
    facts: dict[str, str]  # unit -> HwFact ID
    measurements: dict[str, str]  # kernel unit -> Measurement ID


def _ensure(
    store: Store, clock: Clock, kind: str, body: dict[str, Any], refs: dict[str, Any]
) -> str:
    for rec in store.query(kind, {"actor.name": ACTOR.name}, order="created_at"):
        if rec.body == body and rec.refs == refs:
            return record_id(rec)
    return store.put(Record.new(kind, body, ACTOR, clock.now(), refs=refs))


def _tensors(names: list[tuple[str, str]]) -> list[TensorSpec]:
    return [TensorSpec(name=n, shape=[d], dtype="f32", layout="contiguous") for n, d in names]


def workload(store: Store, clock: Clock, k: PeakKernel, target_id: str) -> tuple[str, Case]:
    """Problem, Contract and Case records of one microbenchmark (reused when present)."""
    problem = Problem(
        name=f"peak_{k.unit}",
        reference_impl=TIMING_ONLY,
        inputs=_tensors(k.inputs),
        outputs=_tensors(k.outputs),
        params={"meaning": k.meaning},
    )
    pid = _ensure(store, clock, "Problem", problem.model_dump(mode="json"), {})
    contract = Contract(tolerances={}, input_dists=[], timing_boundary="single kernel launch")
    cid = _ensure(store, clock, "Contract", contract.model_dump(mode="json"), {"problem": pid})
    case = Case(
        problem=pid,
        shape_dist=[{"shape": k.shape, "weight": 1.0}],
        objective="weighted_latency",
        target=target_id,
    )
    case_id = _ensure(
        store, clock, "Case", case.model_dump(mode="json"), {"problem": pid, "contract": cid}
    )
    return case_id, case


def generated(k: PeakKernel) -> GeneratedKernel:
    launch = LaunchSpec([k.grid], [k.block], 0, k.params)
    return GeneratedKernel("cuda", {"kernel.cu": k.source}, k.entry, launch)


def build(backend: Backend, k: PeakKernel, target: Target, ex: JobRunner) -> BuiltModule:
    result, module = backend.build(generated(k), target, ex)
    if not result.ok:
        raise ExecutionError("nonzero", f"peak kernel {k.entry}: {result.log_summary[-800:]}")
    return module


def _fact(
    store: Store, clock: Clock, unit: str, work: float, m: Any, refs: dict[str, Any]
) -> tuple[float, str]:
    seconds = m.body.median_us * 1e-6
    value = work / seconds
    fact = HwFact(
        unit=unit,  # type: ignore[arg-type]  # validated by the model against Unit
        quantity=f"peak_{unit}",
        value=value,
        ci_low=work / (m.body.ci_high_us * 1e-6),
        ci_high=work / (m.body.ci_low_us * 1e-6),
        method="measured",
    )
    rid = store.put(
        Record.new("HwFact", fact, ACTOR, clock.now(), refs={**refs, "measurement": m.id})
    )
    log.info("peak %s: %.4g per s (median %.3f us)", unit, value, m.body.median_us)
    return value, rid


def next_version(store: Store, target: str) -> int:
    found = store.query("HwModel", {"target": target}, order="-created_at", limit=200)
    return 1 + max((int(r.body["version"]) for r in found), default=0)


def measure_peaks(
    m: Measurer,
    backend: Backend,
    target: Target,
    target_id: str,
    limits: dict[str, int | float],
    protocol_id: str,
    sm_count: int,
) -> PeakResult:
    """Measure every unit's peak and write the HwFacts and the HwModel record."""
    peaks: dict[str, float] = {}
    facts: dict[str, str] = {}
    measurements: dict[str, str] = {}
    for k in kernels(sm_count):
        case_id, case = workload(m.store, m.clock, k, target_id)
        module = build(backend, k, target, m.ex)
        meas = m.measure(module, case, k.shape, protocol_id)
        measurements[k.unit] = meas.id
        refs = {"target": target_id, "case": case_id, "protocol": protocol_id}
        for unit, work in sorted(k.work(k.shape).items()):
            peaks[unit], facts[unit] = _fact(m.store, m.clock, unit, work, meas, refs)
    assumptions = {k: {"value": v, "reason": why} for k, (v, why) in MODEL_ASSUMPTIONS.items()}
    body = HwModel(
        target=target.name,
        version=next_version(m.store, target.name),
        peaks=peaks,  # type: ignore[arg-type]  # keys are Unit names, validated by the model
        limits={**limits, **{k: v for k, (v, _) in MODEL_ASSUMPTIONS.items()}},
    )
    blob = m.ex.put_blob(json.dumps(assumptions, sort_keys=True).encode(), "application/json")
    refs: dict[str, str | list[str]] = {
        "target": target_id,
        "facts": sorted(facts.values()),
        "measurements": sorted(measurements.values()),
        "protocol": protocol_id,
    }
    rec = Record.new("HwModel", body, ACTOR, m.clock.now(), refs=refs, blobs={"assumptions": blob})
    hid = m.store.put(rec)
    log.info("HwModel %s v%d: %s", target.name, body.version, hid[:12])
    return PeakResult(hid, peaks, facts, measurements)


def load_hwmodel(store: Store, hwmodel_id: str) -> HwModel:
    return HwModel.model_validate(store.get(hwmodel_id).body)
