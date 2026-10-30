"""T5 done-check and output (spec/modules/orchestrator-cli.md: "microbenchmark repeated 5
times, coefficient of variation below twice the noise floor"; D404).

Submission layout: ``microbench.yaml`` (``kta.measure.microbench.Microbench`` fields, plus
``source``: the CUDA file, default ``kernel.cu``) and that file. The check builds the
kernel once and measures it ``REPEATS`` times on the GPU worker under the newest
measurement Protocol. The repeat results are kept in the task bookkeeping (``task:<id>``)
so the accepted output, an HwFact with ``refs.measurements``, is written from the same
Measurements without measuring again.
"""

from __future__ import annotations

from kta.core.registry import resolve
from kta.core.schemas import Actor
from kta.measure.microbench import Microbench, Repeats, repeat, write_repeat_fact
from kta.orchestrator.done_checks import CheckInput, Outcome, read_yaml
from kta.orchestrator.proc_common import get_state, update_state

REPEATS = 5
CV_FACTOR = 2.0


def load(inp: CheckInput) -> Microbench:
    doc = read_yaml(inp.workdir / "microbench.yaml")
    src = inp.workdir / str(doc.get("source", "kernel.cu"))
    return Microbench.from_doc(doc, src.read_text(encoding="utf-8"))


def microbench_repeats(inp: CheckInput) -> Outcome:
    mb = load(inp)
    svc = inp.svc
    protocol_id, _ = svc.protocol()
    noise = float(svc.store.get(protocol_id).body["params"].get("noise_floor_rel", 0.0))
    target_id = str(resolve("target", inp.ctx).latest_target_id())
    rep = repeat(
        svc.measurer(), svc.backend("cuda"), svc.target(), target_id, mb, protocol_id, REPEATS
    )
    limit = CV_FACTOR * noise
    ok = noise > 0 and rep.cv < limit
    data = {
        "cv": rep.cv,
        "limit": limit,
        "values": rep.values,
        "measurements": rep.measurements,
        "case": rep.case,
        "protocol": protocol_id,
        "target": target_id,
    }
    update_state(svc.ops, f"task:{inp.task_id}", microbench=data)
    failures = [] if ok else [f"cv {rep.cv:.4f} not below {CV_FACTOR} x noise floor {noise:.4f}"]
    return Outcome("microbench_repeats", ok, failures, data)


def write_microbench(inp: CheckInput, actor: Actor) -> dict[str, str]:
    data = get_state(inp.svc.ops, f"task:{inp.task_id}")["microbench"]
    rep = Repeats(list(data["measurements"]), list(data["values"]), float(data["cv"]), data["case"])
    fid = write_repeat_fact(
        inp.svc.measurer(), load(inp), rep, str(data["target"]), str(data["protocol"]), actor
    )
    return {"hw_fact": fid}
