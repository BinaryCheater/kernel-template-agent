"""What an accepted task writes into the pool (D269).

Outputs are written only after every done-check passed, by the submitter's actor with
``actor.task`` set, so lineage leads from each new object to its task. A ``task_result``
Report lists them; loops read it to resume.
"""

from __future__ import annotations

from typing import Any

import yaml

from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import Family
from kta.knowledge import load_family, propose_template
from kta.knowledge.admission import report_body as admission_report
from kta.knowledge.pool import find_template
from kta.models.calibrate import save_model
from kta.models.unit_model import UnitModel
from kta.orchestrator.done_checks import (
    CheckInput,
    new_patterns,
    t2_admission,
    t2_documents,
    t3_documents,
    t3_pool,
)

RESULT_TITLE = "task_result"


def _put(inp: CheckInput, actor: Actor, kind: str, body: Any, **kw: Any) -> str:
    return inp.store.put(Record.new(kind, body, actor, inp.ctx.clock.now(), **kw))


def _family(inp: CheckInput, actor: Actor) -> dict[str, Any]:
    """Chain and Family records; the package files become the Family's blobs."""
    fp = load_family(inp.workdir)
    chain_id = _put(inp, actor, "Chain", fp.chain)
    blobs = {f"files/{name}": inp.submission.blobs[name] for name in inp.submission.body["files"]}
    refs: dict[str, Any] = {"chain": chain_id}
    previous = [
        i for i in inp.card.inputs if inp.store.exists(i) and inp.store.get(i).kind == "Family"
    ]
    if previous:
        refs["previous"] = previous[0]
    fam_id = _put(
        inp, actor, "Family", Family.model_validate(fp.family.model_dump()), refs=refs, blobs=blobs
    )
    return {"family": fam_id, "chain": chain_id}


def _reconstruction(inp: CheckInput, actor: Actor) -> dict[str, Any]:
    docs = t2_documents(inp.workdir)
    reference = inp.card.inputs[0]
    pattern_ids = [_put(inp, actor, "Pattern", p) for p in new_patterns(inp.store, docs.patterns)]
    known = [find_template(inp.store, i.template) for i in docs.chain.items]
    chain_refs: dict[str, Any] = {}
    if any(known):
        chain_refs["templates"] = sorted({k[0] for k in known if k is not None})
    chain_id = _put(inp, actor, "Chain", docs.chain, refs=chain_refs)
    gap_ids = [_put(inp, actor, "Gap", g, refs={"reference": reference}) for g in docs.gaps]
    refs: dict[str, Any] = {"reference": reference, "chain": chain_id}
    if gap_ids:
        refs["gaps"] = gap_ids
    recon = _put(inp, actor, "Reconstruction", docs.reconstruction, refs=refs)
    checker = Actor(type="checker", name="kta.knowledge.admission", task=inp.task_id)
    body = admission_report(t2_admission(inp), recon, f"T2 {inp.task_id}")
    _put(inp, checker, "Report", body, refs={"reference": reference, "card": inp.card_id})
    gaps = [{"id": g, "missing": inp.store.get(g).body["missing"]} for g in gap_ids]
    out: dict[str, Any] = {"reconstruction": recon, "chain": chain_id, "gaps": gaps}
    return out | {"patterns": pattern_ids} if pattern_ids else out


def _template(inp: CheckInput, actor: Actor) -> dict[str, Any]:
    t, patterns, doc = t3_documents(inp)
    view = t3_pool(inp, t, patterns)  # before the Patterns are written
    pattern_ids = [_put(inp, actor, "Pattern", p) for p in new_patterns(inp.store, patterns)]
    sources = [str(s) for s in doc["sources"]]
    tid = propose_template(inp.ctx, inp.store, t, sources, actor, validate_in=view)
    return {"template": tid, "patterns": pattern_ids} if pattern_ids else {"template": tid}


def _model(inp: CheckInput, actor: Actor) -> dict[str, Any]:
    text = (inp.workdir / "model.json").read_text(encoding="utf-8")
    model = UnitModel.from_dict(yaml.safe_load(text))
    return {"model": save_model(inp.ctx, inp.store, model, actor, previous=inp.card.inputs[0])}


def _microbench(inp: CheckInput, actor: Actor) -> dict[str, Any]:
    from kta.orchestrator.microbench_check import write_microbench  # WP11; import cycle

    return dict(write_microbench(inp, actor))


WRITERS = {
    "T1": _family,
    "T6": _family,
    "T2": _reconstruction,
    "T3": _template,
    "T4": _model,
    "T5": _microbench,
}


def write_outputs(inp: CheckInput, actor: Actor) -> str:
    """Write the accepted outputs and the task_result Report; returns the Report ID."""
    who = actor.model_copy(update={"task": inp.task_id})
    writer = WRITERS.get(inp.card.type)
    outputs = writer(inp, who) if writer else {}
    ids = [v for v in outputs.values() if isinstance(v, str)]
    ids += [g["id"] for g in outputs.get("gaps", [])]
    ids += list(outputs.get("patterns", []))
    refs: dict[str, Any] = {"card": inp.card_id, "submission": inp.submission_id}
    if ids:
        refs["outputs"] = ids
    body = {
        "title": RESULT_TITLE,
        "subject": inp.card_id,
        "summary": f"{inp.card.type} accepted: {sorted(outputs)}",
        "data": {"task": inp.task_id, "outputs": outputs},
    }
    system = Actor(type="system", name="kta.orchestrator.tasks", task=inp.task_id)
    return _put(inp, system, "Report", body, refs=refs)
