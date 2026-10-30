"""What an accepted task writes into the pool (D269).

Outputs are written only after every done-check passed, by the submitter's actor with
``actor.task`` set, so lineage leads from each new object to its task. A ``task_result``
Report lists them; loops read it to resume.
"""

from __future__ import annotations

from typing import Any

import yaml

from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import Chain, Family, Gap, Reconstruction, Template
from kta.knowledge import load_family, propose_template
from kta.models.calibrate import save_model
from kta.models.unit_model import UnitModel
from kta.orchestrator.done_checks import CheckInput, read_yaml

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
    doc = read_yaml(inp.workdir / "reconstruction.yaml")
    reference = inp.card.inputs[0]
    chain_id = _put(inp, actor, "Chain", Chain.model_validate(doc["chain"]))
    gap_ids = [
        _put(inp, actor, "Gap", Gap.model_validate(g), refs={"reference": reference})
        for g in doc.get("gaps", [])
    ]
    refs: dict[str, Any] = {"reference": reference, "chain": chain_id}
    if gap_ids:
        refs["gaps"] = gap_ids
    recon = _put(
        inp,
        actor,
        "Reconstruction",
        Reconstruction.model_validate(doc["reconstruction"]),
        refs=refs,
    )
    gaps = [{"id": g, "missing": inp.store.get(g).body["missing"]} for g in gap_ids]
    return {"reconstruction": recon, "chain": chain_id, "gaps": gaps}


def _template(inp: CheckInput, actor: Actor) -> dict[str, Any]:
    doc = read_yaml(inp.workdir / "template.yaml")
    t = Template.model_validate(doc["template"])
    tid = propose_template(inp.ctx, inp.store, t, [str(s) for s in doc["sources"]], actor)
    return {"template": tid}


def _model(inp: CheckInput, actor: Actor) -> dict[str, Any]:
    text = (inp.workdir / "model.json").read_text(encoding="utf-8")
    model = UnitModel.from_dict(yaml.safe_load(text))
    return {"model": save_model(inp.ctx, inp.store, model, actor, previous=inp.card.inputs[0])}


WRITERS = {"T1": _family, "T6": _family, "T2": _reconstruction, "T3": _template, "T4": _model}


def write_outputs(inp: CheckInput, actor: Actor) -> str:
    """Write the accepted outputs and the task_result Report; returns the Report ID."""
    who = actor.model_copy(update={"task": inp.task_id})
    writer = WRITERS.get(inp.card.type)
    outputs = writer(inp, who) if writer else {}
    ids = [v for v in outputs.values() if isinstance(v, str)]
    ids += [g["id"] for g in outputs.get("gaps", [])]
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
