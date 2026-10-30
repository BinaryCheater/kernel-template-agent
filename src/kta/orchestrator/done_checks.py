"""Fixed done-checks per task type (spec/modules/orchestrator-cli.md "完成检查", D277).

``TaskCard.done_checks`` stores the names below; agents cannot change them. A check that
cannot run (NotSupported, CheckerError, ExecutionError) suspends the task instead of
rejecting it (docs/design/10 §7). Submission layouts are fixed per type:

- T1, T6: a family directory at the submission root (family.yaml, chain.yaml, gen.py, ...)
- T2: ``reconstruction.yaml`` with ``chain`` (Chain body), ``reconstruction`` and ``gaps``
- T3: ``template.yaml`` with ``template`` (Template body) and ``sources`` (Source IDs)
- T4: ``model.json``, a UnitModel in ``UnitModel.to_dict`` form
- T5: ``microbench.yaml`` and its CUDA source (``kta.orchestrator.microbench_check``)
- every type: the delivery note is the Submission's ``note``
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from kta.core.context import Context
from kta.core.errors import SchemaError
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Measurement, Prediction
from kta.core.schemas.knowledge import Chain, Gap, Reconstruction, Template
from kta.core.schemas.process import TaskCard
from kta.knowledge import errors_of, load_family, validate_family, validate_template
from kta.knowledge.pool import find_template
from kta.models import load_model
from kta.models.unit_model import UnitModel
from kta.orchestrator.pipeline import Services
from kta.store.base import Store

# "Delivery note complete": what was done, simplifications and fixtures, known issues.
NOTE_SECTIONS = {
    "what was done": re.compile(r"^#+\s*(what was done|done)\b", re.I | re.M),
    "fixtures and simplifications": re.compile(
        r"^#+\s*(fixtures?|simplifications?)\b", re.I | re.M
    ),
    "known issues": re.compile(r"^#+\s*known issues\b", re.I | re.M),
}


@dataclass(frozen=True)
class CheckInput:
    svc: Services
    ctx: Context
    store: Store
    task_id: str
    card_id: str
    card: TaskCard
    submission_id: str
    submission: Record
    workdir: Path  # the submitted files, materialized


@dataclass(frozen=True)
class Outcome:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=lambda: list[str]())
    data: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SchemaError(f"the submission has no {path.name}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SchemaError(f"{path.name} must be a mapping")
    return cast(dict[str, Any], data)


def delivery_note(inp: CheckInput) -> Outcome:
    note = str(inp.submission.body.get("note", ""))
    missing = [name for name, rx in NOTE_SECTIONS.items() if not rx.search(note)]
    return Outcome("delivery_note", not missing, [f"note lacks a '{m}' section" for m in missing])


def _issues(name: str, issues: list[Any]) -> Outcome:
    errors = errors_of(issues)
    return Outcome(
        name,
        not errors,
        [str(i) for i in errors],
        {"warnings": [str(i) for i in issues if i not in errors]},
    )


def family_validates(inp: CheckInput) -> Outcome:
    fp = load_family(inp.workdir)
    return _issues("validate_family", validate_family(fp, inp.store))


def space_verifies(inp: CheckInput) -> Outcome:
    # Late import: space enumerates, generates and builds through the Executor.
    from kta.orchestrator.space import verify_family_space

    report = verify_family_space(inp.svc, inp.workdir, inp.card)
    failures = [f"{i['path_id']}: {i.get('failures')}" for i in report.items if not i["passed"]]
    return Outcome("verify_space", report.passed, failures, {"report": report.report_id})


def failing_cases_pass(inp: CheckInput) -> Outcome:
    from kta.orchestrator.space import rerun_failed_checks

    failures, report_id = rerun_failed_checks(inp.svc, inp.workdir, inp.card)
    return Outcome("failing_cases_pass", not failures, failures, {"report": report_id})


def _reconstruction(inp: CheckInput) -> tuple[Chain, Reconstruction, list[Gap]]:
    doc = read_yaml(inp.workdir / "reconstruction.yaml")
    chain = Chain.model_validate(doc.get("chain"))
    recon = Reconstruction.model_validate(doc.get("reconstruction"))
    gaps = [Gap.model_validate(g) for g in doc.get("gaps", [])]
    return chain, recon, gaps


def reconstruction_complete(inp: CheckInput) -> Outcome:
    try:
        chain, recon, _ = _reconstruction(inp)
    except (SchemaError, ValueError) as exc:  # pydantic ValidationError is a ValueError
        return Outcome("reconstruction_fields", False, [str(exc)])
    failures: list[str] = []
    if recon.total < recon.matched or recon.total <= 0:
        failures.append(f"matched {recon.matched} of total {recon.total} is inconsistent")
    if recon.matched != len(chain.items):
        failures.append(f"matched {recon.matched} but the chain has {len(chain.items)} items")
    if recon.total - recon.matched != len(recon.unmatched_transactions):
        failures.append("total - matched differs from the number of unmatched transactions")
    return Outcome("reconstruction_fields", not failures, failures)


def matched_templates_exist(inp: CheckInput) -> Outcome:
    chain, _, _ = _reconstruction(inp)
    missing = [i.template for i in chain.items if find_template(inp.store, i.template) is None]
    return Outcome(
        "matched_templates_exist",
        not missing,
        [f"chain item uses {t}, which is not in the store" for t in missing],
    )


def gaps_match_unmatched(inp: CheckInput) -> Outcome:
    _, recon, gaps = _reconstruction(inp)
    ok = len(gaps) == len(recon.unmatched_transactions)
    return Outcome(
        "gaps_match_unmatched",
        ok,
        []
        if ok
        else [f"{len(recon.unmatched_transactions)} unmatched transactions but {len(gaps)} Gaps"],
    )


def template_validates(inp: CheckInput) -> Outcome:
    doc = read_yaml(inp.workdir / "template.yaml")
    t = Template.model_validate(doc.get("template"))
    issues = validate_template(t, inp.store)
    out = _issues("validate_template", issues)
    dups = [str(i) for i in issues if i.kind == "possible_duplicate"]
    note = str(inp.submission.body.get("note", "")).lower()
    if dups and "duplicate" not in note:
        out = Outcome(
            out.name,
            False,
            [*out.failures, "possible_duplicate not explained in the delivery note"],
            out.data,
        )
    if not doc.get("sources"):
        out = Outcome(out.name, False, [*out.failures, "template.yaml lists no sources"], out.data)
    return out


def _pairs(store: Store, ids: list[str]) -> list[tuple[Prediction, Measurement]]:
    recs = [store.get(i) for i in ids]
    preds = [Prediction.model_validate(r.body) for r in recs if r.kind == "Prediction"]
    meas = [Measurement.model_validate(r.body) for r in recs if r.kind == "Measurement"]
    if not preds or len(preds) != len(meas):
        raise SchemaError("T4 inputs need (Prediction, Measurement) pairs after the model")
    return list(zip(preds, meas, strict=True))


def _median_error(model: UnitModel, pairs: list[tuple[Prediction, Measurement]]) -> float:
    errs = sorted(
        abs(model.total_us(p.per_unit, p.serial) - m.median_us) / m.median_us for p, m in pairs
    )
    return errs[len(errs) // 2]


def model_improves(inp: CheckInput) -> Outcome:
    """The new model's median |rel err| on the card's held-out pairs is below the old one."""
    old = load_model(inp.store, inp.card.inputs[0])
    path = inp.workdir / "model.json"
    if not path.is_file():
        return Outcome("model_holdout_improves", False, ["the submission has no model.json"])
    new = UnitModel.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    pairs = _pairs(inp.store, inp.card.inputs[1:])
    e_old, e_new = _median_error(old, pairs), _median_error(new, pairs)
    data = {"old_error": e_old, "new_error": e_new, "n": len(pairs)}
    ok = e_new < e_old
    return Outcome(
        "model_holdout_improves",
        ok,
        [] if ok else [f"held-out error {e_new:.4f} not below {e_old:.4f}"],
        data,
    )


def microbench_repeats(inp: CheckInput) -> Outcome:
    from kta.orchestrator.microbench_check import microbench_repeats as check  # WP11; cycle

    return check(inp)


CHECKS: Mapping[str, Callable[[CheckInput], Outcome]] = {
    "delivery_note": delivery_note,
    "validate_family": family_validates,
    "verify_space": space_verifies,
    "failing_cases_pass": failing_cases_pass,
    "reconstruction_fields": reconstruction_complete,
    "matched_templates_exist": matched_templates_exist,
    "gaps_match_unmatched": gaps_match_unmatched,
    "validate_template": template_validates,
    "model_holdout_improves": model_improves,
    "microbench_repeats": microbench_repeats,
}
DONE_CHECKS: Mapping[str, tuple[str, ...]] = {
    "T1": ("delivery_note", "validate_family", "verify_space"),
    "T2": (
        "delivery_note",
        "reconstruction_fields",
        "matched_templates_exist",
        "gaps_match_unmatched",
    ),
    "T3": ("delivery_note", "validate_template"),
    "T4": ("delivery_note", "model_holdout_improves"),
    "T5": ("delivery_note", "microbench_repeats"),
    "T6": ("delivery_note", "validate_family", "failing_cases_pass", "verify_space"),
}


def run_check(name: str, inp: CheckInput) -> Outcome:
    """One named check. A schema problem in the submission is a failed check."""
    fn = CHECKS.get(name)
    if fn is None:
        raise SchemaError(f"unknown done-check {name!r}")
    try:
        return fn(inp)
    except ValueError as exc:  # SchemaError and pydantic's ValidationError: bad submission data
        return Outcome(name, False, [f"{type(exc).__name__}: {exc}"])
