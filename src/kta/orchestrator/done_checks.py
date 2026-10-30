"""Fixed done-checks per task type (spec/modules/orchestrator-cli.md "完成检查", D277).

``TaskCard.done_checks`` stores the names below; agents cannot change them. A check that
cannot run (NotSupported, CheckerError, ExecutionError) suspends the task instead of
rejecting it (docs/design/10 §7). Submission layouts are fixed per type:

- T1, T6: a family directory at the submission root (family.yaml, chain.yaml, gen.py, ...)
- T2: the decompose-kernel skill's layout: ``chain.yaml`` (Chain body),
  ``reconstruction.yaml`` (Reconstruction body), ``gaps/*.yaml`` (one Gap body per file,
  in file-name order) and ``patterns/*.yaml`` (proposed Patterns); ``templates/`` holds
  drafts the T3 tasks start from (D381).
  The single-file form, ``reconstruction.yaml`` with keys ``chain``, ``reconstruction`` and
  ``gaps``, is also read.
- T3: ``template.yaml`` with ``template`` (Template body), ``sources`` (Source IDs) and
  optional ``patterns`` (Pattern bodies the Template refines that the pool lacks, D383)
- T4: ``model.json``, a UnitModel in ``UnitModel.to_dict`` form
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
from kta.core.errors import NotSupported, SchemaError
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Measurement, Prediction
from kta.core.schemas.knowledge import Chain, Gap, Pattern, Reconstruction, Template
from kta.core.schemas.process import TaskCard
from kta.knowledge import errors_of, load_family, validate_family, validate_template
from kta.knowledge.pending import PendingView
from kta.knowledge.pool import find_by_name, find_template, parse_template_ref
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


@dataclass(frozen=True)
class T2Docs:
    chain: Chain
    reconstruction: Reconstruction
    gaps: list[Gap]
    patterns: list[Pattern]  # proposed Patterns (``patterns/*.yaml``), written on acceptance


def t2_documents(workdir: Path) -> T2Docs:
    """The T2 submission in either layout (module docstring)."""
    doc = read_yaml(workdir / "reconstruction.yaml")
    if "chain" in doc:
        raw_chain, raw_recon, raw_gaps = (
            doc.get("chain"),
            doc.get("reconstruction"),
            doc.get("gaps"),
        )
    else:
        raw_chain, raw_recon = read_yaml(workdir / "chain.yaml"), doc
        raw_gaps = [read_yaml(p) for p in sorted((workdir / "gaps").glob("*.yaml"))]
    raw_patterns = [read_yaml(p) for p in sorted((workdir / "patterns").glob("*.yaml"))]
    return T2Docs(
        Chain.model_validate(raw_chain),
        Reconstruction.model_validate(raw_recon),
        [Gap.model_validate(g) for g in raw_gaps or []],
        [Pattern.model_validate(p) for p in raw_patterns],
    )


def _reconstruction(inp: CheckInput) -> tuple[Chain, Reconstruction, list[Gap]]:
    d = t2_documents(inp.workdir)
    return d.chain, d.reconstruction, d.gaps


def reconstruction_complete(inp: CheckInput) -> Outcome:
    """Counting rules of the decompose-kernel skill (WP10a D102, D380): ``total`` is the
    number of chain items, ``matched`` at most ``total``, and every unmatched step is
    one of the unmatched transactions (which may also hold transactions no step explains)."""
    try:
        chain, recon, _ = _reconstruction(inp)
    except (SchemaError, ValueError) as exc:  # pydantic ValidationError is a ValueError
        return Outcome("reconstruction_fields", False, [str(exc)])
    failures: list[str] = []
    if recon.total <= 0 or recon.total != len(chain.items):
        failures.append(f"total {recon.total} must equal the {len(chain.items)} chain items")
    if not 0 <= recon.matched <= recon.total:
        failures.append(f"matched {recon.matched} of total {recon.total} is inconsistent")
    if len(recon.unmatched_transactions) < recon.total - recon.matched:
        failures.append(
            f"{recon.total - recon.matched} unmatched steps but only "
            f"{len(recon.unmatched_transactions)} unmatched transactions"
        )
    try:
        new_patterns(inp.store, t2_documents(inp.workdir).patterns)
    except SchemaError as exc:
        failures.append(str(exc))
    keys = [i.key for i in chain.items]
    if len(set(keys)) != len(keys):
        failures.append("chain item keys are not unique")
    unknown = sorted({a for i in chain.items for a in i.after} - set(keys))
    if unknown:
        failures.append(f"chain items are after unknown keys {unknown}")
    return Outcome("reconstruction_fields", not failures, failures)


def matched_templates_exist(inp: CheckInput) -> Outcome:
    """``matched`` cannot exceed the chain items whose Template is in the pool; the others
    name a Template to extract (``name@version``), which the loop gives to T3 (D382)."""
    chain, recon, _ = _reconstruction(inp)
    failures: list[str] = []
    known = 0
    missing: list[str] = []
    for item in chain.items:
        try:
            parse_template_ref(item.template)
        except SchemaError as exc:
            failures.append(f"chain item {item.key}: {exc}")
            continue
        if find_template(inp.store, item.template) is None:
            missing.append(item.template)
        else:
            known += 1
    if recon.matched > known:
        failures.append(
            f"matched {recon.matched} but only {known} chain items use Templates in the pool"
        )
    return Outcome(
        "matched_templates_exist", not failures, failures, {"to_extract": sorted(set(missing))}
    )


def gaps_match_unmatched(inp: CheckInput) -> Outcome:
    """One Gap per unmatched transaction, in order, with equal level and Pattern guess."""
    _, recon, gaps = _reconstruction(inp)
    txs = recon.unmatched_transactions
    if len(gaps) != len(txs):
        return Outcome(
            "gaps_match_unmatched",
            False,
            [f"{len(txs)} unmatched transactions but {len(gaps)} Gaps"],
        )
    failures = [
        f"Gap {n} ({g.level}, {g.pattern_guess}) differs from its transaction "
        f"({tx.level}, {tx.pattern_guess})"
        for n, (g, tx) in enumerate(zip(gaps, txs, strict=True), 1)
        if (g.level is not None and g.level != tx.level)
        or (g.pattern_guess is not None and g.pattern_guess != tx.pattern_guess)
    ]
    return Outcome("gaps_match_unmatched", not failures, failures)


def t3_documents(inp: CheckInput) -> tuple[Template, list[Pattern], dict[str, Any]]:
    doc = read_yaml(inp.workdir / "template.yaml")
    raw: list[Any] = list(doc.get("patterns") or [])
    patterns = [Pattern.model_validate(p) for p in raw]
    return Template.model_validate(doc.get("template")), patterns, doc


def new_patterns(store: Store, patterns: list[Pattern]) -> list[Pattern]:
    """The submitted Patterns the pool lacks; a known name with another body is refused."""
    out: list[Pattern] = []
    for p in patterns:
        found = find_by_name(store, "Pattern", p.name)
        if found is None:
            out.append(p)
        elif Pattern.model_validate(found[1].body).model_dump() != p.model_dump():
            raise SchemaError(f"Pattern {p.name} exists with a different body")
    return out


def _inputs_of(inp: CheckInput, kind: str) -> list[Record]:
    recs = (inp.store.get(i) for i in inp.card.inputs if inp.store.exists(i))
    return [r for r in recs if r.kind == kind]


def t3_pool(inp: CheckInput, t: Template, patterns: list[Pattern]) -> Store:
    """The pool a T3 Template is validated against (D383): the store, plus the submitted
    Patterns it lacks, plus the T2 drafts of the other Templates the input Reconstruction's
    chain names that are not extracted yet, so siblings may name each other in relations."""
    now = inp.ctx.clock.now()
    actor = inp.submission.actor
    pending = [Record.new("Pattern", p, actor, now) for p in new_patterns(inp.store, patterns)]
    wanted = {
        str(i["template"])
        for r in _inputs_of(inp, "Reconstruction")
        if isinstance(r.refs.get("chain"), str)
        for i in inp.store.get(str(r.refs["chain"])).body.get("items", [])
    } - {f"{t.name}@{t.version}"}
    for sub in _inputs_of(inp, "Submission"):
        for ref in sorted(wanted):
            name, version = parse_template_ref(ref)
            blob = sub.blobs.get(f"templates/{name}.yaml")
            if blob is None or find_template(inp.store, ref) is not None:
                continue
            draft: Any = yaml.safe_load(inp.store.blob_path(blob).read_text(encoding="utf-8"))
            if isinstance(draft, dict) and cast(dict[str, Any], draft).get("version") == version:
                pending.append(Record.new("Template", Template.model_validate(draft), actor, now))
    return PendingView(inp.store, pending) if pending else inp.store


def template_fills_chain(inp: CheckInput) -> Outcome:
    """For an expand-loop T3 (a Reconstruction among the inputs): the Template is one its
    chain names and the pool does not hold yet (D382)."""
    t, _, _ = t3_documents(inp)
    ref = f"{t.name}@{t.version}"
    chains = [
        inp.store.get(str(r.refs["chain"])).body
        for r in _inputs_of(inp, "Reconstruction")
        if isinstance(r.refs.get("chain"), str)
    ]
    if not chains:
        return Outcome("template_fills_chain", True, data={"note": "no Reconstruction input"})
    failures: list[str] = []
    if not any(ref in {str(i["template"]) for i in c.get("items", [])} for c in chains):
        failures.append(f"{ref} is not named by the input Reconstruction's chain")
    if find_template(inp.store, ref) is not None:
        failures.append(f"{ref} is already in the pool")
    return Outcome("template_fills_chain", not failures, failures)


def template_validates(inp: CheckInput) -> Outcome:
    t, patterns, doc = t3_documents(inp)
    issues = validate_template(t, t3_pool(inp, t, patterns))
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
    raise NotSupported("T5 microbenchmark repeat check: the benchmark format arrives with WP11")


CHECKS: Mapping[str, Callable[[CheckInput], Outcome]] = {
    "delivery_note": delivery_note,
    "validate_family": family_validates,
    "verify_space": space_verifies,
    "failing_cases_pass": failing_cases_pass,
    "reconstruction_fields": reconstruction_complete,
    "matched_templates_exist": matched_templates_exist,
    "gaps_match_unmatched": gaps_match_unmatched,
    "validate_template": template_validates,
    "template_fills_chain": template_fills_chain,
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
    "T3": ("delivery_note", "validate_template", "template_fills_chain"),
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
