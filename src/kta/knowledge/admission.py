"""Admit: the deterministic step after decompose (docs/design/02 §8.6; D560, D561).

A worker's reference directory (the decompose-kernel layout) is validated against the
pool. Issues fall into three classes (``kta.knowledge.learning.issue_class``):

- **error**: the deliverable is refused; in the expand loop the T2 is rejected and the
  failures go back to the same agent session (the task queue retries at most
  ``max_retries`` = 2 times, then escalates);
- **warning**: must be explained in ``warnings.yaml`` at the root of the directory,
  ``"<kind>@<file>:<where>": <explanation>``; an unexplained warning refuses like an error;
- **language gap**: an expressibility gap of the expression language or its vocabulary;
  counted and admitted, the record stays ``proposed``.

Every admission writes (import path) or carries (T2 done-check data, then the T2 writer)
a ``Report`` titled ``admission`` with the counts, which coverage reads.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from kta.core.context import Context
from kta.core.errors import SchemaError
from kta.core.schemas import Actor, Record, RefValue
from kta.knowledge import pool
from kta.knowledge.seed import ImportResult, import_references, reference_order
from kta.knowledge.types import Issue
from kta.store import CodeState
from kta.store.base import Store

EXPLANATIONS = "warnings.yaml"
TITLE = "admission"


def warning_key(issue: Issue, file: str) -> str:
    return f"{issue.kind}@{file}:{issue.where}"


def load_explanations(path: Path) -> dict[str, str]:
    """``warnings.yaml``: warning key -> explanation (absent file: none)."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SchemaError(f"{path.name} must map warning keys to explanations")
    return {str(k): str(v).strip() for k, v in cast(dict[Any, Any], data).items()}


@dataclass(frozen=True)
class Admission:
    errors: list[str] = field(default_factory=lambda: list[str]())
    warnings: list[dict[str, str]] = field(default_factory=lambda: list[dict[str, str]]())
    language_gaps: list[dict[str, str]] = field(default_factory=lambda: list[dict[str, str]]())

    @property
    def unexplained(self) -> list[str]:
        return [w["key"] for w in self.warnings if not w["explanation"]]

    @property
    def admitted(self) -> bool:
        return not self.errors and not self.unexplained

    def failures(self) -> list[str]:
        return self.errors + [
            f"warning {k} is not explained in {EXPLANATIONS}" for k in self.unexplained
        ]

    def counts(self) -> dict[str, Any]:
        return {
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "unexplained": len(self.unexplained),
            "language_gaps": len(self.language_gaps),
            "language_gaps_by_kind": dict(
                sorted(Counter(g["kind"] for g in self.language_gaps).items())
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "counts": self.counts(), "admitted": self.admitted}


def admit(issues: Iterable[tuple[str, Issue]], explanations: dict[str, str]) -> Admission:
    """Classify ``(file, issue)`` pairs (issues already classified by the validators)."""
    out = Admission()
    for file, i in issues:
        key = warning_key(i, file)
        row = {"key": key, "kind": i.kind, "message": i.message}
        if i.severity == "error":
            out.errors.append(f"{key}: {i.message}")
        elif i.severity == "gap":
            out.language_gaps.append(row)
        else:
            out.warnings.append({**row, "explanation": explanations.get(key, "")})
    return out


def report_body(admission: Admission, subject: str, what: str) -> dict[str, Any]:
    c = admission.counts()
    return {
        "title": TITLE,
        "subject": subject,
        "summary": (
            f"{what}: {'admitted' if admission.admitted else 'refused'}; {c['errors']} errors, "
            f"{c['warnings']} warnings ({c['unexplained']} unexplained), "
            f"{c['language_gaps']} language gaps"
        ),
        "data": admission.to_dict(),
    }


def _split(root: Path, issue: Issue) -> tuple[str, str, Issue]:
    """(reference dir, file inside it, issue with the file prefix removed) of a batch issue."""
    path, _, where = issue.where.partition(": ")
    rel = Path(path).resolve().relative_to(root.resolve())
    ref, file = rel.parts[0], "/".join(rel.parts[1:])
    return ref, file, Issue(issue.kind, issue.message, where, issue.severity)


def admit_references(
    ctx: Context, store: Store, root: Path, code: CodeState | None = None
) -> tuple[ImportResult, dict[str, str]]:
    """Import every reference directory under ``root`` through admit, into the store.

    Errors and unexplained warnings refuse the whole batch (nothing is written); language
    gaps are counted. Returns the import result and, per reference name, the ID of its
    ``admission`` Report (subject: its Reconstruction).
    """
    per: dict[str, Admission] = {}

    def explain(result: ImportResult) -> list[Issue]:
        found: dict[str, list[tuple[str, Issue]]] = {d.name: [] for d in reference_order(root)}
        for i in [*result.warnings, *result.language_gaps]:
            ref, file, issue = _split(root, i)
            found.setdefault(ref, []).append((file, issue))
        bad: list[Issue] = []
        for ref, pairs in found.items():
            per[ref] = admit(pairs, load_explanations(root / ref / EXPLANATIONS))
            bad += [
                Issue("unexplained_warning", f"{k} is not explained in {EXPLANATIONS}", ref)
                for k in per[ref].unexplained
            ]
        return bad

    result = import_references(ctx, store, root, code, admit=explain)
    actor = Actor(type="system", name="kta.knowledge.admission")
    reports: dict[str, str] = {}
    for ref, a in per.items():
        found = pool.find_by_name(store, "Reference", ref)
        if found is None:
            continue
        recons = [
            i
            for i in store.neighbors(found[0], "reference", "in")
            if store.get(i).kind == "Reconstruction"
        ]
        subject = recons[-1] if recons else found[0]
        body = report_body(a, subject, f"reference {ref}")
        refs: dict[str, RefValue] = {"reference": found[0]}
        reports[ref] = store.put(Record.new("Report", body, actor, ctx.clock.now(), refs=refs))
    return result, reports


def latest_admissions(store: Store, references: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The newest ``admission`` Report data per Reference ID (absent: never admitted)."""
    out: dict[str, dict[str, Any]] = {}
    wanted = set(references)
    for rec in store.query("Report", {"title": TITLE}, order="created_at", limit=pool.ALL):
        ref = rec.refs.get("reference")
        if isinstance(ref, str) and ref in wanted:
            out[ref] = dict(rec.body["data"])
    return out
