"""Scripted human submissions for the expand loop: WP10a's hand decompositions resubmitted
as T2 and T3 outputs (WP10b). Nothing here imitates an agent: every file comes from
``knowledge/seed/references/<name>/`` or from the T2 Submission a T3 card lists."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from knowhelp import CODE, SEED
from kta.core.context import Context
from kta.core.schemas import Actor, Record, RefValue
from kta.knowledge import import_hw_catalog, import_seed
from kta.knowledge.pool import find_by_name
from kta.knowledge.seed import reference_order
from kta.orchestrator.tasks import TaskQueue, Verdict
from kta.store.base import Store
from kta.targets.gb10 import catalog

REFERENCES = SEED / "references"
HUMAN = Actor(type="human", name="wp10b replay")
OWNER = "human:replay"
NOTE = """# What was done
Resubmitted the WP10a hand decomposition of {what} (knowledge/seed/references/{ref}/),
unchanged, through the human submission path.

# Fixtures and simplifications
None. The files are WP10a's analysis of the frozen source; nothing was built or run.

# Known issues
See deliveries/WP10a.md for the open Gaps of this reference.
"""
T2_FILES = ("chain.yaml", "reconstruction.yaml")
T2_DIRS = ("gaps", "templates", "patterns")
TEMPLATE_IN_GOAL = re.compile(r"Extract Template (\S+@\d+)")


def pool(ctx: Context, store: Store) -> None:
    """What ``kta init`` writes: the seed knowledge and the GB10 feature list."""
    import_hw_catalog(ctx, store, catalog(), CODE)
    import_seed(ctx, store, SEED, CODE)


def references(ctx: Context, store: Store) -> dict[str, str]:
    """Source, Problem and Reference records of every WP10a reference: name -> Reference ID."""
    out: dict[str, str] = {}
    problems: dict[str, str] = {}
    now = ctx.clock.now()
    for d in reference_order(REFERENCES):
        body = _yaml(d / "reference.yaml")
        src = store.put(Record.new("Source", _yaml(d / "source.yaml"), HUMAN, now))
        prob = _yaml(d / "problem.yaml")
        if prob["name"] not in problems:
            problems[prob["name"]] = store.put(Record.new("Problem", prob, HUMAN, now))
        refs: dict[str, RefValue] = {"source": src, "problem": problems[prob["name"]]}
        out[d.name] = store.put(Record.new("Reference", body, HUMAN, now, refs=refs))
    return out


def t2_files(ref: str) -> dict[str, bytes]:
    d = REFERENCES / ref
    files = {name: (d / name).read_bytes() for name in T2_FILES}
    for sub in T2_DIRS:
        files |= {f"{sub}/{p.name}": p.read_bytes() for p in sorted((d / sub).glob("*.yaml"))}
    return files


def t3_files(store: Store, inputs: list[str], goal: str) -> dict[str, bytes]:
    """The Template the card names, from the drafts of the T2 Submission in its inputs."""
    m = TEMPLATE_IN_GOAL.search(goal)
    assert m, goal
    name, _, version = m.group(1).partition("@")
    recs = {store.get(i).kind: (i, store.get(i)) for i in inputs}
    sub = recs["Submission"][1]
    draft = _blob(store, sub, f"templates/{name}.yaml")
    assert draft["version"] == int(version)
    patterns = [
        _blob(store, sub, f"patterns/{p}.yaml")
        for p in [draft["refines"], *draft.get("produces", [])]
        if find_by_name(store, "Pattern", p) is None and f"patterns/{p}.yaml" in sub.blobs
    ]
    doc: dict[str, Any] = {"template": draft, "sources": [recs["Source"][0]]}
    if patterns:
        doc["patterns"] = patterns
    return {"template.yaml": yaml.safe_dump(doc, sort_keys=False).encode()}


def in_dependency_order(files: dict[str, dict[str, bytes]]) -> list[str]:
    """Task IDs so that a T3 comes after the T3s whose Templates its relations name.

    A human answers sibling T3s in this order; agents working in parallel would instead
    see an ``unknown_relation_target`` rejection and a retry (WP10b known issue).
    """
    names: dict[str, tuple[str, set[str]]] = {}
    for task, f in files.items():
        if "template.yaml" in f:
            t = yaml.safe_load(f["template.yaml"])["template"]
            names[task] = (t["name"], {r["target"] for r in t.get("relations", [])})
    out = [task for task in files if task not in names]
    pending = dict(names)
    while pending:
        waiting = {n for n, _ in pending.values()}
        ready = [k for k, (_, deps) in pending.items() if not deps & waiting] or list(pending)[:1]
        out += ready
        for k in ready:
            del pending[k]
    return out


def submit(q: TaskQueue, task: str, files: dict[str, bytes], ref: str, what: str) -> Verdict:
    """``kta task submit --check`` without the CLI: lease, start, submit, check."""
    if q.get(task).state == "queued":
        q.lease(task, OWNER)
    q.start(task, OWNER)
    q.submit(task, OWNER, files, NOTE.format(ref=ref, what=what), HUMAN)
    return q.check(task)


def failures(v: Verdict) -> list[tuple[str, list[str]]]:
    return [(o.name, o.failures) for o in v.outcomes if not o.passed]


def _yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data  # pyright: ignore[reportUnknownVariableType]


def _blob(store: Store, rec: Record, name: str) -> dict[str, Any]:
    data = yaml.safe_load(store.blob_path(rec.blobs[name]).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data  # pyright: ignore[reportUnknownVariableType]
