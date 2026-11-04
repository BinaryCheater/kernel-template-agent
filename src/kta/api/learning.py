"""Learning views (LEARNDEMO-1, D745): /references and a Reference's learning page and source.

One page per Reference tells how the expand loop learned from it: the loop's stages, the
frozen source (``source_tree`` Report, kta.orchestrator.intake), the agent tasks (T2
decomposes, T3 extracts a Template), the chain with the status of every step's Template
(in the pool before this reference, extracted from it, still missing), the unmatched
transactions and Gaps, the admission and coverage Reports. Tasks belong to a Reference
when their card names it as an input.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from kta.api import opsread
from kta.api.common import Api, body, done_check_reports, incoming, iso, machines, ref_one, refs
from kta.api.common import record_ids as _ids
from kta.knowledge.pool import find_template
from kta.orchestrator.outputs import RESULT_TITLE
from kta.store.sqlite import RecordNotFound

TREE_TITLE = "source_tree"  # kta.orchestrator.intake.TITLE (layer 7, not imported for a name)
NOTE_FILES = ("decomposition.md", "DELIVERY.md")
MAX_FILE_LINES = 4000


def _created(api: Api, rid: str) -> str:
    return str(api.store.get_json(rid)["created_at"])


def _reports(api: Api, subject_rel: str, rid: str, title: str) -> list[str]:
    return [r for r in incoming(api, rid, subject_rel, "Report") if body(api, r)["title"] == title]


def _tree(api: Api, source: str | None) -> str | None:
    if not source:
        return None
    found = _reports(api, "source", source, TREE_TITLE)
    return found[-1] if found else None


def _loop(api: Api, ref_id: str) -> opsread.LoopRow | None:
    rows = [lp for lp in opsread.loops(api.ops) if lp.kind == "expand" and lp.subject == ref_id]
    return rows[-1] if rows else None


def _stages(api: Api, loop: opsread.LoopRow | None) -> list[dict[str, Any]]:
    machine = next(m for m in machines()["machines"] if m["id"] == "expand")
    entered: dict[str, str] = {}
    order: list[str] = []
    if loop is not None:
        for ev in api.ops.events_for("loops", loop.id):
            state = ev.change.get("state")
            # every write of a loop row logs its state; a visit is a change of state
            if isinstance(state, str) and (not order or order[-1] != state):
                entered.setdefault(state, iso(ev.at) or "")
                order.append(state)
    skipped = (
        {"measured", "reproduced", "ablated"} if loop and loop.data.get("analysis_only") else set()
    )
    out = []
    for s in machine["states"]:
        sid = s["id"]
        if loop is not None and sid == loop.state:
            status = "current" if s["kind"] != "terminal" else "done"
        elif sid in skipped:  # passed through without doing its work
            status = "skipped"
        elif sid in entered:
            status = "done"
        else:
            status = "pending"
        out.append(
            {
                "id": sid,
                "kind": s["kind"],
                "work": s["work"],
                "does": s["does"],
                "status": status,
                "entered_at": entered.get(sid),
                "visits": order.count(sid),
            }
        )
    return out


def _tasks(api: Api, ref_id: str) -> list[dict[str, Any]]:
    out = []
    for t in opsread.tasks(api.ops):
        if t.card_type not in ("T2", "T3"):
            continue
        card = body(api, t.card_id)
        if ref_id not in card.get("inputs", []):
            continue
        results = [
            r
            for r in incoming(api, t.card_id, "card", "Report")
            if body(api, r)["title"] == RESULT_TITLE
        ]
        outputs = body(api, results[-1])["data"].get("outputs", {}) if results else {}
        subs = incoming(api, t.card_id, "card", "Submission")
        out.append(
            {
                "id": t.id,
                "type": t.card_type,
                "goal": card["goal"],
                "state": t.state,
                "attempts": t.attempts,
                "created_at": iso(t.created_at),
                "updated_at": iso(t.updated_at),
                "submissions": subs,
                "outputs": outputs,
                "checks": [
                    {k: c[k] for k in ("id", "passed", "summary", "reason", "created_at")}
                    for c in done_check_reports(api, t.id)
                ],
            }
        )
    return out


def _step_status(api: Api, ref: str, extracted: set[str], since: str | None) -> dict[str, Any]:
    found = find_template(api.store, ref)
    if found is None:
        return {"status": "missing", "template_id": None}
    tid = found[0]
    if tid in extracted:
        status = "extracted"
    elif since and _created(api, tid) > since:
        status = "added_later"
    else:
        status = "in_pool"
    return {
        "status": status,
        "template_id": tid,
        "summary": found[1].summary,
        "level": [found[1].level_from, found[1].level_to],
    }


def _blob_text(api: Api, sha: str) -> str:
    return api.store.blob_path(sha).read_bytes().decode("utf-8", "replace")


def _notes(api: Api, subs: list[str]) -> dict[str, str]:
    """The decomposition notes of the newest submission that has them."""
    for sub in reversed(subs):
        blobs = api.store.get(sub).blobs
        found = {n: _blob_text(api, blobs[n]) for n in NOTE_FILES if n in blobs}
        if found:
            return found | {"submission": sub}
    return {}


def _reconstruction(api: Api, recon: str, extracted: set[str], since: str | None) -> dict[str, Any]:
    rb = body(api, recon)
    chain_id = ref_one(api, recon, "chain")
    chain = body(api, chain_id) if chain_id else {"items": []}
    items = [
        {
            **{k: i.get(k) for k in ("key", "template", "applies_to", "bindings", "after")},
            **_step_status(api, str(i["template"]), extracted, since),
        }
        for i in chain.get("items", [])
    ]
    gap_ids = refs(api, recon).get("gaps", []) or []
    gaps = [{"id": g, **body(api, g)} for g in gap_ids]
    return {
        "id": recon,
        "created_at": _created(api, recon),
        "chain": chain_id,
        "workload": chain.get("workload"),
        "matched": rb["matched"],
        "total": rb["total"],
        "notes": rb.get("notes"),
        "items": items,
        "unmatched": rb.get("unmatched_transactions", []),
        "gaps": gaps,
    }


def learning(api: Api, ref_id: str) -> dict[str, Any]:
    rec = api.store.get(ref_id)
    if rec.kind != "Reference":
        raise RecordNotFound(f"{ref_id} is a {rec.kind}, not a Reference")
    source = ref_one(api, ref_id, "source")
    problem = ref_one(api, ref_id, "problem")
    tree = _tree(api, source)
    loop = _loop(api, ref_id)
    tasks = _tasks(api, ref_id)
    extracted = {str(t["outputs"]["template"]) for t in tasks if t["outputs"].get("template")}
    since = iso(min((api.ops.events_for("loops", loop.id)[0].at,), default=None)) if loop else None
    recons = sorted(
        incoming(api, ref_id, "reference", "Reconstruction"), key=lambda r: _created(api, r)
    )
    rounds = [_reconstruction(api, r, extracted, since) for r in recons]
    t2_subs = [s for t in tasks if t["type"] == "T2" for s in t["submissions"]]
    tree_view = None
    if tree:
        tb = body(api, tree)
        tree_view = {"id": tree, "summary": tb["summary"], **tb["data"]}
    return {
        "reference": {"id": ref_id, **rec.body},
        "source": {"id": source, **body(api, source)} if source else None,
        "problem": {"id": problem, **body(api, problem)} if problem else None,
        "source_tree": tree_view,
        "loop": None
        if loop is None
        else {
            "id": loop.id,
            "state": loop.state,
            "updated_at": iso(loop.updated_at),
            "analysis_only": loop.data.get("analysis_only"),
            "round": loop.data.get("round", 1),
            "note": loop.data.get("note") or loop.data.get("hold"),
        },
        "stages": _stages(api, loop),
        "tasks": tasks,
        "rounds": rounds,
        "extracted_templates": [
            {"id": t, "name": body(api, t)["name"], "summary": body(api, t).get("summary")}
            for t in sorted(extracted)
        ],
        "admission": [
            {"id": r, "summary": body(api, r)["summary"], "data": body(api, r)["data"]}
            for r in _reports(api, "reference", ref_id, "admission")
        ],
        "coverage": [
            {"id": r, "summary": body(api, r)["summary"], "created_at": _created(api, r)}
            for r in _reports(api, "reference", ref_id, "coverage")
        ],
        "notes": _notes(api, t2_subs),
    }


@lru_cache(maxsize=4)
def _unpacked(path: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(path.read_bytes())), mode="r:") as tar:
        for m in tar.getmembers():
            handle = tar.extractfile(m) if m.isfile() else None
            if handle is not None:
                out[m.name] = handle.read()
    return out


def source_file(api: Api, ref_id: str, path: str) -> dict[str, Any]:
    tree = _tree(api, ref_one(api, ref_id, "source"))
    if tree is None:
        raise RecordNotFound(f"Reference {ref_id[:12]} has no frozen source tree")
    files = _unpacked(api.store.blob_path(api.store.get(tree).blobs["source"]))
    if path not in files:
        raise RecordNotFound(f"{path} is not among the frozen source files")
    lines = files[path].decode("utf-8", "replace").splitlines()
    return {
        "path": path,
        "tree": tree,
        "lines": lines[:MAX_FILE_LINES],
        "truncated": len(lines) > MAX_FILE_LINES,
    }


def references(api: Api) -> list[dict[str, Any]]:
    loops = {lp.subject: lp for lp in opsread.loops(api.ops) if lp.kind == "expand"}
    out = []
    for rid in _ids(api, "Reference"):
        recons = incoming(api, rid, "reference", "Reconstruction")
        last = max(recons, key=lambda r: _created(api, r)) if recons else None
        lp = loops.get(rid)
        out.append(
            {
                "id": rid,
                "name": body(api, rid)["name"],
                "loop_state": lp.state if lp else None,
                "matched": body(api, last)["matched"] if last else None,
                "total": body(api, last)["total"] if last else None,
                "origin": api.store.get(rid).actor.name,
            }
        )
    return out


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/references")
    def list_refs() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return references(api)

    @r.get("/references/{id}/learning")
    def view(id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return learning(api, id)

    @r.get("/references/{id}/source")
    def source(id: str, path: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return source_file(api, id, path)

    return r
