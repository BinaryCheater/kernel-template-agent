"""No-LLM retrieval baseline: task text to ranked rules, plus need text to ranked rules."""

from __future__ import annotations

import math

from template_access.retrieval import ACCESS_WEIGHTS, FLAT_WEIGHTS, Ranker, tokens
from template_access.store import Library

from .gold import Move, Task

SURFACED_K, OPENED_K, SELECTED_K = 20, 8, 5
CUTS = (5, 10, 20, 50)
RETRIEVERS = {"flat": FLAT_WEIGHTS, "access": ACCESS_WEIGHTS}


def first_rank(order: list[str], targets: set[str]) -> int | None:
    return next((i for i, identity in enumerate(order, 1) if identity in targets), None)


def rerank_opened(entries: list[dict], statement: str) -> list[str]:
    """Stage two reads what a GUIDE adds: the deterministic stand-in for reading a page."""
    wanted = set(tokens(statement))

    def overlap(entry: dict) -> float:
        words = tokens(entry["guide_when"] + " " + entry["transformation"]["program"])
        return len(wanted & set(words)) / math.sqrt(len(set(words)) or 1)

    ranked = sorted(entries, key=lambda e: (-overlap(e), e["identity"]))
    return [e["identity"] for e in ranked]


def stage_of(
    move: Move, scoring: str, view: set[str], order: list[str], selected: list[str]
) -> str:
    targets = move.targets(scoring, view)
    if not targets:
        return "absent_from_view"
    rank = first_rank(order, targets)
    if rank is None or rank > SURFACED_K:
        return "not_surfaced"
    if rank > OPENED_K:
        return "surfaced_not_opened"
    return "selected" if targets & set(selected) else "opened_not_selected"


def expected_random(hits: int, size: int, k: int) -> float:
    """Chance that a uniform shuffle puts at least one of `hits` targets in the top k."""
    if hits == 0:
        return 0.0
    return 1.0 - math.comb(size - hits, min(k, size)) / math.comb(size, min(k, size))


def evaluate_task(task: Task, view_name: str, library: Library) -> dict:
    view = set(library.by_identity)
    ranker = Ranker(library.entries)
    rows: dict = {"task": task.id, "view": view_name, "view_size": len(view), "retrievers": {}}
    for retriever, weights in RETRIEVERS.items():
        ranked = ranker.rank(task.statement, weights)
        order = [e["identity"] for e, _ in ranked]
        opened = [e for e, _ in ranked[:OPENED_K]]
        selected = rerank_opened(opened, task.statement)[:SELECTED_K]
        moves = []
        for move in task.moves:
            need_order = [e["identity"] for e, _ in ranker.rank(move.need, weights)]
            row = {"move": move.id, "role": move.role}
            for scoring in ("strict", "lenient"):
                targets = move.targets(scoring, view)
                row[scoring] = {
                    "stage": stage_of(move, scoring, view, order, selected),
                    "targets": sorted(targets),
                    "task_rank": first_rank(order, targets),
                    "need_rank": first_rank(need_order, targets),
                    "random_at": {k: expected_random(len(targets), len(view), k) for k in CUTS},
                }
            moves.append(row)
        rows["retrievers"][retriever] = {"top20": order[:20], "selected": selected, "moves": moves}
    return rows


def recall(moves: list[dict], scoring: str, field: str, k: int, in_view_only: bool) -> dict:
    rows = [m[scoring] for m in moves]
    if in_view_only:
        rows = [r for r in rows if r["targets"]]
    hits = sum(1 for r in rows if r[field] is not None and r[field] <= k)
    return {"hits": hits, "of": len(rows)}


def summarise(result: dict) -> dict:
    out = {}
    for retriever, body in result["retrievers"].items():
        moves = body["moves"]
        row: dict = {}
        for scoring in ("strict", "lenient"):
            stages: dict[str, int] = {}
            for move in moves:
                stage = move[scoring]["stage"]
                stages[stage] = stages.get(stage, 0) + 1
            row[scoring] = {
                "stages": stages,
                "task_recall": {k: recall(moves, scoring, "task_rank", k, True) for k in CUTS},
                "need_recall": {k: recall(moves, scoring, "need_rank", k, True) for k in CUTS},
                "random_expected": {
                    k: sum(m[scoring]["random_at"][k] for m in moves if m[scoring]["targets"])
                    for k in CUTS
                },
            }
        out[retriever] = row
    return out
