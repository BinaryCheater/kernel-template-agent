"""Markdown renderings generated from the pool: family overview and text knowledge (20 §5).

Everything shown is derived from records; author prose appears only as Template
summaries, labelled "author" on the cards.
"""

from __future__ import annotations

from pathlib import Path

from kta.core.errors import SchemaError
from kta.core.schemas import Record
from kta.knowledge import pool
from kta.knowledge.retrieve import card
from kta.store.base import Store


def _family_templates(store: Store, rec: Record) -> list[str]:
    chain_id = rec.refs.get("chain")
    return pool.chain_template_refs(store.get(chain_id).body) if isinstance(chain_id, str) else []


def _refines(store: Store, ref: str) -> str | None:
    hit = pool.find_template(store, ref)
    return hit[1].refines if hit else None


def family_overview(store: Store, pattern_or_problem: str) -> str:
    """One Markdown page: families, reference chains and gaps for a Pattern or a Problem."""
    is_pattern = pool.find_by_name(store, "Pattern", pattern_or_problem) is not None
    problem = pool.find_by_name(store, "Problem", pattern_or_problem)
    if not is_pattern and problem is None:
        raise SchemaError(f"{pattern_or_problem!r} is neither a Pattern nor a Problem")
    kind = "Pattern" if is_pattern else "Problem"
    lines = [f"# {kind} {pattern_or_problem}: family overview", ""]

    def relevant(rec: Record) -> bool:
        if not is_pattern:
            return problem is not None and rec.refs.get("problem") == problem[0]
        return any(_refines(store, r) == pattern_or_problem for r in _family_templates(store, rec))

    families = [(rid, r) for rid, r in pool.with_ids(pool.records(store, "Family")) if relevant(r)]
    lines += ["## Families", ""]
    lines += [f"- {line}" for rid, _ in families for line in card(store, rid).lines[:3]] or [
        "- none"
    ]
    lines += ["", "## Reference chains", ""]
    for rid, rec in pool.with_ids(pool.records(store, "Reconstruction")):
        chain_id = rec.refs.get("chain")
        ref_id = rec.refs.get("reference")
        if not isinstance(chain_id, str) or not isinstance(ref_id, str):
            continue
        reference = store.get(ref_id)
        steps = pool.chain_template_refs(store.get(chain_id).body)
        if (is_pattern and not any(_refines(store, s) == pattern_or_problem for s in steps)) or (
            not is_pattern and reference.body.get("problem") != pattern_or_problem
        ):
            continue
        lines.append(
            f"- {reference.body['name']}: matched {rec.body['matched']}/{rec.body['total']}, "
            f"{len(rec.body.get('unmatched_transactions', []))} unmatched ({rid[:12]})"
        )
    if lines[-1] == "":
        lines.append("- none")
    if is_pattern:
        gaps = [
            r
            for r in pool.records(store, "Gap")
            if r.body.get("pattern_guess") == pattern_or_problem
        ]
        lines += ["", f"## Known gaps ({len(gaps)})", ""]
        lines += [
            f"- L{g.body.get('level')} {g.body['missing']}: "
            f"{' '.join(g.body['description'].split())[:160]}"
            for g in gaps[:20]
        ] or ["- none"]
    return "\n".join(lines) + "\n"


def _pattern_page(store: Store, name: str, members: set[str] | None) -> str:
    hit = pool.find_by_name(store, "Pattern", name)
    if hit is None:
        raise SchemaError(f"Pattern {name} vanished during rendering")
    lines = [f"# Pattern {name}", "", *card(store, hit[0]).lines, "", "## Templates", ""]
    for rid, t in pool.templates(store):
        if t.refines == name and (members is None or rid in members):
            lines += [*card(store, rid).lines, ""]
    negatives = [r for r in pool.records(store, "Negative")]
    names = {t.name for _, t in pool.templates(store) if t.refines == name}
    negs = [n for n in negatives if n.body.get("subject") in names]
    lines += ["## Negative knowledge", ""]
    lines += [
        f"- {n.body['subject']}: when {n.body['condition']}: {n.body['effect']}" for n in negs
    ] or ["- none"]
    return "\n".join(lines) + "\n"


def render_text_knowledge(store: Store, snapshot_id: str | None, out_dir: Path) -> None:
    """Write ``index.md``, one page per Pattern and ``structures.md`` under ``out_dir``."""
    members: set[str] | None = None
    if snapshot_id is not None:
        snap = store.get(snapshot_id)
        if snap.kind != "PoolSnapshot":
            raise SchemaError(f"{snapshot_id} is a {snap.kind}, not a PoolSnapshot")
        members = set(snap.body.get("members", []))
    (out_dir / "patterns").mkdir(parents=True, exist_ok=True)
    patterns = sorted(pool.names(store, "Pattern"))
    index = [
        "# Knowledge pool",
        "",
        f"Snapshot: {snapshot_id or 'whole store'}",
        "",
        "## Patterns",
        "",
    ]
    for name in patterns:
        (out_dir / "patterns" / f"{name}.md").write_text(
            _pattern_page(store, name, members), encoding="utf-8"
        )
        index.append(f"- [{name}](patterns/{name}.md)")
    structures = ["# Exploitable structures", ""]
    for rid, _ in pool.with_ids(pool.records(store, "Structure")):
        structures += [*card(store, rid).lines, ""]
    (out_dir / "structures.md").write_text("\n".join(structures) + "\n", encoding="utf-8")
    index += ["", "- [Structures](structures.md)"]
    (out_dir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
