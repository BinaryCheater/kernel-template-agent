"""Search the Template library by need, then open it one layer at a time.

Layers: 1 index card, 2 GUIDE, 3 template with proof, 4 realisation with witness.
Every command that returns library content appends to the read log.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .retrieval import ACCESS_WEIGHTS, Ranker
from .store import BUNDLE, INDEX, Library, ReadLog, default_log, default_session

FACETS = ("symptom", "resource")


def render_card(entry: dict) -> str:
    card = entry["card"]
    gap = "" if entry["paths"]["realisation"] else "; no realisation"
    return (
        f"{entry['identity']} [{entry['category']}{gap}]\n"
        f"  When: {card['when']}\n  Choice: {card['choice']}\n  Boundary: {card['boundary']}"
    )


def filter_allowed(library: Library, args: argparse.Namespace) -> tuple[set[int] | None, str]:
    """A value equal to a facet tag filters; any other value becomes search terms."""
    allowed: set[int] | None = None
    extra = []
    for facet in FACETS:
        value = getattr(args, facet)
        if not value:
            continue
        if value in library.lexicon[facet]:
            hits = {i for i, e in enumerate(library.entries) if value in e["facets"][facet]}
            allowed = hits if allowed is None else allowed & hits
        else:
            extra.append(value)
    if args.pattern:
        wanted = args.pattern
        hits = {
            i
            for i, e in enumerate(library.entries)
            if wanted in (e["category"], e["pattern"]["refines"], *e["pattern"]["produces"])
        }
        allowed = hits if allowed is None else allowed & hits
    if args.transformation:
        extra.append(args.transformation)
    return allowed, " ".join(extra)


def cmd_search(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    allowed, extra = filter_allowed(library, args)
    query = " ".join(filter(None, [args.query, extra]))
    if query:
        rows = Ranker(library.entries).rank(query, ACCESS_WEIGHTS, allowed)
    else:
        positions = allowed if allowed is not None else set(range(len(library.entries)))
        rows = [(library.entries[i], 0.0) for i in sorted(positions)]
    top = rows[: args.limit]
    body = "\n".join(f"{i}. {render_card(e)}" for i, (e, _) in enumerate(top, 1))
    log.write(
        "search",
        text=body,
        query=args.query,
        filters={k: getattr(args, k) for k in (*FACETS, "pattern", "transformation")},
        matched=len(rows),
        results=[e["identity"] for e, _ in top],
    )
    print(body if top else "no match; relax a filter or record the need with `need`")
    print(f"\n{len(top)} of {len(rows)} matches shown.")
    return 0


def cmd_facets(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    lines = []
    for facet in FACETS:
        counts = Counter(tag for e in library.entries for tag in e["facets"][facet])
        lines.append(f"{facet}:")
        for tag, rule in sorted(library.lexicon[facet].items()):
            lines.append(f"  {tag} ({counts[tag]}): {rule['meaning']}")
        lines.append(f"  other ({counts['other']}): no rule matched this entry")
    patterns = Counter(e["category"] for e in library.entries)
    lines.append("pattern: " + ", ".join(f"{k} ({v})" for k, v in sorted(patterns.items())))
    body = "\n".join(lines)
    log.write("facets", text=body)
    print(body)
    return 0


def cmd_card(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    body = render_card(library.entry(args.identity))
    log.write("card", text=body, identity=args.identity)
    print(body)
    return 0


def cmd_guide(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    body = library.text(args.identity, "guide")
    log.write("guide", text=body, identity=args.identity)
    print(body)
    return 0


def cmd_template(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    template = library.text(args.identity, "template")
    proof = library.text(args.identity, "proof")
    log.write("template", text=template, identity=args.identity)
    log.write("proof", text=proof, identity=args.identity)
    print(f"# template.yaml\n{template}\n# proof.yaml\n{proof}")
    return 0


def cmd_realisation(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    entry = library.entry(args.identity)
    rows = library.realisations(args.identity)
    witness = entry["witness"]
    if not rows:
        notes = [g for g in entry["gaps"] if g.startswith(("realisation", "witness"))]
        body = f"GAP: no realisation for {args.identity}; witness={witness}; {notes}"
        log.write("realisation", text=body, identity=args.identity, gap=True)
    else:
        body = f"witness={witness}\n" + "\n".join(f"# {p}\n{t}" for p, t in rows.items())
        log.write("realisation", text=body, identity=args.identity, gap=False)
    print(body)
    return 0


def cmd_select(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    library.entry(args.identity)
    log.write("select", identity=args.identity, reason=args.reason)
    print(f"selected {args.identity}")
    return 0


def cmd_need(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    log.write("need", need=args.text)
    print("recorded an unmet need")
    return 0


def cmd_gaps(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    counts = Counter(g.split(":")[0] for e in library.entries for g in e["gaps"])
    print(
        json.dumps(
            {"entries": len(library.entries), "gaps": dict(sorted(counts.items()))}, indent=2
        )
    )
    return 0


def cmd_status(library: Library, log: ReadLog, args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    print(json.dumps(dict(Counter(r["event"] for r in rows)), indent=2))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from .build import build, write_index  # PyYAML is only needed here, never in a package.

    if BUNDLE.exists():
        raise SystemExit("this is a frozen package; rebuild from the repository")
    if args.check:
        current = build(Path(__file__).resolve().parents[2])
        text = json.dumps(current, indent=1, sort_keys=True) + "\n"
        same = INDEX.exists() and INDEX.read_text() == text
        print("index is current" if same else "index is stale; run build")
        return 0 if same else 1
    index = write_index()
    print(f"{len(index['entries'])} entries, library {index['library_fingerprint'][:12]}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="template_access", description=__doc__)
    p.add_argument("--log", type=Path, default=None, help="read log path")
    p.add_argument("--session", default=None, help="label stored on every read")
    p.add_argument("--exclude-source", default=None, help="hide rules with this source lineage")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="layer 1: ranked cards by need")
    s.add_argument("query", nargs="?", default="")
    s.add_argument("--symptom", help="a facet tag, or free words about what goes wrong")
    s.add_argument("--resource", help="a facet tag, or free words about the bottleneck")
    s.add_argument("--pattern", help="category or refined pattern, exact")
    s.add_argument("--transformation", help="free words about the change you want to make")
    s.add_argument("--limit", type=int, default=15)
    s.set_defaults(run=cmd_search)
    sub.add_parser("facets", help="list searchable tags").set_defaults(run=cmd_facets)
    for name, run, text in (
        ("card", cmd_card, "layer 1: one index card"),
        ("guide", cmd_guide, "layer 2: GUIDE"),
        ("template", cmd_template, "layer 3: template and proof outline"),
        ("realisation", cmd_realisation, "layer 4: realisation and witness, or the gap"),
    ):
        c = sub.add_parser(name, help=text)
        c.add_argument("identity")
        c.set_defaults(run=run)
    c = sub.add_parser("select", help="record a rule you would apply")
    c.add_argument("identity")
    c.add_argument("--reason", required=True)
    c.set_defaults(run=cmd_select)
    c = sub.add_parser("need", help="record a need no rule satisfied")
    c.add_argument("text")
    c.set_defaults(run=cmd_need)
    sub.add_parser("gaps", help="count missing layer content").set_defaults(run=cmd_gaps)
    sub.add_parser("status", help="count reads by event").set_defaults(run=cmd_status)
    c = sub.add_parser("build", help="rebuild the index from the library")
    c.add_argument("--check", action="store_true")
    c.set_defaults(run=None)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.command == "build":
        return cmd_build(args)
    library = Library(args.exclude_source)
    log = ReadLog(args.log or default_log(), args.session or default_session())
    return args.run(library, log, args)


if __name__ == "__main__":
    sys.exit(main())
