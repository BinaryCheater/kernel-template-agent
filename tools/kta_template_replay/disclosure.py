"""Derive category-first navigation from a frozen canonical library for one fresh run."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

DESCRIPTIONS = {
    "access": "Element access, addressing and packed index mappings.",
    "contract": "Contractions, dot products, instruction choices and blocking.",
    "convert": "Representation and precision conversions.",
    "decode": "Recovering values from encoded representations.",
    "map": "Map-pattern transformations, including computation and layout choices.",
    "reduce": "Combining values, normalized reductions and partial-result interfaces.",
    "scan": "Prefix computation, scans and state recurrences.",
    "select": "Selection and pruning choices.",
    "store": "Physical storage, value lifetime and materialization.",
    "sync": "Ordering, handshakes and completion protocols.",
    "transfer": "Moving values and arranging transfer/compute overlap.",
    "work_assign": "Assigning, ordering and scheduling work.",
}


def add_navigation(run: Path) -> None:
    workspace = run / "workspace"
    package_path = run / "package.json"
    package = json.loads(package_path.read_text())
    entries = json.loads((workspace / "library/catalog.json").read_text())["entries"]
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for entry in entries:
        groups[entry["category"]].append(entry)
    cards = workspace / "library/cards"
    cards.mkdir()
    menu = [
        "# Library starting menu",
        "",
        "Start with these category descriptions, then read a category's cards on demand. "
        "Read a selected rule's guide and paired files before using it. Categories follow "
        "the existing pattern metadata; they do not restrict applicability. Search across "
        "categories or the unchanged plain catalog whenever useful.",
        "",
        "| Category | Entries | Retrieval cue | Cards |",
        "|---|---:|---|---|",
    ]
    for category, items in sorted(groups.items()):
        cue = DESCRIPTIONS.get(category, "Other declared pattern; inspect its cards.")
        relative = f"library/cards/{category}.md"
        menu.append(f"| {category} | {len(items)} | {cue} | `{relative}` |")
        lines = [f"# {category} cards", "", cue, ""]
        for entry in items:
            lines += [
                f"## {entry['identity']}: {entry['title']}",
                "",
                f"When: {entry['when']}",
                "",
                f"Choice: {entry['choice']}",
                "",
                f"Boundary: {entry['boundary']}",
                "",
                f"Guide: library/{entry['guide']}",
                "",
            ]
        path = workspace / relative
        path.write_text("\n".join(lines))
        path.chmod(0o444)
    initial = (run / "prompt.md").read_text().rstrip() + "\n\n" + "\n".join(menu) + "\n"
    (run / "prompt.md").write_text(initial)
    content = initial.encode()
    (run / "disclosure.json").write_text(
        json.dumps(
            {
                "mode": "category-disclosure",
                "source": "unchanged frozen catalog metadata",
                "initial_prompt_bytes": len(content),
                "initial_prompt_characters": len(initial),
                "initial_prompt_identity": hashlib.sha256(content).hexdigest(),
                "initial_context_token_measurement": (
                    "native first-request input tokens after launch; includes harness system "
                    "context and the retained prompt, not a byte-based estimate"
                ),
                "categories": {key: len(value) for key, value in sorted(groups.items())},
                "semantic_library_change": "none; added navigation only",
            },
            indent=2,
        )
    )
    package["mode"] = "category-disclosure"
    for path in cards.glob("*.md"):
        package["files"][str(path.relative_to(workspace))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    package_path.write_text(json.dumps(package, indent=2))
    print(
        json.dumps(
            {
                "categories": len(groups),
                "entries": len(entries),
                "initial_prompt_bytes": len(content),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    add_navigation(parser.parse_args().run)


if __name__ == "__main__":
    main()
