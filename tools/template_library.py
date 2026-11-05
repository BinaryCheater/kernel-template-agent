"""Build and check the file library without modifying source proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from kta.core.schemas.knowledge import Template
from kta.core.schemas.proof import ProofOutline


def discover(root: Path) -> list[dict]:
    """Discover declared rules, pruning retained input and snapshot directories."""
    found = []
    excluded = {"inputs", "pool-records", "snapshots", "snapshot", "library"}
    for directory, dirs, files in os.walk(root / "knowledge"):
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for filename in sorted(files):
            if Path(filename).suffix not in {".yaml", ".yml"}:
                continue
            path = Path(directory) / filename
            body = yaml.safe_load(path.read_text())
            if not isinstance(body, dict) or not all(
                key in body for key in ("name", "refines", "program", "variables")
            ):
                continue
            Template.model_validate(body)
            proof = (
                path.parent / "proof.yaml"
                if filename == "template.yaml"
                else path.parent.parent / "proofs" / filename
            )
            found.append({"path": path, "body": body, "proof": proof})
    return sorted(found, key=lambda row: str(row["path"]))


def identity(body: dict) -> str:
    return f"{body['name']}@{body['version']}"


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact(value: str, limit: int = 300) -> str:
    """Keep cards short; the guide retains the complete contract."""
    sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z])", value.strip())[0]
    return sentence if len(sentence) <= limit else sentence[:limit].rsplit(" ", 1)[0] + "..."


def table_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def guide(body: dict, row: dict, cards: dict, curated: dict) -> str:
    key = identity(body)
    lines = [f"# {key}: {body['object']['type'].replace('_', ' ')}", ""]
    extra = curated.get(key, {})
    lines += ["## When to consider", "", extra.get("when", body["summary"]), ""]
    lines += ["## Computation before and after", "", body["signature"]["algorithm"], ""]
    lines += [
        body["program"]["description"],
        "",
        "Recover the original computation: " + body["backward_map"],
        "",
    ]
    lines += [
        "## Bind the rule",
        "",
        extra.get(
            "binding",
            (
                "1. Bind computation, extents, dtypes and layout to `p.*`; bind "
                "target facts to `hw.*` and workload facts to `case.*`. These are fixed.\n"
                "2. Select values only from the declared variable domains below. A compile binding "
                "is a specialization point; a runtime binding is supplied when executing. Boundary "
                "roles must agree with the adjacent rule or caller.\n"
                "3. Evaluate the scope, every precondition and every constraint. Supply evidence "
                "for semantic booleans; setting an unproved premise to True is insufficient.\n"
                "4. Implement the declared mapping and discharge the obligations for this binding. "
                "Keep numerical acceptance separate from the exact-arithmetic argument."
            ),
        ),
        "",
    ]
    lines += ["| Variable | Binding / role | Declared domain | Meaning |", "|---|---|---|---|"]
    for var in body["variables"]:
        lines.append(
            "| "
            + " | ".join(
                table_cell(x)
                for x in (
                    var["name"],
                    f"{var['binding']} / {var['role']}",
                    json.dumps(var["domain"]),
                    var["meaning"],
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Coupled constraints and scope",
        "",
        body["scope"]["text"],
        "",
        f"Scope predicate: `{body['scope']['predicate']}`.",
        "",
    ]
    for label, field in (("Precondition", "preconditions"), ("Constraint", "constraints")):
        for expr in body[field]:
            lines += [
                f"- {label} `{expr['id']}` ({expr['cls']}): `{expr['expr']}` — {expr['meaning']}"
            ]
    lines += [
        "",
        "## Local example and misuse",
        "",
        extra.get(
            "example",
            (
                f"For a local instance of `{body['refines']}`, identify the object "
                f"`{body['object']['type']}` and replace only that operation by the program above. "
                "Write down one binding for every variable in the table and the caller facts used "
                "by each predicate. This recipe is not a validated backend example. "
                "Reject out-of-scope bindings even if the variable domains accept them."
            ),
        ),
        "",
    ]
    lines += [
        "## Implementations, evidence and extension boundary",
        "",
        extra.get(
            "implementation",
            (
                "The contract above is retained from the proposal. The original source instance "
                "and any implementation-specific choices remain provenance; a declared domain "
                "does not establish that every value has an implemented backend. Before selecting "
                "an instruction or mapping, establish target and compiler support. "
                "Any new tail, numerical policy or mapping outside this contract is an extension "
                "requiring a revised rule/outline and implementation evidence."
            ),
        ),
        "",
        f"Declared capabilities: `{body.get('requires', [])}`. "
        f"Declared probed targets: `{body.get('probed_on', [])}`.",
        "",
        f"Original proposal: `{row['source']}`. "
        f"Original outline: `{row.get('source_proof') or 'consolidation-authored'}`. "
        "Publication here preserves a proposed contract; it does not add proof assurance.",
        "",
    ]
    lines += [
        "## Expected benefit and cost",
        "",
        extra.get(
            "tradeoff",
            (
                f"Reuse: {body['signature'].get('reuse') or 'not specified'}. "
                f"Storage: {body['signature'].get('storage') or 'not specified'}. "
                "Structured effects describe work and resource changes, not timings."
            ),
        ),
        "",
        "```yaml",
        yaml.safe_dump(body.get("effects", {}), sort_keys=False).rstrip(),
        "```",
        "",
        "## Obligations and numerical checks",
        "",
    ]
    for ob in body["obligations"]:
        lines.append(f"- `{ob['name']}` ({ob['method']}): {ob['statement']}")
    lines += [
        "",
        body.get("numeric_risk") or "No numeric risk is declared by this proposal.",
        "",
        "Read [the paired outline](proof.yaml) for assumptions and open checks. "
        "No ProofResult or ProofSummary is created by consolidation.",
        "",
        "## Related rules",
        "",
    ]
    for relation in body.get("relations", []):
        targets = [c for c in cards.values() if c["name"] == relation["target"]]
        links = ", ".join(
            f"[{c['identity']}](../../{c['name']}/{c['version']}/GUIDE.md)" for c in targets
        )
        lines.append(
            f"- {relation['type']}: {links or relation['target'] + ' (outside menu)'}; "
            f"{relation.get('note') or 'check the interface at composition time'}"
        )
    if not body.get("relations"):
        lines.append("No relation is declared; composition still requires matching interfaces.")
    lines += ["", "Structured contract: [template.yaml](template.yaml).", ""]
    return "\n".join(lines)


def build(root: Path) -> dict[Path, str]:
    library = root / "knowledge/library"
    curation = yaml.safe_load((library / "curation.yaml").read_text())
    candidates = discover(root)
    groups = defaultdict(list)
    for candidate in candidates:
        groups[identity(candidate["body"])].append(candidate)
    if set(curation) - set(groups):
        raise ValueError(f"Unknown curated identities: {set(curation) - set(groups)}")
    rows, cards, selected = [], {}, {}
    for key, members in sorted(groups.items()):
        different = len({fingerprint(m["body"]) for m in members}) > 1
        for member in members:
            body, source, proof = member["body"], member["path"], member["proof"]
            row = {
                "identity": key,
                "source": str(source.relative_to(root)),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_proof": str(proof.relative_to(root)) if proof.exists() else None,
            }
            authored = library / "curated-proofs" / f"{key}.yaml"
            if different:
                row.update(
                    disposition="unresolved_conflict",
                    destination=None,
                    reason="Same identity has different contracts; no silent precedence.",
                )
            elif not proof.exists() and not authored.exists():
                row.update(
                    disposition="withheld_incomplete",
                    destination=None,
                    reason="No paired ProofOutline; preserve proposal outside default menu.",
                )
            else:
                chosen_proof = proof if proof.exists() else authored
                outline = yaml.safe_load(chosen_proof.read_text())
                ProofOutline.model_validate(outline)
                obligations = {item["name"] for item in body["obligations"]}
                if any(model["obligation"] not in obligations for model in outline["models"]):
                    raise ValueError(f"Unknown obligation in {chosen_proof}")
                if outline["template"] != key:
                    raise ValueError(f"Outline identity mismatch: {chosen_proof}")
                destination = f"templates/{body['name']}/{body['version']}"
                row.update(
                    disposition="canonical" if key not in selected else "alias",
                    destination=destination,
                    reason="Contract and version retained without semantic merge.",
                    published_proof=str(chosen_proof.relative_to(root)),
                )
                if key not in selected:
                    selected[key] = (member, row, chosen_proof)
                    cards[key] = {
                        "identity": key,
                        "name": body["name"],
                        "version": body["version"],
                        "title": body["object"]["type"].replace("_", " "),
                        "category": body["refines"],
                        "when": compact(curation.get(key, {}).get("when", body["summary"])),
                        "choice": compact(body["signature"]["algorithm"]),
                        "boundary": compact(
                            body["signature"].get("boundary") or body["scope"]["text"]
                        ),
                        "guide": destination + "/GUIDE.md",
                        "template": destination + "/template.yaml",
                        "proof": destination + "/proof.yaml",
                        "status": "proposed",
                    }
            rows.append(row)
    counts = {
        "candidate_files": len(rows),
        "source_identities": len(groups),
        "canonical_entries": len(cards),
        "dispositions": dict(Counter(row["disposition"] for row in rows)),
    }
    outputs = {}
    for member, row, proof in selected.values():
        destination = library / row["destination"]
        outputs[destination / "template.yaml"] = member["path"].read_text()
        outputs[destination / "proof.yaml"] = proof.read_text()
        outputs[destination / "GUIDE.md"] = guide(member["body"], row, cards, curation)
    outputs[library / "mapping.json"] = (
        json.dumps(
            {
                "format_version": 1,
                "discovery_exclusions": [
                    "inputs",
                    "pool-records",
                    "snapshots",
                    "snapshot",
                    "library",
                ],
                "counts": counts,
                "sources": rows,
            },
            indent=2,
        )
        + "\n"
    )
    outputs[library / "catalog.json"] = (
        json.dumps({"format_version": 1, "entries": list(cards.values())}, indent=2) + "\n"
    )
    menu = [
        "# Template catalog",
        "",
        "Generated by `tools/template_library.py`. "
        "Every entry is a proposed contract. Select a version explicitly and read its guide. "
        "Categories are retrieval cues, not applicability restrictions.",
        "",
    ]
    by_category = defaultdict(list)
    for card in cards.values():
        by_category[card["category"]].append(card)
    for category, members in sorted(by_category.items()):
        menu += [f"## {category} ({len(members)})", ""]
        for card in members:
            menu += [
                f"### [{card['identity']}: {card['title']}]({card['guide']})",
                "",
                f"When: {card['when']}",
                "",
                f"Choice: {card['choice']}",
                "",
                f"Boundary: {card['boundary']}",
                "",
            ]
    outputs[library / "catalog.md"] = "\n".join(menu)
    unresolved = [
        "# Withheld candidates",
        "",
        "These source proposals are preserved but "
        "absent from the default catalog. Reconcile conflicts by mapping fields, "
        "domains and obligations in both directions and assigning a fresh version "
        "where contracts differ. Missing outlines require an authored argument "
        "and named open checks before publication.",
        "",
    ]
    for row in rows:
        if row["destination"] is None:
            unresolved += [f"- `{row['identity']}` — `{row['source']}`: {row['reason']}"]
    outputs[library / "unresolved.md"] = "\n".join(unresolved) + "\n"
    return outputs


def verify_links(outputs: dict[Path, str]) -> int:
    """Check relative menu/guide links without fetching sources."""
    links = 0
    for path, content in outputs.items():
        if path.suffix != ".md":
            continue
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if "://" in target or target.startswith("#"):
                continue
            destination = (path.parent / target.split("#")[0]).resolve()
            if destination not in outputs and not destination.exists():
                raise ValueError(f"Broken link in {path}: {target}")
            links += 1
    return links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare, without writing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    outputs = build(root)
    links = verify_links(outputs)
    mismatches = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                mismatches.append(str(path.relative_to(root)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    expected = {
        p for p in outputs if "templates" in p.relative_to(root / "knowledge/library").parts
    }
    actual = set((root / "knowledge/library/templates").rglob("*.*"))
    if actual - expected:
        raise ValueError(f"Unexpected canonical files: {sorted(actual - expected)}")
    if mismatches:
        raise ValueError(f"Generated files differ: {mismatches}")
    mapping = json.loads(outputs[root / "knowledge/library/mapping.json"])
    print(json.dumps(mapping["counts"], indent=2))
    print(f"{'Checked' if args.check else 'Generated'} {len(outputs)} files; {links} links")


if __name__ == "__main__":
    main()
