"""Freeze controlled-replay knowledge treatments and public run packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from kta.core.schemas.knowledge import Template
from kta.core.schemas.proof import ProofOutline

from .proposal_guides import proposal_guide, supplemental


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_json(path: Path, body: Any) -> None:
    if path.exists() and json.loads(path.read_text()) == body:
        return
    write_new(path, json.dumps(body, indent=2) + "\n")


def inventory(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink outside public allowlist: {path}")
        if path.is_file():
            result[str(path.relative_to(directory))] = digest(path)
    return result


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected regular file: {source}")
    write_new(destination, source.read_text())


def receipt(root: Path, stage: str, artifacts: list[Path], checks: list[str]) -> Path:
    path = root / "preparation" / f"{stage}.json"
    write_json(
        path,
        {
            "stage": stage,
            "artifacts": [{"path": str(p.resolve()), "sha256": digest(p)} for p in artifacts],
            "checks": [
                {"name": name, "status": "pass", "evidence": [str(p.resolve()) for p in artifacts]}
                for name in checks
            ],
        },
    )
    return path


def render_catalog(cards: list[dict]) -> str:
    lines = [
        "# Frozen optimization-rule catalog",
        "",
        "Select an explicit identity, then read its "
        "guide and structured rule. Paired outlines remain proposed; entries labelled "
        "source_proposal_missing_outline have no outline and are experimental exposures. "
        "Category names are retrieval cues. Entries are sorted by identity, not application order.",
        "",
    ]
    for card in sorted(cards, key=lambda c: c["identity"]):
        lines.extend(
            [
                f"## [{card['identity']}]({card['guide']})",
                "",
                f"Status: {card['status']}. When: {card['when']}",
                "",
                f"Choice: {card['choice']}",
                "",
                f"Boundary: {card['boundary']}",
                "",
            ]
        )
    return "\n".join(lines)


def freeze_library(repo: Path, root: Path) -> Path:
    exposure = json.loads(
        (repo / "docs/runbooks/learndemo-1/controlled-replay/template-inputs.json").read_text()
    )
    source = repo / "knowledge/library"
    destination = root / "preparation/library"
    catalog = json.loads((source / "catalog.json").read_text())
    cards = catalog["entries"]
    assert len(cards) == 312
    for card in cards:
        directory = Path(card["guide"]).parent
        for name in ("GUIDE.md", "template.yaml", "proof.yaml"):
            original = source / directory / name
            content = original.read_text()
            if card["identity"] == "masked_tile_pruning@2" and name == "GUIDE.md":
                content = content.replace(
                    ", [reduction_partition@2](../../reduction_partition/2/GUIDE.md)",
                    " (the separately retained version2 is outside the prescribed subset)",
                )
                staged = destination / directory / name
                if staged.exists() and staged.read_text() == original.read_text():
                    if (root / "preparation/P3.json").exists():
                        raise ValueError("Cannot change a receipted library")
                    staged.write_text(content)
            write_new(destination / directory / name, content)
        Template.model_validate(
            yaml.safe_load((destination / directory / "template.yaml").read_text())
        )
        outline = ProofOutline.model_validate(
            yaml.safe_load((destination / directory / "proof.yaml").read_text())
        )
        assert outline.template == card["identity"]
    details = supplemental()
    for item in exposure["entries"]:
        key = item["identity"]
        name, version = key.split("@")
        directory = Path("templates") / name / version
        if item["status"] == "canonical_pair":
            for filename, expected in item["files"].items():
                assert digest(source / directory / filename) == expected, (key, filename)
            continue
        proposal = repo / item["source"]
        assert digest(proposal) == item["files"]["template.yaml"]
        body = yaml.safe_load(proposal.read_text())
        Template.model_validate(body)
        copy_file(proposal, destination / directory / "template.yaml")
        write_new(destination / directory / "GUIDE.md", proposal_guide(body, details[key]))
        assert not (destination / directory / "proof.yaml").exists()
        cards.append(
            {
                "identity": key,
                "name": name,
                "version": int(version),
                "title": body["object"]["type"],
                "category": body["refines"],
                "when": details[key]["when"],
                "choice": body["signature"]["algorithm"],
                "boundary": details[key]["boundary"],
                "guide": str(directory / "GUIDE.md"),
                "template": str(directory / "template.yaml"),
                "proof": None,
                "status": "source_proposal_missing_outline",
            }
        )
    assert len(cards) == len({c["identity"] for c in cards}) == 320
    write_json(destination / "catalog.json", {"format_version": 1, "entries": cards})
    write_new(destination / "catalog.md", render_catalog(cards))
    write_new(
        destination / "README.md",
        "# Frozen optimization rules\n\nRead catalog.md, select "
        "versions, then read guides and structured rules. Establish scope and bindings "
        "against TASK.md, preserve numerical and lifetime invariants, and record selected "
        "identities and self-supplied choices in DESIGN.md. Eight labelled source proposals "
        "lack outlines. Existing outlines establish no new assurance. The menu is unordered "
        "with respect to application; composition and implementation are your responsibility.\n",
    )
    subset = sorted(exposure["canonical_subset"] + exposure["proposal_subset"])
    write_json(
        root / "preparation/subset18.json",
        {
            "identities": subset,
            "canonical_pairs": 10,
            "source_proposals_without_outline": 8,
            "ordering": "lexical only; no original composition or tuning choices",
        },
    )
    altered = destination / "templates/masked_tile_pruning/2/GUIDE.md"
    original = source / "templates/masked_tile_pruning/2/GUIDE.md"
    write_json(
        root / "preparation/library-overlays.json",
        [
            {
                "identity": "masked_tile_pruning@2",
                "file": "GUIDE.md",
                "source": str(original),
                "source_sha256": digest(original),
                "frozen_sha256": digest(altered),
                "reason": "Remove hyperlink to an unselected "
                "related version from both P/S; selected18, contracts and obligations unchanged.",
            }
        ],
    )
    write_json(root / "preparation/library-inventory.json", inventory(destination))
    return destination


def main() -> None:
    import argparse

    from .packages import audit_packages, build_packages, prepare_common
    from .source_map import reconciliation, render

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["library", "common", "packages", "audit"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    if args.stage == "library":
        library = freeze_library(repo, root)
        mapping = reconciliation(repo)
        write_json(root / "preparation/source-template-map.json", mapping)
        write_new(root / "preparation/source-template-map.md", render(mapping))
        print(library)
    elif args.stage == "common":
        print(prepare_common(repo, root))
    elif args.stage == "packages":
        print(build_packages(repo, root))
    else:
        result = audit_packages(root)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in [
                        "status",
                        "P_shared_entries",
                        "S_entries",
                        "N_entries",
                        "all_public_links_checked",
                    ]
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
