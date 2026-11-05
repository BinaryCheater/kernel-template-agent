"""Build and compare arm prototypes and nine fresh public workspaces."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .package import copy_file, inventory, receipt, render_catalog, write_json, write_new
from .public import CLAUSES, COMMON_PROMPT, EVALUATE, INTERFACE, TASK


def prepare_common(repo: Path, root: Path) -> Path:
    campaign = json.loads(
        (repo / "docs/runbooks/learndemo-1/controlled-replay/campaign.json").read_text()
    )
    common = root / "preparation/common-public"
    write_new(common / "TASK.md", TASK)
    write_new(common / "INTERFACE.md", INTERFACE)
    write_new(common / "COMMON_PROMPT.txt", COMMON_PROMPT)
    write_new(common / "harness/evaluate", EVALUATE)
    (common / "harness/evaluate").chmod(0o755)
    write_json(root / "preparation/schedule.json", {"schedule": campaign["schedule"]})
    return common


def check_links(directory: Path) -> int:
    count = 0
    for path in directory.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            if not (path.parent / target.split("#")[0]).exists():
                raise ValueError(f"Broken public link: {path}: {target}")
            count += 1
    return count


def build_packages(repo: Path, root: Path) -> Path:
    common = prepare_common(repo, root)
    if not (common / "ENVIRONMENT.md").is_file():
        raise ValueError("Pinned environment document must be staged before public package freeze")
    docs = inventory(common / "docs")
    examples = inventory(common / "examples")
    if not docs or not examples:
        raise ValueError("Stage pinned public documentation and unrelated language example first")
    common_inventory = inventory(common)
    write_json(
        root / "preparation/public-doc-allowlist.json",
        {
            "scope": "same files in all arms; no target attention implementations",
            "docs": docs,
            "examples": examples,
            "inspection": "Only these references and the fixed compile/API recipe are allowed.",
        },
    )
    write_json(root / "preparation/common-public-inventory.json", common_inventory)
    library = root / "preparation/library"
    cards = json.loads((library / "catalog.json").read_text())["entries"]
    subset = set(json.loads((root / "preparation/subset18.json").read_text())["identities"])
    assert len(cards) == 320 and len(subset) == 18
    for arm in CLAUSES:
        package = root / "packages" / arm
        for relative in common_inventory:
            if relative == "COMMON_PROMPT.txt":
                continue
            copy_file(common / relative, package / relative)
        (package / "harness/evaluate").chmod(0o755)
        write_new(
            package / "PROMPT.md", COMMON_PROMPT + ("\n" + CLAUSES[arm] if CLAUSES[arm] else "")
        )
        chosen = cards if arm == "self-selection" else [c for c in cards if c["identity"] in subset]
        chosen = sorted(chosen, key=lambda card: card["identity"])
        if arm != "no-template":
            for card in chosen:
                for field in ("guide", "template", "proof"):
                    if card.get(field):
                        copy_file(library / card[field], package / "library" / card[field])
            copy_file(library / "README.md", package / "library/README.md")
            write_json(package / "library/catalog.json", {"format_version": 1, "entries": chosen})
            write_new(package / "library/catalog.md", render_catalog(chosen))
        elif (package / "library").exists():
            raise ValueError("No-Template package contains a library")
    campaign = json.loads(
        (repo / "docs/runbooks/learndemo-1/controlled-replay/campaign.json").read_text()
    )
    for slot in campaign["schedule"]:
        prototype = root / "packages" / slot["arm"]
        workspace = root / "slots" / slot["slot"] / "workspace"
        for relative in inventory(prototype):
            copy_file(prototype / relative, workspace / relative)
        (workspace / "harness/evaluate").chmod(0o755)
    report = audit_packages(root)
    output = root / "preparation/package-diff.json"
    write_json(output, report)
    readable = "# Public package and prompt comparison\n\n"
    readable += "All nine public workspaces were constructed before implementation launch. "
    readable += "Common task, environment, docs, unrelated example, evaluator wrapper and feedback "
    readable += "instructions are byte-identical. Only the arm-specific prompt suffix and library "
    readable += (
        "exposure differ. Slot identity and runtime settings remain outside each workspace.\n\n"
    )
    readable += "| Arm | Entries | Paired outlines | Unoutlined source proposals | Slots |\n"
    readable += "|---|---:|---:|---:|---:|\n"
    readable += (
        "| Prescribed |18|10|8|3|\n| Self-selection |320|312|8|3|\n| No Template |0|0|0|3|\n\n"
    )
    readable += "All18 prescribed entries' guide/rule/available-outline files equal their "
    readable += (
        "self-selection copies. No source map, study trace, evaluator code, original source, "
    )
    readable += "private inputs, previous candidate or compiled artifact enters a public package. "
    readable += "The eight proposals have no proof.yaml. Catalog ordering is lexical, not the "
    readable += (
        "original composition. Navigation overlay provenance is in library-overlays.json.\n\n"
    )
    readable += "Public prompt differences:\n\n"
    for arm in CLAUSES:
        readable += f"- {arm}: {CLAUSES[arm].strip() or 'No appended treatment clause.'}\n"
    readable += "\nCommon prompt: COMMON_PROMPT.txt. Exact per-file checks, public allowlist, "
    readable += "package inventories and links checked are retained in package-diff.json and "
    readable += "common-public-inventory.json. Native full-access isolation is instruction-only; "
    readable += "P6 verifies homes, caches, persistence and command audit before launch.\n"
    write_new(root / "preparation/package-diff.md", readable)
    receipt(
        root,
        "P5",
        [
            output,
            root / "preparation/package-diff.md",
            root / "preparation/common-public-inventory.json",
            root / "preparation/public-doc-allowlist.json",
        ],
        [
            "common-file byte comparison across arms and all nine slots",
            "P-S subset exact guide/rule/available-outline bytes",
            "N-zero library, Template instructions, source, evaluator and previous answers",
            "common prompt exact; only declared treatment suffix differs",
            "private evaluator/source material excluded by construction allowlist",
        ],
    )
    return output


def audit_packages(root: Path) -> dict:
    prototypes = {arm: inventory(root / "packages" / arm) for arm in CLAUSES}
    common = {k: v for k, v in prototypes["no-template"].items() if k != "PROMPT.md"}
    expected_common = inventory(root / "preparation/common-public")
    expected_common.pop("COMMON_PROMPT.txt")
    assert common == expected_common
    checks = []
    all_links = 0
    for arm, files in prototypes.items():
        package = root / "packages" / arm
        assert {
            k: v for k, v in files.items() if not k.startswith("library/") and k != "PROMPT.md"
        } == common
        assert (package / "PROMPT.md").read_text() == COMMON_PROMPT + (
            "\n" + CLAUSES[arm] if CLAUSES[arm] else ""
        )
        assert not any(
            "source-template-map" in k or "evaluator" in k or "study" in k for k in files
        )
        all_links += check_links(package)
        checks.append({"arm": arm, "common_files_equal": True, "inventory": files})
    selected = json.loads((root / "packages/prescribed/library/catalog.json").read_text())[
        "entries"
    ]
    full = json.loads((root / "packages/self-selection/library/catalog.json").read_text())[
        "entries"
    ]
    assert len(selected) == 18 and len(full) == 320
    assert sum(c.get("proof") is not None for c in selected) == 10
    assert sum(c.get("proof") is not None for c in full) == 312
    for card in selected:
        for field in ("guide", "template", "proof"):
            if card.get(field):
                relative = "library/" + card[field]
                assert prototypes["prescribed"][relative] == prototypes["self-selection"][relative]
    slots = []
    for slot in json.loads((root / "preparation/schedule.json").read_text())["schedule"]:
        workspace = root / "slots" / slot["slot"] / "workspace"
        assert inventory(workspace) == prototypes[slot["arm"]]
        slots.append({**slot, "workspace": str(workspace), "matches_prototype": True})
    return {
        "status": "pass",
        "common_files": common,
        "prompt_differences": CLAUSES,
        "arm_comparisons": checks,
        "slots": slots,
        "P_shared_entries": 18,
        "S_entries": 320,
        "N_entries": 0,
        "all_public_links_checked": all_links,
        "exposure_isolation": "file allowlist plus instructions; no OS read sandbox",
    }
