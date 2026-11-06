"""Build a frozen run package for one session. Gold sets and provenance stay out."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from template_access.build import bundle_bytes
from template_access.store import Library

from .gold import ROOT, Task

ACCESS_PROMPT = """Read TASK.md and .agents/skills/template-access/SKILL.md. You are choosing
optimisation rules for a kernel design. Do not write kernel code, compile or use a GPU.
Use the ./template-access tool for all library reading. Search from several angles,
open what looks relevant, and finish by recording your choices with
./template-access select <identity> --reason "...". Select at most 20 rules that you
would actually compose into a fast implementation. Use ./template-access need "..." for a
design need that no rule satisfies. Stay inside this workspace. Do not read files
outside it, and do not unpack the tool's data files directly. Finish with one short
paragraph on your design."""

FLAT_PROMPT = """Read TASK.md and the library under library/. You are choosing optimisation
rules for a kernel design. Do not write kernel code, compile or use a GPU. Start from
library/catalog.md, read guides and rule files of what looks relevant, and finish by
writing SELECTION.json: a JSON list of objects with fields identity and reason. Select at
most 20 rules that you would actually compose into a fast implementation. Add needs that
no rule satisfies to NEEDS.md. Stay inside this workspace. Do not read files outside it.
Finish with one short paragraph on your design."""

WRAPPER = """#!/bin/sh
cd "$(dirname "$0")"
export KTA_ACCESS_SESSION={session}
exec python3 -m template_access "$@"
"""


def scrub(index: dict, library: Library) -> dict:
    """Keep what the reader needs; origin and paths would reveal the source of each rule."""
    entries = []
    for entry in library.entries:
        row = dict(entry)
        row["origin"] = {"study": "", "lineage": []}
        row["paths"] = {"realisation": entry["paths"]["realisation"]}
        entries.append(row)
    return {**index, "entries": entries, "realisations_without_catalog_entry": []}


def access_arm(workspace: Path, library: Library) -> str:
    target = workspace / "template_access"
    target.mkdir()
    source = ROOT / "tools/template_access"
    for name in ("__init__.py", "__main__.py", "retrieval.py", "store.py", "lexicon.json"):
        shutil.copyfile(source / name, target / name)
    (target / "bundle.json.gz").write_bytes(bundle_bytes(ROOT, library.entries))
    index = json.loads((source / "index.json").read_text())
    (target / "index.json").write_text(json.dumps(scrub(index, library)))
    skill = workspace / ".agents/skills/template-access"
    skill.mkdir(parents=True)
    shutil.copyfile(ROOT / ".agents/skills/template-access/SKILL.md", skill / "SKILL.md")
    wrapper = workspace / "template-access"
    wrapper.write_text(WRAPPER.format(session=workspace.parent.name))
    wrapper.chmod(0o755)
    return ACCESS_PROMPT


def filtered_catalog(text: str, keep: set[str]) -> str:
    """Drop excluded sections from catalog.md and repair its counts; card text is unchanged."""
    head, *sections = re.split(r"(?m)^(?=## |### )", text)
    groups: list[tuple[str, list[str]]] = []
    for section in sections:
        if section.startswith("## "):
            groups.append((section, []))
            continue
        identity = section[len("### [") :].split(":")[0]
        if identity in keep:
            groups[-1][1].append(section)
    out = [head]
    for title, cards in groups:
        if cards:
            out.append(re.sub(r"\(\d+\)", f"({len(cards)})", title, count=1))
            out.extend(cards)
    return "".join(out)


def flat_arm(workspace: Path, library: Library) -> str:
    """The format of the earlier self-selection arm: catalog files plus plain rule files."""
    source = ROOT / "knowledge/library"
    target = workspace / "library"
    keep = set(library.by_identity)
    target.mkdir()
    for identity in sorted(keep):
        name, version = identity.split("@")
        relative = Path("templates") / name / version
        (target / relative).mkdir(parents=True)
        for filename in ("GUIDE.md", "template.yaml", "proof.yaml"):
            shutil.copyfile(source / relative / filename, target / relative / filename)
    catalog = json.loads((source / "catalog.json").read_text())
    catalog["entries"] = [e for e in catalog["entries"] if e["identity"] in keep]
    (target / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (target / "catalog.md").write_text(filtered_catalog((source / "catalog.md").read_text(), keep))
    guides = "\n".join(f"- [{i}](templates/{i.replace('@', '/')}/GUIDE.md)" for i in sorted(keep))
    (target / "README.md").write_text(
        "# Frozen Template library\n\nRead a guide, then its rule and proof outline.\n\n"
        + guides
        + "\n"
    )
    return FLAT_PROMPT


def build_package(task: Task, arm: str, view: str, workspace: Path) -> Path:
    if workspace.exists():
        raise FileExistsError(workspace)
    workspace.mkdir(parents=True)
    (workspace / "TASK.md").write_text(task.statement)
    exclude = task.source_study if view == "loso" else None
    library = Library(exclude)
    prompt = access_arm(workspace, library) if arm == "access" else flat_arm(workspace, library)
    path = workspace.parent / f"{workspace.name}.prompt.md"
    path.write_text(prompt + "\n")
    return path
