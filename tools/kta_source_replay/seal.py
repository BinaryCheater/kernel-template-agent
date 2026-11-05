"""Content-bound prelaunch receipts for the controlled source-replay campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kta.core.errors import ConfigError

REQUIRED = (
    "source",
    "evaluator",
    "packages",
    "common_prompt",
    "subset18",
    "library",
    "cache_settings",
    "settings",
    "schedule",
    "question_chain",
)


def digest(path: Path) -> str:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ConfigError(f"sealed inputs cannot be symlinks: {path}")
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise ConfigError(f"sealed input is absent: {path}")
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ConfigError(f"sealed tree contains a symlink: {path}")
    entries = {
        p.relative_to(path).as_posix(): digest(p) for p in sorted(path.rglob("*")) if not p.is_dir()
    }
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def receipt(path: Path, stage: str) -> dict[str, Any]:
    doc = read(path)
    if doc.get("stage") != stage or not doc.get("artifacts") or not doc.get("checks"):
        raise ConfigError(f"{stage} needs a content-bound receipt, not readiness booleans")
    inventory = {}
    for artifact in doc["artifacts"]:
        target = Path(artifact["path"])
        if not target.is_absolute() or digest(target) != artifact["sha256"]:
            raise ConfigError(f"{stage} artifact identity changed: {target}")
        inventory[str(target)] = artifact["sha256"]
    for check in doc["checks"]:
        evidence = check.get("evidence", [])
        if check.get("status") != "pass" or not check.get("name") or not evidence:
            raise ConfigError(f"{stage} has an unproven check")
        if any(str(Path(e)) not in inventory for e in evidence):
            raise ConfigError(f"{stage} check cites an artifact outside its frozen inventory")
    return doc


def scientific_settings(doc: dict[str, Any], design: dict[str, Any]) -> None:
    for key in (
        "campaign",
        "research_question",
        "source",
        "task",
        "model",
        "arms",
        "repetitions_per_arm",
        "schedule_seed",
        "schedule",
        "implementation_concurrency",
        "budget",
        "validation",
        "timing",
        "infrastructure",
        "outcome",
    ):
        # Readiness receipt links are populated during preparation; science stays fixed.
        left, right = doc.get(key), design.get(key)
        if key == "source":
            left = {k: v for k, v in (left or {}).items() if k != "preflight_receipt"}
            right = {k: v for k, v in (right or {}).items() if k != "preflight_receipt"}
        if left != right:
            raise ConfigError(f"campaign scientific setting changed: {key}")


def verify_chain(path: Path, design: dict[str, Any], receipts: dict[str, Any]) -> None:
    chain = read(path)
    expected = {
        "research_question": design["research_question"],
        "source_revision": design["source"]["revision"],
        "translation_unit": design["source"]["translation_unit"],
        "outputs": ["O", "LSE"],
        "comparator": "unchanged_original",
        "arms": design["arms"],
        "schedule": design["schedule"],
    }
    for key, value in expected.items():
        if chain.get(key) != value:
            raise ConfigError(f"question-to-evidence chain mismatch: {key}")
    links = chain.get("receipts", {})
    for stage in ("P1", "P2", "P3", "P4", "P5", "P6"):
        if links.get(stage) != receipts[stage]["sha256"]:
            raise ConfigError(f"question chain does not bind {stage} evidence")


def verify_exposure(inventory: dict[str, Path], design_path: Path) -> None:
    declared = read(design_path.parent / "template-inputs.json")
    expected = set(declared["canonical_subset"] + declared["proposal_subset"])
    actual = read(inventory["subset18"])["identities"]
    if len(actual) != 18 or set(actual) != expected:
        raise ConfigError("prescribed exposure differs from the declared 18-rule treatment")
    catalog = read(inventory["library"] / "catalog.json")["entries"]
    identities = {e["identity"] for e in catalog}
    if len(catalog) != 320 or len(identities) != 320 or not expected <= identities:
        raise ConfigError("frozen full library is not the declared 320-entry treatment")
    packages = inventory["packages"]
    baseline = None
    common_prompt = inventory["common_prompt"].read_text().rstrip()
    clauses = {
        "prescribed": (
            "Read and use the supplied optimization rules in library/. "
            "Record the rules used and choices supplied from your own knowledge."
        ),
        "self-selection": (
            "Select relevant optimization rules from library/catalog.md, "
            "read and use them, and record the rules used and choices "
            "supplied from your own knowledge."
        ),
        "no-template": "",
    }
    for arm in ("prescribed", "self-selection", "no-template"):
        package = packages / arm
        expected_prompt = common_prompt + ("\n\n" + clauses[arm] if clauses[arm] else "") + "\n"
        if (package / "PROMPT.md").read_text() != expected_prompt:
            raise ConfigError("arm prompt differs from frozen common prompt and declared clause")
        common = {
            p.relative_to(package).as_posix(): digest(p)
            for p in package.rglob("*")
            if p.is_file()
            and p.relative_to(package).parts[0] != "library"
            and p.name != "PROMPT.md"
        }
        if baseline is not None and common != baseline:
            raise ConfigError("common public task/environment/tools differ between arms")
        baseline = common
    if (packages / "no-template/library").exists():
        raise ConfigError("no-Template package exposes a library")
    for p in (packages / "prescribed/library").rglob("*"):
        if p.is_file() and p.name in ("GUIDE.md", "template.yaml", "proof.yaml"):
            other = (
                packages / "self-selection/library" / p.relative_to(packages / "prescribed/library")
            )
            if digest(p) != digest(other):
                raise ConfigError("P/S shared rule bytes differ")


def seal_campaign(
    design_path: Path, root: Path, receipts: dict[str, Path], inventory: dict[str, Path]
) -> Path:
    root = root.resolve()
    target = root / "seal.json"
    if target.exists():
        raise ConfigError("campaign already sealed; preserve it and declare a new revision")
    design = read(design_path)
    if set(receipts) != {f"P{i}" for i in range(1, 8)}:
        raise ConfigError("all P1-P7 receipts are required")
    if not set(REQUIRED).issubset(inventory):
        raise ConfigError(f"missing frozen inventory: {set(REQUIRED) - set(inventory)}")
    frozen = {}
    for stage, path in receipts.items():
        receipt(path, stage)
        frozen[stage] = {"path": str(path.resolve()), "sha256": digest(path)}
    scientific_settings(read(inventory["settings"]), design)
    if read(inventory["schedule"]).get("schedule") != design["schedule"]:
        raise ConfigError("frozen schedule differs from declared balanced schedule")
    verify_chain(inventory["question_chain"], design, frozen)
    verify_exposure(inventory, design_path)
    slots = {}
    for row in design["schedule"]:
        workspace = root / "slots" / row["slot"] / "workspace"
        prototype = inventory["packages"] / row["arm"]
        if digest(workspace) != digest(prototype):
            raise ConfigError(f"slot package differs from sealed prototype: {row['slot']}")
        slots[row["slot"]] = {
            "workspace": str(workspace),
            "sha256": digest(workspace),
            "immutable_files": {
                str(p.relative_to(workspace)): digest(p)
                for p in workspace.rglob("*")
                if p.is_file()
            },
        }
    sealed = {
        "campaign": design["campaign"],
        "launch_allowed": True,
        "design": {"path": str(design_path.resolve()), "sha256": digest(design_path)},
        "receipts": frozen,
        "inventory": {
            k: {"path": str(v.resolve()), "sha256": digest(v)} for k, v in inventory.items()
        },
        "slots": slots,
        "launcher": [
            {"path": str(p), "sha256": digest(p)}
            for p in (
                Path(__file__).resolve(),
                Path(__file__).with_name("session.py").resolve(),
                Path(__file__).resolve().parents[1] / "kta_template_sessions/launch.py",
                Path(__file__).resolve().parents[1] / "kta_template_sessions/adapter.py",
            )
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(sealed, indent=2, sort_keys=True))
    return target


def verify_slot_inputs(entry: dict[str, Any], pristine: bool) -> None:
    workspace = Path(entry["workspace"])
    if pristine and digest(workspace) != entry["sha256"]:
        raise ConfigError("fresh slot package is absent or changed")
    for name, expected in entry["immutable_files"].items():
        if digest(workspace / name) != expected:
            raise ConfigError(f"slot immutable public input changed: {name}")


def verify_seal(root: Path, slot: str | None = None, pristine: bool = True) -> dict[str, Any]:
    path = root / "seal.json"
    if not path.is_file():
        raise ConfigError("no sealed prelaunch evidence; model launch disabled")
    doc = read(path)
    if set(doc.get("receipts", {})) != {f"P{i}" for i in range(1, 8)} or not set(REQUIRED) <= set(
        doc.get("inventory", {})
    ):
        raise ConfigError("sealed campaign lacks required receipt/inventory identities")
    if doc.get("launch_allowed") is not True:
        raise ConfigError("campaign is not launchable")
    for item in [
        doc["design"],
        *doc["inventory"].values(),
        *doc["receipts"].values(),
        *doc.get("launcher", []),
    ]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ConfigError(f"sealed identity changed: {item['path']}")
    design = read(Path(doc["design"]["path"]))
    for stage, item in doc["receipts"].items():
        receipt(Path(item["path"]), stage)
    scientific_settings(read(Path(doc["inventory"]["settings"]["path"])), design)
    verify_chain(Path(doc["inventory"]["question_chain"]["path"]), design, doc["receipts"])
    if slot is not None:
        entry = doc["slots"].get(slot)
        if entry is None:
            raise ConfigError("slot is absent from sealed campaign")
        verify_slot_inputs(entry, pristine)
    return doc
