"""Derive the access index from the unchanged library. Needs PyYAML; the reader does not."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
LIBRARY = "knowledge/library"

# Seed reference packages predate the study directories. Lineage needs them, because a
# later version under another study can still descend from a rule extracted from that source.
REFERENCE_SOURCE = {
    "cutlass_ampere_tf32_gemm": "cutlass",
    "flash_attention_2_fwd": "flash-attention",
    "flash_attention_3_fwd": "flash-attention",
    "flashinfer_batch_decode": "flashinfer",
    "marlin_int4_gemm": "marlin",
    "vllm_fused_moe": "vllm",
    "vllm_rms_norm": "vllm",
}
EXCLUDED_DIRS = {"inputs", "pool-records", "snapshots", "snapshot", "library"}
CITATION = re.compile(r"[\w./-]+\.\w{1,4}:\d+|^\s*[\w./-]+\.\w{1,4}\b")


def origin_of(source: str) -> str:
    parts = source.split("/")
    if parts[1] == "studies":
        return parts[2]
    if parts[2] == "references":
        return REFERENCE_SOURCE[parts[3]]
    return "seed"


def strings(value: object) -> list[str]:
    """Flatten nested YAML values into their text leaves."""
    if isinstance(value, dict):
        return [s for item in value.values() for s in strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    return [str(value)] if value not in (None, "") else []


def realisations(root: Path) -> dict[str, list[dict]]:
    """Find Realisation records by declared template identity, wherever a study keeps them."""
    found: dict[str, list[dict]] = {}
    for directory, dirs, files in os.walk(root / "knowledge"):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        if Path(directory).name != "realisations":
            continue
        for filename in sorted(files):
            path = Path(directory) / filename
            body = yaml.safe_load(path.read_text())
            found.setdefault(body["template"], []).append(
                {"path": str(path.relative_to(root)), "body": body, "text": path.read_text()}
            )
    return found


def witness_of(rows: list[dict], template: dict) -> str:
    """Name the strongest witness kind; 'probed' means a target was recorded, not a proof."""
    if any(r["body"].get("probed_on") for r in rows) or template.get("probed_on"):
        return "probed"
    if any(r["body"].get("expected_sass") or r["body"].get("code_markers") for r in rows):
        return "static"
    return "example_code" if template.get("example_code") else "none"


def tag(text: str, table: dict) -> list[str]:
    hits = [
        name
        for name, rule in sorted(table.items())
        if any(re.search(p, text, re.IGNORECASE) for p in rule["patterns"])
    ]
    return hits or ["other"]


def guide_when(text: str) -> str:
    match = re.search(r"## When to consider\n\n(.*?)\n\n## ", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def entry_card_text(card: dict) -> list[str]:
    return [card["when"], card["choice"], card["boundary"]]


def layer_gaps(entry: dict, rows: list[dict], other_versions: list[str], witness: str) -> list[str]:
    gaps = []
    if CITATION.search(entry["card"]["boundary"]):
        gaps.append("card.boundary_is_source_citation")
    if entry["card"]["choice"].strip() == entry["card"]["when"].strip():
        gaps.append("card.when_repeats_choice")
    if not entry["guide_curated"]:
        gaps.append("guide.generated_without_curated_binding")
    if not rows:
        gaps.append("realisation.none")
        if other_versions:
            gaps.append("realisation.only_other_version:" + ",".join(other_versions))
    if witness == "none":
        gaps.append("witness.none")
    return gaps


def entry_of(
    card: dict, library: Path, origins: dict, found: dict, curated: dict, lex: dict
) -> dict:
    template_text = (library / card["template"]).read_text()
    template = yaml.safe_load(template_text)
    guide_text = (library / card["guide"]).read_text()
    identity, name = card["identity"], card["name"]
    rows = found.get(identity, [])
    others = sorted(key for key in found if key.split("@")[0] == name and key != identity)
    signature = template.get("signature") or {}
    transformation = {
        "object_type": str(template["object"]["type"]),
        "object": str(template["object"].get("description") or ""),
        "algorithm": str(signature.get("algorithm") or ""),
        "storage": str(signature.get("storage") or ""),
        "program": str((template.get("program") or {}).get("description") or ""),
    }
    resource_text = " ".join(strings(signature.get("resources")) + strings(template.get("effects")))
    summary = str(template.get("summary") or "")
    corpus = " ".join(
        [
            card["when"],
            card["choice"],
            card["boundary"],
            summary,
            *transformation.values(),
            resource_text,
            *strings(template.get("scope", {}).get("text")),
            str(template.get("numeric_risk") or ""),
        ]
    )
    entry = {
        "identity": identity,
        "name": name,
        "version": card["version"],
        "title": card["title"],
        "category": card["category"],
        "status": card["status"],
        "card": {k: card[k] for k in ("when", "choice", "boundary")},
        "level": [template.get("level_from"), template.get("level_to")],
        "pattern": {
            "refines": str(template.get("refines") or ""),
            "produces": [str(p) for p in template.get("produces") or []],
        },
        "transformation": transformation,
        "resource_text": resource_text,
        "summary": summary,
        "guide_when": guide_when(guide_text),
        "guide_curated": identity in curated,
        "facets": {
            # Symptoms read only prose that states a situation; contract text names most
            # concepts in passing, which made every tag match most of the library.
            "symptom": tag(" ".join([*entry_card_text(card), summary]), lex["symptom"]),
            "resource": tag(corpus, lex["resource"]),
        },
        "origin": origins[identity],
        "paths": {
            "guide": f"{LIBRARY}/{card['guide']}",
            "template": f"{LIBRARY}/{card['template']}",
            "proof": f"{LIBRARY}/{card['proof']}",
            "realisation": [r["path"] for r in rows],
        },
    }
    witness = witness_of(rows, template)
    entry["witness"] = witness
    entry["gaps"] = layer_gaps(entry, rows, others, witness)
    return entry


def origins_table(mapping: list[dict]) -> dict:
    """Identity origin plus name lineage, counting withheld proposals of the same name."""
    by_name: dict[str, set[str]] = {}
    for row in mapping:
        by_name.setdefault(row["identity"].split("@")[0], set()).add(origin_of(row["source"]))
    return {
        row["identity"]: {
            "study": origin_of(row["source"]),
            "lineage": sorted(by_name[row["identity"].split("@")[0]]),
        }
        for row in mapping
        if row["disposition"] == "canonical"
    }


def fingerprint(root: Path, entries: list[dict]) -> str:
    digest = hashlib.sha256((root / LIBRARY / "catalog.json").read_bytes())
    for entry in entries:
        for key in ("guide", "template", "proof"):
            digest.update((root / entry["paths"][key]).read_bytes())
        for path in entry["paths"]["realisation"]:
            digest.update((root / path).read_bytes())
    return digest.hexdigest()


def build(root: Path) -> dict:
    library = root / LIBRARY
    lex = json.loads((HERE / "lexicon.json").read_text())
    cards = json.loads((library / "catalog.json").read_text())["entries"]
    mapping = json.loads((library / "mapping.json").read_text())["sources"]
    curated = yaml.safe_load((library / "curation.yaml").read_text())
    origins = origins_table(mapping)
    found = realisations(root)
    entries = [entry_of(c, library, origins, found, curated, lex) for c in cards]
    known = {e["identity"] for e in entries}
    return {
        "format_version": 1,
        "library_fingerprint": fingerprint(root, entries),
        "lexicon": {k: lex[k] for k in ("symptom", "resource")},
        "entries": entries,
        "realisations_without_catalog_entry": sorted(k for k in found if k not in known),
    }


def bundle_bytes(root: Path, entries: list[dict]) -> bytes:
    """Layer text for a frozen package, compressed so reads go through the logged tool."""
    layers = {}
    for entry in entries:
        paths = entry["paths"]
        layers[entry["identity"]] = {
            "guide": (root / paths["guide"]).read_text(),
            "template": (root / paths["template"]).read_text(),
            "proof": (root / paths["proof"]).read_text(),
            "realisation": {p: (root / p).read_text() for p in paths["realisation"]},
        }
    return gzip.compress(json.dumps(layers).encode(), mtime=0)


def write_index() -> dict:
    index = build(HERE.parents[1])
    (HERE / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    return index
