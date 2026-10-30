"""WP14 seed migration: knowledge model v2 (D440-D445). Idempotent; safe to re-run.

Run from the repository root:

    uv run python scripts/migrations/WP14_knowledge_v2.py [--seed knowledge/seed] [--check]

Steps (each recorded in ``<seed>/migrations.yaml``):

1. ``WP14-templates``: every Template YAML still in the v1 shape (it has ``requires_hw``)
   gets levels 0-5 mapped to 1-7 (``paper-0to5@1``), ``requires_hw`` (ISA feature names)
   replaced by ``requires`` (abstract Capability names, table ISA_TO_CAPABILITY),
   ``probed_on: []``, and loses ``expected_sass`` and ``program.fragments``; those, plus
   the ISA feature names, move to a Realisation YAML in the sibling ``realisations/``
   directory (backend ``any``). Self-detecting: a file without ``requires_hw`` is skipped.
2. ``WP14-levels``: ``level`` of Patterns and Gaps and of Reconstruction unmatched
   transactions mapped 0-5 -> 1-7. Not self-detecting, so it runs only when the ledger
   does not list it.

Edits are textual so comments and layout of the hand-written YAML stay. ``--check``
reports what would change and exits 1 if anything would.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from kta.core.schemas.knowledge import Gap, Pattern, Realisation, Reconstruction, Template
from kta.core.schemas.levels import LEGACY_MAPPING, map_level

# ISA feature (HwFeature) -> abstract Capability (knowledge/seed/capabilities/).
ISA_TO_CAPABILITY = {
    "ld_st_shared": "managed_scratchpad",
    "cp_async": "async_copy",
    "cp_async_bulk": "bulk_tensor_copy",
    "cp_async_bulk_tensor": "bulk_tensor_copy",
    "mbarrier": "async_transaction_barrier",
    "barrier_cluster": "thread_block_cluster",
    "distributed_shared_memory": "thread_block_cluster",
    "wgmma": "warpgroup_mma",
    "tcgen05": "tensor_memory",
    "ldmatrix": "matrix_fragment_load",
    "shfl_sync": "warp_shuffle",
    "named_barrier": "named_barriers",
    "setmaxnreg": "register_reallocation",
    "fma_f16x2": "packed_half_arith",
    "fma_bf16x2": "packed_half_arith",
    "atom_global_add": "global_atomics",
    "atom_global_add_f32": "global_atomics",
    "griddepcontrol": "programmatic_dependent_launch",
}
# Capabilities a mechanism needs that its v1 ISA list did not name (semantic additions).
EXTRA_REQUIRES = {
    # FA3's TMA multicast: the move is the multicast itself (02 §2 "unprobed mechanisms").
    "cluster_multicast": ["cluster_multicast"],
}
TOP_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:")
LEVEL_LINE = re.compile(r"^(?P<lead>(?:level_from|level_to|level)): (?P<n>\d+)\s*$")
TX_LEVEL = re.compile(r"^(?P<lead>\s*-?\s*level): (?P<n>\d+)\s*$")
FEATURES = Path("src/kta/targets/gb10/features.yaml")


def _since(features: dict[str, str], names: list[str]) -> str | None:
    """Highest ``since`` of the ISA features as an sm number ("90a" -> "90a")."""
    found = [features[n].removeprefix("sm_") for n in names if n in features]
    key = lambda s: (int(re.sub(r"\D", "", s) or 0), s)  # noqa: E731
    return max(found, key=key) if found else None


def _block(lines: list[str], key: str) -> tuple[int, int] | None:
    """[start, end) of the top-level ``key:`` block."""
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            j = i + 1
            while j < len(lines) and not TOP_KEY.match(lines[j]):
                j += 1
            return i, j
    return None


def _drop_fragments(lines: list[str]) -> list[str]:
    span = _block(lines, "program")
    if span is None:
        return lines
    i, j = span
    out = lines[: i + 1]
    k = i + 1
    while k < j:
        line = lines[k]
        if line.strip().startswith("fragments:"):
            indent = len(line) - len(line.lstrip())
            k += 1
            while k < j and lines[k].strip() and len(lines[k]) - len(lines[k].lstrip()) > indent:
                k += 1
            continue
        out.append(line)
        k += 1
    return out + lines[j:]


def _map_levels(text: str, pattern: re.Pattern[str]) -> str:
    def sub(line: str) -> str:
        m = pattern.match(line)
        return f"{m['lead']}: {map_level(int(m['n']))}" if m else line

    return "\n".join(sub(line) for line in text.split("\n"))


def migrate_template(path: Path, features: dict[str, str]) -> Path | None:
    """Rewrite one v1 Template file; returns the Realisation file written, if any."""
    text = path.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(text)
    if "requires_hw" not in data:
        return None
    isa = list(data.get("requires_hw") or [])
    unknown = [n for n in isa if n not in ISA_TO_CAPABILITY]
    if unknown:
        raise SystemExit(f"{path}: no capability mapping for ISA features {unknown}")
    caps = [ISA_TO_CAPABILITY[n] for n in isa] + EXTRA_REQUIRES.get(str(data["name"]), [])
    requires = list(dict.fromkeys(caps))
    sass = data.get("expected_sass") or []
    fragments = (data.get("program") or {}).get("fragments") or {}
    lines = _map_levels(text, LEVEL_LINE).split("\n")
    span = _block(lines, "expected_sass")
    if span is not None:
        del lines[span[0] : span[1]]
    lines = _drop_fragments(lines)
    span = _block(lines, "requires_hw")
    if span is not None:  # flow or block list: replace the whole top-level block
        req = f"requires: [{', '.join(requires)}]"
        lines[span[0] : span[1]] = [req, "probed_on: []"]
    new = "\n".join(lines)
    Template.model_validate(yaml.safe_load(new))
    path.write_text(new, encoding="utf-8")
    if not (sass or fragments or isa):
        return None
    ref = f"{data['name']}@{data['version']}"
    isa_range: dict[str, Any] = {"family": "sm"}
    if (low := _since(features, isa)) is not None:
        isa_range["min"] = low
    body: dict[str, Any] = {
        "template": ref,
        "backend": "any",
        "isa": isa_range,
        "requires": isa,
        "fragments": dict(fragments),
        "expected_sass": sass,
        "probed_on": [],
        "summary": "Migrated by WP14 from the Template's v1 fields requires_hw, "
        "expected_sass and program.fragments (D443); backend-independent.",
    }
    Realisation.model_validate(body)
    out = path.parent.parent / "realisations" / f"{data['name']}.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8")
    return out


def migrate_levels(seed: Path) -> list[Path]:
    changed: list[Path] = []
    groups = [
        ([*seed.glob("patterns/*.yaml"), *seed.glob("references/*/patterns/*.yaml")], LEVEL_LINE),
        (list(seed.glob("references/*/gaps/*.yaml")), LEVEL_LINE),
        (list(seed.glob("references/*/reconstruction.yaml")), TX_LEVEL),
    ]
    for files, pattern in groups:
        for path in sorted(files):
            text = path.read_text(encoding="utf-8")
            new = _map_levels(text, pattern)
            if new != text:
                path.write_text(new, encoding="utf-8")
                changed.append(path)
    for path in changed:
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = {"patterns": Pattern, "gaps": Gap}.get(path.parent.name, Reconstruction)
        model.model_validate(body)
    return changed


def _ledger(seed: Path) -> list[dict[str, Any]]:
    path = seed / "migrations.yaml"
    return list(yaml.safe_load(path.read_text(encoding="utf-8")) or []) if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=Path, default=Path("knowledge/seed"))
    ap.add_argument("--check", action="store_true", help="report pending changes; write nothing")
    args = ap.parse_args()
    seed: Path = args.seed
    ledger = _ledger(seed)
    done = {str(e["id"]) for e in ledger}
    feats = yaml.safe_load(FEATURES.read_text(encoding="utf-8"))["features"]
    features = {str(f["name"]): str(f.get("since", "")) for f in feats}
    templates = sorted(
        [*seed.glob("templates/*.yaml"), *seed.glob("references/*/templates/*.yaml")]
    )
    pending_t = [p for p in templates if "requires_hw" in yaml.safe_load(p.read_text("utf-8"))]
    pending_l = "WP14-levels" not in done
    if args.check:
        print(f"templates to migrate: {len(pending_t)}; levels step pending: {pending_l}")
        return 1 if pending_t or pending_l else 0
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    entries: list[dict[str, Any]] = []
    if pending_t:
        written = [str(r) for p in pending_t if (r := migrate_template(p, features))]
        entries.append(
            {
                "id": "WP14-templates",
                "at": now,
                "level_mapping": LEGACY_MAPPING,
                "files": [str(p) for p in pending_t],
                "realisations_written": written,
            }
        )
    if pending_l:
        changed = migrate_levels(seed)
        entries.append(
            {
                "id": "WP14-levels",
                "at": now,
                "level_mapping": LEGACY_MAPPING,
                "files": [str(p) for p in changed],
            }
        )
    if entries:
        (seed / "migrations.yaml").write_text(
            yaml.safe_dump(ledger + entries, sort_keys=False, width=100), encoding="utf-8"
        )
    print(f"WP14 migration: {', '.join(e['id'] for e in entries) or 'nothing to do'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
