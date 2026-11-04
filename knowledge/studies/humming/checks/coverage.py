"Reconcile all entry/header owners and preserve unresolved per-branch coverage explicitly."

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

STUDY = Path(__file__).resolve().parents[1]


def main() -> None:
    inventory = json.loads((STUDY / "checks/inventory.json").read_text())
    text = (STUDY / "trace-families.md").read_text()
    decisions = {}
    for line in text.splitlines():
        match = re.match(r"\| (D\d+) \|", line)
        if match:
            columns = line.split("|")
            disposition = columns[-2].strip().split()[0].rstrip(";")
            assert disposition in ("explained", "partial", "open"), line
            decisions[match[1]] = {"disposition": disposition, "source_row": line}
    assert set(decisions) == {f"D{i:02}" for i in range(1, 54)}
    gap_rows = yaml.safe_load((STUDY / "delta/gaps.yaml").read_text())
    group_gaps = {
        f"G{i:02}": [row["id"] for row in gap_rows if f"G{i:02}" in row["groups"]]
        for i in range(1, 14)
    }
    group_ranges = {
        "G01": (1, 4),
        "G02": (5, 12),
        "G03": (13, 16),
        "G04": (17, 19),
        "G05": (20, 24),
        "G06": (25, 30),
        "G07": (31, 35),
        "G08": (32, 36),
        "G09": (37, 37),
        "G10": (38, 40),
        "G11": (41, 47),
        "G12": (48, 50),
        "G13": (51, 53),
    }
    for row in [*inventory["device_entries"], *inventory["headers"]]:
        assert row["groups"] and all(
            g in group_gaps or g == "shared_infrastructure" for g in row["groups"]
        )
    variants = []
    for row in inventory["variant_anchors"]:
        groups = row["groups"]
        if groups == ["shared_infrastructure"]:
            groups = ["G01", "G02", "G03", "G06", "G07"]
        ids = sorted({gap for group in groups for gap in group_gaps[group]})
        assert ids
        variants.append(
            {
                "source": f"{row['path']}:{row['line']}",
                "condition": row["source"],
                "groups": groups,
                "disposition": "partial",
                "meaning": (
                    "Source branch belongs to the traced mechanism; this anchor has no"
                    " standalone compiled/replay proof. Consult group decision rows."
                ),
                "next_evidence": ids,
            }
        )
    summary = {
        "outcome": "PASS",
        "source_decisions": decisions,
        "counts": {
            status: sum(r["disposition"] == status for r in decisions.values())
            for status in ("explained", "partial", "open")
        },
        "groups": {
            group: {
                "decisions": [f"D{i:02}" for i in range(bounds[0], bounds[1] + 1)],
                "closed_as": "partial source study",
                "next_evidence": group_gaps[group],
            }
            for group, bounds in group_ranges.items()
        },
        "reconciled": {
            "exports": len(inventory["exports"]),
            "builders": len(inventory["builders"]),
            "device_entries": len(inventory["device_entries"]),
            "headers": len(inventory["headers"]),
            "variant_anchors": len(variants),
        },
        "boundary": (
            "Every source owner/decision has a disposition. This is not a perc"
            "entage of proved branches, semantic completeness or GPU support."
        ),
    }
    (STUDY / "checks/coverage-result.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    external = Path(manifest["humming"]["root"]).parent.parent / "checks"
    external.mkdir(exist_ok=True)
    artifact = external / "variant-dispositions.json"
    artifact.write_text(json.dumps(variants, indent=2) + "\n")
    (STUDY / "checks/variant-dispositions.json").write_text(
        json.dumps(
            {
                "path": str(artifact),
                "anchors": len(variants),
                "disposition": (
                    "All partial; per-group next evidence retained; no branch proof credit."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
