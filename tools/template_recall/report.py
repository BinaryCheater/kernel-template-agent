"""Print the session tables used in the results note from a score report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

STAGES = (
    "selected",
    "opened_not_selected",
    "surfaced_not_opened",
    "not_surfaced",
    "absent_from_view",
)

HEADER = "\n| {} | Arm | Surfaced | Opened | Selected | Absent |\n|---|---|---:|---:|---:|---:|"


def funnel(rows: list[dict], scoring: str) -> str:
    counts = Counter(m[scoring] for r in rows for m in r["moves"])
    in_view = sum(counts[s] for s in STAGES[:4])
    selected = counts["selected"]
    opened = selected + counts["opened_not_selected"]
    surfaced = opened + counts["surfaced_not_opened"]
    return (
        f"{surfaced}/{in_view} | {opened}/{in_view} | {selected}/{in_view} | "
        f"{counts['absent_from_view']}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    report = json.loads(p.parse_args().report.read_text())
    runs = report["runs"]
    print("| Arm | Scoring | Sessions | Surfaced | Opened | Selected | Absent from view |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for arm in ("access", "flat"):
        rows = [r for r in runs if r["arm"] == arm]
        for scoring in ("strict", "lenient"):
            print(f"| {arm} | {scoring} | {len(rows)} | {funnel(rows, scoring)} |")
    print(HEADER.format("Task"))
    for task in sorted({r["task"] for r in runs}):
        for arm in ("access", "flat"):
            rows = [r for r in runs if r["task"] == task and r["arm"] == arm]
            print(f"| {task} | {arm} | {funnel(rows, 'lenient')} |")
    print(HEADER.format("Role"))
    for role in ("distinctive", "generic"):
        for arm in ("access", "flat"):
            rows = [
                {"moves": [m for m in r["moves"] if m["role"] == role]}
                for r in runs
                if r["arm"] == arm
            ]
            print(f"| {role} | {arm} | {funnel(rows, 'lenient')} |")
    print(
        print(
            "\n| Run | Exit | Wall s | Input | Cached | Output | Surfaced | Opened "
            "| Selected | Layers | Outside | Bypass |"
        )
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for r in runs:
        u = r["usage"]
        c = r["counts"]
        a = r["audit"]
        print(
            f"| {r['run']} | {r['exit']} | {r['wall_s']:.0f} | {u['input']} | "
            f"{u['cached_input']} | {u['output']} | {c['surfaced']} | {c['opened']} | "
            f"{c['selected']} | {len(a['outside_workspace'])} | {len(a['bundle_bypass'])} |"
        )
    print("\nPer move, lenient (stage counts over sessions):")
    table: dict = defaultdict(Counter)
    for r in runs:
        for m in r["moves"]:
            table[(r["task"], m["move"], m["role"], r["arm"])][m["lenient"]] += 1
    for key, counts in sorted(table.items()):
        print(key, dict(counts))
    print("\nNon-gold selected, by frequency:")
    print(Counter(i for r in runs for i in r["selected"]).most_common(40))
    print("\nNeeds:")
    for r in runs:
        for n in r["needs"]:
            print(r["run"], str(n)[:240].replace("\n", " "))
    for r in runs:
        if r["audit"]["outside_workspace"] or r["audit"]["bundle_bypass"]:
            print(r["run"], r["audit"]["outside_workspace"][:3], r["audit"]["bundle_bypass"][:3])


if __name__ == "__main__":
    main()
