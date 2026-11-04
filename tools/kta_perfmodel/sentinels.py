"""``perfmodel/sentinels.txt``: the rows a default table run replays. Per row group the row that
moved most often across the historic tables in git, cheap rows first on ties; the 2048³ warm group
whole, because ranking is read from it; the rows worst at first score of the hand and stress sets.
It shows that something moved and never replaces the full table: a back-test on 12 tables caught
90 of 90 moved groups in sample, 38 of 64 held out.

    python -m kta_perfmodel.sentinels    # rewrite the list from the table history
"""

from __future__ import annotations

import itertools
import json
import subprocess

from . import table, upstream

FILE = upstream.ROOT / "perfmodel" / "sentinels.txt"
TABLE = "docs/design/perfmodel/error-table.json"
HISTORY = 12
MOVED = 0.005  # relative change of a row's prediction between two tables
MIN_MOVES = 2  # a group whose rows hardly ever moved has no sentinel
COSTLY = ("cvt", "sfu", "spill", "spill_rw", "lds_narrow", "branch")  # 41 % of replay CPU
WORST_FIRST = {"hand_": 5, "stress_": 5}


def read() -> set[str]:
    if not FILE.exists():
        raise SystemExit(f"{FILE} missing: python -m kta_perfmodel.sentinels")
    return {s for line in FILE.read_text().splitlines() if (s := line.split("#")[0].strip())}


def _tables() -> list[dict[str, dict]]:
    log = ["git", "-C", str(upstream.ROOT), "log", "--format=%h", f"-{HISTORY}", "--", TABLE]
    out = []
    for c in subprocess.run(log, check=True, capture_output=True, text=True).stdout.split():
        show = ["git", "-C", str(upstream.ROOT), "show", f"{c}:{TABLE}"]
        doc = json.loads(subprocess.run(show, check=True, capture_output=True, text=True).stdout)
        out.append({r["id"]: r for r in doc["rows"]})
    return out


def choose(rows: list[dict], tables: list[dict[str, dict]]) -> dict[str, str]:
    moved: dict[str, int] = {}
    for new, old in itertools.pairwise(tables):
        for rid, r in new.items():
            a, b = r.get("predicted"), old.get(rid, {}).get("predicted")
            if a and b and abs(a / b - 1) > MOVED:
                moved[rid] = moved.get(rid, 0) + 1
    last = tables[0]
    cost = {r["id"]: r.get("wall_s", 0.0) for r in last.values()}
    live = [r for r in rows if not r["excluded"]]
    out: dict[str, str] = {}
    for group in dict.fromkeys(r["set"] for r in live):
        if group.removesuffix(".cu") in (f"probe {c}" for c in COSTLY):
            continue
        rs = [r for r in live if r["set"] == group]
        pick = max(rs, key=lambda r: (moved.get(r["id"], 0), -cost.get(r["id"], 1e9)))
        if moved.get(pick["id"], 0) >= MIN_MOVES:
            out[pick["id"]] = f"{group}: moved {moved.get(pick['id'], 0)} of {len(tables) - 1}"
    for r in live:
        if r["set"] == "testset hgemm2048" and "rot" not in r["id"] and r["id"].endswith("kernel"):
            out[r["id"]] = "hgemm2048 warm group"
    first = table.first_scores()
    for prefix, n in WORST_FIRST.items():
        hit = [
            r
            for r in live
            if r["id"].startswith(prefix) and r["id"].endswith(" kernel") and r["id"] in first
        ]
        hit.sort(key=lambda r: -abs(first[r["id"]]["error"]))
        out.update((r["id"], f"worst first score of {prefix}*") for r in hit[:n])
    return out


def main() -> None:
    cal, _ = upstream.calibration_dir()
    picked = choose(table.specs(cal), _tables())
    lines = [
        "# rows of a default table run; python -m kta_perfmodel.sentinels rewrites this file",
        *(f"{rid}  # {why}" for rid, why in picked.items()),
    ]
    FILE.write_text("\n".join(lines) + "\n")
    print(f"{len(picked)} rows in {FILE}")


if __name__ == "__main__":
    main()
