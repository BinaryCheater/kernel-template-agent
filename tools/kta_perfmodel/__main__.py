"""``scripts/perfmodel-table``: replay validation kernels and probes on CPU.

    scripts/perfmodel-table [--top]                    # the sentinels
    scripts/perfmodel-table --focus REGEX [--sentinels]  # the rows REGEX matches, no attribution
    scripts/perfmodel-table --override '{"sim.x": 1}'  # an ablation, on the same selection
    scripts/perfmodel-table --full                     # once per round, on spark; writes the table

Sentinels show that something moved; only the full table scores the round. Every run appends rows,
replays, CPU seconds and wall time to perfmodel/table-runs.jsonl. Local replays run at nice 19 on
at most table.LOCAL_JOBS processes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from . import full, sentinels, table, upstream

OUT = upstream.ROOT / "docs" / "design" / "perfmodel"
LAST = upstream.ROOT / "data" / "perfmodel" / "last-selection.json"


def select(rows: list[dict], focus: str | None, with_sentinels: bool) -> list[dict]:
    keep = sentinels.read() if with_sentinels or not focus else set()
    return [r for r in rows if r["id"] in keep or (focus and re.search(focus, r["id"]))]


def moved(rows: list[dict]) -> None:
    """Rows whose prediction differs from the last selection run's."""
    old = json.loads(LAST.read_text()) if LAST.exists() else {}
    new = {r["id"]: r["predicted"] for r in rows}
    diff = [(new[i] / old[i] - 1, i) for i in new if new[i] and old.get(i) and new[i] != old[i]]
    print(f"{len(diff)} of {len(new)} rows moved from the last run")
    for d, i in sorted(diff, key=lambda x: -abs(x[0]))[:40]:
        print(f"  {i:52s} {100 * d:+7.2f} %")
    LAST.write_text(json.dumps({**old, **new}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/perfmodel-table")
    ap.add_argument("--sets", nargs="+", choices=table.SETS, default=list(table.SETS))
    ap.add_argument("--focus", default=None, help="regex over row ids: only these rows")
    ap.add_argument("--sentinels", action="store_true", help="add the sentinels to --focus")
    ap.add_argument(
        "--wide", action="store_true", help=f"past the {table.LOCAL_CAP_S:.0f} CPU s cap; logged"
    )
    ap.add_argument("--full", action="store_true", help="every row, once per round, on spark")
    ap.add_argument("--full-here", action="store_true", help="what --full runs on the worker")
    ap.add_argument("--upstream", type=Path, default=upstream.UPSTREAM)
    ap.add_argument("--rev", default=upstream.UPSTREAM_REV)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--jobs", type=int, default=table.default_jobs(), help="worker processes")
    ap.add_argument("--no-cache", action="store_true", help="replay every row")
    ap.add_argument("--all", action="store_true", help="also the slow rows (table.SLOW)")
    ap.add_argument("--top", action="store_true", help="print the scorecard summary and top rows")
    ap.add_argument(
        "--cached-only", action="store_true", help="fail instead of replaying an uncached row"
    )
    ap.add_argument(
        "--override", default=None, help="JSON {param: value} applied to every row (an ablation)"
    )
    ap.add_argument("--trace", action="store_true", help="attribution: replays twice as slow")
    a = ap.parse_args(argv)
    if a.full:
        full.on_spark(a.out)
        return 0
    if a.full_here:
        full.refuse_second()
    over = json.loads(a.override) if a.override else None
    trace = a.full_here or (a.trace and not over)
    cal, sha = upstream.calibration_dir(a.upstream, a.rev)
    head, dirty = upstream.kta_commit()
    t0 = time.perf_counter()
    rows = table.specs(cal, tuple(a.sets))
    if not a.full_here:
        rows = select(rows, a.focus, a.sentinels)
    prev = a.out / "error-table.json"
    hint = (
        {r["id"]: r.get("wall_s", 0.0) for r in json.loads(prev.read_text())["rows"]}
        if prev.exists()
        else {}
    )
    for r in table.run(
        cal,
        sha,
        rows,
        a.jobs,
        not a.no_cache,
        hint,
        a.all,
        a.cached_only,
        over,
        trace=trace,
        cap_s=None if a.full_here or a.wide else table.LOCAL_CAP_S,
    ):
        if a.full_here:
            err = "" if r["error"] is None else f"{100 * r['error']:+7.1f} %"
            print(
                f"{r['id']:48s} {r['measured']:12.2f} {table._fmt(r['predicted']):>12s} "
                f"{err:>9s} {r['wall_s']:6.1f} s {r['excluded'] or ''}",
                flush=True,
            )
    # the "mechanisms only" column (LOOP §4.4): the same rows with every fitted parameter replaced;
    # it replays every row that reads one, so only the full table computes it
    mech = {}
    if a.full_here:
        mech = {
            r["id"]: r
            for r in table.run(
                cal,
                sha,
                [dict(r) for r in rows],
                a.jobs,
                not a.no_cache,
                hint,
                a.all,
                a.cached_only,
                {**table.MECH_ONLY, **(over or {})},
                trace=False,
            )
        }
    for r in rows:
        m = mech.get(r["id"])
        r["predicted_mech"] = m["predicted"] if m else None
        r["error_mech"] = m["error"] if m else None
    first = table.first_scores()
    for r in rows:
        f = first.get(r["id"])
        if f:
            r.update(blind=f["error"], blind_batch=f["batch"], blind_predicted=f["predicted"])
    latest_batch = max((f["batch"] for f in first.values()), default=None, key=table.batch_order)
    doc = {
        "kta_commit": head,
        "kta_dirty": dirty,
        "upstream": str(a.upstream),
        "upstream_rev": sha,
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sets": a.sets,
        "wall_s": round(time.perf_counter() - t0, 1),
        "summary": table.summary(rows),
        "blind_batch": latest_batch,
        "blind": table.blind_summary(rows, latest_batch) if latest_batch else [],
        "whatif": (wi := table.whatif_pairs(rows)),
        "whatif_summary": table.whatif_summary(wi),
        "bottlenecks": table.bottlenecks(rows, wi),
        "dynamic_ops": table.dynamic_weights(rows),
        "units": table.unit_compare(rows),
        "convergence": table.convergence(rows),
        "jobs": a.jobs,
        "cached_rows": sum(r["cached"] for r in rows),
        "rows": [{k: v for k, v in r.items() if k not in ("job", "cached")} for r in rows],
    }
    mode = "full" if a.full_here else "override" if over else "selection"
    full.log(
        {
            "mode": mode,
            "commit": head[:12],
            "dirty": dirty,
            "rows": len(rows),
            "replayed": table.STATS["replayed"],
            "cpu_s": round(table.STATS["cpu_s"], 1),
            "wall_s": doc["wall_s"],
            "jobs": a.jobs,
        }
    )
    if over:  # an ablation is read here: it writes no table
        for r in rows:
            err = "" if r["error"] is None else f"{100 * r['error']:+7.1f} %"
            print(f"  {r['id']:52s} {table._fmt(r['predicted']):>12s} {err}")
    n = full.current_round()
    spent = sum(r["cpu_s"] for r in full.runs(n) if r["mode"] != "full")
    print(f"this run {table.STATS['cpu_s']:.0f} CPU s; round {n} so far {spent:.0f} below the full")
    if a.full_here:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "error-table.json").write_text(json.dumps(doc, indent=1) + "\n")
        (a.out / "error-table.md").write_text(table.render(doc))
        print(f"wrote {a.out / 'error-table.json'} and error-table.md")
        return 0
    if not over:
        moved(rows)
    if a.top:
        print(table.render(doc, top=10).split("## All rows")[0])
    else:
        for s in doc["summary"]:
            if s["n"] >= 3:
                print(table.summary_line(s))
    print(
        f"{doc['wall_s']:.0f} s, {table.STATS['replayed']} replays, "
        f"{table.STATS['cpu_s']:.0f} CPU s, {doc['cached_rows']} of {len(rows)} rows cached"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
