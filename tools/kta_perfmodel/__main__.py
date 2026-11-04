"""``scripts/perfmodel-table``: replay the validation kernels and probes on CPU and write the error
table (JSON + Markdown) to docs/design/perfmodel/.

    scripts/perfmodel-table [--sets gemm tma probes] [--out DIR] [--no-write]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from . import table, upstream

OUT = upstream.ROOT / "docs" / "design" / "perfmodel"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/perfmodel-table")
    ap.add_argument("--sets", nargs="+", choices=table.SETS, default=list(table.SETS))
    ap.add_argument("--upstream", type=Path, default=upstream.UPSTREAM)
    ap.add_argument("--rev", default=upstream.UPSTREAM_REV)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-write", action="store_true", help="print only")
    ap.add_argument("--jobs", type=int, default=table.default_jobs(), help="worker processes")
    ap.add_argument("--no-cache", action="store_true", help="replay every row")
    ap.add_argument("--all", action="store_true", help="also the slow rows (table.SLOW)")
    ap.add_argument("--top", action="store_true", help="print the scorecard summary and top rows")
    ap.add_argument(
        "--cached-only", action="store_true", help="fail instead of replaying an uncached row"
    )
    ap.add_argument(
        "--override",
        default=None,
        help="JSON {param: value} applied to every row (an ablation: printed, not written)",
    )
    a = ap.parse_args(argv)
    over = json.loads(a.override) if a.override else None
    if over:
        a.no_write = True
    cal, sha = upstream.calibration_dir(a.upstream, a.rev)
    head, dirty = upstream.kta_commit()
    t0 = time.perf_counter()
    rows = table.specs(cal, tuple(a.sets))
    prev = a.out / "error-table.json"
    hint = (
        {r["id"]: r.get("wall_s", 0.0) for r in json.loads(prev.read_text())["rows"]}
        if prev.exists()
        else {}
    )
    for r in table.run(cal, sha, rows, a.jobs, not a.no_cache, hint, a.all, a.cached_only, over):
        err = "" if r["error"] is None else f"{100 * r['error']:+7.1f} %"
        if a.top:
            continue
        print(
            f"{r['id']:48s} {r['measured']:12.2f} {table._fmt(r['predicted']):>12s} {err:>9s} "
            f"{r['wall_s']:6.1f} s {r['excluded'] or ''}",
            flush=True,
        )
    # the "mechanisms only" column (LOOP §4.4): the same rows with every fitted parameter replaced
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
        "jobs": a.jobs,
        "cached_rows": sum(r["cached"] for r in rows),
        "rows": [{k: v for k, v in r.items() if k not in ("job", "cached")} for r in rows],
    }
    if a.top:
        print(table.render(doc, top=10).split("## All rows")[0])
    else:
        for s in doc["summary"]:
            print(json.dumps(s))
    if not a.no_write and not a.top:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "error-table.json").write_text(json.dumps(doc, indent=1) + "\n")
        (a.out / "error-table.md").write_text(table.render(doc))
        print(f"wrote {a.out / 'error-table.json'} and error-table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
