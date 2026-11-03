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
    a = ap.parse_args(argv)
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
    for r in table.run(cal, sha, rows, a.jobs, not a.no_cache, hint):
        err = "" if r["error"] is None else f"{100 * r['error']:+7.1f} %"
        print(
            f"{r['id']:48s} {r['measured']:12.2f} {r['predicted']:12.2f} {err:>9s} "
            f"{r['wall_s']:6.1f} s {r['excluded'] or ''}",
            flush=True,
        )
    doc = {
        "kta_commit": head,
        "kta_dirty": dirty,
        "upstream": str(a.upstream),
        "upstream_rev": sha,
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sets": a.sets,
        "wall_s": round(time.perf_counter() - t0, 1),
        "summary": table.summary(rows),
        "jobs": a.jobs,
        "cached_rows": sum(r["cached"] for r in rows),
        "rows": [{k: v for k, v in r.items() if k not in ("job", "cached")} for r in rows],
    }
    for s in doc["summary"]:
        print(json.dumps(s))
    if not a.no_write:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "error-table.json").write_text(json.dumps(doc, indent=1) + "\n")
        (a.out / "error-table.md").write_text(table.render(doc))
        print(f"wrote {a.out / 'error-table.json'} and error-table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
