"""``scripts/perfmodel-whatif``: rerun one error-table row with machine or program parameters
changed and report the delta (docs/design/27 §3 item 5).

    scripts/perfmodel-whatif ROW [--set KEY=VALUE ...] [--json]

ROW is a row id of the error table (e.g. "gemm1024 c128x128x32_st3_th256 full"). KEY is a machine
parameter (`sim.<key>` or a top-level key of gb10.json) or a program input of a whole-kernel row
(`h.<key>` of its tile config, `regs`); the SASS stays the compiled one, so a program change moves
only the model inputs derived from it. VALUE is read as JSON (numbers, lists), else as a string.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from kta.models.gb10sim.machine import default_machine

from . import table, upstream


def _value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def whatif(row_id: str, changes: dict[str, Any]) -> dict[str, Any]:
    cal, _ = upstream.calibration_dir()
    spec = next((r for r in table.specs(cal) if r["id"] == row_id), None)
    if spec is None:
        raise SystemExit(f"no row {row_id!r}")
    program = {k: v for k, v in changes.items() if k.startswith("h.") or k == "regs"}
    machine = {k: v for k, v in changes.items() if k not in program}
    base = table.run_job(cal, spec["job"])
    new = table.run_job(cal, spec["job"], default_machine().with_overrides(machine), program)
    # attribution in absolute terms (share x predicted), so the delta per mechanism adds up
    acc: dict[tuple[str, str], list[float]] = {}
    for i, res in enumerate((base, new)):
        for d in res["attribution"]:
            acc.setdefault((d["resource"], d["mechanism"]), [0.0, 0.0])[i] = (
                d["share"] * res["predicted"]
            )
    moved = sorted(
        (
            {"resource": r, "mechanism": m, "before": a, "after": b}
            for (r, m), (a, b) in acc.items()
        ),
        key=lambda d: -abs(d["after"] - d["before"]),
    )
    return {
        "row": row_id,
        "unit": spec["unit"],
        "measured": spec["measured"],
        "changes": changes,
        "before": base["predicted"],
        "after": new["predicted"],
        "delta": new["predicted"] / base["predicted"] - 1,
        "on_path": moved[:8],
        "ledger_after": new["ledger"]["assumed_or_fitted_on_path"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/perfmodel-whatif")
    ap.add_argument("row")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    changes = dict((k, _value(v)) for k, v in (s.split("=", 1) for s in a.set))
    out = whatif(a.row, changes)
    if a.json:
        print(json.dumps(out, indent=1))
        return 0
    print(f"{out['row']}  ({out['unit']}, measured {out['measured']:.2f})")
    print(f"  changes: {out['changes']}")
    print(f"  predicted {out['before']:.3f} -> {out['after']:.3f}  ({100 * out['delta']:+.2f} %)")
    print("  critical path, largest moves (same unit):")
    for d in out["on_path"]:
        print(
            f"    {d['resource']:>12s}  {d['mechanism']:40s} {d['before']:10.3f} -> "
            f"{d['after']:10.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
