"""``scripts/gate [--quick] <WP>``: the done gate (spec/gate.md)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from kta_gate import delivery, runners, scans, specific
from kta_gate.result import CheckResult

ROOT = Path(__file__).resolve().parents[2]
WP_NAME = "WP0 WP1 WP2 WP3 WP4 WP5 WP6 WP7 WP8 WP9a WP9b WP10a WP10b WP11 WP12 WP13".split()


def static_checks(root: Path) -> list[CheckResult]:
    return [
        *runners.check_ruff(root),
        runners.check_imports(root),
        scans.scan_residue(root),
        scans.scan_doubles(root),
        scans.scan_error_handling(root),
        scans.scan_print(root),
        scans.scan_subprocess(root),
        runners.check_secrets(root),
        delivery.check_lock(root),
    ]


def full_checks(root: Path, wp: str, data_root: Path) -> list[CheckResult]:
    results = [runners.check_pyright(root)]
    unit = runners.run_pytest(root, "unit tests", ["-m", "not gpu"])
    results.append(unit.result)
    results += runners.check_contract(root)
    results.append(runners.check_doctor(root, wp, data_root))
    results += [
        delivery.check_delivery_sections(root, wp),
        delivery.check_fixture_declarations(root, wp),
        delivery.check_decisions(root, wp),
        delivery.check_skips(root, wp, unit.skipped),
        delivery.check_evidence(root, wp, data_root if os.environ.get("KTA_DATA") else None),
    ]
    results += specific.run_specific(wp, root, unit.passed)
    return results


def summary(wp: str, quick: bool, results: list[CheckResult]) -> str:
    counts = {s: sum(r.status == s for r in results) for s in ("PASS", "WARN", "FAIL")}
    verdict = "PASSED" if counts["FAIL"] == 0 else "FAILED"
    lines = [
        f"gate {wp}{' --quick' if quick else ''}: {verdict} "
        f"({counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail)"
    ]
    lines += [f"  {r.status} {r.name}" + (f" — {r.detail}" if r.detail else "") for r in results]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/gate")
    parser.add_argument("wp", choices=WP_NAME)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="static checks only (for every commit); not a delivery gate",
    )
    args = parser.parse_args(argv)
    data_env = os.environ.get("KTA_DATA")
    with tempfile.TemporaryDirectory(prefix="kta-gate-") as scratch:
        data_root = Path(data_env) if data_env else Path(scratch)
        results = static_checks(ROOT)
        if not args.quick:
            results += full_checks(ROOT, args.wp, data_root)
    for r in results:
        print(r.render())
    print()
    print("---- summary (paste into the delivery note) ----")
    print(summary(args.wp, args.quick, results))
    return 0 if all(r.status != "FAIL" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
