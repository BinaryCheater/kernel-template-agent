"""PERFMODEL checks (docs/design/perfmodel/LOOP.md): the performance-model loop's own gate.

Both the quick and the full gate run them (they take seconds): pyright, the perfmodel unit tests
(`tests/unit/test_perfmodel_*`), the error table reproduced from its cache without writing (every
row cached: the committed table matches the committed engine), and the state file exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from kta_gate import runners
from kta_gate.result import CheckResult, from_findings


def check_table(root: Path) -> CheckResult:
    proc = runners.run(
        [str(root / "scripts" / "perfmodel-table"), "--no-write", "--cached-only"], root
    )
    if proc.code != 0:
        tail = proc.out.strip().splitlines()[-3:]
        return CheckResult("perfmodel table from cache", "FAIL", f"exit {proc.code}", tail)
    doc = json.loads((root / "docs/design/perfmodel/error-table.json").read_text())
    return from_findings("perfmodel table from cache", [], f"{len(doc['rows'])} rows")


def perfmodel(root: Path, passed: set[str]) -> list[CheckResult]:
    tests = sorted(
        str(p.relative_to(root)) for p in (root / "tests/unit").glob("test_perfmodel_*.py")
    )
    out = [runners.check_pyright(root)]
    out.append(runners.run_pytest(root, "perfmodel unit tests", ["-m", "not gpu", *tests]).result)
    out.append(check_table(root))
    state = root / "docs/design/perfmodel/STATE.md"
    out.append(from_findings("perfmodel STATE.md", [] if state.exists() else ["missing"], "exists"))
    return out
