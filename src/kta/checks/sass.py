"""SASS statistics and expected-feature checks (spec/modules/measure-check.md).

Opcodes are counted from ``cuobjdump -sass`` text (the Build's ``sass`` blob). An
expectation applies when its ``when`` expression holds for the assignment; each
``must_contain`` / ``must_not_contain`` entry matches an opcode by prefix (``HMMA``
matches ``HMMA.16816.F32``). A violated expectation fails with ``intent_not_realized``
and writes a Gap (attribution ``intent_sass``).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from kta.backends.base import BuiltModule
from kta.checks.base import Checked, Checker, Evidence, actor, write_check
from kta.core.context import Context
from kta.core.expr import Expr
from kta.core.schemas import Record
from kta.core.schemas.knowledge import Gap, SassExpect

NAME = "sass_checker"
_INSTR = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")


def opcode_counts(sass: str) -> Counter[str]:
    return Counter(m.group(1) for m in _INSTR.finditer(sass))


def _has(counts: Counter[str], prefix: str) -> bool:
    return any(op == prefix or op.startswith(prefix + ".") for op in counts)


def sass_failures(
    counts: Counter[str], expectations: list[SassExpect], assignment: dict[str, Any]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for e in expectations:
        if not Expr.parse(e.when, set(assignment)).eval(assignment):
            continue
        missing = [p for p in e.must_contain if not _has(counts, p)]
        present = [p for p in e.must_not_contain if _has(counts, p)]
        if missing or present:
            failures.append(
                {
                    "check": "sass",
                    "failure": "intent_not_realized",
                    "when": e.when,
                    "missing": missing,
                    "unexpected": present,
                }
            )
    return failures


def sass_check(
    c: Checker, module: BuiltModule, expectations: list[SassExpect], assignment: dict[str, Any]
) -> Checked:
    if module.sass_blob is None:
        ev = Evidence()
        ev.error("sass", f"build {module.build_id[:12]} has no SASS listing")
        build = c.store.get(module.build_id)
        ev.refs.update({"env": build.refs["env"], "worker": build.refs["worker"]})
        return write_check(c, NAME, ev, module.build_id)
    counts = opcode_counts(c.ex.read_blob(module.sass_blob).decode("utf-8", "replace"))
    build = c.store.get(module.build_id)
    ev = Evidence(refs={"env": build.refs["env"], "worker": build.refs["worker"]})
    ev.blobs["sass"] = module.sass_blob
    ev.detail = {"opcodes": dict(counts.most_common()), "assignment": assignment}
    ev.failures = sass_failures(counts, expectations, assignment)
    checked = write_check(c, NAME, ev, module.build_id)
    for f in ev.failures:
        gap = Gap(
            missing="other",
            description=(
                f"SASS does not realize the declared intent (when {f['when']}): missing "
                f"{f['missing']}, unexpected {f['unexpected']}"
            ),
            attribution="intent_sass",
        )
        refs: dict[str, str | list[str]] = {"check": checked.id, "build": module.build_id}
        c.store.put(Record.new("Gap", gap, actor(NAME), c.clock.now(), refs=refs))
    return checked


def check_sass(
    module: BuiltModule,
    expectations: list[SassExpect],
    assignment: dict[str, Any],
    *,
    ctx: Context,
) -> Checked:
    return sass_check(Checker.from_ctx(ctx), module, expectations, assignment)
