"""The versioned access-discipline table (WP19; docs/design/02 §5.4, D-T20).

Data in ``access_discipline.yaml``; this module loads a version, classifies a pull
(command + record kind) against the row of the task's situation, and renders the table as
the Markdown the worker skill ``use-kta`` carries (a test keeps the two identical).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import yaml

from kta.core.errors import ConfigError

SKILL_BEGIN = "<!-- access-discipline:begin -->"
SKILL_END = "<!-- access-discipline:end -->"


@dataclass(frozen=True)
class Row:
    id: str
    situation: str
    fetch: str
    pulls: tuple[str, ...]  # "<command>" or "<command> <Kind>"
    decide: str


@dataclass(frozen=True)
class Table:
    version: str
    rows: tuple[Row, ...]

    def row(self, row_id: str) -> Row:
        for r in self.rows:
            if r.id == row_id:
                return r
        raise ConfigError(f"access discipline {self.version} has no situation {row_id!r}")

    def classify(self, command: str, kind: str | None, situation: str) -> str | None:
        """The row allowing this pull: the situation's own row, else None (outside)."""
        keys = {command, f"{command} {kind}"} if kind else {command}
        r = next((r for r in self.rows if r.id == situation), None)
        return r.id if r is not None and keys & set(r.pulls) else None


@cache
def _versions() -> dict[str, Any]:
    raw = resources.files("kta.core").joinpath("access_discipline.yaml").read_text("utf-8")
    data: Any = yaml.safe_load(raw)
    return dict(data["versions"])


def versions() -> list[str]:
    return list(_versions())


def table(version: str) -> Table:
    data = _versions().get(version)
    if data is None:
        raise ConfigError(f"unknown access-discipline version {version!r}")
    rows = tuple(
        Row(str(r["id"]), str(r["situation"]), str(r["fetch"]),
            tuple(str(p) for p in r["pulls"]), str(r["decide"]))
        for r in data["rows"]
    )  # fmt: skip
    return Table(version, rows)


def render(t: Table) -> str:
    """The Markdown block of the worker skill (between SKILL_BEGIN and SKILL_END)."""
    lines = [
        SKILL_BEGIN,
        f"Access discipline **{t.version}** (your `TASK.md` header names the situation).",
        "",
        "| Situation (id) | Fetch | Decide |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {r.situation} (`{r.id}`) | {r.fetch} | {r.decide} |" for r in t.rows]
    lines.append(SKILL_END)
    return "\n".join(lines)
