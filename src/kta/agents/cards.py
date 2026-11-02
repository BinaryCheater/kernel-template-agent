"""Task card → prompt: the workspace AGENTS.md, TASK.md and incremental messages.

Templates come from the AgentProfile (blobs, hashed into the record), never from the
source tree at run time, so a run is reproducible from its profile. They use
``string.Template`` ``$name`` placeholders; a missing placeholder is an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from string import Template
from typing import Any

from kta.core.errors import ConfigError
from kta.core.schemas.process import TaskCard


@dataclass(frozen=True)
class InputCard:
    id: str
    kind: str
    title: str
    lines: list[str]


@dataclass(frozen=True)
class RetryNote:
    """Why the previous attempt was rejected: failing check names and their text."""

    failures: list[tuple[str, str]]  # (done check, failure text)

    @property
    def signature(self) -> str:
        return json.dumps(sorted(self.failures), ensure_ascii=False)


def render(template: str, values: dict[str, str]) -> str:
    try:
        return Template(template).substitute(values)
    except KeyError as exc:
        raise ConfigError(f"prompt template needs ${exc.args[0]}") from exc


def skills_table(descriptions: dict[str, str]) -> str:
    return "\n".join(f"- `{name}`: {text}" for name, text in sorted(descriptions.items()))


def prefix(template: str, descriptions: dict[str, str]) -> str:
    """The fixed prefix: byte-identical for every session of one profile."""
    return render(template, {"skills_table": skills_table(descriptions)})


def _bullets(items: list[str], empty: str = "(none)") -> str:
    return "\n".join(f"- {i}" for i in items) if items else empty


def _inputs(cards: list[InputCard], max_lines: int) -> str:
    if not cards:
        return "(none)"
    blocks = []
    for c in cards:
        body = "\n".join(f"  {line}" for line in c.lines[:max_lines])
        blocks.append(f"- `{c.id}` {c.kind}: {c.title}" + (f"\n{body}" if body else ""))
    return "\n".join(blocks)


def _budget(budget: dict[str, Any], used: dict[str, float]) -> str:
    if not budget:
        return "(no explicit budget)"
    lines = []
    for key, limit in sorted(budget.items()):
        spent = used.get(key)
        lines.append(f"- {key}: {limit}" + (f" (used {spent:g})" if spent is not None else ""))
    return "\n".join(lines)


FAILURE_CHARS = 700  # D719: per failing check in TASK.md; the full text is in the rejection report


def failure_lines(retry: RetryNote) -> list[str]:
    """One bounded line per failing check (D719). A check whose text repeats an earlier
    one says so instead of repeating it, so a long failure cannot push TASK.md over
    session.step_header_size and stop the step before the agent runs."""
    out: list[str] = []
    seen: dict[str, str] = {}
    for name, text in retry.failures:
        if text in seen:
            out.append(f"`{name}`: same failure as `{seen[text]}`")
            continue
        seen[text] = name
        short = " ".join(text.split())
        if len(short) > FAILURE_CHARS:
            short = short[:FAILURE_CHARS] + " … (cut; run the check again to see all of it)"
        out.append(f"`{name}`: {short}")
    return out


def task_md(
    template: str,
    *,
    task_id: str,
    card_id: str,
    card: TaskCard,
    cards: list[InputCard],
    skills: list[str],
    commands: list[str],
    used: dict[str, float],
    retry: RetryNote | None,
    policy: dict[str, Any],
    last_submission: str | None = None,
) -> str:
    k = int(policy["cards_k"])
    previous = ""
    if last_submission:
        # D721: the task's state, from the queue: a fresh session must not take notes
        # saying "submitted" for "done"
        previous = (
            "\n## Status of this task\n\nThis task is open. An earlier submission "
            f"(`{last_submission}`) was not accepted; the task ends only when a submission "
            "passes the done checks. Check `out/` against them and submit again.\n"
        )
    if retry is not None and retry.failures:
        previous += "\n## Previous attempt was rejected\n\n" + _bullets(failure_lines(retry))
    out = render(
        template,
        {
            "task_id": task_id,
            "work_unit": card.work_unit,
            "card_id": card_id,
            "goal": card.goal,
            "done_checks": _bullets([f"`{c}`" for c in card.done_checks]),
            "inputs": _inputs(cards[:k], int(policy["card_max_lines"]))
            + (
                f"\n\nAll {len(cards)} inputs are indexed in inputs/index.json."
                if len(cards) > k
                else ""
            ),
            "budget": _budget(card.budget, used),
            "skills": _bullets([f"`{s}` (`.agents/skills/{s}/SKILL.md`)" for s in skills]),
            "commands": " ".join(f"`kta {c}`" for c in commands) or "(none)",
            "previous": previous,
        },
    )
    if card.header is not None:
        out = f"{card.header.text}\n\n{out}"  # case-loop tasks start with the header (WP19)
    limit = int(policy["step_header_max_chars"])  # session.step_header_size
    if len(out) > limit:
        raise ConfigError(f"TASK.md is {len(out)} chars, above step_header_max_chars={limit}")
    return out


def rejection(template: str, task_id: str, retry: RetryNote) -> str:
    return render(
        template,
        {"task_id": task_id, "failures": _bullets(failure_lines(retry))},
    )


def new_session(template: str, work_unit: str, retry: RetryNote | None) -> str:
    extra = ""
    if retry is not None and retry.failures:
        extra = "\nThe previous attempt was rejected; TASK.md lists the failing checks."
    return render(template, {"work_unit": work_unit, "extra": extra})


def human_note(template: str, note: str) -> str:
    return render(template, {"note": note})


def budget_warning(template: str, task_id: str, resource: str, used: str, remaining: str) -> str:
    return render(
        template, {"task_id": task_id, "resource": resource, "used": used, "remaining": remaining}
    )
