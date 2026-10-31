"""The work unit of a new task: loop × role (WP23, D614; ``work_units.yaml``)."""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any, cast

import yaml

from kta.core.errors import ConfigError

RULES = ("step", "case", "case_role")
OTHER = "other"


@cache
def roles() -> dict[str, tuple[str, ...]]:
    raw = yaml.safe_load(
        resources.files("kta.orchestrator").joinpath("work_units.yaml").read_text("utf-8")
    )
    table = {
        str(k): tuple(str(t) for t in v) for k, v in cast(dict[str, Any], raw["roles"]).items()
    }
    seen: dict[str, str] = {}
    for role, types in table.items():
        for t in types:
            if t in seen:
                raise ConfigError(f"work_units.yaml: {t} in roles {seen[t]} and {role}")
            seen[t] = role
    return table


def role_of(task_type: str) -> str:
    return next((r for r, types in roles().items() if task_type in types), OTHER)


def unit_for(loop_id: str, task_type: str, rule: str, task_key: str) -> str:
    """``task_key`` is a fresh ID, used only by the ``step`` rule."""
    if rule not in RULES:
        raise ConfigError(f"work-unit rule {rule!r}; one of {RULES}")
    if loop_id.startswith("human:") or rule == "case":
        return loop_id
    if rule == "step":
        return f"{loop_id}:step:{task_key}"
    return f"{loop_id}:{role_of(task_type)}"


def loop_of(work_unit: str) -> str:
    """The loop a work unit belongs to (the inverse of ``unit_for``)."""
    return work_unit if work_unit.startswith("human:") else work_unit.split(":", 1)[0]
