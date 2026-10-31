"""Budget reservation and settlement across the hierarchy (docs/design/11 §3, D268).

Scopes: ``global`` > ``process:<name>`` > ``loop:<id>`` > ``task:<id>``. A task reserves
its card budget from every ancestor scope that has a limit; if any refuses, the
reservations already made are released and the task stays queued. Scope amounts are in
``agent_min`` and ``gpu_min`` (the keys the API reads, D169); a card's budget is in
``wall_s``, ``gpu_s`` and ``steps``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kta.core.config import Config
from kta.core.controls import Controls
from kta.core.errors import BudgetExceeded, ConfigError
from kta.store.sqlite_ops import SqliteOps


def scope_amounts(card_budget: Mapping[str, Any]) -> dict[str, float]:
    return {
        "agent_min": float(card_budget.get("wall_s", 0.0)) / 60.0,
        "gpu_min": float(card_budget.get("gpu_s", 0.0)) / 60.0,
    }


def ancestors(process: str, loop_id: str) -> list[str]:
    return [f"loop:{loop_id}", f"process:{process}", "global"]


def has_scope(ops: SqliteOps, scope: str) -> bool:
    try:
        ops.budget(scope)
    except ConfigError:
        return False  # no limit set at this level: nothing to reserve against
    return True


def reserve_chain(ops: SqliteOps, scopes: Sequence[str], amount: Mapping[str, float]) -> list[str]:
    """Reserve ``amount`` in each existing scope; [] and nothing reserved if one refuses.

    Returns the scopes reserved in (to settle later). Raises BudgetRefused when refused.
    """
    done: list[str] = []
    for scope in scopes:
        if not has_scope(ops, scope):
            continue
        if not ops.reserve(scope, amount):
            for back in done:
                ops.settle(back, amount, {})
            raise BudgetRefused(scope, dict(amount), ops.budget(scope))
        done.append(scope)
    return done


def settle_chain(
    ops: SqliteOps, scopes: Sequence[str], reserved: Mapping[str, float], used: Mapping[str, float]
) -> None:
    for scope in scopes:
        ops.settle(scope, reserved, used)


def ensure_scope(ops: SqliteOps, scope: str, limit: Mapping[str, float]) -> None:
    """Create a scope with ``limit`` unless it exists (an existing limit is kept)."""
    if not has_scope(ops, scope):
        ops.set_budget(scope, limit)


def configure_scopes(ops: SqliteOps, config: Config, controls: Controls) -> None:
    """Global and process scopes from ``[budget]`` (``agent_min``, ``gpu_min`` per period).

    Without a ``[budget]`` section no limit is set and reservations at these levels are
    skipped. Existing limits are replaced, so the configuration stays the source of the
    global amounts; the process shares are the controls knob ``budget.process_quotas``.
    """
    section = config.section("budget")
    if "quotas" in section:
        raise ConfigError("[budget.quotas] moved to the controls knob budget.process_quotas (D482)")
    limit = {k: float(v) for k, v in section.items() if k in ("agent_min", "gpu_min")}
    if not limit:
        return
    ops.set_budget("global", limit)
    for process, share in controls.process_quotas.items():
        ops.set_budget(f"process:{process}", {k: v * share for k, v in limit.items()})


def view(ops: SqliteOps, scope: str) -> dict[str, dict[str, float]]:
    """Budget, used, reserved and remaining of one scope."""
    b = ops.budget(scope)
    left = {k: v - b["used"].get(k, 0.0) - b["reserved"].get(k, 0.0) for k, v in b["limit"].items()}
    return {**b, "remaining": left}


class BudgetRefused(BudgetExceeded):
    """A reservation was refused: the task stays queued (a normal outcome)."""

    def __init__(self, scope: str, amount: dict[str, float], state: dict[str, Any]) -> None:
        super().__init__(f"budget {scope} cannot reserve {amount}: {state}")
        self.scope = scope
