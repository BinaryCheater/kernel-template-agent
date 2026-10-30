"""Loop instances and transitions: the data `step` reads and returns (decision D260).

A state's work (an effectful handler in ``solve_work`` / ``expand_work``) runs first and
reports what it observed as ``Loop.facts``. ``step`` is then a pure function of the loop
row, those facts and the configuration: it chooses the transition, the records to write
and the tasks to create. The runner applies the result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.core.config import Config

HOLD = "hold"  # not a machines.yaml kind: the instance stays where it is


@dataclass(frozen=True)
class Loop:
    id: str
    kind: str  # machine id: solve | expand
    subject: str  # Case or Reference record ID
    state: str
    data: Mapping[str, Any] = field(default_factory=lambda: dict[str, Any]())
    facts: Mapping[str, Any] = field(default_factory=lambda: dict[str, Any]())


@dataclass(frozen=True)
class TaskRequest:
    type: str  # T1-T6
    goal: str
    inputs: list[str]  # record IDs; the first is the main subject


@dataclass(frozen=True)
class Draft:
    """A record to write when the transition is applied.

    Record IDs depend on the writer's code state (Store.put fills actor.commit), so a draft
    refers to earlier drafts of the same transition by index (``links``), and ``data_key``
    names the loop data key that receives the written ID.
    """

    kind: str
    body: Mapping[str, Any]
    refs: Mapping[str, str | list[str]] = field(default_factory=lambda: dict[str, Any]())
    links: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: dict[str, Any]())
    data_key: str | None = None


@dataclass(frozen=True)
class Transition:
    src: str
    to: str | None  # None: hold (stay in src)
    kind: str  # machines.yaml transition kind, or HOLD
    rule: int | None = None
    data: Mapping[str, Any] = field(default_factory=lambda: dict[str, Any]())
    tasks: tuple[TaskRequest, ...] = ()
    records: tuple[Draft, ...] = ()
    note: str = ""

    @property
    def moves(self) -> bool:
        return self.to is not None


def hold(loop: Loop, note: str, **data: Any) -> Transition:
    return Transition(loop.state, None, HOLD, data=data, note=note)


@dataclass(frozen=True)
class Settings:
    """Numeric defaults of docs/design/10-11; first-fill values to be tuned (D263, D268)."""

    max_rounds: int = 5
    max_retries: int = 2
    shortlist: int = 64
    top_k: int = 8
    rank_min: float = 0.5  # Spearman rho below this (with >= rank_n points) -> rule 2
    rank_n: int = 3
    rel_error_max: float = 0.3  # median |pred - meas| / meas above this -> rule 2
    explained_min: float = 0.8  # share of the gap to the bound the model explains -> rule 3
    suspicious_vs_reference: float = 1.3  # inbox: faster than the strongest reference by this
    suspicious_pct_of_bound: float = 95.0  # inbox: closer to the bound than this
    attention_limit: int = 10
    infra_retries: int = 3
    lease_s: int = 1800
    task_wall_s: float = 3600.0
    task_gpu_s: float = 600.0
    task_steps: int = 200
    solve_time_limit_s: float = 60.0
    epsilon_rel: float = 0.1  # expand: reproduction error allowed

    @classmethod
    def from_config(cls, config: Config) -> Settings:
        section = config.section("orchestrator")
        known = cls.__dataclass_fields__
        values = {k: type(getattr(cls(), k))(v) for k, v in section.items() if k in known}
        return cls(**values)
