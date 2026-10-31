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

from kta.core.controls import Controls
from kta.core.schemas.plan import TaskHeader

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
    header: TaskHeader | None = None  # case-loop tasks: the one-screen header (WP19)


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
    """The orchestrator's view of a controls snapshot (kta.core.controls, WP21 D482).

    Every value comes from the controls registry; nothing here has a default of its own.
    ``controls`` is the version the values came from (None for unstored values).
    """

    max_rounds: int
    max_retries: int
    shortlist: int
    top_k: int  # measured points k (guard.exploration_share)
    rank_min: float  # Spearman rho below this (with >= rank_n points) -> rule 2
    rank_n: int
    rel_error_max: float  # median |pred - meas| / meas above this -> rule 2
    explained_min: float  # share of the gap to the bound the model explains -> rule 3
    suspicious_vs_reference: float  # inbox: faster than the strongest reference by this
    suspicious_pct_of_bound: float  # inbox: closer to the bound than this
    attention_limit: int
    infra_retries: int
    lease_s: int
    task_wall_s: Mapping[str, float]  # per task type
    task_gpu_s: Mapping[str, float]  # per task type
    case_gpu_factor: float  # case (loop) GPU cap = factor x the largest task cap
    task_steps: int
    solve_time_limit_s: float
    epsilon_rel: float  # expand: reproduction error allowed
    controls: str | None = None

    @classmethod
    def of(cls, c: Controls) -> Settings:
        rank, suspicious, gpu = c.model_ranking, c.suspicious_result, c.gpu_s
        factor = gpu.pop("case_factor")
        return cls(
            max_rounds=c.rounds_per_case_loop,
            max_retries=c.retries_per_task,
            shortlist=c.shortlist,
            top_k=c.exploration_share["measured_points"],
            rank_min=rank["rank_min"],
            rank_n=int(rank["rank_n"]),
            rel_error_max=rank["rel_error_max"],
            explained_min=c.explained_min,
            suspicious_vs_reference=suspicious["vs_reference"],
            suspicious_pct_of_bound=suspicious["pct_of_bound"],
            attention_limit=c.user_attention_cap["pending"],
            infra_retries=c.infra_retries,
            lease_s=c.lease_s,
            task_wall_s=c.task_wall_s,
            task_gpu_s=gpu,
            case_gpu_factor=factor,
            task_steps=c.tool_call_steps,
            solve_time_limit_s=c.solve_time_limit_s,
            epsilon_rel=c.reproduction_epsilon,
            controls=c.version_id,
        )

    def card_budget(self, task_type: str) -> dict[str, Any]:
        """The budget a new TaskCard of ``task_type`` gets."""
        return {
            "wall_s": self.task_wall_s[task_type],
            "gpu_s": self.task_gpu_s[task_type],
            "steps": self.task_steps,
        }

    def loop_limit(self) -> dict[str, float]:
        """The budget scope of one loop instance (a case): rounds x the largest task wall
        budget, case_gpu_factor x the largest task GPU budget (12 §1)."""
        return {
            "agent_min": self.max_rounds * max(self.task_wall_s.values()) / 60.0,
            "gpu_min": self.case_gpu_factor * max(self.task_gpu_s.values()) / 60.0,
        }
