"""Typed accessors of the controls registry: one property per knob (``controls.yaml`` ``accessor``).

Adding a knob is one row in ``controls.yaml`` plus one property here; a test checks that
every knob has its accessor and every accessor its knob. Accessors return copies, so a
snapshot cannot be changed through them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast


class Knobs(ABC):
    """Mixed into ``kta.core.controls.Controls``, which provides ``value``."""

    @abstractmethod
    def value(self, knob_id: str) -> Any:
        """The value of ``knob_id`` in this snapshot (a copy)."""

    def _map(self, knob_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self.value(knob_id))

    # ---- typed accessors: one per knob (controls.yaml `accessor`) ----------------------
    # 12 §1
    @property
    def rounds_per_case_loop(self) -> int:
        return int(self.value("budget.rounds_per_case_loop"))

    @property
    def retries_per_task(self) -> int:
        return int(self.value("budget.retries_per_task"))

    @property
    def task_wall_s(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._map("budget.task_wall_s").items()}

    @property
    def hard_cap_factor(self) -> float:
        return float(self.value("budget.hard_cap_factor"))

    @property
    def no_response_timeout_s(self) -> int:
        return int(self.value("budget.no_response_timeout_s"))

    @property
    def lease_s(self) -> int:
        return int(self.value("budget.lease_s"))

    @property
    def gpu_s(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._map("budget.gpu_s").items()}

    @property
    def tool_call_steps(self) -> int:
        return int(self.value("budget.tool_call_steps"))

    @property
    def reminder_fraction(self) -> float:
        return float(self.value("budget.reminder_fraction"))

    @property
    def process_quotas(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._map("budget.process_quotas").items()}

    @property
    def agent_instances(self) -> dict[str, int]:
        return {k: int(v) for k, v in self._map("budget.agent_instances").items()}

    @property
    def user_attention_cap(self) -> dict[str, int]:
        return {k: int(v) for k, v in self._map("budget.user_attention_cap").items()}

    # 12 §2
    @property
    def fresh_session_threshold(self) -> float:
        return float(self.value("session.fresh_session_threshold"))

    @property
    def same_failure_rule(self) -> int:
        return int(self.value("session.same_failure_rule"))

    @property
    def resume_scope(self) -> str:
        return str(self.value("session.resume_scope"))

    @property
    def step_header_size(self) -> dict[str, int]:
        return {k: int(v) for k, v in self._map("session.step_header_size").items()}

    @property
    def tool_output_truncation(self) -> int:
        return int(self.value("session.tool_output_truncation"))

    @property
    def access_discipline_version(self) -> str:
        return str(self.value("session.access_discipline_version"))

    @property
    def notes_required_sections(self) -> list[str]:
        return [str(s) for s in self.value("session.notes_required_sections")]

    # 12 §3
    @property
    def improvement_band(self) -> dict[str, Any]:
        return self._map("guard.improvement_band")

    @property
    def mechanism_attribution(self) -> bool:
        return bool(self.value("guard.mechanism_attribution"))

    @property
    def null_round_limit(self) -> int:
        return int(self.value("guard.null_round_limit"))

    @property
    def one_mechanism_per_round(self) -> int:
        return int(self.value("guard.one_mechanism_per_round"))

    @property
    def exploration_share(self) -> dict[str, int]:
        return {k: int(v) for k, v in self._map("guard.exploration_share").items()}

    @property
    def duplicate_proposal_rejection(self) -> bool:
        return bool(self.value("guard.duplicate_proposal_rejection"))

    @property
    def completion_checks_fixed(self) -> bool:
        return bool(self.value("guard.completion_checks_fixed"))

    @property
    def hidden_shapes_and_seeds(self) -> bool:
        return bool(self.value("guard.hidden_shapes_and_seeds"))

    @property
    def evidence_write_ban(self) -> bool:
        return bool(self.value("guard.evidence_write_ban"))

    @property
    def fixture_declaration(self) -> bool:
        return bool(self.value("guard.fixture_declaration"))

    @property
    def structural_dedupe(self) -> bool:
        return bool(self.value("guard.structural_dedupe"))

    @property
    def evidence_citation_in_plans(self) -> bool:
        return bool(self.value("guard.evidence_citation_in_plans"))

    @property
    def selection_controls(self) -> bool:
        return bool(self.value("guard.selection_controls"))

    @property
    def canary_rate(self) -> dict[str, Any]:
        return self._map("guard.canary_rate")

    # 12 §4
    @property
    def accept(self) -> dict[str, Any]:
        return self._map("escalation.accept")

    @property
    def case_to_knowledge(self) -> dict[str, Any]:
        return self._map("escalation.case_to_knowledge")

    @property
    def to_user(self) -> dict[str, bool]:
        return {k: bool(v) for k, v in self._map("escalation.to_user").items()}

    @property
    def finding_ladder(self) -> dict[str, int]:
        return {k: int(v) for k, v in self._map("escalation.finding_ladder").items()}

    @property
    def model_term_provisionality(self) -> dict[str, Any]:
        return self._map("escalation.model_term_provisionality")

    @property
    def template_assurance_for_use(self) -> str:
        return str(self.value("escalation.template_assurance_for_use"))

    @property
    def level_choice(self) -> float:
        return float(self.value("escalation.level_choice"))

    # 12 §7
    @property
    def shortlist(self) -> int:
        return int(self.value("solve.shortlist"))

    @property
    def solve_time_limit_s(self) -> float:
        return float(self.value("solve.time_limit_s"))

    @property
    def model_ranking(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._map("judging.model_ranking").items()}

    @property
    def explained_min(self) -> float:
        return float(self.value("judging.explained_min"))

    @property
    def suspicious_result(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._map("inbox.suspicious_result").items()}

    @property
    def repeat_failures(self) -> int:
        return int(self.value("inbox.repeat_failures"))

    @property
    def budget_warn(self) -> float:
        return float(self.value("inbox.budget_warn"))

    @property
    def infra_retries(self) -> int:
        return int(self.value("orch.infra_retries"))

    @property
    def reproduction_epsilon(self) -> float:
        return float(self.value("learning.reproduction_epsilon"))

    @property
    def learning_scope(self) -> dict[str, Any]:
        return self._map("learning.scope")

    @property
    def release_stage(self) -> str:
        return str(self.value("release.stage"))

    @property
    def version_admission(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._map("model.version_admission").items()}
