"""The agent-side view of a controls snapshot (kta.core.controls, WP21 D483).

The runner reads session and context knobs only from the controls version in force at
the start of a step and records that version in the AgentRun (``refs.controls``). The
``context_policy`` / ``loop_policy`` fields of AgentProfile records written before WP21
(codex-v1, codex-v2) stay as recorded; they no longer drive a run.
"""

from __future__ import annotations

from typing import Any

from kta.core.controls import Controls


def context_policy(c: Controls) -> dict[str, Any]:
    header = c.step_header_size
    return {
        "cards_k": header["cards"],
        "card_max_lines": header["card_lines"],
        "step_header_max_chars": header["header_max_chars"],
        "tool_output_max_chars": c.tool_output_truncation,
        "notes_required_sections": c.notes_required_sections,
    }


def loop_policy(c: Controls) -> dict[str, Any]:
    return {
        "reset_context_ratio": c.fresh_session_threshold,
        "same_failure_rule": c.same_failure_rule,
        "resume_scope": c.resume_scope,
        "idle_s": c.no_response_timeout_s,
        "hard_cap_factor": c.hard_cap_factor,
        "lease_s": c.lease_s,
    }
