"""Interrupted tool execution must not hide an already completed model response."""

from __future__ import annotations

import json
from pathlib import Path

from kta_template_replay.report import summarize


def test_dsh_reports_reconcile_complete_native_usage_across_resume(tmp_path: Path) -> None:
    def put(relative: str, value: dict) -> None:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    put("manifest.json", {"case": "attention", "shapes": [], "performance_aim": 0.9})
    put("package.json", {"mode": "self-selection"})
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace/candidate.py").write_text("def entry(*args): return None\n")
    first = {"input": 1100, "cached_input": 1000, "output": 20}
    # The second stream has not emitted its final step-end event before interruption.
    second_stream = {"input": 2199, "cached_input": 2000, "output": 29}
    session = {
        "provider": "deepseek_harness",
        "model": "deepseek-flash",
        "effort": "high",
        "wall_s": 1.0,
        "queue_s": 0.0,
        "native_transcripts": [],
        "usage": second_stream,
        "resolved_native": [
            {
                "usage": [
                    {
                        "inputTokens": 100,
                        "cacheReadTokens": 1000,
                        "outputTokens": 20,
                        "totalTokens": 1120,
                    },
                    {
                        "inputTokens": 200,
                        "cacheReadTokens": 2000,
                        "outputTokens": 30,
                        "totalTokens": 2230,
                    },
                ]
            }
        ],
    }
    put("session-state/session.json", session)
    put("session-state/turn-one/result.json", {**session, "usage": first})
    put("session-state/turn-two/result.json", session)

    result = summarize(tmp_path)
    assert result["usage"] == {"input": 3300, "cached_input": 3000, "output": 50}
    assert result["usage_reconciliation"]["stdout_step_end_sums"] == {
        "input": 3299,
        "cached_input": 3000,
        "output": 49,
    }
    assert result["usage_reconciliation"]["match"] is False
    assert result["agent_wall_s"] == 2.0
