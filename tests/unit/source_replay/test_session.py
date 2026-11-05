"""Pure budget arithmetic checks and real process-boundary watchdog checks."""

import json
import sys
from pathlib import Path

import pytest

from kta.agents.launchspec import LaunchSpec
from kta.core.errors import ConfigError
from kta_source_replay.session import campaign_run_cli, queue_account, snapshot


def row(event, at, request="a", submission="one"):
    return {
        "event": event,
        "at_s": at,
        "request_id": request,
        "submission_id": submission,
        "slot": "run-01",
        "worker": "spark-hk",
        "job_id": "job",
    }


def test_open_queue_excluded_and_completed_intervals_counted_once():
    rows = [row("blocked", 1), row("queued", 3), row("waiting", 99)]
    assert queue_account(rows, "run-01", 0, 100)["queue_s"] == 96
    rows += [row("acquired", 103), row("completed", 110)]
    assert queue_account(rows, "run-01", 0, 120)["queue_s"] == 100
    # A resumed native turn clips historical queues; it never charges them twice.
    assert queue_account(rows, "run-01", 115, 120)["queue_s"] == 0
    assert queue_account(rows[:3], "run-01", 50, 100)["queue_s"] == 49


def test_retry_does_not_consume_another_submission():
    rows = [
        row("blocked", 1),
        row("queued", 2),
        row("acquired", 3),
        row("completed", 4),
        row("blocked", 5, "retry"),
        row("queued", 6, "retry"),
        row("acquired", 7, "retry"),
        row("completed", 8, "retry"),
    ]
    assert queue_account(rows, "run-01", 0, 9)["submissions"] == 1
    rows.append(row("blocked", 10, "next", "two"))
    assert queue_account(rows, "run-01", 0, 11)["submissions"] == 2


@pytest.mark.parametrize(
    "rows",
    [
        [row("queued", 1)],
        [row("blocked", 1), row("blocked", 2, "b")],
        [row("blocked", 1), row("queued", 2), row("acquired", 3), row("queued", 4)],
        [row("blocked", 2), row("queued", 1)],
    ],
)
def test_invalid_ledger_is_a_protocol_failure(rows):
    with pytest.raises(ConfigError):
        queue_account(rows, "run-01", 0, 20)


def test_remaining_active_survives_resume_without_evaluator_double_count(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                row("blocked", 1),
                row("queued", 2),
                row("acquired", 102),
                row("completed", 108),
            ]
        )
        + "\n"
    )
    cfg = {
        "slot": "run-01",
        "ledger": str(path),
        "start_s": 0,
        "budget_s": 1800,
        "prior_active_s": 10,
    }
    current = snapshot(cfg, 120)
    assert current["active_s"] == 30  # 10 previous +120wall-100queue; no adding worker time.
    assert current["remaining_active_s"] == 1770


def test_real_watchdog_interrupt_preserves_native_session(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "out").mkdir()
    home = tmp_path / "home"
    home.mkdir()
    ws = tmp_path / "workspace"
    ws.mkdir()
    (job / "prompt.md").write_text("non-model process contract test")
    (job / "campaign-watchdog.json").write_text(
        json.dumps(
            {
                "slot": "run-01",
                "ledger": str(tmp_path / "queue.jsonl"),
                "checkpoint": str(tmp_path / "native.json"),
                "remaining": str(tmp_path / "remaining.json"),
                "budget_s": 0.5,
                "max_submissions": 20,
                "idle_s": 60,
                "prior_active_s": 0,
            }
        )
    )
    program = Path(__file__).resolve().parents[2] / "fakes/native_recorded_process.py"
    outcome = campaign_run_cli(
        LaunchSpec(provider="codex", work_unit="test", run="test", mode="start"),
        [sys.executable, str(program), str(home)],
        {},
        ws,
        home,
        job,
    )
    assert outcome.exit == "active_budget"
    assert json.loads((tmp_path / "native.json").read_text())["session"] == "native-test"
    assert json.loads((tmp_path / "remaining.json").read_text())["remaining_active_s"] == 0
    assert (home / "interrupted").read_text() == "persisted"


def test_real_open_queue_outlasts_wall_budget_without_stopping_native(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "out").mkdir()
    home = tmp_path / "home"
    home.mkdir()
    ws = tmp_path / "workspace"
    ws.mkdir()
    (job / "prompt.md").write_text("non-model queue boundary test")
    queue = tmp_path / "queue.jsonl"
    (job / "campaign-watchdog.json").write_text(
        json.dumps(
            {
                "slot": "run-01",
                "ledger": str(queue),
                "checkpoint": str(tmp_path / "native.json"),
                "remaining": str(tmp_path / "remaining.json"),
                "budget_s": 0.6,
                "max_submissions": 20,
                "idle_s": 0.6,
                "prior_active_s": 0,
            }
        )
    )
    program = Path(__file__).resolve().parents[2] / "fakes/native_recorded_process.py"
    outcome = campaign_run_cli(
        LaunchSpec(provider="codex", work_unit="test", run="test", mode="start"),
        [sys.executable, str(program), str(home), str(queue)],
        {},
        ws,
        home,
        job,
    )
    assert outcome.exit == "ok"
    assert outcome.wall_s > 1.0  # The active/idle cap is0.6s; verified open wait is excluded.
    assert not (home / "interrupted").exists()
    current = json.loads((tmp_path / "remaining.json").read_text())
    assert current["queue_s"] > 0.7
    assert current["active_s"] < 0.6


def test_observed_clock_orders_heartbeat_acquisition_race_conservatively():
    rows = [
        row("blocked", 1),
        dict(row("queued", 2), observed_at_s=2),
        dict(row("waiting", 10), observed_at_s=10),
        dict(row("acquired", 9.9), observed_at_s=11, raw_remote_at_s=9.9),
        row("completed", 12),
    ]
    # A heartbeat emitted just after acquisition cannot enlarge the closed wait.
    assert queue_account(rows, "run-01", 0, 20)["queue_s"] == pytest.approx(7.9)


def test_incomplete_transport_does_not_turn_unverified_execution_into_free_queue():
    rows = [row("blocked", 1), row("queued", 2), row("waiting", 10), row("completed", 100)]
    assert queue_account(rows, "run-01", 0, 110)["queue_s"] == 8
