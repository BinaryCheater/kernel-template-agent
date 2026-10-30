"""WP13 (a): where construction time goes in the F2 v7 Attention batch.

Reads the 27 trajectory ledgers of formal-v7 and, per v7 stage, the raw author
session files, solver receipts and GPU worker cost files. Writes one JSON file
with per-stage rows (each carrying its raw source paths), per-trajectory sums,
per task x arm medians, and the Template-minus-Text comparison.

Usage:
    python3 scripts/analysis/f2_time_breakdown.py [--f2-root DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import f2_common as c

STAGE_FIELDS = (
    "author_wall_s",
    "author_timeout_wall_s",
    "author_failed_wall_s",
    "provider_retry_span_s",
    "provider_startup_stall_s",
    "solve_stage_wall_s",
    "solver_s",
    "gpu_wall_including_queue_s",
    "worker_compile_first_call_s",
    "worker_qualification_s",
    "worker_measurement_s",
    "worker_process_active_s",
    "gpu_transport_residual_s",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "cmd_read",
    "cmd_write",
    "cmd_check",
    "cmd_other",
    "failed_commands",
    "read_bytes",
    "read_bytes_knowledge",
    "read_bytes_feedback",
    "check_output_bytes",
)


def _num(x) -> float:
    return float(x) if isinstance(x, (int, float)) else 0.0


def stage_row(stage: dict, ledger: Path) -> dict:
    cost = stage.get("cost") or {}
    status = cost.get("provider_status") or {}
    worker = cost.get("worker_costs") or {}
    author_wall = _num(cost.get("author_wall_s"))
    timed_out = status.get("reason") == "timeout"
    failed = "failure" in stage or status.get("returncode") not in (0, None)
    row: dict = {
        "stage": stage["name"],
        "provider_returncode": status.get("returncode"),
        "provider_reason": status.get("reason"),
        "quality_valid": stage.get("quality_valid"),
        "author_wall_s": author_wall,
        "author_timeout_wall_s": author_wall if timed_out else 0.0,
        "author_failed_wall_s": author_wall if (failed and not timed_out) else 0.0,
        "solver_s": _num(cost.get("solver_s")),
        "gpu_wall_including_queue_s": _num(cost.get("gpu_wall_including_queue_s")),
        "worker_compile_first_call_s": _num(
            worker.get("construction_compile_first_call_s")
        ),
        "worker_qualification_s": _num(worker.get("qualification_s")),
        "worker_measurement_s": _num(worker.get("measurement_s")),
        "worker_process_active_s": _num(worker.get("process_active_s")),
        "sources": {"ledger": c.rel(ledger)},
    }
    row["gpu_transport_residual_s"] = (
        row["gpu_wall_including_queue_s"] - row["worker_process_active_s"]
        if row["gpu_wall_including_queue_s"]
        else 0.0
    )
    totals = (cost.get("usage") or {}).get("totals") or {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        row[key] = _num(totals.get(key))
    row["solve_stage_wall_s"] = 0.0
    if stage.get("solver"):
        solve_dir = Path(stage["solver"])
        receipts = sorted(solve_dir.glob("*.receipt.json"))
        row["solve_stage_wall_s"] = sum(
            _num(c.load_json(p).get("wall_s")) for p in receipts
        )
        row["sources"]["solve_receipts"] = c.rel(solve_dir) + "/*.receipt.json"
    if stage.get("evaluation"):
        fb = c.feedback_dir_for(stage["evaluation"])
        row["sources"]["worker_costs"] = c.rel(fb / "costs.json")
    _add_session(row, stage)
    return row


def _add_session(row: dict, stage: dict) -> None:
    for key in ("cmd_read", "cmd_write", "cmd_check", "cmd_other"):
        row[key] = 0
    row.update(
        failed_commands=0,
        read_bytes=0,
        read_bytes_knowledge=0,
        read_bytes_feedback=0,
        check_output_bytes=0,
        provider_retry_span_s=0.0,
        provider_startup_stall_s=0.0,
        reads_before_first_write=None,
    )
    author = stage.get("author")
    if not author:
        return
    adir = Path(author)
    stdout = adir / "provider.stdout"
    if stdout.exists():
        s = c.parse_stdout(stdout)
        for cls in ("read", "write", "check", "other"):
            row[f"cmd_{cls}"] = s.commands.get(cls, 0)
        row["failed_commands"] = s.failed_commands
        row["read_bytes"] = s.output_bytes.get("read", 0)
        row["read_bytes_knowledge"] = s.read_bytes_by_target.get("knowledge", 0)
        row["read_bytes_feedback"] = s.read_bytes_by_target.get("feedback", 0)
        row["check_output_bytes"] = s.output_bytes.get("check", 0)
        row["reads_before_first_write"] = s.reads_before_first_write
        row["sources"]["session"] = c.rel(stdout)
    row["provider_retry_span_s"] = c.retry_spans_s(adir / "provider.stderr")
    # The launcher writes provider-command.json immediately before starting
    # the provider; its mtime is the closest recorded session start.
    start = (adir / "provider-command.json").stat().st_mtime
    row["provider_startup_stall_s"] = c.startup_stall_s(adir / "provider.stderr", start)
    row["sources"]["session_start"] = c.rel(adir / "provider-command.json") + " (mtime)"
    row["sources"]["provider_completion"] = c.rel(adir / "provider-completion.json")
    row["sources"]["provider_stderr"] = c.rel(adir / "provider.stderr")


def trajectory_summary(rows: list[dict], record: dict, export_path: Path) -> dict:
    sums = {f: sum(r[f] for r in rows) for f in STAGE_FIELDS}
    sums["author_rounds"] = sum(1 for r in rows if r["author_wall_s"] > 0)
    sums["author_timeouts"] = sum(1 for r in rows if r["author_timeout_wall_s"] > 0)
    sums["author_failures"] = sum(1 for r in rows if r["author_failed_wall_s"] > 0)
    e2e = record.get("trajectory_end_to_end_wall_s")
    sums["trajectory_end_to_end_wall_s"] = e2e
    accounted = (
        sums["author_wall_s"]
        + sums["solve_stage_wall_s"]
        + sums["gpu_wall_including_queue_s"]
    )
    sums["accounted_wall_s"] = accounted
    sums["unaccounted_wait_s"] = (e2e - accounted) if e2e is not None else None
    sums["export_known_incremental_active_s"] = record.get("known_incremental_active_s")
    sums["export_author_wall_s"] = record.get("author_wall_s")
    sums["export_source"] = c.rel(export_path)
    return sums


def shares(traj: dict) -> dict:
    total = traj["trajectory_end_to_end_wall_s"]
    if not total:
        return {}
    a = traj["author_wall_s"]
    return {
        "author_productive": (
            a - traj["author_timeout_wall_s"] - traj["author_failed_wall_s"]
        )
        / total,
        "author_timeout_or_failed": (
            traj["author_timeout_wall_s"] + traj["author_failed_wall_s"]
        )
        / total,
        "solve_stage": traj["solve_stage_wall_s"] / total,
        "gpu_worker_active": traj["worker_process_active_s"] / total,
        "gpu_transport_residual": traj["gpu_transport_residual_s"] / total,
        "unaccounted_wait": (traj["unaccounted_wait_s"] or 0.0) / total,
    }


def build(f2_root: Path) -> dict:
    export_path, records = c.export_records(f2_root)
    stages: list[dict] = []
    trajectories: list[dict] = []
    prior_author_s: dict[str, float] = {}
    for task, arm, rep, path, ledger in c.iter_trajectories(f2_root):
        key = f"{task}/{arm}-rep{rep}"
        rows = []
        prior = 0.0
        for idx, stage in enumerate(ledger["stages"]):
            if not c.is_v7_stage(stage["name"]):
                prior += _num((stage.get("cost") or {}).get("author_wall_s"))
                continue
            row = stage_row(stage, path)
            row.update(trajectory=key, task=task, arm=arm, rep=rep, ledger_index=idx)
            rows.append(row)
        prior_author_s[key] = prior
        record = records[(task, f"{arm}-rep{rep}")]
        summary = trajectory_summary(rows, record, export_path)
        summary.update(
            trajectory=key,
            task=task,
            arm=arm,
            rep=rep,
            ledger=c.rel(path),
            prior_protocol_author_wall_s=prior,
        )
        summary["shares_of_end_to_end"] = (
            shares(summary) if task != "attn-small" else {}
        )
        trajectories.append(summary)
        stages.extend(rows)
    return {
        "schema": "kta-wp13-f2-time-breakdown-1",
        "inputs": {
            "f2_root": c.rel(f2_root),
            "ledgers": c.rel(f2_root / c.V7_DIR) + "/trajectory-*-v7.json",
            "export": c.rel(export_path),
        },
        "notes": NOTES,
        "stages": stages,
        "trajectories": trajectories,
        "by_task_arm": by_task_arm(trajectories),
        "per_author_stage_medians": per_stage_medians(stages),
        "template_minus_text": template_minus_text(trajectories),
        "pooled_fresh_shares": pooled_shares(trajectories),
        "provider_connection": provider_connection(stages),
    }


NOTES = [
    (
        "Only stages whose name contains '-v7-' are counted. attn-small v7 stages are "
        "continuations of v4/v6 trajectories; their earlier author time is reported as "
        "prior_protocol_author_wall_s and excluded from sums."
    ),
    (
        "Author sessions ran as Codex `exec --ephemeral`; stdout has no per-event "
        "timestamps, so time inside a session cannot be split into reading, writing "
        "and debugging. Command counts and bytes are action composition, not time."
    ),
    (
        "unaccounted_wait_s = trajectory_end_to_end_wall_s (export results.json) minus "
        "author, solve-stage and GPU walls: controller queueing for author slots (six "
        "concurrent authors) and the GPU lock. Only defined for fresh medium/long."
    ),
    (
        "gpu_transport_residual_s = GPU wall including queue minus worker process "
        "active time: transfer, remote start-up and GPU-lock queueing."
    ),
]


def _median(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


AGG_FIELDS = (
    *STAGE_FIELDS,
    "author_rounds",
    "author_timeouts",
    "trajectory_end_to_end_wall_s",
    "unaccounted_wait_s",
    "export_known_incremental_active_s",
)


def by_task_arm(trajs: list[dict]) -> dict:
    out: dict = {}
    for task in c.TASKS:
        for arm in c.ARMS:
            group = [t for t in trajs if t["task"] == task and t["arm"] == arm]
            out[f"{task}/{arm}"] = {
                "n": len(group),
                "median": {f: _median([t[f] for t in group]) for f in AGG_FIELDS},
                "per_rep": {f: [t[f] for t in group] for f in AGG_FIELDS},
                "trajectories": [t["trajectory"] for t in group],
            }
    return out


PER_STAGE_FIELDS = (
    "author_wall_s",
    "provider_retry_span_s",
    "provider_startup_stall_s",
    "input_tokens",
    "output_tokens",
    "read_bytes",
    "read_bytes_knowledge",
    "cmd_read",
    "reads_before_first_write",
)


def per_stage_medians(stages: list[dict]) -> dict:
    """Medians over individual author stages (successful provider exits only)."""
    out = {}
    for task in c.TASKS:
        for arm in c.ARMS:
            group = [
                s
                for s in stages
                if s["task"] == task
                and s["arm"] == arm
                and s["provider_returncode"] == 0
            ]
            out[f"{task}/{arm}"] = {
                "n_stages": len(group),
                **{f: _median([s[f] for s in group]) for f in PER_STAGE_FIELDS},
            }
    return out


def template_minus_text(trajs: list[dict]) -> dict:
    agg = by_task_arm(trajs)
    out = {}
    for task in c.TASKS:
        tm = agg[f"{task}/template"]["median"]
        tx = agg[f"{task}/text"]["median"]
        out[task] = {
            f: (tm[f] - tx[f]) if tm[f] is not None and tx[f] is not None else None
            for f in AGG_FIELDS
        }
    return out


def pooled_shares(trajs: list[dict]) -> dict:
    fresh = [t for t in trajs if t["task"] != "attn-small"]
    out = {}
    for arm in (None, *c.ARMS):
        group = [t for t in fresh if arm is None or t["arm"] == arm]
        total = sum(t["trajectory_end_to_end_wall_s"] for t in group)
        keys = group[0]["shares_of_end_to_end"].keys()
        out[arm or "all"] = {
            "n_trajectories": len(group),
            "end_to_end_wall_s": total,
            "shares": {
                k: sum(
                    t["shares_of_end_to_end"][k] * t["trajectory_end_to_end_wall_s"]
                    for t in group
                )
                / total
                for k in keys
            },
        }
    return out


def provider_connection(stages: list[dict]) -> dict:
    sessions = [s for s in stages if "session" in s["sources"]]
    author = sum(s["author_wall_s"] for s in sessions)
    retry = sum(s["provider_retry_span_s"] for s in sessions)
    stall = sum(s["provider_startup_stall_s"] for s in sessions)
    return {
        "sessions": len(sessions),
        "sessions_with_startup_chain": sum(
            1 for s in sessions if s["provider_startup_stall_s"] > 0
        ),
        "sessions_with_chain_ge_60s": sum(
            1 for s in sessions if s["provider_retry_span_s"] >= 60
        ),
        "author_wall_s": author,
        "retry_span_s_lower_bound": retry,
        "startup_stall_s_upper_bound": stall,
        "retry_share_of_author_wall": retry / author if author else None,
        "startup_stall_share_of_author_wall": stall / author if author else None,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f2-root", type=Path, default=c.F2_ROOT)
    ap.add_argument(
        "--out", type=Path, default=Path("deliveries/WP13/f2-time-breakdown.json")
    )
    args = ap.parse_args(argv)
    result = build(args.f2_root)
    c.write_json(args.out, result)
    sys.stdout.write(f"wrote {args.out} ({len(result['stages'])} stages)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
