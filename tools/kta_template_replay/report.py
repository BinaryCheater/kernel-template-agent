"""Coordinator report from retained session and evaluator evidence, never self-assessment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize(run: Path) -> dict[str, Any]:
    manifest = read(run / "manifest.json")
    session = read(run / "session-state/session.json")
    package = read(run / "package.json")
    candidate = run / "workspace/candidate.py"
    candidate_id = hashlib.sha256(candidate.read_bytes()).hexdigest()
    attempts = []
    matches = []
    for directory in sorted((run / "attempts").glob("*")):
        if not directory.is_dir():
            continue
        result_path = directory / "result.json"
        result = read(result_path) if result_path.exists() else None
        execution_path = directory / "execution.json"
        execution = read(execution_path) if execution_path.exists() else None
        attempts.append({"attempt": directory.name, "result": result, "execution": execution})
        if result and result.get("candidate_identity") == candidate_id:
            matches.append({"attempt": directory.name, "result": result})
    final = matches[-1]["result"] if matches else None
    rows = final.get("rows", []) if final else []
    correct = bool(
        final
        and final.get("passed")
        and [row.get("shape") for row in rows] == manifest["shapes"]
        and all(
            row.get("correct")
            and row.get("checks")
            and all(check.get("passed") for check in row["checks"])
            and len(row["checks"]) == len(manifest["private_seeds"]) * len(manifest["patterns"])
            and row.get("production_checks")
            and all(check.get("passed") for check in row["production_checks"])
            for row in rows
        )
    )
    aim = correct and all(row.get("ratio", 0) >= manifest["performance_aim"] for row in rows)
    turns = [
        read(path)
        for path in sorted((run / "session-state").glob("*/result.json"))
        if "usage" in read(path)
    ]
    provider = session.get("provider", "codex")
    usage = session["usage"]
    usage_reconciliation = None
    if provider == "deepseek_harness":
        stream_usage = {
            key: sum(turn["usage"][key] for turn in turns)
            if turns and all(turn["usage"].get(key) is not None for turn in turns)
            else None
            for key in ("input", "output", "cached_input")
        }
        native = [
            item
            for snapshot in session.get("resolved_native", [])
            for item in snapshot.get("usage", [])
        ]
        if not native or any(not isinstance(item, dict) for item in native):
            raise ValueError("DSH report requires complete exported native assistant usage")
        usage = {
            "input": sum(item["totalTokens"] - item["outputTokens"] for item in native),
            "output": sum(item["outputTokens"] for item in native),
            "cached_input": sum(item["cacheReadTokens"] for item in native),
        }
        usage_reconciliation = {
            "source": "latest native assistant/message snapshot, cumulative for the session",
            "native_message_count": len(native),
            "stdout_step_end_sums": stream_usage,
            "native_sums": usage,
            "match": usage == stream_usage,
        }
    executions = [item["execution"] for item in attempts if item["execution"]]
    return {
        "case": manifest["case"],
        "mode": package["mode"],
        "provider": provider,
        "model": session["model"],
        "effort": session["effort"],
        "correct": correct,
        "performance_aim_met": aim,
        "final_candidate": str(candidate),
        "measured_matches": [x["attempt"] for x in matches],
        "rows": rows,
        "attempts": attempts,
        "agent_wall_s": sum(turn["wall_s"] for turn in turns) or session["wall_s"],
        "agent_queue_s": sum(turn["queue_s"] for turn in turns) or session["queue_s"],
        "evaluation_wall_s": sum(item["wall_s"] for item in executions),
        "gpu_queue_s": sum(item.get("queue_s", 0) for item in executions),
        "usage": usage,
        "usage_reconciliation": usage_reconciliation,
        "cost": None,
        "cost_status": "not reported by native harness",
        "compile_s": final.get("compile_s") if final else None,
        "first_call_compile_and_execution_s": final.get("first_call_compile_and_execution_s")
        if final
        else None,
        "run": str(run),
        "native_transcripts": session["native_transcripts"],
    }


def markdown(report: dict[str, Any]) -> str:
    state = (
        "Correct on every prescribed shape"
        if report["correct"]
        else "Final candidate unverified or incorrect"
    )
    target = "met" if report["performance_aim_met"] else "not met"
    lines = [
        f"# {report['case']}: {report['mode']} ({report['provider']})",
        "",
        f"{state}. The per-shape performance aim is {target}.",
        "",
        "| Shape | Production median us (IQR) | Candidate median us (IQR) | Ratio |",
        "|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        shape = " × ".join(map(str, row["shape"]))
        if "candidate" in row and row.get("correct"):
            reference, candidate = row["production"], row["candidate"]
            lines.append(
                f"| {shape} | {reference['median_us']:.3f} ({reference['iqr_us']:.3f}) | "
                f"{candidate['median_us']:.3f} ({candidate['iqr_us']:.3f}) | {row['ratio']:.3f} |"
            )
        else:
            lines.append(f"| {shape} | — | correctness failed | — |")
    usage = report["usage"]
    uncached = (
        usage.get("input") - usage.get("cached_input")
        if all(usage.get(key) is not None for key in ("input", "cached_input"))
        else None
    )
    failed = [
        item["attempt"]
        for item in report["attempts"]
        if not item["result"] or not item["result"].get("passed")
    ]
    lines += [
        "",
        "Ratio is production latency / candidate latency. Raw samples remain in the run.",
        "",
        f"Settings: {report['model']}, effort {report['effort']}; exact native harness settings "
        "and frozen task/library are retained alongside the session.",
        "",
        f"Agent wall: {report['agent_wall_s']:.2f} s; evaluation/transport within that wall: "
        f"{report['evaluation_wall_s']:.2f} s; GPU queue: {report['gpu_queue_s']:.3f} s; "
        f"agent-job queue: {report['agent_queue_s']:.3f} s.",
        "",
        f"Input including cache: {usage.get('input')}; cached input: {usage.get('cached_input')}; "
        f"uncached input: {uncached}; output: {usage.get('output')}. Monetary cost was not "
        "reported by the native harness. Original usage fields remain retained.",
        "",
        f"Evaluation batches: {len(report['attempts'])}; "
        f"failed/unfinished batches: {failed or 'none'}. "
        f"Final candidate matches measured batches: {report['measured_matches'] or 'none'}.",
        "",
        "Compiler-only time is unavailable; first-call compilation/execution and validation "
        "are recorded separately from steady-state timing.",
        "",
        f"[Candidate]({report['final_candidate']}); "
        f"[design note]({report['run']}/workspace/DESIGN.md). "
        f"Retained run: `{report['run']}`. `attempts/` contains source snapshots, errors, "
        "environment records and raw timing samples; `session-state/` contains native "
        "transcripts, input, tool calls, settings and available accounting.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--summary-name", default="coordinator-summary.json")
    args = parser.parse_args()
    report = summarize(args.run.resolve())
    args.output.write_text(markdown(report))
    retained = args.run / args.summary_name
    with retained.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "correct": report["correct"],
                "performance_aim_met": report["performance_aim_met"],
            }
        )
    )


if __name__ == "__main__":
    main()
