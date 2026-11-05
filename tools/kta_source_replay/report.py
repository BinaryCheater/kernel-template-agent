"""Readable progress and final outcomes from retained controlled-replay evidence."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def link(label: str, path: Path) -> str:
    return f"[{label}](<{path.resolve()}>)" if path.exists() else f"{label}: pending"


def native_requests(paths: list[Path]) -> list[dict]:
    """Deduplicate response records across resumed/exported copies of one transcript."""
    seen = {}
    for path in sorted(set(paths)):
        text = path.read_text()
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if index == len(lines) - 1 and not text.endswith("\n"):
                # A live native writer may not have finished its final event.
                continue
            entry = json.loads(line)
            if entry.get("type") != "token_usage_record":
                continue
            payload = entry["payload"]
            key = payload.get("response_id")
            if not key:
                raise ValueError(f"Native usage lacks response identity: {path}:{index + 1}")
            value = {"usage": payload["usage"], "timestamp": entry.get("timestamp")}
            if key in seen and seen[key] != value:
                raise ValueError("Conflicting repeated native response usage")
            seen[key] = value
    return list(seen.values())


def token_summary(requests: list[dict], rates: dict | None = None) -> dict:
    """Price each request at its own context tier; reasoning is included in output."""
    keys = {
        "input": "input_tokens",
        "cached_input": "cached_input_tokens",
        "cache_write_input": "cache_write_input_tokens",
        "output": "output_tokens",
        "reasoning_output": "reasoning_output_tokens",
    }
    totals: dict[str, Any] = {
        key: sum(r["usage"][native] for r in requests)
        if requests and all(native in r["usage"] for r in requests)
        else None
        for key, native in keys.items()
    }
    required = ("input", "cached_input", "cache_write_input", "output")
    complete = all(totals[k] is not None for k in required)
    totals["uncached_input"] = (
        (int(totals["input"]) - int(totals["cached_input"]) - int(totals["cache_write_input"]))
        if complete
        else None
    )
    components = {
        key: 0.0 for key in ("uncached_input", "cached_input", "cache_write_input", "output")
    }
    for row in requests:
        usage = row["usage"]
        if not all(keys[key] in usage for key in required):
            continue
        counts = {key: usage[keys[key]] for key in required}
        counts["uncached_input"] = (
            counts["input"] - counts["cached_input"] - counts["cache_write_input"]
        )
        if min(counts.values()) < 0:
            raise ValueError("Native cache-inclusive token counts are inconsistent")
        if rates:
            long = counts["input"] > rates["long_context_input_threshold"]
            for key in components:
                multiplier = (
                    rates["long_context_output_multiplier"]
                    if key == "output"
                    else rates["long_context_input_multiplier"]
                )
                components[key] += (
                    counts[key]
                    * rates["per_million"][key]
                    * (multiplier if long else 1)
                    / 1_000_000
                )
    return {
        "tokens": totals,
        "native_request_count": len(requests),
        "actual_billed_usd": None,
        "billing_status": "native billing not reported",
        "api_equivalent_usd": sum(components.values()) if complete and rates else None,
        "estimate_components_usd": components if complete and rates else None,
        "estimate_status": "per-request retained-price estimate"
        if complete and rates
        else "unavailable: complete native per-request accounting or retained prices absent",
    }


def milestones(submissions: list[dict], selected: dict) -> dict:
    result = {}
    tests = [
        ("first_correct_provisional", lambda row: row.get("passed")),
        ("first_target_provisional", lambda row: row.get("meets_target")),
        (
            "selected_source_first_measured",
            lambda row: (
                selected and row.get("candidate_identity") == selected.get("candidate_identity")
            ),
        ),
    ]
    for label, test in tests:
        first = next(filter(test, submissions), None)
        result[label] = None
        if first:
            result[label] = {
                "submission": first.get("submission"),
                "active_s": first.get("active_s_at_finish"),
            }
    return result


def slot_summary(root: Path, schedule: dict, campaign: dict, prices: dict) -> dict:
    slot = root / "slots" / schedule["slot"]
    turns = [read(p) for p in sorted((slot / "state").glob("*/result.json"))]
    turns = [t for t in turns if "per_turn_usage" in t]
    transcripts = {Path(p) for turn in turns for p in turn.get("native_transcripts", [])}
    transcripts.update((slot / "state/home/sessions").rglob("*.jsonl"))
    missing = sorted(str(p) for p in transcripts if not p.exists())
    transcripts = {p for p in transcripts if p.exists()}
    requests = native_requests(list(transcripts))
    usage = token_summary(requests, prices.get("gpt-6.1-sol"))
    cumulative = {}
    for turn in turns:
        cumulative[turn["session"]] = turn
    reconciled = {}
    for out, native in [
        ("input", "tokens_in"),
        ("cached_input", "cached_input"),
        ("output", "tokens_out"),
    ]:
        values = [turn.get(native) for turn in cumulative.values()]
        expected = sum(values) if values and all(v is not None for v in values) else None
        reconciled[out] = {
            "latest_session_totals": expected,
            "request_sum": usage["tokens"][out],
            "matches": expected == usage["tokens"][out] if expected is not None else None,
        }
    submissions = []
    for directory in sorted((slot / "submissions").glob("*")):
        if not directory.is_dir():
            continue
        info, result = read(directory / "submission.json"), read(directory / "summary.json")
        submissions.append({**info, **result, "path": str(directory)})
    selected, final = read(slot / "selected-source.json"), read(slot / "confirmation.json")
    shapes = campaign["task"]["shapes"]
    windows = final.get("windows", [])
    checked = [w for w in windows if [r.get("shape") for r in w.get("rows", [])] == shapes]
    hidden_correct = bool(checked) and all(
        all(r.get("correct") for r in w["rows"]) for w in checked
    )
    confirmed = bool(final.get("confirmed") and len(checked) == 3 and hidden_correct)
    if selected and final and final.get("selected_identity") != selected.get("candidate_identity"):
        raise ValueError("Final confirmation does not identify the selected source")
    if selected and (slot / "selected-candidate.py").exists():
        if (
            hashlib.sha256((slot / "selected-candidate.py").read_bytes()).hexdigest()
            != selected["candidate_identity"]
        ):
            raise ValueError("Selected candidate bytes changed")
    timing_rows = final.get("shape_rows", []) if confirmed else []
    closed, started = read(slot / "model-closed.json"), read(slot / "session-started.json")
    accounting = read(slot / "remaining-active.json") or closed.get("accounting", {})
    queue_s = None
    if started:
        from .session import ledger, queue_account

        rows = ledger(slot / "queue-events.jsonl")
        queue_s = queue_account(
            rows,
            schedule["slot"],
            0.0,
            max([r.get("observed_at_s", r["at_s"]) for r in rows], default=0.0),
        )["queue_s"]
    active = accounting.get("active_s")
    if final:
        outcome = (
            "target attained"
            if confirmed and final.get("meets_target")
            else "correct; target not attained"
            if confirmed
            else "measurement invalid/unresolved"
            if hidden_correct
            else "final validation failed"
        )
    elif closed:
        outcome = (
            "confirmation pending"
            if selected
            else "closed; source selection pending"
            if submissions
            else "closed without a submitted candidate"
        )
    else:
        outcome = "running" if started else "pending"
    confirmation_exec = [
        read(p) for p in (slot / "confirmation").glob("*/attempt-*/execution.json")
    ]
    return {
        **schedule,
        "outcome": outcome,
        "complete": bool(final or (closed and not submissions)),
        "hidden_correct": hidden_correct if final else None,
        "confirmed": confirmed,
        "target_attained": bool(confirmed and final.get("meets_target")),
        "submissions": len(submissions),
        "submission_records": submissions,
        "active_s": active,
        "gpu_queue_s": queue_s,
        "session_wall_s": active + queue_s if active is not None and queue_s is not None else None,
        "confirmation_active_s": final.get("confirmation_active_s"),
        "confirmation_queue_s": sum(e["queue_s"] for e in confirmation_exec)
        if confirmation_exec and all("queue_s" in e for e in confirmation_exec)
        else None,
        "usage": usage,
        "usage_reconciliation": reconciled,
        "missing_transcripts": missing,
        "native_transcripts": sorted(str(p) for p in transcripts),
        "milestones": milestones(submissions, selected),
        "shape_rows": timing_rows,
        "confirmation": final,
        "evidence": str(slot),
        "selected_source": str(slot / "selected-candidate.py") if selected else None,
        "final_confirmation_source_fallback": selected.get("fallback_latest"),
        "model_exit": closed.get("exit"),
        "latest_failure": next(
            (
                r.get("error") or r.get("reason")
                for r in reversed(windows + submissions)
                if r.get("error") or r.get("reason")
            ),
            None,
        ),
    }


def summarize(root: Path, price_file: Path | None = None) -> dict:
    root = root.resolve()
    campaign = read(root / "campaign.json") or read(
        root / "preparation/design-snapshot/campaign.json"
    )
    if not campaign:
        raise ValueError("Retained campaign settings absent")
    prices = read(price_file) if price_file else {}
    rows = [slot_summary(root, slot, campaign, prices) for slot in campaign["schedule"]]
    completed = all(r["complete"] for r in rows)
    return {
        "campaign": campaign["campaign"],
        "root": str(root),
        "settings": campaign,
        "slots": rows,
        "all_scheduled_outcomes_complete": completed,
        "actual_billing_usd": None,
        "price_file": str(price_file) if price_file else None,
        "price_basis": {k: prices.get(k) for k in ("as_of", "basis", "currency")},
        "arm_outcomes": {
            arm: {
                "scheduled": sum(r["arm"] == arm for r in rows),
                "target_attained": sum(r["arm"] == arm and r["target_attained"] for r in rows),
                "not_final": sum(r["arm"] == arm and not r["complete"] for r in rows),
            }
            for arm in campaign["arms"]
        },
        "HALT": read(root / "HALT.json"),
    }


def value(number: Any, suffix: str = "") -> str:
    if number is None:
        return "pending"
    return format(number, ",.2f" if isinstance(number, float) else ",") + suffix


def table(headers: list, rows: list[list]) -> str:
    def row(values: list) -> str:
        return "| " + " | ".join(str(x).replace("|", "\\|") for x in values) + " |\n"

    return row(headers) + row(["---"] * len(headers)) + "".join(row(r) for r in rows) + "\n"


def markdown(report: dict) -> str:
    root, rows = Path(report["root"]), report["slots"]
    total = sum(row["target_attained"] for row in rows)
    state = (
        "All nine outcomes recorded"
        if report["all_scheduled_outcomes_complete"]
        else "Execution incomplete"
    )
    text = f"# Source attention control: results\n\n{state}; {total}/9 confirmed target outcomes. "
    text += "Full matched outcomes are required before assessing supplied-knowledge benefit.\n\n"
    if report["HALT"]:
        text += f"Campaign halted: {report['HALT'].get('reason', 'see retained record')}.\n\n"
    p1 = root / "preparation/baseline/remote-retained"
    original = read(p1 / "p1-result.json")
    text += "## Original comparator and computation\n\n"
    text += "Unchanged CUDA FlashAttention-2 dense causal FP16 D128 forward, with a declared thin "
    text += "allocation/parameter/launch host binding. Both entries compute O and natural-log LSE. "
    text += "Q/K/V are BSHD [1,S,8,128], S=512,2048,513; O is FP16 BSHD and LSE is FP32 BHS. "
    text += "GB10 sm_121; primary timing includes the complete GPU entry under graph replay. "
    text += "The target permits about 11.1% more time on every shape.\n\n"
    text += (
        ", ".join(
            [
                link("Pinned source/build inventory", p1 / "source-inventory.json"),
                link("release build recipe", p1 / "build-recipe.json"),
                link("original output checks and runtime identity", p1 / "p1-result.json"),
                link("Source locations / rule map", root / "preparation/source-template-map.md"),
            ]
        )
        + ".\n\n"
    )
    if original.get("passed"):
        names = {name for row in original["rows"] for name in row.get("cuda_kernels", [])}
        traits = sorted(
            {
                match.groups()
                for name in names
                for match in re.finditer(
                    r"Flash_fwd_kernel_traits<(\d+), (\d+), (\d+), (\d+), "
                    r"(true|false), (true|false),",
                    name,
                )
            }
        )
        for head, query, key, warps, qreg, shared in traits:
            text += (
                f"Observed source traits: D{head}, {query} query rows, {key} keys, "
                f"{warps} warps, Q-register flag={qreg}, Q/K alias flag={shared}.\n\n"
            )
    text += "## Every scheduled session\n\n"
    status_rows = []
    for row in rows:
        correct = (
            "pending"
            if row["hidden_correct"] is None
            else "pass"
            if row["hidden_correct"]
            else "fail/unresolved"
        )
        target = (
            "attained"
            if row["target_attained"]
            else "not attained"
            if row["complete"] and row["outcome"] != "measurement invalid/unresolved"
            else "pending/unresolved"
        )
        status_rows.append(
            [
                row["slot"],
                row["arm"],
                correct,
                target,
                value(row["active_s"], " s"),
                value(row["gpu_queue_s"], " s"),
                value(row["session_wall_s"], " s"),
                row["submissions"],
                row["outcome"],
            ]
        )
    text += table(
        "Slot|Arm|Final O/LSE|Target|Active|GPU wait|Session wall|Submissions|Outcome".split("|"),
        status_rows,
    )
    text += "Session wall is accumulated active time plus verified GPU waiting; evaluator time "
    text += "is not added again. Confirmation worker time is separate.\n\n"
    text += "## Confirmed complete-entry performance\n\n"
    times = []
    for row in rows:
        for shape in row["shape_rows"]:
            a, b = shape["original_median_us"], shape["candidate_median_us"]
            delta = 100 * (b / a - 1)
            direction = f"{abs(delta):.2f}% {'more' if delta >= 0 else 'less'} time"
            originals, candidates = (
                shape["original_window_medians_us"],
                shape["candidate_window_medians_us"],
            )
            spread = (
                f"original {min(originals):.3f}–{max(originals):.3f}; "
                f"generated {min(candidates):.3f}–{max(candidates):.3f}"
            )
            times.append(
                [f"{row['slot']} / {shape['shape'][1]}", f"{a:.3f}", f"{b:.3f}", direction, spread]
            )
    text += table(
        ["Slot / S", "Original us", "Generated us", "Generated uses", "Window median ranges us"],
        times or [["No confirmed correct candidate", "pending", "pending", "pending", "pending"]],
    )
    text += "Only correct candidates with valid three-window confirmation enter this table. "
    text += "All failures/invalid samples remain linked below; eager diagnostics are separate.\n\n"
    text += "## Native usage and retained-price estimates\n\n"
    token_keys = (
        "input cached_input uncached_input cache_write_input output reasoning_output".split()
    )
    usage_rows = [
        [
            row["slot"],
            *[value(row["usage"]["tokens"][k]) for k in token_keys],
            value(row["usage"]["api_equivalent_usd"]),
        ]
        for row in rows
    ]
    text += table(
        (
            "Slot|Input incl. cache|Cached|Uncached|Cache write|Output|"
            "Reasoning subset|API-equivalent USD"
        ).split("|"),
        usage_rows,
    )
    text += "Actual billing is unavailable. Estimates apply each request's context tier using "
    text += f"the retained {report['price_basis'].get('as_of') or 'unavailable'} price card. "
    text += "Reasoning tokens are a subset of output. Response IDs deduplicate resumed exports. "
    text += "Missing counts remain unavailable.\n\n"
    for row in rows:
        slot = Path(row["evidence"])
        text += f"### {row['slot']}: retained details\n\n"
        text += (
            ", ".join(
                [
                    link("Native sessions/settings/snapshots", slot / "state"),
                    link("all submitted sources and paired samples", slot / "submissions"),
                    link("selected source", slot / "selected-candidate.py"),
                    link("hidden checks and confirmation", slot / "confirmation.json"),
                    link("design note", slot / "workspace/DESIGN.md"),
                ]
            )
            + ".\n\n"
        )
        text += f"Confirmation active: {value(row['confirmation_active_s'], ' s')}; "
        text += f"confirmation GPU wait: {value(row['confirmation_queue_s'], ' s')}.\n\n"
        for name, milestone in row["milestones"].items():
            observed = (
                f"submission {milestone['submission']}, {value(milestone['active_s'], ' active s')}"
                if milestone
                else "not recorded"
            )
            text += f"- {name.replace('_', ' ')}: {observed}.\n"
        if row["latest_failure"]:
            text += f"- Latest failure: {str(row['latest_failure']).splitlines()[0][:250]}.\n"
        for transcript in row["native_transcripts"]:
            text += f"- {link('Native transcript', Path(transcript))}.\n"
        for missing in row["missing_transcripts"]:
            text += f"- Missing native transcript: `{missing}`.\n"
        if any(v["matches"] is False for v in row["usage_reconciliation"].values()):
            text += "- Request/session token totals differ; reconcile the final session record.\n"
        text += "\n"
    text += "## Matched-arm status and evidence\n\n"
    for arm, outcome in report["arm_outcomes"].items():
        text += (
            f"- {arm}: {outcome['target_attained']}/{outcome['scheduled']} target outcomes; "
            f"{outcome['not_final']} pending outcomes retained.\n"
        )
    text += "\nNo successful-run-only average is used. Development milestones are provisional. "
    text += "Rule reads/use and self-supplied choices remain in design notes and transcripts.\n\n"
    text += link("Frozen campaign", root / "campaign.json") + ", "
    text += link("Prelaunch seal", root / "seal.json") + ", "
    text += link("Public packages", root / "packages") + ".\n"
    return text


def publish(root: Path, output: Path, price_file: Path | None = None) -> dict:
    if output.suffix != ".md" or output.resolve().is_relative_to((root / "slots").resolve()):
        raise ValueError("Reports must be Markdown outside raw slot records")
    report = summarize(root, price_file)
    previous = root / "results.json"
    prior_output = Path(read(previous).get("published_output", root / "results.md"))
    sources = {output: "destination.md", prior_output: "report.md", previous: "results.json"}
    if any(path.exists() for path in sources):
        snapshot = Path(tempfile.mkdtemp(prefix="report-history-", dir=root))
        for path, name in sources.items():
            if path.exists():
                target = snapshot / name
                with target.open("xb") as stream:
                    stream.write(path.read_bytes())
                target.chmod(0o444)
        snapshot.chmod(0o555)
    report["published_output"] = str(output.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown(report))
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
