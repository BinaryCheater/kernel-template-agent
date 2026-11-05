"""Frozen seed schedules, submission/retry limits and coordinator-only confirmation."""

from __future__ import annotations

import random
import secrets
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from kta.core.context import SystemClock
from kta.core.ids import UlidFactory
from kta_template_replay.manifest import identity, load, write_new

from .remote import execute
from .worker import drift


def initialize(data: Path, campaign_path: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    campaign = load(campaign_path)
    if (data / "campaign.json").exists():
        if load(data / "campaign.json") != campaign:
            raise ValueError(
                "Existing campaign settings differ; preserve them and use a new campaign"
            )
    else:
        write_new(data / "campaign.json", campaign)
    generator = random.Random(secrets.randbits(256))
    patterns = campaign["validation"]["patterns"]
    development = {
        str(block): {
            str(sub): {p: [generator.randrange(2**62) for _ in range(2)] for p in patterns}
            for sub in range(1, 21)
        }
        for block in range(1, 4)
    }
    final = {
        str(block): {p: [generator.randrange(2**62) for _ in range(8)] for p in patterns}
        for block in range(1, 4)
    }
    write_new(
        data / "seeds.json",
        {
            "development": development,
            "final": final,
            "generator_identity": identity(Path(__file__).with_name("worker.py").read_bytes()),
        },
    )


def active_s(slot: Path) -> float:
    remaining = slot / "remaining-active.json"
    if remaining.exists():
        return float(load(remaining)["active_s"])
    started_doc = load(slot / "session-started.json")
    started = started_doc.get("at_s", started_doc.get("epoch_s"))
    if started is None:
        raise ValueError("Session start timestamp missing")
    completed_queue = sum(
        load(p).get("queue_s", 0) for p in slot.glob("submissions/*/*/execution.json")
    )
    return started_doc.get("prior_active_s", 0) + time.time() - started - completed_queue


def verify_sealed(data: Path) -> dict:
    campaign = load(data / "campaign.json")
    from .seal import verify_seal

    verify_seal(data)
    if (data / "HALT.json").exists():
        raise ValueError("Campaign halted after exhausted external-failure allowance")
    return campaign


def request_for(
    data: Path, campaign: dict, slot_name: str, mode: str, submission: int, window: int
) -> dict:
    slot = next(s for s in campaign["schedule"] if s["slot"] == slot_name)
    schedule = load(data / "seeds.json")
    if schedule["generator_identity"] != identity(
        Path(__file__).with_name("worker.py").read_bytes()
    ):
        raise ValueError("Generator/evaluator changed after seed schedule freeze")
    seeds = (
        schedule["final"][str(slot["block"])]
        if mode == "final"
        else schedule["development"][str(slot["block"])][str(submission)]
    )
    baseline = load(data / "baseline.json")
    return {
        "request_id": UlidFactory(SystemClock()).new(),
        "slot": slot_name,
        "mode": mode,
        "block": slot["block"],
        "submission": submission,
        "window": window,
        "submission_id": f"{slot_name}:{submission:02d}",
        "seeds": seeds,
        "baseline_module": "/baseline/" + baseline.get("module", "binding.py"),
        "baseline_identity": baseline["identity"],
        "source_revision": campaign["source"]["revision"],
    }


def allowed_retry(slot: Path, window_attempts: int, limit: int = 2) -> bool:
    used = sum(1 for _ in slot.glob("**/external-retry.json"))
    return window_attempts < limit and used < limit


def attempt_window(
    data: Path,
    campaign: dict,
    slot_name: str,
    mode: str,
    submission: int,
    window: int,
    candidate: Path | None,
    timeout_s: float,
    destination: Path,
) -> dict:
    slot = data / "slots" / slot_name
    slot.mkdir(parents=True, exist_ok=True)
    baseline = load(data / "baseline.json")
    request = request_for(data, campaign, slot_name, mode, submission, window)
    if mode == "preflight":
        request["slot"] = "preparation"
    root = Path(__file__).resolve().parents[2]
    candidate_bytes = candidate.read_bytes() if candidate else None
    for retry in range(3):
        attempt = destination / f"attempt-{retry}"
        attempt.mkdir(parents=True)
        request["request_id"] = UlidFactory(SystemClock()).new()
        write_new(attempt / "request.json", request)
        write_new(attempt / "campaign.json", campaign)
        shutil.copyfile(Path(__file__).with_name("worker.py"), attempt / "worker.py")
        if candidate_bytes is not None:
            (attempt / "candidate.py").write_bytes(candidate_bytes)
        write_new(
            attempt / "identities.json",
            {p.name: identity(p.read_bytes()) for p in attempt.iterdir() if p.is_file()},
        )
        ledger = slot / "queue-events.jsonl" if mode == "development" else None
        if mode == "development":
            timeout_s = min(timeout_s, campaign["budget"]["active_seconds"] - active_s(slot))
        if timeout_s <= 0:
            raise ValueError("Active budget exhausted before submission/retry")
        receipt = execute(root, attempt, request, campaign, baseline, timeout_s, ledger)
        write_new(attempt / "execution.json", receipt)
        result = receipt.get("worker", {"passed": False, "reason": receipt.get("reason")})
        write_new(attempt / "result.json", result)
        if not receipt.get("external_failure"):
            return {**result, "attempt_path": str(attempt)}
        timeout_s -= receipt["active_s"]
        if not allowed_retry(slot, retry):
            # Invalid measurements remain invalid; they do not cancel unrelated sessions.
            # A later submission still consumes its normal budget, with no free retry.
            return {**result, "passed": False, "unresolved": True}
        write_new(
            attempt / "external-retry.json",
            {
                "reason": "documented external transport/interference/unstable comparator",
                "same_candidate": candidate_bytes is None or identity(candidate_bytes),
                "same_window": window,
                "extra_attempt": retry + 1,
            },
        )
    raise RuntimeError("Retry loop exceeded declared bound")


def evaluate(data: Path, slot_name: str, candidate: Path) -> dict:
    campaign = verify_sealed(data)
    slot = data / "slots" / slot_name
    submissions = slot / "submissions"
    existing = sorted(submissions.glob("*"))
    if len(existing) >= campaign["budget"]["target_evaluation_submissions"]:
        raise ValueError("20 submissions exhausted")
    if (slot / "selected-source.json").exists():
        raise ValueError("Source already selected; model discovery is closed")
    if active_s(slot) >= campaign["budget"]["active_seconds"]:
        raise ValueError("1800 active seconds exhausted")
    submission = len(existing) + 1
    destination = submissions / f"{submission:02d}"
    destination.mkdir(parents=True)
    shutil.copyfile(candidate, destination / "candidate.py")
    write_new(
        destination / "submission.json",
        {
            "started_at_s": time.time(),
            "active_s_at_start": active_s(slot),
            "submission": submission,
            "candidate_identity": identity((destination / "candidate.py").read_bytes()),
        },
    )
    result = attempt_window(
        data,
        campaign,
        slot_name,
        "development",
        submission,
        0,
        destination / "candidate.py",
        campaign["budget"]["active_seconds"],
        destination,
    )
    previous = load(existing[-1] / "summary.json") if existing else {}
    same_source = bool(existing) and identity(
        (existing[-1] / "candidate.py").read_bytes()
    ) == identity(candidate.read_bytes())
    result["early_stop_eligible"] = (
        same_source and previous.get("meets_target", False) and result.get("meets_target", False)
    )
    result.update({"finished_at_s": time.time(), "active_s_at_finish": active_s(slot)})
    write_new(destination / "summary.json", result)
    return result


def select(data: Path, slot_name: str, candidate: Path | None) -> dict:
    slot = data / "slots" / slot_name
    fallback = candidate is None
    if candidate is None:
        submitted = sorted((slot / "submissions").glob("*/candidate.py"))
        if not submitted:
            raise ValueError("No selected or submitted source exists")
        candidate = submitted[-1]
    target = slot / "selected-candidate.py"
    with target.open("xb") as stream:
        stream.write(candidate.read_bytes())
    receipt = {
        "candidate_identity": identity(target.read_bytes()),
        "fallback_latest": fallback,
        "selected_at_s": time.time(),
        "source": str(candidate.resolve()),
    }
    write_new(slot / "selected-source.json", receipt)
    return receipt


def confirm(data: Path, slot_name: str) -> dict:
    campaign = verify_sealed(data)
    slot = data / "slots" / slot_name
    selected = load(slot / "selected-source.json")
    candidate = slot / "selected-candidate.py"
    if identity(candidate.read_bytes()) != selected["candidate_identity"]:
        raise ValueError("Selected source changed")
    if not (slot / "model-closed.json").exists():
        raise ValueError("Close persistent model before hidden confirmation")
    remaining = campaign["budget"]["confirmation_worker_active_s"]
    windows = []
    for window in range(1, 4):
        destination = slot / "confirmation" / f"window-{window}"
        result = attempt_window(
            data, campaign, slot_name, "final", 1, window, candidate, remaining, destination
        )
        windows.append(result)
        remaining -= sum(load(p)["active_s"] for p in destination.glob("*/execution.json"))
        if not result.get("window_valid") or remaining <= 0:
            break
    shape_rows = []
    valid = len(windows) == 3 and all(w.get("window_valid") for w in windows)
    if valid:
        for index, shape in enumerate(campaign["task"]["shapes"]):
            originals = [w["rows"][index]["timing"]["original"]["median_us"] for w in windows]
            candidates = [w["rows"][index]["timing"]["candidate"]["median_us"] for w in windows]
            shape_rows.append(
                {
                    "shape": shape,
                    "original_window_medians_us": originals,
                    "candidate_window_medians_us": candidates,
                    "original_median_us": statistics.median(originals),
                    "candidate_median_us": statistics.median(candidates),
                    "original_drift": drift(originals),
                }
            )
        valid = all(r["original_drift"] <= 0.05 for r in shape_rows)
    result = {
        "windows": windows,
        "shape_rows": shape_rows,
        "confirmed": valid,
        "meets_target": valid and all(w.get("meets_target") for w in windows),
        "confirmation_active_s": campaign["budget"]["confirmation_worker_active_s"] - remaining,
        "selected_identity": selected["candidate_identity"],
    }
    write_new(slot / "confirmation.json", result)
    return result


def public(value: Any) -> Any:
    if isinstance(value, dict):
        hidden = {
            "seed",
            "input_checksums",
            "baseline_identity",
            "source_revision",
            "request_identity",
            "worker_identity",
            "attempt_path",
            "traceback",
        }
        return {key: public(item) for key, item in value.items() if key not in hidden}
    if isinstance(value, list):
        return [public(item) for item in value]
    return value
