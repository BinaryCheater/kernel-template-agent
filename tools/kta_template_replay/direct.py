"""Private file staging/collection for authorized manual remote experimental runs."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from .coordinator import reserve
from .manifest import identity, load, write_new


def seal_preflight(run: Path, result: Path) -> dict[str, Any]:
    receipt = load(result)
    expected = identity((run / "manifest.json").read_bytes())
    if receipt.get("manifest_identity") != expected or not receipt.get("preflight"):
        raise ValueError("Preflight receipt does not match frozen manifest")
    if receipt.get("passed"):
        shapes = load(run / "manifest.json")["shapes"]
        if [row["shape"] for row in receipt.get("rows", [])] != shapes:
            raise ValueError("Preflight receipt omitted prescribed shapes")
        if not all(row.get("production_checks") for row in receipt["rows"]):
            raise ValueError("Preflight receipt omitted production checks")
    write_new(run / "preflight.json", receipt)
    return receipt


def stage(run: Path, candidate: Path, agent_active_s: float) -> Path:
    manifest = load(run / "manifest.json")
    preflight = load(run / "preflight.json")
    if not preflight.get("passed") or preflight["manifest_identity"] != identity(
        (run / "manifest.json").read_bytes()
    ):
        raise ValueError("Successful unchanged preflight required")
    if len(manifest["shapes"]) > manifest["max_shapes_per_batch"]:
        raise ValueError("Shape budget exceeded")
    attempt = reserve(run, manifest, agent_active_s)
    shutil.copyfile(candidate, attempt / "candidate.py")
    shutil.copyfile(Path(__file__).with_name("worker.py"), attempt / "worker.py")
    # Fresh correctness inputs at every attempt; timing inputs remain fixed across modes.
    manifest["private_seeds"] = [
        int(hashlib.sha256(f"{seed}:{attempt.name}".encode()).hexdigest()[:15], 16)
        for seed in manifest["private_seeds"]
    ]
    write_new(attempt / "manifest.json", manifest)
    write_new(
        attempt / "identity.json",
        {
            "base_manifest": identity((run / "manifest.json").read_bytes()),
            "candidate": identity((attempt / "candidate.py").read_bytes()),
            "worker": identity((attempt / "worker.py").read_bytes()),
            "manifest": identity((attempt / "manifest.json").read_bytes()),
            "code_state": "uncommitted experimental snapshot; not clean WP evidence",
        },
    )
    return attempt


def collect(attempt: Path, worker_result: Path, wall_s: float, queue_s: float) -> dict[str, Any]:
    if not 0 <= queue_s <= wall_s < float("inf"):
        raise ValueError("Expected finite 0 <= queue_s <= wall_s")
    result = load(worker_result)
    snapshot = load(attempt / "identity.json")
    if result.get("manifest_identity") != snapshot["manifest"]:
        raise ValueError("Result manifest differs from staged payload")
    if result.get("candidate_identity") != snapshot["candidate"]:
        raise ValueError("Result candidate differs from retained candidate")
    write_new(
        attempt / "execution.json",
        {
            "wall_s": wall_s - queue_s,
            "queue_s": queue_s,
            "transport": "authorized manual remote run",
            "worker_wall_s": result["worker_wall_s"],
            "measurement_conditions": "see retained remote environment and scheduling records",
        },
    )
    write_new(attempt / "result.json", result)
    return result
