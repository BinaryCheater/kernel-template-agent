"""Frozen, explicit replay controls; private validation seeds never enter packages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHAPES = {
    "gemm": [[1024, 1024, 1024], [32, 4096, 4096], [127, 1000, 769]],
    "attention": [[1, 8, 512, 64], [1, 8, 2048, 128], [1, 8, 513, 64]],
}


def identity(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze(case: str, environment: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    """Tolerance is a launch-time hypothesis; oracle smoke must confirm it unchanged."""
    if case not in SHAPES or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Choose a supported case and at least two distinct private seeds")
    for key in ("target", "container", "requires", "language", "smoke_record"):
        if not environment.get(key):
            raise ValueError(f"Missing environment field: {key}")
    if not re.search(r"(?:^|@)sha256:[0-9a-f]{64}$", environment["container"]):
        raise ValueError("Container must be pinned by digest")
    return {
        "schema": 1,
        "case": case,
        "shapes": [list(shape) for shape in SHAPES[case]],
        "environment": environment,
        "dtype": "float16",
        "accumulation": "float32",
        "output": "float16",
        "rtol": 0.01,
        "atol": 0.01,
        "private_seeds": seeds,
        "timing_seed": seeds[0] ^ 0x5A17,
        "validation_seed_policy": "private base seeds, sha256(base:attempt) for fresh batches",
        "patterns": ["normal", "magnitude", "cancellation", "mask"]
        + (["negative_logits"] if case == "attention" else []),
        "warmup": 10,
        "repetitions": 50,
        "cache_policy": "warm-input-reuse",
        "timing_boundary": "entry(inputs)->new_output; includes packing and dispatch",
        "max_batches": 20,
        "max_shapes_per_batch": 3,
        "active_budget_s": 3600,
        "performance_aim": 0.90,
        "language": environment["language"],
        "isolation": "instruction-only; read-only package files are not an OS sandbox",
    }


def write_new(path: Path, doc: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
