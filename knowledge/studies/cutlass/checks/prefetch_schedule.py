"""Replay the pinned TMA cache-prefetch indices, not generated device code."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
KERNEL = (
    "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/de"
    "nse_blockscaled_gemm_persistent_prefetch.py"
)


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    entries = {entry["path"]: entry for entry in manifest["files"]}
    paths = [KERNEL, "python/CuTeDSL/cutlass/pipeline/helpers.py"]
    for name in paths:
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == entries[name]["sha256"]
    body = (root / KERNEL).read_text()
    assert "future_k_tile = ab_producer_state.count + self.prefetch_dist" in body
    assert "min(self.prefetch_dist, k_tile_cnt)" in body
    assert "self.prefetch_enabled = self.prefetch_dist > 0" in body
    domains = hints = 0
    for k_count, stages, parameter in itertools.product(
        range(65), range(1, 9), [None, -2, -1, *range(17)]
    ):
        distance = stages if parameter is None else parameter
        hinted = []
        if distance > 0:
            hinted.extend(range(min(distance, k_count)))
        # Count resets per persistent output tile; slot and phase continue globally.
        slot, phase = 0, 1
        for output_tile in range(3):
            loaded = []
            tile_hints = list(hinted)
            for k_tile in range(k_count):
                loaded.append(k_tile)
                assert slot == (output_tile * k_count + k_tile) % stages
                assert phase == (1 ^ (((output_tile * k_count + k_tile) // stages) % 2))
                if distance > 0 and k_tile < k_count - distance:
                    tile_hints.append(k_tile + distance)
                slot += 1
                if slot == stages:
                    slot, phase = 0, phase ^ 1
            assert loaded == list(range(k_count))
            assert tile_hints == (list(range(k_count)) if distance > 0 else [])
            hints += 4 * len(tile_hints)  # A, B, SFA and SFB each get a hint.
        domains += 1
    result = {
        "outcome": "PASS",
        "domains": domains,
        "persistent_output_tiles_per_domain": 3,
        "logical_operand_prefetches": hints,
        "witness": {
            "k_tiles": 5,
            "ab_stages": 2,
            "prefetch_distance": 4,
            "initial_hints": [0, 1, 2, 3],
            "rolling_hint_after_load0": 4,
            "ab_load_slots": [0, 1, 0, 1, 0],
        },
        "boundary": (
            "Source-equation replay only; cache residency, emitted ISA and speed unmeasured."
        ),
    }
    (STUDY / "checks/prefetch-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: prefetch coverage, tail bounds and independent AB slot/phase progression.\n"
    )


if __name__ == "__main__":
    main()
