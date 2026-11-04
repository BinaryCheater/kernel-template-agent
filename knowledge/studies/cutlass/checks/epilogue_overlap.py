"""Replay the source TMEM overlap/release regions and output subtile clipping."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    helper = "python/CuTeDSL/cutlass/utils/blackwell_helpers.py"
    data = (root / helper).read_bytes()
    entry = next(x for x in manifest["files"] if x["path"] == helper)
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    # Execute only the retained pure-Python integer helper, without importing CUDA/DSL.
    function = next(
        x
        for x in ast.parse(data).body
        if isinstance(x, ast.FunctionDef) and x.name == "compute_epilogue_tile_size"
    )
    function.decorator_list = []
    code = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    ns: dict = {}
    exec(compile(ast.fix_missing_locations(code), str(root / helper), "exec"), ns)
    tile_size = ns["compute_epilogue_tile_size"]
    overlaps = live = 0
    for width in (64, 128, 192, 256):
        for bits in (8, 16, 32):
            for two_cta in (False, True):
                epi_m, epi_n = tile_size(128, width, two_cta, bits, None, False)
                assert epi_m == 128 and width % epi_n == 0
                if width == 256:
                    sf_columns = (128 // 32 + 256 // 32) * 4
                    slot_stride = 256 - sf_columns
                    release_at = (sf_columns + epi_n - 1) // epi_n - 1
                    for slot in (0, 1):
                        base, next_base = slot * slot_stride, (1 - slot) * slot_stride
                        read = set()
                        for i in range(width // epi_n):
                            subtile = width // epi_n - 1 - i if slot == 0 else i
                            read.update(range(base + subtile * epi_n, base + (subtile + 1) * epi_n))
                            if i == release_at:
                                unread = set(range(base, base + 256)) - read
                                assert not unread.intersection(range(next_base, next_base + 256))
                                overlaps += 1
                        assert read == set(range(base, base + 256))
                for valid_m, valid_n in ((128, width), (16, min(16, width)), (112, width - 16)):
                    coordinates = set()
                    for subtile in range(width // epi_n):
                        for row in range(epi_m):
                            for col in range(subtile * epi_n, (subtile + 1) * epi_n):
                                # Descriptor-domain oracle; generated TMA behavior is separate.
                                if row < valid_m and col < valid_n:
                                    assert (row, col) not in coordinates
                                    coordinates.add((row, col))
                    assert len(coordinates) == valid_m * valid_n
                    live += len(coordinates)
    result = {
        "outcome": "PASS",
        "early_release_regions": overlaps,
        "live_output_domain_coordinates": live,
        "witness_n256": {
            "sf_columns": 48,
            "acc_slots": [[0, 256], [208, 464]],
            "scale_region": [464, 512],
            "subtile_n": 32,
            "release_after_subtiles": 2,
        },
        "boundary": (
            "Interval safety and descriptor-domain oracle; no TMEM/TMA hardware evidence."
        ),
    }
    (STUDY / "checks/epilogue-overlap-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: overlap regions are consumed before early release; output domains partition.\n"
    )


if __name__ == "__main__":
    main()
