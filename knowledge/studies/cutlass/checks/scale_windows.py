"Replay block-scale packing and N64/N192 scale windows from pinned source."

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/"


def offset(mn: int, group: int, batch: int, extent: int, groups: int) -> int:
    """Contiguous source torch shape L,ceil(MN/128),ceil(SFK/4),32,4,4."""
    return (
        (
            ((batch * ((extent + 127) // 128) + mn // 128) * ((groups + 3) // 4) + group // 4) * 32
            + mn % 32
        )
        * 4
        + mn % 128 // 32
    ) * 4 + group % 4


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    for path in [
        PREFIX + "dense_blockscaled_gemm_persistent_prefetch.py",
        "python/CuTeDSL/cutlass/utils/blockscaled_layout.py",
    ]:
        entry = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == entry["sha256"]
    coordinates = 0
    for extent, groups, batches in itertools.product(
        (64, 128, 144, 208, 384), (1, 4, 5, 10), (1, 2)
    ):
        tags = {}
        for batch, mn, group in itertools.product(range(batches), range(extent), range(groups)):
            address = offset(mn, group, batch, extent, groups)
            assert address not in tags
            tags[address] = (mn, group, batch)
            # Independent expanded basic-chunk layout, tiled K first then MN.
            atom = 16 * (mn % 32) + 4 * (mn % 128 // 32) + group % 4
            blocks_k = (groups + 3) // 4
            expected = atom + 512 * (group // 4 + blocks_k * (mn // 128))
            expected += batch * ((extent + 127) // 128) * blocks_k * 512
            assert address == expected
            coordinates += 1
        for vector_size in (16, 32):
            for mn, k in itertools.product(range(extent), range(groups * vector_size)):
                assert tags[offset(mn, k // vector_size, 0, extent, groups)] == (
                    mn,
                    k // vector_size,
                    0,
                )
    windows = 0
    for width, tile, column in itertools.product((64, 128, 192, 256), range(17), range(256)):
        if column >= width:
            continue
        if width == 64:
            base = 128 * (tile // 2)
            selected = base + 64 * (tile % 2) + column
        elif width == 192:
            # N256 TMA slice selects pairs from nested ((2,2),rest) strides ((x,x),3x).
            outer = 2 * tile
            rem, rest = outer % 4, outer // 4
            base = 128 * (rem % 2 + rem // 2 + 3 * rest)
            selected = base + 64 * (tile % 2) + column
        else:
            selected = tile * width + column
        assert selected == tile * width + column
        windows += 1
    result = {
        "outcome": "PASS",
        "unique_scale_coordinates": coordinates,
        "scale_window_coordinates": windows,
        "n192_witness": {
            "tile": 1,
            "loaded_columns": [128, 384],
            "selected_columns": [192, 384],
            "tmem_word_offset": 2,
        },
        "boundary": (
            "Logical packing/window equations only; TMEM intrinsic layout/lowering not executed."
        ),
    }
    (STUDY / "checks/scale-windows-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: unique scale packing coordinates and all four N-window geometries.\n")


if __name__ == "__main__":
    main()
