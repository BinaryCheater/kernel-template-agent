"Marlin stripe equations versus independent interval intersections; no CUDA execution."

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from packing_decode import half, permutation

STUDY = Path(__file__).resolve().parents[1]


def div(n: int, d: int) -> int:
    return (n + d - 1) // d


def metadata(cta: int, column: int, start: int, units: int, ktiles: int) -> tuple[int, int]:
    count, index = 1, 0
    first = units * div(ktiles * column, units)
    if first <= ktiles * (column + 1):
        offset = first - ktiles * column
        count = div(ktiles - offset, units) + int(offset > 0)
        delta = units * cta - first
        if delta < 0 or (offset == 0 and delta == 0):
            index = count - 1
        else:
            index = count - 1 - delta // units - int(offset > 0)
    assert start >= 0
    return count, index


def check_stripes() -> dict[str, object]:
    domains, segments = 0, 0
    witness = []
    for ktiles, columns, parallel, grid, scale_tiles in itertools.product(
        range(1, 18), (1, 2, 3, 8), (1, 2, 4), (1, 2, 3, 5, 16, 132), (1, 2)
    ):
        if ktiles % scale_tiles:
            continue
        total = ktiles * columns * parallel
        units = scale_tiles * div(div(total, grid), scale_tiles)
        owners = {}
        for column in range(columns * parallel):
            lo, hi = column * ktiles, (column + 1) * ktiles
            contributors = [
                cta for cta in range(grid) if max(lo, cta * units) < min(hi, (cta + 1) * units)
            ]
            assert contributors
            for position, cta in enumerate(contributors):
                start = max(lo, cta * units)
                end = min(hi, (cta + 1) * units)
                count, index = metadata(cta, column, start - lo, units, ktiles)
                assert count == len(contributors)
                assert index == len(contributors) - 1 - position
                assert start % scale_tiles == end % scale_tiles == 0
                for unit in range(start, end):
                    assert unit not in owners
                    owners[unit] = cta
                segments += 1
        assert sorted(owners) == list(range(total))
        domains += 1
    ktiles, columns, grid = 8, 4, 5
    units = div(ktiles * columns, grid)
    for cta in range(grid):
        lo, hi = max(0, cta * units), min(ktiles, (cta + 1) * units)
        if lo < hi:
            count, index = metadata(cta, 0, lo, units, ktiles)
            witness.append(
                {"cta": cta, "k_tile_range": [lo, hi], "slice_count": count, "slice_idx": index}
            )
    return {
        "stripe_domains": domains,
        "slice_segments": segments,
        "witness": {
            "k": 1024,
            "n": 512,
            "grid": grid,
            "units_per_cta": units,
            "first_column_contributors": witness,
        },
    }


def check_ordered_rounding() -> dict[str, object]:
    # Source indices visit higher K partitions first; lower K CTA commits last.
    by_k = [-2048.0, 1.0, 2048.0]

    def combine(values: list[float]) -> float:
        saved = half(values[0])
        for value in values[1:-1]:
            saved = half(saved + value)
        return half(saved + values[-1])

    source, increasing = combine(list(reversed(by_k))), combine(by_k)
    assert source == 0 and increasing == 1
    # Locks count intermediate publications; last resets before final output permutation/store.
    states = [0]
    for index in range(3):
        assert states[-1] == index
        states.append(0 if index == 2 else index + 1)
    assert states == [0, 1, 2, 0]
    return {
        "by_increasing_k_partials": by_k,
        "source_reverse_k_result": source,
        "incorrect_increasing_k_result": increasing,
        "lock_states": states,
        "lifetime_boundary": (
            "Reset precedes final output store; it does not authorize concurrent reuse."
        ),
    }


def check_swizzle_and_scale() -> dict[str, int]:
    coordinates, scale_tiles_checked = 0, 0
    for kb, mb in ((8, 1), (4, 1), (4, 2), (4, 3), (4, 4)):
        stride = 2 * kb
        size = 16 * mb * stride
        source = [(stride * (i // stride) + i % stride) ^ (i // stride) for i in range(size)]
        assert sorted(source) == list(range(size))
        # Reads and writes transform the original logical index, not an already transformed address.
        stored = {source[i]: i for i in range(size)}
        assert [stored[source[i]] for i in range(size)] == list(range(size))
        coordinates += size
        scale_tiles = 8 // kb
        assert 4 % scale_tiles == 0
        for tile in range(32):
            pipe = tile % 4
            source_slot = scale_tiles * (pipe // scale_tiles)
            assert source_slot == (tile // scale_tiles * scale_tiles) % 4
            scale_tiles_checked += 1
    return {"swizzled_int4_coordinates": coordinates, "scale_ring_tiles": scale_tiles_checked}


def check_b_fragments() -> dict[str, int]:
    perm = permutation()
    checked = 0
    for tk, tn, n in ((128, 128, 256), (64, 256, 256), (64, 256, 512)):
        k = 256
        tiled = [
            (kb * 16 + row, nb * 16 + col)
            for kb in range(k // 16)
            for nb in range(n // 16)
            for row in range(16)
            for col in range(16)
        ]
        packed = [tiled[base + q] for base in range(0, k * n, 1024) for q in perm]
        b_stride = tn // 2
        nibble_order = ((0, 4, 1, 5), (2, 6, 3, 7))
        for ktile, ntile, thread, step, word, fragment, elem in itertools.product(
            range(k // tk), range(n // tn), range(256), range(2), range(4), range(2), range(4)
        ):
            block_k = thread // b_stride + step * (256 // b_stride)
            vector = (
                (ktile * (tk // 16) + block_k) * (n // 2) + ntile * b_stride + thread % b_stride
            )
            actual = packed[vector * 32 + word * 8 + nibble_order[fragment][elem]]
            lane = thread % 32
            expected = (
                ktile * tk + block_k * 16 + 2 * (lane % 4) + elem % 2 + 8 * (elem // 2),
                ntile * tn + (thread % b_stride // 32) * 64 + word * 16 + fragment * 8 + lane // 4,
            )
            assert actual == expected
            checked += 1
    grouped = [i + 8 * j for i in range(8) for j in range(8)]
    scales = 0
    for tn, thread, word, fragment in itertools.product((128, 256), range(256), range(4), range(2)):
        s_read = 8 * ((thread // 32) % (tn // 64)) + (thread % 32) // 4
        packed_index = s_read * 8 + word * 2 + fragment
        base, inner = divmod(packed_index, 64)
        actual = base * 64 + grouped[inner]
        expected = (
            ((thread // 32) % (tn // 64)) * 64 + word * 16 + fragment * 8 + (thread % 32) // 4
        )
        assert actual == expected
        scales += 1
    return {"mma_b_coordinates": checked, "grouped_scale_column_bindings": scales}


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    body = (root / "marlin/marlin_cuda_kernel.cu").read_text()
    for marker in (
        "slice_idx = slice_count - 1 - delta_first / iters",
        "barrier_acquire(&locks[slice_col], slice_idx)",
        "barrier_release(&locks[slice_col], last)",
        "if (last) // only the last block",
    ):
        assert marker in body
    result = {
        "outcome": "PASS",
        "stripes": check_stripes(),
        "ordered_rounding": check_ordered_rounding(),
        "layout": check_swizzle_and_scale(),
        "b_fragments": check_b_fragments(),
        "boundary": (
            "Source indices and binary16 reduction policy only; no device visi"
            "bility or liveness proof."
        ),
    }
    (STUDY / "checks/striped-combine-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: stripe ownership, reverse-K commit metadata, half rounding and shared swizzle.\n"
    )


if __name__ == "__main__":
    main()
