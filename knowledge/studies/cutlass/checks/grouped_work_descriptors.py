"Replay online group inverse-prefix decoding and per-group descriptor handoffs."

from __future__ import annotations

import bisect
import collections
import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def scan_window(counts: list[int], index: int, first: int = 0, prefix: int = 0) -> tuple[int, int]:
    "Scalar counterpart of inclusive prefix, >= ballot and popcount in windows32."
    while first < len(counts):
        ends, total = [], prefix
        for lane in range(32):
            total += counts[first + lane] if first + lane < len(counts) else 0
            ends.append(total)
        hit = sum(index >= end for end in ends)
        if hit < 32:
            group = first + hit
            start = prefix if hit == 0 else ends[hit - 1]
            return group, index - start
        first, prefix = first + 32, ends[-1]
    return -1, -1


def main() -> None:
    domains = queries = 0
    for groups, pattern in itertools.product((1, 2, 31, 32, 33, 64, 65, 97), range(6)):
        counts = [((i * (pattern + 1) + pattern) % 7) for i in range(groups)]
        ends = list(itertools.accumulate(counts))
        for stride in (1, 2, 5, 17):
            for worker in range(stride):
                group, start, finish = 0, 0, 0
                for index in range(worker, sum(counts), stride):
                    expected_group = bisect.bisect_right(ends, index)
                    expected_start = 0 if expected_group == 0 else ends[expected_group - 1]
                    expected = expected_group, index - expected_start
                    assert scan_window(counts, index) == expected
                    if not (start <= index < finish):
                        group, local = scan_window(counts, index, group, start)
                        start = index - local
                        finish = start + counts[group]
                    assert (group, index - start) == expected
                    queries += 1
                domains += 1
        assert scan_window(counts, sum(counts)) == (-1, -1)
    # Group counts are M/N tiles: zero K still owns output tiles and must not vanish.
    shapes = [(128, 128, 768), (256, 128, 0), (128, 256, 1536)]
    counts = [((m + 127) // 128) * ((n + 127) // 128) for m, n, _ in shapes]
    ends = list(itertools.accumulate(counts))
    visits, input_updates, output_updates, zero_outputs = [], 0, 0, 0
    output_coordinates = collections.Counter()
    for worker in range(2):
        previous = -1
        descriptors = {}
        for index in range(worker, sum(counts), 2):
            group = bisect.bisect_right(ends, index)
            m, n, k = shapes[group]
            changed = group != previous
            if changed:
                descriptors["C"] = (group, (m, n), (n, 1))
                output_updates += 1
                if k:
                    descriptors["A"] = (group, (m, k), (k, 1))
                    descriptors["B"] = (group, (n, k), (k, 1))
                    descriptors["SFA"] = (group, (m, k // 32), "blocked128x4")
                    descriptors["SFB"] = (group, (n, k // 32), "blocked128x4")
                    input_updates += 1
            assert descriptors["C"][0] == group
            local = index - (0 if group == 0 else ends[group - 1])
            tiles_m = (m + 127) // 128
            tile_m, tile_n = local % tiles_m, local // tiles_m
            for row, col in itertools.product(range(128), repeat=2):
                logical = group, tile_m * 128 + row, tile_n * 128 + col
                assert logical[1] < m and logical[2] < n
                output_coordinates[logical] += 1
            if k:
                assert all(descriptors[operand][0] == group for operand in ("A", "B", "SFA", "SFB"))
            else:
                zero_outputs += 1
            visits.append((worker, index, group, k))
            previous = group
    assert sorted(index for _, index, _, _ in visits) == list(range(5))
    assert zero_outputs == 2
    assert len(output_coordinates) == sum(m * n for m, n, _ in shapes)
    assert set(output_coordinates.values()) == {1}
    cache_reuse_domains = same_key_skips = 0
    for grid in (1, 2, 3, 5):
        for worker in range(grid):
            previous, published_c = -1, None
            for index in range(worker, sum(counts), grid):
                group = bisect.bisect_right(ends, index)
                if group != previous:
                    published_c = (group, shapes[group][:2])
                else:
                    same_key_skips += 1
                assert published_c == (group, shapes[group][:2])
                previous = group
        cache_reuse_domains += 1
    assert same_key_skips > 0
    # Incorrectly reusing the input K>0 gate for C rebinding leaves zero-K outputs stale.
    stale_c_group = None
    stale_witness = None
    for group, (_, _, k) in enumerate(shapes[:2]):
        if k:
            stale_c_group = group
        if stale_c_group != group:
            stale_witness = {"expected_group": group, "descriptor_group": stale_c_group}
    assert stale_witness == {"expected_group": 1, "descriptor_group": 0}
    # State retained after decoding index4 in counts[2,3] cannot answer backward index1.
    cached_group, local = scan_window([2, 3], 4)
    cached_start = 4 - local
    backwards = scan_window([2, 3], 1, cached_group, cached_start)
    independent = scan_window([2, 3], 1)
    assert backwards == (1, -1) and independent == (0, 1)
    # A shared descriptor generation may not be overwritten until old readers retire.
    lifetime_domains = 0
    for users in range(1, 9):
        pending = set(range(users))
        for user in range(users):
            assert bool(pending)
            pending.remove(user)
            can_rebind = not pending
            assert can_rebind == (user == users - 1)
        lifetime_domains += 1
    result = {
        "outcome": "PASS",
        "decode_worker_domains": domains,
        "decoded_queries": queries,
        "descriptor_witness": {
            "shapes_mnk": shapes,
            "group_tile_counts": counts,
            "input_tuple_updates": input_updates,
            "output_descriptor_updates": output_updates,
            "zero_k_output_tiles": zero_outputs,
            "visits": visits,
        },
        "descriptor_lifetime_domains": lifetime_domains,
        "logical_output_coordinates": len(output_coordinates),
        "descriptor_cache_domains": cache_reuse_domains,
        "same_key_update_skips": same_key_skips,
        "stale_output_descriptor_witness": stale_witness,
        "backward_query_witness": {"cached": backwards, "independent": independent},
        "rejected_misuses": [
            "backward query with monotone state",
            "skip C update for K0",
            "overwrite descriptor with live readers",
        ],
        "boundary": (
            "Group/metadata/source-lifetime models; no TMA descriptor encoding or GPU execution."
        ),
    }
    (STUDY / "checks/grouped-work-descriptors-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: grouped inverse-prefix decode, descriptor ownership and explicit K0 output path.\n"
    )


if __name__ == "__main__":
    main()
