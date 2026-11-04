"Grid/tile live-row recovery and finite-order reduction partition distinctions."

import json
import struct
from fractions import Fraction as F
from pathlib import Path


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def tree32(values):
    lanes = [0.0] * 32
    for r, value in enumerate(values):
        lane = r % 32
        lanes[lane] = f32(lanes[lane] + value)
    half = 16
    while half:
        lanes = [f32(lanes[i] + lanes[i + half]) for i in range(half)]
        half //= 2
    return lanes[0]


def partitioned_sum(values, grid):
    length = (len(values) + grid - 1) // grid
    partials = []
    for rank in range(grid):
        acc = 0.0
        for value in values[rank * length : (rank + 1) * length]:
            acc = f32(acc + value)
        partials.append(acc)
    return tree32(partials)


if __name__ == "__main__":
    layouts = token_partitions = holes = 0
    for tiles in range(1, 9):
        for grid in range(tiles, 41):
            capacity = (grid + tiles - 1) // tiles
            written = {(b // tiles, b % tiles) for b in range(grid)}
            for tile in range(tiles):
                live = (grid - tile + tiles - 1) // tiles
                assert {(r, tile) for r in range(live)} == {p for p in written if p[1] == tile}
                holes += capacity - live
                for substreams in [1, 2, 4]:
                    for tokens in [0, 1, 3, 7, 31]:
                        streams = live * substreams
                        length = (tokens + streams - 1) // streams
                        observed = []
                        partials = []
                        for rank in range(live):
                            partial = F(0)
                            for sub in range(substreams):
                                stream = rank * substreams + sub
                                start, end = stream * length, min((stream + 1) * length, tokens)
                                for token in range(start, end):
                                    observed.append(token)
                                    partial += F(3 * token - 2, 7)
                            partials.append(partial)
                        assert observed == list(range(tokens))
                        assert sum(partials) == sum(F(3 * token - 2, 7) for token in range(tokens))
                        token_partitions += 1
            layouts += 1
    # Capacity is not the live count for every tile.
    grid, tiles = 5, 3
    assert (grid + tiles - 1) // tiles == 2
    assert (1, 2) not in {(b // tiles, b % tiles) for b in range(grid)}
    # The same lane order can be realized by inline shuffle or shared final reduction.
    for count in [1, 2, 3, 31, 32, 33, 65, 97]:
        values = [f32((i % 7 - 3) / 8) for i in range(count)]
        per_lane = [0.0] * 32
        for lane in range(32):
            for row in range(lane, count, 32):
                per_lane[lane] = f32(per_lane[lane] + values[row])
        stride = 16
        while stride:
            old = per_lane.copy()
            per_lane = [f32(old[lane] + old[lane ^ stride]) for lane in range(32)]
            stride //= 2
        assert per_lane[0] == tree32(values)
    values = [float(2**24), 1.0, float(-(2**24)), 1.0]
    two, four = partitioned_sum(values, 2), partitioned_sum(values, 4)
    assert two == 1.0 and four == 2.0
    result = {
        "passed": True,
        "native_execution": False,
        "grid_tile_layouts": layouts,
        "exact_token_partitions": token_partitions,
        "unread_capacity_holes": holes,
        "partition_rounding_witness": {"two_partials": two, "four_partials": four},
        "open": [
            "cooperative residency/progress",
            "PDL completion visibility",
            "native partial arithmetic",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
