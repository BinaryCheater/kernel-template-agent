"""Small exact matrix witness for chronological right-affine composition."""

import json
from pathlib import Path


def mm(a, b):
    return [
        [sum(x * y for x, y in zip(row, col, strict=False)) for col in zip(*b, strict=False)]
        for row in a
    ]


def add(a, b):
    return [[x + y for x, y in zip(ra, rb, strict=False)] for ra, rb in zip(a, b, strict=False)]


def apply(s, lane_or_matrix, c):
    return add(mm(s, lane_or_matrix), c)


maps = [
    ([[1, 2], [0, 1]], [[1, 0], [0, 2]]),
    ([[2, 0], [1, 1]], [[0, 3], [1, 0]]),
    ([[0, 1], [1, 0]], [[2, 1], [0, 1]]),
]
s0 = [[2, 3], [5, 7]]
expected = s0
for lane_or_matrix, c in maps:
    expected = apply(expected, lane_or_matrix, c)
cases = 0
for size in [1, 2, 3]:
    pieces = []
    for start in range(0, len(maps), size):
        lane_or_matrix, c = [[1, 0], [0, 1]], [[0, 0], [0, 0]]
        for lj, cj in maps[start : start + size]:
            lane_or_matrix, c = mm(lane_or_matrix, lj), add(mm(c, lj), cj)
        pieces.append((lane_or_matrix, c))
    state = s0
    for lane_or_matrix, c in pieces:
        state = apply(state, lane_or_matrix, c)
    assert state == expected
    cases += 1
reverse = s0
for lane_or_matrix, c in reversed(maps):
    reverse = apply(reverse, lane_or_matrix, c)
assert reverse != expected
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "piece_sizes": [1, 2, 3],
            "chronological_result": expected,
            "reversed_result": reverse,
            "scope": "Exact integer2x2 right-affine witness; no native numeric or progress check",
        },
        indent=2,
    )
    + "\n"
)
