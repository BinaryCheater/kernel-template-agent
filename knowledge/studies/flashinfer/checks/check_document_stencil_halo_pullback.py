"""Exact document stencil/transpose identities across short rank fragments."""

import itertools
import json
from fractions import Fraction as F
from pathlib import Path


def compositions(n):
    for cuts in itertools.product([False, True], repeat=n - 1):
        starts = [0, *(i + 1 for i, cut in enumerate(cuts) if cut), n]
        yield list(itertools.pairwise(starts))


def owner(fragments, token):
    return next(f for f, (a, b) in enumerate(fragments) if a <= token < b)


if __name__ == "__main__":
    cases = transpose_edges = multiconsumer_rows = short_multisource_halos = 0
    for n in range(1, 9):
        for fragments in compositions(n):
            for recipe in range(2):
                ranks = [(f * 3 + recipe) % 4 for f in range(len(fragments))]
                x = [F(2 * t - n, 3) for t in range(n)]
                dpre = [F(3 * t + 1, 7) for t in range(n)]
                taps = [F(2), F(-1), F(3, 2), F(1, 3)]
                full_y = [
                    sum(taps[3 - lag] * x[t - lag] for lag in range(4) if t >= lag)
                    for t in range(n)
                ]
                local_y, routed_dx = [None] * n, [None] * n
                dweight = [[F(0)] * 4 for _ in range(4)]
                consumers = {}
                for f, (a, b) in enumerate(fragments):
                    history = {}
                    for t in range(a - 3, a):
                        if t >= 0:
                            history[t] = x[t]
                            assert fragments[owner(fragments, t)][0] <= t
                    if len({owner(fragments, t) for t in history}) > 1:
                        short_multisource_halos += 1
                    tails = {m: dpre[b + m] for m in range(3) if b + m < n}
                    for m in tails:
                        t = b + m
                        source = owner(fragments, t)
                        assert 0 <= t - fragments[source][0] < 3
                        consumers.setdefault(t, set()).add(f)
                    for t in range(a, b):
                        window = [
                            F(0)
                            if t - lag < 0
                            else (x[t - lag] if t - lag >= a else history[t - lag])
                            for lag in range(4)
                        ]
                        local_y[t] = sum(taps[3 - lag] * window[lag] for lag in range(4))
                        for lag in range(4):
                            dweight[ranks[f]][3 - lag] += dpre[t] * window[lag]
                        value = F(0)
                        for out in range(t, min(n, t + 4)):
                            incoming = dpre[out] if out < b else tails[out - b]
                            value += taps[3 - (out - t)] * incoming
                            transpose_edges += 1
                        routed_dx[t] = value
                assert local_y == full_y
                oracle_dx = [F(0)] * n
                oracle_dw = [F(0)] * 4
                for out in range(n):
                    for tap in range(4):
                        source = out + tap - 3
                        if source >= 0:
                            oracle_dx[source] += dpre[out] * taps[tap]
                            oracle_dw[tap] += dpre[out] * x[source]
                assert routed_dx == oracle_dx
                assert [sum(row[tap] for row in dweight) for tap in range(4)] == oracle_dw
                for predecessor_set in consumers.values():
                    assert 1 <= len(predecessor_set) <= 3
                    multiconsumer_rows += len(predecessor_set) > 1
                cases += 1
    # A three-row history can come from three one-token fragments, not one neighbor.
    fragments = [(0, 1), (1, 2), (2, 3), (3, 5)]
    assert {owner(fragments, t) for t in range(3)} == {0, 1, 2}
    # Channel-packing preserves identifiable operand columns under a shuffled row route.
    widths = [2, 3, 1]
    channel_checks = 0
    for route in [(2, 0, 3, 1), (1, 1, 0, 3), ()]:
        rows = [
            [(operand, row, c) for operand, width in enumerate(widths) for c in range(width)]
            for row in range(4)
        ]
        received = [rows[r] for r in route]
        offset = 0
        for operand, width in enumerate(widths):
            for slot, row in enumerate(route):
                assert received[slot][offset : offset + width] == [
                    (operand, row, c) for c in range(width)
                ]
                channel_checks += width
            offset += width
    result = {
        "passed": True,
        "native_execution": False,
        "fragmented_stencil_cases": cases,
        "transpose_edges": transpose_edges,
        "multiconsumer_dpre_rows": multiconsumer_rows,
        "multisource_history_cases": short_multisource_halos,
        "packed_channel_coordinates": channel_checks,
        "oracle": "Independent whole-document stencil and its explicit edge transpose",
        "open": [
            "native SiLU approximation and FP32 order",
            "collective progress and pool lifetime",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
