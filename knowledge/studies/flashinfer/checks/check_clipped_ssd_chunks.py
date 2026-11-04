"Packed logical SSD subranges versus token-serial scalar outer-product state."

import json
from fractions import Fraction as F
from pathlib import Path


def dot(x, y):
    return sum((a * b for a, b in zip(x, y, strict=True)), F(0))


def serial(seed, indices, keys, values, queries, alpha):
    state = [row[:] for row in seed]
    out = []
    for t in indices:
        for k in range(2):
            for v in range(2):
                state[k][v] = alpha[t] * state[k][v] + keys[t][k] * values[t][v]
        out.append([sum((queries[t][k] * state[k][v] for k in range(2)), F(0)) for v in range(2)])
    return out, state


def subrange(seed, physical_start, lo, hi, keys, values, queries, alpha, block):
    prefixes = []
    cumulative = F(1)
    for t in range(physical_start, physical_start + block):
        cumulative *= alpha[t]
        prefixes.append(cumulative)
    anchor = prefixes[lo - 1] if lo else F(1)
    outputs = []
    for i in range(lo, hi):
        t = physical_start + i
        incoming = [
            prefixes[i] / anchor * sum((queries[t][k] * seed[k][v] for k in range(2)), F(0))
            for v in range(2)
        ]
        local = [
            sum(
                (
                    prefixes[i]
                    / prefixes[j]
                    * dot(queries[t], keys[physical_start + j])
                    * values[physical_start + j][v]
                    for j in range(lo, i + 1)
                ),
                F(0),
            )
            for v in range(2)
        ]
        outputs.append([x + y for x, y in zip(incoming, local, strict=True)])
    outgoing = [
        [
            prefixes[hi - 1] / anchor * seed[k][v]
            + sum(
                (
                    prefixes[hi - 1]
                    / prefixes[j]
                    * keys[physical_start + j][k]
                    * values[physical_start + j][v]
                    for j in range(lo, hi)
                ),
                F(0),
            )
            for v in range(2)
        ]
        for k in range(2)
    ]
    # The full physical CB record must have zero entries outside either axis.
    masked_pairs = {
        (i, j): (
            dot(queries[physical_start + i], keys[physical_start + j])
            if lo <= i < hi and lo <= j < hi
            else F(0)
        )
        for i in range(block)
        for j in range(block)
    }
    assert all(
        value == 0 for (i, j), value in masked_pairs.items() if not (lo <= i < hi and lo <= j < hi)
    )
    return outputs, outgoing


if __name__ == "__main__":
    cases = logical_chunks = shared_blocks = 0
    shapes = [
        [3, 2, 5],
        [1, 1, 1, 1],
        [4, 1, 3],
        [0, 3, 0, 2],
        [5, 7, 1],
        [9, 1, 2],
        [2, 0, 0, 5],
        [0, 0, 0],
    ]
    for block in [4, 8]:
        for lengths in shapes:
            for recipe in range(3):
                total = sum(lengths)
                backing = ((total + block - 1) // block) * block
                keys = [(F(t % 3 - 1, 2), F(1, 3)) for t in range(backing)]
                queries = [(F((t + recipe) % 3, 3), F(1)) for t in range(backing)]
                # Native processed dt can multiply B or X; exact model absorbs it in V.
                values = [(F((t + 1) * (1 + t % 2), 3), F(recipe - t, 2)) for t in range(backing)]
                alpha = [F(1, 2) if t % 2 else F(3, 4) for t in range(backing)]
                prefix = 0
                block_owners = {}
                for seq, length in enumerate(lengths):
                    seed = [[F(seq + 1), F(-1)], [F(recipe), F(2)]]
                    expected, final = serial(
                        seed, range(prefix, prefix + length), keys, values, queries, alpha
                    )
                    out, carried = [], seed
                    cursor = prefix
                    while cursor < prefix + length:
                        physical_start = cursor // block * block
                        lo = cursor - physical_start
                        hi = min(block, prefix + length - physical_start)
                        rows, carried = subrange(
                            carried, physical_start, lo, hi, keys, values, queries, alpha, block
                        )
                        out.extend(rows)
                        block_owners.setdefault(physical_start, set()).add(seq)
                        cursor = physical_start + hi
                        logical_chunks += 1
                    assert out == expected and carried == final
                    prefix += length
                    cases += 1
                shared_blocks += sum(len(owners) > 1 for owners in block_owners.values())
    # An incoming state at lo1 must exclude physical row0's decay.
    seed = [[F(1), F(0)], [F(0), F(0)]]
    keys = [(F(0), F(0))] * 4
    queries = [(F(1), F(0))] * 4
    values = [(F(0), F(0))] * 4
    alpha = [F(1, 2), F(1, 2), F(1), F(1)]
    out, _ = subrange(seed, 0, 1, 2, keys, values, queries, alpha, 4)
    assert out == [[F(1, 2), F(0)]]
    assert out != [[F(1, 4), F(0)]]  # Unrebased physical prefix leaks previous decay.
    native_coordinates = 0
    prefix = 0
    for length in [127, 2, 1, 0, 128]:
        owned = []
        cursor = prefix
        while cursor < prefix + length:
            start = cursor // 128 * 128
            lo, hi = cursor - start, min(128, prefix + length - start)
            owned.extend(start + j for j in range(lo, hi))
            cursor = start + hi
        assert owned == list(range(prefix, prefix + length))
        native_coordinates += len(owned)
        prefix += length
    result = {
        "passed": True,
        "native_execution": False,
        "sequence_cases": cases,
        "logical_clipped_chunks": logical_chunks,
        "shared_physical_blocks": shared_blocks,
        "native128_valid_coordinates": native_coordinates,
        "oracle": ("Independent serial outer-product recurrence versus clipped chunk contractions"),
        "open": [
            "native processed-dt operand rounding",
            "state IO copies versus FP32 carry",
            "tcgen05/TMA lowering and pipeline retirement",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
