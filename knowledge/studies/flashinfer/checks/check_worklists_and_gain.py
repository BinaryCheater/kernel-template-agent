"""Logical list ownership and finite FP8-gain witnesses; no native execution."""

import heapq
import json
import math
from pathlib import Path


def tasks(heads):
    requests = [(0, 260, 600), (1, 64, 100), (2, 0, 500), (3, 129, 80)]
    result = []
    for head in range(heads):
        for request, qlen, kvlen in sorted(requests, key=lambda r: -r[2]):
            tiles = (qlen + 127) // 128
            for tile in reversed(range(tiles)):
                end = (
                    kvlen
                    if tile + 1 == tiles
                    else max(min(kvlen - qlen + (tile + 1) * 128, kvlen), 0)
                )
                result.append((request, head, tile, qlen, kvlen, end))
    return result


def plan(work, workers):
    heap = [(0, b) for b in range(workers)]
    lists = [[] for _ in range(workers)]
    for item in work:
        cost, worker = heapq.heappop(heap)
        lists[worker].append(item)
        heapq.heappush(heap, (cost + 256 + item[-1], worker))
    flat, offsets = [], [0]
    for items in lists:
        flat.extend(items)
        offsets.append(len(flat))
    return flat, offsets


def positive_e4m3(code):
    exponent, mantissa = code >> 3, code & 7
    if exponent == 0:
        return mantissa * 2**-9
    return (1 + mantissa / 8) * 2 ** (exponent - 7)


def q_e4m3(value):
    # Finite, nonnegative admitted inputs; the code parity resolves nearest-even ties.
    assert 0 <= value <= 448
    return min((abs(positive_e4m3(c) - value), c % 2, c) for c in range(127))[2]


def quantize(value):
    return positive_e4m3(q_e4m3(value))


if __name__ == "__main__":
    work = tasks(2)
    flat, offsets = plan(work, 3)
    replay = [flat[i] for b in range(3) for i in range(offsets[b], offsets[b + 1])]
    assert len(set(replay)) == len(work) and sorted(replay) == sorted(work)
    assert len(work) == 12
    rows = [
        (r, h, row)
        for r, h, tile, qlen, _, _ in replay
        for row in range(tile * 128, min((tile + 1) * 128, qlen))
    ]
    assert len(rows) == len(set(rows)) == (260 + 64 + 129) * 2
    base, _ = plan(tasks(1), 3)
    replicated = [
        (r, h, tile, qlen, kvlen, end) for h in range(2) for r, _, tile, qlen, kvlen, end in base
    ]
    assert sorted(replicated) == sorted(work)
    empty_flat, empty_offsets = plan(tasks(1), 10)
    assert len(empty_flat) == 6 and empty_offsets.count(6) >= 2
    # Parallel arrays must follow the same flattening permutation.
    wrong = [
        (original[0], stored[1], *stored[2:]) for original, stored in zip(work, flat, strict=True)
    ]
    assert sorted(wrong) != sorted(work)

    gain = 448.0
    tiny = 2**-12
    assert quantize(tiny) == 0 and quantize(gain * tiny) / gain > 0
    # Reference moves from 0 to 1; old numerator and old mass are corrected by 1/2.
    tiles = [([tiny, 1.0], [17.0, 2.0], 1.0), ([1.0, 0.3], [4.0, 9.0], 0.5)]
    numerator = mass = ideal_numerator = residual = quantized_mass = 0.0
    for weights, values, correction in tiles:
        numerator *= correction
        mass *= correction
        ideal_numerator *= correction
        residual *= correction
        quantized_mass *= correction
        for w, v in zip(weights, values, strict=True):
            gained = quantize(gain * w)
            numerator += gained * v
            mass += w
            ideal_numerator += w * v
            residual += (gained - gain * w) * v
            quantized_mass += gained
    output = numerator / (gain * mass)
    assert math.isclose(output, ideal_numerator / mass + residual / (gain * mass), rel_tol=1e-14)
    assert not math.isclose(numerator / quantized_mass, output, rel_tol=1e-8)
    assert not math.isclose(numerator / mass, output)
    assert not math.isclose(numerator / (gain * gain * mass), output)
    # Original-mass LSE has no log2(gain); reference units here are base2.
    lse = 1 + math.log2(mass)
    assert math.isclose(lse, math.log2(tiny + 1 + 2 + 0.6))
    result = {
        "kind": "logical_cost_worklists_and_numerator_gain",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_worklists_and_gain.py"
        ),
        "task_count": len(work),
        "unique_output_rows": len(rows),
        "worker_offsets": offsets,
        "flattened_descriptors": flat,
        "replicated_head_schedule_matches": True,
        "empty_worker_ranges": True,
        "mismatched_parallel_metadata_rejected": True,
        "tiny_weight": tiny,
        "ungained_e4m3": quantize(tiny),
        "gained_e4m3_decoded_weight": quantize(gain * tiny) / gain,
        "streamed_output": output,
        "ideal_output": ideal_numerator / mass,
        "quantization_residual_output": residual / (gain * mass),
        "wrong_quantized_denominator_output": numerator / quantized_mass,
        "ungained_lse2": lse,
        "remaining": (
            "Native plan publication/role state, heap tie policy, FP8 conversi"
            "on/fragment mapping, exponent/MMA arithmetic and admitted error b"
            "ounds"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
