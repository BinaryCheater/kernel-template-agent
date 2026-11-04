"Finite decoder-coordinate and CTA ticket witnesses, not native CUDA execution."

import itertools
import json
import math
import struct
from pathlib import Path


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def decode(code, scale):
    magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    value = math.copysign(magnitudes[code & 7], -1.0 if code & 8 else 1.0)
    return half(half(value) * half(scale))


def dispatch(counts, preferences, initial=(0, 0)):
    counters = list(initial)
    jobs = []
    for preferred in preferences:
        family = preferred
        ticket = counters[family]
        counters[family] += 1
        if ticket >= counts[family]:
            family = 1 - family
            ticket = counters[family]
            counters[family] += 1
        if ticket < counts[family]:
            jobs.append((family, ticket))
    return jobs, counters


if __name__ == "__main__":
    codes = [[(row * 3 + col) % 16 for col in range(32)] for row in range(2)]
    scales = [[0.5, 2.0], [1.5, 0.25]]
    inline = [
        [decode(code, scales[row][col // 16]) for col, code in enumerate(values)]
        for row, values in enumerate(codes)
    ]
    staged = [[None] * 32 for _ in range(2)]
    output_coordinates = set()
    for row in range(2):
        for group in range(2):
            logical_codes = codes[row][group * 16 : (group + 1) * 16]
            payload = bytes(logical_codes[i] | (logical_codes[i + 1] << 4) for i in range(0, 16, 2))
            padded = payload + bytes(8)
            assert len(padded) == 16 and len(padded[:8]) == 8
            recovered = [part for byte in padded[:8] for part in [byte & 15, byte >> 4]]
            for i, code in enumerate(recovered):
                out_block, out_slot = 2 * group + i // 8, i % 8
                coordinate = (row, out_block * 8 + out_slot)
                assert coordinate not in output_coordinates
                output_coordinates.add(coordinate)
                staged[row][coordinate[1]] = decode(code, scales[row][group])
    assert len(output_coordinates) == 64
    assert [[struct.pack("<e", x) for x in row] for row in staged] == [
        [struct.pack("<e", x) for x in row] for row in inline
    ]
    wrong = [
        [decode(code, scales[row][col % 2]) for col, code in enumerate(values)]
        for row, values in enumerate(codes)
    ]
    assert wrong != inline
    probability, value, scale = 0.5, 256.0, 2**-26
    scale_v_first = probability * half(value * scale)
    scale_p_first = half(probability * scale) * value
    assert scale_v_first == 2**-19 and scale_p_first == 0.0

    cases = 0
    for count0 in range(1, 5):
        for count1 in range(1, 5):
            wanted = {
                (family, ticket)
                for family, count in enumerate([count0, count1])
                for ticket in range(count)
            }
            for preferences in itertools.product([0, 1], repeat=count0 + count1):
                jobs, _ = dispatch([count0, count1], preferences)
                assert len(jobs) == len(set(jobs)) and set(jobs) == wanted
                cases += 1
    missing, _ = dispatch([2, 3], [0] * 4)
    assert len(missing) == 4
    reused, counters = dispatch([2, 3], [0] * 5, initial=(2, 3))
    assert reused == [] and counters[0] > 2 and counters[1] > 3

    # Source ratio selector is only a preferred share; the queues still define ownership.
    counts, domain_counters, preferences = [2, 5], [0, 0], []
    for domain in [0, 1, 0, 0, 1, 1, 0]:
        tag = domain_counters[domain] % (counts[1] // counts[0] + 1)
        domain_counters[domain] += 1
        preferences.append(0 if tag == 0 else 1)
    jobs, final_counters = dispatch(counts, preferences)
    assert len(jobs) == 7 and len(set(jobs)) == 7

    result = {
        "kind": "logical_decoded_tile_and_two_family_pool",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_repack_and_pool.py"
        ),
        "e2m1_codes": "all sixteen in adjacent unequal scale groups",
        "padded_input_to_two_output_blocks": "64 logical coordinates covered once",
        "inline_and_staged_fp16_bytes": "equal for tested finite scales",
        "wrong_scale_group_changes_result": True,
        "scale_on_v_result": scale_v_first,
        "scale_on_p_result": scale_p_first,
        "all_small_preference_sequences": cases,
        "too_small_grid_omits_work": True,
        "unreset_counters_omit_work": True,
        "ratio_preferences": preferences,
        "ratio_job_claims": jobs,
        "final_claim_counters": final_counters,
        "remaining": (
            "Native conversion/swizzle/barriers, atomic interleaving, SM-ID sp"
            "an, counter concurrency and full device resource/participation bi"
            "nding"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
