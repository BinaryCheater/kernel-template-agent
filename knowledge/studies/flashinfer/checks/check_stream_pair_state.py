"Logical pair-stream chunk identity and carry ordering; no native GPU replay."

import itertools
import json
import math
import struct
from pathlib import Path


def invoke(state, position, payloads):
    old = state
    emitted = {}
    for i, payload in enumerate(payloads):
        p = position + i
        if p % 2 == 0:
            continue
        even = payloads[i - 1] if i else None
        if i == 0 and old is not None and old[0] == p - 1:
            even = old[1]
        if even is not None:
            emitted[p] = (even, payload)
    if payloads:
        last = position + len(payloads) - 1
        state = (last, payloads[-1]) if last % 2 == 0 else None
    return state, emitted


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def rms(xs):
    inverse = 1 / math.sqrt(sum(x * x for x in xs) / len(xs) + 1e-20)
    return [bf16(x * inverse) for x in xs]


if __name__ == "__main__":
    cases = 0
    for length in range(1, 9):
        for start in (0, 1, 6, 7):
            for have_carry in (False, True):
                initial = (start - 1, ("old", start - 1)) if have_carry and start % 2 else None
                payloads = [("current", start + i) for i in range(length)]
                context = {start + i: v for i, v in enumerate(payloads)}
                if initial:
                    context[initial[0]] = initial[1]
                expected = {
                    p: (context[p - 1], context[p])
                    for p in range(start, start + length)
                    if p % 2 and p - 1 in context
                }
                for cuts in range(1 << (length - 1)):
                    boundaries = (
                        [0] + [i for i in range(1, length) if cuts & (1 << (i - 1))] + [length]
                    )
                    state, observed = initial, {}
                    for begin, end in itertools.pairwise(boundaries):
                        state, chunk = invoke(state, start + begin, payloads[begin:end])
                        observed.update(chunk)
                    assert observed == expected
                    last = start + length - 1
                    assert state == ((last, payloads[-1]) if last % 2 == 0 else None)
                    cases += 1
    old = (0, "even0")
    correct_state, correct = invoke(old, 1, ["odd1", "even2"])
    assert correct == {1: ("even0", "odd1")} and correct_state == (2, "even2")
    _, early_commit = invoke(correct_state, 1, ["odd1", "even2"])
    assert early_commit == {}
    assert invoke(old, 0, [])[0] == old
    assert invoke(old, 5, ["odd5"])[1] == {}
    # Recycled IDs with matching positions need explicit invalidation.
    assert invoke(old, 1, ["new_stream_odd1"])[1]
    assert invoke(None, 1, ["new_stream_odd1"])[1] == {}
    # The source pools per channel then rounds BF16 before RMSNorm.
    pooled = [1.003, 1.013] * 256
    rounded_first = rms([bf16(x) for x in pooled])
    round_only_after_norm = rms(pooled)
    assert rounded_first != round_only_after_norm
    gates0, gates1 = [0.0, 4.0], [4.0, 0.0]
    values0, values1 = [2.0, 6.0], [8.0, 1.0]
    pooled_channels = []
    for a, b, g0, g1 in zip(values0, values1, gates0, gates1, strict=True):
        m = max(g0, g1)
        e0, e1 = math.exp(g0 - m), math.exp(g1 - m)
        pooled_channels.append((a * e0 + b * e1) / (e0 + e1))
    assert pooled_channels[0] > 7 and pooled_channels[1] > 5
    result = {
        "kind": "logical_stream_pair_state_check",
        "native_execution": False,
        "status": "passed",
        "chunk_cases": cases,
        "early_commit_loses_pair": True,
        "checks": [
            "chunk_invariance",
            "absolute_parity",
            "pending_tag",
            "empty_chunk",
            "discontinuous_position",
            "state_reuse_invalidation",
            "read_before_commit",
            "pooled_bf16_checkpoint",
            "per_channel_gates",
        ],
        "numeric_boundary": (
            "Python finite arithmetic and explicit BF16 rounding isolate check"
            "points; native exp/RMSNorm and reduction order are not replayed."
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
