"Ordered matrix-state summaries and source's conservative ragged workspace slots."

import itertools
import json
from fractions import Fraction as F
from pathlib import Path

from check_deferred_prefix_state import affine_reference
from check_delta_feedback import add, mm


def bound(n, total, chunk):
    nonempty = min(n, total)
    return nonempty + (total - nonempty) // chunk


def summary(records):
    linear = [[F(1), F(0)], [F(0), F(1)]]
    offset = [[F(0), F(0)], [F(0), F(0)]]
    for key, value, alpha, beta in records:
        a = [[alpha * (F(i == j) - beta * key[i] * key[j]) for j in range(2)] for i in range(2)]
        b = [[beta * value[i] * key[j] for j in range(2)] for i in range(2)]
        linear, offset = mm(linear, a), add(mm(offset, a), b)
    return linear, offset


def apply(state, affine):
    linear, offset = affine
    return add(mm(state, linear), offset)


if __name__ == "__main__":
    state_cases = piece_cases = 0
    noncommuting = False
    for lengths in [(1, 1), (2, 3, 1), (4, 1, 2), (3, 3, 2)]:
        for recipe in range(4):
            records = []
            for t in range(sum(lengths)):
                records.append(
                    (
                        (F((t + recipe) % 3 - 1, 2), F(1, 2)),
                        (F(t + 1, 3), F(recipe - t)),
                        F(1, 2) if t % 2 else F(1),
                        F(1 + recipe % 2, 3),
                    )
                )
            pieces, start = [], 0
            for length in lengths:
                pieces.append(records[start : start + length])
                start += length
            for initial in [[[F(0), F(0)], [F(0), F(0)]], [[F(1), F(-2)], [F(3), F(1)]]]:
                # Source fixup stores inclusive piece-end states once in order.
                fixed, state = [], initial
                for piece in pieces:
                    state = apply(state, summary(piece))
                    fixed.append(state)
                # Independent local replay starts from exclusive prefix / original seed.
                for index, piece in enumerate(pieces):
                    entry = initial if index == 0 else fixed[index - 1]
                    replayed_prefix = initial
                    for earlier in pieces[:index]:
                        replayed_prefix = apply(replayed_prefix, summary(earlier))
                    assert replayed_prefix == entry
                    direct = tuple(map(tuple, initial))
                    for record in records[: sum(lengths[:index])]:
                        direct = affine_reference(direct, record)
                    assert entry == list(map(list, direct))
                    local = tuple(map(tuple, entry))
                    for record in piece:
                        local = affine_reference(local, record)
                    assert fixed[index] == list(map(list, local))
                    piece_cases += 1
                serial = tuple(map(tuple, initial))
                for record in records:
                    serial = affine_reference(serial, record)
                assert state == list(map(list, serial))
                state_cases += 1
                wrong_order = apply(apply(initial, summary(pieces[1])), summary(pieces[0]))
                right_order = apply(apply(initial, summary(pieces[0])), summary(pieces[1]))
                noncommuting |= wrong_order != right_order
    assert noncommuting
    domains = [(range(6), range(1, 6)), ([0, 1, 63, 64, 65, 128], [64, 128])]
    slot_cases = gap_cases = 0
    for values, chunks in domains:
        for segments in range(1, 5):
            for lengths in itertools.product(values, repeat=segments):
                for chunk in chunks:
                    capacity = bound(segments, sum(lengths), chunk)
                    live_slots = []
                    prefix = 0
                    for i, length in enumerate(lengths):
                        start = bound(i, prefix, chunk)
                        count = (length + chunk - 1) // chunk
                        end = bound(i + 1, prefix + length, chunk)
                        assert start + count <= end <= capacity
                        local = [start + j for j in range(count)]
                        live_slots.extend(local)
                        for j in range(1, count):
                            assert start + j - 1 == local[j - 1]
                        prefix += length
                    assert len(live_slots) == len(set(live_slots))
                    assert all(0 <= slot < capacity for slot in live_slots)
                    gap_cases += len(live_slots) < capacity
                    slot_cases += 1
    assert gap_cases > 0
    assert bound(2, 2, 4) == 2 > (2 + 3) // 4  # Fused-total ceil underallocates.
    # The first slot of sequence2 has an unused predecessor, not its entry state.
    lengths, chunk = [4, 4, 1], 4
    third_start = bound(2, sum(lengths[:2]), chunk)
    assert third_start == 3
    live_before = [bound(i, sum(lengths[:i]), chunk) for i in range(2)]
    assert third_start - 1 not in live_before
    result = {
        "passed": True,
        "native_execution": False,
        "affine_final_state_cases": state_cases,
        "exclusive_piece_entry_cases": piece_cases,
        "ragged_slot_partitions": slot_cases,
        "partitions_with_unread_gaps": gap_cases,
        "oracles": [
            "Independent serial delta recurrence versus matrix affine pieces",
            "Explicit valid segment/chunk intervals versus bound-derived starts",
        ],
        "open": [
            "native FP16/BF16 summary construction",
            "SIMT versus TF32 fixup precision",
            "native workspace publication and descriptor lowering",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
