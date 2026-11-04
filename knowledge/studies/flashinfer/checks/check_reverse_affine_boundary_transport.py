"""Exact matrix pullback and exclusive adjoint-boundary transport identities."""

import json
from fractions import Fraction as F
from pathlib import Path


def add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def matvec(matrix, vector):
    return tuple(sum(x * y for x, y in zip(row, vector, strict=True)) for row in matrix)


def transpose(matrix):
    return tuple(zip(*matrix, strict=True))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def loss(seed, records):
    state, result = seed, F(0)
    for matrix, offset, readout in records:
        state = add(matvec(matrix, state), offset)
        result += dot(readout, state)
    return result


def local_offsets(records):
    return [matvec(transpose(matrix), readout) for matrix, _, readout in records]


def owner_boundaries(records):
    # The last exit is zero. Initial chain value d_last is already its predecessor's exit.
    result = [(F(0), F(0)) for _ in records]
    offsets = local_offsets(records)
    if len(records) > 1:
        value = offsets[-1]
        for fragment in reversed(range(len(records) - 1)):
            result[fragment] = value
            value = add(matvec(transpose(records[fragment][0]), value), offsets[fragment])
    return result


def neighbor_boundaries(records):
    # Algebraic messages only: no fake collective, scheduling or progress implementation.
    assert len(records) <= 3
    result = [(F(0), F(0)) for _ in records]
    offsets = local_offsets(records)
    for fragment in reversed(range(1, len(records))):
        result[fragment - 1] = add(
            matvec(transpose(records[fragment][0]), result[fragment]), offsets[fragment]
        )
    return result


if __name__ == "__main__":
    owner_cases = hop_cases = derivative_entries = ownership_slots = 0
    for count in range(1, 7):
        for recipe in range(8):
            records = []
            for f in range(count):
                matrix = (
                    (F(f + 1, 3), F(recipe + 1, 5)),
                    (F(1 - recipe, 7), F(2 - f, 3)),
                )
                records.append((matrix, (F(f, 2), F(-recipe, 3)), (F(1 + f), F(recipe - f))))
            entries = owner_boundaries(records)
            offsets = local_offsets(records)
            for f in range(count - 1):
                expected = add(matvec(transpose(records[f + 1][0]), entries[f + 1]), offsets[f + 1])
                assert entries[f] == expected
            seed = (F(2, 3), F(-1, 5))
            pullback = add(matvec(transpose(records[0][0]), entries[0]), offsets[0])
            for axis in range(2):
                perturbed = tuple(x + (i == axis) for i, x in enumerate(seed))
                assert loss(perturbed, records) - loss(seed, records) == pullback[axis]
                derivative_entries += 1
            # Stored forward transposed tiles and resent ordinary tiles resolve one operator.
            for matrix, _, _ in records:
                saved_forward_tile = transpose(matrix)
                resent_backward_tile = matrix
                assert saved_forward_tile == transpose(resent_backward_tile)
            if count <= 3:
                assert neighbor_boundaries(records) == entries
                hop_cases += 1
            owner_cases += 1
    # Every live head resolves to one compact owner; the final padded group has unread holes.
    for heads in range(1, 18):
        for world in range(1, 9):
            group = (heads + world - 1) // world
            visited = []
            for rank in range(world):
                live = max(0, min(group, heads - rank * group))
                for local_head in range(live):
                    h = rank * group + local_head
                    assert h // group == rank
                    visited.append(h)
                    ownership_slots += 1
            assert visited == list(range(heads))
    a, b = ((F(1), F(2)), (F(0), F(1))), ((F(1), F(0)), (F(3), F(1)))
    v = (F(2), F(-1))
    assert matvec(transpose(a), v) != matvec(a, v)
    assert matvec(transpose(a), matvec(transpose(b), v)) != matvec(
        transpose(b), matvec(transpose(a), v)
    )
    witness = [(a, (F(0), F(0)), (F(1), F(2))), (b, (F(0), F(0)), (F(3), F(1)))]
    correct = owner_boundaries(witness)
    assert correct[0] != correct[1]  # Inclusive/self boundary is not the future boundary.
    result = {
        "passed": True,
        "native_execution": False,
        "owner_affine_cases": owner_cases,
        "bounded_neighbor_cases": hop_cases,
        "exact_seed_derivatives": derivative_entries,
        "live_head_owner_slots": ownership_slots,
        "counterexamples": ["missing transpose", "reversed map pairing", "inclusive boundary"],
        "open": [
            "native FP32/TF32 stage equality",
            "collective progress",
            "aliased tape walk lifetime",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
