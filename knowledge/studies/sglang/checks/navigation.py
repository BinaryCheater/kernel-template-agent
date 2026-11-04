"""Exact topology, first-failure and resident transition identities."""

import json
from itertools import product
from pathlib import Path

prefix_cases = 0
for n in range(13):
    for bits in product((False, True), repeat=n):
        count = 0
        live = True
        for b in bits:
            live = live and b
            count += live
        first = min([i for i, b in enumerate(bits) if not b] + [n])
        assert count == first
        prefix_cases += 1
tree_cases = 0
for n in range(1, 9):
    for tail in product(*(range(i) for i in range(1, n))):
        parent = [-1, *list(tail)]
        rows = []
        for node in range(n):
            ancestors = {node}
            cur = node
            while parent[cur] >= 0:
                cur = parent[cur]
                ancestors.add(cur)
            rows.append(ancestors)
        first = [-1] * n
        sibling = [-1] * n
        for node in range(n - 1, 0, -1):
            sibling[node] = first[parent[node]]
            first[parent[node]] = node
        for node in range(n):
            proper = [i for i in rows[node] if i < node]
            recovered = max(proper, default=-1)
            assert recovered == parent[node]
            descendants = [i for i in range(node + 1, n) if node in rows[i]]
            assert min(descendants, default=-1) == first[node]
            later = [i for i in range(node + 1, n) if parent[i] == parent[node]] if node else []
            assert min(later, default=-1) == sibling[node]
            # Byte packing round-trip including zero padding to8/16/32 bits.
            if n <= 32:
                width = 8 if n <= 8 else 16 if n <= 16 else 32
                packed = sum(1 << i for i in rows[node])
                data = packed.to_bytes(width // 8, "little")
                assert {i for i in range(width) if data[i // 8] >> (i % 8) & 1} == rows[node]
        tree_cases += 1
# Per-step tape must survive even when only the current index is resident.
walk_cases = 0
for k in range(1, 6):
    for length in range(8):
        transition = [[(prev + slot + 1) % k for prev in range(k)] for slot in range(length)]
        materialized = [0]
        tape = []
        for row in transition:
            nxt = row[materialized[-1]]
            materialized.append(nxt)
            tape.append((nxt, tuple(row)))
        current = 0
        resident_tape = []
        for row in transition:
            current = row[current]
            resident_tape.append((current, tuple(row)))
        assert current == materialized[-1] and resident_tape == tape
        walk_cases += 1
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "prefix_cases": prefix_cases,
            "topological_tree_cases": tree_cases,
            "resident_walk_cases": walk_cases,
            "scope": (
                "Exact finite host structures; no native synchronization, FP32 pro"
                "posal or GPU check"
            ),
        },
        indent=2,
    )
    + "\n"
)
