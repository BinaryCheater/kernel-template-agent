"Chunk-local suffix pullback against an independently expanded prefix Jacobian."

import json
from fractions import Fraction as F
from pathlib import Path


def loss(g, incoming, endpoint):
    prefixes = [sum(g[: i + 1]) for i in range(len(g))]
    total = sum(a * p for a, p in zip(incoming, prefixes, strict=True))
    return total + (endpoint * prefixes[-1] if prefixes else 0)


if __name__ == "__main__":
    cases = entries = 0
    for length in range(65):
        for recipe in range(3):
            incoming = [F((i + recipe) % 7 - 3, 5) for i in range(length)]
            endpoint = F(recipe - 1, 3)
            values = [F(0)] * 64
            values[:length] = incoming
            result, accumulator = [None] * length, endpoint
            for u in range(64):
                s = 63 - u
                if s < length:
                    accumulator += values[s]
                    result[s] = accumulator
            expected = [
                sum(incoming[t] for t in range(length) if t >= i) + endpoint for i in range(length)
            ]
            assert result == expected
            # Independent prefix-loss response to each original gate coordinate.
            gates = [F(i - recipe, 7) for i in range(length)]

            for i in range(length):
                changed = gates.copy()
                changed[i] += 1
                assert (
                    loss(changed, incoming, endpoint) - loss(gates, incoming, endpoint) == result[i]
                )
                entries += 1
            cases += 1
    result = {
        "passed": True,
        "native_execution": False,
        "chunk_length_cases": cases,
        "exact_prefix_jacobian_entries": entries,
        "oracle": "Expanded prefix loss including one terminal prefix cotangent",
        "open": ["native dg_part/gamma_dot arithmetic stages", "native FP32 addition order"],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
