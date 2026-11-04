"""Logical M8S3 row mailbox/merge witnesses, not native DSM execution."""

import json
import math
from pathlib import Path

if __name__ == "__main__":
    owners = [list(range(8 * r // 3, 8 * (r + 1) // 3)) for r in range(3)]
    assert [i for group in owners for i in group] == list(range(8))
    addresses = {}
    for s in range(16):
        for warp in range(4):
            for lane in range(32):
                for bit in range(2):
                    row = lane % 4 * 2 + bit
                    row_lane = warp * 8 + lane // 4
                    offset = s * 512 + row * 64 + row_lane * 2
                    for pair in range(2):
                        address = offset + pair
                        assert address not in addresses
                        addresses[address] = (s, row, row_lane * 2 + pair)
    assert len(addresses) == 8 * 1024 and set(addresses) == set(range(8 * 1024))
    states = [(0.0, 2.0, 6.0), (2.0, 3.0, 15.0), (-math.inf, 0.0, 0.0)]
    m = max(x[0] for x in states)
    a = [2 ** (mx - m) if mass > 0 else 0 for mx, mass, _ in states]
    d = sum(weight * st[1] for weight, st in zip(a, states, strict=True))
    raw_result = sum(weight * st[2] for weight, st in zip(a, states, strict=True)) / d
    normalized_result = (
        sum(
            weight * st[1] * (st[2] / st[1] if st[1] else 0)
            for weight, st in zip(a, states, strict=True)
        )
        / d
    )
    wrong = sum(weight * st[1] * st[2] for weight, st in zip(a, states, strict=True)) / d
    assert raw_result == normalized_result and wrong != raw_result
    empty = [(-math.inf, 0.0, 0.0)] * 3
    empty_weights = [2 ** (mx - (-math.inf)) if mass > 0 else 0 for mx, mass, _ in empty]
    assert empty_weights == [0, 0, 0]
    # Every row owner retains its local coordinate and receives precisely two remote copies.
    reads = 0
    for rank, rows in enumerate(owners):
        for row in rows:
            for feature in range(1024):
                values = [100000 * src + 1000 * row + feature for src in range(3)]
                published = {src: values[src] for src in range(3) if src != rank}
                assert len(published) == 2
                assert values[rank] + sum(published.values()) == sum(values)
                reads += 1
    result = {
        "kind": "logical_three_peer_row_owned_combine",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_cluster_row_combine.py"
        ),
        "owner_rows": owners,
        "row_packed_coordinates": len(addresses),
        "unique_final_coordinates": reads,
        "raw_merge": raw_result,
        "equivalent_normalized_merge": normalized_result,
        "wrong_extra_mass_on_raw_numerator": wrong,
        "empty_peer_weights": empty_weights,
        "remaining": (
            "Native accumulator coordinate mapping, DSM addresses, publication"
            "/retirement joins and cluster launch/resource legality"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
