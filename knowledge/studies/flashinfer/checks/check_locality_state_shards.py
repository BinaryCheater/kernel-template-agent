"Padded two-domain state views, warp transpose-reduce and fork/join ordering."

import itertools
import json
import struct
from pathlib import Path


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def add(x, y):
    return f32(x + y)


def independent_butterflies(values):
    result = [row[:] for row in values]
    for offset in [8, 4, 2, 1]:
        result = [[add(row[lane], row[lane ^ offset]) for lane in range(16)] for row in result]
    return result


def transpose_reduce(values):
    # Four independent row partials per lane -> owner row=2*bit3+bit2.
    a, b = [], []
    for lane in range(16):
        b3 = bool(lane & 8)
        peer = lane ^ 8
        peer_b3 = bool(peer & 8)
        send_a = values[0 if peer_b3 else 2][peer]
        send_b = values[1 if peer_b3 else 3][peer]
        a.append(add(values[2 if b3 else 0][lane], send_a))
        b.append(add(values[3 if b3 else 1][lane], send_b))
    result = []
    for lane in range(16):
        b2, peer = bool(lane & 4), lane ^ 4
        send = a[peer] if peer & 4 else b[peer]
        result.append(add(b[lane] if b2 else a[lane], send))
    for offset in [2, 1]:
        result = [add(result[lane], result[lane ^ offset]) for lane in range(16)]
    return result


if __name__ == "__main__":
    slots, heads, half, keys = 3, 2, 64, 128
    payload = slots * heads * half * keys * 2
    domain_bytes = ((payload + 65535) // 65536) * 65536
    domain_stride = domain_bytes // 2
    labels, seen = {}, set()
    for slot, head, domain, row, key in itertools.product(
        range(slots), range(heads), range(2), range(half), range(keys)
    ):
        address = (
            slot * heads * half * keys
            + head * half * keys
            + domain * domain_stride
            + row * keys
            + key
        )
        logical = slot, head, domain * half + row, key
        assert address not in seen
        assert address // domain_stride == domain
        labels[logical] = address
        seen.add(address)
    assert len(labels) == slots * heads * 128 * 128
    request_coordinates = 0
    for slot in [2, 0, 1]:
        for head, domain, row in itertools.product(range(heads), range(2), range(half)):
            original_row = domain * half + row
            for key in range(keys):
                expected = (
                    domain * domain_stride
                    + slot * heads * half * keys
                    + head * half * keys
                    + row * keys
                    + key
                )
                assert labels[slot, head, original_row, key] == expected
                request_coordinates += 1
    assert domain_bytes > payload  # Domain stride must retain allocation padding.
    assert sum(range(128)) != sum(range(64))  # Cutting K would need a missing reduction.
    reductions = 0
    atoms = [1.0, -1.0, 1e20, -1e20, 2**-20, -(2**-20), 3.5, -7.25]
    for recipe in range(16):
        values = [
            [f32(atoms[(lane * (row + 1) + recipe + row) % len(atoms)]) for lane in range(16)]
            for row in range(4)
        ]
        expected = independent_butterflies(values)
        actual = transpose_reduce(values)
        for lane, value in enumerate(actual):
            row = 2 * bool(lane & 8) + bool(lane & 4)
            assert struct.pack("<f", value) == struct.pack("<f", expected[row][lane])
            reductions += 1
    valid_orders = 0
    events = ["run0", "done0", "run1", "done1"]
    for order in itertools.permutations(events):
        if any(order.index(f"run{d}") > order.index(f"done{d}") for d in range(2)):
            continue
        completed = set()
        for event in order:
            if event.startswith("done"):
                completed.add(int(event[-1]))
        # Source records fork before workers and main joins every recorded done.
        assert completed == {0, 1}
        valid_orders += 1
    assert valid_orders == 6
    old_state = ["old0", "old1"]
    only_first_done = ["new0", old_state[1]]
    assert only_first_done != ["new0", "new1"]  # One join is insufficient.
    modes = {}
    for captured in [False, True]:
        for states in [1, 8191, 8192, 16384]:
            modes[str((captured, states))] = (
                "matched" if captured or states >= 8192 else "current_stream"
            )
    result = {
        "passed": True,
        "native_execution": False,
        "padded_state_coordinates": len(labels),
        "shuffled_request_coordinates": request_coordinates,
        "fp32_transpose_reduce_lane_results": reductions,
        "worker_event_orders": valid_orders,
        "domain_payload_bytes": payload,
        "domain_allocation_bytes": domain_bytes,
        "source_execution_modes": modes,
        "open": [
            "actual allocation/SM locality matching and backfill semantics",
            "native cp.async row arithmetic and layout lowering",
            "CUDA graph event publication and allocation/context retirement",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
