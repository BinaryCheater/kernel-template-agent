"""Check the source address permutation, not CUDA instruction semantics."""

import json
from pathlib import Path


def check(m: int, n: int, layout: str) -> dict[str, object]:
    stride = m if layout == "CM" else n
    frag_stride = stride // 16
    native_to_linear = {}
    for lane in range(32):
        for reg_idx in range(m * n // 64):
            fragment, reg = divmod(reg_idx, 4)
            rx = reg % 2 if layout == "CM" else reg // 2
            ry = reg % 2 if layout == "RM" else reg // 2
            x = lane % 4 * 2 + rx * 8 + fragment % frag_stride * 16
            y = lane // 4 + ry * 8 + fragment // frag_stride * 16
            linear_pair = x + y * stride
            native_pair = (fragment * 32 + lane) * 8 + reg * 2
            for half in range(2):
                native_to_linear[native_pair + half] = linear_pair + half
    assert sorted(native_to_linear) == list(range(m * n))
    assert sorted(native_to_linear.values()) == list(range(m * n))
    # Unique coordinate identities expose row/column and fragment swaps.
    linear = [
        1000 * (i // n) + i % n if layout == "RM" else 1000 * (i % m) + i // m for i in range(m * n)
    ]
    native = [linear[native_to_linear[i]] for i in range(m * n)]
    restored = [None] * (m * n)
    for i, target in native_to_linear.items():
        restored[target] = native[i]
    assert restored == linear
    assert native != linear
    return {
        "rows": m,
        "columns": n,
        "layout": layout,
        "scalar_addresses": m * n,
        "permutation": "passed",
        "coordinate_roundtrip": "passed",
        "native_is_not_linear": True,
        "native_first_12_coordinate_ids": native[:12],
    }


if __name__ == "__main__":
    result = {
        "kind": "logical_address_permutation",
        "source": "include/tiny-cuda-nn/mma.h:171-218,267-285",
        "native_execution": False,
        "cases": [
            check(m, n, layout)
            for m, n in [(16, 16), (32, 16), (16, 64), (32, 64), (64, 64)]
            for layout in ["RM", "CM"]
        ],
        "remaining": (
            "PTX fragment interpretation, shuffle mapping, numeric MMA and con"
            "current parameter layout lifetime"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
