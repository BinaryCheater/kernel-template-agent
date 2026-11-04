"""Small source-distinguishing algebra witnesses; not device execution."""

import itertools
import json
import sys
from fractions import Fraction

# Coordinate update delta must equal full anisotropic objective change.
h, dh, norm_delta, eta = Fraction(1), Fraction(-1, 5), Fraction(1, 10), Fraction(3)
a = norm_delta + (eta - 1) * dh * dh
b = 2 * (eta - 1) * dh
assert a + b * h == norm_delta + (eta - 1) * ((h + dh) ** 2 - h**2)
assert a + b * h == Fraction(-31, 50)

# Bit planes reproduce an asymmetric unsigned integer dot; uniquely varied values expose swaps.
a_codes, b_codes = [1, 2, 3, 0], [3, 1, 2, 1]
plane_dot = 0
for i, j in itertools.product(range(2), repeat=2):
    ai = sum(((v >> i) & 1) << k for k, v in enumerate(a_codes))
    bj = sum(((v >> j) & 1) << k for k, v in enumerate(b_codes))
    plane_dot += (ai & bj).bit_count() << (i + j)
assert plane_dot == sum(a * b for a, b in zip(a_codes, b_codes, strict=False)) == 11

# Recover unique logical bit coordinates after 32x8 transpose, including tail34.
for dim in [1, 7, 8, 33, 34, 255, 256]:
    for coordinate in range(dim):
        lane, bit = coordinate % 32, coordinate // 32
        out_lane = (bit * 4) + lane // 8
        out_bit = lane % 8
        assert 8 * out_lane + out_bit == coordinate

# Nonnegative partial L2 sum bound; signed partial sum fails the same certificate.
assert 7 + 0 + 4 >= 7 > 6
assert 7 - 5 < 6
# Inclusive radius equality must survive, unlike arbitrary-tie top-k rejection.
assert not (7 > 7)

sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            json.dumps(
                {
                    "anisotropic_delta": str(a + b * h),
                    "bitplane_dot": plane_dot,
                    "transpose_dimensions": [1, 7, 8, 33, 34, 255, 256],
                    "signed_bound_counterexample": [7, -5, 6],
                    "scope": (
                        "local algebra/mapping only; no GPU, memory-order, FP32 boundary o"
                        "r recall claim"
                    ),
                },
                indent=2,
            )
        ]
    )
    + "\n"
)
