"""Numeric canaries using real PyTorch, run inside the evaluator container.

python selftest.py (copy beside worker.py for a direct, evaluator-owned smoke).
"""

from __future__ import annotations

import importlib

try:
    from .worker import compare, mathematical
except ImportError:
    from worker import compare, mathematical


def main() -> None:
    torch = importlib.import_module("torch")
    tolerance = {"rtol": 0.01, "atol": 0.01}
    a = torch.tensor([[100, 100, 1], [2, -2, 3]], dtype=torch.float16)
    b = torch.tensor([[1, 2], [-1, -2], [3, 4]], dtype=torch.float16)
    expected = torch.tensor([[3, 4], [13, 20]], dtype=torch.float16)
    assert torch.equal(mathematical(torch, "gemm", (a, b)), expected)
    # Odd sequence length, exact cumulative means, causal row zero, and multiple heads.
    q = torch.zeros((1, 2, 3, 2), dtype=torch.float16)
    k = q.clone()
    v = torch.tensor(
        [[[[2, 4], [4, 8], [12, 24]], [[6, 12], [12, 24], [18, 36]]]], dtype=torch.float16
    )
    expected = torch.tensor(
        [[[[2, 4], [3, 6], [6, 12]], [[6, 12], [9, 18], [12, 24]]]], dtype=torch.float16
    )
    assert torch.equal(mathematical(torch, "attention", (q, k, v)), expected)
    for value in (float("nan"), float("inf"), -float("inf")):
        invalid = expected.clone()
        invalid[0, 0, 0, 0] = value
        assert not compare(torch, invalid, expected, tolerance)["passed"]
    assert not compare(torch, expected.float(), expected, tolerance)["passed"]
    assert not compare(torch, expected[..., :1], expected, tolerance)["passed"]
    assert not compare(torch, expected + 1, expected, tolerance)["passed"]
    assert compare(torch, expected.clone(), expected, tolerance)["passed"]
    print(
        "Numeric canaries passed: cancellation, odd tails, causal masks, finite/dtype/shape/error"
    )


if __name__ == "__main__":
    main()
