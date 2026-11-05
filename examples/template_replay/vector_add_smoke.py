"""Standalone language/toolchain example; unrelated to either target kernel task."""

import json
import sys
from pathlib import Path

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[2])
def build(n=257):
    @T.prim_func
    def main(
        a: T.Tensor((n,), "float16"), b: T.Tensor((n,), "float16"), out: T.Tensor((n,), "float16")
    ):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
            for lane in T.Parallel(128):
                index = block * 128 + lane
                if index < n:
                    out[index] = a[index] + b[index]

    return main


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("The language smoke requires a CUDA worker")
    a = torch.arange(257, device="cuda", dtype=torch.float16)
    b = torch.ones_like(a)
    actual = build()(a, b)
    torch.cuda.synchronize()
    passed = bool(torch.equal(actual, a + b))
    result = {
        "passed": passed,
        "elements": 257,
        "tilelang": tilelang.__version__,
        "device": torch.cuda.get_device_name(),
        "purpose": "language syntax smoke",
    }
    Path("language-smoke.json").write_text(json.dumps(result, indent=2))
    sys.stdout.write(json.dumps(result) + "\n")
    if not passed:
        raise RuntimeError("Vector-add language smoke failed")


if __name__ == "__main__":
    main()
