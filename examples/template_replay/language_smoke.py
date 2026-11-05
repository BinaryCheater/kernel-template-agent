import json
import sys
import time
from pathlib import Path

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[2])
def gemm_smoke():
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), "float16"),
        B: T.Tensor((64, 64), "float16"),
        C: T.Tensor((64, 64), "float16"),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((64, 32), "float16")
            b = T.alloc_shared((32, 64), "float16")
            c = T.alloc_fragment((64, 64), "float32")
            T.clear(c)
            for k in T.Pipelined(2, num_stages=1):
                T.copy(A[0:64, k * 32 : (k + 1) * 32], a)
                T.copy(B[k * 32 : (k + 1) * 32, 0:64], b)
                T.gemm(a, b, c)
            T.copy(c, C)

    return main


start = time.monotonic()
kernel = gemm_smoke()
compiled_s = time.monotonic() - start
g = torch.Generator(device="cuda").manual_seed(119)
a = torch.randn((64, 64), device="cuda", dtype=torch.float16, generator=g)
b = torch.randn((64, 64), device="cuda", dtype=torch.float16, generator=g)
actual = kernel(a, b)
expected = (a.double() @ b.double()).half()
torch.cuda.synchronize()
passed = bool(
    torch.isfinite(actual).all() and torch.allclose(actual, expected, rtol=0.01, atol=0.01)
)
result = {
    "passed": passed,
    "target": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "torch": torch.__version__,
    "tilelang": tilelang.__version__,
    "cuda": torch.version.cuda,
    "compile_s": compiled_s,
    "max_abs_error": float((actual - expected).abs().max()),
    "purpose": "language compilation/execution smoke; no performance claim",
}
Path("language-smoke.json").write_text(json.dumps(result, indent=2))
sys.stdout.write(json.dumps(result) + "\n")
if not passed:
    raise RuntimeError("TileLang GEMM smoke failed")
