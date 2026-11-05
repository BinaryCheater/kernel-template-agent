"""LEARNDEMO1: numeric check of generated `moe_nvfp4` kernels against a NumPy-free torch reference.

Runs in the worker image on a GPU: python perfmodel/testset/check_moe.py
Tokens are grouped by expert; NVFP4 codes hold sign, 2-bit exponent, 1-bit mantissa, two per byte
low nibble first, scaled per 16 elements along K.
"""

import importlib.util
import sys
from pathlib import Path

import tilelang
import torch

spec = importlib.util.spec_from_file_location("gen", Path(__file__).parent / "gen.py")
gen = importlib.util.module_from_spec(spec)
sys.argv = ["gen.py", "/tmp/unused"]
spec.loader.exec_module(gen)

E_, T_, N, K = 4, 3840, 2048, 1536
CASES = [(160, 48, 64, 1, 128), (160, 128, 32, 3, 128), (192, 128, 32, 3, 128)]
torch.manual_seed(7)
dev = "cuda"
code = torch.randint(0, 16, (E_, N, K), device=dev, dtype=torch.int32)
scale = (torch.rand(E_, N, K // 16, device=dev) + 0.5).to(torch.float16)
A = (torch.rand(T_, K, device=dev) - 0.5).to(torch.float16)
packed = (code[:, :, 0::2] | (code[:, :, 1::2] << 4)).to(torch.int8)
ex, m = (code >> 1) & 3, code & 1
mag = torch.where(ex > 0, (2 + m) << (ex - 1).clamp(min=0), m).to(torch.float32) * 0.5
sign = torch.where(code >= 8, -1.0, 1.0)
W = sign * mag * scale.repeat_interleave(16, dim=2).to(torch.float32)
per = T_ // E_
ref = torch.cat([A[i * per : (i + 1) * per].float() @ W[i].T for i in range(E_)])
ok = True
for bm, bn, bk, s, th in CASES:
    f = gen.moe_nvfp4(E_, T_, N, K, bm, bn, bk, s, th)
    kernel = tilelang.compile(f, target="cuda", pass_configs=gen.PASS_CONFIGS)
    C = torch.zeros(T_, N, device=dev, dtype=torch.float16)
    kernel(A, packed, scale, C)
    err = ((C.float() - ref).abs() / (ref.abs() + 1.0)).max().item()
    print(f"RESULT {bm}x{bn}x{bk} s{s} t{th}: max relative error {err:.4f}", flush=True)
    ok &= err < 2e-2
print("PASS" if ok else "FAIL")
