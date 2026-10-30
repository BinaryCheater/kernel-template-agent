# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess=false
"""TileLang codegen, run inside the worker image (it imports only tilelang).

Usage: python gen.py kernel.py <entry> <arch>
Writes out/kernel.cu and prints JSON: {"kernel_entry", "block", "params", "tilelang"}.
"""

import importlib.util
import json
import re
import sys

import tilelang
from tilelang import tvm

KERNEL_DEF = re.compile(
    r'extern "C" __global__ void (?:__launch_bounds__\((\d+)[^)]*\) )?(\w+)\(([^)]*)\)\s*\{'
)


def main(path: str, entry: str, arch: str) -> None:
    spec = importlib.util.spec_from_file_location("kta_kernel", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, entry)
    target = {"kind": "cuda", "arch": arch}
    with tvm.target.Target(target):
        artifact = tilelang.lower(func, target=target)
    source = artifact.kernel_source
    with open("out/kernel.cu", "w") as f:
        f.write(source)
    kernels = KERNEL_DEF.findall(source)
    if len(kernels) != 1:
        raise SystemExit(f"expected one kernel definition, found {len(kernels)}")
    bounds, name, args = kernels[0]
    params = [a.strip().split()[-1] for a in args.split(",") if a.strip()]
    info = {
        "kernel_entry": name,
        "block": int(bounds) if bounds else None,
        "params": params,
        "tilelang": tilelang.__version__,
    }
    json.dump(info, sys.stdout)


if __name__ == "__main__":
    main(*sys.argv[1:4])
