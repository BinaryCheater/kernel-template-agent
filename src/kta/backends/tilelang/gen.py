# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess=false
"""TileLang codegen, run inside the worker image (it imports only tilelang).

Usage: python gen.py kernel.py <entry> <arch>
Writes out/kernel.cu and prints JSON: {"kernel_entry", "block", "params", "tilelang",
"pass_configs", "dynamic_smem", "grid"}.

``kernel.py`` may define ``PASS_CONFIGS`` (a dict of TileLang pass-config keys); they are
applied through a PassContext, as ``tilelang.jit`` does. ``dynamic_smem`` is the device
function's ``dyn_shared_memory_buf`` attribute (bytes the launch must provide) and
``grid`` its blockIdx extents; both are null when the lowered IR does not state them.
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
GRID_TAGS = ("blockIdx.x", "blockIdx.y", "blockIdx.z")


def device_info(artifact):
    """Dynamic shared memory and grid extents of the single device function."""
    mod = getattr(artifact, "device_mod", None)
    if mod is None:
        return None, None
    from tvm import tirx

    smem, grid = None, None
    for _, func in mod.functions.items():
        attrs = func.attrs
        if attrs is None or "global_symbol" not in attrs:
            continue
        if "dyn_shared_memory_buf" in attrs:
            smem = int(attrs["dyn_shared_memory_buf"])
        extents = {}

        def visit(node, extents=extents):
            if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
                extents[str(node.node.thread_tag)] = node.value

        tirx.stmt_functor.post_order_visit(func.body, visit)
        values = [extents.get(t) for t in GRID_TAGS]
        if all(v is None or hasattr(v, "value") for v in values):
            grid = [int(v.value) if v is not None else 1 for v in values]
    return smem, grid


def main(path: str, entry: str, arch: str) -> None:
    spec = importlib.util.spec_from_file_location("kta_kernel", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, entry)
    configs = dict(getattr(module, "PASS_CONFIGS", None) or {})
    target = {"kind": "cuda", "arch": arch}
    with tvm.transform.PassContext(opt_level=3, config=configs), tvm.target.Target(target):
        artifact = tilelang.lower(func, target=target)
    source = artifact.kernel_source
    with open("out/kernel.cu", "w") as f:
        f.write(source)
    kernels = KERNEL_DEF.findall(source)
    if len(kernels) != 1:
        raise SystemExit(f"expected one kernel definition, found {len(kernels)}")
    bounds, name, args = kernels[0]
    params = [a.strip().split()[-1] for a in args.split(",") if a.strip()]
    smem, grid = device_info(artifact)
    info = {
        "kernel_entry": name,
        "block": int(bounds) if bounds else None,
        "params": params,
        "tilelang": tilelang.__version__,
        "pass_configs": configs,
        "dynamic_smem": smem,
        "grid": grid,
    }
    json.dump(info, sys.stdout)


if __name__ == "__main__":
    main(*sys.argv[1:4])
