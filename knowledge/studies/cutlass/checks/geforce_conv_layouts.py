"""Recover SM120 convolution host/shared scale layouts and stage-size policy."""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from geforce_layouts import dot_nested
from moe_dense import ScalarSource

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/"


def code(statements):
    return compile(
        ast.fix_missing_locations(ast.Module(copy.deepcopy(statements), [])), "source", "exec"
    )


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    paths = [
        PREFIX + "dense_blockscaled_implicit_gemm_fprop.py",
        PREFIX + "dense_implicit_gemm_fprop.py",
        "python/CuTeDSL/cutlass/torch.py",
    ]
    trees = {}
    for path in paths:
        data = (Path(manifest["tree"]) / path).read_bytes()
        entry = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        trees[Path(path).name] = ast.parse(data)
    tree = trees["dense_blockscaled_implicit_gemm_fprop.py"]
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
    funcs = {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
    methods = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    activations_checked = 0
    for name in ("dense_implicit_gemm_fprop.py", "dense_blockscaled_implicit_gemm_fprop.py"):
        menu = next(
            x.value
            for x in trees[name].body
            if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "EPILOGUE_ACTIVATIONS"
        )
        assert {x.value for x in menu.keys} == {"identity", "relu"}
        for key, value in zip(menu.keys, menu.values, strict=True):
            device = next(
                v for k, v in zip(value.keys, value.values, strict=True) if k.value == "device"
            )
            cute = SimpleNamespace(
                where=lambda condition, yes, no: yes if condition else no,
                full_like=lambda value, fill: fill,
            )
            fn = eval(compile(ast.Expression(device), "source-activation", "eval"), {"cute": cute})
            for x in (-2.0, -0.0, 0.0, 1.0, 2.0):
                assert fn(x) == (x if key.value == "identity" else max(0, x))
                activations_checked += 1
            if key.value == "relu":
                assert fn(float("nan")) == 0
    dense_funcs = {
        x.name: x
        for x in trees["dense_implicit_gemm_fprop.py"].body
        if isinstance(x, ast.FunctionDef)
    }
    boxed = ast.Module([copy.deepcopy(dense_funcs["_rt_conv_scalars"])], [])
    ns = {}
    exec(
        compile(ast.fix_missing_locations(ScalarSource().visit(boxed)), "runtime-boxing", "exec"),
        ns,
    )
    assert ns["_rt_conv_scalars"](
        upper_pad=(1, 2, 3), lower_pad=(4, 5, 6), stride=(7, 8, 9), dil=(10, 11, 12)
    ) == list(range(1, 13))

    sf = funcs["_sm120_make_smem_layout_sfb"]
    end = next(
        i
        for i, x in enumerate(sf.body)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "smem_layout"
    )
    stage_assignments = [
        x
        for x in sf.body[end + 1 :]
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) in {"k_chunks", "stage_stride"}
    ]
    metadata = code([*sf.body[:end], *stage_assignments])
    shared_domains = shared_coords = narrow_domains = 0
    for n, tile_k, vector, native_k in itertools.product(
        (32, 64, 96, 128, 256), (64, 128), (16, 32), (32, 64)
    ):
        if (vector == 16 and native_k != 64) or (vector == 32 and native_k != 32):
            continue
        if tile_k < vector * 4:
            continue
        env = {
            "tile_shape_mnk": (128, n, tile_k),
            "sf_vec_size": vector,
            "num_stages": 3,
            "tiled_mma": SimpleNamespace(shape_mnk=(16, 8, native_k)),
        }
        exec(metadata, env)
        stage_stride = env["stage_stride"]
        seen = set()
        for row, group in itertools.product(range(n), range(tile_k // vector)):
            nsf = native_k // vector
            coord = (
                ((row % 32, (row // 32) % 4), row // 128),
                ((0, group % nsf), (group // nsf) % (4 // nsf), group // 4),
            )
            addr = dot_nested(coord, env["sSFA_stride"])
            assert (
                addr
                == 512 * ((group // 4) * max(1, n // 128) + row // 128)
                + 16 * (row % 32)
                + 4 * ((row // 32) % 4)
                + group % 4
            )
            assert addr not in seen and 0 <= addr < stage_stride
            seen.add(addr)
            shared_coords += 1
        assert stage_stride == max(1, n // 128) * 512 * (tile_k // vector // 4)
        if n < 128:
            assert max(seen) + 1 < stage_stride
            narrow_domains += 1
        shared_domains += 1

    host = funcs["create_scale_factor_tensor_swizzled"]
    end = next(
        i
        for i, x in enumerate(host.body)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "ref_f32_cpu"
    )
    host_prefix = code(host.body[1:end])
    host_domains = host_scales = 0
    for batches, mn, channels, vector in itertools.product(
        (1, 2), (64, 96, 128, 256), (64, 128, 192, 384), (16, 32)
    ):
        env = {"l": batches, "mn": mn, "k": channels, "sf_vec_size": vector}
        exec(host_prefix, env)
        shape, perm = env["mma_shape"], env["mma_permute_order"]
        assert perm == (3, 4, 1, 5, 2, 0)
        strides = []
        for i in range(6):
            strides.append(math.prod(shape[i + 1 :]))
        used = set()
        for batch, row, group in itertools.product(range(batches), range(mn), range(env["sf_k"])):
            original = (batch, row // 128, group // 4, row % 32, (row // 32) % 4, group % 4)
            logical = (row % 32, (row // 32) % 4, row // 128, group % 4, group // 4, batch)
            recovered = [None] * 6
            for axis, coord in zip(perm, logical, strict=True):
                recovered[axis] = coord
            assert tuple(recovered) == original
            offset = sum(c * s for c, s in zip(original, strides, strict=True))
            assert 0 <= offset < math.prod(shape) and offset not in used
            used.add(offset)
            host_scales += 1
        host_domains += 1

    # Symbolically reconcile the size probe with the actual allocated structure.
    runtime = next(
        x
        for x in ast.walk(methods["__call__"])
        if isinstance(x, ast.ClassDef) and x.name == "SharedStorage"
    )
    probe = next(
        x
        for x in ast.walk(methods["_compute_stages"])
        if isinstance(x, ast.ClassDef) and x.name == "ProbeStorage"
    )
    replacements = {
        "a_dtype": "self.smem_alloc_a_dtype",
        "b_dtype": "self.smem_alloc_b_dtype",
        "sf_dtype": "self.sf_dtype",
        "d_dtype": "self.d_dtype",
        "ab_stage": "self.ab_stage",
        "num_clc_stage": "self.num_clc_stage",
        "buffer_align_bytes": "self.buffer_align_bytes",
        "a_smem": "self.a_smem_layout_staged",
        "b_smem": "self.b_smem_layout_staged",
        "sfa_smem": "self.sfa_smem_layout_staged",
        "sfb_smem": "self.sfb_smem_layout_staged",
        "epi_smem": "self.epi_smem_layout_staged",
        "res_smem": "self.res_smem_layout_staged",
        "has_bias": "self.has_bias",
        "has_residual": "self.has_residual",
        "cta_n": "self.tile_shape_mnk[1]",
        "res_stage": "self.res_stage",
    }

    class BindNames(ast.NodeTransformer):
        def visit_Name(self, node):
            return (
                ast.parse(replacements[node.id], mode="eval").body
                if node.id in replacements
                else node
            )

    actual = {
        ast.unparse(x.target): ast.dump(x.annotation)
        for x in runtime.body
        if isinstance(x, ast.AnnAssign)
    }
    sized = {
        ast.unparse(x.target): ast.dump(BindNames().visit(copy.deepcopy(x.annotation)))
        for x in probe.body
        if isinstance(x, ast.AnnAssign)
    }
    assert actual == sized

    # Replay the retained upward-search tail on bounded monotone aligned storage
    # models. These are mathematical cost functions, not measured SM120 layouts.
    body = methods["_compute_stages"].body
    search_start = next(
        i
        for i, x in enumerate(body)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "align_loss"
    )
    search_body = copy.deepcopy(body[search_start:-1])
    search_code = code(search_body)
    stage_domains = 0
    for slopes, fixed, capacity in itertools.product(
        ((4096, 2048, 512, 512), (8192, 6144, 1024, 1024), (1536, 768, 512, 512)),
        (1024, 2048, 8192),
        (16384, 32768, 65536, 100000),
    ):

        def cost(stages, fixed=fixed, slopes=slopes):
            return fixed + 16 * stages + sum(((s * stages + 1023) // 1024) * 1024 for s in slopes)

        if cost(1) > capacity:
            continue
        env = {
            "buffer_align_bytes": 1024,
            "smem_capacity": capacity,
            "smem_bytes": cost,
            "slope": sum(slopes) + 16,
        }
        exec(search_code, env)
        chosen = env["ab_stage"]
        assert cost(chosen) <= capacity < cost(chosen + 1)
        stage_domains += 1

    result = {
        "outcome": "PASS",
        "shared_metadata_domains": shared_domains,
        "shared_scale_coordinates": shared_coords,
        "helper_only_narrow_domains": narrow_domains,
        "host_layout_domains": host_domains,
        "distinct_host_scales": host_scales,
        "probe_runtime_fields_matched": len(actual),
        "finite_activation_values": activations_checked,
        "abstract_stage_search_domains": stage_domains,
        "boundary": (
            "Retained scale-layout metadata, host permutation constants and sy"
            "mbolic field matching; stage search on abstract monotone costs. N"
            "ative CuTe/ABI cosizes, float encoding, GPU transfer and measured"
            " optimality remain open."
        ),
    }
    (STUDY / "checks/geforce-conv-layouts-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: convolution scale metadata, host coordinates and storage-probe agreement.\n"
    )


if __name__ == "__main__":
    main()
