"""Replay SM120 scale layout metadata and host logical-coordinate preparation."""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/"


def dot_nested(coord, stride):
    if isinstance(stride, int):
        return coord * stride
    return sum(dot_nested(c, s) for c, s in zip(coord, stride, strict=True))


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    names = [
        "dense_blockscaled_gemm_persistent_cooperative.py",
        "dense_blockscaled_gemm_persistent_pingpong.py",
    ]
    paths = [
        *(PREFIX + name for name in names),
        "python/CuTeDSL/cutlass/utils/blockscaled_layout.py",
        "python/CuTeDSL/cutlass/utils/blackwell_helpers.py",
        "python/CuTeDSL/cutlass/torch.py",
    ]
    trees = {}
    for path in paths:
        data = (root / path).read_bytes()
        entry = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        trees[Path(path).name] = ast.parse(data.decode())
    scale = trees["blockscaled_layout.py"]
    shared_cases = coordinates = 0
    for native_k, sf, tile_k, mn, name in itertools.product(
        (32, 64),
        (16, 32),
        (128, 256),
        (128, 256),
        ("sm120_make_smem_layout_sfa", "sm120_make_smem_layout_sfb"),
    ):
        if native_k == 32 and sf != 32:
            continue
        function = next(x for x in scale.body if isinstance(x, ast.FunctionDef) and x.name == name)
        prefix = []
        for statement in function.body:
            if (
                isinstance(statement, ast.Assign)
                and ast.unparse(statement.targets[0]) == "smem_layout"
            ):
                break
            prefix.append(copy.deepcopy(statement))
        env = {
            "sf_vec_size": sf,
            "tile_shape_mnk": (mn, mn, tile_k),
            "tiled_mma": SimpleNamespace(shape_mnk=(16, 8, native_k)),
        }
        exec(compile(ast.Module(prefix, []), "retained-sf-metadata", "exec"), env)
        strides = env["sSFA_stride"]
        nsf = native_k // sf
        physical = {}
        for row, k in itertools.product(range(mn), range(tile_k)):
            group = k // sf
            coord = (
                ((row % 32, (row // 32) % 4), row // 128),
                ((k % sf, group % nsf), (group // nsf) % (4 // nsf), group // 4),
            )
            address = dot_nested(coord, strides)
            expected = (
                512 * ((group // 4) * (mn // 128) + row // 128)
                + 16 * (row % 32)
                + 4 * ((row // 32) % 4)
                + group % 4
            )
            assert address == expected and 0 <= address < mn * tile_k // sf
            if address in physical:
                assert physical[address] == (row, group)
            physical[address] = (row, group)
            coordinates += 1
        assert len(physical) == mn * tile_k // sf
        shared_cases += 1
    host_cases = host_coordinates = 0
    for filename in names:
        tree = trees[filename]
        kernel_class = next(x for x in tree.body if isinstance(x, ast.ClassDef))
        alignment = next(
            x.value
            for x in ast.walk(kernel_class)
            if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "self.buffer_align_bytes"
        )
        assert ast.literal_eval(alignment) == 1024
        if "pingpong" in filename:
            storage = next(
                x
                for x in ast.walk(kernel_class)
                if isinstance(x, ast.ClassDef) and x.name == "SharedStorage"
            )
            fields = [ast.unparse(x.target) for x in storage.body if isinstance(x, ast.AnnAssign)]
            assert fields[:3] == [
                "mainloop_pipeline_array_ptr",
                "math_wg_order_barrier_array_ptr",
                "sA",
            ]
        host = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "run_bs")
        create = next(
            x
            for x in host.body
            if isinstance(x, ast.FunctionDef) and x.name == "create_scale_factor_tensor"
        )
        perm = next(
            x.value
            for x in create.body
            if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "mma_permute_order"
        )
        order = ast.literal_eval(perm)
        assert order == (3, 4, 1, 5, 2, 0)
        for batch, mn, k, sf in itertools.product(
            (1, 2), (128, 144, 256), (128, 160, 256), (16, 32)
        ):
            rm = (mn + 127) // 128
            cols = (k + sf - 1) // sf
            rk = (cols + 3) // 4
            shape = (batch, rm, rk, 32, 4, 4)
            strides = []
            size = 1
            for extent in reversed(shape):
                strides.append(size)
                size *= extent
            strides = tuple(reversed(strides))
            permuted_stride = tuple(strides[axis] for axis in order)
            seen = set()
            for batch_index, row, column in itertools.product(range(batch), range(mn), range(cols)):
                view = (row % 32, (row // 32) % 4, row // 128, column % 4, column // 4, batch_index)
                address = sum(c * s for c, s in zip(view, permuted_stride, strict=True))
                direct = (
                    (((batch_index * rm + row // 128) * rk + column // 4) * 32 + row % 32) * 4
                    + (row // 32) % 4
                ) * 4 + column % 4
                assert address == direct and address not in seen
                seen.add(address)
                host_coordinates += 1
            assert len(seen) == batch * mn * cols
            host_cases += 1
        source = ast.unparse(host)
        assert "RandomInitConfig(min_val=1, max_val=3)" in source
        assert "ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]" in source
        assert "internal_type=cutlass.Int16" in ast.unparse(
            next(x for x in tree.body if isinstance(x, ast.ClassDef))
        )
    helper = next(
        x
        for x in trees["blackwell_helpers.py"].body
        if isinstance(x, ast.FunctionDef) and x.name == "get_permutation_mnk"
    )
    assert "perm_n_shape = (8, 2, 2)" in ast.unparse(helper)
    assert "perm_n_stride = (1, 16, 8)" in ast.unparse(helper)
    permutation = [
        x + 16 * y + 8 * z for z, y, x in itertools.product(range(2), range(2), range(8))
    ]
    assert sorted(permutation) == list(range(32)) and permutation != list(range(32))
    torch_helper = trees["torch.py"]
    matrix = next(
        x for x in torch_helper.body if isinstance(x, ast.FunctionDef) and x.name == "matrix"
    )
    assert "min_val = -2" in ast.unparse(matrix) and "max_val = 2" in ast.unparse(matrix)
    # Declared-struct interval model, conditional on native struct lowering.
    padding = []
    for stages in range(1, 17):
        order_start = 16 * stages
        declared_end = order_start + 16
        next_aligned = ((declared_end + 1023) // 1024) * 1024
        required_end = order_start + 32
        assert required_end <= next_aligned
        padding.append(
            {"stages": stages, "required_end": required_end, "next_field_start": next_aligned}
        )
    result = {
        "outcome": "PASS",
        "shared_metadata_domains": shared_cases,
        "shared_scale_coordinates": coordinates,
        "host_layout_domains": host_cases,
        "host_scale_coordinates": host_coordinates,
        "n_permutation_entries": len(permutation),
        "order_padding_layout_cases": len(padding),
        "boundary": (
            "Scalar source layout/coordinate metadata only; native TMA/SIMT la"
            "ne mapping, struct offsets, output retirement and numerical behav"
            "ior remain obligations."
        ),
    }
    (STUDY / "checks/geforce-layouts-result.json").write_text(json.dumps(result, indent=2) + "\n")
    (STUDY / "inputs/geforce-order-padding.json").write_text(
        json.dumps(
            {
                "premise": (
                    "Source struct lowering preserves declared field order and1024-byt"
                    "e alignment; source ABI not compiled."
                ),
                "cases": padding,
                "conclusion": (
                    "These positive stage counts have backing padding in the scalar mo"
                    "del; two-slot declaration alone was not proof of native out-of-bo"
                    "unds."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    sys.stdout.write(
        "PASS: SM120 host/shared scale coordinates, N permutation and conditional order padding.\n"
    )


if __name__ == "__main__":
    main()
