"Substitute source layer/tile/type bindings and recover generated instruction declarations."

from __future__ import annotations

import ast
import json
import sys
import types

from policy import SOURCE, STUDY, build_scope, load


def main() -> None:
    scope = build_scope()
    path = "humming/config/mma.py"
    constants = {"DTYPE_BIT_WIDTH_MAP", "DTYPE_MAP", "SF_DTYPE_MAP"}
    nodes = [
        n
        for n in ast.parse((SOURCE / path).read_text()).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(SOURCE / path),
            "exec",
        ),
        scope,
    )
    load(
        path,
        scope,
        {
            "calc_reg_count",
            "get_mma_b_dtype",
            "get_mxmma_scale_config",
            "MmaOpClassImpl",
            "WgmmaOpClassImpl",
            "UmmaOpClassImpl",
            "MxMmaOpClassImpl",
            "MmaOpClass",
        },
    )
    cls = next(
        n
        for n in ast.parse((SOURCE / "humming/kernel/humming.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "HummingKernel"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("select_mma_dtypes", "select_mma_op_class")
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])),
            str(SOURCE / "humming/kernel/humming.py"),
            "exec",
        ),
        scope,
    )
    rows = []
    for sm, mma, a, b, block, warp, group in (
        (75, "mma", "float16", "uint4", (16, 128, 64), (16, 32, 32), 32),
        (80, "mma", "float16", "uint4", (16, 128, 64), (16, 32, 32), 32),
        (90, "wgmma", "float16", "uint4", (64, 128, 64), (64, 32, 64), 0),
        (100, "umma", "float16", "uint4", (64, 128, 64), (64, 32, 64), 0),
        (120, "mxmma", "float4e2m1", "float4e2m1", (16, 128, 128), (16, 32, 128), 32),
    ):
        kwargs = {
            "sm_version": sm,
            "shape_n": 128,
            "shape_k": 128,
            "a_dtype": a,
            "b_dtype": b,
            "c_dtype": "float16",
            "weight_scale_group_size": group,
        }
        if mma == "mxmma":
            kwargs |= {"bs_dtype": "float8e8m0", "input_scale_group_size": 32}
        layer = scope["LayerConfig"](**kwargs)
        tuning = scope["TuningConfig"](
            mma_type=mma,
            block_shape=block,
            warp_shape=warp,
            num_stages=4,
            use_warp_spec=mma in ("wgmma", "umma"),
        )
        facts = layer.__dict__ | tuning.__dict__
        facts |= {
            "use_f16_accum": False,
            "use_umma_ss": False,
            "mxmma_native_mixed": False,
            "warp_shape": warp,
        }
        obj = types.SimpleNamespace(**facts)
        scope["select_mma_dtypes"](obj)
        op = scope["select_mma_op_class"](obj)
        text = op.generate_ptx()
        if mma in ("mma", "wgmma", "mxmma"):
            assert ("mma_async" if mma == "wgmma" else "mma.sync") in text
        if sm == 80:
            assert "m16n8k16.row.col.f32.f16.f16.f32" in text
        if sm == 90:
            assert "m64n64k16.f32.f16.f16" in text
        if sm == 120:
            assert "block_scale.scale_vec::2X.m16n8k64" in text
        smem = scope["estimate_smem_size_layer"](
            layer,
            block,
            scope["GemmType"].DENSE,
            4,
            warp_shape=warp,
            mma_type=mma,
            use_warp_spec=tuning.use_warp_spec,
        )
        rows.append(
            {
                "sm": sm,
                "mma": mma,
                "block": block,
                "warp": warp,
                "selected_mma_dtype_B": str(obj.mma_b_dtype),
                "instruction_shape": op.shape,
                "estimated_smem_bytes": smem,
                "generated_instruction": next(
                    (line.strip() for line in text.splitlines() if "mma" in line),
                    "UMMA descriptor path",
                ),
                "boundary": "Source-generated declaration only; no CUDA compilation or launch",
            }
        )
    result = {
        "outcome": "PASS",
        "kind": "source_specialization_replay",
        "gpu_execution": False,
        "rows": rows,
        "open": "Target/compiler validity, complete fragment maps and runtime checks",
    }
    (STUDY / "checks/specializations-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: SM75/80/90/100/120 source type and instruction bindings recovered.\n")


if __name__ == "__main__":
    main()
