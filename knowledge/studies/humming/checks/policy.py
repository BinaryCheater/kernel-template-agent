"Replay a closed dense SM90 source slice; target facts are inputs, never GPU observations."

from __future__ import annotations

import ast
import dataclasses
import enum
import functools
import json
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
SOURCE = Path(json.loads((STUDY / "inputs/sources.json").read_text())["humming"]["root"])


def load(
    path: str, scope: dict, names: set[str] | None = None, dtype_constants: bool = False
) -> None:
    nodes = []
    for node in ast.parse((SOURCE / path).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (
            names is None or node.name in names
        ):
            nodes.append(node)
        if dtype_constants and isinstance(node, ast.Assign) and 189 <= node.lineno <= 223:
            nodes.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(SOURCE / path), "exec"), scope)


def namespace(name: str) -> dict:
    module = types.ModuleType(name)
    sys.modules[name] = module
    module.__dict__.update(
        {
            "dataclasses": dataclasses,
            "enum": enum,
            "Enum": enum.Enum,
            "json": json,
            "re": re,
            "os": os,
            "math": math,
            "functools": functools,
            "ClassVar": ClassVar,
            "Any": Any,
            "Literal": Literal,
            "np": np,
        }
    )
    return module.__dict__


def build_scope() -> dict:
    scope = namespace("humming_source_study_policy")
    dtype_scope = namespace("humming_source_study_dtypes")
    load("humming/dtypes.py", dtype_scope, {"DataType", "IntegerType", "FloatingPointType"}, True)
    scope["dtypes"] = sys.modules[dtype_scope["__name__"]]
    load("humming/config/enum.py", scope)
    load("humming/config/base.py", scope)
    load("humming/utils/math.py", scope)
    load("humming/utils/cubin.py", scope, {"is_cubin_patch_enabled"})
    # All hardware fields below are declared analysis inputs. No CUDA API is called.
    scope["current_device"] = types.SimpleNamespace(is_ppu=False, sm_version=90, sm_count=132)
    load("humming/config/mma.py", scope, {"get_dtype_compatibility_error", "get_default_mma_type"})
    load("humming/config/config.py", scope, {"LayerConfig", "TuningConfig"})
    load(
        "humming/utils/smem.py",
        scope,
        {"_struct_size", "_stage_storage_bytes", "_next_pow2", "estimate_smem_size_layer"},
    )
    scope["_INT4"] = 16
    load("humming/tune/base.py", scope, {"DeviceHeuristics"})
    load("humming/tune/candidate.py", scope, {"DeviceProfile", "TuningProblem", "TuningDecision"})
    selected = {
        "get_sm90_specialized_config",
        "get_compact_scaled_moe_config",
        "get_short_k_config",
        "get_block_scaled_moe_config",
        "get_scaled_a8_config",
        "get_overlap_config",
        "get_packed_wna8_config",
        "fit_sm90_tile",
        "get_compact_moe_config",
        "calc_sm90_num_block_list",
        "select_sm90_block_m",
        "build_sm90_seed_config",
    }
    load("humming/tune/sm90_policies.py", scope, selected)
    # Bind original methods without importing Sm80/device modules. Unsupported inherited
    # paths are absent and fail; no returned configuration is supplied by a replacement.
    wanted = {
        "_select_base_config",
        "_select_config",
        "get_config",
        "get_config1",
        "_make_problem",
        "get_device_profile",
        "_uses_moe_a16_policy",
        "_uses_grouped_scale_candidates",
    }
    node = next(
        n
        for n in ast.parse((SOURCE / "humming/tune/sm90.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Sm90Heuristics"
    )
    methods = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name in wanted]
    cls = ast.ClassDef(
        name="DenseSourcePolicy",
        bases=[ast.Name(id="DeviceHeuristics", ctx=ast.Load())],
        keywords=[],
        body=[
            *methods,
            ast.parse("sm_version = 90").body[0],
            ast.parse("max_smem_size = 227 * 1024").body[0],
        ],
        decorator_list=[],
    )
    tree = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                cls,
            ],
            type_ignores=[],
        )
    )
    exec(compile(tree, str(SOURCE / "humming/tune/sm90.py"), "exec"), scope)
    return scope


def main() -> None:
    scope = build_scope()
    layer = scope["LayerConfig"](
        sm_version=90,
        shape_n=8192,
        shape_k=1024,
        a_dtype="float16",
        b_dtype="int4",
        c_dtype="float16",
    )
    rows = []
    for m in (1, 8, 9, 16, 17, 32, 33, 64, 65, 128):
        config = scope["DenseSourcePolicy"].get_config(layer, m)
        mma = scope["MmaType"](config.get("mma_type") or scope["get_default_mma_type"](layer))
        tuning = scope["TuningConfig"](
            **{k: v for k, v in config.items() if k in scope["TuningConfig"].get_field_names()}
        )
        smem = scope["estimate_smem_size_layer"](
            layer,
            config["block_shape"],
            scope["GemmType"].DENSE,
            config["num_stages"],
            warp_shape=config["warp_shape"],
            use_tma=config.get("use_tma", False),
            use_warp_spec=config.get("use_warp_spec", False),
            mma_type=mma,
            smem_reuse_mode=config.get("smem_reuse_mode", "all_stages"),
        )
        assert smem * config.get("num_ctas_per_sm", 1) <= 227 * 1024
        sampled = scope["DenseSourcePolicy"].get_shape_m_sample(layer, m)
        seed_menu = []
        if mma == scope["MmaType"].WGMMA:
            counts = scope["calc_sm90_num_block_list"](layer, sampled, 176, scope["GemmType"].DENSE)
            seed_menu = [
                {"block_m": bm, "M_tile_count_score": count}
                for bm, count in zip(range(8, 177, 8), counts, strict=True)
            ]
        rows.append(
            {
                "M": m,
                "seed_candidate_menu": seed_menu,
                "sampled_M": scope["DenseSourcePolicy"].get_shape_m_sample(layer, m),
                "mma": mma.value,
                "config": config,
                "estimated_smem_bytes": smem,
                "threads": tuning.num_threads,
            }
        )
    assert rows[5]["mma"] == "mma" and rows[6]["mma"] == "wgmma"
    result = {
        "outcome": "PASS",
        "kind": "closed_source_slice_replay",
        "gpu_execution": False,
        "target_facts": vars(scope["current_device"]),
        "layer": "W4A16, N8192 K1024, per-channel scale",
        "rows": rows,
        "boundary": (
            "No CUDA/compiler/driver, no empirical performance; dense FP16 sli"
            "ce only; no invented return value for unavailable source paths."
        ),
    }
    (STUDY / "checks/policy-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: dense SM90 M menu and full tile fitting replayed from source.\n")


if __name__ == "__main__":
    main()
