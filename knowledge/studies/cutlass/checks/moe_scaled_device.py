"Replay scaled MoE frontend, overlap regions, K0 clocks and numeric ordering."

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import struct
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import yaml
from moe_dense import EpilogueScalar, scalar_module

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Template

STUDY = Path(__file__).resolve().parents[1]
SOURCE = "examples/python/CuTeDSL/cute/blackwell/kernel/moe/torch_scaled_grouped_mm.py"


def evaluate(expression: ast.expr, values: dict) -> object:
    return eval(
        compile(ast.Expression(copy.deepcopy(expression)), "retained-expression", "eval"), values
    )


def f32(value: Fraction | float | int) -> Fraction:
    return Fraction(struct.unpack("<f", struct.pack("<f", float(value)))[0])


def bf16(value: Fraction) -> Fraction:
    bits = int.from_bytes(struct.pack("<f", float(value)), "little")
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return Fraction(struct.unpack("<f", bits.to_bytes(4, "little"))[0])


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    paths = [SOURCE, "python/CuTeDSL/cutlass/utils/blackwell_helpers.py"]
    trees = {}
    for path in paths:
        data = (root / path).read_bytes()
        entry = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        trees[Path(path).name] = ast.parse(data.decode())
    tree = trees["torch_scaled_grouped_mm.py"]
    ns = {}
    exec(
        compile(
            scalar_module(
                tree, {"ScaledGroupedGemmKernel": {"_validate_mma_tiler_and_cluster_shape"}}
            ),
            "retained-frontend",
            "exec",
        ),
        ns,
    )
    validate = ns["ScaledGroupedGemmKernel"]._validate_mma_tiler_and_cluster_shape
    accepted = rejected = 0
    for two, m, n, sf, k, cluster in itertools.product(
        (False, True),
        (64, 128, 256),
        (64, 128, 192, 256),
        (16, 32),
        (64, 128, 256, 768),
        ((1, 1), (2, 1), (4, 4), (8, 1)),
    ):
        config = SimpleNamespace(
            mma_tiler_mnk=(m, n, k), cluster_shape_mn=cluster, use_2cta_instrs=two, sf_vec_size=sf
        )
        expected = (
            m == (256 if two else 128)
            and n in (64, 128, 256)
            and k % (4 * sf) == 0
            and (not two or cluster[0] % 2 == 0)
            and max(cluster) <= 4
        )
        try:
            validate(config)
        except ValueError:
            assert not expected
            rejected += 1
        else:
            assert expected
            accepted += 1
    cls = next(
        x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ScaledGroupedGemmKernel"
    )
    constructor = next(
        x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"
    )
    accumulate_guard = next(
        x
        for x in constructor.body
        if isinstance(x, ast.If) and "accumulate_on_output" in ast.unparse(x.test)
    )
    code = compile(
        ast.fix_missing_locations(ast.Module([copy.deepcopy(accumulate_guard)], [])),
        "retained-accumulation-guard",
        "exec",
    )
    for scenario, accumulate in itertools.product(("2Dx3D", "2Dx2D"), (False, True)):
        try:
            exec(code, {"scenario": scenario, "accumulate_on_output": accumulate})
        except ValueError:
            assert scenario == "2Dx3D" and accumulate
        else:
            assert scenario != "2Dx3D" or not accumulate
    setup = next(
        x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_setup_attributes"
    )
    kernel = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "kernel")
    expressions = {
        ast.unparse(x.targets[0]): x.value
        for x in ast.walk(setup)
        if isinstance(x, ast.Assign) and len(x.targets) == 1
    }
    helper_tree = trees["blackwell_helpers.py"]
    atom_factory = next(
        x
        for x in helper_tree.body
        if isinstance(x, ast.FunctionDef) and x.name == "_make_blockscaled_trivial_tiled_mma_impl"
    )
    # Actual atom branch fixes native K64 for both FP4 formats and K32 for MXFP8.
    atom_text = ast.unparse(atom_factory)
    assert "(*mma_tiler_mn, 64)" in atom_text and "(*mma_tiler_mn, 32)" in atom_text
    epi = copy.deepcopy(
        next(
            x
            for x in helper_tree.body
            if isinstance(x, ast.FunctionDef) and x.name == "compute_epilogue_tile_size"
        )
    )
    epi.decorator_list = []
    module = ast.Module([ast.ImportFrom("__future__", [ast.alias("annotations")], 0), epi], [])
    exec(compile(ast.fix_missing_locations(module), "retained-epilogue-size", "exec"), ns)
    storage_path = STUDY.parents[1] / "seed/templates/physical_storage.yaml"
    storage = Template.model_validate(yaml.safe_load(storage_path.read_text()))
    lifetime = Expr.parse(storage.preconditions[0].expr, ["level", "buffer", "share_order"])
    region_domains = positive_releases = 0
    release_counters = []
    for kind, native_k, sf, k, bits, two in itertools.product(
        ("mxfp8", "mxfp4", "nvfp4"),
        (32, 64),
        (16, 32),
        (128, 256, 384, 512, 640),
        (16, 32),
        (False, True),
    ):
        if native_k != (32 if kind == "mxfp8" else 64) or sf != (16 if kind == "nvfp4" else 32):
            continue
        tile = (128, 256, k)
        config = SimpleNamespace(
            cta_tile_shape_mnk=tile,
            cta_tile_shape_mnk_sfb=tile,
            num_acc_stage=2,
            overlapping_accum=True,
        )
        env = {"self": config, "sf_atom_mn": 32, "mma_inst_tile_k": k // native_k}
        config.num_sfa_tmem_cols = evaluate(expressions["self.num_sfa_tmem_cols"], env)
        config.num_sfb_tmem_cols = evaluate(expressions["self.num_sfb_tmem_cols"], env)
        config.num_sf_tmem_cols = evaluate(expressions["self.num_sf_tmem_cols"], env)
        config.num_accumulator_tmem_cols = evaluate(
            expressions["self.num_accumulator_tmem_cols"], env
        )
        _, config.epi_tile_n = ns["compute_epilogue_tile_size"](128, 256, two, bits, None, False)
        release_at = evaluate(expressions["self.iter_acc_early_release_in_epilogue"], env)
        stride = 256 - config.num_sf_tmem_cols
        if stride <= 0:
            continue
        assert config.num_accumulator_tmem_cols + config.num_sf_tmem_cols == 512
        for slot in (0, 1):
            base, next_base = slot * stride, (1 - slot) * stride
            overlap = set(range(base, base + 256)) & set(range(next_base, next_base + 256))
            read = set()
            release_seen = False
            for i in range(256 // config.epi_tile_n):
                subtile = 256 // config.epi_tile_n - 1 - i if slot == 0 else i
                read.update(
                    range(
                        base + subtile * config.epi_tile_n, base + (subtile + 1) * config.epi_tile_n
                    )
                )
                if i == release_at:
                    assert overlap <= read
                    # Logical event order: required read, fence/release, then overwrite.
                    assert lifetime.eval(
                        {"level": "tmem", "buffer": "shared", "share_order": "this_first"},
                        p={"last_read": 3 * i, "delta": 1, "next_write": 3 * i + 2},
                    )
                    assert not lifetime.eval(
                        {"level": "tmem", "buffer": "shared", "share_order": "this_first"},
                        p={"last_read": 3 * i, "delta": 1, "next_write": 3 * i},
                    )
                    release_seen = True
                    positive_releases += 1
            assert release_seen and read == set(range(base, base + 256))
            region_domains += 1
        release_counters.append(release_at + 1)
    # Valid source frontend does not establish legal compressed TMEM geometry.
    bad = SimpleNamespace(
        mma_tiler_mnk=(128, 256, 768),
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
        sf_vec_size=32,
    )
    validate(bad)
    bad_sf = (4 + 8) * (768 // 32)
    assert bad_sf == 288 and 256 - bad_sf == -32
    assert bad_sf // 32 >= 256 // 32

    # Source-selected stage expressions, including repeated phases across K0 work.
    mma = next(
        x
        for x in kernel.body
        if isinstance(x, ast.If) and ast.unparse(x.test) == "warp_idx == self.mma_warp_id"
    )
    epilogue = next(
        x
        for x in kernel.body
        if isinstance(x, ast.If) and ast.unparse(x.test) == "warp_idx < self.mma_warp_id"
    )
    producer_slot = next(
        x.value
        for x in ast.walk(mma)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "acc_stage_index"
    )
    consumer_slot = next(
        x.value
        for x in ast.walk(epilogue)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "acc_stage_index"
    )
    streams = 0
    for stream in itertools.product((0, 1, 3), repeat=5):
        generation = c_items = 0
        for k_tiles in stream:
            producer = SimpleNamespace(phase=1 ^ (generation % 2), index=0)
            consumer = SimpleNamespace(phase=generation % 2, index=0)
            pslot = evaluate(producer_slot, {"acc_producer_state": producer})
            cslot = evaluate(consumer_slot, {"acc_consumer_state": consumer})
            assert pslot == cslot == generation % 2
            if k_tiles:
                generation += 1
            c_items += 1
        assert generation == sum(x > 0 for x in stream) and c_items == len(stream)
        streams += 1
    subtile = next(
        x
        for x in ast.walk(epilogue)
        if isinstance(x, ast.For) and ast.unparse(x.target) == "subtile_idx"
    )
    zero = next(
        i
        for i, x in enumerate(subtile.body)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "acc_vec"
    )
    scalar = ast.Module(copy.deepcopy(subtile.body[zero : zero + 3]), [])
    code = compile(
        ast.fix_missing_locations(EpilogueScalar().visit(scalar)), "retained-epilogue-scale", "exec"
    )
    for scenario, k_tiles, alpha in itertools.product(("2Dx3D", "2Dx2D"), (0, 1), (None, 2, 4)):
        env = {
            "scenario": scenario,
            "k_tile_cnt": k_tiles,
            "stored_acc": 37,
            "global_scale_a": None if alpha is None else 1,
            "alpha": alpha,
        }
        exec(code, env)
        raw = 0 if scenario == "2Dx2D" and k_tiles == 0 else 37
        assert env["acc_vec"] == raw * (alpha if alpha is not None else 1)
    # A missing/nonfinite paired global scale is not covered by the finite identity binding.
    env = {
        "scenario": "2Dx2D",
        "k_tile_cnt": 0,
        "stored_acc": 37,
        "global_scale_a": 1,
        "alpha": float("inf"),
    }
    exec(code, env)
    assert env["acc_vec"] != env["acc_vec"]

    epsilon = Fraction(1, 2**23)
    ga, gb = 1 + epsilon, 1 + 3 * epsilon
    a, b = (3, -2), (2, 3)
    pre = f32(sum(f32(f32(q * ga) * f32(r * gb)) for q, r in zip(a, b, strict=True)))
    post = f32(f32(sum(q * r for q, r in zip(a, b, strict=True))) * f32(ga * gb))
    assert pre == Fraction(1, 2**21) and post == 0 and bf16(pre) != bf16(post)
    # Exact dot of stored pre-scaled FP32 operands also differs, independent of product rounding.
    exact_stored = sum(f32(q * ga) * f32(r * gb) for q, r in zip(a, b, strict=True))
    assert exact_stored != 0
    # SFB N64 pairs use a128-wide load and alternate the selected half.
    slice_if = next(x for x in ast.walk(kernel) if isinstance(x, ast.If) and x.lineno == 1740)
    assert ast.unparse(slice_if.body[0].value) == "work_tile_info.tile_n_idx // 2"
    offset_expr = next(
        x.value.args[0]
        for x in ast.walk(mma)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "offset"
    )
    for n in range(9):
        assert (
            evaluate(offset_expr, {"work_tile_info": SimpleNamespace(tile_n_idx=n)}) == (n % 2) * 2
        )
        assert (n // 2) * 128 + (n % 2) * 64 == n * 64
    sched = next(
        x
        for x in kernel.body
        if isinstance(x, ast.If) and ast.unparse(x.test) == "warp_idx == self.sched_warp_id"
    )
    last_commit = max(
        i
        for i, x in enumerate(sched.body)
        if isinstance(x, ast.Expr)
        and isinstance(x.value, ast.Call)
        and ast.unparse(x.value.func) == "sched_pipeline.producer_commit"
    )
    assert not any(
        "sched_producer_state.advance" in ast.unparse(x) for x in sched.body[last_commit + 1 :]
    )
    source_text = (root / SOURCE).read_text()
    assert "if max_diff == 0.0:" in source_text and "skip ref check." in source_text
    assert "args.consistent_token_padding = False" in source_text
    fusion_path = (
        STUDY.parents[1] / "seed/references/cutlass_ampere_tf32_gemm/templates/operator_fusion.yaml"
    )
    fusion = Template.model_validate(yaml.safe_load(fusion_path.read_text()))
    variables = {
        "consumer_kind": "elementwise",
        "intermediate_site": "registers",
        "fused_ops": "scale_and_cast",
    }
    for guard in [*fusion.constraints, *fusion.preconditions]:
        assert Expr.parse(guard.expr, variables).eval(variables, p={})
    result = {
        "outcome": "PASS",
        "frontend_accepted": accepted,
        "frontend_rejected": rejected,
        "overlap_region_domains": region_domains,
        "positive_k_release_events": positive_releases,
        "mixed_k_clock_streams": streams,
        "accepted_but_invalid_geometry": {
            "tile_k": 768,
            "sf_columns": bad_sf,
            "slot_stride_columns": -32,
            "release_index": 9,
            "subtile_count": 8,
        },
        "global_scale_counterexample": {
            "ga": str(ga),
            "gb": str(gb),
            "pre_scaled_fp32_dot": str(pre),
            "post_scaled_raw_dot": str(post),
            "exact_dot_of_stored_inputs": str(exact_stored),
            "survives_bf16_cast": True,
        },
        "boundary": (
            "Scalar retained frontend/expressions and region/numeric models; n"
            "ative descriptor, TMEM mapping, overlap progress and accumulation"
            " arithmetic remain open."
        ),
    }
    binding = {
        "format": "mxfp8",
        "data_scale_output_acc": ["E4M3", "E8M0", "BF16", "FP32"],
        "sf_vec_size": 32,
        "mma_mnk": [128, 256, 128],
        "native_instruction_k": 32,
        "cluster_mnk": [1, 1, 1],
        "hidden": 256,
        "intermediate": 256,
        "expert_token_lengths": [32, 0, 160, 64],
        "sf_columns": 48,
        "accumulator_regions_columns": [[0, 256], [208, 464]],
        "scale_region_columns": [464, 512],
        "epilogue_n": 32,
        "physical_accumulator_regions": 2,
        "accumulator_pipeline_stages": 1,
        "release_after_subtiles": 2,
        "region_reuse": "physical_storage@1 pre_lifetime only; traversal/layout stay source glue",
        "final_scale_reuse": {"identity": "operator_fusion@1", "variables": variables},
        "native_obligations": [
            "TMEM/lane maps and completion",
            "descriptor/proxy visibility",
            "finite computed alpha for empty identity",
            "admitted global-scale factoring",
            "legal derived resource stages",
            "corrected scheduler exit protocol",
        ],
    }
    (STUDY / "inputs/moe-scaled-device-binding.json").write_text(
        json.dumps(binding, indent=2) + "\n"
    )
    (STUDY / "checks/moe-scaled-device-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: scaled MoE overlap regions, K0 clocks, frontend and numeric boundaries.\n"
    )


if __name__ == "__main__":
    main()
