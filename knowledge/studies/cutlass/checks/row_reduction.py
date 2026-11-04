"""Replay RMSNorm configuration/TV ownership and replicated scalar reductions."""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import operator
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import yaml
from moe_dense import ScalarSource

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Chain, Template

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/"


def xor_fold(values: list[int], op) -> list[int]:
    offset = len(values) // 2
    while offset:
        previous = values[:]
        values = [op(value, previous[lane ^ offset]) for lane, value in enumerate(previous)]
        offset //= 2
    return values


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    paths = [
        PREFIX + "reduce/reduce.py",
        PREFIX + "rmsnorm/rmsnorm.py",
        "python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py",
        "python/CuTeDSL/cutlass/cute/nvgpu/cpasync/copy.py",
    ]
    trees = {}
    for path in paths:
        data = (root / path).read_bytes()
        entry = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        trees[Path(path).name] = ast.parse(data.decode())
    rms = trees["rmsnorm.py"]
    config = copy.deepcopy(
        next(x for x in rms.body if isinstance(x, ast.ClassDef) and x.name == "RMSNormConfig")
    )
    module = ast.Module([config], [])
    ns = {}
    exec(
        compile(
            ast.fix_missing_locations(ScalarSource().visit(module)),
            "retained-rmsnorm-config",
            "exec",
        ),
        ns,
    )
    config_type = ns["RMSNormConfig"]
    config_cases = ownership_domains = ownership_coordinates = 0
    for bits, sm, n in itertools.product(
        (16, 32),
        (80, 90, 100),
        (
            64,
            96,
            128,
            256,
            3072,
            3073,
            4096,
            4097,
            6144,
            16384,
            16385,
            20000,
            32767,
            32768,
            32769,
            65536,
            131072,
            262144,
        ),
    ):
        cfg = config_type(SimpleNamespace(width=bits), n, True, sm)
        assert cfg.vec_size == 128 // bits
        assert cfg.N_per_cta == n // cfg.cluster_n
        assert cfg.cols_per_tile * cfg.rows_per_block == (
            cfg.num_threads * cfg.vec_size * cfg.num_vec_blocks
        )
        assert cfg.rows_per_block * cfg.threads_per_row == cfg.num_threads
        shape, stride = cfg._make_tv_layout(
            cfg.threads_per_row, cfg.rows_per_block, cfg.vec_size, cfg.num_vec_blocks
        )
        assert shape == (
            (cfg.threads_per_row, cfg.rows_per_block),
            (cfg.vec_size, cfg.num_vec_blocks),
        )
        if n in (64, 128, 4096, 32768) and sm == 100:
            owned = Counter()
            for tid, element, vector in itertools.product(
                range(cfg.num_threads), range(cfg.vec_size), range(cfg.num_vec_blocks)
            ):
                thread_col, row = tid % cfg.threads_per_row, tid // cfg.threads_per_row
                flat = thread_col * stride[0][0] + row * stride[0][1]
                flat += element * stride[1][0] + vector * stride[1][1]
                got_row, col = flat % cfg.rows_per_block, flat // cfg.rows_per_block
                assert got_row == row
                assert (
                    col
                    == thread_col * cfg.vec_size
                    + element
                    + vector * cfg.vec_size * cfg.threads_per_row
                )
                owned[(row, col)] += 1
            assert len(owned) == cfg.rows_per_block * cfg.cols_per_tile
            assert set(owned.values()) == {1}
            ownership_coordinates += len(owned)
            ownership_domains += 1
        config_cases += 1
    missing = config_type(SimpleNamespace(width=16), 4097, True, 100)
    assert missing.cluster_n * missing.cols_per_tile == 4096
    cluster_missing = config_type(SimpleNamespace(width=16), 32769, True, 100)
    assert cluster_missing.cluster_n * cluster_missing.cols_per_tile == 32768
    partial = config_type(SimpleNamespace(width=16), 32767, True, 100)
    last_vector_base = 32760
    assert last_vector_base < partial.N < last_vector_base + partial.vec_size
    padded = config_type(SimpleNamespace(width=16), 20000, True, 100)
    assert padded.cluster_n * padded.cols_per_tile == 20480
    assert 20480 - padded.N == 480

    reduction = trees["reduce.py"]
    functions = {x.name: x for x in reduction.body if isinstance(x, ast.FunctionDef)}
    cluster = functions["cluster_reduce"]
    exprs = {
        ast.unparse(x.targets[0]): x.value for x in ast.walk(cluster) if isinstance(x, ast.Assign)
    }

    def evaluate(expr, env):
        return eval(compile(ast.Expression(expr), "retained-cluster-expr", "eval"), env)

    incidence_domains = deliveries = final_results = 0
    for peers, warps, rows in itertools.product((1, 2, 4, 8, 16), (1, 2, 4, 8), (1, 2)):
        if peers * warps <= 1:
            continue
        received = {d: {} for d in range(peers)}
        count = Counter()
        for sender, warp, destination in itertools.product(
            range(peers), range(rows * warps), range(peers)
        ):
            row = evaluate(exprs["row_idx"], {"warp_idx": warp, "warps_per_row": warps})
            column = evaluate(exprs["col_idx"], {"warp_idx": warp, "warps_per_row": warps})
            slot = column + warps * sender
            label = 1 + sender * 10000 + row * 100 + column
            assert (row, slot) not in received[destination]
            received[destination][(row, slot)] = label
            count[destination] += 4
            deliveries += 1
        expected_bytes = evaluate(
            exprs["expected_bytes"], {"num_warps": rows * warps, "cluster_n": peers}
        )
        assert all(count[d] == expected_bytes for d in range(peers))
        for destination, row, op in itertools.product(
            range(peers), range(rows), (operator.add, max)
        ):
            identity = 0 if op is operator.add else -(10**9)
            lane_values = []
            for lane in range(32):
                value = identity
                for index in range(lane, warps * peers, 32):
                    value = op(value, received[destination][(row, index)])
                lane_values.append(value)
            result = xor_fold(lane_values, op)
            all_values = [received[destination][(row, index)] for index in range(warps * peers)]
            expected = sum(all_values) if op is operator.add else max(all_values)
            assert result == [expected] * 32
            final_results += 1
        incidence_domains += 1
    assert "phase=0" in ast.unparse(cluster)
    assert "peer_cta_rank_in_cluster=lane_idx" in ast.unparse(cluster)
    # Remote destination is addressed by receiver rank; slot identity uses the sender rank.
    assert "(row_idx, (col_idx, cta_rank_in_cluster))" in ast.unparse(cluster)
    store = ast.unparse(functions["store_shared_remote"])
    assert "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32" in store
    assert "set_block_rank(mbar_ptr, peer_cta_rank_in_cluster" in store
    assert "set_block_rank(smem_ptr, peer_cta_rank_in_cluster" in store
    # A current source fixed-phase call cannot safely represent a second generation.
    first_completed_phase = 1
    second_generation_pending_bytes = 32
    # Fixed phase0 would still report the previous completion during the next generation.
    assert first_completed_phase != 0 and second_generation_pending_bytes > 0
    warp_domains = 0
    for width in (8, 16, 32):
        labels = list(range(32))
        result = []
        for start in range(0, 32, width):
            result.extend(xor_fold(labels[start : start + width], operator.add))
        for lane in range(32):
            begin = lane // width * width
            assert result[lane] == sum(labels[begin : begin + width])
        warp_domains += 1
    source_kernel = next(
        x for x in rms.body if isinstance(x, ast.ClassDef) and x.name == "RMSNormKernel"
    )
    kernel = next(
        x for x in source_kernel.body if isinstance(x, ast.FunctionDef) and x.name == "kernel"
    )
    text = ast.unparse(kernel)
    assert "cute.math.rsqrt(mean_sq + eps, fastmath=True)" in text
    assert text.count("cute.autovec_copy(tXsX, tXrX)") == 2
    assert "cute.arch.cp_async_wait_group(0)" in text
    assert "if row_in_bounds:" in text
    assert "tXpX = predicate_k(tXcX, limit=cfg.N)" in text
    # Public source skips row loads but later reads sX unconditionally; no neutral-tail claim.
    assert "x_sq = x * x" in text and "mean_sq = sum_sq / cfg.N" in text
    reuse = [
        {
            "path": "seed/references/cutlass_ampere_tf32_gemm/templates/vectorized_access.yaml",
            "identity": "vectorized_access@1",
            "variables": {"vector_bytes": 16, "thread_arrangement": "strided_loop", "threads": 128},
            "p": {"contiguous_elems": 32768, "elem_bytes": 2},
            "case": {"base_alignment_bytes": 16},
        },
        {
            "path": "seed/references/flash_attention_2_fwd/templates/cross_lane_reduction.yaml",
            "identity": "cross_lane_reduction@1",
            "variables": {"lanes": 32, "result_to": "all_lanes", "combine_op": "add"},
            "p": {"assoc": True, "comm": True},
            "case": {},
        },
        {
            "path": (
                "seed/references/flash_attention_2_fwd/templates/approximate_math_rewrite.yaml"
            ),
            "identity": "approximate_math_rewrite@1",
            "variables": {"rewrite": "rsqrt_approx", "fused_fma": False},
            "p": {},
            "case": {"tolerance_allows_fast_math": True},
        },
    ]
    for instance in reuse:
        rule = Template.model_validate(
            yaml.safe_load((STUDY.parents[1] / instance["path"]).read_text())
        )
        assert f"{rule.name}@{rule.version}" == instance["identity"]
        for guard in [*rule.constraints, *rule.preconditions]:
            assert Expr.parse(guard.expr, instance["variables"]).eval(
                instance["variables"], p=instance["p"], case=instance["case"]
            )
        if rule.name == "vectorized_access":
            assert not Expr.parse(rule.constraints[0].expr, instance["variables"]).eval(
                instance["variables"], p={"contiguous_elems": 32767, "elem_bytes": 2}
            )
    (STUDY / "inputs/row-reduction-reuse.json").write_text(
        json.dumps(
            {
                "instances": reuse,
                "boundary": (
                    "Applicability guards only; native vector lowering, numeric reasso"
                    "ciation and rsqrt error bound remain conditional."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    chain = Chain.model_validate(
        yaml.safe_load((STUDY / "delta/row-reduction-chain.yaml").read_text())
    )
    resolved = {instance["identity"]: STUDY.parents[1] / instance["path"] for instance in reuse}
    resolved["operator_fusion@1"] = (
        STUDY.parents[1] / "seed/references/cutlass_ampere_tf32_gemm/templates/operator_fusion.yaml"
    )
    resolved["replicated_partial_reduce@1"] = (
        STUDY / "delta/replicated_partial_reduce@1/template.yaml"
    )
    previous = set()
    for item in chain.items:
        assert item.key not in previous and set(item.after) <= previous
        rule = Template.model_validate(yaml.safe_load(resolved[item.template].read_text()))
        assert set(item.bindings) == {v.name for v in rule.variables}
        for variable in rule.variables:
            value = item.bindings[variable.name]
            if isinstance(variable.domain, list):
                assert value in variable.domain
            else:
                assert variable.domain.lo <= value <= variable.domain.hi
        previous.add(item.key)
    result = {
        "outcome": "PASS",
        "partial_chain_steps": len(chain.items),
        "retained_configuration_cases": config_cases,
        "tv_ownership_domains": ownership_domains,
        "tv_owned_coordinates": ownership_coordinates,
        "replicated_incidence_domains": incidence_domains,
        "scalar_deliveries": deliveries,
        "replicated_reader_results": final_results,
        "subwarp_domains": warp_domains,
        "source_tail_witnesses": {
            "N4097_covered": 4096,
            "N32769_covered": 32768,
            "N32767_last_vector_end": 32768,
            "N20000_allocated_columns": 20480,
        },
        "boundary": (
            "Scalar source configuration, coordinate and reduction incidence m"
            "odels; native remote PTX, predicate zero-fill, vector tails, mask"
            "s and RMS numerics remain open."
        ),
    }
    (STUDY / "checks/row-reduction-result.json").write_text(json.dumps(result, indent=2) + "\n")
    witness = config_type(SimpleNamespace(width=16), 32768, True, 100)
    binding = {
        "dtype": "Float16",
        "N": 32768,
        "M": 2,
        "has_weight": True,
        "eps": 1e-6,
        "cluster_n": witness.cluster_n,
        "threads_per_row": witness.threads_per_row,
        "rows_per_block": witness.rows_per_block,
        "cols_per_tile": witness.cols_per_tile,
        "warps_per_row": witness.warps_per_row,
        "num_threads": witness.num_threads,
        "declared_smem_bytes": witness.smem_size_in_bytes(),
        "rule": "replicated_partial_reduce@1",
        "generation": "one fresh phase0",
        "boundary": (
            "Fully covered aligned vectors; native execution and numeric admission required."
        ),
    }
    (STUDY / "inputs/row-reduction-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    sys.stdout.write(
        "PASS: row-reduction configuration, TV ownership, replicated slots and tail witnesses.\n"
    )


if __name__ == "__main__":
    main()
