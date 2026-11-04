"""Replay retained MoE scalar scheduling and descriptor-coordinate contracts."""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import yaml

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Chain, Template

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/moe/"


class ScalarSource(ast.NodeTransformer):
    "Erase DSL annotations and lower only scalar casts, constexpr and tuple size."

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.decorator_list = [
            x for x in node.decorator_list if ast.unparse(x) in ("property", "staticmethod")
        ]
        node.returns = None
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            arg.annotation = None
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "Int32":
            return ast.copy_location(ast.Name("int", node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if ast.unparse(node) == "cutlass.Int32":
            return ast.copy_location(ast.Name("int", ast.Load()), node)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        name = ast.unparse(node.func)
        if name in ("const_expr", "cutlass.const_expr"):
            return self.visit(node.args[0])
        if name == "cute.size":
            node.func = ast.Name("tuple_size", ast.Load())
        node.keywords = [x for x in node.keywords if x.arg not in ("loc", "ip")]
        return self.generic_visit(node)


class EpilogueScalar(ScalarSource):
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if ast.unparse(node) == "self.scenario":
            return ast.copy_location(ast.Name("scenario", ast.Load()), node)
        return super().visit_Attribute(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if ast.unparse(node.func) == "cute.zeros_like":
            return ast.copy_location(ast.Constant(0), node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "load":
            return ast.copy_location(ast.Name("stored_acc", ast.Load()), node)
        return super().visit_Call(node)


def scalar_module(tree: ast.Module, selected: dict[str, set[str]]) -> ast.Module:
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in selected:
            node = copy.deepcopy(node)
            node.body = [
                x
                for x in node.body
                if isinstance(x, ast.FunctionDef) and x.name in selected[node.name]
            ]
            classes.append(node)
    return ast.fix_missing_locations(ScalarSource().visit(ast.Module(classes, [])))


def evaluate(node: ast.expr, env: dict) -> object:
    return eval(compile(ast.Expression(copy.deepcopy(node)), "source-expression", "eval"), env)


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    sources = {}
    for entry in json.loads((STUDY / "inputs/moe-variants.json").read_text())["entries"]:
        source = (root / entry["path"]).read_bytes()
        assert hashlib.sha256(source).hexdigest() == entry["sha256"]
        sources[Path(entry["path"]).name] = ast.parse(source.decode())
    scheduler = sources["moe_persistent_scheduler.py"]
    selected = {
        "MoEWorkTileInfo": {"__init__", "is_valid_tile"},
        "MoEStaticSchedulerParams": {
            "__init__",
            "cluster_tile_m",
            "cluster_tile_n",
            "cta_tile_k",
            "get_grid_shape",
        },
        "MoEStaticPersistentTileScheduler": {
            "__init__",
            "scenario",
            "expert_cnt",
            "intermediate",
            "hidden",
            "cta_tile_shape_mnk",
            "cluster_shape_mn",
            "cluster_tile_m",
            "cluster_tile_n",
            "cta_tile_k",
            "create",
            "initial_work_tile_info",
            "advance_to_next_work",
            "_get_work_tile_for_linear_idx",
            "_advance_expert_to_contain",
            "_compute_tiles_for_expert",
            "_decompose_local_idx",
            "_get_cluster_tile_counts",
            "_compute_k_tile_cnt",
        },
    }
    ns = {"tuple_size": math.prod}
    exec(compile(scalar_module(scheduler, selected), "retained-moe-scheduler", "exec"), ns)
    params_type = ns["MoEStaticSchedulerParams"]
    sched_type = ns["MoEStaticPersistentTileScheduler"]
    domains = queries = zero_k_outputs = 0
    short_orders = set()
    for scenario, lengths, dims, tile, cluster, workers in itertools.product(
        ("2Dx3D", "2Dx2D"),
        ((0, 8, 0, 40), (8, 0, 40, 16), (0, 0, 0, 0), (128,), (1, 17, 129)),
        ((64, 128), (256, 64)),
        ((64, 32, 16), (128, 64, 64)),
        ((1, 1), (2, 1), (1, 2)),
        (1, 2, 7),
    ):
        hidden, intermediate = dims
        ends = list(itertools.accumulate(lengths))
        params = params_type(scenario, (len(lengths), intermediate, hidden), tile, cluster)
        reference = []
        for expert, length in enumerate(lengths):
            m = length if scenario == "2Dx3D" else hidden
            mt = (m + tile[0] * cluster[0] - 1) // (tile[0] * cluster[0])
            nt = (intermediate + tile[1] * cluster[1] - 1) // (tile[1] * cluster[1])
            kt = ((hidden if scenario == "2Dx3D" else length) + tile[2] - 1) // tile[2]
            coords = list(itertools.product(range(mt), range(nt)))
            coords.sort(key=(lambda x: (x[1], x[0])) if mt <= nt else lambda x: x)
            if coords:
                short_orders.add("m" if mt <= nt else "n")
            reference.extend((expert, m_idx, n_idx, kt) for m_idx, n_idx in coords)
        seen = set()
        grid = params_type.get_grid_shape(params, workers)
        for peer_m, peer_n, worker in itertools.product(
            range(cluster[0]), range(cluster[1]), range(workers)
        ):
            current = sched_type.create(params, ends, (peer_m, peer_n, worker), grid)
            work = current.initial_work_tile_info()
            query = worker
            while work.is_valid_tile:
                expert, cm, cn, kt = reference[query]
                actual = (work.expert_idx, work.tile_m_idx, work.tile_n_idx, work.k_tile_cnt)
                expected = (expert, cm * cluster[0] + peer_m, cn * cluster[1] + peer_n, kt)
                assert actual == expected
                repeated = current._get_work_tile_for_linear_idx(query)
                assert vars(repeated) == vars(work)
                identity = (query, peer_m, peer_n)
                assert identity not in seen
                seen.add(identity)
                zero_k_outputs += int(kt == 0)
                work = current.advance_to_next_work()
                query += workers
                queries += 2
            assert query >= len(reference)
        assert len(seen) == len(reference) * math.prod(cluster)
        domains += 1
    assert short_orders == {"m", "n"} and zero_k_outputs > 0
    params = params_type("2Dx3D", (0, 64, 64), (64, 32, 16), (1, 1))
    empty_table = sched_type.create(params, [], (0, 0, 0), (1, 1, 1))
    try:
        empty_table.initial_work_tile_info()
    except IndexError:
        empty_table_rejected = True
    else:
        raise AssertionError("Expected retained source group0 read for an empty expert table")

    # Evaluate source coordinate tuples and pointer arithmetic, without tensor/driver substitutes.
    extension = next(
        x
        for x in sources["moe_sched_extension.py"].body
        if isinstance(x, ast.ClassDef) and x.name == "GroupedMmSchedExtension"
    )
    method = next(
        x for x in extension.body if isinstance(x, ast.FunctionDef) and x.name == "get_gmem_tensor"
    )
    scenario_if = next(x for x in method.body if isinstance(x, ast.If))
    branches = {"2Dx3D": scenario_if.body, "2Dx2D": scenario_if.orelse[0].body}
    offsets = {}
    for scenario, body in branches.items():
        branch = body[0]
        for tensor in ("a", "b", "c"):
            calls = [
                x
                for statement in branch.body
                for x in ast.walk(statement)
                if isinstance(x, ast.Call) and ast.unparse(x.func) == "cute.domain_offset"
            ]
            offsets[(scenario, tensor)] = (
                calls[0].args[0] if calls else ast.parse("(0,0,0)", mode="eval").body
            )
            branch = branch.orelse[0] if branch.orelse else None
    utilities = sources["moe_utils.py"]
    constructor = next(
        x
        for x in utilities.body
        if isinstance(x, ast.ClassDef) and x.name == "MoEGroupedGemmTensormapConstructor"
    )
    pointer_exprs = {}
    for method in constructor.body:
        if isinstance(method, ast.FunctionDef):
            for node in ast.walk(method):
                if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in (
                    "a_ptr_i",
                    "b_ptr_i",
                    "c_ptr_i",
                ):
                    pointer_exprs[ast.unparse(node.targets[0])[0]] = node.value
    coordinate_checks = 0
    for scenario, expert, token_offset in itertools.product(
        ("2Dx3D", "2Dx2D"), range(4), (0, 8, 48)
    ):
        for tensor in ("a", "b", "c"):
            encoded = tensor == "c" if scenario == "2Dx3D" else tensor in ("a", "b")
            coord_offset = evaluate(
                offsets[(scenario, tensor)], {"token_offset": token_offset, "expert_idx": expert}
            )
            stride = (257, 3, 100003)
            base = 17
            env = {f"{tensor}_ptr": base, f"{tensor}_stride": stride, "token_offset": token_offset}
            descriptor_base = evaluate(pointer_exprs[tensor], env) if encoded else base
            for m, k, batch in itertools.product(range(3), range(5), (0,)):
                local = (m, k, batch)
                got = descriptor_base + sum(
                    (c + d) * s for c, d, s in zip(local, coord_offset, stride, strict=True)
                )
                target_offset = (
                    (token_offset, 0, 0)
                    if scenario == "2Dx3D" and tensor != "b"
                    else (0, 0, expert)
                    if (scenario, tensor) in (("2Dx3D", "b"), ("2Dx2D", "c"))
                    else (0, token_offset, 0)
                )
                expected = base + sum(
                    (c + d) * s for c, d, s in zip(local, target_offset, stride, strict=True)
                )
                assert got == expected
                coordinate_checks += 1
    assert 17 + 8 * 3 + 8 * 3 != 17 + 8 * 3  # descriptor plus repeated offset is wrong
    table_addresses = 0
    for experts, width in itertools.product((1, 4, 33), (1, 2, 5)):
        locations = [128 * (e * width + slot) for e in range(experts) for slot in range(width)]
        assert len(set(locations)) == experts * width
        assert all(x % 128 == 0 and x + 128 <= experts * width * 128 for x in locations)
        table_addresses += len(locations)

    dense = sources["torch_grouped_mm.py"]
    cls = next(
        x for x in dense.body if isinstance(x, ast.ClassDef) and x.name == "GroupedGemmKernel"
    )
    kernel = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "kernel")
    sched_branch = next(
        x
        for x in kernel.body
        if isinstance(x, ast.If) and ast.unparse(x.test) == "warp_idx == self.sched_warp_id"
    )
    commits = [
        i
        for i, x in enumerate(sched_branch.body)
        if isinstance(x, ast.Expr)
        and isinstance(x.value, ast.Call)
        and ast.unparse(x.value.func) == "sched_pipeline.producer_commit"
    ]
    assert len(commits) == 2  # initial and terminal; middle commits are inside the loop
    tail = sched_branch.body[-1]
    assert ast.unparse(tail.value.func) == "sched_pipeline.producer_tail"
    assert not any(
        "sched_producer_state.advance" in ast.unparse(x)
        for x in sched_branch.body[commits[-1] + 1 :]
    )
    # Source publishes its initial result and an unconditional terminal sentinel.
    params = params_type("2Dx3D", (4, 64, 64), (64, 32, 16), (1, 1))
    idle = sched_type.create(params, [0, 0, 0, 0], (0, 0, 0), (1, 1, 1))
    initial_invalid = idle.initial_work_tile_info()
    assert not initial_invalid.is_valid_tile
    sentinel_expr = next(
        x.value
        for x in sched_branch.body
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "sentinel"
    )
    sentinel = evaluate(
        ast.fix_missing_locations(ScalarSource().visit(copy.deepcopy(sentinel_expr))), ns
    )
    sentinel_stream = [initial_invalid.expert_idx, sentinel.expert_idx]
    assert sentinel_stream.index(-1) + 1 < len(sentinel_stream)
    tail_counterexamples = 0
    for valid_work in range(1, 18):
        publications = valid_work + 1
        last = publications - 1
        phases = [0, 0]
        for q in range(publications):
            phases[q % 2] ^= 1  # all published items consumed and released
        wrong_phase = 1 ^ ((last // 2) % 2)
        assert phases[last % 2] == wrong_phase  # wait cannot pass after quiescence
        correct = publications
        for q in range(correct, correct + 2):
            assert phases[q % 2] != (1 ^ ((q // 2) % 2))
        tail_counterexamples += 1
    for helper in ("helpers.py", "sm90.py"):
        path = f"python/CuTeDSL/cutlass/pipeline/{helper}"
        pinned = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == pinned["sha256"]
    helpers = (root / "python/CuTeDSL/cutlass/pipeline/helpers.py").read_text()
    sm90 = (root / "python/CuTeDSL/cutlass/pipeline/sm90.py").read_text()
    assert (
        "Producers are assumed to start with an empty buffer and have a flipped phase bit of 1"
    ) in helpers
    assert "for i in range(self.num_stages):" in sm90
    assert "self.sync_object_empty.wait(state.index, state.phase" in sm90
    # Check empty backward epilogue selects zero rather than an old accumulator value.
    zero = next(
        x
        for x in ast.walk(kernel)
        if isinstance(x, ast.Assign)
        and ast.unparse(x.targets[0]) == "acc_vec"
        and isinstance(x.value, ast.Call)
        and ast.unparse(x.value.func) == "cute.zeros_like"
    )
    assert zero.lineno == 1312
    subtile = next(
        x
        for x in ast.walk(kernel)
        if isinstance(x, ast.For) and ast.unparse(x.target) == "subtile_idx"
    )
    zero_index = subtile.body.index(zero)
    block = copy.deepcopy(subtile.body[zero_index : zero_index + 2])
    module = ast.Module(block, [])
    code = compile(
        ast.fix_missing_locations(EpilogueScalar().visit(module)), "retained-empty-epilogue", "exec"
    )
    for scenario, k_tile_cnt in itertools.product(("2Dx3D", "2Dx2D"), (0, 1, 3)):
        env = {"scenario": scenario, "k_tile_cnt": k_tile_cnt, "stored_acc": 37}
        exec(code, env)
        expected = 0 if scenario == "2Dx2D" and k_tile_cnt == 0 else 37
        assert env["acc_vec"] == expected
    calls = [x for x in ast.walk(kernel) if isinstance(x, ast.Call)]
    assert not any(ast.unparse(x.func).endswith("construct_and_write") for x in calls)
    assert (
        "Change separate_tensormap_init to True"
        in (root / PREFIX / "torch_grouped_mm.py").read_text()
    )

    old_dir = STUDY / "delta/online_group_work_decode@1"
    new_dir = STUDY / "delta/online_group_work_decode@2"
    old = Template.model_validate(yaml.safe_load((old_dir / "template.yaml").read_text()))
    new = Template.model_validate(yaml.safe_load((new_dir / "template.yaml").read_text()))
    assert new.variables[1:] == old.variables
    assert new.constraints[:-1] == old.constraints and new.preconditions == old.preconditions
    assert new.scope.predicate == old.scope.predicate and new.effects == old.effects
    for cache in (False, True):
        assert Expr.parse(new.constraints[-1].expr, ["search", "cache_shapes"]).eval(
            {"search": "warp_prefix_32", "cache_shapes": cache}, p={}
        )
    chain = Chain.model_validate(yaml.safe_load((STUDY / "delta/moe-chain.yaml").read_text()))
    previous = set()
    for item in chain.items:
        assert item.key not in previous and set(item.after) <= previous
        record = Template.model_validate(
            yaml.safe_load((STUDY / "delta" / item.template / "template.yaml").read_text())
        )
        assert set(item.bindings) == {variable.name for variable in record.variables}
        for variable in record.variables:
            value = item.bindings[variable.name]
            if isinstance(variable.domain, list):
                assert value in variable.domain
            else:
                assert variable.domain.lo <= value <= variable.domain.hi
        previous.add(item.key)
    witness = {"expert_token_lengths": [8, 0, 40, 16], "expert_ends": [8, 8, 48, 64]}
    witness.update(hidden=128, intermediate=256, mma_mnk=[128, 128, 64], cluster_mnk=[1, 1, 1])
    for scenario in ("2Dx3D", "2Dx2D"):
        params = params_type(scenario, (4, 256, 128), (128, 128, 64), (1, 1))
        source = sched_type.create(params, witness["expert_ends"], (0, 0, 0), (1, 1, 3))
        witness[scenario] = {
            "output_counts": [source._compute_tiles_for_expert(e) for e in range(4)],
            "k_tile_counts": [source._compute_k_tile_cnt(e) for e in range(4)],
            "descriptor_tuple": ["c"] if scenario == "2Dx3D" else ["a", "b"],
        }
    witness["source_scope"] = {
        "expert_count_at_least_one": True,
        "positive_hidden_and_intermediate": True,
        "separate_tensormap_init": True,
        "accumulate_on_output": False,
        "bias": None,
        "dtype_a_b_c_acc": ["Float16", "Float16", "Float16", "Float32"],
        "native_transfer_alignment_layout_visibility": "Required, not scalar-proved",
        "source_scheduler_exit_protocol": "Gap: terminal state and duplicate idle sentinels",
    }
    (STUDY / "inputs/moe-dense-binding.json").write_text(json.dumps(witness, indent=2) + "\n")
    result = {
        "outcome": "PASS",
        "partial_chain_steps": len(chain.items),
        "source_scheduler_domains": domains,
        "source_scheduler_queries": queries,
        "backward_zero_k_cta_outputs": zero_k_outputs,
        "short_axis_orders": sorted(short_orders),
        "empty_expert_table_source_read": empty_table_rejected,
        "source_coordinate_checks": coordinate_checks,
        "table_slot_addresses": table_addresses,
        "quiescent_tail_counterexamples": tail_counterexamples,
        "initially_invalid_worker_publications": sentinel_stream,
        "version1_projection": (
            "Old variables/constraints/premises/effects preserved; warp_prefix_32 branch."
        ),
        "boundary": (
            "Retained scalar source and protocol/coordinate models only; nativ"
            "e descriptor proxy visibility, zero extents, clipping and pipelin"
            "e progress remain open."
        ),
    }
    (STUDY / "checks/moe-dense-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: MoE scalar scheduler, coordinate ownership and source protocol counterexamples.\n"
    )


if __name__ == "__main__":
    main()
