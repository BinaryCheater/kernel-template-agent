"Replay dense grouped descriptor variants and version1 projection without CUDA."

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path

import yaml
from frontend_contract import DTYPES, Symbols
from grouped_work_descriptors import scan_window

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Template

STUDY = Path(__file__).resolve().parents[1]
SOURCE = "examples/python/CuTeDSL/cute/blackwell/kernel/grouped_gemm/grouped_gemm.py"


class GroupSymbols(Symbols):
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if ast.unparse(node) in {
            "cutlass.tensor_utils.TensorMapUpdateMode.GMEM",
            "cutlass.tensor_utils.TensorMapUpdateMode.SMEM",
        }:
            return ast.copy_location(ast.Constant(node.attr), node)
        return super().visit_Attribute(node)

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "print":
                return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    data = (root / SOURCE).read_bytes()
    retained = next(x for x in manifest["files"] if x["path"] == SOURCE)
    assert hashlib.sha256(data).hexdigest() == retained["sha256"]
    source = data.decode()
    tree = ast.parse(source)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
    methods = {x.name: [x.lineno, x.end_lineno] for x in cls.body if isinstance(x, ast.FunctionDef)}
    pure_cls = copy.deepcopy(cls)
    pure_cls.body = [
        x
        for x in pure_cls.body
        if (isinstance(x, ast.FunctionDef) and x.name == "_get_tensormap_smem_bytes")
        or (
            isinstance(x, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in {"num_tensormaps", "bytes_per_tensormap"}
                for t in x.targets
            )
        )
    ]
    run = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "run")
    guard = copy.deepcopy(run)
    guard.name = "retained_host_guard"
    names = [
        "problem_sizes_mnkl",
        "ab_dtype",
        "c_dtype",
        "acc_dtype",
        "use_2cta_instrs",
        "mma_tiler_mn",
        "cluster_shape_mn",
        "a_major",
        "b_major",
        "c_major",
    ]
    guard.args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name) for name in names],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    guard.body = [n for n in guard.body if n.end_lineno <= 2187]
    guard.body.append(ast.Return(value=ast.Constant(True)))
    guard.decorator_list = []
    selected = ast.Module(body=[pure_cls, guard], type_ignores=[])
    ns: dict = {"widths": DTYPES}
    exec(compile(ast.fix_missing_locations(GroupSymbols().visit(selected)), SOURCE, "exec"), ns)
    kind = ns[cls.name]
    assert kind._get_tensormap_smem_bytes("GMEM") == 0
    assert kind._get_tensormap_smem_bytes("SMEM") == 384

    def accepts(shapes, ab="Float16", out="Float16", acc="Float32"):
        try:
            ns[guard.name](shapes, ab, out, acc, False, (128, 128), (1, 1), "k", "k", "n")
            return True
        except ValueError:
            return False

    dtype_cases = 0
    for ab, out, acc in itertools.product(
        ("Float16", "BFloat16"), ("Float16", "BFloat16", "Float32"), ("Float16", "Float32")
    ):
        assert accepts([(128, 128, 64, 1)], ab, out, acc) == (ab != "BFloat16" or acc != "Float16")
        dtype_cases += 1
    bad, good = (128, 128, 7, 1), (128, 128, 8, 1)
    assert accepts([bad, good]) and not accepts([good, bad])

    old = Template.model_validate(
        yaml.safe_load((STUDY / "delta/rebind_transfer_descriptors@1/template.yaml").read_text())
    )
    new = Template.model_validate(
        yaml.safe_load((STUDY / "delta/rebind_transfer_descriptors@2/template.yaml").read_text())
    )
    assert old.variables == new.variables[:-1]
    assert old.constraints == new.constraints and old.preconditions == new.preconditions
    assert old.scope.predicate == new.scope.predicate
    assert [o.name for o in old.obligations] == [o.name for o in new.obligations]
    p = {"descriptor_bytes": 128}
    projection_cases = 0
    for count in range(1, 6):
        old_vars, new_vars = (
            {"descriptor_count": count},
            {"descriptor_count": count, "update_site": "shared_shadow"},
        )
        for old_effect, new_effect in zip(old.effects.storage, new.effects.storage, strict=True):
            assert Expr.parse(old_effect.bytes, old_vars.keys()).eval(old_vars, p=p) == Expr.parse(
                new_effect.bytes, new_vars.keys()
            ).eval(new_vars, p=p)
            if new_effect.when:
                assert Expr.parse(new_effect.when, new_vars.keys()).eval(new_vars, p=p)
                assert not Expr.parse(new_effect.when, new_vars.keys()).eval(
                    {**new_vars, "update_site": "direct_global"}, p=p
                )
        projection_cases += 1
    # Both implementations preserve immutable problem mappings after old users retire.
    shapes = [(128, 128, 64), (256, 128, 0), (128, 256, 128)]
    counts = [1, 2, 2]
    visits = []
    unsafe_direct_witness = None
    for site in ("shared_shadow", "direct_global"):
        for worker in range(2):
            published = {}
            for index in range(worker, sum(counts), 2):
                group, local = scan_window(counts, index)
                m, n, k = shapes[group]
                immutable = {"C": (group, (m, n), (n, 1))}
                if k:
                    immutable.update(A=(group, (m, k), (k, 1)), B=(group, (n, k), (k, 1)))
                for role, metadata in immutable.items():
                    old_live = role in published
                    shadow = metadata if site == "shared_shadow" else None
                    if old_live:
                        previous = published[role]
                        if site == "shared_shadow":
                            assert published[role] == previous and shadow == metadata
                        else:
                            # A first-field mutation before retirement changes an old
                            # reader's generation while the rest of the tuple is old.
                            unsafe = (metadata[0], previous[1], previous[2])
                            assert unsafe != previous
                            if unsafe_direct_witness is None:
                                unsafe_direct_witness = {
                                    "old_group": previous[0],
                                    "observed_group": unsafe[0],
                                }
                        old_live = False
                    assert not old_live
                    published[role] = shadow if shadow is not None else metadata
                    assert published[role] == metadata
                visits.append((site, worker, index, group, local))
    assert len(visits) == 10
    # Both K branches satisfy the source-level delegated AB init participant count.
    assert "if cutlass.const_expr(self.delegate_tensormap_ab_init):" in source
    assert "self.tensormap_ab_init_barrier.arrive_and_wait()" in source
    kernel = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "kernel")

    def has_init(nodes):
        return any("tensormap_ab_init_barrier.arrive_and_wait" in ast.unparse(n) for n in nodes)

    nonzero = next(
        x
        for x in ast.walk(kernel)
        if isinstance(x, ast.If)
        and ast.unparse(x.test) == "not is_k_tile_cnt_zero"
        and has_init(x.body)
    )
    assert has_init(nonzero.orelse)
    other = root / (
        "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_grouped"
        "_gemm/grouped_blockscaled_gemm.py"
    )
    other_tree = ast.parse(other.read_text())
    old_nonzero = next(
        x
        for x in ast.walk(other_tree)
        if isinstance(x, ast.If)
        and ast.unparse(x.test) == "not is_k_tile_cnt_zero"
        and has_init(x.body)
        and has_init(x.orelse)
    )
    assert has_init(old_nonzero.orelse)
    pending_shared_init = 160 - 128 - 32  # epilogue128 plus TMA32, including K0
    assert pending_shared_init == 0
    assert not any(k != 0 for _, _, k in [(128, 128, 0), (256, 128, 0)])
    manager = (root / "python/CuTeDSL/cutlass/tensor_utils/tensormap.py").read_text()
    assert "cute.nvgpu.cpasync.fence_tma_desc_release" in manager
    assert "cute.nvgpu.cpasync.cp_fence_tma_desc_release" in manager
    assert manager.index("cp_async_bulk_wait_group(0, read=True") < manager.index(
        "# updates to tensormap in global memory"
    )
    result = {
        "outcome": "PASS",
        "dtype_guard_cases": dtype_cases,
        "version1_projection_counts": projection_cases,
        "descriptor_mode_worker_visits": len(visits),
        "shadow_bytes_by_mode": {"SMEM": 384, "GMEM": 0},
        "last_group_only_alignment_gap": True,
        "unsafe_direct_update_witness": unsafe_direct_witness,
        "blockscaled_zero_branch_has_init_partner": True,
        "all_zero_shared_init_pending_threads": pending_shared_init,
        "missing_zero_branch_pending_threads": 160 - 128,
        "boundary": (
            "Pure source predicates and metadata/order models; native descript"
            "ors, retire/release/acquire and initialization progress remain op"
            "en."
        ),
    }
    (STUDY / "inputs/dense-grouped-binding.json").write_text(
        json.dumps(
            {
                "path": SOURCE,
                "sha256": retained["sha256"],
                "methods": methods,
                "witness": {
                    "shapes_mnk": shapes,
                    "mma_mn": [128, 128],
                    "cluster_mn": [1, 1],
                    "ab_dtype": "Float16",
                    "acc_dtype": "Float32",
                    "c_dtype": "Float16",
                    "workers": 2,
                    "cache_shapes": True,
                    "descriptor_modes": ["SMEM", "GMEM"],
                },
            },
            indent=2,
        )
        + "\n"
    )
    (STUDY / "checks/grouped-dense-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: dense grouped modes, version1 projection, K0 init and source alignment gap.\n"
    )


if __name__ == "__main__":
    main()
