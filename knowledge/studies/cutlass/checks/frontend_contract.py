"Evaluate retained static predicates on symbolic dtype identities, without CUDA."

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
DTYPES = {
    "Float4E2M1FN": 4,
    "Float8E5M2": 8,
    "Float8E4M3FN": 8,
    "Float8E8M0FNU": 8,
    "Float32": 32,
    "Float16": 16,
    "BFloat16": 16,
}


class Symbols(ast.NodeTransformer):
    "Only replace enum identity and width queries in otherwise unchanged predicates."

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.returns = None
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            arg.annotation = None
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == "cutlass":
            assert node.attr in DTYPES
            return ast.copy_location(ast.Constant(node.attr), node)
        if node.attr == "width":
            return ast.copy_location(
                ast.Subscript(ast.Name("widths", ast.Load()), self.visit(node.value), ast.Load()),
                node,
            )
        return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        # Dtype symbols have unique names; source type identity becomes symbol equality.
        node.ops = [
            ast.Eq() if isinstance(x, ast.Is) else ast.NotEq() if isinstance(x, ast.IsNot) else x
            for x in node.ops
        ]
        return node


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    path = (
        "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/de"
        "nse_blockscaled_gemm_persistent_prefetch.py"
    )
    data = (Path(manifest["tree"]) / path).read_bytes()
    entry = next(x for x in manifest["files"] if x["path"] == path)
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    cls = next(
        x
        for x in ast.parse(data).body
        if isinstance(x, ast.ClassDef) and x.name == "Sm100BlockScaledPersistentDenseGemmKernel"
    )
    names = {
        "is_valid_dtypes_and_scale_factor_vec_size",
        "is_valid_layouts",
        "is_valid_mma_tiler_and_cluster_shape",
        "is_valid_tensor_alignment",
        "can_implement",
    }
    cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in names]
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
            Symbols().visit(cls),
        ],
        type_ignores=[],
    )
    ns: dict = {"widths": DTYPES}
    exec(compile(ast.fix_missing_locations(module), path, "exec"), ns)
    predicates = ns[cls.name]
    allowed = {
        ("Float4E2M1FN", "Float8E8M0FNU", 16),
        ("Float4E2M1FN", "Float8E8M0FNU", 32),
        ("Float4E2M1FN", "Float8E4M3FN", 16),
        ("Float8E5M2", "Float8E8M0FNU", 32),
        ("Float8E4M3FN", "Float8E8M0FNU", 32),
    }
    outputs = {"Float32", "Float16", "BFloat16", "Float8E5M2", "Float8E4M3FN"}
    count = 0
    for ab, sf, vector, output in itertools.product(DTYPES, DTYPES, (8, 16, 32, 64), DTYPES):
        actual = predicates.is_valid_dtypes_and_scale_factor_vec_size(ab, sf, vector, output)
        assert actual == ((ab, sf, vector) in allowed and output in outputs)
        count += 1
    clusters = 0
    for tm, tn, cm, cn in itertools.product(
        (64, 128, 256, 512), (32, 64, 128, 192, 256), range(9), range(9)
    ):
        expected = tm in (128, 256) and tn in (64, 128, 192, 256)
        expected &= cm in (1, 2, 4) and cn in (1, 2, 4) and cm * cn <= 16
        expected &= tm != 256 or cm % 2 == 0
        assert predicates.is_valid_mma_tiler_and_cluster_shape((tm, tn), (cm, cn)) == expected
        clusters += 1
    args = (
        "Float8E4M3FN",
        "Float8E8M0FNU",
        32,
        "Float32",
        (128, 192),
        (1, 1),
        144,
        208,
        320,
        2,
        "k",
        "k",
        "n",
    )
    assert predicates.can_implement(*args)
    misaligned = list(args)
    misaligned[8] = 321
    assert not predicates.can_implement(*misaligned)
    result = {
        "outcome": "PASS",
        "dtype_domains": count,
        "tile_cluster_domains": clusters,
        "frontend_witness": {
            "mnkl": [144, 208, 320, 2],
            "ab": "Float8E4M3FN",
            "sf": "Float8E8M0FNU",
            "sf_vector": 32,
            "c": "Float32",
            "mma_mn": [128, 192],
            "cluster_mn": [1, 1],
            "major": ["k", "k", "n"],
            "derived_tile_k": 128,
        },
        "boundary": "Static frontend predicates only, not compilation or device acceptance.",
    }
    (STUDY / "checks/frontend-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: static dtype/tile/cluster predicates and aligned tail witness.\n")


if __name__ == "__main__":
    main()
