"Recover the SM120 dense entry's actual CLI, dtype gates and resource arithmetic."

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from dense_contracts import DenseSymbols
from frontend_contract import DTYPES
from moe_dense import ScalarSource

STUDY = Path(__file__).resolve().parents[1]
PATH = "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py"


class DtypeNames(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id in DTYPES:
            return ast.copy_location(ast.Constant(node.id), node)
        if node.id == "DSLUserCodeError":
            return ast.copy_location(ast.Name("ValueError", ast.Load()), node)
        return node


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    backend = "python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py"
    trees = {}
    for path in (PATH, backend):
        data = (Path(manifest["tree"]) / path).read_bytes()
        retained = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(data).hexdigest() == retained["sha256"]
        trees[path] = ast.parse(data)
    tree = trees[PATH]
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
    methods = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    funcs = {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
    # Preserve the parser with only dtype objects represented by stable strings.
    parser_functions = [
        copy.deepcopy(funcs[name]) for name in ("parse_comma_separated_ints", "parse_arguments")
    ]
    for fn in parser_functions:
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
    parser_ns = {
        "argparse": argparse,
        "cutlass": SimpleNamespace(dtype=str, Float16="Float16", Float32="Float32"),
    }
    exec(compile(ast.Module(parser_functions, []), "retained-parser", "exec"), parser_ns)
    menu_cases = accepted = 0
    tile_choices = []
    parser_ast = funcs["parse_arguments"]
    for node in ast.walk(parser_ast):
        if (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--tile_shape_mnk"
        ):
            tile_choices = ast.literal_eval(
                next(x.value for x in node.keywords if x.arg == "choices")
            )
    original_argv = sys.argv
    try:
        for tile, stages, dims in itertools.product(
            [*tile_choices, (32, 32, 32)], (-1, 0, 1, 4, 16), ("1,1,1,1", "1,1,1")
        ):
            sys.argv = [
                "retained-parser",
                "--tile_shape_mnk",
                ",".join(map(str, tile)),
                "--epi_stage",
                str(stages),
                "--mnkl",
                dims,
            ]
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    parser_ns["parse_arguments"]()
                    actual = True
                except SystemExit as error:
                    assert error.code == 2
                    actual = False
            assert actual == (tile in tile_choices and stages > 0 and dims.count(",") == 3)
            accepted += actual
            menu_cases += 1
    finally:
        sys.argv = original_argv

    source_guards = [
        copy.deepcopy(x)
        for x in methods["__call__"].body
        if isinstance(x, ast.If) and any(isinstance(y, ast.Raise) for y in ast.walk(x))
    ]
    module = ast.Module(source_guards, [])
    guard_code = compile(
        ast.fix_missing_locations(DenseSymbols().visit(ScalarSource().visit(module))),
        "retained-width-guards",
        "exec",
    )
    op = next(
        x for x in trees[backend].body if isinstance(x, ast.ClassDef) and x.name == "MmaF16BF16Op"
    )
    validate = next(
        x for x in op.body if isinstance(x, ast.FunctionDef) and x.name == "__post_init__"
    )
    validator = ast.Module([copy.deepcopy(validate)], [])
    ns = {}
    exec(
        compile(
            ast.fix_missing_locations(DtypeNames().visit(validator)), "retained-mma-op", "exec"
        ),
        ns,
    )
    dtype_cases = front_accepted = op_accepted = 0
    rejected_after_frontend = []
    for a, b, acc in itertools.product(DTYPES, repeat=3):
        front = True
        try:
            exec(guard_code, {"self": SimpleNamespace(a_dtype=a, b_dtype=b), "widths": DTYPES})
        except TypeError:
            front = False
        expected = DTYPES[a] == DTYPES[b] and DTYPES[a] in (8, 16) and (DTYPES[a] != 16 or a == b)
        assert front == expected
        native_op = False
        if front:
            try:
                ns["__post_init__"](
                    SimpleNamespace(ab_dtype=a, acc_dtype=acc, shape_mnk=(16, 8, 16))
                )
                native_op = True
            except ValueError:
                rejected_after_frontend.append([a, b, acc])
        assert native_op == (
            (a == b == "Float16" and acc in {"Float16", "Float32"})
            or (a == b == "BFloat16" and acc == "Float32")
        )
        dtype_cases += 1
        front_accepted += front
        op_accepted += native_op

    stage_fn = copy.deepcopy(methods["_compute_stages"])
    stage_fn.decorator_list = []
    ns = {"widths": DTYPES}
    exec(
        compile(
            ast.fix_missing_locations(DenseSymbols().visit(ast.Module([stage_fn], []))),
            "retained-stage-policy",
            "exec",
        ),
        ns,
    )
    stage_cases = negative = zero = 0
    for tile, epi, out, capacity in itertools.product(
        tile_choices, (1, 4, 16), ("Float16", "Float32"), (16384, 65536, 100000)
    ):
        em, en = min(tile[0], 64), min(tile[1], 64)
        stages, returned_epi = ns["_compute_stages"](
            tile, "Float16", "Float16", (em, en), out, capacity, 1, epi
        )
        stage_bytes = (tile[0] + tile[1]) * tile[2] * 2
        expected = (capacity - 2048 - em * en * (DTYPES[out] // 8) * epi) // stage_bytes
        assert stages == expected and returned_epi == epi
        negative += stages < 0
        zero += stages == 0
        stage_cases += 1
    assert negative and zero
    # Setup exits only on exactly zero; negative sizes remain outside this guard.
    equality = next(x.test for x in methods["_setup_attributes"].body if isinstance(x, ast.If))
    assert ast.unparse(equality) == "self.ab_stage == 0"

    # Follow the retained monotonic output slot expression across work tiles;
    # modeled source-read completion is separated from native TMA semantics.
    kernel = methods["kernel"]
    increment = next(
        x.value
        for x in ast.walk(kernel)
        if isinstance(x, ast.Assign)
        and ast.unparse(x.targets[0]) == "epi_buffer"
        and isinstance(x.value, ast.BinOp)
    )
    increment_code = compile(ast.Expression(increment), "source-output-slot", "eval")
    output_domains = 0
    for stages, subtiles, items in itertools.product(range(1, 9), range(1, 9), range(1, 8)):
        slot = issued = 0
        outstanding = []
        for _item in range(items):
            for _subtile in range(subtiles):
                assert slot == issued % stages and slot not in outstanding
                outstanding.append(slot)
                if len(outstanding) > stages - 1:
                    outstanding.pop(0)
                slot = eval(
                    increment_code, {"epi_buffer": slot, "self": SimpleNamespace(epi_stage=stages)}
                )
                issued += 1
            outstanding.clear()  # per-tile source tail followed by CTA math barrier
        output_domains += 1

    entry = next(x for x in manifest["files"] if x["path"] == PATH)
    inventory = {
        "entries": [
            {
                "path": PATH,
                "sha256": entry["sha256"],
                "methods": {x: [y.lineno, y.end_lineno] for x, y in methods.items()},
                "functions": {x: [y.lineno, y.end_lineno] for x, y in funcs.items()},
                "status": ("Source-extracted; native layout/resource/numeric obligations retained"),
            }
        ]
    }
    (STUDY / "inputs/geforce-dense-variants.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    result = {
        "outcome": "PASS",
        "cli_cases": menu_cases,
        "cli_accepted": accepted,
        "dtype_cases": dtype_cases,
        "width_guard_accepted": front_accepted,
        "selected_op_accepted": op_accepted,
        "front_to_op_rejections": len(rejected_after_frontend),
        "tile_menu": tile_choices,
        "resource_cases": stage_cases,
        "negative_stage_results": negative,
        "zero_stage_results": zero,
        "output_ring_domains": output_domains,
        "boundary": (
            "Retained parser, predicates, MMA constructor validation, stage/sl"
            "ot arithmetic and abstract output retirement; no native compile, "
            "layout or GPU execution."
        ),
    }
    (STUDY / "checks/geforce-dense-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: GeForce dense CLI, actual MMA gate, stage boundaries and output slots.\n"
    )


if __name__ == "__main__":
    main()
