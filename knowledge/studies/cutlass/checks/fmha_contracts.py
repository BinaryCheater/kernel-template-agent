"Replay retained FMHA forward gates and distinguish reference freezing from pruning."

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

from dense_contracts import DenseSymbols
from frontend_contract import DTYPES
from moe_dense import ScalarSource

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/"


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    path = PREFIX + "fmha.py"
    data = (Path(manifest["tree"]) / path).read_bytes()
    retained = next(x for x in manifest["files"] if x["path"] == path)
    assert hashlib.sha256(data).hexdigest() == retained["sha256"]
    cls = next(x for x in ast.parse(data).body if isinstance(x, ast.ClassDef))
    methods = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    guards = copy.deepcopy(cls)
    guards.body = [
        copy.deepcopy(methods[name])
        for name in ("check_supported_dtypes", "check_invalid_shape", "can_implement")
    ]
    module = DenseSymbols().visit(ast.Module([guards], []))
    ns = {"widths": DTYPES}
    exec(compile(ast.fix_missing_locations(module), "retained-fmha-gates", "exec"), ns)
    kind = ns[cls.name]
    obj = kind.__new__(kind)
    dtype_cases = accepted = 0
    supported = {"Float16", "BFloat16", "Float8E4M3FN"}
    for qk, pv, out, qacc, pacc in itertools.product(DTYPES, repeat=5):
        actual = obj.can_implement((1, 64, 8, 64), (1, 128, 2, 64), qk, pv, out, qacc, pacc)
        assert actual == (
            qk in supported and pv in supported and out in supported and qacc == pacc == "Float32"
        )
        dtype_cases += 1
        accepted += actual
    shape_cases = 0
    for batch, other_batch, hq, hk, dim, other_dim, dtype in itertools.product(
        (1, 2),
        (1, 2),
        (1, 2, 4, 8),
        (1, 2, 4),
        (32, 64, 96, 128, 192),
        (64, 128, 192),
        ("Float16", "Float8E4M3FN"),
    ):
        actual = obj.can_implement(
            (batch, 17, hq, dim),
            (other_batch, 33, hk, other_dim),
            dtype,
            "Float16",
            "Float16",
            "Float32",
            "Float32",
        )
        expected = batch == other_batch and dim == other_dim and dim in {32, 64, 128, 192}
        expected &= hq % hk == 0 and (dim != 192 or dtype == "Float8E4M3FN")
        assert actual == expected
        shape_cases += 1
    # Sequence positivity and per-item variable lengths are not checked here.
    assert obj.can_implement(
        (1, 0, 8, 64), (1, 0, 2, 64), "Float16", "Float16", "Float16", "Float32", "Float32"
    )
    assert obj.can_implement(
        (2, (3, -1), 8, 64),
        (2, (5, 0), 2, 64),
        "Float16",
        "Float16",
        "Float16",
        "Float32",
        "Float32",
    )
    try:
        obj.can_implement(
            (1, 64, 8, 64), (1, 64, 0, 64), "Float16", "Float16", "Float16", "Float32", "Float32"
        )
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("source h_k0 division did not occur")

    skip = methods["calculate_skip_softmax_flag"]
    scalar_prefix = [
        x
        for x in skip.body
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "thread_wants_skip"
    ]
    skip_code = compile(ast.Module(copy.deepcopy(scalar_prefix), []), "source-row-skip", "exec")
    vote_cases = 0
    for flags in itertools.product((False, True), repeat=4):
        packed_bytes = sum(int(flag) << (8 * i) for i, flag in enumerate(flags))
        assert (packed_bytes.bit_count() == 4) == all(flags)
        vote_cases += 1
    row_skip_cases = 0
    for delta, threshold, lane, length in itertools.product(
        (-16, -10, -8, -1, 0, 1), (-10, -8, 0), (0, 31, 32, 127), (1, 32, 128)
    ):
        env = {
            "tile_row_max": delta,
            "row_max": 0,
            "scale_softmax_log2": 1,
            "skip_softmax_threshold_log2": threshold,
            "logical_offset": (0, 0),
            "thread_idx": lane,
            "seqlen_q": length,
        }
        exec(skip_code, env)
        assert env["thread_wants_skip"] == (delta < threshold or lane >= length)
        row_skip_cases += 1

    softmax = methods["softmax_step"]
    freeze = next(
        x
        for x in softmax.body
        if isinstance(x, ast.If) and "self.rescale_threshold > 0.0" in ast.unparse(x.test)
    )
    transformed = ScalarSource().visit(ast.Module([copy.deepcopy(freeze)], []))
    freeze_code = compile(ast.fix_missing_locations(transformed), "source-reference-freeze", "exec")
    freeze_cases = 0
    for old, delta, threshold in itertools.product((-4, 0, 8), (0, 1, 8, 9), (0, 8)):
        env = {
            "self": SimpleNamespace(rescale_threshold=threshold),
            "row_max_safe": old + delta,
            "old_row_max": old,
            "scale_softmax_log2": 1,
        }
        exec(freeze_code, env)
        assert env["row_max_safe"] == (old if threshold > 0 and delta <= threshold else old + delta)
        freeze_cases += 1
    # A frozen finite reference can preserve the normalized value while violating
    # the old rule's exact-running-maximum invariant.
    frozen_weights = [1, 2**4]
    maximum_weights = [2**-4, 1]
    assert frozen_weights[1] / sum(frozen_weights) == maximum_weights[1] / sum(maximum_weights)

    # Low maximum per element does not bound total dropped mass or output error
    # independently of tile cardinality and the value-vector range.
    dropped = []
    for count, value in itertools.product((1, 8, 128), (1, 1000)):
        env = {
            "tile_row_max": -10,
            "row_max": 0,
            "scale_softmax_log2": 1,
            "skip_softmax_threshold_log2": math.log2(0.01),
            "logical_offset": (0, 0),
            "thread_idx": 0,
            "seqlen_q": 128,
        }
        exec(skip_code, env)
        assert env["thread_wants_skip"]
        mass = count * 2**-10
        exact = mass * value / (1 + mass)
        assert exact > 0
        dropped.append(
            {
                "elements": count,
                "value": value,
                "dropped_mass": mass,
                "full_output": exact,
                "pruned_output": 0,
            }
        )

    result = {
        "outcome": "PASS",
        "dtype_cases": dtype_cases,
        "dtype_accepted": accepted,
        "shape_cases": shape_cases,
        "row_skip_cases": row_skip_cases,
        "four_warp_votes": vote_cases,
        "finite_reference_freeze_cases": freeze_cases,
        "pruning_witnesses": dropped,
        "boundary": (
            "Retained host predicates and scalar decision expressions; finite "
            "reference algebra and pruning counterexamples. Full recurrence, T"
            "MEM/packed arithmetic, masking, cast precision, votes/visibility "
            "and backward remain active."
        ),
    }
    entries = []
    for filename in ("fmha.py", "fmha_bwd.py"):
        entry_path = PREFIX + filename
        blob = (Path(manifest["tree"]) / entry_path).read_bytes()
        record = next(x for x in manifest["files"] if x["path"] == entry_path)
        assert hashlib.sha256(blob).hexdigest() == record["sha256"]
        source = ast.parse(blob)
        kind = next(x for x in source.body if isinstance(x, ast.ClassDef))
        entries.append(
            {
                "path": entry_path,
                "sha256": record["sha256"],
                "methods": {
                    x.name: [x.lineno, x.end_lineno]
                    for x in kind.body
                    if isinstance(x, ast.FunctionDef)
                },
                "functions": {
                    x.name: [x.lineno, x.end_lineno]
                    for x in source.body
                    if isinstance(x, ast.FunctionDef)
                },
                "status": (
                    "Forward frame revision proposed; masks/protocol/backward remain active"
                    if filename == "fmha.py"
                    else "Inventoried; backward mechanism trace pending"
                ),
            }
        )
    (STUDY / "inputs/fmha-variants.json").write_text(
        json.dumps({"entries": entries}, indent=2) + "\n"
    )
    (STUDY / "checks/fmha-contracts-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: FMHA entry gates, reference-freeze decisions and nonidentity pruning witnesses.\n"
    )


if __name__ == "__main__":
    main()
