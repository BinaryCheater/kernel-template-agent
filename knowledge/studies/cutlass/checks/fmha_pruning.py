"""Check normalized-pruning certificates and source reference/skip protocols."""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import yaml
from moe_dense import ScalarSource

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Template
from kta.core.schemas.proof import ProofOutline

STUDY = Path(__file__).resolve().parents[1]


class ScalarIndex:
    """Interpret a retained one-row tensor expression with fixed scalar operands."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return self.value


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    path = "examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py"
    blob = (Path(manifest["tree"]) / path).read_bytes()
    entry = next(x for x in manifest["files"] if x["path"] == path)
    assert hashlib.sha256(blob).hexdigest() == entry["sha256"]
    source = ast.parse(blob)
    cls = next(x for x in source.body if isinstance(x, ast.ClassDef))
    methods = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    selected = [
        copy.deepcopy(x)
        for x in methods["calculate_skip_softmax_flag"].body
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "thread_wants_skip"
    ]
    skip_code = compile(ast.Module(selected, []), "source-row-skip", "exec")
    freeze = next(
        x
        for x in methods["softmax_step"].body
        if isinstance(x, ast.If) and "self.rescale_threshold > 0.0" in ast.unparse(x.test)
    )
    freeze_code = compile(
        ast.fix_missing_locations(ScalarSource().visit(ast.Module([copy.deepcopy(freeze)], []))),
        "source-reference-freeze",
        "exec",
    )
    host_compare = next(
        x
        for x in ast.walk(source)
        if isinstance(x, ast.Compare)
        and "skip_softmax_threshold_log2.value" in ast.unparse(x)
        and "block_cur_max" in ast.unparse(x)
    )
    compare_code = compile(ast.Expression(host_compare), "source-host-prune-test", "eval")
    device_ref = host_ref = -math.inf
    decisions = []
    for tile_max in (0, 4, -4):
        env = {
            "tile_row_max": tile_max,
            "row_max": device_ref,
            "scale_softmax_log2": 1,
            "skip_softmax_threshold_log2": -5,
            "logical_offset": (0, 0),
            "thread_idx": 0,
            "seqlen_q": 128,
        }
        exec(skip_code, env)
        device_skip = env["thread_wants_skip"]
        host_keep = eval(
            compare_code,
            {
                "block_max": ScalarIndex(tile_max),
                "block_cur_max": ScalarIndex(host_ref),
                "idx": (),
                "i": 0,
                "skip_softmax_threshold_log2": SimpleNamespace(value=-5),
            },
        )
        decisions.append(
            {
                "tile_max_log2": tile_max,
                "device_reference_before": device_ref,
                "host_reference_before": host_ref,
                "device_skip": device_skip,
                "host_skip": not host_keep,
            }
        )
        if not device_skip:
            fenv = {
                "self": SimpleNamespace(rescale_threshold=8),
                "row_max_safe": max(device_ref, tile_max),
                "old_row_max": device_ref,
                "scale_softmax_log2": 1,
            }
            exec(freeze_code, fenv)
            device_ref = fenv["row_max_safe"]
        if host_keep:
            host_ref = max(host_ref, tile_max)
    assert decisions[-1]["device_skip"] is False and decisions[-1]["host_skip"] is True

    rational_cases = frame_cases = tight_cases = 0
    undercount_witness = None
    for weights, values, keep_mask in itertools.product(
        itertools.product((0, 1, 2, 4), repeat=3),
        itertools.product((-3, 0, 2), repeat=3),
        range(1, 8),
    ):
        kept = [i for i in range(3) if keep_mask & (1 << i)]
        dropped = [i for i in range(3) if i not in kept]
        k, d = sum(weights[i] for i in kept), sum(weights[i] for i in dropped)
        if not k:
            continue
        yk = Fraction(sum(weights[i] * values[i] for i in kept), k)
        yf = Fraction(sum(w * v for w, v in zip(weights, values, strict=True)), k + d)
        diameter = max(values) - min(values)
        bound = Fraction(diameter * d, k + d)
        assert abs(yf - yk) <= bound
        if bound > 0 and abs(yf - yk) == bound:
            tight_cases += 1
            underestimated = Fraction(diameter) * Fraction(d, 2) / (k + Fraction(d, 2))
            assert abs(yf - yk) > underestimated
            if undercount_witness is None:
                undercount_witness = {
                    "weights": weights,
                    "values": values,
                    "kept_indices": kept,
                    "actual_error": str(abs(yf - yk)),
                    "false_bound_with_half_dropped_mass": str(underestimated),
                }
        # Conservative mass bounds are deliberately looser than actual masses.
        bloose = d + 1
        klower = Fraction(k, 2)
        conservative = diameter * bloose / (klower + bloose)
        assert bound <= conservative
        for exponent in (-8, 0, 8):
            factor = Fraction(2) ** exponent
            assert diameter * (bloose * factor) / ((klower + bloose) * factor) == conservative
            frame_cases += 1
        assert math.log1p(d / k) <= math.log1p(float(bloose / klower)) + 1e-15
        rational_cases += 1
    assert tight_cases > 0
    assert undercount_witness is not None

    directory = STUDY / "delta/bounded_normalized_weight_pruning@1"
    rule = Template.model_validate(yaml.safe_load((directory / "template.yaml").read_text()))
    outline = ProofOutline.model_validate(yaml.safe_load((directory / "proof.yaml").read_text()))
    assert outline.template == "bounded_normalized_weight_pruning@1"
    variables = {
        "ratio_threshold": 0.01,
        "output_error_budget": 112,
        "require_lse": True,
        "lse_error_budget": 0.125,
    }
    premises = {
        key: True
        for key in (
            "nonnegative_finite_weights",
            "kept_dropped_partition_exact",
            "certificate_covers_all_skips",
            "kept_mass_lower_valid",
            "dropped_mass_upper_valid",
            "coordinate_value_ranges_valid",
            "bounds_outward_rounded",
            "all_mass_bounds_share_frame",
            "extra_denominator_terms_accounted",
            "kept_output_is_normalized",
            "absolute_error_contract",
            "additional_error_bounds_valid",
            "lse_bound_valid",
            "collective_skip_uniform",
            "skipped_work_protocol_retired",
            "initial_kept_state_valid",
        )
    }
    flags = list(premises)
    premises.update(
        kept_mass_lower=1,
        dropped_mass_upper=0.125,
        value_diameter=1000,
        output_abs_scale=1,
        additional_output_error_upper=0,
        lse_error_upper=math.log1p(0.125),
    )
    for row in [*rule.constraints, *rule.preconditions]:
        assert Expr.parse(row.expr, variables.keys()).eval(variables, p=premises)
    constraints = [Expr.parse(row.expr, variables.keys()) for row in rule.constraints]
    assert not all(
        row.eval({**variables, "output_error_budget": 1}, p=premises) for row in constraints
    )
    assert not all(
        row.eval({**variables, "lse_error_budget": 0.01}, p=premises) for row in constraints
    )
    scope = Expr.parse(rule.scope.predicate, variables.keys())
    for flag in flags:
        assert not scope.eval(variables, p={**premises, flag: False})

    # Source control-path accounting: skipped P is not produced, but its flag is
    # published/fenced/committed; the MMA bypass still retires P and O handles.
    step = methods["softmax_step"]
    skip_branch = next(
        x
        for x in ast.walk(step)
        if isinstance(x, ast.If) and ast.unparse(x.test) == "not skip_softmax"
    )
    calls = [
        ast.unparse(x.func)
        for stmt in skip_branch.orelse
        for x in ast.walk(stmt)
        if isinstance(x, ast.Call)
    ]
    for required in (
        "inplace_consumer.wait_and_advance",
        "pi_mma_producer.acquire_and_advance",
        "cute.copy",
        "cute.arch.fence_view_async_tmem_store",
        "pi_handle.commit",
    ):
        assert required in calls
    pv = methods["mma_pv"]
    op_guard = next(
        x for x in ast.walk(pv) if isinstance(x, ast.If) and ast.unparse(x.test) == "not skip_pv"
    )
    assert any(
        isinstance(x, ast.Call) and ast.unparse(x.func) == "cute.gemm" for x in ast.walk(op_guard)
    )
    top_calls = [
        ast.unparse(x.value.func)
        for x in pv.body
        if isinstance(x, ast.Expr) and isinstance(x.value, ast.Call)
    ]
    assert top_calls[-2:] == ["pi_handle.release", "oi_handle.commit"]

    binding = {
        "identity": "bounded_normalized_weight_pruning@1",
        "variables": variables,
        "p": premises,
        "negative_scope_fields": flags,
        "invalid_variables": [
            {**variables, "output_error_budget": 1},
            {**variables, "lse_error_budget": 0.01},
        ],
        "source_binding": (
            "Conditional tau0.01/N128 weight2^-10 example with separate certif"
            "icate; source computes no bounds and native retirement remains un"
            "proved."
        ),
    }
    (STUDY / "inputs/fmha-pruning-binding-cases.json").write_text(
        json.dumps({"cases": [binding]}, indent=2) + "\n"
    )
    # Strict JSON: encode initial -infinite references as descriptive strings.
    for row in decisions:
        for key in ("device_reference_before", "host_reference_before"):
            if row[key] == -math.inf:
                row[key] = "-inf"
    result = {
        "outcome": "PASS",
        "rational_domains": rational_cases,
        "common_frame_checks": frame_cases,
        "tight_output_bounds": tight_cases,
        "undercounted_mass_witness": undercount_witness,
        "scope_mutations": len(flags),
        "frozen_device_vs_host_pruning": decisions,
        "boundary": (
            "Exact rational inequalities and retained scalar decisions/control"
            " paths; no native TMEM/collective progress, outward rounding or g"
            "radient assurance."
        ),
    }
    (STUDY / "checks/fmha-pruning-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: normalized-pruning error bounds and source reference/protocol boundaries.\n"
    )


if __name__ == "__main__":
    main()
