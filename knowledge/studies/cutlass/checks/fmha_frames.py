"""Recover FMHA framed recurrence and source sink/pre-scale mismatch."""

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

import yaml
from moe_dense import ScalarSource

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Template
from kta.core.schemas.proof import ProofOutline

STUDY = Path(__file__).resolve().parents[1]


class MathSource(ScalarSource):
    def visit_Name(self, node):
        if node.id == "Float32":
            return ast.copy_location(ast.Name("float", node.ctx), node)
        return super().visit_Name(node)

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        if name in {"cute.math.exp2", "cute.math.log"}:
            node.func = ast.Attribute(ast.Name("math", ast.Load()), name.split(".")[-1], ast.Load())
            node.keywords = []
        return super().visit_Call(node)


def expression(node):
    tree = MathSource().visit(ast.Expression(copy.deepcopy(node)))
    return compile(ast.fix_missing_locations(tree), "retained-frame-expression", "eval")


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    path = "examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py"
    data = (Path(manifest["tree"]) / path).read_bytes()
    record = next(x for x in manifest["files"] if x["path"] == path)
    assert hashlib.sha256(data).hexdigest() == record["sha256"]
    cls = next(x for x in ast.parse(data).body if isinstance(x, ast.ClassDef))
    methods = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    setup = [
        x
        for x in methods["_setup_attributes"].body
        if isinstance(x, ast.Assign)
        and ast.unparse(x.targets[0]) in {"self.rescale_threshold", "self.p_fp8_prescale_log2"}
    ]
    setup_code = compile(ast.Module(copy.deepcopy(setup), []), "source-frame-policy", "exec")
    source_policies = []
    for enabled in (False, True):
        obj = SimpleNamespace(enable_skip_correction=enabled)
        exec(setup_code, {"self": obj, "math": math})
        assert (obj.rescale_threshold, obj.p_fp8_prescale_log2) == ((8, 0) if enabled else (0, 8))
        assert 2 ** (obj.rescale_threshold + obj.p_fp8_prescale_log2) <= 448
        source_policies.append([obj.rescale_threshold, obj.p_fp8_prescale_log2])

    directory = STUDY / "delta/online_rescaled_reduction@2"
    rule = Template.model_validate(yaml.safe_load((directory / "template.yaml").read_text()))
    proof = ProofOutline.model_validate(yaml.safe_load((directory / "proof.yaml").read_text()))
    assert proof.template == "online_rescaled_reduction@2"
    variables = {
        "state": "max_sum_acc",
        "rescale_point": "per_tile",
        "finalize": "deferred_divide",
        "traversal": "ascending",
        "inf_guard": True,
        "reference_policy": "lagged_max",
        "threshold_log2": 8,
        "weight_gain_log2": 0,
    }
    premises = {
        name: True
        for name in (
            "assoc",
            "rescalable",
            "complete_disjoint_tiles",
            "no_nonidentity_terms_dropped",
            "initial_state_valid",
            "masked_terms_are_zero",
            "final_denominator_positive_finite",
            "common_gain_constant",
            "extra_denominator_terms_share_frame",
            "numerator_denominator_reference_match",
            "state_transitions_match_recurrence",
            "casts_and_math_admitted",
            "finalize_log_compensation_exact",
            "reference_and_gain_within_numeric_range",
        )
    }
    for row in [*rule.constraints, *rule.preconditions]:
        assert Expr.parse(row.expr, variables.keys()).eval(variables, p=premises)
    scope = Expr.parse(rule.scope.predicate, variables.keys())
    for name in premises:
        assert not scope.eval(variables, p={**premises, name: False})
    old_path = STUDY.parents[1] / (
        "seed/references/flash_attention_2_fwd/templates/online_rescaled_reduction.yaml"
    )
    old = Template.model_validate(yaml.safe_load(old_path.read_text()))
    for variable in old.variables:
        new_variable = next(x for x in rule.variables if x.name == variable.name)
        assert variable.domain == new_variable.domain
    projection = {
        **variables,
        "reference_policy": "running_max",
        "threshold_log2": 0,
        "weight_gain_log2": 0,
    }
    for row in [*rule.constraints, *rule.preconditions]:
        assert Expr.parse(row.expr, projection.keys()).eval(projection, p=premises)
    old_variables = {x.name: projection[x.name] for x in old.variables}
    for row in old.preconditions:
        assert Expr.parse(row.expr, old_variables.keys()).eval(old_variables, p=premises)

    recurrence_cases = prefix_checks = 0
    for logits, values, cut, reverse, threshold, gamma, sink in itertools.product(
        ((-4, -1, 0, 3, 7), (0, 1, 2, 4, 8)),
        ((-2, 1, 5, -1, 4), (1, 1, 1, 1, 1)),
        (1, 2, 4),
        (False, True),
        (0, 8),
        (0, 8),
        (None, 0.5),
    ):
        paired = list(zip(logits, values, strict=True))
        blocks = [paired[:cut], paired[cut:]]
        if reverse:
            blocks.reverse()
        ref, denominator, numerator = max(x for x, _ in blocks[0]), 0.0, 0.0
        gain, seen = 2**gamma, []
        for block in blocks:
            candidate = max(ref, max(x for x, _ in block))
            new_ref = (
                ref if threshold and (candidate - ref) / math.log(2) <= threshold else candidate
            )
            correction = math.exp(ref - new_ref)
            denominator = correction * denominator + gain * sum(
                math.exp(x - new_ref) for x, _ in block
            )
            numerator = correction * numerator + gain * sum(
                math.exp(x - new_ref) * v for x, v in block
            )
            ref = new_ref
            seen.extend(block)
            assert math.isclose(
                denominator, gain * sum(math.exp(x - ref) for x, _ in seen), rel_tol=1e-12
            )
            assert math.isclose(
                numerator,
                gain * sum(math.exp(x - ref) * v for x, v in seen),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            prefix_checks += 1
        if sink is not None:
            denominator += gain * math.exp(sink - ref)
        total = sum(math.exp(x) for x in logits) + (math.exp(sink) if sink is not None else 0)
        target = sum(math.exp(x) * v for x, v in paired) / total
        assert math.isclose(numerator / denominator, target, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(
            ref + math.log(denominator) - math.log(gain), math.log(total), rel_tol=1e-12
        )
        recurrence_cases += 1

    # Reproduce the packed-sum correction algebra: each half starts with half
    # the old corrected denominator, then the two accumulated halves are added.
    step_assignments = [x for x in ast.walk(methods["softmax_step"]) if isinstance(x, ast.Assign)]
    half_expr = next(x.value for x in step_assignments if ast.unparse(x.targets[0]) == "acc_scale")
    packed_cases = 0
    for old_sum, correction, weights in itertools.product(
        (0, 1, 17), (0.125, 1), ((1, 2), (4, 8, 16, 32))
    ):
        half = old_sum * eval(
            expression(half_expr), {"math": math, "acc_scale_": math.log2(correction)}
        )
        result = half + sum(weights[::2]) + half + sum(weights[1::2])
        assert result == old_sum * correction + sum(weights)
        packed_cases += 1

    epilog = methods["correction_epilog"]
    assignments = [x for x in ast.walk(epilog) if isinstance(x, ast.Assign)]
    sink_expr = next(x.value for x in assignments if ast.unparse(x.targets[0]) == "sink_exp")
    sum_expr = next(
        x.value
        for x in assignments
        if ast.unparse(x.targets[0]) == "row_sum" and isinstance(x.value, ast.BinOp)
    )
    scale_expr = next(x.value for x in assignments if ast.unparse(x.targets[0]) == "scale")
    lse_rows = [x.value for x in assignments if ast.unparse(x.targets[0]) == "lse"]
    assert len(lse_rows) == 2
    sink_cases = []
    for gamma in (0, 8):
        gain = 2**gamma
        env = {
            "math": math,
            "sink_val": 0,
            "row_max_raw": 0,
            "scale_softmax": 1,
            "log2_e": 1.4426950408889634,
            "row_sum": gain,
            "scale_output": 1,
            "scaled_tmp": 0,
            "tTMEM_LOAD_VECrS": [gain, 0],
            "self": SimpleNamespace(p_fp8_prescale_log2=gamma),
        }
        env["sink_exp"] = eval(expression(sink_expr), env)
        env["row_sum"] = eval(expression(sum_expr), env)
        output = gain * eval(expression(scale_expr), env)
        env["lse"] = eval(expression(lse_rows[0]), env)
        if gamma:
            env["lse"] = eval(expression(lse_rows[1]), env)
        assert math.isclose(output, gain / (gain + 1))
        assert math.isclose(env["lse"], math.log2(gain + 1) - gamma)
        sink_cases.append(
            {
                "gain_log2": gamma,
                "source_output": output,
                "contract_output": 0.5,
                "source_lse_log2": env["lse"],
                "contract_lse_log2": 1,
            }
        )
    assert sink_cases[0]["source_output"] == 0.5 and sink_cases[1]["source_output"] != 0.5

    binding = {
        "identity": "online_rescaled_reduction@2",
        "variables": variables,
        "p": premises,
        "negative_scope_fields": list(premises),
        "invalid_variables": [
            {**variables, "reference_policy": "running_max", "threshold_log2": 8}
        ],
        "source_binding": (
            "FMHA positive scale/mass lag8/gain1 without pruning; backend recu"
            "rrence/casts conditional. FP8 gain256 sink violates frame."
        ),
    }
    (STUDY / "inputs/fmha-binding-cases.json").write_text(
        json.dumps({"cases": [binding]}, indent=2) + "\n"
    )
    result = {
        "outcome": "PASS",
        "source_policy_threshold_gain": source_policies,
        "recurrence_cases": recurrence_cases,
        "prefix_invariants": prefix_checks,
        "packed_denominator_cases": packed_cases,
        "sink_frame_cases": sink_cases,
        "guard_mutations": len(premises),
        "boundary": (
            "Retained scalar setup/epilogue expressions and real-arithmetic re"
            "currence; no native casts/TMEM execution, error certificate or ba"
            "ckward assurance."
        ),
    }
    (STUDY / "checks/fmha-frames-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: FMHA reference/gain frame, complete recurrence and sink compensation boundary.\n"
    )


if __name__ == "__main__":
    main()
