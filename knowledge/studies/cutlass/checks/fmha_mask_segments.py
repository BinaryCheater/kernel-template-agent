"""Recover three-segment mask choices and reject unnormalized source metadata."""

from __future__ import annotations

import ast
import copy
import enum
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from fmha_masks import MaskSource
from moe_dense import ScalarSource

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Template
from kta.core.schemas.proof import ProofOutline

STUDY = Path(__file__).resolve().parents[1]


def plan(info, masking, traversal):
    start, end, leading, core, trailing = info
    assert min(start, end, leading, core, trailing) >= 0
    assert end == start + leading + core + trailing
    result = [
        (i, masking == "mask_all_kept" or i < start + leading or i >= end - trailing)
        for i in range(start, end)
    ]
    return result if traversal == "ascending" else list(reversed(result))


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    paths = [
        "examples/python/CuTeDSL/helpers/fmha_helpers.py",
        "examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py",
    ]
    trees = {}
    for path in paths:
        blob = (root / path).read_bytes()
        record = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(blob).hexdigest() == record["sha256"]
        trees[Path(path).name] = ast.parse(blob)
    helpers = trees["fmha_helpers.py"]
    selected = [
        copy.deepcopy(x)
        for x in helpers.body
        if isinstance(x, ast.ClassDef) and x.name in {"MaskEnum", "FusedMask"}
    ]
    ns = {
        "enum": enum,
        "Float32": SimpleNamespace(inf=math.inf),
        "tuple_size": len,
        "cute": SimpleNamespace(
            ceil_div=lambda a, b: -(-a // b), arch=SimpleNamespace(thread_idx=lambda: (0, 0, 0))
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(MaskSource().visit(ast.Module(selected, []))),
            "retained-mask-segments",
            "exec",
        ),
        ns,
    )
    mask, kinds = ns["FusedMask"], ns["MaskEnum"]
    metadata_domains = normalized = unnormalized = plans = 0
    for sq, sk, bm, bn, left, right in itertools.product(
        (1, 3, 5, 9), (1, 3, 5, 9), (2, 4), (2, 4), (None, 0, 2), (None, 0, 2)
    ):
        for qb in range(math.ceil(sq / bm)):
            info = mask.get_masked_info(
                kinds.WINDOW_MASK_INFERENCE, (qb, 0, (0, 0)), (bm, bn, 64), sq, sk, left, right
            )
            total = math.ceil(sk / bn)
            start, end = info[:2]
            valid = min(info) >= 0 and start <= end <= total
            metadata_domains += 1
            if not valid:
                unnormalized += 1
                continue
            normalized += 1
            for masking, traversal in itertools.product(
                ("mask_all_kept", "peel_boundaries"), ("ascending", "descending")
            ):
                emitted = plan(info, masking, traversal)
                ids = [i for i, _ in emitted]
                assert set(ids) == set(range(start, end)) and len(ids) == len(set(ids))
                assert ids == sorted(ids, reverse=traversal == "descending")
                plans += 1
    assert unnormalized > 0

    witness = mask.get_masked_info(
        kinds.WINDOW_MASK, (1, 0, (0, 0)), (256, 128, 64), 768, 1024, 256, 256
    )
    assert witness == (0, 6, 2, 2, 2)
    forward = next(x for x in trees["fmha.py"].body if isinstance(x, ast.ClassDef))
    softmax = next(
        x for x in forward.body if isinstance(x, ast.FunctionDef) and x.name == "softmax"
    )
    # Execute the actual three loop-argument expressions and literal mask flags.
    loops = []
    for x in ast.walk(softmax):
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "loop_args":
            loops.append(x.value)
    assert len(loops) == 3
    env = dict(
        zip(("st_cnt", "end_cnt", "ld_mask_cnt", "unmask_cnt", "tl_mask_cnt"), witness, strict=True)
    )
    source_plan = []
    for node, apply in zip(loops, (True, False, True), strict=True):
        start, count, end = eval(compile(ast.Expression(node), "source-segment-loop", "eval"), env)
        assert end == witness[1]
        source_plan.extend((i, apply) for i in range(start, start + count))
    assert source_plan == plan(witness, "peel_boundaries", "ascending")
    coordinate_cases = 0
    for masking, traversal in itertools.product(
        ("mask_all_kept", "peel_boundaries"), ("ascending", "descending")
    ):
        emitted = plan(witness, masking, traversal)
        for q in range(256, 512):
            expected = {k for k in range(1024) if q - 256 <= k <= q + 256}
            kept = [
                k
                for tile, apply in emitted
                for k in range(tile * 128, (tile + 1) * 128)
                if not apply or q - 256 <= k <= q + 256
            ]
            assert len(kept) == len(set(kept)) and set(kept) == expected
            coordinate_cases += 1024
    # A one-edge-first schedule cannot reproduce both masked edges while also
    # preserving the source's unmasked middle. Masking all kept tiles is valid.
    assert source_plan[4][1] is True and source_plan[2][1] is False

    negative_actual = mask.get_masked_info(
        kinds.WINDOW_MASK_INFERENCE, (0, 0, (0, 0)), (256, 128, 64), 513, 1, None, 0
    )
    assert negative_actual == (0, -2, 0, -2, 0)
    guard = next(
        x
        for x in ast.walk(trees["fmha.py"])
        if isinstance(x, ast.FunctionDef) and x.name == "check_seqlen_valid"
    )
    guard_ns = {}
    exec(
        compile(
            ast.fix_missing_locations(ScalarSource().visit(ast.Module([copy.deepcopy(guard)], []))),
            "source-window-check",
            "exec",
        ),
        guard_ns,
    )
    assert guard_ns["check_seqlen_valid"](513, 1, None, 0, True)

    directory = STUDY / "delta/masked_tile_pruning@2"
    rule = Template.model_validate(yaml.safe_load((directory / "template.yaml").read_text()))
    proof = ProofOutline.model_validate(yaml.safe_load((directory / "proof.yaml").read_text()))
    assert proof.template == "masked_tile_pruning@2"
    variables = {"masking": "peel_boundaries", "traversal": "ascending"}
    premises = {
        x: True
        for x in (
            "tile_partition_exact",
            "counts_integral_and_representable",
            "mask_metadata_stable",
            "mask_value_is_identity",
            "outside_interval_all_identity",
            "element_mask_exact",
            "padded_output_rows_unobserved",
            "core_all_visible_elements_valid",
            "traversal_numerically_admitted",
            "empty_interval_policy_admitted",
            "loop_and_producer_ranges_match",
        )
    }
    flags = list(premises)
    premises.update(
        total_tiles=8,
        start=0,
        end=6,
        leading=2,
        core=2,
        trailing=2,
        empty_interval_result_defined=False,
    )
    for row in [*rule.constraints, *rule.preconditions]:
        assert Expr.parse(row.expr, variables.keys()).eval(variables, p=premises)
    scope = Expr.parse(rule.scope.predicate, variables.keys())
    for flag in flags:
        assert not scope.eval(variables, p={**premises, flag: False})
    assert scope.eval(
        {**variables, "masking": "mask_all_kept"},
        p={**premises, "core_all_visible_elements_valid": False},
    )
    for invalid in (
        {"end": -2, "leading": 0, "core": -2, "trailing": 0},
        {"end": 0, "leading": 0, "core": 0, "trailing": 0},
    ):
        assert not all(
            Expr.parse(row.expr, variables.keys()).eval(variables, p={**premises, **invalid})
            for row in rule.constraints
        )

    # Map every old one-edge intent to explicit normalized metadata. Old proofs
    # are not imported: the new arithmetic and visible mask obligations remain.
    old = Template.model_validate(
        yaml.safe_load(
            (
                STUDY.parents[1]
                / "seed/references/flash_attention_2_fwd/templates/masked_tile_pruning.yaml"
            ).read_text()
        )
    )
    assert {v.name for v in old.variables} == {"mask_kind", "masked_steps", "masked_first"}
    projections = 0
    for edge_steps, first, traversal in itertools.product(
        range(1, 9), (False, True), ("ascending", "descending")
    ):
        left_edge = first == (traversal == "ascending")
        info = (
            0,
            edge_steps + 2,
            edge_steps if left_edge else 0,
            2,
            0 if left_edge else edge_steps,
        )
        emitted = plan(info, "peel_boundaries", traversal)
        flags_in_order = [flag for _, flag in emitted]
        assert flags_in_order == (
            [True] * edge_steps + [False] * 2 if first else [False] * 2 + [True] * edge_steps
        )
        projections += 1
    binding = {
        "identity": "masked_tile_pruning@2",
        "variables": variables,
        "p": premises,
        "negative_scope_fields": flags,
        "invalid_p": [
            {"end": -2, "leading": 0, "core": -2, "trailing": 0},
            {"end": 0, "leading": 0, "core": 0, "trailing": 0},
        ],
        "source_binding": (
            "Conditional CTA256/N128 window256/256 with2/2/2 segments; scalar "
            "classification checked; native coordinates/ranges are premises."
        ),
    }
    (STUDY / "inputs/fmha-mask-binding-cases.json").write_text(
        json.dumps({"cases": [binding]}, indent=2) + "\n"
    )
    result = {
        "outcome": "PASS",
        "source_metadata_domains": metadata_domains,
        "normalized_domains": normalized,
        "unnormalized_domains_excluded": unnormalized,
        "normalized_plans": plans,
        "two_edge_witness": witness,
        "witness_coordinate_checks": coordinate_cases,
        "one_edge_intent_projections": projections,
        "source_negative_counts_actual_tile": negative_actual,
        "negative_case_host_guard_accepts": True,
        "scope_mutations": len(flags),
        "boundary": (
            "Retained classifier/loop expressions and exact coordinate plans; "
            "native producer/TMEM mapping, traversal and empty-row behavior re"
            "main open."
        ),
    }
    (STUDY / "checks/fmha-mask-segments-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: FMHA three-segment mask refinement and rejected negative/empty metadata.\n"
    )


if __name__ == "__main__":
    main()
