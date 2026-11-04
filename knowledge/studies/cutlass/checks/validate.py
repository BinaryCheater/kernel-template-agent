"Validate proposed records and bounded premise witnesses; no Store admission."

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Chain, Template
from kta.core.schemas.proof import ProofOutline

STUDY = Path(__file__).resolve().parents[1]


def main() -> None:
    directory = STUDY / "delta/cache_prefetch@1"
    template = Template.model_validate(yaml.safe_load((directory / "template.yaml").read_text()))
    proof = ProofOutline.model_validate(yaml.safe_load((directory / "proof.yaml").read_text()))
    assert proof.template == f"{template.name}@{template.version}" == "cache_prefetch@1"
    snapshot = json.loads((STUDY / "inputs/pool-reconciled-snapshot.json").read_text())
    for entry in snapshot["files"]:
        path = STUDY.parents[2] / entry.get("retained_path", entry["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        if path.suffix == ".yaml":
            body = yaml.safe_load(path.read_text())
            assert not (
                isinstance(body, dict)
                and body.get("name") == template.name
                and body.get("version") == template.version
            )
    premises = {
        "hints_value_neutral": True,
        "hint_addresses_valid": True,
        "immutable_source": True,
        "tiles": 3,
    }
    variables = {"distance": 4}
    expressions = [*template.constraints, *template.preconditions, template.scope]
    assert all(
        Expr.parse(x.expr if hasattr(x, "expr") else x.predicate, variables.keys()).eval(
            variables, p=premises
        )
        for x in expressions
    )
    rejected = []
    for key in ("hints_value_neutral", "hint_addresses_valid", "immutable_source"):
        altered = {**premises, key: False}
        assert not Expr.parse(template.scope.predicate, variables.keys()).eval(variables, p=altered)
        rejected.append(key)
    window_dir = STUDY / "delta/cross_buffer_operand_window@1"
    window = Template.model_validate(yaml.safe_load((window_dir / "template.yaml").read_text()))
    outline = ProofOutline.model_validate(yaml.safe_load((window_dir / "proof.yaml").read_text()))
    assert outline.template == f"{window.name}@{window.version}"
    window_premises = {
        "window_extent": 96,
        "two_chunk_descriptor_supported": True,
        "chunk_layout_legal": True,
        "chunks_ready_and_live": True,
        "tail_padding_valid": True,
    }
    window_vars = {"chunk_extent": 256}
    for expression in [*window.constraints, *window.preconditions]:
        assert Expr.parse(expression.expr, window_vars.keys()).eval(window_vars, p=window_premises)
    assert not Expr.parse(window.constraints[0].expr, window_vars.keys()).eval(
        {"chunk_extent": 64}, p=window_premises
    )
    window_scope = Expr.parse(window.scope.predicate, window_vars.keys())
    assert window_scope.eval(window_vars, p=window_premises)
    for name in ("chunks_ready_and_live", "tail_padding_valid"):
        assert not window_scope.eval(window_vars, p={**window_premises, name: False})
    for entry in snapshot["files"]:
        path = STUDY.parents[2] / entry.get("retained_path", entry["path"])
        if path.suffix == ".yaml":
            body = yaml.safe_load(path.read_text())
            assert not (
                isinstance(body, dict)
                and body.get("name") == window.name
                and body.get("version") == window.version
            )
    store_dir = STUDY / "delta/staged_async_output_store@1"
    output_store = Template.model_validate(
        yaml.safe_load((store_dir / "template.yaml").read_text())
    )
    output_outline = ProofOutline.model_validate(
        yaml.safe_load((store_dir / "proof.yaml").read_text())
    )
    assert output_outline.template == f"{output_store.name}@{output_store.version}"
    output_vars = {"stages": 2}
    output_p = {
        "stage_bytes": 8192,
        "capacity_bytes": 16384,
        "async_store_legal": True,
        "source_read_wait_available": True,
        "output_tiles_disjoint": True,
        "source_visibility_fenced": True,
        "output_observed_after_write_completion": True,
    }
    for expression in [*output_store.constraints, *output_store.preconditions]:
        assert Expr.parse(expression.expr, output_vars.keys()).eval(output_vars, p=output_p)
    assert not Expr.parse(output_store.constraints[0].expr, output_vars.keys()).eval(
        {"stages": 3}, p=output_p
    )
    store_scope = Expr.parse(output_store.scope.predicate, output_vars.keys())
    assert store_scope.eval(output_vars, p=output_p)
    assert not store_scope.eval(
        output_vars, p={**output_p, "output_observed_after_write_completion": False}
    )
    for entry in snapshot["files"]:
        path = STUDY.parents[2] / entry.get("retained_path", entry["path"])
        if path.suffix == ".yaml":
            body = yaml.safe_load(path.read_text())
            assert not (
                isinstance(body, dict)
                and body.get("name") == output_store.name
                and body.get("version") == output_store.version
            )
    cancellation_dir = STUDY / "delta/launch_cancellation_work_reuse@1"
    cancellation = Template.model_validate(
        yaml.safe_load((cancellation_dir / "template.yaml").read_text())
    )
    cancellation_outline = ProofOutline.model_validate(
        yaml.safe_load((cancellation_dir / "proof.yaml").read_text())
    )
    assert cancellation_outline.template == f"{cancellation.name}@{cancellation.version}"
    cancellation_vars = {"participants": 416}
    cancellation_p = {
        "response_readers": 416,
        "response_slots": 1,
        "cancel_removes_launch": True,
        "success_ids_unique": True,
        "success_checked_before_coordinates": True,
        "stop_after_observed_failure": True,
        "slot_published_before_read": True,
        "slot_reused_after_all_readers": True,
        "group_decode_bijective": True,
        "group_launch_fair": True,
        "independent_groups": True,
        "response_bytes": 16,
    }
    for expression in [*cancellation.constraints, *cancellation.preconditions]:
        assert Expr.parse(expression.expr, cancellation_vars.keys()).eval(
            cancellation_vars, p=cancellation_p
        )
    cancellation_scope = Expr.parse(cancellation.scope.predicate, cancellation_vars.keys())
    assert cancellation_scope.eval(cancellation_vars, p=cancellation_p)
    assert not cancellation_scope.eval(
        cancellation_vars, p={**cancellation_p, "success_checked_before_coordinates": False}
    )
    for entry in snapshot["files"]:
        path = STUDY.parents[2] / entry.get("retained_path", entry["path"])
        if path.suffix == ".yaml":
            body = yaml.safe_load(path.read_text())
            assert not (
                isinstance(body, dict)
                and body.get("name") == cancellation.name
                and body.get("version") == cancellation.version
            )
    scatter_dir = STUDY / "delta/scatter_reduce_output_shards@1"
    scatter = Template.model_validate(yaml.safe_load((scatter_dir / "template.yaml").read_text()))
    scatter_outline = ProofOutline.model_validate(
        yaml.safe_load((scatter_dir / "proof.yaml").read_text())
    )
    assert scatter_outline.template == f"{scatter.name}@{scatter.version}"
    scatter_vars = {"peers": 2}
    scatter_p = {
        "partial_count": 2,
        "elements_per_thread": 128,
        "threads": 128,
        "element_bytes": 4,
        "control_bytes": 16,
        "mailbox_capacity_bytes": 32784,
        "local_partials_defined": True,
        "accumulator_fp32": True,
        "numeric_reassociation_allowed": True,
        "fragment_map_equal_and_injective": True,
        "async_peer_store_supported": True,
        "full_bytes_and_empty_acks_correct": True,
        "mailbox_phase_ordered": True,
    }
    for expression in [*scatter.constraints, *scatter.preconditions]:
        assert Expr.parse(expression.expr, scatter_vars.keys()).eval(scatter_vars, p=scatter_p)
    scatter_scope = Expr.parse(scatter.scope.predicate, scatter_vars.keys())
    assert scatter_scope.eval(scatter_vars, p=scatter_p)
    assert not scatter_scope.eval(scatter_vars, p={**scatter_p, "local_partials_defined": False})
    for name in ("local_partials_defined", "accumulator_fp32", "fragment_map_equal_and_injective"):
        assert not scatter_scope.eval(scatter_vars, p={**scatter_p, name: False})
    assert not Expr.parse(scatter.constraints[0].expr, scatter_vars.keys()).eval(
        scatter_vars, p={**scatter_p, "elements_per_thread": 127}
    )
    assert not cancellation_scope.eval(
        cancellation_vars, p={**cancellation_p, "stop_after_observed_failure": False}
    )
    for entry in snapshot["files"]:
        path = STUDY.parents[2] / entry.get("retained_path", entry["path"])
        if path.suffix == ".yaml":
            body = yaml.safe_load(path.read_text())
            assert not (
                isinstance(body, dict)
                and body.get("name") == scatter.name
                and body.get("version") == scatter.version
            )
    grouped_cases = json.loads((STUDY / "inputs/grouped-binding-cases.json").read_text())["cases"]
    extra_pairs = []
    blockwise_cases = json.loads((STUDY / "inputs/blockwise-binding-cases.json").read_text())[
        "cases"
    ]
    conv_cases = json.loads((STUDY / "inputs/conv-binding-cases.json").read_text())["cases"]
    mixed_cases = json.loads((STUDY / "inputs/mixed-input-binding-cases.json").read_text())["cases"]
    grouped_mixed = json.loads((STUDY / "inputs/grouped-mixed-binding-cases.json").read_text())[
        "cases"
    ]
    moe_cases = json.loads((STUDY / "inputs/moe-binding-cases.json").read_text())["cases"]
    moe_scaled = json.loads((STUDY / "inputs/moe-scaled-binding-cases.json").read_text())["cases"]
    row_cases = json.loads((STUDY / "inputs/row-reduction-binding-cases.json").read_text())["cases"]
    geforce_cases = json.loads((STUDY / "inputs/geforce-binding-cases.json").read_text())["cases"]
    fmha_cases = json.loads((STUDY / "inputs/fmha-binding-cases.json").read_text())["cases"]
    pruning_cases = json.loads((STUDY / "inputs/fmha-pruning-binding-cases.json").read_text())[
        "cases"
    ]
    for case in [
        *pruning_cases,
        *fmha_cases,
        *geforce_cases,
        *row_cases,
        *moe_scaled,
        *grouped_cases,
        *blockwise_cases,
        *conv_cases,
        *mixed_cases,
        *grouped_mixed,
        *moe_cases,
    ]:
        directory = STUDY / "delta" / case["identity"]
        record = Template.model_validate(yaml.safe_load((directory / "template.yaml").read_text()))
        record_proof = ProofOutline.model_validate(
            yaml.safe_load((directory / "proof.yaml").read_text())
        )
        assert record_proof.template == f"{record.name}@{record.version}" == case["identity"]
        assert set(case["variables"]) == {variable.name for variable in record.variables}
        for variable in record.variables:
            value = case["variables"][variable.name]
            if isinstance(variable.domain, list):
                assert value in variable.domain
            else:
                assert variable.domain.lo <= value <= variable.domain.hi
        for expression in [*record.constraints, *record.preconditions]:
            assert Expr.parse(expression.expr, case["variables"].keys()).eval(
                case["variables"], p=case["p"]
            )
        for invalid in case.get("invalid_variables", []):
            assert not all(
                Expr.parse(x.expr, case["variables"].keys()).eval(invalid, p=case["p"])
                for x in record.constraints
            )
        for invalid in case.get("invalid_p", []):
            assert not all(
                Expr.parse(x.expr, case["variables"].keys()).eval(
                    case["variables"], p={**case["p"], **invalid}
                )
                for x in record.constraints
            )
        scope = Expr.parse(record.scope.predicate, case["variables"].keys())
        assert scope.eval(case["variables"], p=case["p"])
        for field in case["negative_scope_fields"]:
            assert not scope.eval(case["variables"], p={**case["p"], field: False})
        for effect in record.effects.storage:
            if effect.when is not None:
                assert not Expr.parse(effect.when, case["variables"].keys()).eval(
                    case["variables"], p=case["p"]
                )
            assert (
                Expr.parse(effect.bytes, case["variables"].keys()).eval(
                    case["variables"], p=case["p"]
                )
                > 0
            )
        for entry in snapshot["files"]:
            path = STUDY.parents[2] / entry.get("retained_path", entry["path"])
            if path.suffix != ".yaml":
                continue
            body = yaml.safe_load(path.read_text())
            assert not (
                isinstance(body, dict)
                and body.get("name") == record.name
                and body.get("version") == record.version
            )
        extra_pairs.append(case["identity"])
    chain = Chain.model_validate(yaml.safe_load((STUDY / "delta/chain.yaml").read_text()))
    local_templates = {
        f"{x.name}@{x.version}": x for x in (template, window, output_store, cancellation, scatter)
    }
    previous = set()
    for item in chain.items:
        assert item.key not in previous and set(item.after) <= previous
        bound = local_templates[item.template]
        assert set(item.bindings) == {variable.name for variable in bound.variables}
        for variable in bound.variables:
            value = item.bindings[variable.name]
            if isinstance(variable.domain, list):
                assert value in variable.domain
            else:
                assert variable.domain.lo <= value <= variable.domain.hi
        previous.add(item.key)
    reuse = json.loads((STUDY / "inputs/reuse-bindings.json").read_text())
    reused = []
    resolved = {}
    for case in reuse["cases"]:
        matches = []
        for entry in snapshot["files"]:
            path = STUDY.parents[2] / entry.get("retained_path", entry["path"])
            if path.suffix != ".yaml":
                continue
            body = yaml.safe_load(path.read_text())
            if isinstance(body, dict) and "level_from" in body:
                if f"{body.get('name')}@{body.get('version')}" == case["identity"]:
                    matches.append(Template.model_validate(body))
        assert len(matches) == 1
        old = matches[0]
        assert set(case["variables"]) == {v.name for v in old.variables}
        for variable in old.variables:
            value = case["variables"][variable.name]
            if isinstance(variable.domain, list):
                assert value in variable.domain
            else:
                assert variable.domain.lo <= value <= variable.domain.hi
        for expression in [*old.constraints, *old.preconditions]:
            assert Expr.parse(expression.expr, case["variables"].keys()).eval(
                case["variables"], p=case["p"]
            )
        assert Expr.parse(old.scope.predicate, case["variables"].keys()).eval(
            case["variables"], p=case["p"]
        )
        resolved[case["identity"]] = old
        reused.append(case["identity"])
    outside = reuse["outside_old_scope"]
    old = resolved[outside["identity"]]
    domain = next(v.domain for v in old.variables if v.name == outside["variable"])
    assert not isinstance(domain, list) and not domain.lo <= outside["value"] <= domain.hi
    result = {
        "outcome": "PASS",
        "template_proof_pairs": 5 + len(extra_pairs),
        "partial_chain_steps": len(chain.items),
        "duplicate_identities": 0,
        "reused_binding_identities": reused,
        "outside_old_domain": outside,
        "rejected_missing_premises": rejected,
        "boundary": "Schema/expression checks only; no ProofSummary or admission.",
    }
    (STUDY / "checks/validation-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        f"PASS: {5 + len(extra_pairs)} Template/ProofOutline pairs "
        "and scoped premise expressions.\n"
    )


if __name__ == "__main__":
    main()
