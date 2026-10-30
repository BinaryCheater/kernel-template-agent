"""WP4 acceptance (spec/modules/knowledge.md "验收", unit-test part; spec/workplan.md WP4).

- the five seed Templates pass validate_template;
- each validator returns no Issue on correct input and has one counterexample per error class;
- free variables of expressions are checked;
- duplicates are reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from knowhelp import SEED_TEMPLATES, family_dir, seed, template, template_body
from kta.core.context import Context
from kta.core.schemas.knowledge import Chain, Template
from kta.knowledge import (
    errors_of,
    load_family,
    load_patterns,
    validate_chain,
    validate_family,
    validate_template,
)
from kta.knowledge.pool import find_template
from kta.store.base import Store

pytestmark = pytest.mark.acceptance

GOOD_CHAIN: dict[str, Any] = {
    "workload": "fp16 GEMM, M = N = K = 4096",
    "items": [
        {
            "key": "k_split",
            "template": "reduction_partition@1",
            "applies_to": "sum over K",
            "bindings": {
                "num_groups": "KT",
                "group_size": 32,
                "assignment": "contiguous_sequential",
            },
        },
        {
            "key": "ab_smem",
            "template": "physical_storage@1",
            "applies_to": "A and B tiles",
            "bindings": {"level": "smem", "buffer": "dedicated", "share_order": "none"},
            "after": ["k_split"],
        },
        {
            "key": "ab_stages",
            "template": "multi_buffer_rotation@1",
            "applies_to": "A and B k-tiles",
            "bindings": {"S": 3, "P": 1, "beta": 1},
            "after": ["ab_smem"],
        },
    ],
}


@pytest.fixture
def pool(ctx: Context, store: Store) -> Store:
    seed(ctx, store)
    return store


def kinds(t: Template, store: Store) -> set[str]:
    return {i.kind for i in errors_of(validate_template(t, store))}


def test_seed_templates_pass_validate_template(pool: Store) -> None:
    for name in SEED_TEMPLATES:
        hit = find_template(pool, f"{name}@1")
        assert hit is not None, name
        issues = validate_template(hit[1], pool, None)
        assert errors_of(issues) == [], (name, issues)


def test_seed_pattern_table_has_the_eight_categories(pool: Store) -> None:
    patterns = load_patterns(pool)
    expected = {"reduce", "contract", "map", "store", "transfer", "sync", "access", "other"}
    assert expected <= set(patterns)
    assert all(len(p.examples) >= 2 for p in patterns.values())


def test_validate_template_error_classes(pool: Store) -> None:
    body = template_body("multi_buffer_rotation")
    cases: dict[str, Template] = {
        "missing_field": template("multi_buffer_rotation", summary=" "),
        "bad_expression": template(
            "multi_buffer_rotation",
            constraints=[{"id": "c", "expr": "P <=", "meaning": "m", "cls": "semantic"}],
        ),
        "unknown_pattern": template("multi_buffer_rotation", refines="no_such_pattern"),
        "unknown_hw_feature": template("multi_buffer_rotation", requires_hw=["no_such_feature"]),
        "unknown_relation_target": template(
            "multi_buffer_rotation", relations=[{"type": "requires", "target": "nothing_here"}]
        ),
        "bad_same_as": template(
            "multi_buffer_rotation",
            preconditions=[{**body["preconditions"][0], "same_as": "c_missing"}],
        ),
        "same_as_mismatch": template(
            "multi_buffer_rotation",
            preconditions=[{**body["preconditions"][0], "expr": "P <= S - 1"}],
        ),
        "duplicate_variable": template(
            "multi_buffer_rotation", variables=[*body["variables"], body["variables"][0]]
        ),
    }
    for kind, t in cases.items():
        assert kind in kinds(t, pool), kind


def test_expression_free_variables_are_checked(pool: Store) -> None:
    t = template(
        "multi_buffer_rotation",
        constraints=[{"id": "c", "expr": "P <= slots", "meaning": "m", "cls": "semantic"}],
    )
    issues = errors_of(validate_template(t, pool))
    assert any(i.kind == "bad_expression" and "slots" in i.message for i in issues)
    # Γ may only use p.*, hw.*, case.*: a Template variable there is undeclared.
    scoped = template("multi_buffer_rotation", scope={"predicate": "S >= 1", "text": "t"})
    assert "bad_expression" in kinds(scoped, pool)


def test_duplicate_is_reported_as_a_hint(pool: Store) -> None:
    copy = template("multi_buffer_rotation", name="ring_buffer")
    issues = validate_template(copy, pool)
    dups = [i for i in issues if i.kind == "possible_duplicate"]
    assert dups and all(i.severity == "warning" for i in dups)
    assert errors_of(issues) == []


def test_validate_chain_good_and_error_classes(pool: Store) -> None:
    assert errors_of(validate_chain(Chain.model_validate(GOOD_CHAIN), pool)) == []

    def broken(**item_changes: Any) -> set[str]:
        data: dict[str, Any] = {**GOOD_CHAIN, "items": [dict(i) for i in GOOD_CHAIN["items"]]}
        data["items"][2].update(item_changes)
        return {i.kind for i in errors_of(validate_chain(Chain.model_validate(data), pool))}

    assert "unknown_template" in broken(template="nope@1")
    assert "unbound_variable" in broken(bindings={"S": 3, "P": 1})
    assert "unknown_binding" in broken(bindings={"S": 3, "P": 1, "beta": 1, "Q": 2})
    assert "binding_out_of_domain" in broken(bindings={"S": 99, "P": 1, "beta": 1})
    assert "unknown_after" in broken(after=["missing"])
    assert "cycle" in broken(key="k_split")  # duplicate key also creates a self-loop
    no_storage: dict[str, Any] = {**GOOD_CHAIN, "items": [GOOD_CHAIN["items"][2] | {"after": []}]}
    found = {i.kind for i in validate_chain(Chain.model_validate(no_storage), pool)}
    assert "missing_required" in found  # multi_buffer_rotation requires physical_storage


def test_validate_family_good_and_error_classes(pool: Store, tmp_path: Path) -> None:
    good = load_family(family_dir(tmp_path / "good"))
    assert validate_family(good, pool, {"smem_per_cta": 101376}) == []

    def broken(name: str, **changes: Any) -> set[str]:
        fp = load_family(family_dir(tmp_path / name, **changes))
        return {i.kind for i in errors_of(validate_family(fp, pool))}

    base = yaml.safe_load((tmp_path / "good" / "family.yaml").read_text(encoding="utf-8"))
    extra = [*base["variables"], {**base["variables"][0], "name": "unused"}]
    assert "extra_family_variable" in broken("extra", family={"variables": extra})
    assert "unbound_name" in broken("unbound", family={"variables": base["variables"][1:]})
    no_meaning = [{**base["glue_constraints"][0], "meaning": " "}]
    assert "missing_field" in broken("glue", family={"glue_constraints": no_meaning})
    assert "missing_marker" in broken(
        "markers", gen="def generate(a, c):\n    return {'files': {}}\n"
    )
    assert "generate_failed" in broken("edge", edge_cases=[{"M": -1}])
