"""Seed import, Template proposal, Pattern matching and chain details (kta.knowledge)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from knowhelp import CODE, SEED, gemm_chain, seed, source_id, template, write_yaml
from kta.core.context import Context
from kta.core.errors import SchemaError
from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import Chain
from kta.knowledge import (
    SeedError,
    TxDescription,
    errors_of,
    import_references,
    import_seed,
    load_patterns,
    match_pattern,
    propose_template,
    validate_chain,
    validate_template,
)
from kta.knowledge.pending import PendingView
from kta.knowledge.pool import find_template, records
from kta.knowledge.seed import reference_order
from kta.store.base import Store
from storehelp import T0


def test_seed_import_is_idempotent(ctx: Context, store: Store) -> None:
    first = seed(ctx, store)
    assert first.counts == {"Source": 1, "Pattern": 8, "Structure": 9, "Template": 7}
    assert first.warnings == []
    again = seed(ctx, store)
    assert again.written == [] and len(again.reused) == len(first.written)
    assert len(records(store, "Template")) == 7


def test_seed_templates_cite_the_seed_source(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    source = records(store, "Source")[0]
    for rec in records(store, "Template"):
        assert rec.refs["sources"] and store.get(rec.refs["sources"][0]).body == source.body


def _copy_seed(tmp: Path) -> Path:
    root = tmp / "seed"
    for sub in ("sources", "patterns", "structures", "templates"):
        for f in (SEED / sub).glob("*.yaml"):
            (root / sub).mkdir(parents=True, exist_ok=True)
            (root / sub / f.name).write_bytes(f.read_bytes())
    return root


def test_bad_batch_writes_nothing(ctx: Context, store: Store, tmp_path: Path) -> None:
    root = _copy_seed(tmp_path)
    bad = yaml.safe_load((root / "templates" / "async_transfer.yaml").read_text())
    bad["refines"] = "no_such_pattern"
    write_yaml(root / "templates" / "async_transfer.yaml", bad)
    with pytest.raises(SeedError) as exc:
        import_seed(ctx, store, root, CODE)
    assert any(i.kind == "unknown_pattern" for i in exc.value.issues)
    assert records(store, "Pattern") == [] and records(store, "Template") == []


def test_changed_body_under_same_name_is_a_conflict(
    ctx: Context, store: Store, tmp_path: Path
) -> None:
    root = _copy_seed(tmp_path)
    import_seed(ctx, store, root, CODE)
    body = yaml.safe_load((root / "patterns" / "map.yaml").read_text())
    body["description"] = "changed"
    write_yaml(root / "patterns" / "map.yaml", body)
    with pytest.raises(SeedError) as exc:
        import_seed(ctx, store, root, CODE)
    assert [i.kind for i in exc.value.issues] == ["conflict"]


def _reference(root: Path, name: str, matched: int) -> None:
    d = root / name
    write_yaml(
        d / "source.yaml", {"name": f"{name}_src", "type": "git", "locator": "x", "revision": "r"}
    )
    write_yaml(d / "problem.yaml", {"name": f"{name}_problem", "reference_impl": "torch"})
    write_yaml(d / "reference.yaml", {"name": name, "problem": f"{name}_problem", "build": "make"})
    chain = gemm_chain(markers=False)
    chain["workload"] = f"{name} gemm"
    write_yaml(d / "chain.yaml", chain)
    write_yaml(
        d / "gaps" / "ux-01-tile.yaml",
        {"missing": "template", "description": f"{name} tile", "level": 2, "attribution": "space"},
    )
    write_yaml(
        d / "reconstruction.yaml",
        {
            "matched": matched,
            "total": 4,
            "notes": f"{name} notes",
            "unmatched_transactions": [
                {"level": 2, "pattern_guess": "map", "description": "[ux-01]", "where": "k.cu:1"}
            ],
        },
    )


def test_import_references_in_readme_order(ctx: Context, store: Store, tmp_path: Path) -> None:
    seed(ctx, store)
    refs = tmp_path / "references"
    _reference(refs, "b_ref", 3)
    _reference(refs, "a_ref", 2)
    (refs / "README.md").write_text("## Order\n\n1. `b_ref`\n2. `a_ref`\n", encoding="utf-8")
    assert [p.name for p in reference_order(refs)] == ["b_ref", "a_ref"]
    result = import_references(ctx, store, refs, CODE)
    assert result.counts["Reconstruction"] == 2 and result.counts["Gap"] == 2
    recon = records(store, "Reconstruction", order="created_at")
    assert [r.body["matched"] for r in recon] == [3, 2]
    chain = store.get(str(recon[0].refs["chain"]))
    assert len(chain.refs["templates"]) == 3


def test_pending_view_sees_batch_records(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    t = template("multi_buffer_rotation", name="ring_buffer", relations=[])
    rec = Record.new("Template", t, Actor(type="human", name="t"), T0)
    view = PendingView(store, [rec])
    assert find_template(view, "ring_buffer@1") is not None
    assert view.status(next(iter(view.pending))) == ("proposed", None)
    assert find_template(store, "ring_buffer@1") is None


def test_propose_template_writes_proposed_declared(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    src = source_id(store)
    t = template("multi_buffer_rotation", name="ring_buffer", version=1)
    rid = propose_template(ctx, store, t, [src])
    assert store.status(rid) == ("proposed", None)  # no assurance change = declared (D203)
    assert propose_template(ctx, store, t, [src]) == rid
    changed = template("multi_buffer_rotation", name="ring_buffer", summary="other words")
    with pytest.raises(SchemaError, match="bump the version"):
        propose_template(ctx, store, changed, [src])
    v2 = propose_template(
        ctx, store, template("multi_buffer_rotation", name="ring_buffer", version=2), [src]
    )
    assert store.get(v2).refs["previous"] == rid
    with pytest.raises(SchemaError):
        propose_template(ctx, store, template("multi_buffer_rotation", refines="nope"), [src])


def test_hw_contract_names_are_checked_when_given(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    t = template("shared_resource_binding")
    warnings = [
        i for i in validate_template(t, store, {"smem_banks": 32}) if i.kind == "unknown_hw_param"
    ]
    assert {w.where for w in warnings} == {
        "constraints.c_queue_fixed",
        "constraints.c_min_wavefronts",
    }
    assert errors_of(validate_template(t, store, {"smem_banks": 32})) == []


def test_match_pattern(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    patterns = load_patterns(store)
    hits = match_pattern(
        TxDescription(
            "cp.async copy of the A tile from global to shared memory", level=2, category="transfer"
        ),
        patterns,
    )
    assert hits[0][0] == "transfer"
    assert match_pattern(TxDescription("zzz qqq"), patterns) == [("other", 0.0)]


def _chain(**item: Any) -> Chain:
    data = gemm_chain()
    data["items"][2]["bindings"] = item
    return Chain.model_validate(data)


def test_chain_constant_bindings_are_evaluated(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    issues = validate_chain(_chain(S=2, P=1, beta=0), store)
    assert [i.kind for i in issues] == ["constraint_false"]
    assert issues[0].severity == "warning"
    assert validate_chain(_chain(S=2, P=1, beta=1), store) == []


def test_chain_may_bind_pattern_parameters(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    bound = _chain(S=4, P=3, beta=1, **{"p.size_bytes": 16384, "p.capacity_bytes": 32768})
    assert [i.kind for i in validate_chain(bound, store)] == ["constraint_false"]  # 4*16K > 32K


def test_undeclared_pattern_parameter_is_a_warning(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    t = template("multi_buffer_rotation", scope={"predicate": "p.iteratons >= 1", "text": "t"})
    issues = validate_template(t, store)
    typo = [i for i in issues if i.kind == "undeclared_pattern_param"]
    assert [(i.where, i.severity) for i in typo] == [("scope.predicate", "warning")]
    assert errors_of(issues) == []
