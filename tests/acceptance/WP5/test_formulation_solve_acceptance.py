"""WP5 acceptance (spec/modules/formulation-solve-models.md, "验收", unit-test part).

The real-run part (GEMM family on three shapes, top 8 measured, rank correlation) is WP7's.
"""

from __future__ import annotations

import pytest

from kta.core.context import Context
from kta.core.schemas.knowledge import Chain
from kta.formulation import GLUE, Formulation, assemble, per_unit_work
from kta.models import UnitModel
from kta.solve import enumerate_space, solve, solve_mip
from kta.store.base import Store
from solvehelp import TEST_HW, case, gemm, linear_family

pytestmark = pytest.mark.acceptance


def test_constraint_sources_after_assembly(ctx: Context, store: Store) -> None:
    fp, templates, f = gemm(ctx, store)
    keys = {i.key: i.template for i in fp.chain.items}
    # Every constraint names the chain item (or the glue) it came from ...
    assert {c.source for c in f.constraints} == set(keys) | {GLUE}
    for c in f.constraints:
        if c.source == GLUE:
            assert c.id in {g.id for g in fp.family.glue_constraints}
            continue
        t = templates[keys[c.source]]
        assert c.id in {x.id for x in [*t.constraints, *t.preconditions]}
    # ... and so does every cost term, with the Template's unit and overlap.
    for term in f.cost_terms:
        spec = templates[keys[term.source]].cost[term.index]
        assert (term.unit, term.overlap) == (spec.unit, spec.overlap)
    # Spot checks: the lifetime constraint of the rotation step, the glue constraint.
    assert f.constraint("ab_stages:c_lifetime").expr.free_vars == {"P", "S"}
    assert f.constraint("glue:g_acc_regs").kind == GLUE


def test_constant_binding_removes_variable(ctx: Context, store: Store) -> None:
    fp, templates, f = gemm(ctx, store)
    # beta of multi_buffer_rotation is bound to the constant 1: it is not a search
    # variable, the block records it as fixed and the constraint carries the constant.
    stages = f.blocks["ab_stages"]
    assert stages.fixed == {"beta": 1}
    assert "beta" not in f.variables
    assert f.constraint("ab_stages:c_lifetime").expr.src == "P <= S - 2 + 1"
    # Binding a variable that was free to a constant removes it from the search too.
    items = [i.model_copy(deep=True) for i in fp.chain.items]
    stages_item = next(i for i in items if i.key == "ab_stages")
    stages_item.bindings["P"] = 1
    family_vars = [v for v in fp.family.variables if v.name != "P"]
    fp2 = fp.__class__(
        fp.path,
        fp.family.model_copy(update={"variables": family_vars}),
        Chain(workload=fp.chain.workload, items=items),
        fp.edge_cases,
        fp.readme,
        fp.gen_path,
        fp.gen_function,
    )
    f2 = assemble(fp2, fp2.chain, templates, case(), TEST_HW.limits, previous=f)
    assert "P" not in f2.variables
    assert f2.blocks["ab_stages"].fixed == {"P": 1, "beta": 1}
    assert f2.constraint("ab_stages:c_lifetime").expr.free_vars == {"S"}
    # Only the changed step's block was rebuilt.
    assert set(f2.reused) == set(f.blocks) - {"ab_stages"}


def test_enumeration_and_mip_agree_on_small_linear_problem() -> None:
    fp, templates = linear_family(hi=100)
    f = assemble(fp, fp.chain, templates, case(), {})
    assert f.linear and f.cost_linear
    model = UnitModel(target=TEST_HW.target)
    shape = {"cap": 150, "base": 200000}
    found: list[tuple[float, dict[str, int]]] = []

    def visit(a: dict[str, int]) -> None:
        p = model.predict(*_work(f, a, shape), TEST_HW)
        found.append((p.total, a))

    stats = enumerate_space(f, shape, TEST_HW.limits, visit)
    assert stats.exhaustive and stats.feasible + stats.pruned == 100 * 100
    found.sort(key=lambda x: x[0])
    mip = solve_mip(f, shape, TEST_HW, model, shortlist=3, time_limit_s=30)
    assert mip.first_optimal
    assert mip.assignments[0] == found[0][1]
    mip_totals = [model.predict(*_work(f, a, shape), TEST_HW).total for a in mip.assignments]
    assert mip_totals == pytest.approx([t for t, _ in found[:3]], rel=1e-12)
    # solve() itself: below 10^6 points it enumerates and claims model-optimality.
    result = solve(f, case(), shape, model, TEST_HW, shortlist=3)
    assert result.method == "enumerate" and result.exhaustive and result.model_optimal
    assert result.candidates[0].assignment == found[0][1]


def _work(
    f: Formulation, a: dict[str, int], shape: dict[str, int]
) -> tuple[dict[str, float], dict[str, float]]:
    return per_unit_work(f, a, shape, TEST_HW.limits)
