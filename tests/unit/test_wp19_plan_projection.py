"""WP19 (+ WP17): FormulationPlan, chain enumerator, access discipline, projection compiler.

Code-stage acceptance (spec/workplan.md WP17, WP19): the GEMM seed formulation is
compiled and solved by the projection compiler and the MIP objective equals L1's
projection at its points; the default plan is produced for the GEMM seed case; the
enumerator finds the seed chain within its bound; an agent-plan fixture is diffed and
scored; the task header fits one screen; a pull log is written for a fake-adapter task.
"""

from __future__ import annotations

import pytest
from test_wp15_effects_hwmodel import FAMILY, SHAPE, gb10, gemm_case, gemm_formulation

from kta.core.context import Context
from kta.core.errors import NotSupported
from kta.core.schemas import Actor
from kta.models import MachineModel
from kta.solve import projection
from kta.solve.mip import NotExportable
from kta.store.base import Store


@pytest.fixture
def mm() -> MachineModel:
    arch, hw = gb10()
    return MachineModel.build(hw, arch)


def test_gemm_seed_formulation_compiles_and_solves(
    ctx: Context, store: Store, mm: MachineModel
) -> None:
    f = gemm_formulation(ctx, store, mm)
    out = projection.compile_and_solve(f, SHAPE, mm, shortlist=4)
    assert out.assignments and out.first_optimal, out
    # The objective is L1's projection at peak at every returned point (translation).
    for a, obj in zip(out.assignments, out.objective_us, strict=True):
        assert obj == pytest.approx(projection.projection_us(f, a, SHAPE, mm), rel=1e-6)
    assert out.objective_us == sorted(out.objective_us)
    assert {"one_hot", "resmii", "table_term"} <= set(out.rules), out.rules
    assert out.menus["KT"] <= 2  # narrowed by the constraints pinning KT = ceil_div(K, BK)


def test_projection_is_the_minimum_of_enumerated_projection(
    ctx: Context, store: Store, mm: MachineModel
) -> None:
    from kta.solve.enumerate import enumerate_space

    f = gemm_formulation(ctx, store, mm)
    shape = {"M": 1024, "N": 1024, "K": 16384}
    best: list[float] = []
    enumerate_space(
        f, shape, mm.hw, lambda a: best.append(projection.projection_us(f, a, shape, mm))
    )
    out = projection.compile_and_solve(f, shape, mm, shortlist=1)
    assert out.objective_us[0] == pytest.approx(min(best), rel=1e-6)


def test_outer_enumeration_and_its_limit(ctx: Context, store: Store, mm: MachineModel) -> None:
    f = gemm_formulation(ctx, store, mm)
    out = projection.compile_and_solve(f, SHAPE, mm, outer=["S"], shortlist=2)
    assert "S" in out.outer and out.outer_points >= 3
    with pytest.raises(NotExportable, match="outer enumeration"):
        projection.compile_and_solve(f, SHAPE, mm, outer=["S", "BM"], outer_max=4)
    # A table limit below every joint table forces atoms outward until they fit.
    small = projection.compile_and_solve(f, SHAPE, mm, table_max=64, shortlist=1)
    assert small.outer and any("outer" in n for n in small.notes)


def test_consistency_check_waits_for_l2() -> None:
    with pytest.raises(NotSupported, match="WP16"):
        projection.consistency_check()


# ---- FormulationPlan -------------------------------------------------------------------


def _default(ctx: Context, store: Store, mm: MachineModel):
    from kta.knowledge import load_family
    from kta.solve.plan import default_plan

    f = gemm_formulation(ctx, store, mm)
    fp = load_family(FAMILY)
    return fp, f, default_plan(fp, f, mm, SHAPE)


def test_default_plan_for_the_gemm_seed_case(ctx: Context, store: Store, mm: MachineModel) -> None:
    from kta.core.schemas.plan import FormulationPlan
    from kta.solve.plan import diff_plans, record_plan

    fp, f, d = _default(ctx, store, mm)
    assert d.author == "default" and [s.key for s in d.chain] == [i.key for i in fp.chain.items]
    assert {v.name for v in d.variables} == set(f.variables)
    assert {v.status for v in d.variables} <= {"open", "outer"}
    assert d.table_terms and all(len(t.over) >= 2 for t in d.table_terms)
    assert [g.id for g in d.glue] == ["g_prefetch", "g_acc_regs", "g_no_combine"]
    assert diff_plans(d, d) == []
    # The same object round-trips through an immutable record.
    rid = record_plan(store, d, Actor(type="system", name="wp19-test"), ctx.clock.now())
    assert FormulationPlan.model_validate(store.get(rid).body) == d


def test_agent_plan_fixture_is_diffed_and_scored(
    ctx: Context, store: Store, mm: MachineModel
) -> None:
    from kta.core.schemas.plan import FormulationPlan
    from kta.solve.plan import diff_plans, score_plan

    _, f, d = _default(ctx, store, mm)
    # The agent-plan fixture: one cited fix, one uncited menu cut, one uncited glue addition.
    body = d.model_dump(mode="json")
    body["author"] = "agent"
    for v in body["variables"]:
        if v["name"] == "BM":
            v.update(status="fixed", value=128, structure="low_effective_dimension",
                     evidence=["01FINDINGFIXTURE0000000000"])  # fmt: skip
        if v["name"] == "threads":
            v["menu"] = [128]
    body["glue"].append({"id": "g_extra", "expr": "BN >= BM", "why": "fixture"})
    agent = FormulationPlan.model_validate(body)
    changes = diff_plans(d, agent)
    assert [(c.section, c.key, c.cited) for c in changes] == [
        ("variables", "BM", True),
        ("variables", "threads", False),
        ("glue", "g_extra", False),
    ]
    s = score_plan(agent, d, f, mm, SHAPE)
    assert s.changes == 3 and s.uncited == ["variables:threads", "glue:g_extra"]
    assert not s.complete and s.space < s.default_space
    assert s.bound_us is not None and s.default_bound_us is not None
    assert s.bound_us >= s.default_bound_us - 1e-9  # a restriction cannot beat the default
    body["citations"] = {"glue": {"structure": "dominance", "evidence": ["01FINDINGFIXTURE01"]}}
    agent = FormulationPlan.model_validate(body)
    assert score_plan(agent, d, f, mm, SHAPE, solve=False).uncited == ["variables:threads"]
    assert score_plan(agent, d, f, mm, SHAPE, require_citation=False, solve=False).complete


# ---- chain enumerator --------------------------------------------------------------------


def test_enumerator_finds_the_seed_chain_within_its_bound(
    ctx: Context, store: Store, mm: MachineModel
) -> None:
    from kta.knowledge import load_family
    from kta.models import min_work
    from kta.models.l1 import sol_us
    from kta.solve.chains import enumerate_chains, start_symbols

    f = gemm_formulation(ctx, store, mm)
    del f
    fp = load_family(FAMILY)
    from kta_wp7.pipeline import templates_of

    tmpl = templates_of(store, fp)
    seed_steps = [tmpl[i.template] for i in fp.chain.items]
    start = start_symbols(seed_steps)
    problem, case = gemm_case()
    sol = sol_us(min_work(problem, SHAPE), mm)
    res = enumerate_chains(store, case, start, mm, sol)
    hit = res.find(i.template for i in fp.chain.items)
    assert hit is not None, res.coverage()
    assert hit.bound_us == pytest.approx(max(sol.values()))
    assert res.best is not None and res.best.bound_us == hit.bound_us
    cov = res.coverage()
    assert cov["missing_rules"] == [] and cov["chains"] == len(res.chains) >= 1
    # Pruning against an incumbent faster than SOL removes every bounded chain.
    pruned = enumerate_chains(store, case, start, mm, sol, incumbent_us=0.5 * max(sol.values()))
    assert pruned.find(i.template for i in fp.chain.items) is None
    assert pruned.counts["pruned_incumbent"] >= 1
    # A start symbol no Template refines is reported as a missing rule.
    missing = enumerate_chains(store, case, [*start, ("nonexistent", 1)], mm, sol)
    assert missing.missing == [("nonexistent", 1)] and not missing.chains


def test_composition_stats_is_a_view_over_chains(
    ctx: Context, store: Store, mm: MachineModel
) -> None:
    from kta.knowledge import load_family
    from kta.solve.chains import composition_stats, composition_view
    from kta_wp7.pipeline import templates_of

    gemm_formulation(ctx, store, mm)
    fp = load_family(FAMILY)
    stats = composition_stats([fp.chain, fp.chain], templates_of(store, fp))
    assert stats["chains"] == 2
    assert stats["conditional"]["contract@L4 → compute_instruction_selection"] == 1.0
    assert stats["order"]["output_tiling<compute_instruction_selection"] == 2
    from kta.core.schemas import Record

    rec = Record.new(
        "Chain", fp.chain.model_dump(mode="json"), Actor(type="system", name="t"), ctx.clock.now()
    )
    store.put(rec)
    view = composition_view(store)
    assert view["chains"] == 1 and view["unknown_templates"] == {}


# ---- access discipline: header, skill table, pull log ----------------------------------


def _header(**kw: object):
    from kta.orchestrator.access import task_header

    args: dict[str, object] = dict(
        case="gemm_fp16", case_id="01CASE", bound_us=1495.0, bottleneck="tensor",
        gap=None, plan_id="01PLAN", score={"changes": 0, "open_dims": 6, "space": 432,
        "bound_us": 1578.3}, budget={"tokens": 200000, "wall_s": 3600}, discipline="v1",
    )  # fmt: skip
    args.update(kw)
    return task_header(**args)  # type: ignore[arg-type]


def test_header_fits_one_screen_and_states_the_situation() -> None:
    from kta.core.errors import ConfigError
    from kta.orchestrator.access import HEADER_MAX_COLS, HEADER_MAX_LINES, situation_of

    h = _header()
    lines = h.text.splitlines()
    assert len(lines) <= HEADER_MAX_LINES and max(map(len, lines)) <= HEADER_MAX_COLS
    assert h.situation == "start" and "01PLAN" in h.text and "1495" in h.text
    gap = {"class": "l_rep", "idle": "dram", "eps": 1.0, "delta": 2.0, "l_rep": 30.0}
    assert situation_of(gap) == "idle_resource" and situation_of({"class": "eps"}) == "eps"
    assert _header(gap=gap).situation == "idle_resource"
    with pytest.raises(ConfigError, match="lines"):
        _header(max_lines=5)


def test_skill_carries_the_rendered_table() -> None:
    from kta.core.access import SKILL_BEGIN, SKILL_END, render, table
    from kta.core.config import repo_root

    text = (repo_root() / "src/kta/agents/skills/use-kta/SKILL.md").read_text("utf-8")
    block = text[text.index(SKILL_BEGIN) : text.index(SKILL_END) + len(SKILL_END)]
    assert block == render(table("v1"))
    t = table("v1")
    assert t.classify("query", None, "start") is None
    assert t.classify("query chain", None, "start") == "start"
    assert t.classify("show", "Finding", "open_variable") == "open_variable"
    assert t.classify("show", "Finding", "start") is None  # outside the row: logged


def test_task_md_starts_with_the_header() -> None:
    from kta.agents import cards
    from kta.core.schemas.process import TaskCard

    card = TaskCard(type="T1", goal="g", profile="p", work_unit="w", header=_header())
    out = cards.task_md(
        "# $task_id\n$goal\n$inputs\n$budget\n$skills\n$commands\n$done_checks\n$previous",
        task_id="t", card_id="c", card=card, cards=[], skills=[], commands=[], used={},
        retry=None, policy={"cards_k": 5, "card_max_lines": 5, "step_header_max_chars": 8000},
    )  # fmt: skip
    assert out.startswith("## Case header")


def test_pull_log_written_for_a_task_and_shown_on_release(tmp_path) -> None:
    from test_api_agent import _auth, _first_id

    from apihelp import put, request, seeded_api
    from kta.api.app import create_app
    from kta.api.controls import release_view

    api = seeded_api(tmp_path / "data", {"token_key_file": str(tmp_path / "key")})
    plan_id = put(api, "FormulationPlan", {"family": "f@1", "author": "default",
                                           "chain_source": "family", "chain": []})  # fmt: skip
    header = _header(plan_id=plan_id).model_dump(mode="json")
    body = {
        "type": "T1",
        "goal": "case task",
        "profile": "p",
        "work_unit": "wp19",
        "header": header,
    }
    card = put(api, "TaskCard", body, {"plan": plan_id})
    queued = api.ops.enqueue(card, 1000, "wp19")
    view = {h["task"]: h for h in release_view(api)["held"]}
    # A held call shows its plan before it runs (and no pull log yet).
    assert view[queued]["plan"]["id"] == plan_id and view[queued]["pull_log"] is None
    task = api.ops.lease("wp19-test", ["T1"], 600)
    assert task is not None and task.id == queued
    app, auth = create_app(api), _auth(api, tmp_path, queued)
    tid = _first_id(api, "Template")
    case = {"what": "chain", "case": _first_id(api, "Case")}
    assert request(app, "POST", "/agent/query", case, auth)[0] == 200
    assert request(app, "GET", f"/agent/records/{tid}", None, auth)[0] == 200
    status, log = request(app, "GET", f"/pulls?task={queued}", None)
    assert status == 200 and [p["command"] for p in log["pulls"]] == ["query chain", "show"]
    assert log["by_row"] == {"outside": 1, "start": 1} and log["in_table"] == 1
    status, stats = request(app, "GET", "/pulls", None)
    assert status == 200 and stats["by_version"] == {"v1": 2}
    assert request(app, "GET", f"/plans/{plan_id}", None)[1]["id"] == plan_id


def test_solve_selects_the_projection_arm(ctx: Context, store: Store, mm: MachineModel) -> None:
    from kta.models import UnitModel
    from kta.solve import solve

    f = gemm_formulation(ctx, store, mm)
    _, case = gemm_case()
    r = solve(f, case, SHAPE, UnitModel(target="gb10"), mm, shortlist=4, projection={})
    assert r.method == "projection" and r.candidates and r.mip_status == "Optimal"
    assert any(n.startswith("projection: outer") for n in r.notes)
