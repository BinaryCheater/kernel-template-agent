"""WP24: verification in the loop (paper §2.2, §4; spec/workplan.md WP24).

Every acceptance item has a test here: the five seed steps end with a recorded outcome
(no silent pass), a planted unsound Pre gives a replayed counterexample, a K1 failure is
caught, composition rejects an incompatible assumption, a premise change invalidates,
reuse keeps unchanged certificates, assurance comes only from proof records, and the
weakest-precondition search and Δpre follow the paper's tiling case.
"""

from __future__ import annotations

from typing import Any

import yaml

from knowhelp import SEED, SEED_TEMPLATES
from kta.api.common import Api
from kta.api.studies import pool as pool_view
from kta.checks.proof import compose, records, wp
from kta.checks.proof.candidate import check_candidate
from kta.checks.proof.encode import Translator, constant
from kta.checks.proof.models.slots import linearize, replay, schedule_queries
from kta.checks.proof.prove import find_template, prove_template
from kta.checks.proof.smt import run
from kta.checks.proof.template import check_template
from kta.core.context import Context
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import Chain, Template
from kta.core.schemas.process import StatusChange
from kta.core.schemas.proof import ProofOutline
from kta.knowledge import load_family
from kta.orchestrator.gap import decompose
from kta.store import open_ops
from kta.store.assurance import effective
from kta.store.sqlite import SqliteStore
from proofhelp import GEMM_CASES, gb10_hw, gemm_family, put_outline, seeded, template_id
from storehelp import T0

TILING = "shared_memory_tiling@1"


def _yaml(rel: str) -> dict[str, Any]:
    return yaml.safe_load((SEED / rel).read_text(encoding="utf-8"))


def _tiling(pre: str | None = None) -> Template:
    t = Template.model_validate(_yaml("templates/shared_memory_tiling.yaml"))
    if pre is None:
        return t
    pres = [p.model_copy(update={"expr": pre}) for p in t.preconditions]
    cons = [
        c.model_copy(update={"expr": pre}) if c.id == "c3_lifetime" else c for c in t.constraints
    ]
    return t.model_copy(update={"preconditions": pres, "constraints": cons})


def _tiling_outline() -> ProofOutline:
    return ProofOutline.model_validate(_yaml("proofs/shared_memory_tiling.yaml"))


# ---------------------------------------------------------------- the paper's tiling case


def test_event_model_matches_paper_table16() -> None:
    """Appendix A: the safe set over S in 1..3, P in 0..3 and beta equals P <= S - 2 + beta,
    and every unsafe witness replays to a wrong term."""
    for s in (1, 2, 3):
        for p in range(4):
            for beta in (0, 1):
                phases = max(5, s + p + 2)
                q = schedule_queries(s, p, beta, 2, phases, 5000)
                safe = q["readiness"].result == "unsat" and q["lifetime"].result == "unsat"
                assert safe == (p <= s - 2 + beta), (s, p, beta)
                if not safe:
                    bad = q["readiness"] if q["readiness"].result == "sat" else q["lifetime"]
                    rep = replay(s, 2, phases, q["progs"], linearize(bad.model, q["progs"]))
                    assert rep["admissible"] and not rep["agrees"]


def test_tiling_template_index_proved_schedule_bounded() -> None:
    chk = check_template(_tiling(), _tiling_outline())
    out = {(d.condition, d.obligation): d.outcome for d in chk.drafts}
    assert out[("K1", "pre_lifetime")] == "proved" and out[("K3", "*")] == "proved"
    assert all(out[("K2", o)] == "proved" for o in ("coverage", "exact_once", "no_phantom"))
    assert all(out[("K2", o)] == "bounded" for o in ("readiness", "lifetime", "agreement"))
    assert chk.level == "bounded" and not chk.unchecked
    k1 = next(d for d in chk.drafts if d.condition == "K1")
    assert k1.data["same_as"]["confirmed"] is True


def test_planted_unsound_pre_yields_replayed_counterexample() -> None:
    """The canary: P <= S - 1 (barrier ignored) admits the race of the paper's Figure 7."""
    chk = check_template(_tiling("P <= S - 1"), _tiling_outline())
    life = next(d for d in chk.drafts if d.obligation == "lifetime")
    assert life.outcome == "counterexample" and life.replay is not None
    assert (
        life.replay["admissible"] and not life.replay["agrees"] and life.replay["differences"] > 0
    )
    admitted_unsafe = [
        (r["S"], r["P"], r["beta"]) for r in life.data["table"] if r["pre"] and not r["safe"]
    ]
    assert (2, 1, 0) in admitted_unsafe
    assert chk.level is None


def test_weakest_precondition_relaxes_the_conservative_contract() -> None:
    exact = wp.search(_tiling(), _tiling_outline())
    assert exact.relation == "exact" and exact.weaker is None
    cons = wp.search(_tiling("beta == 1 and P <= S - 1"), _tiling_outline())
    assert cons.relation == "stronger" and cons.verified == "bounded"
    assert cons.weaker is not None
    r = wp.relaxed(_tiling("beta == 1 and P <= S - 1"), "pre_lifetime", cons.weaker)
    assert next(c for c in r.constraints if c.id == "c3_lifetime").expr == cons.weaker
    for row in cons.table:  # the found predicate is the safe set on the menu
        env = {"S": row["S"], "P": row["P"], "beta": row["beta"]}
        assert eval(cons.weaker, {}, env) == row["safe"]
    assert wp.search(_tiling("P <= S - 1"), _tiling_outline()).relation == "unsound"


def test_delta_pre_reads_the_relaxed_prediction() -> None:
    band = {"use_measurement_ci": False, "rel_of_t": 0.01}
    b = {"best_us": 60.0, "bound_us": 40.0, "predicted_us": 54.0, "top_pred_us": 54.0,
         "solver": {"model_optimal": True}, "relaxed_pred_us": 53.3,
         "delta_pre_note": "x: stronger"}  # fmt: skip
    dec = decompose(b, band)
    assert dec["terms"]["delta_pre"]["status"] == "value"
    assert abs(dec["terms"]["delta_pre"]["us"] - 0.7) < 1e-9
    assert (
        decompose({**b, "relaxed_pred_us": None}, band)["terms"]["delta_pre"]["status"]
        == "not_assessed"
    )


# ---------------------------------------------------------------- K1, K3 and the builder


def test_k1_failure_caught_on_a_fixture() -> None:
    """C admits a point Pre rejects (P <= S with Pre P <= S - 1): K1 is a counterexample
    whose point the interpreter confirms (C true, Pre false, in Γ)."""
    t = _tiling()
    loose = [
        c.model_copy(update={"expr": "P <= S", "id": "c3_loose"}) if c.id == "c3_lifetime" else c
        for c in t.constraints
    ]
    pres = [p.model_copy(update={"same_as": None}) for p in t.preconditions]
    t = t.model_copy(update={"constraints": loose, "preconditions": pres})
    chk = check_template(t, None, conditions=("K1",))
    k1 = chk.drafts[0]
    assert k1.outcome == "counterexample" and k1.replay is not None and k1.replay["agrees"]
    cx = k1.counterexample or {}
    assert cx["P"] <= cx["S"] and not cx["P"] <= cx["S"] - 2 + cx["beta"]


def test_unsupported_construct_is_unresolved_never_proved() -> None:
    pre = _tiling().preconditions[0].model_copy(update={"expr": "log2(S) <= 2", "same_as": None})
    t = _tiling().model_copy(update={"preconditions": [pre]})
    k1 = check_template(t, None, conditions=("K1",)).drafts[0]
    assert k1.outcome == "unresolved" and "unsupported" in k1.reason


def test_floor_division_and_modulo_follow_python() -> None:
    import z3

    for a in (-7, -1, 0, 5, 7):
        for b in (-3, -2, 2, 3):
            env = {"a": constant("a", a), "b": constant("b", b)}
            for src, want in (
                ("a // b", a // b),
                ("a % b", a % b),
                ("ceil_div(a, b)", -(-a // b)),
            ):
                got = z3.simplify(Translator(env).num(src))
                assert got.as_long() == want, (src, a, b)  # type: ignore[union-attr]


def test_assumption_naming_a_design_variable_is_refused() -> None:
    cheat = {"name": "cheat", "statement": "x", "expr": "P <= S - 2 + beta"}
    o = ProofOutline.model_validate({**_tiling_outline().model_dump(), "assumes": [cheat]})
    chk = check_template(_tiling("P <= S - 1"), o)
    assert any("cannot become an assumption" in g for g in chk.gaps)
    assert chk.level is None


def test_solver_witness_and_unknown_are_never_proofs() -> None:
    import z3

    x = z3.Int("x")
    assert run(z3.And(x > 0, x < 0)).result == "unsat"
    assert run(x * x == 2, {"x": x}).result in ("unsat", "unknown")
    q = run(x > 3, {"x": x})
    assert q.result == "sat" and q.model["x"] > 3


# ---------------------------------------------------------------- seed, records, reuse


def test_seed_steps_end_with_recorded_outcomes(ctx: Context, store: SqliteStore) -> None:
    """Each of the five seed steps ends with a summary; every obligation of Σ is either a
    checked result (with a reason when not positive) or listed as unchecked."""
    seeded(ctx, store)
    assert store.count("ProofOutline") == 7
    for name in SEED_TEMPLATES:
        tid = template_id(store, f"{name}@1")
        p = prove_template(store, tid, T0)
        s = store.get(p.summary).body
        names = {c["obligation"] for c in s["checked"] if c["condition"] == "K2"}
        names |= {u["obligation"] for u in s["unchecked"]}
        t = store.get(tid).body
        assert names == {o["name"] for o in t["obligations"]}, name
        for c in s["checked"]:
            assert c["outcome"] in ("proved", "bounded", "counterexample", "unresolved")
            if c["outcome"] not in ("proved", "bounded"):
                assert c.get("reason") or store.get(c["result"]).body["counterexample"], (name, c)
        assert (s["level"] is None) == bool(s["gaps"])


def test_assurance_only_from_live_proof_records(ctx: Context, store: SqliteStore) -> None:
    seeded(ctx, store)
    tid = template_id(store, TILING)
    claim = StatusChange(subject=tid, to_assurance="proved", reason="author says so")
    store.put(Record.new("StatusChange", claim, Actor(type="checker", name="someone"), T0))
    assert store.status(tid)[1] == "proved" and effective(store, tid) == "declared"
    p = prove_template(store, tid, T0)
    assert p.level == "bounded" and effective(store, tid) == "bounded"


def test_invalidation_on_premise_change(ctx: Context, store: SqliteStore) -> None:
    seeded(ctx, store)
    tid = template_id(store, TILING)
    p = prove_template(store, tid, T0)
    assert effective(store, tid) == "bounded"
    current = records.versions()
    marked = records.drift(store, T0, {**current, "checker": "proof@older"})
    assert p.summary in marked and set(p.results) <= set(marked)
    assert effective(store, tid) == "declared"
    again = prove_template(store, tid, T0)
    assert again.reused == 0 and effective(store, tid) == "bounded"
    # a new outline version supersedes the old one: results built on it go stale
    o = _yaml("proofs/shared_memory_tiling.yaml")
    put_outline(store, {**o, "version": 2})
    assert again.summary in records.drift(store, T0)


def test_reuse_keeps_certificates_when_only_constraints_strengthen(
    ctx: Context, store: SqliteStore
) -> None:
    """Corollary 1: a new version that only tightens C reuses the K2 certificates."""
    seeded(ctx, store)
    tid = template_id(store, TILING)
    first = prove_template(store, tid, T0)
    body = store.get(tid).body
    c4 = {"id": "c4", "expr": "Tk <= 32", "meaning": "tighter", "cls": "resource"}
    body2 = {**body, "version": 2, "constraints": [*body["constraints"], c4]}
    t2 = store.put(
        Record.new(
            "Template",
            body2,
            Actor(type="tool", name="t"),
            T0,
            refs={"sources": store.get(tid).refs["sources"]},
        )
    )
    put_outline(
        store, {**_yaml("proofs/shared_memory_tiling.yaml"), "template": "shared_memory_tiling@2"}
    )
    second = prove_template(store, t2, T0)
    reused = [store.get(r) for r in second.results if store.get(r).refs.get("reused")]
    assert {r.body["obligation"] for r in reused} == {
        "coverage",
        "exact_once",
        "no_phantom",
        "readiness",
        "lifetime",
        "agreement",
    }
    assert all(r.refs["reused"] in first.results for r in reused)
    assert second.level == "bounded" and second.queries < first.queries


# ---------------------------------------------------------------- composition and candidates


def test_composition_rejects_an_incompatible_assumption(ctx: Context, store: SqliteStore) -> None:
    seeded(ctx, store)
    fam = gemm_family(store)
    without = compose.prove_family(store, fam, T0, case_domain=GEMM_CASES, hw=gb10_hw())
    assert without.level is None
    assert any("bounded_arbitration" in g and "neither guaranteed" in g for g in without.gaps)
    declared = compose.prove_family(
        store, fam, T0, case_domain=GEMM_CASES, declared=("bounded_arbitration",), hw=gb10_hw()
    )
    assert not any("bounded_arbitration" in g for g in declared.gaps)
    rows = next(
        store.get(r).body
        for r in declared.results
        if store.get(r).body["obligation"] == "assumptions"
    )
    how = {x["assumption"]: x["how"] for x in rows["data"]["rows"]}
    assert how["bounded_arbitration"] == "declared assumption of the whole result"
    assert how["queue_exists"] == "discharged by the chain bindings"


def test_chain_steps_are_checked_under_their_bindings(ctx: Context, store: SqliteStore) -> None:
    """Bindings narrow a step: physical_storage's p-only Pre fails K1 for the Template but
    holds for the chain (0 + 0 <= 1); output_tiling with index_map=identity is bounded."""
    seeded(ctx, store)
    alone = prove_template(store, template_id(store, "physical_storage@1"), T0)
    assert any(g.startswith("K1 pre_lifetime: counterexample") for g in alone.gaps)
    fam = gemm_family(store)
    p = compose.prove_family(
        store, fam, T0, case_domain=GEMM_CASES, declared=("bounded_arbitration",), hw=gb10_hw()
    )
    steps = {
        store.get(s).body["part"]: store.get(s).body for s in store.get(p.summary).refs["steps"]
    }
    assert steps["step:ab_smem"]["level"] == "bounded"
    assert steps["step:tiles"]["level"] == "bounded"
    assert steps["step:ab_stages"]["level"] == "bounded"
    assert p.level is None and any("ab_async" in g for g in p.gaps)


def test_candidate_check_in_c_and_gamma() -> None:
    fp = load_family(SEED / "families" / "gemm_fp16_tilelang")
    chain = Chain.model_validate(fp.chain.model_dump())
    tpl = {
        n: Template.model_validate(_yaml(f"templates/{n.split('@')[0]}.yaml"))
        for n in {i.template for i in chain.items}
    }
    steps = [(it, tpl[it.template]) for it in chain.items]
    shape, hw = {"M": 512, "N": 512, "K": 512}, gb10_hw()
    good = {"BM": 128, "BN": 128, "BK": 32, "S": 3, "P": 2, "threads": 128, "split": 1, "KT": 16}
    ok = check_candidate(steps, good, shape, hw)
    assert ok.ok and any(u.startswith("mma:") for u in ok.unknown)  # p.operand_bits unbound
    bad = check_candidate(steps, {**good, "P": 3}, shape, hw)  # beta = 1: P <= S - 1 fails
    assert not bad.ok and any("ab_stages" in f for f in bad.failures)


# ---------------------------------------------------------------- API view


def test_pool_view_lists_assurance_and_every_obligation(ctx: Context, store: SqliteStore) -> None:
    seeded(ctx, store)
    prove_template(store, template_id(store, TILING), T0)
    prove_template(store, template_id(store, "async_transfer@1"), T0)
    api = Api(ctx=ctx, store=store, ops=open_ops(ctx))
    view = pool_view(api, None)
    assert view["assurance"] == {"declared": 7, "tested": 0, "bounded": 1, "proved": 0}
    rows = {f"{t['name']}@{t['version']}": t["proof"] for t in view["templates"]}
    assert rows[TILING]["level"] == "bounded" and rows[TILING]["last_run"]
    at = {o["name"]: o for o in rows["async_transfer@1"]["obligations"]}
    assert at["readiness"]["status"] == "counterexample"
    assert at["mechanism_available"]["status"] == "unchecked"
    assert all(o["status"] == "never checked" for o in rows["output_tiling@2"]["obligations"])
    assert find_template(store, TILING) == record_id(store.get(template_id(store, TILING)))


# ---------------------------------------------------------------- CLI


def test_kta_prove_commands(tmp_path: Any, monkeypatch: Any, capsys: Any) -> None:
    import json

    from kta.cli.main import EXIT_OK, main

    monkeypatch.setenv("KTA_ENV", "test")
    monkeypatch.setenv("KTA_TEST_DATA", str(tmp_path / "data"))

    def run(*argv: str) -> tuple[int, Any]:
        code = main([*argv, "--json"])
        return code, json.loads(capsys.readouterr().out)

    assert run("init")[0] == EXIT_OK
    code, canary = run("prove", "canary")
    assert code == EXIT_OK and canary["caught"] and canary["counterexamples"]
    code, everything = run("prove", "all")
    levels = {r["template"]: r["level"] for r in everything["templates"]}
    assert code == EXIT_OK and levels[TILING] == "bounded" and len(levels) == 7
    fam_dir = str(SEED / "families" / "gemm_fp16_tilelang")
    code, fam = run(
        "prove",
        "family",
        fam_dir,
        "--from-dir",
        "--target",
        "gb10",
        "--declare",
        "bounded_arbitration",
    )
    assert code == EXIT_OK and fam["level"] is None and fam["gaps"]
    code, again = run("prove", "family", fam_dir, "--from-dir", "--target", "gb10")
    assert again["family"] == fam["family"]  # recorded once
    code, w = run("prove", "wp", TILING)
    assert code == EXIT_OK and w["relation"] == "exact"
    assert run("prove", "status", fam["family"])[1]["assurance"] == "declared"
