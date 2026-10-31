"""WP20 learning-loop runner: admit, language gaps, coverage levels, ablation, plans.

The seven WP10a decompositions are imported through admit into a test store (never into
``knowledge/seed/``). The T2 retry test runs the real agent runner with the fake Codex
CLI (tests/fakes/fake_codex.py); its submissions go through the human path because the
fake CLI does no model work.
"""

from __future__ import annotations

import random
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from knowhelp import CODE, template
from kta.agents import batches
from kta.agents.cards import RetryNote
from kta.agents.config import agents_config
from kta.agents.profile import ProfileSpec, write_profile
from kta.agents.runner import Deps, run_step
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.errors import ConfigError
from kta.core.ids import UlidFactory
from kta.core.registry import Registry
from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import Chain, Family, Reference, ReferencePriority, Template
from kta.core.trace import new_tracer, tracing
from kta.execution.executor import JobExecutor
from kta.execution.registry import register, worker_configs
from kta.knowledge import SeedError, coverage_report, validate_template
from kta.knowledge.ablation import (
    Factor,
    Pair,
    design,
    estimate,
    factors,
    propose_couplings,
    suspected_pairs,
)
from kta.knowledge.admission import admit_references
from kta.knowledge.coverage import coverage_page
from kta.knowledge.learning import (
    ReferenceFacts,
    issue_class,
    performance_core,
    priority,
    stages,
    system_feature,
    tiers,
)
from kta.knowledge.types import Issue
from kta.orchestrator.learning_plan import plan
from kta.orchestrator.tasks import CURRENT_PROFILE, TaskQueue
from kta.orchestrator.transition import TaskRequest
from kta.store import CodeState, code_state, open_ops, open_store
from kta.store.base import Store
from proxyhelp import direct
from replayhelp import HUMAN, NOTE, REFERENCES, pool, references, t2_files
from storehelp import T0

FAKE = Path(__file__).resolve().parents[1] / "fakes" / "fake_codex.py"


def _admit_seven(ctx: Context, store: Store) -> dict[str, str]:
    pool(ctx, store)
    result, reports = admit_references(ctx, store, REFERENCES, CODE)
    assert result.counts["Reconstruction"] == 7 and len(result.language_gaps) == 61
    return reports


def test_seven_decompositions_admitted_with_counted_language_gaps(
    ctx: Context, store: Store
) -> None:
    reports = _admit_seven(ctx, store)
    data = {k: store.get(v).body["data"] for k, v in reports.items()}
    assert all(d["admitted"] and not d["errors"] for d in data.values())
    gaps = {k: d["counts"]["language_gaps"] for k, d in data.items()}
    assert gaps["cutlass_ampere_tf32_gemm"] == 19 and gaps["vllm_rms_norm"] == 0
    explained = [w for d in data.values() for w in d["warnings"]]
    assert len(explained) == 3 and all(w["explanation"] for w in explained)
    # Admitted at proposed: no StatusChange moved any imported Template.
    assert not store.query("StatusChange", limit=10)
    rep = coverage_report(store, None, random.Random(0))
    page = coverage_page(rep, {})
    levels = {r["name"]: r for r in page["learning"]["sota_levels"]}
    assert list(levels) == ["expressible", "reproducible", "reachable", "surpassed"]
    assert levels["expressible"]["references"] == 2 and levels["expressible"]["of"] == 7
    assert round(levels["expressible"]["step_rate"], 2) == 0.73  # 99 / 135 (WP10a)
    assert levels["reproducible"]["pending"] and levels["surpassed"]["pending"]
    sys_f = page["learning"]["system_features"]
    assert sys_f["total"] == 9 and sys_f["by_feature"]["dispatch"] == 5
    assert page["learning"]["language_gaps"]["total"] == 61
    tl = page["learning"]["backends"]["tilelang"]
    assert tl["steps"] == 135 and tl["expressible"] == 0 and tl["intent_confirmed"] is None


def test_an_unexplained_warning_refuses_the_batch(
    ctx: Context, store: Store, tmp_path: Path
) -> None:
    pool(ctx, store)
    root = tmp_path / "refs"
    shutil.copytree(REFERENCES / "cutlass_ampere_tf32_gemm", root / "cutlass_ampere_tf32_gemm")
    (root / "cutlass_ampere_tf32_gemm" / "warnings.yaml").unlink()
    with pytest.raises(SeedError, match="unexplained_warning"):
        admit_references(ctx, store, root, CODE)
    assert not store.query("Reconstruction", limit=1)  # nothing written


def test_language_gaps_are_classified_from_the_table(ctx: Context, store: Store) -> None:
    pool(ctx, store)
    gap = Issue("bad_expression", "operator BitXor is not allowed", "constraints.c")
    typo = Issue("bad_expression", "undeclared variable 'Sx'", "constraints.c")
    assert issue_class(gap) == "language_gap" and issue_class(typo) == "error"
    t = template(
        "multi_buffer_rotation",
        constraints=[{"id": "c", "expr": "S ^ P", "meaning": "m", "cls": "semantic"}],
    )
    issues = validate_template(Template.model_validate(t.model_dump()), store)
    found = [i for i in issues if i.kind == "bad_expression"]
    assert [i.severity for i in found] == ["gap"]


def test_system_features_tiers_priority_and_core() -> None:
    assert system_feature({"description": "griddepcontrol.wait before reading q"}) == "pdl"
    assert system_feature({"description": "x", "pattern_guess": "system"}) == "other"
    assert system_feature({"description": "register allocation by nvcc"}) is None
    f = ReferenceFacts("r", "gb10", runnable_on=("gb10",), unreproduced_steps=3)
    assert tiers(f) == ["L0", "L1", "L2"]
    assert tiers(ReferenceFacts("r", "gb10")) == ["L0"]
    assert stages("closed_library") == ["intake", "cover"] and stages("nonsense") == stages("other")
    p = ReferencePriority(
        cases_affected=2, gap_closable=0.5, uncertainty=1.0, reasons=["strongest_baseline"]
    )
    assert priority(p) == 2.0 and priority(None) == 0.0
    core = performance_core(
        "fi",
        {"unmatched_transactions": [{"description": "fetched through the page table"}]},
        {"items": [{"key": "a"}, {"key": "b"}]},
        {"dtype": "f16"},
        "rtol 1e-2",
    )
    assert core.excluded == {"page_table": ["fetched through the page table"]}
    assert core.hot_loop == ["a", "b"] and core.epsilon_rel == 0.10


def _family_and_chain() -> tuple[Family, Chain, dict[str, Template]]:
    fam = Family.model_validate(
        {
            "name": "f",
            "version": 1,
            "backend": "tilelang",
            "entry": "gen.py",
            "variables": [
                {
                    "name": n,
                    "kind": "numeric",
                    "binding": "compile",
                    "role": "software",
                    "domain": [1, 2, 4],
                    "meaning": n,
                }
                for n in ("S", "P")
            ],
            "glue_constraints": [
                {"id": "g", "expr": "P <= S", "meaning": "lookahead", "cls": "semantic"}
            ],
        }
    )
    chain = Chain.model_validate(
        {
            "workload": "gemm",
            "items": [
                {"key": "copy", "template": "async_transfer@1", "applies_to": "A"},
                {"key": "ring", "template": "multi_buffer_rotation@1", "applies_to": "A"},
            ],
        }
    )
    ts = {
        "async_transfer@1": template("async_transfer"),
        "multi_buffer_rotation@1": template("multi_buffer_rotation"),
    }
    return fam, chain, ts


def test_ablation_design_and_coupling_evidence(ctx: Context, store: Store) -> None:
    fam, chain, ts = _family_and_chain()
    fs = factors(fam, chain)
    pairs = suspected_pairs(fs, fam, chain, ts)
    assert Pair("P", "S", "shared_variable") in pairs
    d = design(fs, pairs)
    assert (
        d.mode == "pairwise" and d.runs[0] == {} and len(d.runs) == 1 + len(d.factors) + len(pairs)
    )
    # Synthetic times: P and S interact (joint 1.5 vs product 1.2 x 1.1), others multiply.
    single = {"P": 1.2, "S": 1.1, "copy": 1.3, "ring": 1.05}
    times = []
    for run in d.runs:
        off = sorted(run)
        t = 1.0
        for n in off:
            t *= single[n]
        times.append(1.5 if off == ["P", "S"] else t)
    est = {(e.pair.a, e.pair.b): e for e in estimate(d, times)}
    assert est[("P", "S")].coupled and not any(e.coupled for k, e in est.items() if k != ("P", "S"))
    who = Actor(type="tool", name="t")
    runs = [  # stand-ins for the run Measurements (test data, no GPU)
        store.put(
            Record.new("Report", {"title": "run", "subject": str(i), "summary": "s"}, who, T0)
        )
        for i in range(2)
    ]
    fam_id = store.put(Record.new("Family", fam, Actor(type="tool", name="t"), ctx.clock.now()))
    ids = propose_couplings(
        store,
        d,
        list(est.values()),
        family_id=fam_id,
        target="gb10",
        workload="gemm",
        evidence=runs,
        actor=Actor(type="tool", name="t"),
        now=ctx.clock.now(),
    )
    rec = store.get(ids[0])
    assert len(ids) == 1 and rec.kind == "Coupling" and rec.refs["evidence"] == sorted(runs)
    assert {m["name"] for m in rec.body["members"]} == {"P", "S"}
    with pytest.raises(ConfigError):
        estimate(d, times[:-1])


def test_fractional_design_when_large() -> None:
    fs = [Factor(f"x{i}", "variable", on=1, off=0) for i in range(8)]
    pairs = [Pair(f"x{i}", f"x{j}", "shared_variable") for i in range(8) for j in range(i + 1, 8)]
    d = design(fs, pairs)
    assert d.mode == "fractional" and len(d.runs) <= 64 and d.aliases
    # A pure multiplicative response shows no coupling in the fractional estimate either.
    times = [2.0 ** sum(1 - r[f"x{i}"] for i in range(8)) for r in d.runs]
    assert not any(e.coupled for e in estimate(d, times))


def test_dry_run_plan_of_a_runnable_reference(ctx: Context, store: Store) -> None:
    pool(ctx, store)
    ref_id = references(ctx, store)["cutlass_ampere_tf32_gemm"]
    p = plan(store, ref_id, "gb10")
    by = {s.name: s for s in p.stages}
    assert p.dry_run and p.tiers == ["L0", "L1"]  # no Reconstruction yet: L2 not entered
    assert by["intake"].status == "planned" and len(by["intake"].jobs) == 8
    assert any("e13f5cd4" in " ".join(j.argv) for j in by["intake"].jobs)
    assert by["decompose"].status == "task" and by["reproduce"].status == "not applicable"
    body = Reference.model_validate(store.get(ref_id).body)
    assert body.source_class == "open_code" and body.priority is None


# ---- T2 rejected, retried in the same fake-adapter session ----------------------------


@pytest.fixture
def actx(tmp_path: Path) -> Context:
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text('{"fake": "credential"}')
    config = {
        "workers": {"local": {"transport": "local", "pools": {"cpu": 1, "agent": 1}}},
        "executor": {"queue_timeout_s": 60},
        "agents": {
            "worker": "local",
            "batch": "b1",
            "pool": "agent",
            "token_key_file": str(tmp_path / "keys" / "task-token.key"),
            "hosts": {
                "local": {
                    "codex_cmd": [sys.executable, str(FAKE)],
                    "auth_dir": str(auth),
                    "nvidia_smi": "/nonexistent/nvidia-smi",
                }
            },
        },
    }
    direct(tmp_path / "d")  # codex goes direct in these tests: no network (WP22)
    return Context(
        env="test", config=config_from_mapping(config), data_dir=tmp_path / "d", registry=Registry()
    )


def test_t2_admission_rejection_goes_back_to_the_same_session(actx: Context) -> None:
    store = open_store(actx, CodeState(code_state().commit, False))
    ops = open_ops(actx)
    register(actx, store, ops, worker_configs(actx)["local"], code_state().commit)
    deps = Deps(store, ops, JobExecutor(actx, store, ops, code_state().commit), agents_config(actx))
    profile = write_profile(
        store, datetime.now(UTC), ProfileSpec(name="codex-test", isolation="config")
    )
    batches.build(store, actx.data_dir, "b1", profile, code_state().commit)
    ops.set_pointer(CURRENT_PROFILE, None, profile)
    pool(actx, store)
    ref = references(actx, store)["cutlass_ampere_tf32_gemm"]
    q = TaskQueue(actx, store, ops)
    task = q.create(
        TaskRequest("T2", "Decompose this reference into a composition chain", [ref]),
        "expand-cutlass",
        "learn",
    )
    owner = "agent:0"

    q.lease(task, owner)
    q.start(task, owner)
    tracer = new_tracer(ops, actx.clock, UlidFactory(actx.clock), {})
    with tracing(tracer):
        first = run_step(actx, q.get(task), deps=deps)
    files = t2_files("cutlass_ampere_tf32_gemm")
    del files["warnings.yaml"]  # the deliverable leaves its admission warning unexplained
    q.submit(task, owner, files, NOTE.format(ref="cutlass", what="T2"), HUMAN)
    v = q.check(task)
    adm = next(o for o in v.outcomes if o.name == "admission")
    assert v.state == "queued" and not adm.passed
    assert adm.failures == [
        "warning constraint_false@chain.yaml:items.frag_stages is not explained in warnings.yaml"
    ]
    assert adm.data["counts"]["language_gaps"] == 19

    q.lease(task, owner)
    q.start(task, owner)
    failures, _ = q.retry_context(task)
    assert ("admission", adm.failures[0]) in failures
    with tracing(tracer):
        second = run_step(actx, q.get(task), deps=deps, retry=RetryNote(failures))
    assert second.decision.startswith("resume") and second.run.session == first.run.session
    prompt = store.blob_path(store.get(second.run_id).blobs["prompt"]).read_text()
    assert "constraint_false@chain.yaml:items.frag_stages" in prompt
    q.submit(
        task, owner, t2_files("cutlass_ampere_tf32_gemm"), NOTE.format(ref="c", what="T2"), HUMAN
    )
    assert q.check(task).state == "accepted"
    admission = store.query("Report", {"title": "admission"}, limit=5)
    assert len(admission) == 1 and admission[0].body["data"]["admitted"]
