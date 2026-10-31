"""WP14 knowledge model v2: upcast of old records, capabilities and the unprobed filter,
realisations, findings, the status ladder, promotion/demotion and the pool diff.

Each mechanism has a failure fixture here (spec/workplan.md round-4 rules).
"""

from __future__ import annotations

import copy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from apihelp import request
from fakes.clock import ManualClock
from knowhelp import CODE, SEED, seed, template, template_body
from kta.api.app import create_app
from kta.api.common import Api, envelope
from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.controls import baseline
from kta.core.errors import PermissionDenied
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.levels import LEVEL_MAPPINGS, map_level
from kta.knowledge import coverage_report, import_references, retrieve_templates, validate_template
from kta.knowledge.capabilities import target_capabilities, template_availability
from kta.knowledge.findings import PRIOR, couplings, findings
from kta.knowledge.pooldiff import diff
from kta.knowledge.promotion import demote, promote, review
from kta.knowledge.realisations import expected_sass, realisations_of
from kta.knowledge.retrieve import unavailable_templates
from kta.knowledge.types import errors_of
from kta.store.snapshot import snapshot
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from storehelp import T0

HUMAN = Actor(type="human", name="tester")
TOOL = Actor(type="tool", name="test")
REFERENCES = SEED / "references"
CONTROLS = baseline()


def _v1_template() -> dict[str, Any]:
    """A seed Template body in the pre-WP14 (schema 1) shape."""
    body = template_body("multi_buffer_rotation")
    body["level_from"], body["level_to"] = 2, 3  # paper numbering
    body["requires_hw"] = body.pop("requires")
    body.pop("probed_on")
    body["program"] = {**body["program"], "fragments": {"cuda": "a" * 64}}
    body["expected_sass"] = [{"when": "S > 1", "must_contain": ["LDGSTS"], "must_not_contain": []}]
    return body


def _stored(
    kind: str, schema: int, body: dict[str, Any], refs: dict[str, Any] | None = None
) -> Record:
    return Record.model_validate(
        {
            "kind": kind,
            "schema": schema,
            "created_at": T0.isoformat(),
            "actor": {"type": "human", "name": "old"},
            "refs": refs or {},
            "body": body,
            "origin": "real",
        }
    )


# ---- upcast of old-format records (D440, D441) ----------------------------------------


def test_level_mapping_is_versioned_and_total_on_old_levels() -> None:
    assert [map_level(n) for n in range(6)] == [2, 3, 4, 5, 6, 7]
    assert "paper-0to5@1" in LEVEL_MAPPINGS
    with pytest.raises(ValueError):
        map_level(6)  # the old numbering had no level 6


def test_old_template_reads_back_upcast_with_its_stored_id(
    ctx: Context, store: SqliteStore
) -> None:
    seed(ctx, store)
    old = _stored("Template", 1, _v1_template() | {"name": "old_ring", "version": 1})
    rid = store.put(old)
    back = store.get(rid)
    assert back.upcast and back.schema_ == 1
    assert record_id(back) == rid  # the stored form is what is hashed
    assert back.body["level_from"] == 4 and back.body["level_to"] == 5
    assert back.body["requires"] == [] and back.body["probed_on"] == []
    assert "expected_sass" not in back.body and "fragments" not in back.body["program"]
    assert store.get_json(rid)["body"]["expected_sass"]  # nothing rewritten
    legacy = realisations_of(store, "old_ring@1")
    assert len(legacy) == 1 and legacy[0].legacy and legacy[0].id is None
    assert legacy[0].body.fragments == {"cuda": "a" * 64}
    assert expected_sass(store, ["old_ring@1"], "tilelang")["old_ring@1"][0].must_contain == [
        "LDGSTS"
    ]


def test_old_pattern_gap_and_reconstruction_levels_are_mapped(store: SqliteStore) -> None:
    p = store.put(
        _stored(
            "Pattern",
            1,
            {
                "name": "p0",
                "level": 0,
                "category": "other",
                "description": "d",
                "examples": ["a", "b"],
            },
        )
    )
    g = store.put(
        _stored(
            "Gap", 1, {"missing": "other", "description": "g", "level": 5, "attribution": "other"}
        )
    )
    tx = {"level": 0, "pattern_guess": "map", "description": "x", "where": "f:1"}
    r = store.put(
        _stored(
            "Reconstruction",
            1,
            {"matched": 0, "total": 1, "unmatched_transactions": [tx], "notes": "n"},
        )
    )
    assert store.get(p).body["level"] == 2
    assert store.get(g).body["level"] == 7
    assert store.get(r).body["unmatched_transactions"][0]["level"] == 2


def test_a_newer_schema_than_the_code_is_refused() -> None:
    with pytest.raises(ValueError, match="schema"):
        _stored("Template", 9, _v1_template())


def test_api_envelope_shows_upcast_body_and_stored_body(
    ctx: Context, store: SqliteStore, ops: SqliteOps
) -> None:
    seed(ctx, store)
    rid = store.put(_stored("Template", 1, _v1_template() | {"name": "old_ring", "version": 1}))
    env = envelope(Api(ctx=ctx, store=store, ops=ops), rid)
    assert env["upcast_from"] == 1 and env["body"]["level_from"] == 4
    assert env["stored_body"]["level_from"] == 2


# ---- capabilities and the unprobed filter (D442, D445) -------------------------------


def _pool(ctx: Context, store: SqliteStore) -> None:
    seed(ctx, store)
    import_references(ctx, store, REFERENCES, CODE)


def test_seed_declares_gb10_capabilities(ctx: Context, store: SqliteStore) -> None:
    seed(ctx, store)
    caps = target_capabilities(store, "gb10")
    assert caps["warpgroup_mma"].status == "no" and caps["warpgroup_mma"].basis == "documented"
    assert caps["async_copy"].status == "yes"
    assert caps["cluster_multicast"].status == "unknown"


def test_unprobed_template_is_filtered_on_gb10_and_counted(
    ctx: Context, store: SqliteStore
) -> None:
    _pool(ctx, store)
    wgmma = "cross_iteration_software_pipelining@1"
    hidden = unavailable_templates(store, "gb10")
    assert wgmma in hidden["unavailable"]
    assert "cluster_multicast@1" in hidden["unconfirmed"]
    shown = {c.name for c in retrieve_templates(store, k=500, target="gb10")}
    assert wgmma not in shown and "multi_buffer_rotation@1" in shown
    everywhere = {c.name for c in retrieve_templates(store, k=500)}
    assert wgmma in everywhere  # no target: nothing is filtered
    import random

    rep = coverage_report(store, None, random.Random(0), targets=["gb10"])
    gb10 = rep.availability["gb10"]
    assert gb10["known_unavailable_here"] >= 1 and wgmma in gb10["unavailable"]
    assert gb10["available"] + gb10["known_unavailable_here"] + gb10["unconfirmed_here"] == len(
        store.query("Template", limit=1000)
    )


def test_target_declaring_a_capability_wakes_the_template_up(
    ctx: Context, store: SqliteStore
) -> None:
    _pool(ctx, store)
    ref = "cross_iteration_software_pipelining@1"
    assert not template_availability(store, ["warpgroup_mma"], "h100").available
    decl = {"target": "h100", "capability": "warpgroup_mma", "status": "yes", "basis": "documented"}
    store.put(Record.new("CapabilityDecl", decl, HUMAN, T0))
    assert ref not in unavailable_templates(store, "h100")["unavailable"]
    assert template_availability(store, ["warpgroup_mma"], "h100").available


def test_capability_is_inferred_from_an_isa_probe(ctx: Context, store: SqliteStore) -> None:
    seed(ctx, store)
    feature = {"name": "wgmma", "unit": "tensor", "description": "d", "support": {"lab": "yes"}}
    store.put(Record.new("HwFeature", feature, TOOL, T0))
    state = target_capabilities(store, "lab")["warpgroup_mma"]
    assert (state.status, state.basis) == ("yes", "inferred")


def test_requires_must_name_a_capability(ctx: Context, store: SqliteStore) -> None:
    seed(ctx, store)
    bad = template("multi_buffer_rotation", requires=["no_such_capability"])
    assert "unknown_hw_feature" in {i.kind for i in errors_of(validate_template(bad, store))}
    store.put(
        Record.new("HwFeature", {"name": "cp_async", "unit": "lsu", "description": "d"}, TOOL, T0)
    )
    legacy = template("multi_buffer_rotation", requires=["cp_async"])
    issues = validate_template(legacy, store)
    assert errors_of(issues) == []
    assert "hw_feature_not_capability" in {i.kind for i in issues}


# ---- findings and couplings (D444, D447) ---------------------------------------------


def _finding(
    store: SqliteStore,
    claim: str,
    targets: list[str],
    subject: str = "multi_buffer_rotation",
    **extra: Any,
) -> str:
    body = {
        "claim": claim,
        "kind": "regime",
        "subject": subject,
        "scope": {"targets": targets},
        "basis": "measured",
        **extra,
    }
    return store.put(Record.new("Finding", body, TOOL, T0))


def test_legacy_regime_and_negative_read_as_findings(store: SqliteStore) -> None:
    store.put(
        Record.new(
            "Negative",
            {"subject": "async_transfer", "condition": "decode", "effect": "no gain"},
            TOOL,
            T0,
        )
    )
    store.put(
        Record.new(
            "Regime",
            {"structure": "monotone", "context": "gemm", "holds": True, "how": "proved"},
            TOOL,
            T0,
        )
    )
    got = {f.body.kind: f for f in findings(store)}
    assert got["negative"].legacy and got["negative"].body.claim == "no gain"
    assert got["regime"].body.basis == "proved" and got["regime"].body.subject == "monotone"


def test_a_finding_from_another_chip_is_a_prior(store: SqliteStore) -> None:
    _finding(store, "split >= 4 saturates DRAM", ["gb10"])
    assert findings(store, target="gb10")[0].status == "proposed"
    assert findings(store, target="h100")[0].status == PRIOR


def test_couplings_filter_by_member_and_target(store: SqliteStore) -> None:
    body = {
        "members": [{"kind": "variable", "name": "BK"}, {"kind": "variable", "name": "S"}],
        "scope": {"targets": ["gb10"]},
        "effect": "joint speed-up exceeds the product of single ones",
        "basis": "measured",
    }
    store.put(Record.new("Coupling", body, TOOL, T0))
    assert len(couplings(store, member="BK", target="gb10")) == 1
    assert couplings(store, member="BK", target="h100") == []
    assert couplings(store, member="BM") == []
    with pytest.raises(ValueError):
        Record.new("Coupling", {**body, "members": body["members"][:1]}, TOOL, T0)


# ---- the ladder, promotion and demotion (D446, D448) ---------------------------------


def _evidence(store: SqliteStore, n: int, **data: Any) -> list[str]:
    return [
        store.put(
            Record.new(
                "Report",
                {"title": "ablation", "subject": f"run {i}", "summary": "s", "data": data},
                TOOL,
                T0,
            )
        )
        for i in range(n)
    ]


def test_promotion_needs_the_evidence_of_its_step(store: SqliteStore, clock: ManualClock) -> None:
    fid = _finding(store, "claim", ["gb10"])
    with pytest.raises(PermissionDenied, match="needs 2 real evidence"):
        promote(store, fid, "replicated", _evidence(store, 1), clock.now(), CONTROLS)
    fixture = store.put(
        Record.new(
            "Report", {"title": "t", "subject": "s", "summary": "s"}, TOOL, T0, origin="fixture"
        )
    )
    with pytest.raises(PermissionDenied, match="origin fixture"):
        promote(store, fid, "reproduced", [*_evidence(store, 1), fixture], clock.now(), CONTROLS)
    promote(store, fid, "replicated", _evidence(store, 2), clock.now(), CONTROLS)
    assert store.status(fid)[0] == "reproduced"
    with pytest.raises(PermissionDenied, match="explained_by"):
        promote(store, fid, "explained", [], clock.now(), CONTROLS)
    promote(
        store,
        fid,
        "calibrated",
        _evidence(store, 1, ci_low=0.1, ci_high=0.2),
        clock.now(),
        CONTROLS,
    )
    with pytest.raises(PermissionDenied, match="hold-out"):
        promote(
            store,
            fid,
            "holdout_validated",
            _evidence(store, 1, holdout=True, agrees=False),
            clock.now(),
            CONTROLS,
        )
    promote(
        store,
        fid,
        "holdout_validated",
        _evidence(store, 1, holdout=True, agrees=True),
        clock.now(),
        CONTROLS,
    )
    assert store.status(fid)[0] == "holdout_validated"


def test_templates_keep_assurance_and_are_not_on_the_ladder(
    ctx: Context, store: SqliteStore, clock: ManualClock
) -> None:
    seed(ctx, store)
    tid = store.query("Template", limit=1)[0]
    with pytest.raises(PermissionDenied, match="assurance"):
        promote(store, record_id(tid), "reproduced", _evidence(store, 2), clock.now(), CONTROLS)


def test_demotion_steps_down_and_propagates_stale(store: SqliteStore, clock: ManualClock) -> None:
    fid = _finding(store, "claim", ["gb10"])
    promote(store, fid, "reproduced", _evidence(store, 2), clock.now(), CONTROLS)
    dependent = _finding(store, "builds on it", ["gb10"], subject="other")
    # A record resting on the finding (refs.evidence) must be re-verified when it falls.
    rest = store.put(
        Record.new(
            "Coupling",
            {
                "members": [{"kind": "template", "name": "a"}, {"kind": "template", "name": "b"}],
                "scope": {},
                "effect": "e",
                "basis": "observed",
            },
            TOOL,
            T0,
            refs={"evidence": [fid]},
        )
    )
    with pytest.raises(PermissionDenied, match="counter-evidence"):
        demote(store, fid, [], "no reason", clock.now())
    d = demote(store, fid, _evidence(store, 1), "counter-example on decode", clock.now())
    assert (d.from_status, d.to_status) == ("reproduced", "proposed")
    assert store.status(fid)[0] == "proposed"
    assert rest in d.stale and store.status(rest)[0] == "stale"
    assert store.status(dependent)[0] == "proposed"  # unrelated records untouched
    bottom = demote(store, fid, _evidence(store, 1), "again", clock.now())
    assert bottom.to_status is None and store.status(fid)[0] == "proposed"  # waits for a human


def test_only_the_system_demotes_and_never_up(store: SqliteStore) -> None:
    from kta.core.schemas.process import StatusChange

    fid = _finding(store, "claim", [])
    checker = Actor(type="checker", name="c")
    system = Actor(type="system", name="s")
    store.put(
        Record.new(
            "StatusChange",
            StatusChange(subject=fid, to_status="calibrated", reason="r"),
            checker,
            T0,
        )
    )
    with pytest.raises(PermissionDenied):
        store.put(
            Record.new(
                "StatusChange",
                StatusChange(subject=fid, to_status="holdout_validated", reason="r"),
                system,
                T0,
            )
        )
    store.put(
        Record.new(
            "StatusChange",
            StatusChange(subject=fid, to_status="reproduced", reason="r"),
            system,
            T0,
        )
    )
    assert store.status(fid)[0] == "reproduced"


def test_review_proposes_promotions_out_of_the_finding_layer(
    store: SqliteStore, clock: ManualClock
) -> None:
    for target in ("gb10", "h100"):
        fid = _finding(store, f"holds on {target}", [target])
        promote(store, fid, "reproduced", _evidence(store, 2), clock.now(), CONTROLS)
    for i in range(3):
        gap = {
            "missing": "pattern",
            "description": f"gap {i}",
            "pattern_guess": "scan",
            "attribution": "space",
        }
        store.put(Record.new("Gap", gap, TOOL, T0))
    data = store.get(review(store, clock.now(), CONTROLS)).body["data"]
    assert data["into_scope"][0]["targets"] == ["gb10", "h100"]
    assert data["into_scope"][0]["into"] == "structure_regime"
    assert data["new_patterns"] == [
        {"pattern_guess": "scan", "gaps": 3, "into": "new_pattern_or_template"}
    ]
    assert data["into_model"] == []
    assert data["thresholds"] == {"min_targets_for_scope": 2, "min_repeats_for_pattern": 3}
    stricter = CONTROLS.with_changes(
        {"knowledge.promotion_review": {"min_targets_for_scope": 3, "min_repeats_for_pattern": 4}}
    )
    data = store.get(review(store, clock.now(), stricter)).body["data"]
    assert data["into_scope"] == [] and data["new_patterns"] == []


# ---- pool diff (D449) --------------------------------------------------------------


def test_pool_diff_between_two_snapshots(
    ctx: Context, store: SqliteStore, ops: SqliteOps, clock: ManualClock
) -> None:
    seed(ctx, store)
    members = [record_id(r) for k in ("Pattern", "Template") for r in store.query(k, limit=1000)]
    fid = _finding(store, "claim", ["gb10"])
    a = snapshot(store, ops, [*members, fid], None, "a", HUMAN, clock.now())
    clock.advance(60)
    old = store.query("Template", {"name": "async_transfer"}, limit=1)[0]
    newer = {**old.body, "version": 2, "summary": old.body["summary"] + " Revised."}
    t2 = store.put(
        Record.new("Template", newer, HUMAN, clock.now(), refs={"previous": record_id(old)})
    )
    promote(store, fid, "reproduced", _evidence(store, 2), clock.now(), CONTROLS)
    extra = store.put(
        Record.new(
            "Structure",
            {"name": "s", "effect": "variables", "description": "d", "evidence_needed": "e"},
            HUMAN,
            clock.now(),
        )
    )
    clock.advance(60)
    keep = [m for m in members if m != record_id(old)]
    b = snapshot(store, ops, [*keep, t2, fid, extra], a, "b", HUMAN, clock.now())
    d = diff(store, a, b)
    assert [(c.kind, c.label) for c in d.changed] == [("Template", "async_transfer@2")]
    assert [e.label for e in d.added] == ["s"] and d.removed == []
    assert [(s.old, s.new) for s in d.status] == [("proposed", "reproduced")]
    status, page = request(
        create_app(Api(ctx=ctx, store=store, ops=ops)), "GET", f"/pool/diff?a={a}&b={b}"
    )
    assert status == 200 and page["changed"][0]["new"] == t2
    status, _ = request(
        create_app(Api(ctx=ctx, store=store, ops=ops)), "GET", f"/pool/diff?a={fid}&b={b}"
    )
    assert status == 400  # not a PoolSnapshot


# ---- the seed migration (scripts/migrations/WP14_knowledge_v2.py) --------------------


def test_seed_migration_is_idempotent(tmp_path: Path) -> None:
    root = repo_root()
    script = root / "scripts" / "migrations" / "WP14_knowledge_v2.py"
    seed_copy = tmp_path / "seed"
    shutil.copytree(SEED, seed_copy)
    v1 = copy.deepcopy(_v1_template())
    v1["requires_hw"] = ["wgmma"]
    (seed_copy / "templates" / "old_ring.yaml").write_text(
        yaml.safe_dump(v1 | {"name": "old_ring"})
    )

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, str(script), "--seed", str(seed_copy), *args]
        return subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)

    assert run("--check").returncode == 1
    assert run().returncode == 0
    migrated = yaml.safe_load((seed_copy / "templates" / "old_ring.yaml").read_text())
    assert migrated["requires"] == ["warpgroup_mma"] and migrated["level_from"] == 4
    realisation = yaml.safe_load((seed_copy / "realisations" / "old_ring.yaml").read_text())
    assert realisation["requires"] == ["wgmma"] and realisation["isa"]["min"] == "90a"
    assert run("--check").returncode == 0
    second = run()
    assert "nothing to do" in second.stdout
    assert run("--check", "--seed", str(SEED)).returncode == 0  # the repository seed is migrated
