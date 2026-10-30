"""Γ evaluation, reuse plans, retrieval, cards, coverage and rendering (kta.knowledge)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from knowhelp import put, put_template, seed, source_id, template
from kta.core.context import Context
from kta.core.schemas.evaluation import Case
from kta.knowledge import (
    card,
    coverage_report,
    diff_templates,
    family_overview,
    nearest_chains,
    render_text_knowledge,
    retrieve_families,
    retrieve_templates,
    reuse_plan,
    scope_holds,
)
from kta.knowledge.coverage import chao1, chao1_interval, coverage_page
from kta.knowledge.pool import find_template
from kta.store.base import Store
from storehelp import status_change

PARAMS = {
    "assoc": True,
    "comm": True,
    "iterations": 8,
    "size_bytes": 4096,
    "align_bytes": 16,
    "length": 4096,
}
HW = {"smem_banks": 32, "smem_bank_bytes": 4, "smem_queue_depth": 8}


def case(*shapes: dict[str, Any], problem: str = "gemm_fp16") -> Case:
    dist = [{"shape": s, "weight": 1.0} for s in shapes]
    return Case(problem=problem, shape_dist=dist, objective="weighted_latency", target="gb10")


@pytest.fixture
def pool(ctx: Context, store: Store) -> Store:
    seed(ctx, store)
    put(
        store,
        "Problem",
        {"name": "gemm_fp16", "reference_impl": "torch.matmul fp16 gemm", "params": PARAMS},
    )
    return store


def item(
    key: str, tref: str, bindings: dict[str, Any], after: list[str] | None = None
) -> dict[str, Any]:
    return {
        "key": key,
        "template": tref,
        "applies_to": key,
        "bindings": bindings,
        "after": after or [],
    }


def put_chain(store: Store, workload: str, items: list[dict[str, Any]]) -> str:
    ids = sorted({hit[0] for i in items if (hit := find_template(store, i["template"]))})
    return put(store, "Chain", {"workload": workload, "items": items}, {"templates": ids})


GEMM_ITEMS = [
    item(
        "k_split",
        "reduction_partition@1",
        {"num_groups": 128, "group_size": 32, "assignment": "contiguous_sequential"},
    ),
    item(
        "ab_smem",
        "physical_storage@1",
        {"level": "smem", "buffer": "dedicated", "share_order": "none"},
    ),
    item(
        "ab_async",
        "async_transfer@1",
        {
            "mechanism": "instruction",
            "issue_point": "prologue",
            "complete_point": "before_read",
            "granularity_bytes": 16,
            "p.align_bytes": 2,
        },
    ),
    item("ab_stages", "multi_buffer_rotation@1", {"S": 3, "P": 1, "beta": 1}),
    item("ab_banks", "shared_resource_binding@1", {"binding": "xor_swizzle", "queue_depth": 8}),
]


def test_scope_holds_per_shape(pool: Store) -> None:
    t = template("multi_buffer_rotation", scope={"predicate": "case.M >= 16", "text": "t"})
    assert scope_holds(t, case({"M": 64}, {"M": 32}), {}, {}).holds is True
    result = scope_holds(t, case({"M": 64}, {"M": 8}), {}, {})
    assert result.holds is False and "'M': 8" in result.reason
    assert scope_holds(t, case({"N": 64}), {}, {}).holds is None


def test_reuse_plan_marks_steps(pool: Store) -> None:
    src = source_id(pool)
    sync_copy = template(
        "async_transfer",
        name="sync_copy",
        scope={"predicate": "True", "text": "any"},
        relations=[{"type": "alternative_of", "target": "async_transfer"}],
    )
    put_template(pool, sync_copy, [src])
    chain_id = put_chain(pool, "fp16 gemm", GEMM_ITEMS)
    plan = reuse_plan(pool, chain_id, case({"M": 64}), {"smem_banks": 32})
    verdicts = {s.key: s.verdict for s in plan.steps}
    assert verdicts == {
        "k_split": "keep",
        "ab_smem": "keep",
        "ab_async": "invalid",
        "ab_stages": "keep",
        "ab_banks": "keep",
    }
    assert plan.by_verdict("invalid")[0].alternatives == ["sync_copy@1"]
    no_hw = reuse_plan(pool, chain_id, case({"M": 64}), {})
    assert {s.key for s in no_hw.by_verdict("unknown")} == {"ab_banks"}


def test_retrieve_templates_filters_and_ranks(pool: Store) -> None:
    names = [c.name for c in retrieve_templates(pool, pattern="store")]
    assert sorted(names) == ["multi_buffer_rotation@1", "physical_storage@1"]
    by_unit = [c.name for c in retrieve_templates(pool, unit="dram")]  # cost or resources
    assert sorted(by_unit) == ["async_transfer@1", "multi_buffer_rotation@1", "output_tiling@2"]
    assert [c.name for c in retrieve_templates(pool, text="slots rotate")] == [
        "multi_buffer_rotation@1"
    ]
    cards = retrieve_templates(pool, case=case({"M": 64}), hw={})
    unknown = [c.name for c in cards if c.scope_unknown]
    assert "shared_resource_binding@1" in unknown and cards[-1].scope_unknown
    assert all(len(c.lines) <= 5 for c in cards)
    rid = find_template(pool, "physical_storage@1")
    assert rid is not None
    pool.put(status_change(rid[0], "human", to_status="rejected"))
    assert "physical_storage@1" not in [c.name for c in retrieve_templates(pool)]


def test_required_hw_feature_must_be_yes_on_target(pool: Store) -> None:
    src = source_id(pool)
    for feature, support in (("cp_async", "yes"), ("tmem", "no")):
        put(
            pool,
            "HwFeature",
            {"name": feature, "unit": "smem", "description": feature, "support": {"gb10": support}},
        )
        t = template("async_transfer", name=f"copy_{feature}", requires_hw=[feature], relations=[])
        put_template(pool, t, [src])
    found = [c.name for c in retrieve_templates(pool, pattern="transfer", case=case({"M": 1}))]
    assert "copy_cp_async@1" in found and "copy_tmem@1" not in found


def test_card_nearest_and_diff(pool: Store) -> None:
    full = put_chain(pool, "fp16 gemm with smem pipeline", GEMM_ITEMS)
    small = put_chain(pool, "rmsnorm row reduction", GEMM_ITEMS[:2])
    near = nearest_chains(pool, case({"M": 1}), k=2)
    assert [c.id for c in near] == [full, small]
    diff = [c.name for c in diff_templates(pool, full, small)]
    assert diff == ["async_transfer@1", "multi_buffer_rotation@1", "shared_resource_binding@1"]
    c = card(pool, full)
    assert c.kind == "Chain" and len(c.lines) <= 5
    hit = find_template(pool, "reduction_partition@1")
    assert hit is not None
    tcard = card(pool, hit[0])
    assert tcard.lines[1].startswith("author:") and "used in 2 chain(s)" in tcard.text()


def test_retrieve_families_excludes_false_scope(pool: Store) -> None:
    good = put_chain(pool, "fp16 gemm", [GEMM_ITEMS[0], GEMM_ITEMS[3]])
    bad = put_chain(pool, "fp16 gemm misaligned", GEMM_ITEMS[:3])
    for name, chain in (("gemm_ok", good), ("gemm_bad", bad)):
        put(
            pool,
            "Family",
            {"name": name, "version": 1, "backend": "tilelang", "entry": "gen.py:generate"},
            {"chain": chain},
        )
    assert [c.name for c in retrieve_families(pool, case({"M": 64}), hw=HW)] == ["gemm_ok@1"]


def _recon(store: Store, chain: str, name: str, matched: int, unmatched: int) -> str:
    src = source_id(store)
    ref = put(
        store, "Reference", {"name": name, "problem": "gemm_fp16", "build": "make"}, {"source": src}
    )
    txs = [
        {"level": 2, "pattern_guess": "map", "description": f"tx{i}", "where": "k.cu"}
        for i in range(unmatched)
    ]
    return put(
        store,
        "Reconstruction",
        {
            "matched": matched,
            "total": matched + unmatched,
            "unmatched_transactions": txs,
            "notes": name,
        },
        {"reference": ref, "chain": chain},
    )


def test_chao1_formula() -> None:
    assert chao1(10, 4, 2) == 14.0
    assert chao1(3, 2, 0) == 4.0  # bias-corrected form when f2 = 0


def test_coverage_report(pool: Store) -> None:
    chains = [
        put_chain(pool, "a", [GEMM_ITEMS[0], GEMM_ITEMS[1]]),
        put_chain(pool, "b", [GEMM_ITEMS[0], GEMM_ITEMS[3]]),
        put_chain(pool, "c", [GEMM_ITEMS[0]]),
    ]
    for i, ch in enumerate(chains):
        _recon(pool, ch, f"ref{i}", matched=2, unmatched=i)
    put(
        pool,
        "Family",
        {"name": "f", "version": 1, "backend": "cuda", "entry": "gen.py:g"},
        {"chain": chains[0]},
    )
    rep = coverage_report(pool, None, random.Random(1))
    assert rep.reconstructions == 3 and rep.discovery_observed == [2, 3, 3]
    assert (rep.templates_observed, rep.singletons, rep.doubletons) == (3, 2, 0)
    assert rep.chao1 == 4.0 and rep.good_turing == pytest.approx(1 - 2 / 5)
    assert len(rep.discovery_mean) == 3 and rep.discovery_mean[-1] == 3
    assert rep.unmatched_by_level == {str(n): 3 if n == 2 else 0 for n in range(1, 8)}
    assert rep.unmatched_by_pattern == {"map": 3}
    assert rep.assurance == {"declared": 7, "tested": 0, "bounded": 0, "proved": 0}
    assert rep.other_patterns == 0
    assert rep.hw_pattern_grid["none"]["reduce"] == {"declared": 1}
    fams = rep.families_per_template
    assert fams["reduction_partition"] == 1 and fams["physical_storage"] == 1
    assert len(fams) == 7 and sum(fams.values()) == 2  # every Template, 0 when in no family
    assert rep.direct_test == {"references": 3, "all_steps_matched": 1, "nothing_unexplained": 1}
    assert rep.gaps_by_missing["template"] == 0 and rep.templates_missing == []
    assert set(rep.pending) >= {"instruction_coverage", "gap_explained"}
    again = coverage_report(pool, None, random.Random(1))
    assert again.discovery_mean == rep.discovery_mean
    snap = put(pool, "PoolSnapshot", {"members": [], "note": "empty"})
    assert coverage_report(pool, snap, random.Random(1)).reconstructions == 0


def test_coverage_counts_the_latest_reconstruction_per_reference(pool: Store) -> None:
    first = put_chain(pool, "a", [GEMM_ITEMS[0], GEMM_ITEMS[1]])
    second = put_chain(pool, "a2", [GEMM_ITEMS[0]])
    ref_id = pool.get(_recon(pool, first, "ref", matched=2, unmatched=1)).refs["reference"]
    put(
        pool,
        "Reconstruction",
        {"matched": 1, "total": 1, "notes": "re-match"},
        {"reference": str(ref_id), "chain": second},
    )
    rep = coverage_report(pool, None, random.Random(1))
    assert rep.reconstructions == 1 and rep.discovery_observed == [1]
    assert rep.references[0]["unmatched"] == 0


def test_chao1_interval_and_page_fields(pool: Store) -> None:
    lo, hi = chao1_interval(10, 4, 2)
    assert lo >= 10 and lo < chao1(10, 4, 2) < hi
    assert chao1_interval(5, 0, 3) == (5.0, 5.0)
    _recon(pool, put_chain(pool, "a", [GEMM_ITEMS[0]]), "ref", matched=1, unmatched=1)
    rep = coverage_report(pool, None, random.Random(1))
    ref_id = str(rep.references[0]["reference_id"])
    page = coverage_page(
        rep, {ref_id: "covered"}, [{"snapshot": "s0", "good_turing": 0.1, "expressible": 0.0}]
    )
    want = {"scope", "discovery", "trend", "instruction_families", "hw_patterns"}
    want |= {"references", "gap_explained", "backlog", "structure"}
    assert want <= set(page)
    assert page["references"] == [
        {
            "name": "ref",
            "state": "covered",
            "record": rep.references[0]["reconstruction"],
            "gaps": 1,
        }
    ]
    assert [t["snapshot"] for t in page["trend"]] == ["s0", "whole store"]
    cells = page["hw_patterns"]["cells"]
    assert cells and all(c["n"] >= 0 for c in cells)


def test_render_text_knowledge_and_overview(pool: Store, tmp_path: Path) -> None:
    render_text_knowledge(pool, None, tmp_path)
    page = (tmp_path / "patterns" / "store.md").read_text(encoding="utf-8")
    assert "multi_buffer_rotation@1" in page and "physical_storage@1" in page
    assert "bottleneck_dominance" in (tmp_path / "structures.md").read_text(encoding="utf-8")
    assert "(patterns/reduce.md)" in (tmp_path / "index.md").read_text(encoding="utf-8")
    overview = family_overview(pool, "store")
    assert overview.startswith("# Pattern store") and "## Families" in overview
    assert family_overview(pool, "gemm_fp16").startswith("# Problem gemm_fp16")
