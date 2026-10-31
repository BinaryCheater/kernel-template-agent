"""WP22: GET /mechanisms shows the round-4 mechanisms' records with their origin."""

from __future__ import annotations

from pathlib import Path

from apihelp import put, request, seeded_api
from kta.api.app import create_app


def test_mechanisms_view_lists_each_mechanism_with_origin(tmp_path: Path) -> None:
    api = seeded_api(tmp_path / "d")
    plan = {"family": "gemm@1", "author": "default", "chain_source": "family", "chain": []}
    default = put(api, "FormulationPlan", plan)
    agent = put(api, "FormulationPlan", {**plan, "author": "agent", "generator": "g1"},
                {"default": default})  # fmt: skip
    put(api, "Report", {"title": "what_if", "subject": "hw@1", "summary": "wall smem=2: 3 us",
                        "data": {"walls": {"smem": 2}, "bottleneck": "dram",
                                 "shadow_price_us": 3.0, "delta_total_us": -3.0}})  # fmt: skip
    for batch, verdict in (("b1", "caught"), ("b2", "missed"), ("b3", "caught")):
        data = {"kind": "slow_input", "expect": "noise", "verdict": verdict}
        put(api, "Report", {"title": "canary", "subject": batch, "summary": verdict, "data": data})
    sel = {"model_top_e_us": 10.0, "random_best_us": 9.0, "random_n": 4, "copied_us": None,
           "model_beaten_by_random": True}  # fmt: skip
    put(api, "GapDecomposition", {"round": 2, "gap_class": "none", "route": {}, "routing": "r1",
                                  "buckets": [{"name": "main", "selection": sel}]})  # fmt: skip

    status, view = request(create_app(api), "GET", "/mechanisms", None)
    assert status == 200, view
    row = next(p for p in view["plans"] if p["id"] == agent)
    assert row["default"] == default and row["origin"] == "fixture"
    assert [(c["section"], c["key"]) for c in row["diff"]] == [("generator", "generator")]
    assert row["uncited"] == ["generator:generator"]
    (w,) = view["what_if"]
    assert w["shadow_price_us"] == 3.0 and w["walls"] == {"smem": 2}
    assert view["canaries"]["by_kind"] == {"slow_input": {"caught": 2, "missed": 1}}
    (s,) = view["selection"]
    assert s["bucket"] == "main" and s["model_beaten_by_random"] is True and s["round"] == 2
    assert view["stages"] == ["intake", "decompose", "admit", "reproduce", "ablate", "cover"]
    for ref in view["learning"]:  # the seed world's references: a plan or the reason why not
        assert ("stages" in ref) != ("error" in ref)
    assert view["experiments"]["count"] == 12 and view["experiments"]["page"] == "/settings"
