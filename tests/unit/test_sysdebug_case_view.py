"""The real solve loop links cached artifacts through round data, not Build->Candidate."""

from pathlib import Path

from apihelp import put, seeded_api
from kta.api import cases, opsread
from kta.api.common import incoming


def test_case_view_uses_round_links_and_keeps_baselines_in_their_bucket(tmp_path: Path) -> None:
    api = seeded_api(tmp_path)
    case = next(c for c in cases.cases(api) if c["name"] == "vadd_f32")
    lp = next(lp for lp in opsread.loops(api.ops) if lp.subject == case["id"])
    old = cases.timeline(api, case["id"])["candidates"][0]
    source = api.store.get(old["record"])
    cid = put(api, "Candidate", {**source.body, "bucket": "s0"}, dict(source.refs))
    bid = incoming(api, old["record"], "candidate", "Build")[0]
    build = api.store.get(bid)
    bid = put(
        api,
        "Build",
        {**build.body, "log_summary": "cache hit"},
        {k: v for k, v in build.refs.items() if k != "candidate"},
    )
    check = incoming(api, bid, "build", "Check")
    assert not check
    chk = put(
        api,
        "Check",
        {"passed": True},
        {"build": bid, "env": build.refs["env"], "worker": build.refs["worker"]},
    )
    previous = api.store.get(incoming(api, case["id"], "case", "Measurement")[0])
    mid = put(
        api, "Measurement", {**previous.body, "median_us": 55.0}, {**previous.refs, "build": bid}
    )
    baseline = put(api, "Report", {"title": "baseline", "subject": "baseline", "summary": "torch"})
    base_refs = {k: v for k, v in previous.refs.items() if k not in ("build", "baseline")}
    m0 = put(
        api,
        "Measurement",
        {**previous.body, "median_us": 100.0},
        {**base_refs, "baseline": baseline},
    )
    m1 = put(
        api,
        "Measurement",
        {**previous.body, "median_us": 60.0},
        {**base_refs, "baseline": baseline},
    )
    bound = case["bound_record"]
    d = {
        **lp.data,
        "candidates": {"s0": [cid]},
        "top": {"s0": [cid]},
        "builds": {cid: bid},
        "checks": {cid: chk},
        "measurements": {cid: mid},
        "bounds": {"s0": bound, "s1": bound},
        "baseline_measurements": {"s0": m0, "s1": m1},
    }
    api.ops.put_loop(lp.id, lp.kind, lp.subject, lp.state, d)
    view = cases.timeline(api, case["id"])
    assert view["bound_us"] is None  # one shape's bound must not label the entire case
    assert len(view["candidates"]) == 1
    candidate = view["candidates"][0]
    assert candidate["bucket"] == "s0" and candidate["selected"]
    assert candidate["build_record"] == bid and candidate["check_passed"]
    assert candidate["measured_us"] == 55.0
    assert [b["references"] for b in view["buckets"]] == [{"baseline": 100.0}, {"baseline": 60.0}]

    api.ops.put_loop(lp.id + "NEXT", lp.kind, lp.subject, "checking", {**d, "round": 2})
    assert cases.timeline(api, case["id"])["round"] == 2
    assert cases.timeline(api, case["id"])["state"] == "checking"
