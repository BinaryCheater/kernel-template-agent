"""WP7 specific checks (spec/gate.md): the first real chain, FP16 GEMM (decision D288).

Static, always:
- the named WP7 unit tests passed;
- the seed family ``gemm_fp16_tilelang`` validates (markers for every chain item) and its
  chain uses the five seed GEMM Templates plus the two contraction-cost Templates;
- the Case file has M=N=K=4096 plus two non-square shapes, the FP16 contract of
  docs/design/50 §3 and the torch.matmul baseline.

Records in ``$KTA_DATA/dev/WP7/kta.db`` listed in the evidence manifest, real, written by a
clean commit in this branch's history (pending real runs in the code stage, D320):
- a solve-loop judgement (Report titled ``judgement``) or an accepted Solution;
- the three fault injections caught: the ``wp7_faults`` Report says so and its three
  Check records are listed, failed, and ran on a GPU worker;
- a replay Report (subject ``replay``) whose outputs agree;
- a measured HwModel, a baseline Measurement and the ``wp7_reuse`` Report as expected.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import yaml

from kta_gate.result import CheckResult, from_findings

WP7_UNIT_TESTS = (
    "test_family_validates_with_full_markers",
    "test_generator_launch_spec",
    "test_formulation_costs_contraction_and_traffic",
    "test_wrong_legality_admits_racy_split",
    "test_reuse_plan_marks_only_the_instruction_step",
    "test_peak_kernels_count_work_per_unit",
    "test_tilelang_launch_mismatch_grid_and_dynamic_smem",
)
FAMILY = Path("knowledge") / "seed" / "families" / "gemm_fp16_tilelang"
CASE = Path("knowledge") / "seed" / "cases" / "gemm_fp16.yaml"
CHAIN_TEMPLATES = (
    "reduction_partition",
    "physical_storage",
    "async_transfer",
    "multi_buffer_rotation",
    "shared_resource_binding",
    "output_tiling",
    "compute_instruction_selection",
)
WP7_CODE = "tools/kta_wp7/pipeline.py"
FAULTS = ("oob_write", "racy_legality", "cublas")


def family_findings(root: Path, data: Path) -> list[str]:
    """The seed family validates in a scratch store with the seed imported."""
    from kta.core.config import config_from_mapping
    from kta.core.context import Context
    from kta.core.registry import Registry
    from kta.knowledge import SeedError, errors_of, import_seed, load_family, validate_family
    from kta.store import CodeState, open_store

    path = root / FAMILY
    if not path.is_dir():
        return [f"{FAMILY} missing"]
    ctx = Context(env="test", config=config_from_mapping({}), data_dir=data, registry=Registry())
    code = CodeState("0" * 40, False)
    store = open_store(ctx, code)
    try:
        import_seed(ctx, store, root / "knowledge" / "seed", code)
    except SeedError as exc:
        return [f"seed import refused: {exc}"]
    fp = load_family(path)
    out = [str(i) for i in errors_of(validate_family(fp, store, None))]
    used = {item.template.split("@")[0] for item in fp.chain.items}
    out += [f"chain does not use {t}" for t in CHAIN_TEMPLATES if t not in used]
    if fp.family.backend != "tilelang":
        out.append(f"backend is {fp.family.backend}, not tilelang")
    return out


def case_findings(root: Path) -> list[str]:
    path = root / CASE
    if not path.is_file():
        return [f"{CASE} missing"]
    doc: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    shapes = [s["shape"] for s in doc["case"]["shape_dist"]]
    out: list[str] = []
    if {"M": 4096, "N": 4096, "K": 4096} not in shapes:
        out.append("no M=N=K=4096 shape")
    if sum(1 for s in shapes if len({s["M"], s["N"], s["K"]}) > 1) < 2:
        out.append("fewer than two non-square shapes")
    tol = doc["contract"]["tolerances"].get("f16", {})
    if tol.get("rtol") != 0.01 or tol.get("atol") != 0.01 or "rel_l2" not in tol:
        out.append("FP16 tolerances are not rtol = atol = 1e-2 plus rel_l2 (docs/design/50 §3)")
    dists = set(doc["contract"]["input_dists"])
    out += [
        f"input distribution {d} missing" for d in ("normal", "zeros", "extreme") if d not in dists
    ]
    edges = doc["contract"]["edge_shapes"]
    if not any(min(e.values()) == 1 for e in edges):
        out.append("no length-1 edge shape")
    if not any(any(v % 64 for v in e.values()) and min(e.values()) > 1 for e in edges):
        out.append("no non-tile-multiple edge shape")
    if "torch.matmul" not in str(doc.get("baseline", {}).get("call", "")):
        out.append("baseline is not torch.matmul")
    return out


def _clean(con: sqlite3.Connection, git_root: Path, rec: dict[str, Any]) -> bool:
    from kta_gate.specific import _git_ok

    commit = str(rec["actor"].get("commit"))
    return (
        rec["_origin"] == "real"
        and rec["actor"].get("dirty") is False
        and _git_ok(git_root, "merge-base", "--is-ancestor", commit, "HEAD")
        and _git_ok(git_root, "cat-file", "-e", f"{commit}:{WP7_CODE}")
    )


def _on_gpu(con: sqlite3.Connection, rec: dict[str, Any]) -> bool:
    from kta_gate.specific import _record

    worker = _record(con, rec["refs"].get("worker"))
    return worker is not None and "gpu" in worker["body"].get("labels", {})


def _reports(good: list[dict[str, Any]], key: str, value: str) -> list[dict[str, Any]]:
    return [r for r in good if r["kind"] == "Report" and r["body"].get(key) == value]


def record_findings(git_root: Path, con: sqlite3.Connection, listed: set[str]) -> list[str]:
    from kta_gate.specific import _record

    good: list[dict[str, Any]] = []
    for rid in sorted(listed):
        rec = _record(con, rid)
        if rec is not None and _clean(con, git_root, rec):
            good.append({**rec, "_id": rid})
    ids = {r["_id"] for r in good}
    out: list[str] = []
    if not _reports(good, "title", "judgement") and not any(r["kind"] == "Solution" for r in good):
        out.append("no listed solve-loop judgement Report or Solution")
    faults = _reports(good, "subject", "wp7_faults")
    if not faults:
        out.append("no listed wp7_faults Report")
    for name in FAULTS:
        data = faults[-1]["body"]["data"].get(name, {}) if faults else {}
        chk = _record(con, data.get("check")) if data else None
        if not data.get("caught"):
            out.append(f"fault {name} not caught")
        elif data["check"] not in ids or chk is None or chk["body"].get("passed") is not False:
            out.append(f"fault {name}: its failed Check is not listed")
        elif not _on_gpu(con, chk):
            out.append(f"fault {name}: Check did not run on a GPU worker")
    replays = _reports(good, "subject", "replay")
    if not any(r["body"]["data"].get("consistent") is True for r in replays):
        out.append("no listed replay Report with consistent outputs")
    if not any(r["kind"] == "HwModel" for r in good):
        out.append("no listed HwModel")
    if not any(r["kind"] == "Measurement" and "baseline" in r["refs"] for r in good):
        out.append("no listed baseline Measurement")
    reuse = _reports(good, "subject", "wp7_reuse")
    if not any(r["body"]["data"].get("bf16", {}).get("as_expected") for r in reuse):
        out.append("no listed wp7_reuse Report marking exactly the BF16-invalid steps")
    return out


def wp7(
    root: Path, passed: set[str], data_root: Path | None = None, git_root: Path | None = None
) -> list[CheckResult]:
    from kta_gate.specific import _listed, code_stage, pending_real

    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP7_UNIT_TESTS if t not in names]
    results = [from_findings("WP7 unit tests", absent, f"{len(WP7_UNIT_TESTS)} named tests")]
    with tempfile.TemporaryDirectory(prefix="kta-gate-wp7-") as tmp:
        results.append(from_findings("WP7 family", family_findings(root, Path(tmp)), str(FAMILY)))
    results.append(from_findings("WP7 case", case_findings(root), str(CASE)))
    if code_stage():
        return [*results, pending_real("WP7 records", "solve loop, faults, replay, baseline")]
    if data_root is None and not os.environ.get("KTA_DATA"):
        return [*results, CheckResult("WP7 records", "FAIL", "set KTA_DATA")]
    db = (data_root or Path(os.environ["KTA_DATA"])) / "dev" / "WP7" / "kta.db"
    if not db.exists():
        return [*results, CheckResult("WP7 records", "FAIL", f"{db} not found")]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        findings = record_findings(git_root or root, con, _listed(root, "WP7"))
    finally:
        con.close()
    detail = "judgement, three caught faults, replay, HwModel, baseline, reuse plan"
    return [*results, from_findings("WP7 records", findings, detail)]
