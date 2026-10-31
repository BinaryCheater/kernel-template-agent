"""Work-package specific checks (spec/gate.md, second table).

A work package whose specific checks are not implemented here fails the gate: the
WP that owns them adds them, so no check is silently skipped.
"""

from __future__ import annotations

import importlib
import io
import json
import math
import os
import re
import sqlite3
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import yaml

from kta_gate.delivery import split_sections
from kta_gate.result import CheckResult, from_findings
from kta_gate.specific_wp3 import wp3
from kta_gate.specific_wp7 import wp7
from kta_gate.specific_wp8 import wp8
from kta_gate.specific_wp9b import wp9b
from kta_gate.specific_wp10b import wp10b
from kta_gate.specific_wp11 import wp11
from kta_gate.specific_wp14 import wp14
from kta_gate.specific_wp15 import wp15
from kta_gate.specific_wp19 import wp19
from kta_gate.specific_wp21 import wp21

MODULES = (
    "core",
    "store",
    "targets",
    "execution",
    "backends",
    "knowledge",
    "models",
    "measure",
    "formulation",
    "checks",
    "solve",
    "orchestrator",
    "agents",
    "cli",
    "api",
)
WP0_VIOLATION_TESTS = (
    "test_gate_blocks_layer_violation",
    "test_gate_blocks_fake_class_in_src",
    "test_gate_blocks_bare_except",
    "test_gate_blocks_print_in_src",
    "test_gate_blocks_fake_api_key",
    "test_gate_blocks_fixture_impl_in_dev",
)
REFERENCE_KINDS = ("Reference", "Chain", "Reconstruction")
CITATION = re.compile(r"\[S\d+\]")
SOURCE_DEF = re.compile(r"^\s*[-*]?\s*\[(S\d+)\]:?\s+(\S+)")
MEASURED_NUMBER = re.compile(r"\d+\.\d+|\d+(?:\.\d+)?\s*(?:%|×|x\b|s\b|ms\b|us\b|µs\b|min\b|h\b)")


def wp0(root: Path, passed: set[str]) -> list[CheckResult]:
    missing = [
        f"src/kta/{m}/README.md"
        for m in MODULES
        if not (root / "src" / "kta" / m / "README.md").exists()
    ]
    names = {t.split("::")[-1] for t in passed}
    absent = [f"{t} did not pass" for t in WP0_VIOLATION_TESTS if t not in names]
    return [
        from_findings("module READMEs", missing, f"{len(MODULES)} modules"),
        from_findings(
            "gate violation samples", absent, f"{len(WP0_VIOLATION_TESTS)} samples blocked"
        ),
    ]


WP1_ACCEPTANCE = (
    "test_main_env_rejects_fixture_and_simulated_records",
    "test_main_env_rejects_dirty_writes",
    "test_registry_rejects_fixture_store_in_dev",
    "test_record_id_is_stable_across_processes",
    "test_update_and_delete_are_refused_by_triggers",
    "test_put_fails_when_refs_are_missing",
    "test_status_promotion_rules",
    "test_stale_propagation_on_five_level_graph",
    "test_concurrent_cas_pointer_update_has_one_winner",
)
BENCH_SUBJECT = "store-benchmark"
BENCH_MIN_RECORDS = 100_000
BENCH_P95_MS = 50.0
STORE_SOURCE = "src/kta/store/sqlite.py"


def _p95(values: list[float]) -> float:
    """Same rule as kta.store.bench: nearest rank on the sorted samples."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def _git_ok(root: Path, *args: str) -> bool:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    return proc.returncode == 0


def _bench_problems(
    root: Path, data: Path, rid: str, record: dict[str, Any], con: Any
) -> list[str]:
    """Why a store-benchmark Report does not count as WP1 evidence (empty when it does)."""
    problems: list[str] = []
    actor, body = record.get("actor", {}), record.get("body", {})
    commit = str(actor.get("commit"))
    if actor.get("dirty") is not False:
        problems.append("written by dirty code")
    # "Created during WP1": the code that ran contains the store and is in this branch.
    if not (
        _git_ok(root, "merge-base", "--is-ancestor", commit, "HEAD")
        and _git_ok(root, "cat-file", "-e", f"{commit}:{STORE_SOURCE}")
    ):
        problems.append(f"commit {commit[:12]} is not a WP1 commit in this branch")
    env_row = con.execute(
        "SELECT json FROM records WHERE id = ? AND kind = 'EnvManifest'",
        (record.get("refs", {}).get("env"),),
    ).fetchone()
    if env_row is None or json.loads(env_row[0])["body"].get("role") != "worker":
        problems.append("refs.env is not a worker EnvManifest (the run belongs on spark)")
    if body.get("data", {}).get("records_in_store", 0) < BENCH_MIN_RECORDS:
        problems.append(f"fewer than {BENCH_MIN_RECORDS} records in the benchmarked store")
    sha = str(record.get("blobs", {}).get("latencies_ms", ""))
    blob = data / "objects" / sha[:2] / sha[2:]
    if not sha or not blob.exists():
        return [*problems, "raw latencies blob is missing"]
    samples: dict[str, list[float]] = json.loads(blob.read_text(encoding="utf-8"))
    for name, values in sorted(samples.items()):
        p95 = _p95(values) if values else math.inf
        if not p95 < BENCH_P95_MS:
            problems.append(f"{name}: p95 {p95:.3f} ms >= {BENCH_P95_MS} ms")
    if not any(n.startswith("query") for n in samples) or not any(
        n.startswith("search") for n in samples
    ):
        problems.append("samples lack query or search timings")
    return [f"Report {rid[:12]}: {p}" for p in problems]


def wp1(
    root: Path, passed: set[str], data_root: Path | None = None, git_root: Path | None = None
) -> list[CheckResult]:
    """Acceptance tests passed; a real store-benchmark Report from spark is in the dev store.

    WP1 has no GPU work, so "pytest -m gpu tests/acceptance/WP1" is the acceptance run
    inside the unit tests plus this record check (decision D154).
    """
    names = {t.split("::")[-1].split("[")[0] for t in passed if "acceptance.WP1" in t}
    absent = [f"{t} did not pass" for t in WP1_ACCEPTANCE if t not in names]
    results = [from_findings("WP1 acceptance tests", absent, f"{len(names)} tests passed")]
    if data_root is None and not os.environ.get("KTA_DATA"):
        return [*results, CheckResult("store benchmark record", "FAIL", "set KTA_DATA")]
    data = (data_root or Path(os.environ["KTA_DATA"])) / "dev" / "WP1"
    db = data / "kta.db"
    if not db.exists():
        return [*results, CheckResult("store benchmark record", "FAIL", f"{db} not found")]
    evidence_path = root / "deliveries" / "WP1.evidence.json"
    listed: set[str] = set()
    if evidence_path.exists():
        for claim in json.loads(evidence_path.read_text(encoding="utf-8")):
            listed |= set(claim.get("record_ids", []))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT id, json FROM records WHERE kind = 'Report' AND origin = 'real' "
            "AND json_extract(json, '$.body.subject') = ?",
            (BENCH_SUBJECT,),
        ).fetchall()
        findings = {
            rid: _bench_problems(git_root or root, data, rid, json.loads(js), con)
            for rid, js in rows
        }
    finally:
        con.close()
    good = sorted(rid for rid, p in findings.items() if not p and rid in listed)
    if good:
        detail = f"Report {good[0][:12]} (listed in the evidence manifest)"
        return [*results, CheckResult("store benchmark record", "PASS", detail)]
    problems = [p for ps in findings.values() for p in ps] or [
        f"no real {BENCH_SUBJECT} Report in {db}"
    ]
    problems += [
        f"Report {r[:12]} passes but is not in WP1.evidence.json"
        for r, p in findings.items()
        if not p and r not in listed
    ]
    return [*results, CheckResult("store benchmark record", "FAIL", "no valid record", problems)]


WP2_ACCEPTANCE = (
    "test_spark_registers_as_gpu_worker_over_ssh",
    "test_probe_writes_target_facts_and_feature_support",
    "test_vector_add_on_both_backends_is_correct",
    "test_exclusive_job_defers_concurrent_shared_jobs",
    "test_gpu_is_usable_after_a_timed_out_kernel",
    "test_slai_cpu_worker_compiles_with_the_same_lock",
)
WP2_CODE = "src/kta/execution/executor.py"
WP2_KINDS = ("Target", "Build", "Worker", "EnvManifest", "Check")


def _listed(root: Path, wp: str) -> set[str]:
    path = root / "deliveries" / f"{wp}.evidence.json"
    if not path.exists():
        return set()
    return {
        r for c in json.loads(path.read_text(encoding="utf-8")) for r in c.get("record_ids", [])
    }


def _record(con: sqlite3.Connection, rid: object) -> dict[str, Any] | None:
    row = con.execute("SELECT origin, json FROM records WHERE id = ?", (str(rid),)).fetchone()
    if row is None:
        return None
    return {**json.loads(row[1]), "_origin": row[0]}


def _wp2_records(git_root: Path, con: sqlite3.Connection, listed: set[str]) -> list[str]:
    """Real, clean WP2 records of every required kind, listed in the evidence manifest.

    Every record must come from a clean commit of this branch that contains the
    Executor. The Target and a Worker must come from / be a GPU-labelled worker, and
    both backends need an ok Build whose vector-add Check passed on a GPU worker.
    """

    def clean(rec: dict[str, Any]) -> bool:
        commit = str(rec["actor"].get("commit"))
        return (
            rec["_origin"] == "real"
            and rec["actor"].get("dirty") is False
            and _git_ok(git_root, "merge-base", "--is-ancestor", commit, "HEAD")
            and _git_ok(git_root, "cat-file", "-e", f"{commit}:{WP2_CODE}")
        )

    def on_gpu(rec: dict[str, Any]) -> bool:
        worker = rec if rec["kind"] == "Worker" else _record(con, rec["refs"].get("worker"))
        return worker is not None and "gpu" in worker["body"].get("labels", {})

    good = [r for r in (_record(con, rid) for rid in sorted(listed)) if r and clean(r)]
    findings = [
        f"no listed, real, clean WP2 {kind} record"
        for kind in WP2_KINDS
        if not any(r["kind"] == kind for r in good)
    ]
    for kind in ("Target", "Worker"):
        if not any(r["kind"] == kind and on_gpu(r) for r in good):
            findings.append(f"no listed {kind} from a GPU-labelled worker")
    backends: set[str] = set()
    for check in (r for r in good if r["kind"] == "Check" and on_gpu(r)):
        build = _record(con, check["refs"].get("build"))
        if check["body"].get("passed") is True and build and build["body"].get("ok") is True:
            backends.add("tilelang" if "codegen" in build["blobs"] else "cuda")
    findings += [
        f"no passed vector-add Check for a {b} Build"
        for b in ("cuda", "tilelang")
        if b not in backends
    ]
    return findings


def wp2(
    root: Path,
    passed: set[str],
    data_root: Path | None = None,
    git_root: Path | None = None,
    gpu_passed: set[str] | None = None,
) -> list[CheckResult]:
    """GPU acceptance tests run through the Executor on spark; real WP2 records exist.

    In the code stage (D320) the spark run is reported as pending; the records check stays.
    """
    results: list[CheckResult] = []
    if gpu_passed is None and code_stage():
        results.append(pending_real("WP2 acceptance tests", "GPU acceptance run on spark/slai"))
    elif gpu_passed is None:
        from kta_gate.runners import run_pytest

        run = run_pytest(root, "WP2 GPU acceptance run", ["-m", "gpu", "tests/acceptance/WP2"])
        results.append(run.result)
        gpu_passed = run.passed
    if gpu_passed is not None:
        names = {t.split("::")[-1].split("[")[0] for t in gpu_passed if "acceptance.WP2" in t}
        absent = [f"{t} did not pass" for t in WP2_ACCEPTANCE if t not in names]
        results.append(from_findings("WP2 acceptance tests", absent, f"{len(names)} tests passed"))
    if data_root is None and not os.environ.get("KTA_DATA"):
        return [*results, CheckResult("WP2 records", "FAIL", "set KTA_DATA")]
    db = (data_root or Path(os.environ["KTA_DATA"])) / "dev" / "WP2" / "kta.db"
    if not db.exists():
        return [*results, CheckResult("WP2 records", "FAIL", f"{db} not found")]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        findings = _wp2_records(git_root or root, con, _listed(root, "WP2"))
    finally:
        con.close()
    kinds = ", ".join(WP2_KINDS)
    return [*results, from_findings("WP2 records", findings, f"{kinds} listed and real")]


def _yaml_docs(path: Path) -> list[dict[str, Any]]:
    docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d is not None]
    return [d for d in docs if isinstance(d, dict)]


def _kind_of(path: Path, doc: dict[str, Any]) -> str:
    """A document's kind: its ``kind`` field, else the file name (``gap-*.yaml`` -> Gap)."""
    if isinstance(doc.get("kind"), str):
        return str(doc["kind"])
    stem = path.stem.lower()
    for kind in (*REFERENCE_KINDS, "Gap", "Pattern", "Template"):
        if stem.startswith(kind.lower()) or path.parent.name.lower() == f"{kind.lower()}s":
            return kind
    return "unknown"


def _model_validator() -> Callable[[str, dict[str, Any]], None] | None:
    """kta.core.schemas validation once WP1 provides it."""
    try:
        schemas = importlib.import_module("kta.core.schemas")
    except ModuleNotFoundError:
        return None
    validate = getattr(schemas, "validate_body", None)
    if not callable(validate):
        return None
    return cast(Callable[[str, dict[str, Any]], None], validate)


WP4_ACCEPTANCE = (
    "test_seed_templates_pass_validate_template",
    "test_seed_pattern_table_has_the_eight_categories",
    "test_validate_template_error_classes",
    "test_expression_free_variables_are_checked",
    "test_duplicate_is_reported_as_a_hint",
    "test_validate_chain_good_and_error_classes",
    "test_validate_family_good_and_error_classes",
)
WP4_SEED_TEMPLATES = (
    "reduction_partition",
    "physical_storage",
    "async_transfer",
    "multi_buffer_rotation",
    "shared_resource_binding",
)
WP4_PATTERNS = ("reduce", "contract", "map", "store", "transfer", "sync", "access", "other")
WP4_STRUCTURES = 9


def _seed_check(seed: Path, data: Path) -> tuple[CheckResult, CheckResult | None]:
    """Import the seed into a scratch test store; every Template must validate.

    Reference decompositions under seed/references belong to WP10a: they are imported in
    the same scratch store and reported, as WARN when they fail (decision D215).
    """
    # Imported here so the gate of a branch without kta.knowledge still loads.
    from kta.core.config import config_from_mapping
    from kta.core.context import Context
    from kta.core.registry import Registry
    from kta.knowledge import (
        SeedError,
        errors_of,
        import_hw_catalog,
        import_references,
        import_seed,
    )
    from kta.knowledge.pool import find_template, names
    from kta.knowledge.templates import validate_template
    from kta.store import CodeState, open_store

    ctx = Context(env="test", config=config_from_mapping({}), data_dir=data, registry=Registry())
    code = CodeState("0" * 40, False)
    store = open_store(ctx, code)
    # Same order as `kta init`: the declared GB10 feature catalogue first (D98).
    from kta.targets.gb10 import catalog

    import_hw_catalog(ctx, store, catalog(), code)
    try:
        result = import_seed(ctx, store, seed, code)
    except SeedError as exc:
        return CheckResult(
            "seed validates", "FAIL", "seed import refused", [str(i) for i in exc.issues]
        ), None
    findings: list[str] = []
    for name in WP4_SEED_TEMPLATES:
        hit = find_template(store, f"{name}@1")
        if hit is None:
            findings.append(f"Template {name}@1 missing")
            continue
        findings += [f"{name}: {i}" for i in errors_of(validate_template(hit[1], store, None))]
    findings += [f"Pattern {p} missing" for p in WP4_PATTERNS if p not in names(store, "Pattern")]
    if len(names(store, "Structure")) < WP4_STRUCTURES:
        findings.append(f"fewer than {WP4_STRUCTURES} Structures")
    detail = ", ".join(f"{n} {k}" for k, n in sorted(result.counts.items()))
    seed_result = from_findings("seed validates", findings, detail)
    refs = seed / "references"
    if not refs.is_dir():
        return seed_result, None
    try:
        rr = import_references(ctx, store, refs, code)
    except SeedError as exc:
        kinds: dict[str, int] = {}
        for i in exc.issues:
            kinds[i.kind] = kinds.get(i.kind, 0) + 1
        summary = ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))
        return seed_result, CheckResult(
            "reference decompositions",
            "WARN",
            f"not importable: {summary}",
            [str(i) for i in exc.issues[:40]],
        )
    detail = ", ".join(f"{n} {k}" for k, n in sorted(rr.counts.items()))
    return seed_result, CheckResult("reference decompositions", "PASS", detail)


def wp4(root: Path, passed: set[str]) -> list[CheckResult]:
    """The seed imports and validates; the WP4 acceptance tests passed by name."""
    import tempfile

    names_ = {t.split("::")[-1].split("[")[0] for t in passed if "acceptance.WP4" in t}
    absent = [f"{t} did not pass" for t in WP4_ACCEPTANCE if t not in names_]
    results = [from_findings("WP4 acceptance tests", absent, f"{len(names_)} tests passed")]
    with tempfile.TemporaryDirectory(prefix="kta-gate-wp4-") as tmp:
        seed_result, ref_result = _seed_check(root / "knowledge" / "seed", Path(tmp))
    results.append(seed_result)
    if ref_result is not None:
        results.append(ref_result)
    return results


WP5_ACCEPTANCE = (
    "test_constraint_sources_after_assembly",
    "test_constant_binding_removes_variable",
    "test_enumeration_and_mip_agree_on_small_linear_problem",
)
WP5_TEST_FAMILY = Path("tests") / "acceptance" / "WP5" / "gemm_family"


def _wp5_families(root: Path) -> list[Path]:
    seed = root / "knowledge" / "seed" / "families"
    found = sorted(p for p in seed.iterdir() if p.is_dir()) if seed.is_dir() else []
    test_family = root / WP5_TEST_FAMILY
    return [*found, test_family] if test_family.is_dir() else found


def _assemble_check(root: Path, data: Path) -> CheckResult:
    """Every family (seed families and the WP5 GEMM test family) validates and assembles
    over the seed Templates in a scratch store; together they use all seed Templates
    (decision D252)."""
    from kta.core.config import config_from_mapping
    from kta.core.context import Context
    from kta.core.errors import SchemaError
    from kta.core.registry import Registry
    from kta.core.schemas.evaluation import Case
    from kta.formulation import assemble
    from kta.knowledge import SeedError, errors_of, import_seed, load_family, validate_family
    from kta.knowledge.pool import find_template
    from kta.store import CodeState, open_store

    ctx = Context(env="test", config=config_from_mapping({}), data_dir=data, registry=Registry())
    code = CodeState("0" * 40, False)
    store = open_store(ctx, code)
    try:
        import_seed(ctx, store, root / "knowledge" / "seed", code)
    except SeedError as exc:
        return CheckResult("seed formulations assemble", "FAIL", f"seed import refused: {exc}")
    families = _wp5_families(root)
    if not families:
        return CheckResult("seed formulations assemble", "FAIL", "no family to assemble")
    case = Case(problem="gate", shape_dist=[], objective="weighted_latency", target="gate")
    findings: list[str] = []
    used: set[str] = set()
    sizes: list[str] = []
    for path in families:
        name = path.relative_to(root)
        try:
            fp = load_family(path)
        except SchemaError as exc:
            findings.append(f"{name}: {exc}")
            continue
        findings += [f"{name}: {i}" for i in errors_of(validate_family(fp, store, None))]
        templates = {}
        for item in fp.chain.items:
            hit = find_template(store, item.template)
            if hit is not None:
                templates[item.template] = hit[1]
        try:
            f = assemble(fp, fp.chain, templates, case, {})
        except SchemaError as exc:
            findings.append(f"{name}: {exc}")
            continue
        used |= {t.split("@")[0] for t in templates}
        sizes.append(
            f"{fp.family.name}: {len(f.constraints)} constraints, {len(f.cost_terms)} costs"
        )
    findings += [
        f"seed Template {t} is not used by any assembled family"
        for t in WP4_SEED_TEMPLATES
        if t not in used
    ]
    return from_findings("seed formulations assemble", findings, "; ".join(sizes))


def wp5(root: Path, passed: set[str]) -> list[CheckResult]:
    """WP5 acceptance tests passed; the seed validates; seed formulations assemble."""
    import tempfile

    names_ = {t.split("::")[-1].split("[")[0] for t in passed if "acceptance.WP5" in t}
    absent = [f"{t} did not pass" for t in WP5_ACCEPTANCE if t not in names_]
    results = [from_findings("WP5 acceptance tests", absent, f"{len(names_)} tests passed")]
    with tempfile.TemporaryDirectory(prefix="kta-gate-wp5-") as tmp:
        seed_result, _ = _seed_check(root / "knowledge" / "seed", Path(tmp) / "seed")
        results.append(seed_result)
        results.append(_assemble_check(root, Path(tmp) / "assemble"))
    return results


WP6_TESTS = (
    "test_step_produces_exactly_the_yaml_transitions",
    "test_created_task_types_are_listed_on_the_edge",
    "test_lifecycle_edges_used_are_in_the_yaml",
    "test_budget_refusal_keeps_the_task_queued",
)
WP6_MACHINES = Path("src") / "kta" / "orchestrator" / "machines.yaml"


def wp6(root: Path, passed: set[str]) -> list[CheckResult]:
    """Every transition and judging rule of machines.yaml has a passing test (D275)."""
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP6_TESTS if t not in names]
    results = [from_findings("WP6 transition tests", absent, f"{len(WP6_TESTS)} tests passed")]
    path = root / WP6_MACHINES
    if not path.exists():
        return [*results, CheckResult("judging rules tested", "FAIL", f"{path} is missing")]
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules = sorted(
        {
            int(t["rule"])
            for m in doc.get("machines", [])
            for t in m.get("transitions", [])
            if t.get("rule") is not None
        }
    )
    untested = [
        f"rule {n} has no passing test_judging_rule_{n}_*"
        for n in rules
        if not any(x.startswith(f"test_judging_rule_{n}_") for x in names)
    ]
    detail = f"rules {rules} each tested" if rules else "no judging rules found"
    if not rules:
        untested.append("machines.yaml lists no judging rules")
    results.append(from_findings("judging rules tested", untested, detail))
    return results


def wp10a(root: Path, passed: set[str]) -> list[CheckResult]:
    base = root / "knowledge" / "seed" / "references"
    refs = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
    if not refs:
        return [CheckResult("references", "FAIL", f"no reference directories under {base}")]
    validate = _model_validator()
    findings: list[str] = []
    for ref in refs:
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for path in sorted([*ref.rglob("*.yaml"), *ref.rglob("*.yml")]):
            try:
                docs = _yaml_docs(path)
            except yaml.YAMLError as exc:
                findings.append(f"{path.relative_to(root)}: YAML error {exc}")
                continue
            for doc in docs:
                kind = _kind_of(path, doc)
                by_kind.setdefault(kind, []).append(doc)
                if validate is not None and kind != "unknown":
                    try:
                        validate(kind, doc.get("body", doc))
                    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
                        findings.append(f"{path.relative_to(root)}: {kind} invalid: {exc}")
        findings += [f"{ref.name}: no {k}" for k in REFERENCE_KINDS if k not in by_kind]
        recon = by_kind.get("Reconstruction", [{}])[0]
        body = recon.get("body", recon)
        unmatched = body.get("unmatched_transactions", []) if isinstance(body, dict) else []
        gaps = len(by_kind.get("Gap", []))
        if len(unmatched) != gaps:
            findings.append(f"{ref.name}: {len(unmatched)} unmatched transactions, {gaps} Gaps")
    results = [from_findings("references", findings, f"{len(refs)} references")]
    if validate is None:
        results.append(
            CheckResult(
                "reference schemas", "WARN", "kta.core.schemas not available yet; YAML parsed only"
            )
        )
    return results


def wp13(root: Path, passed: set[str]) -> list[CheckResult]:
    """Every measured number in the note cites a source defined in a Sources list."""
    note = root / "deliveries" / "WP13.md"
    if not note.exists():
        return [CheckResult("number sources", "FAIL", "deliveries/WP13.md is missing")]
    text = note.read_text(encoding="utf-8")
    gate_body = split_sections(text).get("gate", "")
    sources = {
        m.group(1): m.group(2) for line in text.splitlines() if (m := SOURCE_DEF.match(line))
    }
    findings: list[str] = []
    for no, line in enumerate(text.splitlines(), 1):
        if line in gate_body or SOURCE_DEF.match(line) or line.lstrip().startswith("#"):
            continue
        if MEASURED_NUMBER.search(line) and not CITATION.search(line):
            findings.append(f"WP13.md:{no}: number without [S<n>] citation")
        findings += [
            f"WP13.md:{no}: {c} has no source"
            for c in CITATION.findall(line)
            if c.strip("[]") not in sources
        ]
    # Remote sources ("host:/path") cannot be checked from here; local ones must exist.
    findings += [
        f"source {sid} path does not exist: {src}"
        for sid, src in sorted(sources.items())
        if src.startswith("/") and not Path(src).exists()
    ]
    return [from_findings("number sources", findings, f"{len(sources)} sources")]


WP9A_TESTS = (
    "test_active_maps_ops_states_onto_the_task_lifecycle",
    "test_case_list_and_timeline",
    "test_lineage_builds_the_full_evidence_chain",
    "test_families_follow_snapshot_membership",
    "test_coverage_reads_the_stored_report_or_reports_missing",
    "test_trace_window_includes_spans_started_before_it",
    "test_app_serves_every_spec_read_endpoint",
    "test_machine_is_well_formed[solve]",
    "test_judging_rules_are_ordered_one_to_four",
)
WEB_FIXTURE = re.compile(r"kta:fixture|SAMPLE DATA")


def web_build(root: Path) -> CheckResult:
    """Build ``web/`` as committed at HEAD, in a fresh directory.

    Untracked or git-ignored sources and a cached ``tsconfig.tsbuildinfo`` exist only in
    the working tree, so building the archive of HEAD (with the lockfile-installed
    ``node_modules`` linked in) fails exactly when a clean checkout would.
    """
    web = root / "web"
    if not (web / "node_modules").is_dir():
        return CheckResult("web build", "FAIL", "web/node_modules missing: run `npm ci` in web/")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", "web"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    if dirty:
        return CheckResult(
            "web build", "FAIL", "web/ differs from HEAD; commit it first", dirty[:10]
        )
    archive = subprocess.run(
        ["git", "-C", str(root), "archive", "--format=tar", "HEAD", "web"],
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        return CheckResult("web build", "FAIL", archive.stderr.decode(errors="replace").strip())
    with tempfile.TemporaryDirectory(prefix="kta-gate-web-") as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(tmp, filter="data")
        clean = Path(tmp) / "web"
        (clean / "node_modules").symlink_to(web / "node_modules", target_is_directory=True)
        proc = subprocess.run(
            ["npm", "run", "--silent", "build"],
            cwd=clean,
            capture_output=True,
            text=True,
            check=False,
        )
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-10:]
    if proc.returncode != 0:
        return CheckResult("web build", "FAIL", "\n".join(tail))
    return CheckResult("web build", "PASS", "tsc + vite build of HEAD in a clean directory")


def wp9a(root: Path, passed: set[str]) -> list[CheckResult]:
    """API tests pass, the web build (type check + bundle) passes, no sample data in web/."""
    names = {t.split("::")[-1] for t in passed}
    absent = [f"{t} did not pass" for t in WP9A_TESTS if t not in names]
    web = root / "web"
    build = web_build(root)
    sample = [
        f"{p.relative_to(root)} contains sample-data markers"
        for p in sorted((web / "src").rglob("*"))
        if p.is_file() and WEB_FIXTURE.search(p.read_text(encoding="utf-8", errors="replace"))
    ]
    if (web / "src" / "sample").exists():
        sample.append("web/src/sample still exists (D95: deleted in phase B)")
    return [
        from_findings("WP9a API tests", absent, f"{len(WP9A_TESTS)} tests passed"),
        build,
        from_findings("no sample data in web/", sample, "web/src reads the API only"),
    ]


SPECIFIC: dict[str, Callable[[Path, set[str]], list[CheckResult]]] = {
    "WP9a": wp9a,
    "WP9b": wp9b,
    "WP0": wp0,
    "WP1": wp1,
    "WP4": wp4,
    "WP5": wp5,
    "WP6": wp6,
    "WP2": wp2,
    "WP3": wp3,
    "WP7": wp7,
    "WP8": wp8,
    "WP11": wp11,
    "WP10a": wp10a,
    "WP10b": wp10b,
    "WP13": wp13,
    "WP14": wp14,
    "WP15": wp15,
    "WP19": wp19,
    "WP21": wp21,
}


_STAGE: list[str] = ["full"]


@contextmanager
def stage(name: str) -> Generator[None]:
    """Set the gate stage that WP-specific checks see (D320)."""
    _STAGE.append(name)
    try:
        yield
    finally:
        _STAGE.pop()


def code_stage() -> bool:
    return _STAGE[-1] == "code"


def pending_real(name: str, what: str) -> CheckResult:
    """Use in place of a check that needs a real run or measurement, in the code stage only."""
    return CheckResult(name, "WARN", f"pending real run (code stage): {what}")


def run_specific(wp: str, root: Path, passed: set[str]) -> list[CheckResult]:
    check = SPECIFIC.get(wp)
    if check is None:
        return [
            CheckResult(
                f"{wp} specific checks",
                "FAIL",
                "not implemented in tools/kta_gate/specific.py; this WP must add them",
            )
        ]
    return check(root, passed)
