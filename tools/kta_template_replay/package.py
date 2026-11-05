"""Allowlist package construction, with no evaluator code or validation inputs."""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import Any

from .manifest import identity, write_new

USAGE = """# Template use
Read the supplied library. Select concrete rules, read their guides, bind variables,
then implement and validate. Keep DESIGN.md brief: selected id/version, bindings,
coupled constraints, inferred extensions and choices supplied from your own knowledge.
Do not fetch answer implementations or original source-study material. Language docs
are allowed only when supplied in ENVIRONMENT.md. Preserve the best correct candidate.
"""
INTERFACE = """# Evaluator interface
Run `harness/evaluate` from this workspace, optionally supplying a candidate file path.
Active time is recorded automatically by the coordinator. Each call retains the
candidate, checks correctness and benchmarks all shapes under the shared GPU lock.
Feedback contains compiler/runtime failures, numerical errors and raw timing samples.
All target-kernel correctness and performance evaluations must use harness/evaluate.
Every call counts toward the declared batch budget. Do not run separate target
benchmarking or correctness sweeps. Direct remote compilation and API inspection
remain available; retain those sources and logs in scratch/.
Use candidate.py, DESIGN.md, scratch/ and candidate_versions/ for implementation,
local experiments and saved versions. The submitted candidate.py must be self-contained;
scratch/support files are not submitted to the evaluator. Never modify TASK.md,
ENVIRONMENT.md, library/ or harness/. The evaluator owns inputs and tests.
No source search or network browsing for target implementations is permitted.
Filesystem/tool isolation here is instruction-only; readonly permissions are not a sandbox.
"""


def safe_copy(source: Path, dest: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Only regular files may enter package: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def template_directory(library: Path, selector: str) -> Path:
    name, sep, version = selector.partition("@")
    if not sep or not name or not version or any(c in selector for c in ("/", "\\", "..")):
        raise ValueError("Template selector must be name@version")
    directory = library / "templates" / name / version
    if directory.is_symlink() or not directory.resolve().is_relative_to(library.resolve()):
        raise ValueError("Template directory escapes canonical library")
    for part in directory.relative_to(library).parts:
        library = library / part
        if library.is_symlink():
            raise ValueError("Symlink directory in library")
    return directory


def build(run: Path, library: Path, destination: Path, selectors: list[str]) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    if not manifest["environment"].get("agent_environment_md"):
        raise ValueError("Environment must include agent_environment_md")
    if destination.exists():
        raise FileExistsError(destination)
    if not (run / "preflight.json").exists():
        raise ValueError("Run evaluator smoke before preparing implementation package")
    preflight = json.loads((run / "preflight.json").read_text())
    if preflight.get("manifest_identity") != identity((run / "manifest.json").read_bytes()):
        raise ValueError("Manifest changed after preflight")
    if not preflight.get("passed"):
        raise ValueError("Oracle preflight must pass unchanged tolerances")
    # Validate inputs before writing any workspace files.
    directories = [template_directory(library, s) for s in selectors]
    if not selectors:
        directories = [p for p in (library / "templates").glob("*/*") if p.is_dir()]
        if not directories:
            raise ValueError("Canonical library is empty")
        for directory in directories:
            template_directory(library, f"{directory.parent.name}@{directory.name}")
    files = []
    for directory in directories:
        for name in ("GUIDE.md", "template.yaml", "proof.yaml"):
            source = directory / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Missing regular canonical file: {source}")
            files.append(source)
    destination.mkdir(parents=True)
    for source in files:
        safe_copy(source, destination / "library" / source.relative_to(library))
    guide_list = "\n".join(
        f"- [{p.parent.name}@{p.name}](templates/{p.parent.name}/{p.name}/GUIDE.md)"
        for p in directories
    )
    (destination / "library/README.md").write_text(
        "# Frozen Template library\n\nRead a guide, then its rule and proof outline. "
        "Record selected identities and bindings in DESIGN.md.\n\n" + guide_list + "\n"
    )
    if not selectors:
        for name in ("catalog.md", "catalog.json"):
            safe_copy(library / name, destination / "library" / name)
    entry = "entry(a, b)" if manifest["case"] == "gemm" else "entry(q, k, v)"
    math = (
        "C = A @ B; A[M,K] and B[K,N] row-major, no bias or activation."
        if manifest["case"] == "gemm"
        else "O = softmax(Q @ K.transpose(-2,-1) / sqrt(D) + causal_mask) @ V. "
        "Q,K,V contiguous [B,H,S,D]; mask excludes key positions greater than query. "
        "Equal head counts, no dropout, backward or LSE output."
    )
    task = (
        f"# {manifest['case']}\n\n{math}\n\n"
        f"Shapes: {manifest['shapes']}. All inputs/output FP16, accumulation/statistics FP32. "
        f"Return one new contiguous CUDA FP16 tensor from candidate.py::{entry}. "
        "Inputs must remain unchanged. No persistent output caching, hidden host compute, "
        "shape removal or semantic changes. "
        "Include all packing/preprocessing/dispatch per call.\n\n"
        f"Target: {manifest['environment']['target']}; language: {manifest['language']}. "
        f"Tolerance: atol={manifest['atol']}, rtol={manifest['rtol']}; finite outputs required. "
        f"Budget: {manifest['active_budget_s']} active seconds, "
        f"{manifest['max_batches']} evaluation batches, "
        f"{manifest['max_shapes_per_batch']} shapes per batch. "
        f"Timing: {manifest['warmup']} warmups, {manifest['repetitions']} event samples, "
        f"{manifest['cache_policy']}; performance aim {manifest['performance_aim']}.\n"
    )
    (destination / "TASK.md").write_text(task)
    (destination / "TEMPLATE_USAGE.md").write_text(USAGE)
    environment_text = manifest["environment"].get("agent_environment_md")
    if not environment_text:
        raise ValueError("Environment must include agent_environment_md with smoke instructions")
    environment_text += (
        "\n## Working language example\n\n"
        "`harness/language_smoke.py` is a standalone, real-worker-tested vector-add example. "
        "It demonstrates TileLang JIT, tensor annotations, launch geometry, guarded tails "
        "and invocation. Copy it to a fresh remote scratch directory and run it inside "
        "the recorded pinned image under the shared host lock, using the environment's "
        "remote compile instructions. It writes language-smoke.json and raises on failure. "
        "Installed public API docstrings and compilation-only inspection are available. "
        "Target-kernel validation/timing continues through harness/evaluate.\n"
    )
    (destination / "ENVIRONMENT.md").write_text(environment_text)
    (destination / "harness").mkdir()
    safe_copy(
        Path(__file__).resolve().parents[2] / "examples/template_replay/vector_add_smoke.py",
        destination / "harness/language_smoke.py",
    )
    (destination / "harness/README.md").write_text(INTERFACE)
    tool = Path(__file__).resolve().parents[2] / "scripts/template-replay"
    (destination / "harness/evaluate").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"exec {shlex.quote(str(tool))} remote-evaluate {shlex.quote(str(run.resolve()))} "
        '"${1:-$PWD/candidate.py}" "${@:2}"\n'
    )
    (destination / "candidate.py").write_text(
        f"def {entry}:\n    raise NotImplementedError('Implement TASK.md')\n"
    )
    (destination / "DESIGN.md").write_text("# Design\n")
    (destination / "scratch").mkdir()
    (destination / "candidate_versions").mkdir()
    snapshot = {
        str(p.relative_to(destination)): identity(p.read_bytes())
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    for path in destination.rglob("*"):
        if path.is_file() and path.name not in ("candidate.py", "DESIGN.md"):
            path.chmod(0o555 if path == destination / "harness/evaluate" else 0o444)
    write_new(
        run / "package.json",
        {
            "workspace": str(destination.resolve()),
            "mode": "prescribed" if selectors else "self-selection",
            "selectors": selectors,
            "files": snapshot,
            "isolation": manifest["isolation"],
        },
    )
    return snapshot
