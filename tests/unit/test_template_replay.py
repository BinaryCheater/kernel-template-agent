"""Isolation and ledger tests require no model, remote process or GPU."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from kta.core.errors import BudgetExceeded
from kta_template_replay.coordinator import reserve
from kta_template_replay.manifest import freeze, identity, write_new
from kta_template_replay.package import build, template_directory
from kta_template_replay.worker import summary


def environment() -> dict:
    return {
        "target": "test-target",
        "container": "sha256:" + "a" * 64,
        "requires": {"gpu": "sm_121"},
        "language": "tilelang",
        "smoke_record": "test-environment-only",
        "agent_environment_md": "# Environment\nUse the coordinator interface.\n",
    }


def prepared(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    write_new(run / "manifest.json", freeze("gemm", environment(), [41, 59]))
    write_new(
        run / "preflight.json",
        {"passed": True, "manifest_identity": identity((run / "manifest.json").read_bytes())},
    )
    library = tmp_path / "library"
    rule = library / "templates/tile/1"
    rule.mkdir(parents=True)
    for name in ("GUIDE.md", "template.yaml", "proof.yaml"):
        (rule / name).write_text("canonical content")
    for name in ("README.md", "catalog.md", "catalog.json", "mapping.json"):
        (library / name).write_text("menu")
    (library / "hidden-answer.py").write_text("answer")
    return run, library


def test_package_excludes_seeds_sources_and_evaluator(tmp_path: Path) -> None:
    run, library = prepared(tmp_path)
    destination = tmp_path / "workspace"
    snapshot = build(run, library, destination, ["tile@1"])
    assert set(snapshot) == {
        "TASK.md",
        "TEMPLATE_USAGE.md",
        "ENVIRONMENT.md",
        "candidate.py",
        "DESIGN.md",
        "harness/README.md",
        "harness/evaluate",
        "harness/language_smoke.py",
        "library/README.md",
        "library/templates/tile/1/GUIDE.md",
        "library/templates/tile/1/template.yaml",
        "library/templates/tile/1/proof.yaml",
    }
    assert "private_seeds" not in "".join(
        p.read_text() for p in destination.rglob("*") if p.is_file()
    )
    ast.parse((destination / "candidate.py").read_text())
    ast.parse((destination / "harness/language_smoke.py").read_text())
    assert not (destination / "TASK.md").stat().st_mode & 0o222


def test_changed_manifest_cannot_be_packaged(tmp_path: Path) -> None:
    run, library = prepared(tmp_path)
    with (run / "manifest.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed"):
        build(run, library, tmp_path / "workspace", ["tile@1"])


def test_symlink_and_traversal_are_rejected(tmp_path: Path) -> None:
    run, library = prepared(tmp_path)
    for selector in ("../outside@1", "tile@../../secret", "tile"):
        with pytest.raises(ValueError):
            template_directory(library, selector)
    source = library / "templates/tile/1/proof.yaml"
    source.unlink()
    source.symlink_to(library / "hidden-answer.py")
    with pytest.raises(ValueError, match="regular"):
        build(run, library, tmp_path / "workspace", ["tile@1"])


def test_budget_counts_failures_and_blocks_unfinished(tmp_path: Path) -> None:
    manifest = freeze("gemm", environment(), [41, 59])
    attempt = reserve(tmp_path, manifest, 0)
    with pytest.raises(BudgetExceeded, match="unfinished"):
        reserve(tmp_path, manifest, 0)
    write_new(attempt / "execution.json", {"wall_s": 3599})
    with pytest.raises(BudgetExceeded, match="budget"):
        reserve(tmp_path, manifest, 1)


def test_budget_batch_limit_and_nonfinite_time(tmp_path: Path) -> None:
    manifest = freeze("gemm", environment(), [41, 59])
    manifest["max_batches"] = 1
    attempt = reserve(tmp_path, manifest, 0)
    write_new(attempt / "execution.json", {"wall_s": 1})
    with pytest.raises(BudgetExceeded, match="batch"):
        reserve(tmp_path, manifest, 0)
    for value in (math.nan, math.inf, -1):
        with pytest.raises(BudgetExceeded):
            reserve(tmp_path, manifest, value)


def test_event_statistics_and_invalid_samples() -> None:
    stats = summary([1, 2, 3, 4, 100])
    assert stats["median_us"] == 3
    assert stats["iqr_us"] == 2
    for values in ([1, 2, 3], [1, 2, 3, math.nan], [1, 2, 3, 0]):
        with pytest.raises(ValueError):
            summary([float(x) for x in values])


def test_freeze_rejects_mutable_image_and_repeated_seeds() -> None:
    env = environment()
    env["container"] = "image:latest"
    with pytest.raises(ValueError, match="digest"):
        freeze("gemm", env, [1, 2])
    with pytest.raises(ValueError, match="distinct"):
        freeze("gemm", environment(), [1, 1])
