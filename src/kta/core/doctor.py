"""``kta doctor``: which implementation each interface uses, and whether the environment
matches the lock files (spec/conventions.md §2, §7).

In dev and main, selecting a fixture implementation is a failure. Lock mismatches fail
in main and are warnings elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.envmanifest import EnvManifestData, Role, collect_manifest
from kta.core.errors import ConfigError
from kta.core.registry import REAL_ONLY_ENVS

DEFAULT_CUDA_HOME = "/usr/local/cuda"


@dataclass(frozen=True)
class InterfaceReport:
    interface: str
    implementation: str | None
    fixture: bool
    module: str | None
    error: str | None


@dataclass(frozen=True)
class DoctorReport:
    env: str
    data_dir: str
    config_path: str | None
    config_hash: str
    interfaces: list[InterfaceReport]
    manifest: EnvManifestData
    problems: list[str] = field(default_factory=lambda: list[str]())
    warnings: list[str] = field(default_factory=lambda: list[str]())

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "env": self.env,
            "data_dir": self.data_dir,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "interfaces": [vars(i) for i in self.interfaces],
            "manifest": self.manifest.to_dict(),
            "problems": self.problems,
            "warnings": self.warnings,
        }


def check_interfaces(ctx: Context) -> list[InterfaceReport]:
    names = set(ctx.registry.interfaces()) | set(ctx.config.section("impl"))
    reports: list[InterfaceReport] = []
    for name in sorted(names):
        try:
            impl = ctx.registry.select(name, ctx.config)
        except ConfigError as exc:
            reports.append(InterfaceReport(name, None, False, None, str(exc)))
            continue
        reports.append(InterfaceReport(name, impl.name, impl.fixture, impl.module, None))
    return reports


def cuda_home(ctx: Context) -> Path:
    return Path(str(ctx.config.section("toolchain").get("cuda_home", DEFAULT_CUDA_HOME)))


def run_doctor(ctx: Context, role: Role = "control", root: Path | None = None) -> DoctorReport:
    source_root = repo_root() if root is None else root
    interfaces = check_interfaces(ctx)
    manifest = collect_manifest(role, source_root, cuda_home(ctx))
    problems: list[str] = []
    warnings: list[str] = []
    for report in interfaces:
        if report.error is not None:
            problems.append(f"{report.interface}: {report.error}")
        elif report.fixture and ctx.env in REAL_ONLY_ENVS:
            problems.append(
                f"{report.interface}: fixture implementation {report.implementation!r} "
                f"selected in env {ctx.env}"
            )
    target = problems if ctx.env == "main" else warnings
    target.extend(f"lock: {m}" for m in manifest.lock_mismatches)
    if ctx.env == "main" and manifest.dirty:
        problems.append("main env requires committed code (working tree is dirty)")
    return DoctorReport(
        env=ctx.env,
        data_dir=str(ctx.data_dir),
        config_path=str(ctx.config.path) if ctx.config.path else None,
        config_hash=ctx.config.hash,
        interfaces=interfaces,
        manifest=manifest,
        problems=problems,
        warnings=warnings,
    )
