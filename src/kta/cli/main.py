"""``kta`` command entry point. WP0 provides ``doctor``; WP6 adds the rest."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from kta.cli.output import emit_error, emit_json, emit_text
from kta.core.config import default_config_path, load_config
from kta.core.context import Context, data_dir_for, parse_env
from kta.core.doctor import DoctorReport, run_doctor
from kta.core.errors import ConfigError, ExecutionError, KtaError, NotSupported, PermissionDenied
from kta.core.registry import build_registry

# Exit codes, spec/conventions.md §12.
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_SUPPORTED, EXIT_INFRA, EXIT_PERMISSION = range(6)


def context_from_environ(process: str) -> Context:
    env = parse_env(os.environ.get("KTA_ENV"))
    return Context(
        env=env,
        config=load_config(default_config_path()),
        data_dir=data_dir_for(env),
        registry=build_registry(),
        process=process,
    )


def _format_doctor(report: DoctorReport) -> str:
    m = report.manifest
    lines = [
        f"env        {report.env}",
        f"data_dir   {report.data_dir}",
        f"config     {report.config_path} (sha256 {report.config_hash[:12]})",
        f"code       {m.commit}{' (dirty)' if m.dirty else ''}",
        f"host       {m.host} {m.os} {m.arch}",
        f"python     {m.python}  uv {m.uv}  uv.lock {m.uv_lock_sha256[:12]}",
        f"node       {m.node}  npm {m.npm}",
        f"cuda       {m.cuda}  driver {m.driver}  nvcc {m.nvcc}  ptxas {m.ptxas}",
        "interfaces:",
    ]
    if not report.interfaces:
        lines.append("  (none registered)")
    for i in report.interfaces:
        chosen = i.error if i.error else f"{i.implementation} [{i.module}]"
        lines.append(f"  {i.interface:24} {chosen}{'  FIXTURE' if i.fixture else ''}")
    lines += [f"WARN {w}" for w in report.warnings]
    lines += [f"FAIL {p}" for p in report.problems]
    lines.append("doctor: OK" if report.ok else "doctor: FAILED")
    return "\n".join(lines)


def cmd_doctor(args: argparse.Namespace) -> int:
    ctx = context_from_environ("doctor")
    report = run_doctor(ctx, role=args.role)
    if args.json:
        emit_json(report.to_dict())
    else:
        emit_text(_format_doctor(report))
    return EXIT_OK if report.ok else EXIT_FAIL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kta")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="show selected implementations and check lock files")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--role", choices=["control", "worker"], default="control")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        emit_error(f"config error: {exc}")
        return EXIT_USAGE
    except NotSupported as exc:
        emit_error(f"not supported: {exc}")
        return EXIT_NOT_SUPPORTED
    except PermissionDenied as exc:
        emit_error(f"permission denied: {exc}")
        return EXIT_PERMISSION
    except ExecutionError as exc:
        emit_error(f"execution error: {exc}")
        return EXIT_INFRA
    except KtaError as exc:
        emit_error(f"error: {exc}")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
