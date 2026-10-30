"""``kta measure`` and ``kta check``: WP3 commands (thin wrappers over kta.measure, kta.checks)."""

from __future__ import annotations

import argparse
from typing import Any

from kta.checks import selftest as check_selftest
from kta.checks.base import Checker
from kta.cli.exec_cmds import ContextFactory, _out, command
from kta.core.context import Context
from kta.core.errors import ConfigError
from kta.core.registry import resolve
from kta.core.schemas.hardware import Target
from kta.measure import noise, protocol, selftest
from kta.measure.profile import run_profile
from kta.measure.runner import Measurer
from kta.store.base import Store


def _setup(ctx: Context) -> tuple[Measurer, Any, Target, str]:
    store: Store = resolve("store", ctx)
    plugin = resolve("target", ctx)
    target_id = plugin.latest_target_id()
    target = Target.model_validate(store.get(target_id).body)
    m = Measurer(store, resolve("executor", ctx), ctx.clock, ctx.rng, ctx.seed)
    return m, resolve("backend.cuda", ctx), target, target_id


def _protocol_id(store: Store, name: str) -> str:
    version = int(name.removeprefix("measure-v"))
    pid = protocol.find_protocol(store, protocol.NAME, version)
    if pid is None:
        raise ConfigError(f"no Protocol record {name}; run `kta measure protocol` / noise --revise")
    return pid


def cmd_protocol(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "measure-protocol", "kta measure protocol") as ctx:
        store: Store = resolve("store", ctx)
        pid = protocol.ensure_v1(store, ctx.clock)
        body = store.get(pid).body
    _out(args, {"protocol": pid, **body}, f"measure-v1 {pid}")
    return 0


def cmd_noise(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    reports: list[dict[str, Any]] = []
    with command(make_ctx, "measure-noise", f"kta measure noise {args.kernel}") as ctx:
        m, cuda, target, target_id = _setup(ctx)
        pid = _protocol_id(m.store, args.protocol)
        names = ["vadd", "gemm"] if args.kernel == "all" else [args.kernel]
        for name in names:
            w = selftest.workload(m.store, m.clock, name, target_id)
            gk = (
                selftest.vadd_kernel(w.shape["n"])
                if name == "vadd"
                else selftest.gemm_kernel(w.shape)
            )
            module = selftest.build(cuda, gk, target, m.ex)
            rid, data = noise.noise_probe(m, module, w.case, w.shape, pid, args.groups, name)
            reports.append({"kernel": name, "report": rid, "cv": data["cv"]})
        revised = None
        if args.revise:
            revised = noise.revise_protocol(m, pid, [r["report"] for r in reports])
    text = "\n".join(
        f"{r['kernel']:5} report {r['report']} cv {r['cv'] * 100:.3f}%" for r in reports
    )
    if revised:
        text += f"\nmeasure-v2 {revised}"
    _out(args, {"reports": reports, "measure_v2": revised}, text)
    return 0


def cmd_compare(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    rows: list[dict[str, Any]] = []
    with command(make_ctx, "measure-compare", "kta measure compare") as ctx:
        m, cuda, target, target_id = _setup(ctx)
        pid = _protocol_id(m.store, args.protocol)
        w = selftest.workload(m.store, m.clock, "fma", target_id)
        base = selftest.build(cuda, selftest.fma_kernel(1000), target, m.ex)
        slower = selftest.build(cuda, selftest.fma_kernel(1050), target, m.ex)
        for label, b in (("1000 vs 1050 steps", slower), ("1000 vs 1000 steps", base)):
            c = m.compare(base, b, w.case, w.shape, pid)
            rows.append(
                {
                    "pair": label,
                    "report": c.report_id,
                    "rel_diff": c.rel_diff,
                    "ci": [c.ci_low, c.ci_high],
                    "significant": c.significant,
                }
            )
    text = "\n".join(
        f"{r['pair']:20} {r['rel_diff'] * 100:+.2f}% significant={r['significant']} "
        f"report {r['report']}"
        for r in rows
    )
    _out(args, {"comparisons": rows}, text)
    return 0


def cmd_profile(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "measure-profile", "kta measure profile") as ctx:
        m, cuda, target, target_id = _setup(ctx)
        w = selftest.workload(m.store, m.clock, "gemm", target_id)
        module = selftest.build(cuda, selftest.gemm_kernel(w.shape), target, m.ex)
        pid, body = run_profile(m, module, w.case, w.shape)
    _out(args, {"profile": pid, **body.model_dump()}, f"profile {pid} {body.metrics}")
    return 0


def cmd_check_selftest(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "check-selftest", "kta check selftest") as ctx:
        m, cuda, target, target_id = _setup(ctx)
        c = Checker(m.store, m.ex, ctx.clock, ctx.rng, ctx.data_dir / "checker", ctx.seed)
        variants = tuple(args.variant) if args.variant else check_selftest.VARIANTS
        flags = resolve("target", ctx).compile_flags("cuda")
        rid, results = check_selftest.run_all(c, cuda, target, target_id, flags, variants)
    rows = [
        {
            "variant": r.variant,
            "check": r.check.id,
            "passed": r.check.passed,
            "failed_checks": sorted(r.failed_checks),
            "as_expected": r.as_expected,
        }
        for r in results
    ]
    text = "\n".join(
        f"{r['variant']:12} passed={r['passed']!s:5} as_expected={r['as_expected']!s:5} "
        f"{','.join(r['failed_checks'])} check {r['check'][:12]}"
        for r in rows
    )
    _out(args, {"report": rid, "variants": rows}, f"report {rid}\n{text}")
    return 0 if all(r["as_expected"] for r in rows) else 1


def add_measure_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    meas = sub.add_parser("measure", help="measurement protocol and selftests on the GPU worker")
    mc = meas.add_subparsers(dest="measure_command", required=True)
    p = mc.add_parser("protocol", help="write (or find) the Protocol record measure-v1")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=lambda a: cmd_protocol(a, make_ctx))
    n = mc.add_parser("noise", help="noise probe: repeated measurements of one kernel")
    n.add_argument("--kernel", choices=["vadd", "gemm", "all"], default="all")
    n.add_argument("--groups", type=int, default=30)
    n.add_argument("--protocol", default="measure-v1")
    n.add_argument("--revise", action="store_true", help="write measure-v2 from the probe")
    n.add_argument("--json", action="store_true")
    n.set_defaults(func=lambda a: cmd_noise(a, make_ctx))
    c = mc.add_parser("compare", help="interleaved comparison: ~5%% apart and identical pairs")
    c.add_argument("--protocol", default="measure-v2")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=lambda a: cmd_compare(a, make_ctx))
    pr = mc.add_parser("profile", help="Nsight Compute profile of the selftest GEMM")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=lambda a: cmd_profile(a, make_ctx))
    chk = sub.add_parser("check", help="checker selftests on the GPU worker")
    cc = chk.add_subparsers(dest="check_command", required=True)
    st = cc.add_parser("selftest", help="full_check of a correct GEMM and three faulty variants")
    st.add_argument("--variant", action="append", choices=list(check_selftest.VARIANTS))
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=lambda a: cmd_check_selftest(a, make_ctx))
