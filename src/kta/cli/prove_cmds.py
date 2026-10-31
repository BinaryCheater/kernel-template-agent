"""``kta prove``: the verification loop's commands (WP24; thin wrappers over kta.checks.proof).

    kta prove template <name@version | id> [--no-reuse]   K1–K3 once per Template
    kta prove all                                          every Template that has an outline
    kta prove family <id> [--target T] [--declare A] [--case E]   Theorem 1 over its chain
    kta prove wp <name@version | id>                       weakest-precondition search
    kta prove drift                                        invalidate on changed premises
    kta prove status <id>                                  effective assurance and summary
    kta prove canary [--pre EXPR]                          planted unsound Pre must be refuted

Checker runs are CPU work on the control plane here; a heavy run can go to the compute
host as a job like any other CPU job (they are deterministic and need only this package).
"""

from __future__ import annotations

import argparse
from typing import Any

from kta.checks.proof import compose, records, wp
from kta.checks.proof.prove import CHECKER, find_template, outline_for, prove_template
from kta.checks.proof.template import check_template
from kta.cli.exec_cmds import ContextFactory, _out, command
from kta.core.errors import ConfigError
from kta.core.ids import record_id
from kta.core.registry import resolve
from kta.core.schemas import Record
from kta.core.schemas.knowledge import Template
from kta.models import load_machine
from kta.store.assurance import effective, proof_summary
from kta.store.sqlite import SqliteStore

GEMM_CASES = ("case.M >= 1", "case.N >= 1", "case.K >= 1")


def _store(ctx: Any) -> SqliteStore:
    store = resolve("store", ctx)
    if not isinstance(store, SqliteStore):
        raise ConfigError("kta prove needs the SQLite store")
    return store


def _template(store: SqliteStore, ref: str) -> str:
    return find_template(store, ref) if "@" in ref else ref


def _lines(p: Any) -> str:
    head = f"summary {p.summary}  level {p.level or 'none'}  results {len(p.results)}"
    return "\n".join([head, *(f"  gap: {g}" for g in p.gaps)])


def cmd_template(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "prove", f"kta prove template {args.ref}") as ctx:
        store = _store(ctx)
        p = prove_template(
            store, _template(store, args.ref), ctx.clock.now(), reuse=not args.no_reuse
        )
    data = {
        "summary": p.summary,
        "level": p.level,
        "results": p.results,
        "reused": p.reused,
        "queries": p.queries,
        "gaps": p.gaps,
    }
    _out(args, data, _lines(p))
    return 0


def cmd_all(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    rows: list[dict[str, Any]] = []
    with command(make_ctx, "prove", "kta prove all") as ctx:
        store = _store(ctx)
        refs = sorted({o.body["template"] for o in store.query("ProofOutline", limit=10000)})
        for ref in refs:
            p = prove_template(
                store, find_template(store, ref), ctx.clock.now(), reuse=not args.no_reuse
            )
            rows.append(
                {
                    "template": ref,
                    "summary": p.summary,
                    "level": p.level,
                    "reused": p.reused,
                    "gaps": p.gaps,
                }
            )
    _out(
        args,
        {"templates": rows},
        "\n".join(f"{r['template']:32} {r['level'] or 'none':8} {r['summary']}" for r in rows),
    )
    return 0


def cmd_family(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "prove", f"kta prove family {args.family}") as ctx:
        store = _store(ctx)
        hw = None
        if args.target:
            found = store.query("HwModel", {"target": args.target}, order="-created_at", limit=1)
            if not found:
                raise ConfigError(f"no HwModel for target {args.target}")
            hw = dict(load_machine(store, record_id(found[0])).hw)
        p = compose.prove_family(
            store, args.family, ctx.clock.now(), case_domain=args.case or GEMM_CASES,
            declared=args.declare or [], hw=hw,
        )  # fmt: skip
    _out(
        args,
        {"summary": p.summary, "level": p.level, "results": p.results, "gaps": p.gaps},
        _lines(p),
    )
    return 0


def cmd_wp(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "prove", f"kta prove wp {args.ref}") as ctx:
        store = _store(ctx)
        rid, w = wp.record_search(store, _template(store, args.ref), ctx.clock.now())
    data = {
        "result": rid,
        "relation": w.relation,
        "current": w.current,
        "weaker": w.weaker,
        "verified": w.verified,
    }
    _out(
        args,
        data,
        f"{w.relation}: {w.current}" + (f" -> {w.weaker} ({w.verified})" if w.weaker else ""),
    )
    return 0


def cmd_drift(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "prove", "kta prove drift") as ctx:
        marked = records.drift(_store(ctx), ctx.clock.now())
    _out(args, {"invalidated": marked}, f"invalidated {len(marked)} record(s)")
    return 0


def cmd_status(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "prove", f"kta prove status {args.id}") as ctx:
        store = _store(ctx)
        s = proof_summary(store, args.id)
        level = effective(store, args.id)
    _out(
        args,
        {"assurance": level, "summary": s},
        f"assurance {level}; summary {s['id'] if s else 'none'}",
    )
    return 0


CANARY_TEMPLATE = "shared_memory_tiling@1"
CANARY_PRE = "P <= S - 1"  # paper §4.3: the barrier ignored; admits the race of Figure 7


def cmd_canary(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """The checker's canary: the tiling Template with a planted unsound precondition, checked
    in memory (nothing enters the pool); the outcome is a Report."""
    with command(make_ctx, "prove", "kta prove canary") as ctx:
        store = _store(ctx)
        tid = find_template(store, CANARY_TEMPLATE)
        t = Template.model_validate(store.get(tid).body)
        found = outline_for(store, CANARY_TEMPLATE)
        planted = wp.relaxed(t, t.preconditions[0].id, args.pre)
        chk = check_template(planted, found[1] if found else None)
        bad = [d for d in chk.drafts if d.outcome == "counterexample"]
        caught = bool(bad) and all(
            d.replay is not None and not d.replay.get("agrees", True) for d in bad
        )
        data = {
            "template": CANARY_TEMPLATE,
            "planted_pre": args.pre,
            "caught": caught,
            "counterexamples": [
                {"obligation": d.obligation, "reason": d.reason, "counterexample": d.counterexample,
                 "replay": {k: v for k, v in (d.replay or {}).items() if k != "trace"}}
                for d in bad
            ],
            "outcomes": {f"{d.condition}:{d.obligation}": d.outcome for d in chk.drafts},
            **records.versions(),
        }  # fmt: skip
        body = {
            "title": "proof_canary",
            "subject": tid,
            "summary": f"planted {args.pre}: caught={caught}",
            "data": data,
        }
        rid = store.put(Record.new("Report", body, CHECKER, ctx.clock.now()))
    _out(args, {"report": rid, **data}, f"canary {'caught' if caught else 'MISSED'}: report {rid}")
    return 0 if caught else 1


def add_prove_commands(sub: Any, make_ctx: ContextFactory) -> None:
    prove = sub.add_parser("prove", help="Template proofs, composition and proof reuse (WP24)")
    psub = prove.add_subparsers(dest="prove_command", required=True)

    def add(name: str, fn: Any, help: str) -> argparse.ArgumentParser:
        p = psub.add_parser(name, help=help)
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=lambda a: fn(a, make_ctx))
        return p

    t = add("template", cmd_template, "K1–K3 of one Template")
    t.add_argument("ref", help="name@version or record ID")
    t.add_argument("--no-reuse", action="store_true")
    a = add("all", cmd_all, "every Template with a ProofOutline")
    a.add_argument("--no-reuse", action="store_true")
    f = add("family", cmd_family, "Theorem 1 over a family's chain")
    f.add_argument("family", help="Family record ID")
    f.add_argument("--target", help="hardware values of this Target's newest HwModel")
    f.add_argument("--declare", action="append", help="an assumption of the whole result")
    f.add_argument("--case", action="append", help="case domain expression (default M, N, K >= 1)")
    w = add("wp", cmd_wp, "weakest-precondition search")
    w.add_argument("ref")
    add("drift", cmd_drift, "invalidate proofs whose premises changed")
    s = add("status", cmd_status, "effective assurance of a record")
    s.add_argument("id")
    c = add(
        "canary", cmd_canary, "a planted unsound precondition must give a replayed counterexample"
    )
    c.add_argument("--pre", default=CANARY_PRE)
