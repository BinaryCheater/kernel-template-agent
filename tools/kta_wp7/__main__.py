"""``PYTHONPATH=tools uv run python -m kta_wp7 <step>``: WP7's construction runs before
``kta solve`` exists (decision D99). Steps, in order:

- ``setup``: measure-v2, the FP16 and BF16 Cases (Problem, Contract, baseline), the
  family's Chain and Family records, the uncalibrated unit model;
- ``peaks``: per-unit peak microbenchmarks on the GPU worker -> HwFacts and an HwModel;
- ``topk``: solve, build, re-rank, check and measure the top k of every shape bucket,
  with the torch.matmul baseline and the predicted-vs-measured rank correlation;
- ``faults``: the three fault injections;
- ``reuse``: reuse_plan of the chain on the BF16 case;
- ``replay``: re-derive the latest top-k run's conclusions from its records (fresh process).

Needs ``KTA_ENV=dev KTA_WP=WP7`` and a deployed GPU worker; prints the record IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from kta.cli.exec_cmds import command
from kta.cli.main import context_from_environ
from kta.cli.output import emit_text
from kta.core.context import Context
from kta.core.ids import record_id
from kta.core.registry import resolve
from kta.measure.peaks import measure_peaks
from kta.models import UnitModel, load_machine
from kta.models.calibrate import MODEL_SUBJECT, load_model, save_model
from kta.store.base import Store
from kta_wp7 import records as r
from kta_wp7.faults import run_faults
from kta_wp7.pipeline import make_env, run_topk
from kta_wp7.replay import run_replay
from kta_wp7.reuse import best_measured, run_reuse


def _model(ctx: Context, store: Store, target: str) -> tuple[str, UnitModel]:
    model = UnitModel(target=target)
    for rec in store.query("Report", {"subject": MODEL_SUBJECT}, order="created_at", limit=500):
        if rec.body["data"].get("version") == model.version:
            return record_id(rec), load_model(store, record_id(rec))
    return save_model(ctx, store, model, r.ACTOR), model


def _setup(ctx: Context) -> dict[str, Any]:
    store: Store = resolve("store", ctx)
    target, target_id = r.target(ctx)
    fp16 = r.write_case(store, ctx.clock, r.CASE_FP16, target.name, target.image)
    bf16 = r.write_case(store, ctx.clock, r.CASE_BF16, target.name, None)
    family, chain, _ = r.write_family(store, ctx.clock, fp16.problem)
    model_id, _ = _model(ctx, store, target.name)
    return {
        "protocol": r.protocol_v2(store, ctx.clock),
        "case": fp16.case,
        "case_bf16": bf16.case,
        "baseline": fp16.baseline,
        "family": family,
        "chain": chain,
        "model": model_id,
        "target": target_id,
    }


def _refs(ctx: Context) -> dict[str, Any]:
    """Records every later step needs (setup is idempotent)."""
    ids = _setup(ctx)
    store: Store = resolve("store", ctx)
    target, _ = r.target(ctx)
    hw_id, _ = r.latest_hwmodel(store, target.name)
    return {**ids, "hw_model": hw_id, "hw": load_machine(store, hw_id)}


def step_peaks(ctx: Context) -> dict[str, Any]:
    ids = _setup(ctx)
    target, target_id = r.target(ctx)
    e = make_env(ctx, target, target_id)
    limits = resolve("target", ctx).limits()
    sm_count = int(limits["sm_count"])
    res = measure_peaks(e.m, e.cuda, target, target_id, limits, ids["protocol"], sm_count)
    return {"hw_model": res.hwmodel_id, "peaks": res.peaks, "facts": res.facts}


def _case_ids(ctx: Context, refs: dict[str, Any]) -> tuple[r.CaseIds, r.CaseIds, Any]:
    store: Store = resolve("store", ctx)
    target, _ = r.target(ctx)
    fp16 = r.write_case(store, ctx.clock, r.CASE_FP16, target.name, target.image)
    bf16 = r.write_case(store, ctx.clock, r.CASE_BF16, target.name, None)
    _, _, fp = r.write_family(store, ctx.clock, fp16.problem)
    return fp16, bf16, fp


def step_topk(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    refs = _refs(ctx)
    fp16, _, fp = _case_ids(ctx, refs)
    target, target_id = r.target(ctx)
    e = make_env(ctx, target, target_id)
    model = load_model(e.store, refs["model"])
    keep = {k: refs[k] for k in ("family", "hw_model", "model", "protocol")}
    rid = run_topk(e, fp16, fp, keep, refs["hw"], model, args.k, args.shortlist, args.workers)
    return {"report": rid, "summary": e.store.get(rid).body["summary"]}


def step_faults(ctx: Context) -> dict[str, Any]:
    refs = _refs(ctx)
    fp16, _, fp = _case_ids(ctx, refs)
    target, target_id = r.target(ctx)
    e = make_env(ctx, target, target_id)
    model = load_model(e.store, refs["model"])
    keep = {k: refs[k] for k in ("family", "hw_model", "model")}
    rid = run_faults(e, fp16, fp, keep, refs["hw"], model)
    return {"report": rid, "summary": e.store.get(rid).body["summary"]}


def step_reuse(ctx: Context) -> dict[str, Any]:
    refs = _refs(ctx)
    fp16, bf16, _ = _case_ids(ctx, refs)
    store: Store = resolve("store", ctx)
    cases = {"bf16": bf16, "fp16": fp16}
    best = best_measured(store)
    rid = run_reuse(store, ctx.clock, refs["chain"], cases, refs["hw"].hw, best)
    return {"report": rid, "summary": store.get(rid).body["summary"]}


def step_replay(ctx: Context) -> dict[str, Any]:
    _setup(ctx)
    fp16, _, _ = _case_ids(ctx, {})
    store: Store = resolve("store", ctx)
    rid = run_replay(ctx, store, fp16)
    return {"report": rid, "summary": store.get(rid).body["summary"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kta_wp7")
    parser.add_argument("step", choices=["setup", "peaks", "topk", "faults", "reuse", "replay"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    with command(context_from_environ, "wp7", f"kta_wp7 {args.step}") as ctx:
        if args.step == "setup":
            out = _setup(ctx)
        elif args.step == "peaks":
            out = step_peaks(ctx)
        elif args.step == "topk":
            out = step_topk(ctx, args)
        elif args.step == "faults":
            out = step_faults(ctx)
        elif args.step == "reuse":
            out = step_reuse(ctx)
        else:
            out = step_replay(ctx)
    emit_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
