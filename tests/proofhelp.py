"""Builders for WP24 proof tests: the real seed (with its proof outlines) and the seed
GEMM family as Chain and Family records."""

from __future__ import annotations

from typing import Any

import yaml

from knowhelp import SEED, seed
from kta.core.context import Context
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.proof import ProofOutline
from kta.knowledge import load_family
from kta.store.sqlite import SqliteStore
from storehelp import T0

TOOL = Actor(type="tool", name="test")
GEMM_CASES = ("case.M >= 1", "case.N >= 1", "case.K >= 1")


def seeded(ctx: Context, store: SqliteStore) -> None:
    seed(ctx, store)


def template_id(store: SqliteStore, ref: str) -> str:
    name, _, version = ref.partition("@")
    return record_id(store.query("Template", where={"name": name, "version": int(version)})[0])


def gemm_family(store: SqliteStore) -> str:
    fp = load_family(SEED / "families" / "gemm_fp16_tilelang")
    tids = [template_id(store, it.template) for it in fp.chain.items]
    chain = store.put(Record.new("Chain", fp.chain, TOOL, T0, refs={"templates": tids}))
    return store.put(Record.new("Family", fp.family, TOOL, T0, refs={"chain": chain}))


def gb10_hw() -> dict[str, Any]:
    from kta.core.schemas.hardware import HwModel
    from kta.core.schemas.hwmodel import HwArch
    from kta.models.machine import MachineModel

    hw = SEED / "hardware"
    arch = HwArch.model_validate(yaml.safe_load((hw / "arch_sm12x.yaml").read_text()))
    model = HwModel.model_validate(yaml.safe_load((hw / "model_gb10.yaml").read_text()))
    return dict(MachineModel.build(model, arch).hw)


def put_outline(store: SqliteStore, body: dict[str, Any]) -> str:
    o = ProofOutline.model_validate(body)
    return store.put(
        Record.new("ProofOutline", o, TOOL, T0, refs={"template": template_id(store, o.template)})
    )
