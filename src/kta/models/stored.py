"""The hardware model as stored: an HwModel record and the HwArch it is on (D463)."""

from __future__ import annotations

from kta.core.errors import NotSupported
from kta.core.ids import record_id
from kta.core.schemas.hardware import HwModel
from kta.core.schemas.hwmodel import HwArch
from kta.models.machine import MachineModel
from kta.store.base import Store


def load_arch(store: Store, hw_id: str, hw: HwModel) -> HwArch | None:
    """The HwArch of ``hw``: ``refs.arch`` of its record, else the newest of that name@version."""
    if hw.arch is None:
        return None
    ref = store.get(hw_id).refs.get("arch")
    if isinstance(ref, str):
        return HwArch.model_validate(store.get(ref).body)
    name, _, version = hw.arch.partition("@")
    found = store.query(
        "HwArch", {"name": name, "version": int(version)}, order="-created_at", limit=1
    )
    if not found:
        raise NotSupported(f"HwModel {hw.target}@{hw.version}: HwArch {hw.arch} is not stored")
    return HwArch.model_validate(found[0].body)


def load_machine(store: Store, hw_id: str) -> MachineModel:
    hw = HwModel.model_validate(store.get(hw_id).body)
    return MachineModel.build(hw, load_arch(store, hw_id, hw))


def latest_machine(store: Store, target: str) -> tuple[str, MachineModel]:
    """The newest HwModel of ``target``, resolved on its class."""
    found = store.query("HwModel", {"target": target}, order="-created_at", limit=1)
    if not found:
        raise NotSupported(f"no HwModel record for target {target}")
    rid = record_id(found[0])
    return rid, load_machine(store, rid)
