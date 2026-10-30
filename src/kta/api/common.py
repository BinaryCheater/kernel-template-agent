"""Shared helpers for the read views: record envelopes with derived status, small lookups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from importlib import resources
from typing import Any

import yaml

from kta.core.config import Config
from kta.core.errors import ConfigError
from kta.store import SqliteOps, SqliteStore


@dataclass(frozen=True)
class Api:
    """What every view needs: the store, the ops tables, the config and the clock."""

    store: SqliteStore
    ops: SqliteOps
    config: Config
    env: str

    def now(self) -> datetime:
        return self.store.clock.now()


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@cache
def machines() -> dict[str, Any]:
    """The orchestrator's state machines (D96), as loaded from machines.yaml."""
    text = resources.files("kta.orchestrator").joinpath("machines.yaml").read_text("utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError("machines.yaml must be a mapping")
    return data  # pyright: ignore[reportUnknownVariableType]


def machine_limit(machine: str, key: str, default: int) -> int:
    for m in machines()["machines"]:
        if m["id"] == machine:
            return int(m.get("limits", {}).get(key, default))
    return default


def envelope(api: Api, id: str) -> dict[str, Any]:
    """GET /records/{id}: the stored JSON plus derived status and assurance."""
    data = api.store.get_json(id)
    status, assurance = api.store.status(id)
    return {"id": id, **data, "status": status, "assurance": assurance}


def body(api: Api, id: str) -> dict[str, Any]:
    return api.store.get_json(id)["body"]


def refs(api: Api, id: str) -> dict[str, Any]:
    return api.store.get_json(id).get("refs", {})


def ref_one(api: Api, id: str, name: str) -> str | None:
    value = refs(api, id).get(name)
    if isinstance(value, list):
        return value[0] if value else None  # pyright: ignore[reportUnknownVariableType]
    return value


def incoming(api: Api, id: str, rel: str, kind: str) -> list[str]:
    """IDs of records of ``kind`` whose refs[rel] point at ``id``."""
    return [x for x in api.store.neighbors(id, rel, direction="in") if api.store.kind_of(x) == kind]


def latest(api: Api, ids: list[str]) -> str | None:
    if not ids:
        return None
    return max(ids, key=lambda x: api.store.get_json(x)["created_at"])


def config_int(api: Api, section: str, key: str) -> int:
    value = api.config.section(section).get(key)
    if not isinstance(value, int):
        raise ConfigError(f"config [{section}] {key} must be an integer")
    return value
