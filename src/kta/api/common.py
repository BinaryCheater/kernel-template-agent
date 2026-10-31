"""Shared helpers for the read views: record envelopes with derived status, small lookups."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from kta.api.wiring import Orchestration, Unwired
from kta.core.config import Config
from kta.core.context import Context
from kta.core.controls import Controls, definitions
from kta.core.errors import ConfigError
from kta.store import SqliteOps, SqliteStore
from kta.store import controls as controls_store
from kta.store.assurance import effective


@dataclass(frozen=True)
class Api:
    """What every view needs: the context (config, environment, data directory), the store
    and the ops tables. ``orchestration`` carries the operations delegated to WP6/WP8
    (wiring.py); ``open_api`` wires it, ``Unwired`` is only the default for bare tests.
    """

    ctx: Context
    store: SqliteStore
    ops: SqliteOps
    orchestration: Orchestration = field(default_factory=Unwired)

    @property
    def config(self) -> Config:
        return self.ctx.config

    @property
    def env(self) -> str:
        return self.ctx.env

    @property
    def data_dir(self) -> Path:
        return self.ctx.data_dir

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


def controls(api: Api) -> Controls:
    """The controls version in force (read only: nothing is written by a GET)."""
    return controls_store.current(api.store, api.ops)


def _limit(c: Controls, raw: Any) -> Any:
    # machines.yaml names controls knobs for its numeric limits (WP21): show the value in force
    return c.value(raw) if isinstance(raw, str) and raw in definitions().ids() else raw


def machines_view(api: Api) -> dict[str, Any]:
    """machines.yaml with each limit that names a controls knob replaced by its value;
    ``limit_controls`` keeps the knob names."""
    c = controls(api)
    out = dict(machines())
    resolved: list[dict[str, Any]] = []
    for m in out["machines"]:
        limits: dict[str, Any] = dict(m.get("limits") or {})
        named = {k: v for k, v in limits.items() if isinstance(v, str) and v in definitions().ids()}
        resolved.append(
            {**m, "limits": {k: _limit(c, v) for k, v in limits.items()}, "limit_controls": named}
        )
    out["machines"] = resolved
    return out


def machine_limit(api: Api, machine: str, key: str) -> int | None:
    for m in machines()["machines"]:
        if m["id"] == machine:
            raw = m.get("limits", {}).get(key)
            return None if raw is None else int(_limit(controls(api), raw))
    return None


# kinds whose assurance is read through proof records (WP24, D634)
ASSURED = frozenset({"Template", "Family", "Chain"})


def envelope(api: Api, id: str) -> dict[str, Any]:
    """GET /records/{id}: the stored JSON plus derived status and assurance.

    A record stored under an older schema shows its upcast ``body`` (the current shape)
    with the stored one under ``stored_body`` and ``upcast_from`` (D440).
    """
    data = api.store.get_json(id)
    status, raw = api.store.status(id)
    assurance = effective(api.store, id) if data["kind"] in ASSURED else raw
    out = {"id": id, **data, "status": status, "assurance": assurance}
    rec = api.store.get(id)
    if rec.upcast:
        out |= {"body": rec.body, "stored_body": data["body"], "upcast_from": data["schema"]}
    return out


def body(api: Api, id: str) -> dict[str, Any]:
    """The record body in the current schema (upcast when stored older, D440)."""
    return api.store.get(id).body


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


def done_check_reports(api: Api, task_id: str) -> list[dict[str, Any]]:
    """The task's done-check verdicts: Reports titled ``done_checks`` written by the
    orchestrator in the task (D276), oldest first, with the failed checks summarised."""
    with api.store.db.read() as con:
        rows = con.execute(
            "SELECT id FROM records WHERE task = ? AND kind = 'Report' ORDER BY created_at, rowid",
            (task_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for (rid,) in rows:
        rec = api.store.get_json(rid)
        b = rec["body"]
        if b["title"] != "done_checks":
            continue
        checks = b["data"].get("checks", [])
        failed = [c for c in checks if not c.get("passed")]
        out.append(
            {
                "id": rid,
                "created_at": rec["created_at"],
                "actor": rec["actor"]["name"],
                "passed": bool(b["data"].get("passed")),
                "summary": b["summary"],
                "checks": checks,
                "reason": "; ".join(
                    f"{c['name']}: {(c.get('failures') or ['failed'])[0]}" for c in failed
                )
                or b["summary"],
            }
        )
    return out
