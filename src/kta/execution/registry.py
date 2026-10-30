"""Worker registration, heartbeat and capability-label matching.

A worker is registered by probing it through its transport (``worker info``): the
control plane writes an EnvManifest (role ``worker``) and a Worker record whose labels
come from the probe, never from configuration alone. The Executor dispatches only to
the latest registration of each configured worker, and only when that registration's
commit is the control plane's commit (decision D197).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from kta.core.context import Context
from kta.core.errors import ConfigError, ExecutionError
from kta.core.ids import record_id
from kta.core.registry import resolve
from kta.core.schemas import Actor, Record
from kta.core.schemas.hardware import Worker
from kta.execution.base import TransportName, WorkerClient, WorkerConfig
from kta.execution.transport import make_transport
from kta.store import code_state
from kta.store.base import Ops, Store

log = logging.getLogger("kta.execution.registry")
ACTOR = Actor(type="system", name="kta worker register")


def worker_configs(ctx: Context) -> dict[str, WorkerConfig]:
    section = ctx.config.section("workers")
    out: dict[str, WorkerConfig] = {}
    for name, data in section.items():
        if not isinstance(data, Mapping):
            raise ConfigError(f"[workers.{name}] must be a table")
        out[str(name)] = WorkerConfig.from_mapping(str(name), dict(data))  # pyright: ignore[reportUnknownArgumentType]
    return out


def data_root(ctx: Context) -> str:
    """The control plane's KTA_DATA (the local transport shares it with the worker)."""
    if ctx.env == "test":
        return str(ctx.data_dir)
    root = os.environ.get("KTA_DATA")
    if not root:
        raise ConfigError("KTA_DATA is not set")
    return root


def client_for(ctx: Context, config: WorkerConfig, commit: str) -> WorkerClient:
    wp = os.environ.get("KTA_WP") if ctx.env == "dev" else None
    transport = make_transport(config, ctx.env, wp, Path(data_root(ctx)), commit)
    return WorkerClient(config, transport)


def label_matches(labels: Mapping[str, str], requires: Mapping[str, str]) -> bool:
    """``k=v`` matches a label ``k`` equal to ``v`` or a label key ``k.v`` (D182).

    Multi-valued capabilities are keys: ``pool.gpu-shared``, ``backend.tilelang``.
    """
    return all(labels.get(k) == v or f"{k}.{v}" in labels for k, v in requires.items())


@dataclass(frozen=True)
class Registration:
    worker_id: str
    env_id: str
    worker: Worker


def register(
    ctx: Context, store: Store, ops: Ops, config: WorkerConfig, commit: str
) -> Registration:
    """Probe a worker and write its EnvManifest and Worker records."""
    info: dict[str, Any] = client_for(ctx, config, commit).info()
    if info["commit"] != commit:
        raise ExecutionError(
            "infra", f"{config.name} runs commit {info['commit']}, control plane {commit}"
        )
    now = ctx.clock.now()
    env_id = store.put(Record.new("EnvManifest", info["manifest"], ACTOR, now))
    body = Worker(
        name=config.name,
        host=str(info["host"]),
        transport=config.transport,
        labels=dict(info["labels"]),
        devices=list(info["devices"]),
        commit=str(info["commit"]),
    )
    worker_id = store.put(Record.new("Worker", body, ACTOR, now, refs={"env": env_id}))
    ops.heartbeat(config.name, body.host, True)
    log.info(
        "registered worker %s (%s) labels=%s", config.name, worker_id[:12], sorted(body.labels)
    )
    return Registration(worker_id, env_id, body)


def register_worker(ctx: Context, name: str, host: str, transport: TransportName) -> Worker:
    """spec/modules/target-exec.md: probe capability labels, write Worker and EnvManifest."""
    configs = worker_configs(ctx)
    if name not in configs:
        raise ConfigError(f"no [workers.{name}] in {ctx.config.path}")
    config = replace(configs[name], transport=transport)
    if transport == "ssh":
        config = replace(config, ssh=host)
    store: Store = resolve("store", ctx)
    ops: Ops = resolve("ops", ctx)
    return register(ctx, store, ops, config, code_state().commit).worker


def latest_registration(store: Store, name: str) -> Registration | None:
    found = store.query("Worker", {"name": name}, order="-created_at", limit=1)
    if not found:
        return None
    rec = found[0]
    env = rec.refs.get("env")
    if not isinstance(env, str):
        return None
    return Registration(record_id(rec), env, Worker.model_validate(rec.body))
