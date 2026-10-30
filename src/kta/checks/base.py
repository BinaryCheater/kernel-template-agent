"""Shared parts of the checkers: the Checker context, Check records, checker errors.

Every check function writes one Check record and returns it with its ID (``Checked``,
decision D224). "Did not pass" is a normal result. A checker that cannot reach a
verdict (its own job failed on infrastructure, its reference failed, its tool printed
no summary) records ``error_kind = "checker_error"`` in the failure item, so callers
park the task instead of blaming the kernel (spec/modules/measure-check.md).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kta.core.context import Clock, Context
from kta.core.jobs import JobResult, JobRunner
from kta.core.registry import resolve
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Check
from kta.store.base import Store

CHECKER_ERROR = "checker_error"
RngFactory = Callable[[str], random.Random]


def actor(name: str, version: int = 1) -> Actor:
    return Actor(type="checker", name=f"{name}@{version}")


@dataclass(frozen=True)
class Checked:
    id: str
    body: Check

    @property
    def passed(self) -> bool:
        return self.body.passed

    @property
    def checker_error(self) -> bool:
        return any(f.get("error_kind") == CHECKER_ERROR for f in self.body.failures)


@dataclass(frozen=True)
class Checker:
    store: Store
    ex: JobRunner
    clock: Clock
    rng: RngFactory
    private_dir: Path  # holds the hidden-shape seed; never mounted into agent workspaces
    seed: int = 0
    image: str | None = None  # worker image with torch, for FP64 references

    @classmethod
    def from_ctx(cls, ctx: Context) -> Checker:
        return cls(
            resolve("store", ctx),
            resolve("executor", ctx),
            ctx.clock,
            ctx.rng,
            ctx.data_dir / "checker",
            ctx.seed,
        )


@dataclass
class Evidence:
    """What a check accumulates before its record is written."""

    failures: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    blobs: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    refs: dict[str, str | list[str]] = field(default_factory=lambda: dict[str, str | list[str]]())
    detail: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def job(self, result: JobResult, prefix: str) -> None:
        """Keep the job's logs and tie the record to its worker and environment."""
        self.refs.setdefault("env", result.env_manifest)
        self.refs.setdefault("worker", result.worker)
        self.blobs[f"{prefix}_stdout"] = result.stdout_blob
        self.blobs[f"{prefix}_stderr"] = result.stderr_blob

    def error(self, check: str, detail: str) -> None:
        self.failures.append({"check": check, "error_kind": CHECKER_ERROR, "detail": detail})


def write_check(c: Checker, name: str, ev: Evidence, build_id: str) -> Checked:
    body = Check(passed=not ev.failures, failures=ev.failures)
    refs = {**ev.refs, "build": build_id}
    blobs = dict(ev.blobs)
    if ev.detail:
        blobs["detail"] = c.ex.put_blob(
            json.dumps(ev.detail, sort_keys=True, default=str).encode(), "application/json"
        )
    rec = Record.new("Check", body, actor(name), c.clock.now(), refs=refs, blobs=blobs)
    return Checked(c.store.put(rec), body)


def stderr_tail(c: Checker, result: JobResult, chars: int = 1500) -> str:
    return c.ex.read_blob(result.stderr_blob).decode("utf-8", "replace")[-chars:]
