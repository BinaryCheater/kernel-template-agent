"""Context object: the only way dependencies reach code (spec/conventions.md §3.1).

WP0 provides environment, configuration, data directory, registry, clock and rng;
WP21 adds the controls snapshot. Store, ops and executor are resolved from the registry.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from kta.core.config import Config
from kta.core.controls import Controls, baseline
from kta.core.errors import ConfigError
from kta.core.registry import Registry

Env = Literal["test", "dev", "main"]
ENVS: tuple[Env, ...] = ("test", "dev", "main")


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def parse_env(value: str | None) -> Env:
    if value not in ENVS:
        raise ConfigError(f"KTA_ENV must be one of {', '.join(ENVS)}; got {value!r}")
    return value


def data_dir_for(env: Env, environ: Mapping[str, str] | None = None) -> Path:
    """Data directory per spec/conventions.md §2; test data dirs are created by the test harness."""
    source = os.environ if environ is None else environ
    if env == "test":
        explicit = source.get("KTA_TEST_DATA")
        if not explicit:
            raise ConfigError("KTA_ENV=test needs KTA_TEST_DATA (a temporary directory)")
        return Path(explicit)
    root = source.get("KTA_DATA")
    if not root:
        raise ConfigError("KTA_DATA is not set")
    if env == "main":
        return Path(root) / "main"
    wp = source.get("KTA_WP")
    if not wp:
        raise ConfigError("KTA_ENV=dev needs KTA_WP (the work package, e.g. WP2)")
    return Path(root) / "dev" / wp


@dataclass(frozen=True)
class Context:
    env: Env
    config: Config
    data_dir: Path
    registry: Registry
    clock: Clock = field(default_factory=SystemClock)
    seed: int = 0
    process: str = "kta"
    # The controls snapshot pure code reads (WP21, D482). Entry points that hold a store
    # replace it with the stored version in force (kta.store.controls.ensure).
    controls: Controls = field(default_factory=baseline)

    def rng(self, name: str) -> random.Random:
        """Independent, reproducible stream per name; record the seed where it is used."""
        digest = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))
