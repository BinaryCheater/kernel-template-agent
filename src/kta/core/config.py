"""TOML configuration loading and hashing."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kta.core.errors import ConfigError

CONFIG_ENV_VAR = "KTA_CONFIG"
# A deployment mode names config/modes/<mode>.toml (WP23, D600): dev-mac, spark-hk.
MODE_ENV_VAR = "KTA_MODE"


def repo_root() -> Path:
    """Root of the deployed source tree (works for editable installs and worker deployments)."""
    return Path(__file__).resolve().parents[3]


def default_config_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    explicit = env.get(CONFIG_ENV_VAR)
    if explicit:
        return Path(explicit)
    mode = env.get(MODE_ENV_VAR)
    if mode:
        path = repo_root() / "config" / "modes" / f"{mode}.toml"
        if not path.is_file():
            known = sorted(p.stem for p in (repo_root() / "config" / "modes").glob("*.toml"))
            raise ConfigError(f"{MODE_ENV_VAR}={mode}: no {path.name}; modes: {', '.join(known)}")
        return path
    return repo_root() / "config" / "kta.toml"


@dataclass(frozen=True)
class Config:
    path: Path | None
    data: Mapping[str, Any] = field(default_factory=lambda: dict[str, Any]())
    hash: str = ""

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"config section [{name}] must be a table")
        return value  # pyright: ignore[reportUnknownVariableType]

    def impl(self, interface: str) -> str | None:
        """Implementation name chosen in the [impl] section, if any."""
        value = self.section("impl").get(interface)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError(f"[impl] {interface} must be a string")
        return value


def config_hash(data: Mapping[str, Any]) -> str:
    # Hash the parsed content, not the file bytes, so comments and key order do not matter.
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_from_mapping(data: Mapping[str, Any], path: Path | None = None) -> Config:
    return Config(path=path, data=dict(data), hash=config_hash(data))


def load_config(path: Path) -> Config:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return config_from_mapping(data, path)
