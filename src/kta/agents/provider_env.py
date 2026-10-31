"""Reuse host provider connections without importing personal agent instructions.

Runs on the agent host. Connection values stay there: manifests contain source hashes
and key names, and the merged config is excluded from session archives.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

from kta.core.errors import ConfigError

MARKER = ".kta-provider.json"
CONNECTION_KEYS = frozenset(
    {
        "model_provider",
        "model_providers",
        "openai_base_url",
        "chatgpt_base_url",
        "cli_auth_credentials_store",
        "forced_login_method",
        "forced_chatgpt_workspace_id",
    }
)


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()


def auth_directory(configured: str) -> Path:
    return Path(configured).expanduser() if configured else codex_home()


def command(name: str, configured: list[str] | None = None) -> list[str]:
    if configured:
        return configured
    found = shutil.which(name)
    local = Path.home() / ".local" / "bin" / name
    return [found or (str(local) if local.is_file() else name)]


def _toml(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {_toml(v)}" for k, v in value.items()) + "}"
    raise ConfigError(f"unsupported provider configuration value: {type(value).__name__}")


def inherit_connection(home: Path, source: Path) -> dict[str, Any]:
    """Pin connection settings once per experimental home; profile values override them."""
    marker = home / MARKER
    if marker.is_file():
        return json.loads(marker.read_text())
    source = source / "config.toml"
    inherited: dict[str, Any] = {}
    raw = b""
    try:
        if source.is_file():
            raw = source.read_bytes()
            inherited = {
                k: v for k, v in tomllib.loads(raw.decode()).items() if k in CONNECTION_KEYS
            }
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read native provider configuration at {source}") from exc
    target = home / "config.toml"
    if inherited:
        try:
            profile = tomllib.loads(target.read_text()) if target.is_file() else {}
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Cannot read experimental configuration at {target}") from exc
        merged = {**inherited, **profile}
        pending = home / ".kta-config.pending"
        pending.touch(mode=0o600, exist_ok=True)
        pending.write_text(
            "\n".join(f"{json.dumps(k)} = {_toml(v)}" for k, v in merged.items()) + "\n"
        )
        pending.replace(target)
    # No connection values or credentials enter metadata.
    meta = {
        "source": str(source),
        "sha256": hashlib.sha256(raw).hexdigest() if raw else None,
        "keys": sorted(inherited),
    }
    marker.write_text(json.dumps(meta, sort_keys=True))
    return meta
