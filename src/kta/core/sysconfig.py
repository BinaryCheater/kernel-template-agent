"""System configuration of the control plane (WP22, D580): infrastructure, not research knobs.

The controls registry (``core/controls.yaml``) holds versioned research knobs that runs
record and experiments set. Proxy settings are a property of the machines, so they live
here instead: one JSON file ``<data dir>/system/config.json`` with a revision counter.
The API is its only writer (operator token, a reason, the revision the author read); each
write also leaves a HumanAction ``set_system`` record, which is the audit trail. Absent
file = the defaults below. Nothing here may hold a credential: proxy URLs are refused when
they carry user:password (that part lives in ``secrets.env``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kta.core.errors import ConfigError
from kta.core.proxy import parse_url

FILE = Path("system") / "config.json"
Policy = Literal["required", "off"]
# Defaults the user set (2026-10-31): Codex and Claude Code always through the proxy,
# pi and the DeepSeek harness never. Hosts are what the preflight tunnels to.
DEFAULT_PROVIDERS: dict[str, dict[str, Any]] = {
    "codex": {"proxy": "required", "check_hosts": ["chatgpt.com:443", "api.openai.com:443"]},
    "claude_code": {"proxy": "required", "check_hosts": ["api.anthropic.com:443"]},
    "pi": {"proxy": "off", "check_hosts": []},
    "deepseek_harness": {"proxy": "off", "check_hosts": []},
}


def _no_credentials(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    try:
        _, creds = parse_url(value)
    except ConfigError as exc:
        raise ValueError(str(exc)) from exc
    if creds:
        raise ValueError(
            "a proxy URL here must not carry user:password; put it in secrets.env "
            "as KTA_PROXY_USERINFO"
        )
    return value.strip()


class ProxySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    override: str | None = None  # None: auto-detect (scutil, environment, Clash default)
    credentials_from_secrets: bool = False  # add KTA_PROXY_USERINFO on the agent host
    no_proxy_extra: list[str] = Field(default_factory=lambda: list[str]())

    @field_validator("override")
    @classmethod
    def _plain(cls, v: str | None) -> str | None:
        return _no_credentials(v)


class ProviderProxy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proxy: Policy
    check_hosts: list[str] = Field(default_factory=lambda: list[str]())


class WorkerProxy(BaseModel):
    """Per-worker proxy for intake fetches only (git clone, package downloads); default off."""

    model_config = ConfigDict(extra="forbid")
    intake_proxy: str | None = None

    @field_validator("intake_proxy")
    @classmethod
    def _plain(cls, v: str | None) -> str | None:
        return _no_credentials(v)


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = 0
    updated_at: str | None = None
    updated_by: str | None = None
    reason: str | None = None
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    providers: dict[str, ProviderProxy] = Field(
        default_factory=lambda: {k: ProviderProxy(**v) for k, v in DEFAULT_PROVIDERS.items()}
    )
    workers: dict[str, WorkerProxy] = Field(default_factory=lambda: dict[str, WorkerProxy]())

    def provider(self, name: str) -> ProviderProxy:
        """The provider's policy; a provider the table does not name goes direct (off)."""
        return self.providers.get(name, ProviderProxy(proxy="off"))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def path(data_dir: Path) -> Path:
    return data_dir / FILE


def _validate(data: Mapping[str, Any]) -> SystemConfig:
    try:
        cfg = SystemConfig.model_validate(dict(data))
    except ValidationError as exc:
        raise ConfigError(f"system configuration: {exc.errors()[0]['msg']}") from exc
    # Providers the defaults know stay present even when a stored file predates them.
    for name, default in DEFAULT_PROVIDERS.items():
        cfg.providers.setdefault(name, ProviderProxy(**default))
    return cfg


def load(data_dir: Path) -> SystemConfig:
    p = path(data_dir)
    if not p.is_file():
        return _validate({})
    try:
        return _validate(json.loads(p.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{p} is not valid JSON: {exc}") from exc


def apply(cfg: SystemConfig, changes: Mapping[str, Any]) -> SystemConfig:
    """``changes`` merges field by field: ``proxy``, ``providers.<name>``, ``workers.<name>``."""
    unknown = set(changes) - {"proxy", "providers", "workers"}
    if unknown:
        raise ConfigError(f"system configuration has no section {sorted(unknown)}")
    data = cfg.to_dict()
    if "proxy" in changes:
        data["proxy"] = {**data["proxy"], **dict(changes["proxy"])}
    for section in ("providers", "workers"):
        for name, patch in dict(changes.get(section) or {}).items():
            base = data[section].get(name, {})
            data[section][name] = {**base, **dict(patch)}
    return _validate(data)


def save(
    data_dir: Path,
    changes: Mapping[str, Any],
    expected: int,
    author: str,
    at: datetime,
    reason: str,
) -> SystemConfig:
    """Write a new revision; refused when ``expected`` is not the revision on disk."""
    if not reason.strip():
        raise ConfigError("a system configuration change needs a reason")
    current = load(data_dir)
    if current.revision != expected:
        raise ConfigError(
            f"system configuration is at revision {current.revision}, not {expected}: reload"
        )
    new = apply(current, changes).model_copy(
        update={
            "revision": current.revision + 1,
            "updated_at": at.isoformat(),
            "updated_by": author,
            "reason": reason.strip(),
        }
    )
    p = path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return new
