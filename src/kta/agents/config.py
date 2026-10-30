"""The ``[agents]`` configuration section (paths come from config, conventions §3.1).

```toml
[agents]
worker = "spark"                    # Executor worker that runs agent CLIs (D300)
pool = "agent"                      # its pool for agent jobs ([workers.<w>] pools)
agent_user = "kta-agent"            # system user of config+user isolation (D09)
token_key_file = "~/.config/kta/task-token.key"   # on the control host
api_url = ""                        # KTA_API given to the agent-side kta (WP6/WP9)
secrets_file = "~/.config/kta/secrets.env"
watch = ["~/.codex", "~"]           # host paths checked for writes after each run

[agents.hosts.spark]                # per agent host
codex_cmd = ["/path/to/node", "/path/to/codex"]
auth_dir = "~/.config/kta/auth/codex"
nvidia_smi = "nvidia-smi"
```
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.core.context import Context
from kta.core.errors import ConfigError

DEFAULT_WATCH = ("~/.codex", "~")


@dataclass(frozen=True)
class HostConfig:
    codex_cmd: list[str] = field(default_factory=lambda: list[str]())
    auth_dir: str = "~/.config/kta/auth/codex"
    nvidia_smi: str = "nvidia-smi"


@dataclass(frozen=True)
class AgentsConfig:
    worker: str
    pool: str = "agent"
    agent_user: str | None = None
    token_key_file: str = "~/.config/kta/task-token.key"
    api_url: str = ""
    secrets_file: str = "~/.config/kta/secrets.env"
    watch: list[str] = field(default_factory=lambda: list(DEFAULT_WATCH))
    hosts: dict[str, HostConfig] = field(default_factory=lambda: dict[str, HostConfig]())

    @property
    def host(self) -> HostConfig:
        if self.worker not in self.hosts:
            raise ConfigError(f"no [agents.hosts.{self.worker}] section in the configuration")
        return self.hosts[self.worker]


def _host(name: str, data: Mapping[str, Any]) -> HostConfig:
    known = set(HostConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"[agents.hosts.{name}] unknown keys {sorted(unknown)}")
    cmd = data.get("codex_cmd", [])
    if not isinstance(cmd, list) or not all(isinstance(c, str) for c in cmd):  # pyright: ignore[reportUnknownVariableType]
        raise ConfigError(f"[agents.hosts.{name}] codex_cmd must be a list of strings")
    return HostConfig(**{**data, "codex_cmd": list(cmd)})  # pyright: ignore[reportUnknownArgumentType]


def agents_config(ctx: Context) -> AgentsConfig:
    data = dict(ctx.config.section("agents"))
    if "worker" not in data:
        raise ConfigError("[agents] worker is not set: name the Executor worker that runs agents")
    hosts = {str(k): _host(str(k), v) for k, v in dict(data.pop("hosts", {})).items()}
    known = set(AgentsConfig.__dataclass_fields__) - {"hosts"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"[agents] unknown keys {sorted(unknown)}")
    if "watch" in data:
        data["watch"] = [str(w) for w in data["watch"]]
    return AgentsConfig(**data, hosts=hosts)
