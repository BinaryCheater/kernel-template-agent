"""System-configuration helpers for tests that run the real launcher without network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kta.core import sysconfig


def direct(data_dir: Path, provider: str = "codex") -> None:
    """Set ``provider`` to proxy ``off`` so a test's launch makes no network connection."""
    cfg = sysconfig.load(data_dir)
    sysconfig.save(
        data_dir,
        {"providers": {provider: {"proxy": "off"}}},
        cfg.revision,
        "tests",
        datetime.now(UTC),
        "unit tests run without network (WP22)",
    )


def proxy_at(data_dir: Path, url: str, hosts: list[str], provider: str = "codex") -> None:
    """Require the proxy for ``provider`` with an override URL and preflight hosts."""
    cfg = sysconfig.load(data_dir)
    sysconfig.save(
        data_dir,
        {
            "proxy": {"override": url},
            "providers": {provider: {"proxy": "required", "check_hosts": hosts}},
        },
        cfg.revision,
        "tests",
        datetime.now(UTC),
        "test proxy",
    )
