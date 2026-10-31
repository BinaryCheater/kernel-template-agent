"""``python -m kta.api``: serve the API on the control host (``[api]`` host and port in config)."""

from __future__ import annotations

import logging
import os

import uvicorn

from kta.api.app import create_app, open_api
from kta.api.common import config_int
from kta.core.config import default_config_path, load_config
from kta.core.context import Context, data_dir_for, parse_env
from kta.core.errors import ConfigError
from kta.core.logging import setup
from kta.core.registry import build_registry
from kta.store.backup import check_owner


def main() -> None:
    env = parse_env(os.environ.get("KTA_ENV"))
    ctx = Context(
        env=env,
        config=load_config(default_config_path()),
        data_dir=data_dir_for(env),
        registry=build_registry(),
        process="api",
    )
    setup(ctx)
    check_owner(ctx.data_dir)  # one control plane per store (WP23, D601)
    api = open_api(ctx)
    host = api.config.section("api").get("host")
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ConfigError(
            f"[api] host must be a loopback address, got {host!r} (api-web.md: local port only)"
        )
    logging.getLogger("kta.api").info("serving", extra={"data_dir": str(ctx.data_dir)})
    uvicorn.run(
        create_app(api), host=str(host), port=config_int(api, "api", "port"), log_level="warning"
    )


if __name__ == "__main__":
    main()
