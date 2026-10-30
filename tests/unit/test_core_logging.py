from __future__ import annotations

import dataclasses
import json
import logging

from kta.core.context import Context
from kta.core.logging import bind, setup


def test_json_lines_with_context_and_redaction(ctx: Context) -> None:
    dev_ctx = dataclasses.replace(ctx, env="dev", process="unit")
    path = setup(dev_ctx)
    log = logging.getLogger("kta.test")
    secret = "sk-ant-" + "x" * 40
    with bind(task_id="T1", job_id="J9"):
        log.debug("starting with key %s", secret)
    log.info("outside")
    for handler in logging.getLogger("kta").handlers:
        handler.flush()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.parent.name == "unit" and path.parent.parent.name == "logs"
    first, second = lines
    for key in ("ts", "level", "logger", "msg", "env", "process"):
        assert key in first
    assert first["task_id"] == "T1" and first["job_id"] == "J9" and first["env"] == "dev"
    assert secret not in first["msg"] and "[REDACTED]" in first["msg"]
    assert "task_id" not in second


def test_debug_only_in_dev(ctx: Context) -> None:
    setup(dataclasses.replace(ctx, env="test"))
    assert logging.getLogger("kta").level == logging.INFO


def test_secret_patterns_skip_hyphenated_names() -> None:
    from kta.core.secrets import find_secrets

    assert find_secrets("task-" + "attention-medium-template-round-one-feedback.json") == []
    assert find_secrets("key " + "sk-" + "proj-" + "Ab1" * 12) != []
