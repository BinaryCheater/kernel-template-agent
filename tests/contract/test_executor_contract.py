"""The Executor stand-in has the real JobExecutor's method signatures (and, once WP2's
``cancel`` is merged, the same ``cancel``)."""

from __future__ import annotations

import inspect

import pytest

from fakes.executor import RecordingExecutor
from kta.core.jobs import JobRunner
from kta.execution import JobExecutor

METHODS = ("submit", "wait", "run", "read_blob", "put_blob")


def _sig(cls: type, name: str) -> list[str]:
    return list(inspect.signature(getattr(cls, name)).parameters)


@pytest.mark.contract
@pytest.mark.parametrize("cls", [JobExecutor, RecordingExecutor], ids=lambda c: c.__name__)
def test_executor_has_the_job_runner_methods(cls: type) -> None:
    for name in METHODS:
        assert _sig(cls, name) == _sig(JobRunner, name), name


@pytest.mark.contract
def test_cancel_signature() -> None:
    assert _sig(RecordingExecutor, "cancel") == ["self", "job_id"]
    real = getattr(JobExecutor, "cancel", None)
    if real is not None:  # WP2's cancel(job_id) -> None
        assert list(inspect.signature(real).parameters) == ["self", "job_id"]
