"""Error hierarchy shared by all modules (spec/conventions.md §3.3)."""

from __future__ import annotations

from typing import Literal

ExecutionKind = Literal["timeout", "oom", "gpu_hang", "infra", "nonzero"]


class KtaError(Exception):
    """Base class; every error KTA raises on purpose derives from it."""


class ConfigError(KtaError):
    pass


class SchemaError(KtaError, ValueError):
    """A record or knowledge object does not match its schema.

    Also a ValueError, so callers that validate data (and pydantic validators) can
    treat schema problems like any other invalid value.
    """


class NotSupported(KtaError):
    """The current Target, backend or environment cannot provide this capability."""


class ExecutionError(KtaError):
    def __init__(self, kind: ExecutionKind, message: str) -> None:
        super().__init__(f"[{kind}] {message}")
        self.kind: ExecutionKind = kind


class CheckerError(KtaError):
    """The checker itself failed; distinct from a check that did not pass."""


class BudgetExceeded(KtaError):
    pass


class PermissionDenied(KtaError):
    pass
