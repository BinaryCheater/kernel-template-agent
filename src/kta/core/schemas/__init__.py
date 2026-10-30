"""Pydantic models for every record kind (spec/schemas.md).

``validate_body(kind, body)`` raises ``InvalidBody`` (a ``SchemaError`` and a
``ValueError``) when a body does not match its kind.
"""

from kta.core.schemas.common import (
    ActorType,
    Assurance,
    Binding,
    Body,
    ChangeKind,
    EvidenceStatus,
    Level,
    Origin,
    Unit,
    VarKind,
    VarRole,
)
from kta.core.schemas.envelope import Actor, Record, RefValue
from kta.core.schemas.registry import (
    KINDS,
    InvalidBody,
    KindSpec,
    kind_spec,
    model_for,
    normalize_body,
    parse_body,
    validate_body,
)

__all__ = [
    "KINDS",
    "Actor",
    "ActorType",
    "Assurance",
    "Binding",
    "Body",
    "ChangeKind",
    "EvidenceStatus",
    "InvalidBody",
    "KindSpec",
    "Level",
    "Origin",
    "Record",
    "RefValue",
    "Unit",
    "VarKind",
    "VarRole",
    "kind_spec",
    "model_for",
    "normalize_body",
    "parse_body",
    "validate_body",
]
