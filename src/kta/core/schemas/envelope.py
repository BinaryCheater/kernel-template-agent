"""The record envelope (spec/schemas.md §1, docs/design/20 §2.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kta.core.ids import is_record_id
from kta.core.schemas.common import ActorType, Origin
from kta.core.schemas.registry import InvalidBody, kind_spec, normalize_body, parse_body

RefValue = str | list[str]


class Actor(BaseModel):
    """Who produced a record.

    ``commit`` and ``dirty`` identify the code that wrote it (decision D84). The Store
    fills them from its own code state when they are absent; records produced
    elsewhere (a worker) keep the values they were created with.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ActorType
    name: str
    task: str | None = None
    commit: str | None = None
    dirty: bool | None = None

    @model_validator(mode="after")
    def _commit_and_dirty_together(self) -> Actor:
        if (self.commit is None) != (self.dirty is None):
            raise ValueError("actor.commit and actor.dirty are set together")
        return self


class Record(BaseModel):
    """An immutable evidence record; its ID is the sha256 of its canonical JSON.

    The body is validated against the kind's model and stored in normalized form
    (defaults filled in), so equal content always yields the same ID. The JSON field
    ``schema`` is ``schema_`` in Python because ``schema`` shadows a BaseModel attribute.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    kind: str
    schema_: int = Field(alias="schema", ge=1)
    created_at: datetime
    actor: Actor
    refs: dict[str, RefValue] = Field(default_factory=lambda: dict[str, RefValue]())
    body: dict[str, Any]
    blobs: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    origin: Origin

    @field_validator("created_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (UTC)")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _check(self) -> Record:
        spec = kind_spec(self.kind)
        if not 1 <= self.schema_ <= spec.version:
            raise InvalidBody(f"{self.kind} schema {self.schema_}; current is {spec.version}")
        for key, value in self.refs.items():
            for rid in [value] if isinstance(value, str) else value:
                if not is_record_id(rid):
                    raise InvalidBody(f"refs.{key}: {rid!r} is not a record ID")
        for name, sha in self.blobs.items():
            if not is_record_id(sha):
                raise InvalidBody(f"blobs.{name}: {sha!r} is not a sha256")
        missing = [k for k in spec.required_refs if k not in self.refs]
        if missing:
            raise InvalidBody(f"{self.kind} needs refs {missing}")
        if self.kind == "Template" and self.actor.type != "human" and not self.refs.get("sources"):
            raise InvalidBody("Template needs refs.sources unless the actor is human")
        if self.schema_ < spec.version:
            # A stored older body (D461): validate its upcast form, keep the body as stored
            # so the record's ID does not change. Readers parse it with ``body_model``.
            parse_body(self.kind, self.body, self.schema_)
            return self
        # Normalize in place: the model is frozen, but validation is construction time.
        object.__setattr__(self, "body", normalize_body(self.kind, self.body))
        return self

    def body_model(self) -> BaseModel:
        """The body as the current schema's model (older bodies upcast, D461)."""
        return parse_body(self.kind, self.body, self.schema_)

    @classmethod
    def new(
        cls,
        kind: str,
        body: BaseModel | dict[str, Any],
        actor: Actor,
        created_at: datetime,
        refs: dict[str, RefValue] | None = None,
        blobs: dict[str, str] | None = None,
        origin: Origin = "real",
    ) -> Record:
        data = body.model_dump(mode="json") if isinstance(body, BaseModel) else body
        return cls(
            kind=kind,
            schema=kind_spec(kind).version,
            created_at=created_at,
            actor=actor,
            refs=dict(refs or {}),
            body=data,
            blobs=dict(blobs or {}),
            origin=origin,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """The stored JSON form (``schema`` under its JSON name); the ID is computed from it."""
        return self.model_dump(mode="json", by_alias=True)

    def ref_ids(self) -> list[tuple[str, str]]:
        """(rel, id) pairs from refs, list values expanded."""
        pairs: list[tuple[str, str]] = []
        for key, value in self.refs.items():
            pairs += [(key, value)] if isinstance(value, str) else [(key, v) for v in value]
        return pairs
