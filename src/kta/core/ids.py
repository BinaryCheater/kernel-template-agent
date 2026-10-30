"""Content-hash record IDs and ULIDs for runtime objects (spec/schemas.md §1, conventions §3.4).

``canonical_json``: sorted keys, no extra whitespace, UTF-8 (non-ASCII kept as is),
datetimes as ISO 8601 in UTC, floats via ``repr`` (Python's shortest round-trip form,
which ``json`` already uses). NaN and infinity are refused because they have no
JSON form that every reader agrees on.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kta.core.errors import SchemaError

if TYPE_CHECKING:
    from kta.core.context import Clock
    from kta.core.schemas.envelope import Record

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise SchemaError(f"naive datetime {value!r}; timestamps must be timezone-aware UTC")
    return value.astimezone(UTC).isoformat()


def _default(value: object) -> object:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, (set, frozenset)):
        raise SchemaError("sets have no canonical order; use a sorted list")
    raise SchemaError(f"value of type {type(value).__name__} has no canonical JSON form")


def canonical_json(obj: Any) -> bytes:
    try:
        text = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_default,
        )
    except ValueError as exc:
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError(f"cannot canonicalize: {exc}") from exc
    return text.encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_id(record: Record | Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of the whole record (envelope and body)."""
    data = record if isinstance(record, Mapping) else record.to_json_dict()
    return sha256_hex(canonical_json(data))


def is_record_id(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class UlidFactory:
    """ULIDs from the context clock.

    The 80 random bits come from ``randbytes`` (default ``os.urandom``), not from
    ``ctx.rng``: IDs must be unique across processes started with the same seed, and
    they never influence results. Tests pass a deterministic ``randbytes``.
    """

    def __init__(self, clock: Clock, randbytes: Callable[[int], bytes] = os.urandom) -> None:
        self._clock = clock
        self._randbytes = randbytes

    def new(self) -> str:
        ms = int(self._clock.now().timestamp() * 1000)
        if not 0 <= ms < 1 << 48:
            raise SchemaError(f"timestamp {ms} ms is outside the ULID range")
        value = (ms << 80) | int.from_bytes(self._randbytes(10), "big")
        return "".join(_CROCKFORD[(value >> (5 * i)) & 31] for i in reversed(range(26)))


def ulid_time(ulid: str) -> datetime:
    """Timestamp encoded in a ULID (millisecond precision)."""
    value = 0
    for ch in ulid.upper():
        value = value * 32 + _CROCKFORD.index(ch)
    return datetime.fromtimestamp((value >> 80) / 1000, UTC)
