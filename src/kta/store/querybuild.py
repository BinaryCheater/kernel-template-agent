"""``Store.query`` filters and ordering to SQL (decision D147).

``where`` maps a dotted path to a value or to ``{op: value}``. A path whose first part
is an envelope field (kind, schema, created_at, actor, refs, origin, blobs, body) is
taken from the record root; any other path is inside ``body``. Operators: ``=`` ``!=``
``<`` ``<=`` ``>`` ``>=`` ``in`` (value is a list) and ``contains`` (the field is a list
holding the value). ``order`` is a path, ``-path`` for descending.

Paths are validated against an identifier pattern and then written into the SQL as
literals, so SQLite can use expression indexes such as ``(kind, $.body.name)``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from kta.core.errors import ConfigError

ENVELOPE = {"kind", "schema", "created_at", "actor", "refs", "origin", "blobs", "body"}
# Envelope paths that have their own column.
COLUMNS = {
    "kind": "kind",
    "schema": "schema",
    "created_at": "created_at",
    "origin": "origin",
    "actor.type": "actor_type",
    "actor.name": "actor_name",
    "actor.task": "task",
}
OPS = {"=": "=", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_LIMIT = 100_000


def column(path: str) -> str:
    """SQL expression for a dotted path."""
    parts = path.split(".")
    if not path or not all(_SEGMENT.match(p) for p in parts):
        raise ConfigError(f"invalid field path {path!r}")
    if path in COLUMNS:
        return COLUMNS[path]
    json_path = "$." + (path if parts[0] in ENVELOPE else f"body.{path}")
    return f"json_extract(json, '{json_path}')"


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    raise ConfigError(f"query values must be str, int, float, bool or None; got {value!r}")


def condition(path: str, spec: Any) -> tuple[str, list[Any]]:
    col = column(path)
    if not isinstance(spec, Mapping):
        if spec is None:
            return f"{col} IS NULL", []
        return f"{col} = ?", [_scalar(spec)]
    if len(spec) != 1:  # pyright: ignore[reportUnknownArgumentType]
        raise ConfigError(f"{path}: use one operator per field, got {spec!r}")
    op, value = next(iter(spec.items()))  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if op in OPS:
        return f"{col} {OPS[op]} ?", [_scalar(value)]
    if op == "in":
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{path}: 'in' needs a non-empty list")
        items = [_scalar(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
        return f"{col} IN ({', '.join('?' * len(items))})", items
    if op == "contains":
        json_path = col.removeprefix("json_extract(json, ").removesuffix(")")
        if json_path == col:
            raise ConfigError(f"{path}: 'contains' needs a JSON list field")
        return (
            f"EXISTS (SELECT 1 FROM json_each(records.json, {json_path}) WHERE value = ?)",
            [_scalar(value)],
        )
    raise ConfigError(f"{path}: unknown operator {op!r}")


def build(
    kind: str, where: Mapping[str, Any] | None, order: str | None, limit: int
) -> tuple[str, list[Any]]:
    if not 0 < limit <= MAX_LIMIT:
        raise ConfigError(f"limit must be in 1..{MAX_LIMIT}")
    clauses = ["kind = ?"]
    params: list[Any] = [kind]
    for path, spec in (where or {}).items():
        sql, values = condition(path, spec)
        clauses.append(sql)
        params += values
    order = order or "created_at"
    desc = order.startswith("-")
    direction = "DESC" if desc else "ASC"
    sql = (
        f"SELECT json FROM records WHERE {' AND '.join(clauses)} "
        f"ORDER BY {column(order.lstrip('-'))} {direction}, rowid {direction} LIMIT ?"
    )
    return sql, [*params, limit]
