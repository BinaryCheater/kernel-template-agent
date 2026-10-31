"""The controls registry: definitions, value validation and the typed snapshot (WP21, D480).

``controls.yaml`` defines every knob of docs/design/12-agent-controls.md (value type, the
failure it prevents, where it lives, status, the X experiment that sets it);
``experiments.yaml`` defines X1–X9 (02 §11). Both are read-only package data. The values
in force are a ``Controls`` snapshot: the newest ControlsVersion record
(``kta.store.controls``), or the definitions' initial values (``baseline()``) where no store
is at hand. Code reads knobs only through the typed accessors below; adding a knob is one
row in ``controls.yaml`` plus one accessor here (a test checks they match).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any, Literal, cast

import yaml

from kta.core.errors import ConfigError, PermissionDenied
from kta.core.knobs import Knobs

KnobType = Literal["int", "float", "bool", "str", "enum", "per_task_type", "mapping", "list_str"]
KNOB_TYPES = ("int", "float", "bool", "str", "enum", "per_task_type", "mapping", "list_str")
FIELD_TYPES = ("int", "float", "bool", "str", "list_str")
WHERE = ("profile", "orch", "protocol", "ui", "gate", "knowledge")
STATUSES = ("spec", "default", "set")
EXPERIMENT_STATUSES = ("defined", "approved", "running", "done")
SET_FROM = ("", "S1/S2 logs")  # 02 §11 step 5: revised from logs, no dedicated experiment


@dataclass(frozen=True)
class Knob:
    id: str
    row: str
    section: str
    controls: str
    type: KnobType
    value: Any
    prevents: str
    where: tuple[str, ...]
    status: str
    source: str
    set_by: tuple[str, ...]
    editable: bool
    accessor: str
    consumer: str
    unit: str = ""
    choices: tuple[str, ...] = ()
    fields: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())
    min: float | None = None
    max: float | None = None
    sum_max: float | None = None
    set_run: str | None = None
    set_from: str = ""

    @property
    def set_by_cell(self) -> str:
        """The `set_by` cell of 12: the experiments, then "S1/S2 logs" when revised from logs."""
        parts = [*self.set_by, *([self.set_from] if self.set_from else [])]
        return ", ".join(parts) or "—"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            k: getattr(self, k) for k in self.__dataclass_fields__ if k not in ("fields",)
        }
        out["fields"] = dict(self.fields)
        out["set_by_cell"] = self.set_by_cell
        for k in ("where", "set_by", "choices"):
            out[k] = list(out[k])
        return out


@dataclass(frozen=True)
class Experiment:
    id: str
    question: str
    design: str
    arms: tuple[str, ...]
    metrics: tuple[str, ...]
    sets: tuple[str, ...]
    profile_fields: tuple[str, ...]
    sets_elsewhere: tuple[str, ...]  # values outside the registry (e.g. HwModel data)
    depends_on: tuple[str, ...]
    step: int  # 02 §11 order, 1 (no agent, GPU minutes) .. 6 (claims)
    method: str  # run | logs (revised from S1/S2 logs, no dedicated run)
    status: str

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            v: Any = getattr(self, name)
            out[name] = list(cast(tuple[str, ...], v)) if isinstance(v, tuple) else v
        return out


@dataclass(frozen=True)
class Definitions:
    knobs: tuple[Knob, ...]
    experiments: tuple[Experiment, ...]
    task_types: tuple[str, ...]
    hash: str  # sha256 of both files' parsed content: a version records which definitions it used

    def knob(self, knob_id: str) -> Knob:
        for k in self.knobs:
            if k.id == knob_id:
                return k
        raise ConfigError(f"no control {knob_id!r} in the registry")

    def ids(self) -> list[str]:
        return [k.id for k in self.knobs]


def _load(name: str) -> dict[str, Any]:
    data: Any = yaml.safe_load(resources.files("kta.core").joinpath(name).read_text("utf-8"))
    if not isinstance(data, dict) or cast(dict[str, Any], data).get("format") != 1:
        raise ConfigError(f"{name} must be a mapping with format: 1")
    return cast(dict[str, Any], data)


def _tuple(raw: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in cast(list[Any], raw or []))


def _knob(raw: dict[str, Any]) -> Knob:
    if raw.get("type") not in KNOB_TYPES:
        raise ConfigError(f"control {raw.get('id')}: type {raw.get('type')!r} unknown")
    k = Knob(
        id=str(raw["id"]),
        row=str(raw["row"]),
        section=str(raw["section"]),
        controls=str(raw["controls"]),
        type=cast(KnobType, raw["type"]),
        value=raw["value"],
        prevents=str(raw["prevents"]),
        where=_tuple(raw["where"]),
        status=str(raw["status"]),
        source=str(raw.get("source", "")),
        set_by=_tuple(raw["set_by"]),
        editable=bool(raw["editable"]),
        accessor=str(raw["accessor"]),
        consumer=str(raw.get("consumer", "")),
        unit=str(raw.get("unit", "")),
        choices=_tuple(raw.get("choices")),
        fields={str(a): str(b) for a, b in cast(dict[Any, Any], raw.get("fields") or {}).items()},
        min=None if raw.get("min") is None else float(raw["min"]),
        max=None if raw.get("max") is None else float(raw["max"]),
        sum_max=None if raw.get("sum_max") is None else float(raw["sum_max"]),
        set_run=None if raw.get("set_run") is None else str(raw["set_run"]),
        set_from=str(raw.get("set_from", "")),
    )
    bad = [w for w in k.where if w not in WHERE]
    if bad or k.status not in STATUSES:
        raise ConfigError(f"control {k.id}: where {bad} or status {k.status!r} unknown")
    if k.type == "mapping" and any(t not in FIELD_TYPES for t in k.fields.values()):
        raise ConfigError(f"control {k.id}: field types must be among {FIELD_TYPES}")
    if k.set_from not in SET_FROM:
        raise ConfigError(f"control {k.id}: set_from must be one of {SET_FROM}")
    if (k.status == "set") != (k.set_run is not None):
        raise ConfigError(f"control {k.id}: status set needs set_run, and only then")
    return k


@cache
def definitions() -> Definitions:
    """The registry definitions, validated: unique ids and accessors, initial values valid,
    ``set_by`` and the experiments' ``sets`` agreeing."""
    knobs_raw, exp_raw = _load("controls.yaml"), _load("experiments.yaml")
    task_types = _tuple(knobs_raw["task_types"])
    knobs = tuple(_knob(cast(dict[str, Any], r)) for r in knobs_raw["knobs"])
    exps = tuple(
        Experiment(
            id=str(r["id"]),
            question=str(r["question"]),
            design=str(r["design"]),
            arms=_tuple(r["arms"]),
            metrics=_tuple(r["metrics"]),
            sets=_tuple(r["sets"]),
            profile_fields=_tuple(r.get("profile_fields")),
            sets_elsewhere=_tuple(r.get("sets_elsewhere")),
            depends_on=_tuple(r.get("depends_on")),
            step=int(r["step"]),
            method=str(r["method"]),
            status=str(r["status"]),
        )
        for r in cast(list[dict[str, Any]], exp_raw["experiments"])
    )
    canonical = json.dumps([knobs_raw, exp_raw], sort_keys=True, separators=(",", ":"))
    defs = Definitions(knobs, exps, task_types, hashlib.sha256(canonical.encode()).hexdigest())
    _validate(defs)
    return defs


def _validate(defs: Definitions) -> None:
    ids = defs.ids()
    for what, values in (("id", ids), ("accessor", [k.accessor for k in defs.knobs])):
        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            raise ConfigError(f"controls.yaml: duplicate {what} {dupes}")
    exp_ids = {e.id for e in defs.experiments}
    for k in defs.knobs:
        coerce(k, k.value, defs.task_types)
        unknown = [x for x in k.set_by if x not in exp_ids]
        if unknown:
            raise ConfigError(f"control {k.id}: set_by names undefined experiments {unknown}")
    for e in defs.experiments:
        if e.method not in ("run", "logs") or not 1 <= e.step <= 6:
            raise ConfigError(f"experiment {e.id}: method run|logs and step 1..6")
        if e.status not in EXPERIMENT_STATUSES:
            raise ConfigError(f"experiment {e.id}: status {e.status!r} unknown")
        for knob_id in e.sets:
            if e.id not in defs.knob(knob_id).set_by:
                raise ConfigError(f"experiment {e.id} sets {knob_id}, whose set_by omits it")
    for k in defs.knobs:
        for x in k.set_by:
            if k.id not in next(e for e in defs.experiments if e.id == x).sets:
                raise ConfigError(f"control {k.id}: set_by {x}, but {x} does not list it")


# ---- values -------------------------------------------------------------------------------


def _number(knob: Knob, name: str, value: Any, kind: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{knob.id}{name}: expected a number, got {value!r}")
    if kind == "int":
        if float(value) != int(value):
            raise ConfigError(f"{knob.id}{name}: expected an integer, got {value!r}")
        out: int | float = int(value)
    else:
        out = float(value)
    if knob.min is not None and out < knob.min:
        raise ConfigError(f"{knob.id}{name}: {out} is below the minimum {knob.min:g}")
    if knob.max is not None and out > knob.max:
        raise ConfigError(f"{knob.id}{name}: {out} is above the maximum {knob.max:g}")
    return out


def _scalar(knob: Knob, name: str, value: Any, kind: str) -> Any:
    if kind in ("int", "float"):
        return _number(knob, name, value, kind)
    if kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{knob.id}{name}: expected true or false, got {value!r}")
        return value
    if kind == "str":
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{knob.id}{name}: expected a non-empty string")
        return value.strip()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in cast(list[Any], value)):
        raise ConfigError(f"{knob.id}{name}: expected a list of strings")
    return list(cast(list[str], value))


def _table(knob: Knob, value: Any, fields: Mapping[str, str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{knob.id}: expected a mapping with keys {sorted(fields)}")
    given = cast(Mapping[str, Any], value)
    if set(given) != set(fields):
        raise ConfigError(f"{knob.id}: keys must be {sorted(fields)}, got {sorted(given)}")
    out = {f: _scalar(knob, f".{f}", given[f], kind) for f, kind in fields.items()}
    numeric = [v for f, v in out.items() if fields[f] in ("int", "float")]
    if knob.sum_max is not None and sum(numeric) > knob.sum_max + 1e-9:
        raise ConfigError(f"{knob.id}: values sum to {sum(numeric):g} > {knob.sum_max:g}")
    return out


def coerce(knob: Knob, value: Any, task_types: tuple[str, ...]) -> Any:
    """``value`` validated against the knob's type and bounds, in canonical form."""
    if knob.type == "enum":
        if value not in knob.choices:
            raise ConfigError(f"{knob.id}: {value!r} is not one of {list(knob.choices)}")
        return str(value)
    if knob.type == "per_task_type":
        return _table(knob, value, dict.fromkeys(task_types, "float"))
    if knob.type == "mapping":
        return _table(knob, value, knob.fields)
    return _scalar(knob, "", value, knob.type)


# ---- the snapshot ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    knob: str
    old: Any
    new: Any

    def to_dict(self) -> dict[str, Any]:
        return {"knob": self.knob, "old": self.old, "new": self.new}


@dataclass(frozen=True)
class Controls(Knobs):
    """The knob values in force. ``version_id`` is the ControlsVersion record ID, or None for
    values not (yet) stored: the baseline, or a proposal before it is written."""

    values: Mapping[str, Any]
    version: int = 0
    version_id: str | None = None
    definitions_hash: str = ""

    def value(self, knob_id: str) -> Any:
        if knob_id not in self.values:
            raise ConfigError(f"control {knob_id!r} has no value in version {self.version}")
        return json.loads(json.dumps(self.values[knob_id]))  # a copy: snapshots stay immutable

    def with_changes(self, changes: Mapping[str, Any], allow_locked: bool = False) -> Controls:
        """A proposal: these values changed, validated; ``version_id`` None until stored."""
        defs = definitions()
        values = dict(self.values)
        for knob_id, raw in changes.items():
            knob = defs.knob(knob_id)
            if not knob.editable and not allow_locked:
                raise PermissionDenied(f"{knob_id} is an invariant enforced in code (not editable)")
            values[knob_id] = coerce(knob, raw, defs.task_types)
        return Controls(values, self.version + 1, None, defs.hash)

    def diff(self, other: Controls) -> list[Change]:
        """Changes from ``self`` to ``other``, in registry order (knobs only one side has too)."""
        keys = [*definitions().ids(), *sorted(set(self.values) | set(other.values))]
        seen: set[str] = set()
        out: list[Change] = []
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            old, new = self.values.get(k), other.values.get(k)
            if old != new:
                out.append(Change(k, old, new))
        return out


def baseline() -> Controls:
    """The definitions' initial values (what version 1 stores)."""
    defs = definitions()
    values = {k.id: coerce(k, k.value, defs.task_types) for k in defs.knobs}
    return Controls(values, 0, None, defs.hash)


def upgrade(stored: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Stored values carried onto the current definitions: (values, added, dropped).

    A knob the stored version lacks takes its initial value; a value the definitions no
    longer have is dropped; a stored value that no longer validates is an error (a
    definition change must not silently reinterpret a value the user set).
    """
    defs = definitions()
    values: dict[str, Any] = {}
    added: list[str] = []
    for k in defs.knobs:
        if k.id in stored:
            values[k.id] = coerce(k, stored[k.id], defs.task_types)
        else:
            values[k.id] = coerce(k, k.value, defs.task_types)
            added.append(k.id)
    dropped = sorted(set(stored) - set(values))
    return values, added, dropped
