"""The machine description (machine.yaml) and its parameters (gb10.json) as one object.

The engine reads everything about the resource network through a `Machine`: which resources
exist and at which scope, which operation class each opcode belongs to and which resources it
gates on and holds, and every numeric parameter. `Machine.with_overrides` makes a changed copy for
what-if runs; nothing is changed in place.
"""

from __future__ import annotations

import copy
import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kta.core.errors import ConfigError

HERE = Path(__file__).parent
DESCRIPTION = HERE / "machine.yaml"
PARAMETERS = HERE / "gb10.json"


class _Reads(dict):
    """A parameter dict that notes which keys were read: empty at first, each key is fetched once
    through `__missing__`, so later reads are plain dict lookups and the replay does not slow down.
    Iterating counts as reading everything ("*")."""

    def __init__(self, src: dict[str, Any], seen: set[str], prefix: str):
        super().__init__()
        self.src, self.seen, self.prefix = src, seen, prefix

    def __missing__(self, key: str) -> Any:
        value = self.src[key]
        self.seen.add("*" if key == "sim" else self.prefix + key)
        self[key] = value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.src:
            return self[key]
        self.seen.add(self.prefix + key)
        return default

    def __contains__(self, key: object) -> bool:
        if key not in self.src:  # an entry added later changes the answer
            self.seen.add(self.prefix + str(key))
        return key in self.src

    def __iter__(self):
        self.seen.add("*")
        return iter(self.src)

    def keys(self):
        self.seen.add("*")
        return self.src.keys()

    def items(self):
        self.seen.add("*")
        return self.src.items()

    def values(self):
        self.seen.add("*")
        return self.src.values()


@dataclass(frozen=True)
class OpClass:
    name: str
    handler: str
    gates: tuple[
        str, ...
    ] = ()  # resources that must be free; "either" = the first free of `either`
    either: tuple[str, ...] = ()
    hold: tuple[tuple[str, str], ...] = ()  # (resource or "either", parameter reference)
    latency: str | None = None
    rbar: str | None = None
    count: str | None = None
    beside: tuple[tuple[str, str], ...] = ()  # (resource, delay it suffers if held at this issue)


@dataclass(frozen=True)
class Machine:
    desc: dict[str, Any]
    raw: dict[str, Any]  # gb10.json as loaded (values with source and status)
    classes: tuple[tuple[dict[str, Any], OpClass], ...]
    default: OpClass
    overrides: tuple[tuple[str, Any], ...] = ()
    _cache: dict[tuple[str, str], OpClass] = field(default_factory=dict, compare=False)

    @functools.cached_property
    def C(self) -> dict[str, Any]:
        return {k: v["value"] for k, v in self.raw.items() if not k.startswith("_")}

    @functools.cached_property
    def P(self) -> dict[str, Any]:
        return self.raw["sim"]["value"]

    def tracked(self) -> Machine:
        """The same machine with `reads`: the parameters a replay on it reads (`sim.<key>` or
        `<key>`, as `with_overrides` names them). A replay is deterministic, so an override of a
        parameter it never read cannot change its result."""
        m = Machine(self.desc, self.raw, self.classes, self.default, self.overrides, self._cache)
        seen: set[str] = set()
        m.__dict__.update(P=_Reads(self.P, seen, "sim."), C=_Reads(self.C, seen, ""), reads=seen)
        return m

    def param(self, ref: str) -> Any:
        if ref.startswith("sim."):
            return self.P[ref[4:]]
        return self.C[ref]

    def status(self, ref: str) -> str:
        """measured / fitted / structural / assumed / withdrawn, or `override` for a what-if."""
        if any(k == ref for k, _ in self.overrides):
            return "override"
        if ref.startswith("sim."):
            return self.raw["sim"].get("value_status", {}).get(ref[4:], "unknown")
        return self.raw.get(ref, {}).get("status", "unknown")

    @property
    def smsp_per_sm(self) -> int:
        return int(self.param(self.desc["hierarchy"]["smsp"]["count"]))

    @property
    def sms(self) -> int:
        return int(self.param(self.desc["hierarchy"]["sm"]["total"]))

    @property
    def sms_per_tpc(self) -> int:
        return int(self.desc["sms_per_tpc"])

    def resources(self, scope: str | None = None) -> list[str]:
        return [
            n for n, r in self.desc["resources"].items() if scope is None or r["scope"] == scope
        ]

    def gating(self, scope: str) -> list[str]:
        return [
            n
            for n, r in self.desc["resources"].items()
            if r["scope"] == scope and r.get("gates_issue")
        ]

    def classify(self, base: str, op: str) -> OpClass:
        key = (base, op)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = self.default
        for rule, cls in self.classes:
            if "base" in rule and base not in rule["base"]:
                continue
            if "op_prefix" in rule and not op.startswith(rule["op_prefix"]):
                continue
            if "op_contains" in rule and rule["op_contains"] not in op:
                continue
            out = cls
            break
        self._cache[key] = out
        return out

    def with_overrides(self, changes: dict[str, Any]) -> Machine:
        """A copy with parameters replaced: keys `sim.<key>` or `<key>` (gb10.json); an unknown
        key is a ConfigError."""
        raw = copy.deepcopy(self.raw)
        for ref, value in changes.items():
            if ref.startswith("sim."):
                if ref[4:] not in raw["sim"]["value"]:
                    raise ConfigError(f"unknown machine parameter {ref}")
                raw["sim"]["value"][ref[4:]] = value
            else:
                if ref not in raw or ref.startswith("_"):
                    raise ConfigError(f"unknown machine parameter {ref}")
                raw[ref]["value"] = value
        return build(self.desc, raw, (*self.overrides, *changes.items()))


def _op_class(rule: dict[str, Any]) -> OpClass:
    gates: list[str] = []
    either: tuple[str, ...] = ()
    for g in rule.get("gates", []):
        if isinstance(g, dict):
            either = tuple(g["either"])
            gates.append("either")
        else:
            gates.append(g)
    return OpClass(
        name=rule["name"],
        handler=rule["handler"],
        gates=tuple(gates),
        either=either,
        hold=tuple(rule.get("hold", {}).items()),
        latency=rule.get("latency"),
        rbar=rule.get("rbar"),
        count=rule.get("count"),
        beside=tuple(rule.get("beside", {}).items()),
    )


def build(desc: dict[str, Any], raw: dict[str, Any], overrides=()) -> Machine:
    classes = tuple((rule, _op_class(rule)) for rule in desc["classes"])
    m = Machine(desc, raw, classes, _op_class(desc["default_class"]), tuple(overrides))
    _check(m)
    return m


def _check(m: Machine) -> None:
    """Every parameter the description names exists; every gate and hold names a resource."""
    known = set(m.desc["resources"]) | {"either"}
    for _, c in m.classes:
        for ref in [*(p for _, p in (*c.hold, *c.beside)), c.latency, c.rbar]:
            if ref and ref not in ("lsu", "tma_store_flush"):
                try:
                    m.param(ref)
                except KeyError as e:
                    raise ConfigError(f"class {c.name}: unknown parameter {ref}") from e
        for r in [*c.gates, *(r for r, _ in c.hold), *c.either]:
            if r not in known:
                raise ConfigError(f"class {c.name}: unknown resource {r}")


@functools.cache
def default_machine() -> Machine:
    desc = yaml.safe_load(DESCRIPTION.read_text())
    raw = json.loads(PARAMETERS.read_text())
    return build(desc, raw)
