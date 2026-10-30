"""Loading the state-machine definition (machines.yaml, decision D96)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, cast

import yaml

from kta.core.errors import ConfigError

ANY_STATE = "*"


@dataclass(frozen=True)
class Edge:
    src: str
    to: str
    kind: str
    rule: int | None
    creates: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class Machine:
    id: str
    states: dict[str, str]  # state id -> state kind (step | wait | terminal)
    initial: str
    edges: tuple[Edge, ...]
    limits: dict[str, Any]

    def edge(self, src: str, to: str, kind: str, rule: int | None = None) -> Edge:
        """The listed edge for a transition, or ConfigError: step never invents an edge."""
        for e in self.edges:
            if e.to == to and e.kind == kind and e.rule == rule and e.src in (src, ANY_STATE):
                return e
        raise ConfigError(f"{self.id}: no transition {src} -> {to} ({kind}, rule {rule})")

    def terminal(self, state: str) -> bool:
        return self.states.get(state) == "terminal"


def _edge(raw: dict[str, Any]) -> Edge:
    rule = raw.get("rule")
    return Edge(
        src=str(raw["from"]),
        to=str(raw["to"]),
        kind=str(raw["kind"]),
        rule=int(rule) if rule is not None else None,
        creates=tuple(str(c) for c in raw.get("creates", [])),
        label=str(raw.get("label", "")),
    )


def _machine(raw: dict[str, Any]) -> Machine:
    states = {str(s["id"]): str(s["kind"]) for s in raw["states"]}
    initial = [str(s["id"]) for s in raw["states"] if s.get("initial")]
    if len(initial) != 1:
        raise ConfigError(f"machine {raw['id']} needs exactly one initial state")
    edges = tuple(_edge(e) for e in raw["transitions"])
    for e in edges:
        for state in (e.src, e.to):
            if state != ANY_STATE and state not in states:
                raise ConfigError(f"machine {raw['id']}: unknown state {state}")
    return Machine(str(raw["id"]), states, initial[0], edges, dict(raw.get("limits", {})))


def definition() -> dict[str, Any]:
    text = resources.files("kta.orchestrator").joinpath("machines.yaml").read_text("utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError("machines.yaml must be a mapping")
    return cast(dict[str, Any], data)


def machine(machine_id: str) -> Machine:
    for raw in definition()["machines"]:
        if raw["id"] == machine_id:
            return _machine(raw)
    raise ConfigError(f"machines.yaml has no machine {machine_id!r}")
