"""The resolved hardware model: an HwModel (chip instance) on its HwArch (architecture
class), with every expression parsed and every rate evaluated (spec/modules/
formulation-solve-models.md "Hardware model", decisions D463, D465).

``MachineModel.build(hw, arch)`` is the one place where the declarative data becomes
numbers. L1 (``kta.models.l1``) prices effects against it; the L2 simulator (WP16) and
the projection compiler (WP17) read the same object.

Unmodelled means infinite capacity (D465): a resource without a peak costs no time and
an operation class without a reservation table reserves nothing, and both are reported
by L1 (never silently zero-cost).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.core.errors import NotSupported, SchemaError
from kta.core.expr import Expr, ExprError
from kta.core.schemas.effects import LEGACY_KIND_PREFIX
from kta.core.schemas.hardware import HwModel
from kta.core.schemas.hwmodel import (
    ROOTS_FEATURE,
    ROOTS_RESERVATION,
    ROOTS_SCOPE,
    EfficiencySpec,
    HwArch,
    InteractionSpec,
    TableSpec,
    WallSpec,
)

CHIP = "chip"


@dataclass(frozen=True)
class Resource:
    name: str
    kind: str
    scope: str
    amount: str
    rate: float | None  # chip-level amount per second at peak; None = unmodelled
    latency_s: float | None
    ncu: tuple[str, ...]
    status: str  # measured | documented | default | unmodelled
    source: str


@dataclass(frozen=True)
class Row:
    resource: str
    amount: Expr  # over tx.*, hw.*, tbl.*


@dataclass(frozen=True)
class OpClass:
    name: str
    rows: tuple[Row, ...]
    latency_s: float | None
    serial: bool
    legacy: bool = False  # the built-in unit:<u> class of upcast cost terms (D461)


@dataclass(frozen=True)
class Wall:
    spec: WallSpec
    limit: float
    reserved: float


@dataclass(frozen=True)
class MachineModel:
    target: str
    hw_version: int
    arch: str | None
    hw: Mapping[str, Any]  # the hw.* values
    resources: Mapping[str, Resource]
    op_classes: Mapping[str, OpClass]
    walls: tuple[Wall, ...]
    tables: Mapping[str, TableSpec]
    efficiency: Mapping[str, EfficiencySpec]
    interactions: tuple[InteractionSpec, ...]
    clock_hz: float | None
    scope_counts: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())

    # ---- construction --------------------------------------------------------------

    @staticmethod
    def build(hw: HwModel, arch: HwArch | None = None) -> MachineModel:
        """Resolve ``hw`` on ``arch``; without ``arch`` only measured peaks are resources."""
        if hw.arch is not None and arch is None:
            raise NotSupported(f"HwModel {hw.target}@{hw.version} needs its HwArch {hw.arch}")
        if arch is not None and hw.arch is not None and hw.arch != arch.ref:
            raise SchemaError(f"HwModel {hw.target}@{hw.version} is on {hw.arch}, not {arch.ref}")
        env: dict[str, Any] = dict(hw.limits)
        if arch is None:
            return _flat(hw, env)
        missing = [p for p in arch.params if p not in env]
        if missing:
            raise SchemaError(f"HwModel {hw.target}@{hw.version} lacks {arch.ref} params {missing}")
        scopes = {
            name: float(_num(src, env, f"scopes.{name}")) for name, src in arch.scopes.items()
        }
        clock = _opt(arch.composition.clock_hz, env, "composition.clock_hz")
        resources = {r.name: _resource(r, hw, env, scopes, clock) for r in arch.resources}
        ops = {o.name: _op_class(o, env, clock) for o in arch.operation_classes}
        walls = tuple(
            Wall(w, _num(w.limit, env, f"walls.{w.name}"), _opt(w.reserved, env, w.name) or 0.0)
            for w in arch.walls
        )
        tables = {t.name: t for t in arch.tables} | {t.name: t for t in hw.tables}
        eff = {e.resource: e for e in arch.efficiency} | {e.resource: e for e in hw.efficiency}
        inter = {i.name: i for i in arch.interactions} | {i.name: i for i in hw.interactions}
        return MachineModel(
            target=hw.target,
            hw_version=hw.version,
            arch=arch.ref,
            hw=env,
            resources=resources,
            op_classes=ops,
            walls=walls,
            tables=tables,
            efficiency=eff,
            interactions=tuple(inter.values()),
            clock_hz=clock,
            scope_counts=scopes,
        )

    # ---- lookups -------------------------------------------------------------------

    @property
    def version(self) -> str:
        """Identity of everything that changes a price."""
        text = json.dumps(
            {
                "hw": {k: self.hw[k] for k in sorted(self.hw)},
                "arch": self.arch,
                "rates": {r: v.rate for r, v in sorted(self.resources.items())},
                "ops": sorted(self.op_classes),
                "eff": {k: e.expr for k, e in sorted(self.efficiency.items())},
                "inter": [[i.name, i.gamma] for i in self.interactions],
            },
            sort_keys=True,
            default=str,
        )
        return f"{self.target}@{self.hw_version}:" + hashlib.sha256(text.encode()).hexdigest()[:10]

    def op_class(self, kind: str) -> OpClass | None:
        """The operation class of ``kind``; legacy ``unit:<u>`` maps to resource ``u``."""
        if kind in self.op_classes:
            return self.op_classes[kind]
        if kind.startswith(LEGACY_KIND_PREFIX):
            unit = kind[len(LEGACY_KIND_PREFIX) :]
            row = Row(unit, Expr.parse("1", set(), ROOTS_RESERVATION))
            return OpClass(kind, (row,), None, serial=False, legacy=True)
        return None

    def rate(self, resource: str) -> float | None:
        r = self.resources.get(resource)
        return r.rate if r is not None else None

    def time_s(self, resource: str, amount: float) -> float | None:
        """Time of ``amount`` on ``resource`` at peak; None when the resource is unmodelled."""
        rate = self.rate(resource)
        return None if rate is None else amount / rate

    def table(self, name: str, scopes: Mapping[str, Mapping[str, Any]]) -> float:
        spec = self.tables.get(name)
        if spec is None:
            raise NotSupported(f"{self.target}: no table term {name!r}")
        keys: list[float] = []
        for ref in spec.keys:
            root, _, attr = ref.partition(".")
            scope = scopes.get(root, {})
            if attr not in scope:
                raise NotSupported(f"table {name}: no value for key {ref}")
            keys.append(float(scope[attr]))
        if spec.lookup == "floor":
            rows = [r for r in spec.rows if r[0] <= keys[0]]
            if rows:
                return max(rows, key=lambda r: r[0])[-1]
        else:
            for r in spec.rows:
                if r[:-1] == keys:
                    return r[-1]
        raise NotSupported(f"table {name}: no row for keys {keys}")

    def evaluate(
        self, expr: Expr, tx: Mapping[str, Any] | None = None, f: Mapping[str, Any] | None = None
    ) -> float:
        """Evaluate an expression of the hardware model, resolving ``tbl.*`` first."""
        scopes: dict[str, Mapping[str, Any]] = {"tx": tx or {}, "f": f or {}, "hw": self.hw}
        tbl = {
            ref[4:]: self.table(ref[4:], scopes)
            for ref in sorted(expr.attr_refs)
            if ref.startswith("tbl.")
        }
        try:
            extra = {"tx": tx or {}, "f": f or {}, "tbl": tbl}
            return float(expr.eval({}, hw=self.hw, extra=extra))
        except ExprError as exc:
            raise SchemaError(f"{self.target}: {exc}") from exc

    def efficiency_of(self, resource: str, features: Mapping[str, Any]) -> float:
        spec = self.efficiency.get(resource)
        if spec is None:
            return 1.0
        expr = parse_hw(spec.expr, ROOTS_FEATURE, f"efficiency.{resource}")
        value = self.evaluate(expr, f=features)
        if not 0.0 < value <= 1.0:
            raise SchemaError(f"efficiency of {resource} is {value}, not in (0, 1]")
        return value


# ---- helpers ------------------------------------------------------------------------


def parse_hw(src: str, roots: tuple[str, ...], where: str) -> Expr:
    try:
        return Expr.parse(src, set(), roots)
    except ExprError as exc:
        raise SchemaError(f"{where}: {exc}") from exc


def _num(src: str, env: Mapping[str, Any], where: str) -> float:
    try:
        return float(parse_hw(src, ROOTS_SCOPE, where).eval({}, hw=env))
    except ExprError as exc:
        raise SchemaError(f"{where}: {exc}") from exc


def _opt(src: str | None, env: Mapping[str, Any], where: str) -> float | None:
    return None if src is None else _num(src, env, where)


def _resource(
    r: Any, hw: HwModel, env: Mapping[str, Any], scopes: Mapping[str, float], clock: float | None
) -> Resource:
    prov = hw.provenance.get(f"peaks.{r.name}")
    if r.name in hw.peaks:
        rate: float | None = float(hw.peaks[r.name])
        status = prov.status if prov else "measured"
        source = prov.source if prov else "HwModel.peaks"
    elif r.peak is not None:
        rate = _num(r.peak, env, f"resources.{r.name}.peak") * scopes[r.scope]
        status, source = "documented", r.source
    else:
        rate, status, source = None, "unmodelled", r.source
    if rate is not None and rate <= 0:
        raise SchemaError(f"resource {r.name}: peak {rate} is not positive")
    lat = _opt(r.latency_cycles, env, f"resources.{r.name}.latency_cycles")
    return Resource(
        name=r.name,
        kind=r.kind,
        scope=r.scope,
        amount=r.amount,
        rate=rate,
        latency_s=lat / float(clock) if lat is not None and clock else None,
        ncu=tuple(r.ncu),
        status=status,
        source=source,
    )


def _op_class(o: Any, env: Mapping[str, Any], clock: float | None) -> OpClass:
    rows = tuple(
        Row(row.resource, parse_hw(row.amount, ROOTS_RESERVATION, f"{o.name}.reservation"))
        for row in o.reservation
    )
    lat = _opt(o.latency_cycles, env, f"operation_classes.{o.name}.latency_cycles")
    return OpClass(o.name, rows, lat / clock if lat is not None and clock else None, o.serial)


def _flat(hw: HwModel, env: Mapping[str, Any]) -> MachineModel:
    """Schema-1 HwModel: each measured peak is a chip-scope throughput resource."""
    resources = {
        u: Resource(u, "throughput", CHIP, "work", float(p), None, (), "measured", "HwModel.peaks")
        for u, p in hw.peaks.items()
        if p > 0
    }
    return MachineModel(
        target=hw.target,
        hw_version=hw.version,
        arch=None,
        hw=env,
        resources=resources,
        op_classes={},
        walls=(),
        tables={t.name: t for t in hw.tables},
        efficiency={e.resource: e for e in hw.efficiency},
        interactions=tuple(hw.interactions),
        clock_hz=None,
        scope_counts={CHIP: 1.0},
    )
