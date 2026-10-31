"""The symbolic space of one Template: X with its domains, the free p.* / hw.* / case.*
symbols, Γ, C, Pre and the outline's environment assumptions.

``restrict`` fixes design variables (a chain binding, a candidate); ``attrs`` fixes or
rebinds p.* / hw.* / case.* (composition binds p.* to expressions of family variables).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import z3

from kta.core.schemas.knowledge import Template, VarSpec
from kta.core.schemas.proof import Assumption

from .encode import (
    Sym,
    Translator,
    Unsupported,
    constant,
    declare,
    domain_values,
    enum_range,
    infer,
)
from .encode import free as free_sym


@dataclass
class Space:
    template: Template
    assumes: list[Assumption]
    restrict: Mapping[str, Any] = field(default_factory=lambda: dict[str, Any]())
    attrs: Mapping[str, Sym] = field(default_factory=lambda: dict[str, Sym]())
    # design variable -> the values a chain allows (a family variable's finite domain)
    menus: Mapping[str, list[Any]] = field(default_factory=lambda: dict[str, list[Any]]())
    extra: Any = None  # z3 constraint conjoined to the domain (a family's domains and glue)
    ns: str = ""  # prefix of the solver names of free p.* symbols (one namespace per chain step)
    env: dict[str, Sym] = field(init=False)
    domain: Any = field(init=False)  # z3 constraint: X domains and enum ranges
    free: dict[str, Sym] = field(init=False)  # symbols of p/hw/case left free

    def __post_init__(self) -> None:
        t = self.template
        srcs = [t.scope.predicate, *(c.expr for c in t.constraints)]
        srcs += [c.expr for c in t.preconditions]
        srcs += [a.expr for a in self.assumes if a.expr]
        kinds = infer(srcs)
        self.env, self.free = {}, {}
        parts: list[Any] = []
        for v in t.variables:
            lits = kinds.get(v.name, ("int", ()))[1]
            sym, dom = declare(v, lits)
            fixed = self.restrict.get(v.name)
            if isinstance(fixed, Sym):
                # aliased to a family variable: keep the step's own domain on that term
                self.env[v.name] = Sym(fixed.name, fixed.term, sym.kind, sym.enum)
                parts.append(z3.substitute(dom, (sym.term, fixed.term)))
            elif v.name in self.restrict:
                self.env[v.name] = constant(v.name, fixed, sym)
            else:
                self.env[v.name] = sym
                parts.append(dom)
            if v.name in self.menus:
                term = self.env[v.name].term
                parts.append(
                    z3.Or(*[term == constant(v.name, x, sym).term for x in self.menus[v.name]])
                )
        for name, (kind, lits) in kinds.items():
            if name in self.env:
                continue
            if name in self.attrs:
                self.env[name] = self.attrs[name]
                continue
            if "." not in name:
                raise Unsupported(f"undeclared name {name}")
            s = free_sym(name, kind, lits)
            if self.ns and name.startswith("p."):
                s = Sym(name, z3.Const(self.ns + name, s.term.sort()), s.kind, s.enum)
            self.env[name] = self.free[name] = s
            parts.append(enum_range(s))
        if self.extra is not None:
            parts.append(self.extra)
        self.domain = z3.And(*parts) if parts else z3.BoolVal(True)

    @property
    def tr(self) -> Translator:
        return Translator(self.env)

    def gamma(self) -> Any:
        return self.tr.bool(self.template.scope.predicate)

    def assumptions(self) -> Any:
        return z3.And(*[self.tr.bool(a.expr) for a in self.assumes if a.expr] or [z3.BoolVal(True)])

    def constraints(self, env: Mapping[str, Sym] | None = None) -> Any:
        tr = Translator(env or self.env)
        return z3.And(*[tr.bool(c.expr) for c in self.template.constraints] or [z3.BoolVal(True)])

    def design(self) -> list[VarSpec]:
        """Variables still free in this space (aliased ones included)."""
        return [v for v in self.template.variables if not self.is_fixed(v.name)]

    def is_fixed(self, name: str) -> bool:
        return name in self.restrict and not isinstance(self.restrict[name], Sym)

    def decls(self) -> dict[str, Any]:
        out = {v.name: self.env[v.name].term for v in self.design()}
        out.update({k: s.term for k, s in self.free.items()})
        out.update({k: s.term for k, s in self.attrs.items() if k not in out})
        return out

    def decode(self, model: Mapping[str, Any]) -> dict[str, Any]:
        point = {k: self.env[k].decode(v) for k, v in model.items() if k in self.env}
        point.update({k: v for k, v in model.items() if k not in self.env})
        point.update({k: v for k, v in self.restrict.items() if not isinstance(v, Sym)})
        return point

    def finite_points(self, limit: int) -> Iterator[dict[str, Any]] | None:
        """Every design point when X is finite and has at most ``limit`` points."""
        menus: list[tuple[str, list[Any]]] = []
        size = 1
        for v in self.design():
            vals = self.menus.get(v.name) or domain_values(v)
            if vals is None:
                return None
            menus.append((v.name, vals))
            size *= max(len(vals), 1)
            if size > limit:
                return None
        names = [n for n, _ in menus]
        return (
            dict(zip(names, combo, strict=True))
            for combo in itertools.product(*(vals for _, vals in menus))
        )

    def fixed_env(self, point: Mapping[str, Any]) -> dict[str, Sym]:
        env = dict(self.env)
        for k, v in point.items():
            env[k] = constant(k, v, self.env.get(k))
        return env

    def describe(self) -> str:
        parts = [f"p in Gamma ({self.template.scope.predicate})"]
        if any(a.expr for a in self.assumes):
            parts.append("assumes " + "; ".join(str(a.expr) for a in self.assumes if a.expr))
        dom = ", ".join(f"{v.name} in {self.menus.get(v.name) or _dom(v)}" for v in self.design())
        if dom:
            parts.append(dom)
        if self.restrict:
            parts.append(
                "bound "
                + ", ".join(
                    f"{k}={v.name if isinstance(v, Sym) else v}" for k, v in self.restrict.items()
                )
            )
        if self.attrs:
            parts.append("with chain bindings for " + ", ".join(sorted(self.attrs)))
        return "; ".join(parts)


def _dom(v: VarSpec) -> str:
    d = v.domain
    if isinstance(d, list):
        return "{" + ",".join(str(x) for x in d) + "}"
    return (
        f"{d.lo}..{d.hi}" + (f" step {d.step}" if d.step > 1 else "") + (" pow2" if d.pow2 else "")
    )
