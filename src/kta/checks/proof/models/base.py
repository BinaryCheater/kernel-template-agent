"""What a checker model receives and the helpers models share."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kta.core.schemas.knowledge import Obligation
from kta.core.schemas.proof import ObligationModel

from .. import interp
from ..encode import Unsupported, domain_values
from ..space import Space
from ..types import Draft


@dataclass(frozen=True)
class ModelInput:
    space: Space
    binding: ObligationModel
    obligation: Obligation
    timeout_ms: int

    def param(self, name: str, default: Any = None) -> Any:
        return self.binding.params.get(name, default)

    def var(self, role: str) -> str:
        """The Template variable (or p.* name) that plays ``role`` in this model."""
        name = self.binding.params.get(role)
        if not isinstance(name, str):
            raise Unsupported(f"model {self.binding.model}: parameter {role!r} not bound")
        return name

    def menu(self, name: str) -> list[Any]:
        """Values of a design variable: the fixed value, the outline's menu or the domain."""
        sp = self.space
        if sp.is_fixed(name):
            return [sp.restrict[name]]
        if name in sp.menus:
            return list(sp.menus[name])
        override = (self.binding.params.get("menu") or {}).get(name)
        if override is not None:
            return list(override)
        spec = next((v for v in sp.template.variables if v.name == name), None)
        if spec is None:
            raise Unsupported(f"model {self.binding.model}: {name} is not a design variable")
        vals = domain_values(spec)
        if vals is None:
            raise Unsupported(f"{name}: domain too wide to enumerate; give a menu")
        return vals

    def upto(self, name: str, bound: int) -> list[int]:
        """Values 1..bound that the variable's domain allows (bounded enumeration)."""
        sp = self.space
        if sp.is_fixed(name):
            return [int(sp.restrict[name])]
        if name in sp.menus:
            # a chain's menu: its smallest values, whatever their size
            return sorted(int(v) for v in sp.menus[name] if int(v) >= 1)[:bound]
        spec = next((v for v in sp.template.variables if v.name == name), None)
        if spec is None:
            raise Unsupported(f"model {self.binding.model}: {name} is not a design variable")
        vals = domain_values(spec)
        if vals is not None:
            return [int(v) for v in vals if 1 <= int(v) <= bound]
        d = spec.domain
        assert not isinstance(d, list)
        return [
            v for v in range(max(d.lo, 1), min(d.hi, bound) + 1) if (v - d.lo) % max(d.step, 1) == 0
        ]

    def pre_holds(self, point: dict[str, Any]) -> bool:
        """All preconditions at ``point`` (interpreter); Unsupported when undecided."""
        for pre in self.space.template.preconditions:
            got = interp.holds(pre.expr, point)
            if got is None:
                raise Unsupported(f"precondition {pre.id} not decided by {sorted(point)}")
            if not got:
                return False
        return True

    def draft(self, outcome: str, domain: str, groups: tuple[str, ...]) -> Draft:
        return Draft("K2", self.obligation.name, outcome, domain, groups=groups)


ModelFn = Callable[[ModelInput], Draft]
