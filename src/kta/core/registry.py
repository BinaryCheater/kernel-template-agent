"""Registration and selection of interface implementations (spec/conventions.md §3.2).

The registry is an object carried by the Context rather than a module-level table, so
tests can build their own and no process-wide mutable state exists. Each top-level
package under ``src/kta/`` may define ``register_implementations(registry)``;
``build_registry`` calls those hooks and nothing else, so only code in ``src/kta/``
reaches a production registry.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kta.core.errors import ConfigError, PermissionDenied

if TYPE_CHECKING:
    from kta.core.config import Config
    from kta.core.context import Context

Factory = Callable[["Context"], Any]

# Environments in which a fixture implementation must never be selected.
REAL_ONLY_ENVS = frozenset({"dev", "main"})


@dataclass(frozen=True)
class Implementation:
    interface: str
    name: str
    factory: Factory
    fixture: bool
    module: str


def _is_src_module(module: str) -> bool:
    return module == "kta" or module.startswith("kta.")


class Registry:
    def __init__(self) -> None:
        self._impls: dict[str, dict[str, Implementation]] = {}

    def register(self, interface: str, name: str, factory: Factory, fixture: bool = False) -> None:
        module = str(getattr(factory, "__module__", ""))
        if _is_src_module(module) and fixture:
            raise ConfigError(f"{interface}/{name}: implementations in src/kta are never fixtures")
        if not _is_src_module(module) and not fixture:
            raise ConfigError(
                f"{interface}/{name}: implementations outside src/kta ({module}) "
                "must be registered with fixture=True"
            )
        by_name = self._impls.setdefault(interface, {})
        if name in by_name:
            raise ConfigError(f"{interface}/{name} registered twice")
        by_name[name] = Implementation(interface, name, factory, fixture, module)

    def interfaces(self) -> list[str]:
        return sorted(self._impls)

    def implementations(self, interface: str) -> list[Implementation]:
        return [self._impls[interface][n] for n in sorted(self._impls.get(interface, {}))]

    def select(self, interface: str, config: Config) -> Implementation:
        """Pick the implementation named in [impl]; without a name, the only real one."""
        candidates = self._impls.get(interface, {})
        chosen = config.impl(interface)
        if chosen is not None:
            if chosen not in candidates:
                raise ConfigError(f"[impl] {interface} = {chosen!r} is not registered")
            return candidates[chosen]
        real = [impl for impl in candidates.values() if not impl.fixture]
        if len(real) == 1:
            return real[0]
        raise ConfigError(
            f"interface {interface!r} needs an [impl] entry "
            f"({len(real)} real implementations registered)"
        )

    def resolve(self, interface: str, ctx: Context) -> Any:
        impl = self.select(interface, ctx.config)
        if impl.fixture and ctx.env in REAL_ONLY_ENVS:
            raise PermissionDenied(
                f"fixture implementation {interface}/{impl.name} selected in env {ctx.env}"
            )
        return impl.factory(ctx)


def resolve(interface: str, ctx: Context) -> Any:
    return ctx.registry.resolve(interface, ctx)


def build_registry() -> Registry:
    """Collect implementations from every top-level package under src/kta."""
    import kta

    registry = Registry()
    for info in sorted(pkgutil.iter_modules(kta.__path__), key=lambda i: i.name):
        if not info.ispkg:
            continue
        package = importlib.import_module(f"kta.{info.name}")
        hook = getattr(package, "register_implementations", None)
        if callable(hook):
            hook(registry)
    return registry
