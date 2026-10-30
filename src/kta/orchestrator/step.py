"""``step(loop, ctx) -> Transition``: the pure transition function of every loop machine."""

from __future__ import annotations

from kta.core.context import Context
from kta.core.errors import ConfigError
from kta.orchestrator import expand_loop, solve_loop
from kta.orchestrator.transition import Loop, Transition

STEPS = {"solve": solve_loop.step, "expand": expand_loop.step}


def step(loop: Loop, ctx: Context) -> Transition:
    fn = STEPS.get(loop.kind)
    if fn is None:
        raise ConfigError(f"no step function for loop kind {loop.kind!r}")
    return fn(loop, ctx)
