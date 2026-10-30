"""Measurement protocols: one frozen Protocol record per version (spec/modules/measure-check.md).

``measure-v1`` carries the defaults of the specification. Its noise floor is the
provisional 1.1% median capture A/B deviation of earlier GB10 runs (WP13 finding 6)
until the WP3 noise probe measures one; the probe's result is frozen as ``measure-v2``
(decision D226). A version is never changed in place: different values need a new
version, and Measurements name the Protocol record they were taken under.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from kta.core.context import Clock
from kta.core.errors import ConfigError
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import Protocol
from kta.store.base import Store

ACTOR = Actor(type="tool", name="kta measure protocol")
NAME = "measure"


@dataclass(frozen=True)
class MeasureParams:
    timing: str = "cuda_events_per_iteration"
    l2_flush_factor: float = 2.0  # flush buffer = factor x L2 bytes, written before each iteration
    min_input_sets: int = 4
    nan_fill: bool = True
    warmup_min_iters: int = 10
    warmup_min_ms: float = 50.0
    min_iters: int = 100
    min_ms: float = 200.0
    max_iters: int = 100_000  # safety cap per phase for very short kernels
    statistic: str = "median"
    ci_level: float = 0.95
    bootstrap_resamples: int = 1000
    noise_floor_rel: float = 0.011
    significance_factor: float = 3.0  # significant: CIs disjoint and |diff| > factor x floor
    memory: str = "cuMemAlloc"
    clock_lock: bool = False  # locking needs root on the worker; throttling is recorded instead
    exclusive: bool = True  # runs in the gpu-exclusive pool under the GPU lease

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasureParams:
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown measurement protocol parameters: {sorted(unknown)}")
        return cls(**data)


V1 = MeasureParams()
V1_DESCRIPTION = (
    "Default measurement protocol of spec/modules/measure-check.md: CUDA events per "
    "iteration; 2x L2 flush before each iteration; at least 4 rotating input sets; outputs "
    "NaN-filled before each launch; warm-up max(10 iterations, 50 ms); measured "
    "max(100 iterations, 200 ms); median with bootstrap 95% CI (1000 resamples); "
    "significance: CIs disjoint and difference > 3x noise floor. Noise floor 1.1% is "
    "provisional (WP13 finding 6) until the WP3 noise probe."
)


def find_protocol(store: Store, name: str, version: int) -> str | None:
    found = store.query("Protocol", {"name": name, "version": version}, order="created_at")
    return record_id(found[0]) if found else None


def ensure_protocol(
    store: Store,
    clock: Clock,
    version: int,
    params: MeasureParams,
    description: str,
    refs: dict[str, str | list[str]] | None = None,
) -> str:
    """The Protocol record ``measure-v<version>``; an existing one must hold the same params."""
    existing = find_protocol(store, NAME, version)
    if existing is not None:
        if store.get(existing).body["params"] != params.to_dict():
            raise ConfigError(f"{NAME}-v{version} exists with other parameters; use a new version")
        return existing
    body = Protocol(
        name=NAME,
        version=version,
        type="measurement",
        description=description,
        params=params.to_dict(),
    )
    return store.put(Record.new("Protocol", body, ACTOR, clock.now(), refs=refs or {}))


def ensure_v1(store: Store, clock: Clock) -> str:
    return ensure_protocol(store, clock, 1, V1, V1_DESCRIPTION)


def load_params(store: Store, protocol_id: str) -> MeasureParams:
    rec = store.get(protocol_id)
    if rec.kind != "Protocol" or rec.body.get("type") != "measurement":
        raise ConfigError(f"{protocol_id[:12]} is not a measurement Protocol record")
    return MeasureParams.from_dict(dict(rec.body["params"]))


def revise(base: MeasureParams, noise_floor_rel: float) -> MeasureParams:
    """Parameters of the next version from a measured noise floor (decision D226)."""
    return replace(base, noise_floor_rel=noise_floor_rel)
