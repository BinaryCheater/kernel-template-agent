"""The GB10 target plugin (NVIDIA DGX Spark, sm_121)."""

from __future__ import annotations

from kta.core.context import Clock
from kta.core.errors import NotSupported
from kta.core.ids import record_id
from kta.core.jobs import JobRunner
from kta.core.schemas.hardware import HwFact, HwFeature, Target
from kta.store.base import Store
from kta.targets.base import ProbeReport
from kta.targets.gb10.probe import Gb10Probe, load_features
from kta.targets.gb10.units import UNITS

NAME = "gb10"
SM = "sm_121"
# The arch-specific variant enables sm_120a/121a-only instructions (e.g. kind::f8f6f4 MMA).
ARCH_FLAG = "-arch=sm_121a"
COMPILE_FLAGS: dict[str, list[str]] = {
    "cuda": [ARCH_FLAG, "-O3", "-lineinfo", "-std=c++17"],
    "tilelang": [
        ARCH_FLAG,
        "-O3",
        "-lineinfo",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-DENABLE_BF16",
    ],
}


class Gb10Target:
    name = NAME

    def __init__(self, store: Store, clock: Clock) -> None:
        self.store = store
        self.clock = clock
        self.last_probe: ProbeReport | None = None

    def probe(self, ex: JobRunner) -> Target:
        report = Gb10Probe(self.store, ex, self.clock, NAME, SM, ARCH_FLAG).run()
        self.last_probe = report
        return Target.model_validate(self.store.get(report.target_id).body)

    def units(self) -> list[str]:
        return list(UNITS)

    def latest_target_id(self) -> str:
        found = self.store.query("Target", {"name": NAME}, order="-created_at", limit=1)
        if not found:
            raise NotSupported(f"no Target record for {NAME}; run `kta target probe` first")
        return record_id(found[0])

    def limits(self) -> dict[str, int | float]:
        """hw.* values from the latest probe's HwFacts (documented, then measured ones)."""
        facts = self.store.query(
            "HwFact", {"refs.target": self.latest_target_id()}, order="created_at", limit=1000
        )
        out: dict[str, int | float] = {}
        # Measured facts are written after documented ones and so take precedence.
        for rec in sorted(facts, key=lambda r: r.body["method"] == "measured"):
            fact = HwFact.model_validate(rec.body)
            if isinstance(fact.value, float):
                out[fact.quantity] = int(fact.value) if fact.value.is_integer() else fact.value
        return out

    def features(self) -> list[HwFeature]:
        """The feature list with support from the latest probe (unknown when never probed)."""
        probed: dict[str, HwFeature] = {}
        found = self.store.query("Target", {"name": NAME}, order="-created_at", limit=1)
        if found:
            target_id = record_id(found[0])
            for rec in self.store.query("HwFeature", {"refs.target": target_id}, limit=1000):
                feature = HwFeature.model_validate(rec.body)
                probed[feature.name] = feature
        return [
            probed.get(s.name)
            or HwFeature(
                name=s.name,
                unit=s.unit,
                instructions=s.instructions,
                description=s.description,
                support={NAME: "unknown"},
            )
            for s in load_features()
        ]

    def compile_flags(self, backend: str) -> list[str]:
        if backend not in COMPILE_FLAGS:
            raise NotSupported(f"backend {backend!r} on {NAME}")
        return list(COMPILE_FLAGS[backend])


__all__ = ["ARCH_FLAG", "NAME", "SM", "Gb10Target"]
