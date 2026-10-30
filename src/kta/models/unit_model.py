"""Per-unit bottleneck cost model (docs/design/50 §7, paper §5.2).

    time_u = work_u / (peak_u * eff_u * occ)          occ = 1 before compilation
    total  = max_u time_u + serial_coef * serial + serial_const_us
             + sum over fitted pairs (u, v) of gamma_uv * min(time_u, time_v)

``Prediction.per_unit`` and ``Prediction.serial`` hold the times at peak (eff = 1,
occ = 1) in microseconds, so a fit can reuse predictions made by any model version
(decision D241). ``Prediction.total`` is this model's prediction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from kta.core.errors import NotSupported, SchemaError
from kta.core.schemas.evaluation import Measurement, Prediction
from kta.core.schemas.hardware import HwModel
from kta.models.bound import unit_time_us

if TYPE_CHECKING:
    from kta.models.fit import FitReport

# Units whose throughput does not depend on latency hiding by resident warps.
OCCUPANCY_EXEMPT = frozenset({"launch"})
# Occupancy at and above which latency is assumed hidden (decision D242).
DEFAULT_OCCUPANCY_KNEE = 0.25
OCCUPANCY_LIMITS = ("max_ctas_per_sm", "smem_per_sm", "regs_per_sm", "max_threads_per_sm")


class NoResidentCta(NotSupported):
    """The compile feedback leaves no CTA resident on an SM: the kernel cannot launch."""


@dataclass(frozen=True)
class Interaction:
    units: tuple[str, str]
    gamma: float
    basis: str  # why it was added: the significance test and its numbers


def _empty_info() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class UnitModel:
    target: str
    eff: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())
    serial_coef: float = 1.0
    serial_const_us: float = 0.0
    interactions: tuple[Interaction, ...] = ()
    occupancy_knee: float = DEFAULT_OCCUPANCY_KNEE
    parent: str | None = None  # version of the model this one was fitted from
    fit_info: Mapping[str, Any] = field(default_factory=_empty_info)

    # ---- identity ------------------------------------------------------------

    def params(self) -> dict[str, Any]:
        """Everything that changes a prediction (fit bookkeeping excluded)."""
        return {
            "target": self.target,
            "eff": {u: self.eff[u] for u in sorted(self.eff)},
            "serial_coef": self.serial_coef,
            "serial_const_us": self.serial_const_us,
            "interactions": [[*i.units, i.gamma] for i in self.interactions],
            "occupancy_knee": self.occupancy_knee,
        }

    @property
    def version(self) -> str:
        text = json.dumps(self.params(), sort_keys=True, separators=(",", ":"))
        return "unit-" + hashlib.sha256(text.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.params(),
            "version": self.version,
            "parent": self.parent,
            "interaction_basis": [i.basis for i in self.interactions],
            "fit_info": dict(self.fit_info),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> UnitModel:
        basis = list(data.get("interaction_basis", []))
        inter = tuple(
            Interaction((str(u), str(v)), float(g), str(basis[i]) if i < len(basis) else "")
            for i, (u, v, g) in enumerate(data.get("interactions", []))
        )
        model = UnitModel(
            target=str(data["target"]),
            eff={str(k): float(v) for k, v in dict(data.get("eff", {})).items()},
            serial_coef=float(data.get("serial_coef", 1.0)),
            serial_const_us=float(data.get("serial_const_us", 0.0)),
            interactions=inter,
            occupancy_knee=float(data.get("occupancy_knee", DEFAULT_OCCUPANCY_KNEE)),
            parent=data.get("parent"),
            fit_info=dict(data.get("fit_info", {})),
        )
        if "version" in data and data["version"] != model.version:
            raise SchemaError(f"model version {data['version']} does not match its parameters")
        return model

    # ---- prediction ----------------------------------------------------------

    def peak_times(
        self, per_unit_work: Mapping[str, float], serial: Mapping[str, float], hw: HwModel
    ) -> tuple[dict[str, float], float]:
        """Per-unit and serial times at peak (us); units with zero work are left out."""
        per_unit = {u: unit_time_us(u, w, hw) for u, w in sorted(per_unit_work.items()) if w}
        serial_us = sum(unit_time_us(u, w, hw) for u, w in sorted(serial.items()) if w)
        return per_unit, serial_us

    def total_us(
        self,
        per_unit_peak_us: Mapping[str, float],
        serial_peak_us: float,
        occupancy_factor: float = 1.0,
    ) -> float:
        times = {
            u: t / (self.eff.get(u, 1.0) * (1.0 if u in OCCUPANCY_EXEMPT else occupancy_factor))
            for u, t in per_unit_peak_us.items()
        }
        total = max(times.values(), default=0.0)
        total += self.serial_coef * serial_peak_us + self.serial_const_us
        for i in self.interactions:
            total += i.gamma * min(times.get(i.units[0], 0.0), times.get(i.units[1], 0.0))
        return total

    def occupancy_factor(self, feedback: Mapping[str, Any] | None, hw: HwModel) -> float:
        """Latency-hiding factor in (0, 1] from compile feedback; 1 without feedback.

        Raises NotSupported when the feedback leaves no CTA resident on an SM.
        """
        if not feedback:
            return 1.0
        occ = occupancy(feedback, hw)
        if occ <= 0:
            raise NoResidentCta(f"compile feedback {dict(feedback)} fits no CTA on an SM")
        return min(1.0, occ / self.occupancy_knee)

    def predict(
        self,
        per_unit_work: dict[str, float],
        serial: Mapping[str, float],
        hw: HwModel,
        feedback: dict[str, Any] | None = None,
    ) -> Prediction:
        per_unit, serial_us = self.peak_times(per_unit_work, serial, hw)
        total = self.total_us(per_unit, serial_us, self.occupancy_factor(feedback, hw))
        return Prediction(per_unit=per_unit, serial=serial_us, total=total, model=self.version)

    def repredict(
        self, p: Prediction, hw: HwModel, feedback: Mapping[str, Any] | None = None
    ) -> Prediction:
        """The same peak times under this model and optional compile feedback."""
        total = self.total_us(p.per_unit, p.serial, self.occupancy_factor(feedback, hw))
        return Prediction(per_unit=p.per_unit, serial=p.serial, total=total, model=self.version)

    # ---- calibration ---------------------------------------------------------

    def fit(self, data: list[tuple[Prediction, Measurement]]) -> UnitModel:
        """A new version fitted on ``data``; this one is left unchanged."""
        model, _ = self.fit_report(data)
        return model

    def fit_report(self, data: list[tuple[Prediction, Measurement]]) -> tuple[UnitModel, FitReport]:
        from kta.models.fit import fit_model  # the fitting code needs numpy; load it lazily

        report = fit_model(self, data)
        model = replace(
            self,
            eff=report.eff,
            serial_coef=report.serial_coef,
            serial_const_us=report.serial_const_us,
            interactions=report.interactions,
            parent=self.version,
            fit_info=report.info,
        )
        return model, report


def occupancy(feedback: Mapping[str, Any], hw: HwModel) -> float:
    """Resident threads per SM over the maximum, from regs/smem/threads per CTA.

    Without ``threads`` the fraction is over CTA slots, limited by shared memory only.
    """
    missing = [k for k in OCCUPANCY_LIMITS if k not in hw.limits]
    if missing:
        raise NotSupported(f"HwModel {hw.target} v{hw.version} lacks limits {missing}")
    lim = hw.limits
    ctas = int(lim["max_ctas_per_sm"])
    smem = int(feedback.get("smem") or 0) + int(lim.get("smem_reserved_per_cta", 0))
    if smem > 0:
        ctas = min(ctas, int(lim["smem_per_sm"]) // smem)
    threads = feedback.get("threads")
    if not threads:
        return ctas / float(lim["max_ctas_per_sm"])
    ctas = min(ctas, int(lim["max_threads_per_sm"]) // int(threads))
    regs = int(feedback.get("regs") or 0)
    if regs > 0:
        ctas = min(ctas, int(lim["regs_per_sm"]) // (regs * int(threads)))
    return max(0, ctas) * int(threads) / float(lim["max_threads_per_sm"])
