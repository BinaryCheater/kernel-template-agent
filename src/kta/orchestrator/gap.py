"""Gap decomposition and routing (docs/design/02 §1 P7, §6.1; WP18 D520-D523).

Per bucket, with ``P1`` the solver's best prediction and ``Popt`` the model optimum::

    T_meas - T_SOL = eps + delta + dpre + L_rep + irreducible
    eps   = T_meas - P1                    model error at the solver's choice
    delta = P1 - Popt                      0 when certified, mip_gap x P1 when bounded
    dpre  = Popt - P_relaxed               needs the model optimum with fixed variables
                                           opened (FormulationPlan F3, WP19); else 0, not assessed
    irreducible                            declared irreducible time; else 0, not assessed
    L_rep = P_relaxed - T_SOL - irreducible   (SOL bounds it: L_rep <= P1 - T_SOL)

A term the inputs cannot settle is ``unbounded`` (delta without a certificate); one with no
input is ``not_assessed`` and its share stays in ``L_rep``. The class is the first entry of
the routing table's ``order`` whose term is significant (beyond the improvement band) or
unbounded while gap remains; later classes are not judged before earlier ones (D-T6).

Everything here is pure: ``judge`` (solve_loop) calls it with facts and Settings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from kta.core.errors import ConfigError

ROUTING_PATH = Path(__file__).with_name("routing.yaml")
TERMS = ("epsilon", "delta", "delta_pre", "l_rep", "irreducible")


@dataclass(frozen=True)
class Routing:
    version: int
    digest: str  # sha256 of the file, first 12 hex digits
    order: tuple[str, ...]
    classes: Mapping[str, Mapping[str, Any]]
    escalations: Mapping[str, Mapping[str, Any]]
    canaries: tuple[Mapping[str, Any], ...]

    @property
    def id(self) -> str:
        return f"routing@{self.version}:{self.digest}"


@cache
def routing(path: Path = ROUTING_PATH) -> Routing:
    text = path.read_text()
    raw = yaml.safe_load(text)
    order = tuple(raw["order"])
    classes = raw["classes"]
    missing = [c for c in (*order, "none") if c not in classes]
    if missing:
        raise ConfigError(f"routing table {path.name}: no route for {missing}")
    for name, r in {**classes, **raw.get("escalations", {})}.items():
        if r.get("to") not in ("waiting", "closed", "accepted"):
            raise ConfigError(f"routing {name}: to must be waiting, closed or accepted")
        if r["to"] == "waiting" and not r.get("task"):
            raise ConfigError(f"routing {name}: a waiting route needs a task")
    return Routing(
        int(raw["version"]),
        hashlib.sha256(text.encode()).hexdigest()[:12],
        order,
        classes,
        raw.get("escalations", {}),
        tuple(raw.get("canaries", [])),
    )


def band_us(t_us: float, ci_us: float | None, band: Mapping[str, Any]) -> float:
    """The improvement band (12 §3): max(measurement CI half-width, rel_of_t x T)."""
    ci = float(ci_us or 0.0) if band["use_measurement_ci"] else 0.0
    return max(ci, float(band["rel_of_t"]) * t_us)


def _term(us: float | None, status: str, note: str = "") -> dict[str, Any]:
    return {"us": us, "status": status, "note": note}


def _delta(b: Mapping[str, Any], p1: float) -> dict[str, Any]:
    solver: Mapping[str, Any] = b.get("solver") or {}
    if solver.get("model_optimal"):
        return _term(0.0, "value", "model-optimal")
    gap = solver.get("mip_gap")
    if gap is not None and solver.get("method") == "mip":
        return _term(float(gap) * p1, "bounded", f"MIP gap {float(gap):.3g}")
    return _term(None, "unbounded", "no optimality certificate")


def decompose(b: Mapping[str, Any], band: Mapping[str, Any]) -> dict[str, Any]:
    """Terms of one judged bucket (see the module docstring); ``b`` carries ``best_us``,
    ``ci_us``, ``top_pred_us`` (else the smallest prediction measured), ``bound_us``,
    ``solver``, optional ``relaxed_pred_us`` and ``irreducible_us``."""
    t, sol = float(b["best_us"]), float(b["bound_us"])
    preds = [float(p) for p, _ in b.get("pairs", [])] + [float(b["predicted_us"])]
    p1 = float(b.get("top_pred_us") or min(preds))
    d = _delta(b, p1)
    popt = p1 - (d["us"] or 0.0)
    relaxed = b.get("relaxed_pred_us")
    irr = b.get("irreducible_us")
    if relaxed is None:
        dpre = _term(0.0, "not_assessed", "no relaxed solve (FormulationPlan, WP19)")
        p_rel = popt
    else:
        dpre, p_rel = _term(popt - float(relaxed), "value"), float(relaxed)
    irreducible = (
        _term(0.0, "not_assessed", "no irreducible term declared")
        if irr is None
        else _term(float(irr), "value")
    )
    l_rep = p_rel - sol - float(irr or 0.0)
    terms = {
        "epsilon": _term(t - p1, "value"),
        "delta": d,
        "delta_pre": dpre,
        "l_rep": _term(l_rep, "value", "includes delta_pre" if relaxed is None else ""),
        "irreducible": irreducible,
    }
    return {
        "t_meas_us": t,
        "t_sol_us": sol,
        "gap_us": t - sol,
        "top_pred_us": p1,
        "band_us": band_us(t, b.get("ci_us"), band),
        "l_rep_bound_us": p1 - sol,
        "terms": terms,
        "unpriced": dict(b.get("unpriced") or {}),
    }


def classify(dec: Mapping[str, Any], order: Sequence[str], failed: bool, model_off: bool) -> str:
    """The first class in attribution order whose term is significant or unjudgeable."""
    if failed:
        return "correctness"
    band = float(dec["band_us"])
    terms: Mapping[str, Mapping[str, Any]] = dec["terms"]
    remaining = float(dec["gap_us"])
    for cls in order:
        if cls == "correctness":
            continue
        term = terms[cls]
        if cls == "epsilon" and model_off:
            return cls
        if term["status"] == "unbounded":
            if remaining > band:
                return cls
            continue
        us = float(term["us"])
        if abs(us) > band:
            return cls
        remaining -= us
    return "none"


def accepts(dec: Mapping[str, Any], reference_us: float | None, rule: Mapping[str, Any]) -> bool:
    """12 §4 accept: not worse than the strongest reference (within the band) and the
    unexplained part (|eps|) below max_unexplained_rel x T; or the gap is explained as
    irreducible (gap - irreducible within the band). No reference: the first clause fails
    unless the reference condition is switched off (D522)."""
    t, band = float(dec["t_meas_us"]), float(dec["band_us"])
    eps = abs(float(dec["terms"]["epsilon"]["us"]))
    small = eps < float(rule["max_unexplained_rel"]) * t
    if rule["not_worse_than_strongest_reference"]:
        ref_ok = reference_us is not None and t <= float(reference_us) + band
    else:
        ref_ok = True
    irr = float(dec["terms"]["irreducible"]["us"] or 0.0)
    explained = float(dec["gap_us"]) - irr <= band
    return (ref_ok and small) or (bool(rule["accept_if_fully_irreducible"]) and explained)


def case_class(classes: Sequence[str], order: Sequence[str]) -> str:
    """The most upstream bucket class: the case is routed by the earliest failure."""
    rank = {c: i for i, c in enumerate(order)}
    known = [c for c in classes if c in rank]
    return min(known, key=lambda c: rank[c]) if known else "none"


def task_for(route: Mapping[str, Any], has_reference: bool) -> tuple[str, str]:
    """(task type, goal) of a waiting route."""
    if not has_reference and route.get("if_no_reference"):
        return str(route["if_no_reference"]), str(route.get("goal_no_reference", route["goal"]))
    return str(route["task"]), str(route["goal"])


def selection(arms: Mapping[str, Any], e: int) -> dict[str, Any]:
    """Selection controls of one bucket (12 §3, D-T13): best measured time of the model's
    points, of the model's first ``e`` by prediction (the same count as the random arm),
    of the random arm and of the copied assignment. A record, not a route (D524)."""
    model = sorted((float(p), float(m)) for p, m in arms.get("model", []))
    rnd = [float(m) for m in arms.get("random", [])]
    copied = [float(m) for m in arms.get("copied", [])]
    out: dict[str, Any] = {
        "model_best_us": min((m for _, m in model), default=None),
        "model_top_e_us": min((m for _, m in model[:e]), default=None) if e else None,
        "random_best_us": min(rnd, default=None),
        "random_n": len(rnd),
        "copied_us": copied[0] if copied else None,
        "copied_from": arms.get("copied_from"),
    }
    mb, rb = out["model_top_e_us"], out["random_best_us"]
    out["model_beaten_by_random"] = mb is not None and rb is not None and rb < mb
    return out
