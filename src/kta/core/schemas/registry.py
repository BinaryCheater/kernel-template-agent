"""Kind name -> body model, schema version and envelope rules.

This is a constant table, not a mutable registry: adding a kind is a code change with a
schema version, so stored records always know which model validated them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from kta.core.errors import SchemaError
from kta.core.schemas import evaluation as ev
from kta.core.schemas import hardware as hw
from kta.core.schemas import knowledge as kn
from kta.core.schemas import process as pr


class InvalidBody(SchemaError):
    """A body does not validate against its kind's model (a ValueError, see SchemaError)."""


@dataclass(frozen=True)
class KindSpec:
    model: type[BaseModel]
    version: int = 1
    # refs keys that must be present (spec/schemas.md §1 and per-kind "refs" lines).
    required_refs: tuple[str, ...] = ()
    # refs key -> kind the referenced record must have.
    ref_kinds: dict[str, str] = field(default_factory=lambda: dict[str, str]())


_ENV = {"env": "EnvManifest"}
_JOB = {"env": "EnvManifest", "worker": "Worker"}

KINDS: dict[str, KindSpec] = {
    # knowledge
    "Pattern": KindSpec(kn.Pattern),
    "Template": KindSpec(kn.Template, ref_kinds={"previous": "Template", "sources": "Source"}),
    "Chain": KindSpec(kn.Chain, ref_kinds={"templates": "Template"}),
    "Family": KindSpec(
        kn.Family, ref_kinds={"chain": "Chain", "previous": "Family", "problem": "Problem"}
    ),
    "Reference": KindSpec(kn.Reference, ref_kinds={"source": "Source", "problem": "Problem"}),
    "Reconstruction": KindSpec(
        kn.Reconstruction,
        ref_kinds={
            "reference": "Reference",
            "chain": "Chain",
            "gaps": "Gap",
            "measurement": "Measurement",
        },
    ),
    "Structure": KindSpec(kn.Structure),
    "Regime": KindSpec(kn.Regime),
    "Negative": KindSpec(kn.Negative),
    "Gap": KindSpec(kn.Gap),
    # hardware
    "Target": KindSpec(hw.Target),
    "HwFeature": KindSpec(hw.HwFeature),
    "HwFact": KindSpec(hw.HwFact),
    "Worker": KindSpec(hw.Worker),
    "EnvManifest": KindSpec(hw.EnvManifest),
    "HwModel": KindSpec(hw.HwModel),
    # solving and evaluation
    "Problem": KindSpec(ev.Problem),
    "Contract": KindSpec(ev.Contract),
    "Case": KindSpec(ev.Case),
    "Candidate": KindSpec(ev.Candidate),
    "Prediction": KindSpec(ev.Prediction),
    "Build": KindSpec(ev.Build, required_refs=("env", "worker"), ref_kinds=_JOB),
    "Check": KindSpec(ev.Check, required_refs=("env", "worker"), ref_kinds=_JOB),
    "Measurement": KindSpec(ev.Measurement, required_refs=("env", "worker"), ref_kinds=_JOB),
    "Profile": KindSpec(ev.Profile, required_refs=("env", "worker"), ref_kinds=_JOB),
    "Bound": KindSpec(ev.Bound),
    "Solution": KindSpec(ev.Solution),
    # process and audit
    "StatusChange": KindSpec(pr.StatusChange),
    "TaskCard": KindSpec(pr.TaskCard),
    "AgentRun": KindSpec(pr.AgentRun, required_refs=("env",), ref_kinds=_ENV),
    "AgentProfile": KindSpec(pr.AgentProfile),
    "Submission": KindSpec(pr.Submission),
    "HumanAction": KindSpec(pr.HumanAction),
    "PoolSnapshot": KindSpec(pr.PoolSnapshot, ref_kinds={"previous": "PoolSnapshot"}),
    "Source": KindSpec(pr.Source),
    "Protocol": KindSpec(pr.Protocol),
    "ExperimentRun": KindSpec(pr.ExperimentRun),
    "Report": KindSpec(pr.Report, ref_kinds=_ENV),
    "ControlsVersion": KindSpec(pr.ControlsVersion, ref_kinds={"previous": "ControlsVersion"}),
}


def kind_spec(kind: str) -> KindSpec:
    spec = KINDS.get(kind)
    if spec is None:
        raise InvalidBody(f"unknown record kind {kind!r}")
    return spec


def model_for(kind: str) -> type[BaseModel]:
    return kind_spec(kind).model


def parse_body(kind: str, body: Any) -> BaseModel:
    """The validated body model; raises InvalidBody (a ValueError) on any problem."""
    model = model_for(kind)
    if not isinstance(body, dict):
        raise InvalidBody(f"{kind} body must be a mapping, got {type(body).__name__}")
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        raise InvalidBody(f"{kind}: {exc}") from exc


def validate_body(kind: str, body: dict[str, Any]) -> None:
    """Raise InvalidBody (a ValueError subclass) when ``body`` is not a valid ``kind``."""
    parse_body(kind, body)


def normalize_body(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    """Validated body in JSON form with every default filled in.

    Records are hashed after normalization, so a body written with or without its
    default values gets the same ID.
    """
    return parse_body(kind, body).model_dump(mode="json")
