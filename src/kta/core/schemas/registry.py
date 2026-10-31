"""Kind name -> body model, schema version, upcasters and envelope rules.

This is a constant table, not a mutable registry: adding a kind is a code change with a
schema version, so stored records always know which model validated them. A kind's
version is ``1 + len(upcasters)``; a stored body of version n is read through
``upcasters[n - 1:]`` (D440).
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
from kta.core.schemas import upcast as up


class InvalidBody(SchemaError):
    """A body does not validate against its kind's model (a ValueError, see SchemaError)."""


@dataclass(frozen=True)
class KindSpec:
    model: type[BaseModel]
    # refs keys that must be present (spec/schemas.md §1 and per-kind "refs" lines).
    required_refs: tuple[str, ...] = ()
    # refs key -> kind the referenced record must have.
    ref_kinds: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    # upcasters[n - 1] maps a stored body of version n to version n + 1.
    upcasters: tuple[up.Upcaster, ...] = ()
    # Readable but superseded: new code writes the replacement kind (D442, D444).
    legacy_of: str | None = None

    @property
    def version(self) -> int:
        return 1 + len(self.upcasters)


_ENV = {"env": "EnvManifest"}
_JOB = {"env": "EnvManifest", "worker": "Worker"}

KINDS: dict[str, KindSpec] = {
    # knowledge
    "Pattern": KindSpec(kn.Pattern, upcasters=(up.pattern_v1,)),
    "Template": KindSpec(
        kn.Template,
        ref_kinds={"previous": "Template", "sources": "Source"},
        upcasters=(up.template_v1,),
    ),
    "Realisation": KindSpec(
        kn.Realisation,
        required_refs=("template",),
        ref_kinds={"template": "Template", "sources": "Source", "previous": "Realisation"},
    ),
    "Chain": KindSpec(kn.Chain, ref_kinds={"templates": "Template"}),
    "Family": KindSpec(
        kn.Family, ref_kinds={"chain": "Chain", "previous": "Family", "problem": "Problem"}
    ),
    "Reference": KindSpec(
        kn.Reference,
        ref_kinds={"source": "Source", "problem": "Problem"},
        upcasters=(up.reference_v1,),
    ),
    "Reconstruction": KindSpec(
        kn.Reconstruction,
        ref_kinds={
            "reference": "Reference",
            "chain": "Chain",
            "gaps": "Gap",
            "measurement": "Measurement",
        },
        upcasters=(up.reconstruction_v1,),
    ),
    "Structure": KindSpec(kn.Structure),
    "Finding": KindSpec(kn.Finding, ref_kinds={"sources": "Source"}),
    "Coupling": KindSpec(kn.Coupling, ref_kinds={"sources": "Source", "family": "Family"}),
    "Regime": KindSpec(kn.Regime, legacy_of="Finding"),
    "Negative": KindSpec(kn.Negative, legacy_of="Finding"),
    "Gap": KindSpec(kn.Gap, upcasters=(up.gap_v1,)),
    # hardware
    "Target": KindSpec(hw.Target),
    "Capability": KindSpec(hw.Capability, ref_kinds={"sources": "Source"}),
    "CapabilityDecl": KindSpec(
        hw.CapabilityDecl, ref_kinds={"target": "Target", "sources": "Source"}
    ),
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


def upcast_body(kind: str, schema: int, body: dict[str, Any]) -> dict[str, Any]:
    """``body`` stored as version ``schema`` of ``kind``, in the current version's shape."""
    spec = kind_spec(kind)
    if schema < 1 or schema > spec.version:
        raise InvalidBody(f"{kind} schema {schema}; current is {spec.version}")
    for step in spec.upcasters[schema - 1 :]:
        try:
            body = step(body)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidBody(f"{kind} v{schema} body cannot be upcast: {exc}") from exc
    return body


def validate_body(kind: str, body: dict[str, Any]) -> None:
    """Raise InvalidBody (a ValueError subclass) when ``body`` is not a valid ``kind``."""
    parse_body(kind, body)


def normalize_body(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    """Validated body in JSON form with every default filled in.

    Records are hashed after normalization, so a body written with or without its
    default values gets the same ID.
    """
    return parse_body(kind, body).model_dump(mode="json")
