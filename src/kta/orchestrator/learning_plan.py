"""Dry-run plans of the learning-loop pipeline per reference (02 §8.6; WP20, D569).

intake → decompose → admit → reproduce → ablate → cover. Intake is a list of Executor
jobs on the frozen source; reproduce and ablate are family variants that go through the
solve pipeline's existing path (``pipeline.generate`` → ``build`` → full checks →
measurement under the Protocol), one per run. Here they are *defined* and planned, not
submitted (code stage, D320): a plan lists every job or variant and the records it writes.
Which stages apply follows the reference's source class and cost tiers
(``kta.knowledge.learning``); decompose and admit are the T2 task and its done-checks,
cover is the expand loop's ``covered`` state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from kta.core.errors import SchemaError
from kta.core.jobs import Job
from kta.core.schemas.knowledge import Reference
from kta.knowledge.ablation import Design
from kta.knowledge.learning import (
    PerformanceCore,
    ReferenceFacts,
    priority,
    source_class,
    stages,
    tiers,
)
from kta.store.base import Store
from kta.store.querybuild import MAX_LIMIT

StageStatus = Literal["planned", "task", "not applicable"]
STAGE_TIER = {"intake": "L1", "reproduce": "L2", "ablate": "L3"}
SRC = "src"  # job-directory name of the frozen checkout


@dataclass(frozen=True)
class Stage:
    name: str
    status: StageStatus
    why: str
    jobs: list[Job] = field(default_factory=lambda: list[Job]())
    runs: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    writes: list[str] = field(default_factory=lambda: list[str]())


@dataclass(frozen=True)
class Plan:
    reference: str
    target: str
    source_class: str
    priority: float
    tiers: list[str]
    stages: list[Stage]
    dry_run: bool = True  # nothing is submitted (D320)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def intake_jobs(source: dict[str, Any], ref: Reference, target: str) -> list[Job]:
    """Clone at commit, license, hash, build in its container, run, SASS, profile."""
    rev, url = str(source["revision"]), str(source["locator"])
    gpu = {"target": target}
    return [
        Job("shell", ["git", "clone", "--no-checkout", url, SRC], timeout_s=1800),
        Job("shell", ["git", "-C", SRC, "checkout", "--detach", rev]),
        Job(
            "shell", ["sh", "-c", f"ls {SRC} | grep -iE '^(license|copying)' && cat {SRC}/LICENSE*"]
        ),
        Job("shell", ["git", "-C", SRC, "rev-parse", "HEAD^{tree}"]),
        Job("compile", ["sh", "-c", f"cd {SRC} && {ref.build}"], requires=gpu, timeout_s=3600),
        # run.sh: the launcher the reference build provides (the reference harness is a
        # pending real-run item, D569); measured under the Protocol by the measure path.
        Job("run", ["sh", "-c", f"cd {SRC} && ./run.sh"], requires=gpu),
        Job("shell", ["sh", "-c", f"cuobjdump -sass $(find {SRC} -name '*.o' -o -name '*.so')"]),
        Job(
            "profile", ["ncu", "--set", "full", "-o", "out/profile", f"{SRC}/run.sh"], requires=gpu
        ),
    ]


PIPELINE = ("generate", "build", "full_check", "measure")  # kta.orchestrator.pipeline path


def reproduce_runs(core: PerformanceCore, family: str | None) -> list[dict[str, Any]]:
    """The family at the core's configuration; compared with the reference within ε."""
    return [
        {
            "family": family or "<family of T1>",
            "config": dict(core.configuration),
            "tolerance": core.tolerance,
            "ops": list(PIPELINE),
            "compare": {"against": core.reference, "epsilon_rel": core.epsilon_rel},
        }
    ]


def ablation_runs(d: Design, family: str) -> list[dict[str, Any]]:
    """One variant per design run: chain steps dropped and variable values set."""
    return [
        {"family": family, "run": i, **d.settings(run), "ops": list(PIPELINE)}
        for i, run in enumerate(d.runs)
    ]


def unreproduced_steps(store: Store, reference_id: str) -> int:
    """Chain steps of the reference's latest Reconstruction whose Template no Family uses."""
    recons = [
        i
        for i in store.neighbors(reference_id, "reference", "in")
        if store.get(i).kind == "Reconstruction"
    ]
    if not recons:
        return 0
    chain_id = store.get(recons[-1]).refs.get("chain")
    if not isinstance(chain_id, str):
        return 0
    used = {
        str(i["template"])
        for f in store.query("Family", limit=MAX_LIMIT)
        if isinstance(f.refs.get("chain"), str)
        for i in store.get(str(f.refs["chain"])).body.get("items", [])
    }
    return sum(1 for i in store.get(chain_id).body.get("items", []) if i["template"] not in used)


def plan(
    store: Store,
    reference_id: str,
    target: str,
    facts: ReferenceFacts | None = None,
    core: PerformanceCore | None = None,
    ablation: Design | None = None,
    family: str | None = None,
) -> Plan:
    rec = store.get(reference_id)
    if rec.kind != "Reference":
        raise SchemaError(f"{reference_id} is a {rec.kind}, not a Reference")
    ref = Reference.model_validate(rec.body)
    src_id = rec.refs.get("source")
    source = store.get(src_id).body if isinstance(src_id, str) else {}
    facts = facts or ReferenceFacts(
        ref.name,
        target,
        runnable_on=tuple(ref.runnable_on),
        unreproduced_steps=unreproduced_steps(store, reference_id),
    )
    entered = tiers(facts)
    applies = stages(ref.source_class)
    out: list[Stage] = []
    for name in ("intake", "decompose", "admit", "reproduce", "ablate", "cover"):
        tier = STAGE_TIER.get(name)
        if name not in applies:
            out.append(
                Stage(name, "not applicable", f"source class {source_class(ref.source_class)}")
            )
        elif tier is not None and tier not in entered:
            out.append(Stage(name, "not applicable", f"tier {tier} not entered"))
        elif name == "intake":
            if not source.get("revision"):
                raise SchemaError(f"Reference {ref.name}: its Source has no revision (not frozen)")
            jobs = intake_jobs(source, ref, target)
            out.append(
                Stage(
                    name,
                    "planned",
                    "L1: runnable on the target",
                    jobs,
                    writes=["EnvManifest", "Build", "Measurement", "Profile"],
                )
            )
        elif name == "reproduce":
            if core is None:
                out.append(
                    Stage(name, "planned", "L2: needs the performance core (after decompose)")
                )
            else:
                out.append(
                    Stage(
                        name,
                        "planned",
                        f"L2 within eps {core.epsilon_rel}",
                        runs=reproduce_runs(core, family),
                        writes=["Build", "Check", "Measurement", "Report reproduction"],
                    )
                )
        elif name == "ablate":
            if ablation is None or family is None:
                out.append(Stage(name, "planned", "L3: needs the reproduced family"))
            else:
                out.append(
                    Stage(
                        name,
                        "planned",
                        f"L3 {ablation.mode}, {len(ablation.runs)} runs",
                        runs=ablation_runs(ablation, family),
                        writes=["Measurement", "Coupling"],
                    )
                )
        elif name in ("decompose", "admit"):
            out.append(Stage(name, "task", "T2 (decompose-kernel) and its admission done-check"))
        else:
            out.append(
                Stage(
                    name,
                    "planned",
                    "pool snapshot and coverage Report",
                    writes=["PoolSnapshot", "Report coverage"],
                )
            )
    return Plan(
        ref.name, target, source_class(ref.source_class), priority(ref.priority), entered, out
    )
