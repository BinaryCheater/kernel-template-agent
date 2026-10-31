"""L3: adapter for the external SASS-level GB10 simulator (docs/design/02 §4 "Four
fidelity levels"; D506).

The simulator lives in the research repository
(``kernel-lab/exports/gb10-simulator-calibration-20261020-v2``, entry ``tools/simulate.py``
with an upstream AI-Infra-Co-Design checkout). It is read-only reference material: KTA never
imports it. It runs as an external tool through the Executor on a CPU worker that has a
copy of the bundle, with the compiled binary as a job input. Until that job has run for
real and its output format has been checked against this adapter, ``simulate`` raises
``NotSupported`` (pending real run).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kta.core.errors import NotSupported
from kta.core.jobs import Job
from kta.models.base import L3Result

BUNDLE = "kernel-lab/exports/gb10-simulator-calibration-20261020-v2"


@dataclass(frozen=True)
class ExternalSassSimulator:
    """The research repository's GB10 simulator as an external tool (never imported)."""

    bundle_dir: str  # on the worker
    upstream_dir: str  # AI-Infra-Co-Design checkout on the worker
    timeout_s: int = 600

    def job(self, cubin_sha256: str, launch: Mapping[str, int]) -> Job:
        grid = ",".join(f"{k}={v}" for k, v in sorted(launch.items()))
        return Job(
            kind="shell",
            argv=[
                "{python}",
                f"{self.bundle_dir}/tools/simulate.py",
                "--cubin",
                "{job}/kernel.cubin",
                "--launch",
                grid,
                "--upstream",
                self.upstream_dir,
                "--out",
                "{job}/out/l3.json",
            ],
            requires={"pool": "cpu"},
            inputs={"kernel.cubin": cubin_sha256},
            timeout_s=self.timeout_s,
        )

    def simulate(self, cubin_sha256: str, launch: Mapping[str, int]) -> L3Result:
        raise NotSupported(
            "L3 SASS-level simulator: the external tool has not run through the Executor for "
            "real yet, and its output format is unchecked (pending real run, D506)"
        )
