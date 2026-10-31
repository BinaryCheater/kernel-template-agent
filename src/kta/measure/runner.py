"""measure and compare: timed runs in the gpu-exclusive pool, Measurement records.

Every timed run is one ``measure`` job (pool ``gpu-exclusive``, so the Executor holds
the GPU lease for it). The worker returns per-iteration times and NVML samples; the
control plane computes the median and bootstrap CI and writes the Measurement. Seeds
of the input, order and bootstrap streams go into the ``params`` blob.

Quiet host (WP23, D605): each timing job also returns host readings; a job whose host
was not quiet (``kta.measure.quiet``, thresholds from the controls knob
``measure.quiet_thresholds``) is written as a Report ``measurement_interfered`` and run
again, up to ``max_requeues`` times. Its times never become a Measurement. When every
attempt was interfered the Measurer raises ``HostNotQuiet`` (kind infra), so the caller's
infrastructure retry applies.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kta.backends.base import BuiltModule
from kta.core.context import Clock, Context
from kta.core.controls import baseline
from kta.core.errors import ExecutionError
from kta.core.ids import canonical_json
from kta.core.jobs import JobResult, JobRunner
from kta.core.registry import resolve
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Case, Measurement
from kta.core.schemas.process import Report
from kta.measure import jobs, quiet, stats
from kta.measure.protocol import MeasureParams, load_params
from kta.store.base import Store

log = logging.getLogger("kta.measure")
ACTOR = Actor(type="tool", name="kta measure@1")
RngFactory = Callable[[str], random.Random]


@dataclass(frozen=True)
class Measured:
    id: str
    body: Measurement
    times_us: list[float]


@dataclass(frozen=True)
class Comparison:
    a: Measured
    b: Measured
    rel_diff: float  # (median_b - median_a) / median_a
    ci_low: float
    ci_high: float
    significant: bool
    verdict: stats.Verdict
    report_id: str


@dataclass(frozen=True)
class Timed:
    result: JobResult
    timing: dict[str, Any]
    params_blob: str
    params: MeasureParams
    quiet: quiet.QuietVerdict | None = None
    interfered: tuple[str, ...] = ()  # Report IDs of the attempts that were not counted


class HostNotQuiet(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__("infra", message)


class Measurer:
    def __init__(
        self,
        store: Store,
        ex: JobRunner,
        clock: Clock,
        rng: RngFactory,
        seed: int = 0,
        thresholds: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.ex = ex
        self.clock = clock
        self.rng = rng
        self.seed = seed
        # None = the baseline values of measure.quiet_thresholds (callers with a controls
        # snapshot pass theirs).
        self.thresholds = thresholds if thresholds is not None else baseline().quiet_thresholds

    @classmethod
    def from_ctx(cls, ctx: Context) -> Measurer:
        return cls(
            resolve("store", ctx),
            resolve("executor", ctx),
            ctx.clock,
            ctx.rng,
            ctx.seed,
            ctx.controls.quiet_thresholds,
        )

    def _interfered(self, result: JobResult, timing: dict[str, Any], v: quiet.QuietVerdict) -> str:
        """Keep the uncounted attempt as evidence of the interference (never a Measurement)."""
        body = Report(
            title="measurement interfered",
            subject="measurement_interfered",
            summary="; ".join(v.reasons),
            data={"job_id": result.job_id, "quiet": v.to_dict(), "thresholds": self.thresholds},
        )
        refs: dict[str, str | list[str]] = {"env": result.env_manifest, "worker": result.worker}
        blobs = {"timing": result.outputs["timing.json"], "stderr": result.stderr_blob}
        rid = self.store.put(
            Record.new("Report", body, ACTOR, self.clock.now(), refs=refs, blobs=blobs)
        )
        log.warning("measure job %s interfered (%s): requeued", result.job_id, body.summary)
        return rid

    def _run_quiet(
        self, job: Any
    ) -> tuple[JobResult, dict[str, Any], quiet.QuietVerdict, list[str]]:
        interfered: list[str] = []
        attempts = 1 + int(self.thresholds.get("max_requeues", 0))
        for _ in range(attempts):
            result = self.ex.run(job)
            if not result.ok or "timing.json" not in result.outputs:
                stderr = self.ex.read_blob(result.stderr_blob).decode("utf-8", "replace")[-2000:]
                raise ExecutionError(
                    "nonzero" if result.error_kind == "none" else result.error_kind,
                    f"measure job {result.job_id} failed: {stderr}",
                )
            timing = json.loads(self.ex.read_blob(result.outputs["timing.json"]))
            verdict = quiet.assess(timing, self.thresholds)
            if not verdict.interfered:
                return result, timing, verdict, interfered
            interfered.append(self._interfered(result, timing, verdict))
        raise HostNotQuiet(
            f"host not quiet in {attempts} attempts (reports {', '.join(interfered)})"
        )

    def _time(
        self, modules: list[BuiltModule], case: Case, shape: dict[str, Any], protocol_id: str
    ) -> Timed:
        params = load_params(self.store, protocol_id)
        problem = jobs.problem_of(self.store, case)
        tag = f"{canonical_json(shape).decode()}:{','.join(m.build_id for m in modules)}"
        streams = {"inputs": f"measure.inputs:{tag}", "order": f"measure.order:{tag}"}
        rng_in = self.rng(streams["inputs"])
        sets = [
            jobs.random_inputs(problem, shape, "normal", rng_in)
            for _ in range(params.min_input_sets)
        ]
        settings = {
            k: getattr(params, k)
            for k in (
                "l2_flush_factor",
                "nan_fill",
                "warmup_min_iters",
                "warmup_min_ms",
                "min_iters",
                "min_ms",
                "max_iters",
            )
        }
        settings["order_seed"] = self.rng(streams["order"]).getrandbits(31)
        requires = {"gpu": jobs.target_sm(self.store, case.target)}
        bench = jobs.bench_job(
            self.ex,
            self.store,
            modules,
            problem,
            shape,
            sets,
            "time",
            "measure",
            requires,
            settings,
        )
        result, timing, verdict, interfered = self._run_quiet(bench.job)
        meta = {
            "job_id": result.job_id,  # names the bootstrap streams, so a replay can recompute CIs
            "spec": bench.spec_blob,
            "shape": shape,
            "ctx_seed": self.seed,
            "rng_streams": streams,
            "input_dist": "normal",
            "input_sets": params.min_input_sets,
        }
        blob = self.ex.put_blob(json.dumps(meta, sort_keys=True).encode(), "application/json")
        return Timed(result, timing, blob, params, verdict, tuple(interfered))

    def _record(
        self,
        t: Timed,
        index: int,
        module: BuiltModule,
        case: Case,
        protocol_id: str,
    ) -> Measured:
        body, times_us = self.measurement_body(t, index, module.build_id, protocol_id)
        refs: dict[str, str | list[str]] = {
            "env": t.result.env_manifest,
            "worker": t.result.worker,
            "build": module.build_id,
            "protocol": protocol_id,
        }
        return self.write_measurement(t, body, times_us, refs, case)

    def measurement_body(
        self, t: Timed, index: int, subject: str, protocol_id: str
    ) -> tuple[Measurement, list[float]]:
        """Median, bootstrap CI and GPU state of the ``index``-th timed module of ``t``."""
        times_us = [ms * 1000 for ms in t.timing["times_ms"][index]]
        boot = self.rng(f"measure.bootstrap:{subject}:{t.result.job_id}:{index}")
        med, lo, hi = stats.bootstrap_median(
            times_us, t.params.bootstrap_resamples, t.params.ci_level, boot
        )
        s, e = t.timing["env_start"], t.timing["env_end"]
        body = Measurement(
            median_us=med,
            ci_low_us=lo,
            ci_high_us=hi,
            n=len(times_us),
            clocks={
                "sm_mhz_start": s["sm_mhz"],
                "sm_mhz_end": e["sm_mhz"],
                "sm_max_mhz": s["sm_max_mhz"],
                "mem_mhz_start": s["mem_mhz"],
                "mem_mhz_end": e["mem_mhz"],
                "throttle_reasons_start": s["throttle_reasons"],
                "throttle_reasons_end": e["throttle_reasons"],
                "locked": False,
                "flush_bytes": t.timing["flush_bytes"],
                "warmup_iters": t.timing["warmup_iters"],
                "loop_s": t.timing["loop_s"],
                "quiet": t.quiet.to_dict() if t.quiet is not None else None,
                "interfered_attempts": list(t.interfered),
            },
            temps={
                "temp_c_start": s["temp_c"],
                "temp_c_end": e["temp_c"],
                "power_w_start": s["power_w"],
                "power_w_end": e["power_w"],
            },
            protocol=protocol_id,
        )
        return body, times_us

    def write_measurement(
        self,
        t: Timed,
        body: Measurement,
        times_us: list[float],
        refs: dict[str, str | list[str]],
        case: Case,
    ) -> Measured:
        if self.store.exists(case.problem):
            refs["problem"] = case.problem
        blobs = {
            "timing": t.result.outputs["timing.json"],
            "params": t.params_blob,
            "stdout": t.result.stdout_blob,
            "stderr": t.result.stderr_blob,
        }
        rec = Record.new("Measurement", body, ACTOR, self.clock.now(), refs=refs, blobs=blobs)
        mid = self.store.put(rec)
        log.info(
            "measurement %s: median %.3f us [%.3f, %.3f] n=%d",
            mid[:12],
            body.median_us,
            body.ci_low_us,
            body.ci_high_us,
            body.n,
        )
        return Measured(mid, body, times_us)

    def measure(
        self, module: BuiltModule, case: Case, shape: dict[str, Any], protocol_id: str
    ) -> Measured:
        t = self._time([module], case, shape, protocol_id)
        return self._record(t, 0, module, case, protocol_id)

    def compare(
        self,
        a: BuiltModule,
        b: BuiltModule,
        case: Case,
        shape: dict[str, Any],
        protocol_id: str,
    ) -> Comparison:
        """Both modules in one job, interleaved in random order per round."""
        t = self._time([a, b], case, shape, protocol_id)
        ma = self._record(t, 0, a, case, protocol_id)
        mb = self._record(t, 1, b, case, protocol_id)
        p = t.params
        boot = self.rng(f"measure.compare:{ma.id}:{mb.id}")
        rel, lo, hi = stats.bootstrap_rel_diff(
            ma.times_us, mb.times_us, p.bootstrap_resamples, p.ci_level, boot
        )
        verdict = stats.significance(
            (ma.body.ci_low_us, ma.body.ci_high_us),
            (mb.body.ci_low_us, mb.body.ci_high_us),
            rel,
            p.noise_floor_rel,
            p.significance_factor,
        )
        data = {
            "a": {"measurement": ma.id, "build": a.build_id, "median_us": ma.body.median_us},
            "b": {"measurement": mb.id, "build": b.build_id, "median_us": mb.body.median_us},
            "rel_diff": rel,
            "rel_diff_ci": [lo, hi],
            "significant": verdict.significant,
            "cis_disjoint": verdict.cis_disjoint,
            "above_floor": verdict.above_floor,
            "threshold_rel": verdict.threshold_rel,
            "noise_floor_rel": p.noise_floor_rel,
            "protocol": protocol_id,
            "job_id": t.result.job_id,
        }
        summary = (
            f"b vs a: {rel * 100:+.2f}% (95% CI {lo * 100:+.2f}%..{hi * 100:+.2f}%); "
            f"significant={verdict.significant} (threshold {verdict.threshold_rel * 100:.2f}%)"
        )
        body = Report(title="measure compare", subject="compare", summary=summary, data=data)
        refs: dict[str, str | list[str]] = {
            "env": t.result.env_manifest,
            "measurements": [ma.id, mb.id],
            "protocol": protocol_id,
        }
        rid = self.store.put(Record.new("Report", body, ACTOR, self.clock.now(), refs=refs))
        log.info("compare %s: %s", rid[:12], summary)
        return Comparison(ma, mb, rel, lo, hi, verdict.significant, verdict, rid)


def measure(
    ctx: Context, module: BuiltModule, case: Case, shape: dict[str, Any], protocol_id: str
) -> Measured:
    return Measurer.from_ctx(ctx).measure(module, case, shape, protocol_id)


def compare(
    ctx: Context,
    a: BuiltModule,
    b: BuiltModule,
    case: Case,
    shape: dict[str, Any],
    protocol_id: str,
) -> Comparison:
    return Measurer.from_ctx(ctx).compare(a, b, case, shape, protocol_id)
