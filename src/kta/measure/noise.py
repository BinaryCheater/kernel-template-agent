"""Noise probe: repeated measurements of one kernel, and the protocol revision it supports.

Each group is a separate ``measure`` job (a fresh process, context and input set), so
the spread of the group medians includes process-to-process effects, not only the
spread within one loop. The noise floor of the next protocol version is the largest
coefficient of variation of the group medians over the probed kernels, rounded up to
0.1% (decision D226).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from kta.backends.base import BuiltModule
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Case
from kta.core.schemas.process import Report
from kta.measure import stats
from kta.measure.protocol import V1, ensure_protocol, load_params, revise
from kta.measure.runner import ACTOR, Measurer

log = logging.getLogger("kta.measure.noise")


def noise_probe(
    m: Measurer,
    module: BuiltModule,
    case: Case,
    shape: dict[str, Any],
    protocol_id: str,
    groups: int = 30,
    label: str = "",
) -> tuple[str, dict[str, Any]]:
    """``groups`` consecutive measurements of ``module``; a Report with their spread."""
    ids: list[str] = []
    medians: list[float] = []
    widths: list[float] = []
    temps: list[Any] = []
    clocks: list[Any] = []
    for g in range(groups):
        r = m.measure(module, case, shape, protocol_id)
        ids.append(r.id)
        medians.append(r.body.median_us)
        widths.append((r.body.ci_high_us - r.body.ci_low_us) / r.body.median_us)
        temps.append(r.body.temps.get("temp_c_end"))
        clocks.append(r.body.clocks.get("sm_mhz_end"))
        log.info("noise %s group %d/%d: %.3f us", label, g + 1, groups, r.body.median_us)
    summary = stats.noise_summary(medians)
    data: dict[str, Any] = {
        "label": label,
        "build": module.build_id,
        "shape": shape,
        "protocol": protocol_id,
        "medians_us": medians,
        "ci_rel_widths": widths,
        "temp_c_end": temps,
        "sm_mhz_end": clocks,
        "iterations": [m.store.get(i).body["n"] for i in ids],
        **summary,
    }
    text = (
        f"{label}: {groups} groups, median {summary['median_of_medians_us']:.3f} us, "
        f"CV of group medians {summary['cv'] * 100:.3f}%, deviation from the median "
        f"median {summary['dev_median'] * 100:.3f}% p90 {summary['dev_p90'] * 100:.3f}% "
        f"max {summary['dev_max'] * 100:.3f}%"
    )
    first = m.store.get(ids[0])
    body = Report(title="measure noise probe", subject="noise", summary=text, data=data)
    refs: dict[str, str | list[str]] = {
        "env": first.refs["env"],
        "measurements": ids,
        "protocol": protocol_id,
    }
    rid = m.store.put(Record.new("Report", body, ACTOR, m.clock.now(), refs=refs))
    log.info("noise report %s: %s", rid[:12], text)
    return rid, data


def floor_from(cvs: list[float]) -> float:
    return math.ceil(max(cvs) * 1000) / 1000


def revise_protocol(m: Measurer, base_id: str, report_ids: list[str]) -> str:
    """``measure-v2``: ``measure-v1`` with the noise floor measured by the probe reports."""
    base = load_params(m.store, base_id)
    if base != V1:
        raise ValueError("revise_protocol expects measure-v1 as its base")
    cvs = [float(m.store.get(r).body["data"]["cv"]) for r in report_ids]
    floor = floor_from(cvs)
    params = revise(base, floor)
    description = (
        f"measure-v1 with the noise floor measured by the WP3 noise probe on GB10: max CV of "
        f"group medians {max(cvs) * 100:.3f}% over {len(cvs)} kernel(s), rounded up to "
        f"{floor * 100:.1f}%. Significance threshold = {params.significance_factor:g} x floor "
        f"= {params.significance_factor * floor * 100:.1f}%. Other values unchanged."
    )
    refs: dict[str, str | list[str]] = {"base": base_id, "noise_reports": report_ids}
    return ensure_protocol(m.store, m.clock, 2, params, description, refs)
