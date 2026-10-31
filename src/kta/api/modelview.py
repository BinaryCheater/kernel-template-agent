"""Hardware-model views (WP16, D511): /model/scoreboard and
/model/versions/{id}/timeline (spec/modules/api-web.md).

The scoreboard lists ModelVersion records with their regression numbers (rank correlation
and error per resource, kta.models.regression). The timeline view pairs, for one kernel of
one version, each configuration's stored L2 prediction (an ``l2_simulation`` Report: waves
and busy intervals) with its measured time. Every row carries its record's origin so the
page can mark fixture data.
"""

from __future__ import annotations

from typing import Any

from kta.api.common import Api
from kta.core.schemas.hwmodel import ModelVersion
from kta.models.regression import scoreboard as rows
from kta.store.sqlite import RecordNotFound


def scoreboard(api: Api, target: str | None) -> dict[str, Any]:
    versions = rows(api.store, target)
    keys = sorted({k for v in versions for r in v["results"] for k in r["error_by_resource"]})
    return {
        "versions": versions,
        "error_keys": keys,
        "fixture": any(v["origin"] != "real" for v in versions),
    }


def timeline(api: Api, version_id: str, kernel: str) -> dict[str, Any]:
    rec = api.store.get(version_id)
    if rec.kind != "ModelVersion":
        raise RecordNotFound(f"{version_id} is a {rec.kind}, not a ModelVersion")
    mv = ModelVersion.model_validate(rec.body)
    found = [r for r in mv.results if r.kernel == kernel]
    if not found:
        raise RecordNotFound(f"ModelVersion {mv.label} has no regression result for {kernel}")
    res = found[0]
    configs = []
    for i, measured in enumerate(res.measured_us):
        pid = res.predictions[i] if i < len(res.predictions) else ""
        pred = api.store.get(pid) if pid else None
        data = pred.body.get("data", {}) if pred else {}
        configs.append(
            {
                "index": i,
                "prediction": pid or None,
                "measurement": (res.measurements[i] if i < len(res.measurements) else "") or None,
                "configuration": data.get("configuration"),
                "predicted_us": res.predicted_us[i],
                "measured_us": measured,
                "waves": data.get("waves", []),
                "intervals": data.get("intervals", {}),
                "busy_us": data.get("busy_us", {}),
                "idle_us": data.get("idle_us", {}),
                "bottleneck": data.get("bottleneck"),
                "origin": pred.origin if pred else rec.origin,
            }
        )
    return {
        "version": version_id,
        "label": mv.label,
        "kernel": kernel,
        "rank_correlation": res.rank_correlation,
        "error_by_resource": res.error_by_resource,
        "holdout": res.holdout,
        "origin": rec.origin,
        "configurations": configs,
    }
