"""Replay a WP7 top-k run from its records in a fresh process (no new jobs).

Everything the run concluded is derived again from stored inputs and compared with what
was recorded:

- each bucket's solve: the family is materialized from the Family record's ``files/*``
  blobs, the formulation is assembled over the stored Templates, HwModel and model
  version, and the solver's shortlist (assignments, order, predicted totals) must equal
  the recorded Candidates and Predictions;
- each re-ranked prediction: the stored Build's feedback under the same model;
- each Measurement: median, CI and n from its ``timing`` blob and the seeds in its
  ``params`` blob (the bootstrap stream is named by the build or baseline and the job);
- each bucket's rank correlation.

The result is a Report (subject ``replay``) with ``consistent`` and every difference.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any

from kta.core.context import Context
from kta.core.ids import record_id
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Build, Prediction
from kta.core.schemas.hardware import HwModel
from kta.core.schemas.process import Report
from kta.formulation import assemble
from kta.knowledge import FamilyPackage, load_family
from kta.measure import stats
from kta.measure.protocol import load_params
from kta.models.calibrate import load_model
from kta.solve import rerank, solve
from kta.solve.result import Candidate
from kta.store.base import Store
from kta_wp7.pipeline import templates_of
from kta_wp7.records import ACTOR, CaseIds
from kta_wp7.stats import spearman

REL_TOL = 1e-9


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-12)


def materialize(store: Store, family_id: str, root: Path) -> FamilyPackage:
    rec = store.get(family_id)
    for key, sha in rec.blobs.items():
        if key.startswith("files/"):
            path = root / key.removeprefix("files/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(store.blob_path(sha).read_bytes())
    return load_family(root)


def replay_measurement(store: Store, mid: str, subject: str) -> list[str]:
    rec = store.get(mid)
    timing = json.loads(store.blob_path(rec.blobs["timing"]).read_bytes())
    meta = json.loads(store.blob_path(rec.blobs["params"]).read_bytes())
    params = load_params(store, str(rec.refs["protocol"]))
    times = [ms * 1000 for ms in timing["times_ms"][0]]
    if "job_id" not in meta:
        return [f"measurement {mid[:12]}: params blob has no job_id"]
    ctx_seed = int(meta["ctx_seed"])
    name = f"measure.bootstrap:{subject}:{meta['job_id']}:0"
    rng = _rng(ctx_seed, name)
    med, lo, hi = stats.bootstrap_median(times, params.bootstrap_resamples, params.ci_level, rng)
    body = rec.body
    out: list[str] = []
    for key, value in (("median_us", med), ("ci_low_us", lo), ("ci_high_us", hi)):
        if not _close(float(body[key]), value):
            out.append(f"measurement {mid[:12]} {key}: recorded {body[key]}, replayed {value}")
    if int(body["n"]) != len(times):
        out.append(f"measurement {mid[:12]} n: recorded {body['n']}, replayed {len(times)}")
    return out


def _rng(seed: int, name: str) -> random.Random:
    """Same derivation as ``Context.rng`` (kta.core.context)."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _candidates(store: Store, ids: list[str]) -> list[tuple[dict[str, Any], float]]:
    out: list[tuple[dict[str, Any], float]] = []
    for cid in ids:
        assignment = store.get(cid).body["assignment"]
        pred = store.query("Prediction", {"refs.candidate": cid}, order="created_at", limit=1)
        out.append((assignment, float(pred[0].body["total"])))
    return out


def replay_bucket(
    ctx: Context,
    store: Store,
    fp: FamilyPackage,
    ids: CaseIds,
    row: dict[str, Any],
    hw: HwModel,
    model_id: str,
) -> list[str]:
    model = load_model(store, model_id)
    report = store.get(row["solve_report"]).body["data"]
    f = assemble(fp, fp.chain, templates_of(store, fp), ids.body, hw.limits)
    bucket = {"name": row["bucket"], "shape": row["shape"], "weight": 1.0}
    r = solve(
        f, ids.body, bucket, model, hw, shortlist=len(report["candidate_ids"]), rng=random.Random(0)
    )
    out: list[str] = []
    recorded = _candidates(store, report["candidate_ids"])
    replayed = [(c.assignment, c.prediction.total) for c in r.candidates]
    if [a for a, _ in recorded] != [a for a, _ in replayed]:
        out.append(f"bucket {row['bucket']}: shortlist differs from the recorded Candidates")
    out += [
        f"bucket {row['bucket']}: prediction of {a} recorded {t0}, replayed {t1}"
        for (a, t0), (_, t1) in zip(recorded, replayed, strict=False)
        if not _close(t0, t1)
    ]
    for x in row["top"]:
        build = Build.model_validate(store.get(x["build"]).body)
        cand = Candidate(
            x["assignment"],
            row["bucket"],
            Prediction.model_validate(
                store.query(
                    "Prediction", {"refs.candidate": x["candidate"]}, order="created_at", limit=1
                )[0].body
            ),
        )
        again = rerank([cand], {cand.key: build}, model, hw, k=1)
        if not again or not _close(again[0].prediction.total, x["predicted_us"]):
            out.append(f"bucket {row['bucket']}: re-ranked prediction of {x['assignment']} differs")
        if "measurement" in x:
            out += replay_measurement(store, x["measurement"], x["build"])
    if row.get("baseline"):
        bm = row["baseline"]["measurement"]
        out += replay_measurement(store, bm, str(store.get(bm).refs["baseline"]))
    measured = [x for x in row["top"] if "median_us" in x]
    rho = spearman([x["predicted_us"] for x in measured], [x["median_us"] for x in measured])
    if not _close(rho, row["spearman"]):
        out.append(f"bucket {row['bucket']}: spearman recorded {row['spearman']}, replayed {rho}")
    return out


def run_replay(ctx: Context, store: Store, ids: CaseIds, topk_id: str | None = None) -> str:
    if topk_id is None:
        found = store.query("Report", {"subject": "wp7_topk"}, order="-created_at", limit=1)
        if not found:
            raise LookupError("no wp7_topk Report to replay")
        topk_id = record_id(found[0])
    data = store.get(topk_id).body["data"]
    hw = HwModel.model_validate(store.get(data["hw_model"]).body)
    diffs: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix="kta-wp7-replay-") as tmp:
        fp = materialize(store, data["family"], Path(tmp))
        for row in data["buckets"]:
            diffs[row["bucket"]] = replay_bucket(ctx, store, fp, ids, row, hw, data["model"])
    consistent = not any(diffs.values())
    summary = f"replay of {topk_id[:12]}: " + (
        "consistent" if consistent else f"{sum(map(len, diffs.values()))} differences"
    )
    body = Report(
        title="wp7 replay",
        subject="replay",
        summary=summary,
        data={"replayed": topk_id, "consistent": consistent, "differences": diffs},
    )
    refs: dict[str, str | list[str]] = {"replayed": topk_id, "case": ids.case}
    return store.put(Record.new("Report", body, ACTOR, ctx.clock.now(), refs=refs))
