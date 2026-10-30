"""WP13 (b): prediction error of the GB10 latency models written in F2 v7.

Each F2 stage freezes an author-written analytic latency model (formulation +
the shared GB10 hardware.json) and measures the model-selected point, the
family default and up to four pre-registered probes on the GB10 worker. This
script pairs, per stage, the frozen model's prediction for every measured
assignment (selection-freeze.json: cpu_formula_audit) with the pooled capture
median (mechanical-summary.json), and reports absolute and ranking accuracy.

Usage:
    python3 scripts/analysis/f2_model_error.py [--f2-root DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import f2_common as c

ROLE_SHORT = {
    "frozen_model_recommendation": "model",
    "family_default": "default",
    "preregistered_diagnostic_probe": "probe",
}
WITHIN = (0.05, 0.10, 0.25)
# Measured pairs closer than this are treated as unresolved: about 3x the median
# A/B capture deviation of the batch (see capture_ab_max_dev in the output).
RESOLVED_GAP = 0.03


def _key(assignment: dict) -> str:
    return json.dumps(assignment, sort_keys=True)


def stage_pairs(fb: Path, seen: set[str]) -> tuple[list[dict], dict]:
    """Measured/predicted pairs of one stage, deduplicated by assignment."""
    sf_path = fb / "selection-freeze.json"
    ms_path = fb / "mechanical-summary.json"
    sf = c.load_json(sf_path)
    ms = c.load_json(ms_path)
    pred = {_key(x["assignment"]): x["prediction_us"] for x in sf.get("cpu_formula_audit") or []}
    info = {
        "feedback": c.rel(fb),
        "cpu_audit_complete": bool(sf.get("cpu_audit_complete")),
        "configs": len(sf["measurement_configs"]),
        "not_qualified": [],
        "no_prediction": [],
        "selection_freeze_prediction_us": sf.get("prediction_us"),
        "formulation_size": {
            k: sf.get(k)
            for k in (
                "domain_combinations",
                "cpu_legal",
                "factor_rows",
                "columns",
                "rows",
                "seconds",
            )
        },
    }
    pairs: dict[str, dict] = {}
    for mc in sf["measurement_configs"]:
        label = mc["label"]
        k = _key(mc["assignment"])
        lat = ms["latency"].get(label)
        if lat is None:
            info["not_qualified"].append(label)
            continue
        if k not in pred:
            info["no_prediction"].append(label)
            continue
        if k in pairs:
            pairs[k]["labels"].append(label)
            continue
        measured = lat["pooled_capture_median_us"]
        ab = [abs(r - 1.0) for r in lat.get("aa_b_over_a") or []]
        pairs[k] = {
            "labels": [label],
            "role": ROLE_SHORT.get(mc["role"], mc["role"]),
            "assignment": mc["assignment"],
            "predicted_us": pred[k],
            "measured_us": measured,
            "rel_error": (pred[k] - measured) / measured,
            "log_ratio": math.log(pred[k] / measured),
            "capture_ab_max_dev": max(ab) if ab else None,
            "seen_in_earlier_stage": k in seen,
        }
    for pair in pairs.values():
        if "model" in pair["labels"]:
            pair["role"] = "model"
    info["sources"] = [c.rel(sf_path), c.rel(ms_path)]
    return list(pairs.values()), info


def ranking(pairs: list[dict]) -> dict:
    if len(pairs) < 2:
        return {"n_points": len(pairs)}
    pred = [p["predicted_us"] for p in pairs]
    meas = [p["measured_us"] for p in pairs]
    best_meas = min(meas)
    best_pred_idx = pred.index(min(pred))
    model = [p for p in pairs if p["role"] == "model"]
    resolved = agree = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            gap = abs(meas[i] - meas[j]) / min(meas[i], meas[j])
            if gap <= RESOLVED_GAP:
                continue
            resolved += 1
            if (pred[i] - pred[j]) * (meas[i] - meas[j]) > 0:
                agree += 1
    out = {
        "n_points": len(pairs),
        "resolved_pairs": resolved,
        "resolved_pairs_order_agree": agree,
        "kendall_tau_b": c.kendall_tau_b(pred, meas),
        "top1_hit": meas[best_pred_idx] == best_meas,
        "regret_of_predicted_best": meas[best_pred_idx] / best_meas - 1.0,
    }
    if model:
        out["model_point_regret"] = model[0]["measured_us"] / best_meas - 1.0
    return out


def error_summary(pairs: list[dict]) -> dict:
    if not pairs:
        return {"n": 0}
    rel = [p["rel_error"] for p in pairs]
    absr = [abs(x) for x in rel]
    logs = [p["log_ratio"] for p in pairs]
    out = {
        "n": len(pairs),
        "signed_rel_error": c.describe(rel),
        "abs_rel_error": c.describe(absr),
        "geo_mean_pred_over_meas": math.exp(statistics.fmean(logs)),
        "frac_overpredicted": sum(1 for x in rel if x > 0) / len(rel),
    }
    for w in WITHIN:
        out[f"frac_within_{int(w * 100)}pct"] = sum(1 for x in absr if x <= w) / len(absr)
    return out


def ranking_summary(stages: list[dict]) -> dict:
    ranked = [s["ranking"] for s in stages if s["ranking"].get("n_points", 0) >= 3]
    taus = [r["kendall_tau_b"] for r in ranked if r.get("kendall_tau_b") is not None]
    regrets = [r["model_point_regret"] for r in ranked if "model_point_regret" in r]
    all_ranked = [s["ranking"] for s in stages]
    resolved = sum(r.get("resolved_pairs", 0) for r in all_ranked)
    agree = sum(r.get("resolved_pairs_order_agree", 0) for r in all_ranked)
    return {
        "resolved_gap": RESOLVED_GAP,
        "resolved_pairs": resolved,
        "resolved_pair_order_accuracy": agree / resolved if resolved else None,
        "stages_with_3plus_points": len(ranked),
        "kendall_tau_b": c.describe(taus),
        "top1_hit_rate": (sum(1 for r in ranked if r["top1_hit"]) / len(ranked))
        if ranked
        else None,
        "model_point_regret": c.describe(regrets),
        "frac_model_point_is_measured_best": (
            sum(1 for x in regrets if x == 0.0) / len(regrets) if regrets else None
        ),
    }


def collect(f2_root: Path) -> list[dict]:
    stages = []
    for task, arm, rep, path, ledger in c.iter_trajectories(f2_root):
        seen: set[str] = set()
        v7_index = 0
        for stage in ledger["stages"]:
            evaluation = stage.get("evaluation")
            if not evaluation:
                continue
            fb = c.feedback_dir_for(evaluation)
            sf_path = fb / "selection-freeze.json"
            if not sf_path.exists():
                continue
            if not (fb / "mechanical-summary.json").exists():
                if c.is_v7_stage(stage["name"]):
                    raise FileNotFoundError(fb / "mechanical-summary.json")
                # Earlier-protocol checkpoint without a latency summary: its
                # measured configs still count as feedback the author has seen.
                sf = c.load_json(sf_path)
                seen.update(_key(m["assignment"]) for m in sf["measurement_configs"])
                continue
            pairs, info = stage_pairs(fb, seen)
            seen.update(_key(p["assignment"]) for p in pairs)
            if not c.is_v7_stage(stage["name"]):
                continue
            v7_index += 1
            stages.append(
                {
                    "trajectory": f"{task}/{arm}-rep{rep}",
                    "task": task,
                    "arm": arm,
                    "rep": rep,
                    "stage": stage["name"],
                    "v7_round": v7_index,
                    "fresh_trajectory": task != "attn-small",
                    "ledger": c.rel(path),
                    "info": info,
                    "pairs": pairs,
                    "ranking": ranking(pairs),
                }
            )
    return stages


def formulation_sizes(stages: list[dict]) -> dict:
    out = {}
    for task in c.TASKS:
        for arm in c.ARMS:
            group = [
                s["info"]["formulation_size"]
                for s in stages
                if s["task"] == task and s["arm"] == arm
            ]
            out[f"{task}/{arm}"] = {
                k: statistics.median([g[k] for g in group if g[k] is not None])
                for k in ("domain_combinations", "cpu_legal", "factor_rows", "seconds")
            }
    return out


def summarise(stages: list[dict], select) -> dict:
    chosen = [s for s in stages if select(s)]
    pairs = [p for s in chosen for p in s["pairs"]]
    unseen = [p for p in pairs if not p["seen_in_earlier_stage"]]
    by_role = {
        r: error_summary([p for p in pairs if p["role"] == r])
        for r in ("model", "default", "probe")
    }
    noise = [p["capture_ab_max_dev"] for p in pairs if p["capture_ab_max_dev"] is not None]
    return {
        "stages": len(chosen),
        "all_pairs": error_summary(pairs),
        "unseen_pairs": error_summary(unseen),
        "by_role": by_role,
        "ranking": ranking_summary(chosen),
        "capture_ab_max_dev": c.describe(noise),
        "not_qualified_configs": sum(len(s["info"]["not_qualified"]) for s in chosen),
        "not_qualified_model_points": sum(
            1 for s in chosen if "model" in s["info"]["not_qualified"]
        ),
        "configs": sum(s["info"]["configs"] for s in chosen),
        "configs_without_prediction": sum(len(s["info"]["no_prediction"]) for s in chosen),
    }


def final_points(f2_root: Path) -> dict:
    export_path, records = c.export_records(f2_root)
    rows = []
    for (task, run_id), r in sorted(records.items()):
        pred, meas = r["model_prediction_us"], r["final_region_us"]
        rows.append(
            {
                "task": task,
                "run_id": run_id,
                "predicted_us": pred,
                "measured_us": meas,
                "rel_error": (pred - meas) / meas,
                "log_ratio": math.log(pred / meas),
            }
        )
    out = {"source": c.rel(export_path), "rows": rows, "all": error_summary(rows)}
    for task in c.TASKS:
        out[task] = error_summary([x for x in rows if x["task"] == task])
    return out


HARDWARE = "formal-v5/common-assets/hardware.json"
BF16_BYTES = 2


def working_sets(f2_root: Path) -> dict:
    """K+V bytes of each task against the GB10 L2 figures the authors were given.

    Shapes come from the trajectory ledgers; the hardware file is the one passed
    to every v7 author launch (launch-*.receipt.json --hardware).
    """
    hw_path = f2_root / HARDWARE
    hw = c.load_json(hw_path)
    out = {
        "hardware": c.rel(hw_path),
        "hw_l2_bytes": hw["hw_l2_bytes"],
        "hw_l2_effective_bytes": hw["hw_l2_effective_bytes"],
        "tasks": {},
    }
    for task in c.TASKS:
        path = c.trajectory_path(f2_root, task, "template", 0)
        t = c.load_json(path)["task"]
        kv = 2 * t["N"] * t["HK"] * t["D"] * BF16_BYTES
        out["tasks"][task] = {
            "source": c.rel(path),
            "shape": {k: t[k] for k in ("H", "HK", "D", "N")},
            "kv_bytes": kv,
            "kv_over_l2": kv / hw["hw_l2_bytes"],
            "kv_over_l2_effective": kv / hw["hw_l2_effective_bytes"],
        }
    return out


def build(f2_root: Path) -> dict:
    stages = collect(f2_root)
    groups = {
        "all": lambda s: True,
        "fresh_round01": lambda s: s["fresh_trajectory"] and s["v7_round"] == 1,
        "fresh_round02plus": lambda s: s["fresh_trajectory"] and s["v7_round"] > 1,
        "small_continuations": lambda s: not s["fresh_trajectory"],
    }
    for task in c.TASKS:
        groups[task] = lambda s, t=task: s["task"] == t
    for arm in c.ARMS:
        groups[f"arm:{arm}"] = lambda s, a=arm: s["arm"] == a
    return {
        "schema": "kta-wp13-f2-model-error-1",
        "inputs": {
            "f2_root": c.rel(f2_root),
            "per_stage": "formal-v7/feedback-<stage>/selection-freeze.json "
            "(cpu_formula_audit, measurement_configs) + mechanical-summary.json",
            "final": c.rel(f2_root / c.EXPORT_DIR / "results.json"),
        },
        "notes": NOTES,
        "summaries": {name: summarise(stages, sel) for name, sel in groups.items()},
        "final_recommendations": final_points(f2_root),
        "formulation_size_medians": formulation_sizes(stages),
        "working_sets": working_sets(f2_root),
        "stages": stages,
    }


NOTES = [
    (
        "rel_error = (predicted - measured) / measured; measured is the pooled "
        "capture median of the stage's GPU batch on GB10."
    ),
    (
        "Pairs are deduplicated by assignment within a stage. seen_in_earlier_stage "
        "marks assignments measured in an earlier stage of the same trajectory "
        "(including v4/v6 stages for attn-small); authors saw that feedback, so those "
        "pairs are in-sample. unseen_pairs is the out-of-sample view."
    ),
    (
        "Probes are chosen by the author, not sampled at random; ranking metrics are "
        "over the measured set only (model point, default, probes)."
    ),
    "Configs that failed qualification have no latency and are counted, not paired.",
]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f2-root", type=Path, default=c.F2_ROOT)
    ap.add_argument("--out", type=Path, default=Path("deliveries/WP13/f2-model-error.json"))
    args = ap.parse_args(argv)
    result = build(args.f2_root)
    c.write_json(args.out, result)
    sys.stdout.write(f"wrote {args.out} ({len(result['stages'])} stages)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
