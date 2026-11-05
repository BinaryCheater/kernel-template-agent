"""Read retained replay evidence for plots without modifying it."""

import json
import math
import statistics


def read(path):
    return json.loads(path.read_text())


def collect(root):
    campaign = read(root / "campaign.json")
    usage = {r["slot"]: r for r in read(root / "final-findings.json")["slots"]}
    arm_order = ("prescribed", "self-selection", "no-template")
    colors = ("#376da8", "#ce7c23", "#208778")
    names = ("Given Templates", "Choose Templates", "No Templates")
    runs, iterations, submissions, samples, windows, inputs = [], [], [], [], [], set()
    for meta in sorted(campaign["schedule"], key=lambda r: (arm_order.index(r["arm"]), r["block"])):
        slot = root / "slots" / meta["slot"]
        extra = root / "frozen-validation-v1" / meta["slot"] / "result.json"
        final_path = extra if extra.exists() else slot / "confirmation.json"
        final = read(final_path)
        inputs.add(final_path)
        arm = arm_order.index(meta["arm"])
        run = {
            **meta,
            "label": f"{names[arm]} - {meta['block']}",
            "color": colors[arm],
            "final": final,
            "usage": usage[meta["slot"]],
            "shapes": [],
        }
        for index, shape in enumerate(campaign["task"]["shapes"]):
            timings = [w["rows"][index]["timing"] for w in final["windows"]]
            original = [t["original"]["median_us"] for t in timings]
            candidate = [t["candidate"]["median_us"] for t in timings]
            run["shapes"].append(
                {
                    "S": shape[1],
                    "original": original,
                    "candidate": candidate,
                    "original_samples": [],
                    "candidate_samples": [],
                }
            )
            for window, (w, t) in enumerate(zip(final["windows"], timings, strict=True), 1):
                assert math.isclose(
                    t["ratio"],
                    t["original"]["median_us"] / t["candidate"]["median_us"],
                    rel_tol=1e-8,
                )
                windows.append(
                    {
                        "slot": meta["slot"],
                        "arm": meta["arm"],
                        "S": shape[1],
                        "window": window,
                        "valid": w["window_valid"],
                        "original_us": t["original"]["median_us"],
                        "generated_us": t["candidate"]["median_us"],
                        "source": str(final_path),
                    }
                )
                for key in ("original", "candidate"):
                    vals = t[key]["samples_us"]
                    assert len(vals) == 30
                    assert math.isclose(statistics.median(vals), t[key]["median_us"], rel_tol=1e-8)
                    run["shapes"][-1][f"{key}_samples"].extend(vals)
                for block, pair in enumerate(t["paired_blocks"], 1):
                    samples.append(
                        {
                            "slot": meta["slot"],
                            "S": shape[1],
                            "window": window,
                            "block": block,
                            "original_us": pair["original"],
                            "generated_us": pair["candidate"],
                            "first": pair["first"],
                            "valid_window": w["window_valid"],
                            "source": str(final_path),
                        }
                    )
        for sub in sorted((slot / "submissions").glob("*")):
            info = read(sub / "submission.json")
            path = sub / "summary.json"
            inputs.add(sub / "submission.json")
            result = read(path) if path.exists() else {}
            if path.exists():
                inputs.add(path)
            active = result.get("active_s_at_finish")
            rows = result.get("rows", [])
            usable = len(rows) == 3 and all(r.get("correct") and r.get("timing") for r in rows)
            submissions.append(
                {
                    "slot": meta["slot"],
                    "submission": info["submission"],
                    "active_minutes": active / 60 if active else "",
                    "usable_timing": usable,
                    "valid": result.get("window_valid", False),
                    "reason": result.get("error")
                    or result.get("reason")
                    or ("unfinished submission" if not path.exists() else ""),
                    "source": str(path if path.exists() else sub / "submission.json"),
                }
            )
            if usable:
                for row in rows:
                    timing = row["timing"]
                    iterations.append(
                        {
                            "slot": meta["slot"],
                            "submission": info["submission"],
                            "S": row["shape"][1],
                            "active_minutes": active / 60 if active is not None else None,
                            "valid": bool(result.get("window_valid")),
                            "change_percent": 100
                            * (
                                timing["candidate"]["median_us"] / timing["original"]["median_us"]
                                - 1
                            ),
                            "original_us": timing["original"]["median_us"],
                            "generated_us": timing["candidate"]["median_us"],
                            "source": str(path),
                        }
                    )
        runs.append(run)
    inputs.update(
        root / p for p in ("campaign.json", "final-findings.json", "final-accounting.json")
    )
    return runs, iterations, submissions, windows, samples, inputs
