"""Render the WP13 delivery tables from the two analysis outputs.

Usage:
    python3 scripts/analysis/f2_report.py [--dir deliveries/WP13]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import f2_common as c

F2 = str(c.F2_ROOT)
# Citation tags used in deliveries/WP13.md; the gate requires every line with a
# measured number to carry one, and each tag to name an existing path.
SOURCES = {
    "S1": f"{F2}/formal-v7",
    "S2": f"{F2}/formal-v7/export-attention-20261027-01/results.json",
    "S3": f"{F2}/formal-v5/common-assets/hardware.json",
}
TIME_SRC = "[S1] [S2]"
STAGE_SRC = "[S1]"


def source_lines() -> list[str]:
    return [f"- [{tag}] {path}" for tag, path in SOURCES.items()]


def _f(x, nd=0) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def time_tables(tb: dict) -> list[str]:
    out = ["### A1. Share of end-to-end wall, fresh trajectories (medium + long)", ""]
    out.append(
        "| arm | n | end-to-end s | author ok | author timeout/failed "
        "| solve | GPU worker active | GPU transport+queue | controller wait | src |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm, v in tb["pooled_fresh_shares"].items():
        s = v["shares"]
        out.append(
            f"| {arm} | {v['n_trajectories']} | {_f(v['end_to_end_wall_s'])} "
            f"| {_pct(s['author_productive'])} | {_pct(s['author_timeout_or_failed'])} "
            f"| {_pct(s['solve_stage'])} | {_pct(s['gpu_worker_active'])} "
            f"| {_pct(s['gpu_transport_residual'])} | {_pct(s['unaccounted_wait'])} "
            f"| {TIME_SRC} |"
        )
    pc = tb["provider_connection"]
    out += [
        "",
        (
            f"Provider connection: {pc['sessions']} author sessions, "
            f"{pc['sessions_with_startup_chain']} with a reconnect chain starting "
            f"right after launch, {pc['sessions_with_chain_ge_60s']} losing >= 60 s; "
            f"retry span {_f(pc['retry_span_s_lower_bound'])} s (lower bound, "
            f"{_pct(pc['retry_share_of_author_wall'])} of author wall), start-to-"
            f"fallback {_f(pc['startup_stall_s_upper_bound'])} s (upper bound, "
            f"{_pct(pc['startup_stall_share_of_author_wall'])}); author wall "
            f"{_f(pc['author_wall_s'])} s. {STAGE_SRC}"
        ),
    ]
    out += ["", "### A2. Per trajectory, median of 3 replicates (v7 stages only)", ""]
    cols = [
        ("author_wall_s", "author s", 0),
        ("author_rounds", "rounds", 0),
        ("author_timeouts", "timeouts", 0),
        ("provider_retry_span_s", "retry s", 0),
        ("provider_startup_stall_s", "startup stall s", 0),
        ("solve_stage_wall_s", "solve s", 0),
        ("worker_compile_first_call_s", "compile s", 1),
        ("worker_measurement_s", "measure s", 1),
        ("gpu_wall_including_queue_s", "GPU wall s", 0),
        ("unaccounted_wait_s", "wait s", 0),
        ("input_tokens", "input tok", 0),
        ("output_tokens", "output tok", 0),
        ("cmd_read", "reads", 0),
        ("cmd_write", "writes", 0),
        ("cmd_check", "checks", 0),
        ("read_bytes", "read B", 0),
        ("read_bytes_knowledge", "knowledge B", 0),
    ]
    out.append("| task/arm | " + " | ".join(h for _, h, _ in cols) + " | src |")
    out.append("|---|" + "---|" * (len(cols) + 1))
    for key, v in tb["by_task_arm"].items():
        m = v["median"]
        cells = " | ".join(_f(m[f], nd) for f, _, nd in cols)
        out.append(f"| {key} | {cells} | {TIME_SRC} |")
    out += ["", "### A3. Template minus Text (difference of medians)", ""]
    diff_cols = [x for x in cols if x[0] != "author_rounds"]
    out.append("| task | " + " | ".join(h for _, h, _ in diff_cols) + " | src |")
    out.append("|---|" + "---|" * (len(diff_cols) + 1))
    for task, d in tb["template_minus_text"].items():
        cells = " | ".join(_f(d[f], nd) for f, _, nd in diff_cols)
        out.append(f"| {task} | {cells} | {TIME_SRC} |")
    out += ["", "### A4. Per author stage, median (successful provider exits)", ""]
    pcols = [
        ("author_wall_s", "author s"),
        ("provider_retry_span_s", "retry s"),
        ("provider_startup_stall_s", "startup stall s"),
        ("input_tokens", "input tok"),
        ("output_tokens", "output tok"),
        ("read_bytes_knowledge", "knowledge B"),
        ("cmd_read", "reads"),
        ("reads_before_first_write", "reads before 1st write"),
    ]
    out.append("| task/arm | n | " + " | ".join(h for _, h in pcols) + " | src |")
    out.append("|---|---|" + "---|" * (len(pcols) + 1))
    for key, v in tb["per_author_stage_medians"].items():
        cells = " | ".join(_f(v[f], 0) for f, _ in pcols)
        out.append(f"| {key} | {v['n_stages']} | {cells} | {STAGE_SRC} |")
    return out


def error_tables(me: dict) -> list[str]:
    out = ["### B1. Per-stage prediction error (predicted vs measured, GB10)", ""]
    out.append(
        "| group | stages | pairs | median abs err | p90 abs err "
        "| signed median | within 10 pct | unseen pairs | unseen median abs "
        "| resolved-pair order acc. | model-point regret median / p90 "
        "| model point = measured best | unqualified configs | src |"
    )
    out.append("|---|" + "---|" * 13)
    for name, v in me["summaries"].items():
        a, u, r = v["all_pairs"], v["unseen_pairs"], v["ranking"]
        reg = r["model_point_regret"]
        out.append(
            f"| {name} | {v['stages']} | {a['n']} "
            f"| {_pct(a['abs_rel_error']['median'])} "
            f"| {_pct(a['abs_rel_error']['p90'])} "
            f"| {_pct(a['signed_rel_error']['median'])} "
            f"| {_pct(a['frac_within_10pct'])} "
            f"| {u['n']} | {_pct(u.get('abs_rel_error', {}).get('median'))} "
            f"| {_pct(r['resolved_pair_order_accuracy'])} ({r['resolved_pairs']}) "
            f"| {_pct(reg.get('median'))} / {_pct(reg.get('p90'))} "
            f"| {_pct(r['frac_model_point_is_measured_best'])} "
            f"| {v['not_qualified_configs']}/{v['configs']} | {STAGE_SRC} |"
        )
    out += ["", "### B2. Error by measured role (all v7 stages)", ""]
    out.append("| role | pairs | median abs err | p90 abs err | signed median | src |")
    out.append("|---|---|---|---|---|---|")
    for role, v in me["summaries"]["all"]["by_role"].items():
        out.append(
            f"| {role} | {v['n']} | {_pct(v['abs_rel_error']['median'])} "
            f"| {_pct(v['abs_rel_error']['p90'])} "
            f"| {_pct(v['signed_rel_error']['median'])} | {STAGE_SRC} |"
        )
    fr = me["final_recommendations"]
    out += ["", "### B3. Final recommendations in the common final batch", ""]
    out.append("| task | n | median abs err | max abs err | signed median | src |")
    out.append("|---|---|---|---|---|---|")
    for task in ("all", *c.TASKS):
        v = fr[task]
        out.append(
            f"| {task} | {v['n']} | {_pct(v['abs_rel_error']['median'])} "
            f"| {_pct(v['abs_rel_error']['max'])} "
            f"| {_pct(v['signed_rel_error']['median'])} | [S2] |"
        )
    out += ["", "### B4. Frozen formulation size per stage, median", ""]
    out.append("| task/arm | domain assignments | CPU-legal | factor rows | solver s | src |")
    out.append("|---|---|---|---|---|---|")
    for key, v in me["formulation_size_medians"].items():
        out.append(
            f"| {key} | {_f(v['domain_combinations'])} | {_f(v['cpu_legal'])} "
            f"| {_f(v['factor_rows'])} | {_f(v['seconds'], 2)} | {STAGE_SRC} |"
        )
    ws = me["working_sets"]
    out += [
        "",
        "### B5. K+V working set vs GB10 L2 (hardware.json given to authors)",
        "",
    ]
    out.append("| task | H/HK/D/N | K+V bytes | / hw_l2_bytes | / hw_l2_effective | src |")
    out.append("|---|---|---|---|---|---|")
    for task, v in ws["tasks"].items():
        sh = v["shape"]
        out.append(
            f"| {task} | {sh['H']}/{sh['HK']}/{sh['D']}/{sh['N']} "
            f"| {_f(v['kv_bytes'])} | {v['kv_over_l2']:.4f} "
            f"| {v['kv_over_l2_effective']:.2f} | [S1] [S3] |"
        )
    noise = me["summaries"]["all"]["capture_ab_max_dev"]
    out += [
        "",
        (
            f"Capture A/B deviation (noise reference), all pairs: median "
            f"{_pct(noise['median'])}, p90 {_pct(noise['p90'])}. {STAGE_SRC}"
        ),
    ]
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("deliveries/WP13"))
    args = ap.parse_args(argv)
    tb = c.load_json(args.dir / "f2-time-breakdown.json")
    me = c.load_json(args.dir / "f2-model-error.json")
    lines = [*source_lines(), "", *time_tables(tb), "", *error_tables(me)]
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
