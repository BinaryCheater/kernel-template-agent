# /// script
# requires-python = "==3.12.*"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Plot recorded agent tuning results; never execute or modify a kernel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".uv/mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from source_replay_plot_data import collect


def decorate(fig, title, subtitle, footer, top=0.85, bottom=0.12):
    fig.suptitle(title, x=0.08, y=0.98, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.08,
        0.932 if fig.get_figheight() >= 8 else 0.90,
        subtitle,
        ha="left",
        fontsize=10,
        color="#4c5664",
    )
    fig.text(0.08, 0.025, footer, va="bottom", fontsize=9, color="#4c5664", linespacing=1.5)
    fig.subplots_adjust(left=0.19, right=0.97, top=top, bottom=bottom, wspace=0.15, hspace=0.35)


def clean(ax, axis="x"):
    ax.grid(axis=axis, color="#e6e9ed", linewidth=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b8c0ca")
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_axisbelow(True)


def save(fig, out, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=190, facecolor="white")
    plt.close(fig)


def final_times(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    for index, ax in enumerate(axes):
        maximum = 0
        for y, run in enumerate(runs):
            shape = run["shapes"][index]
            stable = run["final"]["confirmed"]
            if not stable:
                ax.axhspan(y - 0.44, y + 0.44, color="#fff4df", zorder=0)
            for key, offset, color, marker in (
                ("original", -0.14, "#77818c", "s"),
                ("candidate", 0.14, run["color"], "o"),
            ):
                values = shape[key]
                center = statistics.median(values)
                ax.plot([min(values), max(values)], [y + offset] * 2, color=color, lw=1.6)
                ax.scatter(values, [y + offset] * len(values), color=color, s=11, alpha=0.55)
                ax.scatter(
                    [center],
                    [y + offset],
                    s=44,
                    marker=marker,
                    edgecolor=color,
                    facecolor=color if stable or key == "original" else "white",
                    zorder=4,
                )
                ax.annotate(
                    f"{center:.2f}",
                    (center, y + offset),
                    xytext=(6, -3),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=color,
                )
                maximum = max(maximum, max(values))
        ax.set_xlim(0, maximum * 1.24)
        ax.set_title(f"Sequence length {runs[0]['shapes'][index]['S']}", loc="left", fontsize=12)
        ax.set_xlabel("GPU time (microseconds); lower is faster")
        clean(ax)
    axes[0].set_yticks(range(9), [r["label"] for r in runs])
    axes[0].set_ylim(8.65, -0.65)
    for ax in axes:
        for boundary in (2.5, 5.5):
            ax.axhline(boundary, color="#c5cbd3", lw=0.8)
    handles = [
        Line2D([], [], color="#77818c", marker="s", ls="", label="Original FlashAttention"),
        Line2D([], [], color="#202b3c", marker="o", ls="", label="Generated implementation"),
        Line2D(
            [],
            [],
            color="#202b3c",
            marker="o",
            mfc="white",
            ls="",
            label="Open marker / shaded row: timing unresolved",
        ),
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.185, 0.905), ncol=3, frameon=False
    )
    decorate(
        fig,
        "Final GPU time: compare each generated kernel with its own measured original",
        "Nine model sessions; the same attention computation and outputs. "
        "The original is remeasured in every window.",
        "Point = median of three window medians; whisker = their minimum to "
        "maximum (not a confidence interval).\n"
        "Unresolved rows retain observations but do not establish a "
        "confirmed performance result. No MIP or performance-prediction "
        "model was used.",
    )
    save(fig, out, "01_final_performance")


def variation(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    for index, ax in enumerate(axes):
        for y, run in enumerate(runs):
            shape = run["shapes"][index]
            if not run["final"]["confirmed"]:
                ax.axhspan(y - 0.44, y + 0.44, color="#fff4df", zorder=0)
            for key, offset, color in (
                ("original", -0.17, "#8d969f"),
                ("candidate", 0.17, run["color"]),
            ):
                data = shape[f"{key}_samples"]
                plot = ax.boxplot(
                    [data],
                    positions=[y + offset],
                    widths=0.26,
                    orientation="horizontal",
                    whis=(10, 90),
                    showfliers=False,
                    patch_artist=True,
                    manage_ticks=False,
                    medianprops={"color": "#1d2837", "linewidth": 1.2},
                    whiskerprops={"color": color},
                    capprops={"color": color},
                )
                plot["boxes"][0].set(facecolor=color, alpha=0.6, edgecolor=color)
                ax.scatter([min(data), max(data)], [y + offset] * 2, s=5, color=color, alpha=0.5)
        ax.set_title(f"Sequence length {runs[0]['shapes'][index]['S']}", loc="left", fontsize=12)
        ax.set_xlabel("GPU time per call (microseconds)")
        clean(ax)
    axes[0].set_yticks(range(9), [r["label"] for r in runs])
    axes[0].set_ylim(8.65, -0.65)
    decorate(
        fig,
        "Timing variation: the original kernel also moves between measurements",
        "Gray = original; colored = generated. Each box pools 90 "
        "paired-block samples across three final windows.",
        "Box = middle 50%; center line = pooled sample median; whiskers = "
        "10th to 90th percentiles; small dots = minimum / maximum.\n"
        "Each sample averages 100 graph replays. These show observed "
        "spread, not a confidence interval; shaded rows remain unresolved.",
    )
    save(fig, out, "02_timing_variation")


def iteration_progress(runs, iterations, submissions, out):
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True, sharey=True)
    colors = ("#376da8", "#ce7c23", "#208778")
    for run, ax in zip(runs, axes.flat, strict=True):
        own = [r for r in iterations if r["slot"] == run["slot"]]
        all_subs = [s for s in submissions if s["slot"] == run["slot"]]
        for shape, color in zip((512, 2048, 513), colors, strict=True):
            data = [r for r in own if r["S"] == shape]
            lookup = {r["submission"]: r for r in data}
            xx = list(range(1, max(s["submission"] for s in all_subs) + 1))
            yy = [
                lookup[x]["change_percent"] if x in lookup and lookup[x]["valid"] else float("nan")
                for x in xx
            ]
            ax.plot(xx, yy, color=color, lw=1.4, marker="o", markersize=3)
            invalid = [r for r in data if not r["valid"]]
            ax.scatter(
                [r["submission"] for r in invalid],
                [r["change_percent"] for r in invalid],
                facecolor="white",
                edgecolor=color,
                s=24,
                alpha=0.65,
                zorder=4,
            )
        failed = [s["submission"] for s in all_subs if not s["usable_timing"]]
        ax.scatter(
            failed,
            [-0.065] * len(failed),
            transform=ax.get_xaxis_transform(),
            color="#b34c49",
            marker="x",
            s=22,
            clip_on=False,
        )
        ax.axhline(0, color="#56616e", lw=0.8)
        ax.axhline(100 / 9, color="#8c949d", ls="--", lw=0.8)
        ax.set_title(f"{run['label']}  ({run['slot']})", loc="left", fontsize=11)
        ax.set_ylim(-35, 105)
        ax.set_xticks((1, 4, 8, 12, 16))
        ax.set_xlim(0.5, 16.7)
        clean(ax, "y")
    for ax in axes[-1]:
        ax.set_xlabel("Target submission number", labelpad=17)
    for ax in axes[:, 0]:
        ax.set_ylabel("GPU time change (%)")
    handles = [
        Line2D([], [], color=c, marker="o", label=f"S={s}")
        for s, c in zip((512, 2048, 513), colors, strict=True)
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.905), ncol=3, frameon=False
    )
    decorate(
        fig,
        "Iteration history: real progress, regressions, and failed attempts",
        "Negative means less GPU time than the original. These are "
        "development observations, not final confirmation.",
        "Filled points / lines = valid timing; open points = unresolved "
        "timing; red x below a panel = no usable timing (including compile "
        "failures).\n"
        "Dashed line = +11.1% target limit. Missing / invalid points break "
        "the line. All submitted versions are retained; no smoothing.",
        top=0.84,
        bottom=0.14,
    )
    fig.subplots_adjust(left=0.08, hspace=0.38)
    save(fig, out, "03_iteration_history")


def best_progress(runs, iterations, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharey=True)
    styles = ("-", "--", ":")
    for group, ax in enumerate(axes):
        for run in runs[group * 3 : group * 3 + 3]:
            own = [
                r
                for r in iterations
                if r["slot"] == run["slot"] and r["valid"] and r["active_minutes"] is not None
            ]
            by_submission = {}
            for row in own:
                by_submission.setdefault(row["submission"], []).append(row)
            x, y, best = [], [], float("inf")
            for _, rows in sorted(by_submission.items()):
                if len(rows) != 3:
                    continue
                best = min(best, max(r["change_percent"] for r in rows))
                x.append(rows[0]["active_minutes"])
                y.append(best)
            ax.step(
                x,
                y,
                where="post",
                color=run["color"],
                ls=styles[run["block"] - 1],
                lw=2,
                label=f"Repeat {run['block']}",
            )
            ax.scatter(x, y, color=run["color"], s=12)
        ax.axhline(0, color="#56616e", lw=0.8)
        ax.axhline(100 / 9, color="#8c949d", ls="--", lw=0.8)
        ax.set_title(runs[group * 3]["label"].rsplit(" - ", 1)[0], loc="left", fontsize=12)
        ax.set_xlabel("Active minutes (verified GPU queue wait excluded)")
        ax.set_xlim(0, 31)
        ax.set_ylim(-15, 80)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        clean(ax, "y")
    axes[0].set_ylabel("Best observed slowdown on the slowest shape (%)")
    decorate(
        fig,
        "How much active time did it take to improve the weakest shape?",
        "At each valid submission, take the worst percentage across all "
        "three shapes; retain the best value seen so far.",
        "Lower is better; below +11.1% means a provisional all-shape target "
        "observation. Final independent confirmation is a separate result.\n"
        "Curves exclude invalid timings and failed builds. No-Template "
        "repeat 1 was interrupted by a tool error with budget remaining.",
        top=0.80,
        bottom=0.19,
    )
    fig.subplots_adjust(left=0.08, wspace=0.18)
    save(fig, out, "04_improvement_vs_active_time")


def costs(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    for y, run in enumerate(runs):
        usage = run["usage"]
        active, wait = usage["active_s"] / 60, usage["gpu_queue_s"] / 60
        axes[0].barh(y, active, height=0.58, color=run["color"])
        axes[0].barh(y, wait, left=active, height=0.58, color="#e1e5ea", edgecolor="white")
        axes[0].text(
            active / 2, y, f"{active:.1f}", va="center", ha="center", color="white", fontsize=9
        )
        axes[0].text(active + wait / 2, y, f"{wait:.1f}", va="center", ha="center", fontsize=9)
        cost = usage["usage"]["api_equivalent_usd"]
        axes[1].barh(y, cost, height=0.58, color=run["color"])
        axes[1].text(cost + 0.03, y, f"${cost:.3f}", va="center", fontsize=10)
    axes[0].set_yticks(range(9), [r["label"] for r in runs])
    axes[0].set_ylim(8.65, -0.65)
    axes[0].set_xlim(0, 81)
    axes[1].set_xlim(0, 2.1)
    axes[0].set_xlabel("Accumulated session minutes (colored = active; gray = GPU wait)")
    axes[1].set_xlabel("Estimated model cost for this session (USD)")
    for ax in axes:
        clean(ax)
    decorate(
        fig,
        "Experiment cost: distinguish active work, GPU waiting, and model usage",
        "All nine sessions, including interrupted work. Repeated cached "
        "context is counted in retained request-level usage.",
        "Costs use the retained price card and are not account charges; "
        "preparation / coordinator costs are excluded.\n"
        "Sessions overlapped, so row durations must not be added to infer "
        "calendar duration. No-Template repeat 1 ended early because of a "
        "tool error.",
    )
    fig.subplots_adjust(left=0.22, wspace=0.15)
    save(fig, out, "05_time_and_cost")


def export(out, name, rows):
    with (out / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data.resolve(), args.output.resolve()
    if out == root or out.is_relative_to(root / "slots"):
        raise ValueError("Use a separate derived-figure directory, never a raw slot directory")
    out.mkdir(parents=True, exist_ok=True)
    runs, iterations, submissions, windows, samples, inputs = collect(root)
    assert len(runs) == 9 and len(samples) == 2430
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }
    with plt.rc_context(style):
        final_times(runs, out)
        variation(runs, out)
        iteration_progress(runs, iterations, submissions, out)
        best_progress(runs, iterations, out)
        costs(runs, out)
    for name, data in (
        ("final_windows.csv", windows),
        ("paired_samples.csv", samples),
        ("iterations.csv", iterations),
        ("submissions.csv", submissions),
    ):
        export(out, name, data)
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    scripts = (
        Path(__file__).resolve(),
        Path(__file__).with_name("source_replay_plot_data.py"),
        Path(__file__).with_suffix(".py.lock"),
    )
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "source_files": provenance,
                "script": str(Path(__file__).resolve()),
                "code_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
                "matplotlib_version": matplotlib.__version__,
                "sessions": len(runs),
                "paired_samples": len(samples),
                "no_new_measurements": True,
            },
            indent=2,
        )
    )
    text = "# Recorded agent tuning: figures\n\n"
    text += "One attention operator, nine independent model sessions, three sequence lengths. "
    text += (
        "These are empirical coding/tuning results; no MIP solver or "
        "performance-prediction model was used.\n\n"
    )
    for name in (
        "01_final_performance",
        "02_timing_variation",
        "03_iteration_history",
        "04_improvement_vs_active_time",
        "05_time_and_cost",
    ):
        text += f"## {name.replace('_', ' ')}\n\n![{name}]({name}.png)\n\n"
        text += f"[PDF]({name}.pdf) / [SVG]({name}.svg)\n\n"
    text += "## Data and reproduction\n\n"
    text += "Final plots use the fixed-source follow-up for every eligible unresolved run, "
    text += "and initial confirmation for the other runs, matching final-findings.json. "
    text += "Shaded rows remain unresolved; no invalid measurements are counted as formal passes. "
    text += (
        "Within-window spread and across-window spread are different "
        "quantities, not confidence intervals.\n\n"
    )
    text += (
        "CSV exports retain the values behind each plot. provenance.json "
        "identifies every source file. "
    )
    text += "No kernel or model was executed to generate these figures.\n\n"
    text += (
        "```sh\nUV_CACHE_DIR=.uv/plot-cache uv run --locked --script tools/plot_source_replay.py "
    )
    text += f"{root} --output {out}\n```\n"
    (out / "README.md").write_text(text)
    print(f"Saved 5 figures in PNG/PDF/SVG and {len(samples)} paired samples to {out}")


if __name__ == "__main__":
    main()
