# /// script
# requires-python = "==3.12.*"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Plot the E3 records from `tools/template_chain/figdata.py`; runs no kernel and no model.

uv run --locked --script tools/plot_template_chain.py FIGDATA OUT
"""

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ARMS = ("none", "prescribed", "selection")
COLORS = dict(zip(ARMS, ("#2f6fb5", "#d9772b", "#2a9d7a"), strict=True))
ROUNDS = {1: "#9aa5b1", 2: "#1f2933"}
TASKS = {
    "paged_decode": "Paged decode (FlashInfer FA2)",
    "causal_forward": "Causal forward (FlashAttention-2)",
}
INK, MUTED, GRID = "#1f2933", "#616e7c", "#e4e7eb"
MARKERS = {
    (1, "mip"): dict(marker="o", facecolors="none"),
    (2, "mip"): dict(marker="o"),
    (1, "anchor"): dict(marker="D"),
    (1, "random"): dict(marker="x"),
}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def percent_axis(ax, axis="x"):
    fmt = FuncFormatter(lambda v, _: f"{(v - 1) * 100:+.0f} %" if abs(v - 1) > 1e-9 else "equal")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(fmt)
    target.set_minor_formatter(NullFormatter())


def cells_of(rows, task):
    order = sorted({(ARMS.index(r["arm"]), r["repeat"]) for r in rows if r["task"] == task})
    return [(ARMS[a], rep) for a, rep in order]


def all_configs(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ax, (task, title) in zip(axes, TASKS.items(), strict=True):
        cells = cells_of(rows, task)
        y = {c: len(cells) - 1 - i for i, c in enumerate(cells)}
        for r in rows:
            if r["task"] != task:
                continue
            style = dict(MARKERS.get((r["round"], r["role"]), MARKERS[(2, "mip")]))
            kw = {"s": 34, "linewidths": 1.3, "alpha": 0.9, "zorder": 3}
            color = COLORS[r["arm"]]
            pos = y[(r["arm"], r["repeat"])] + (0.17 if r["round"] == 1 else -0.17)
            if style.get("facecolors") == "none":
                ax.scatter(r["ratio"], pos, edgecolors=color, **style, **kw)
            else:
                ax.scatter(r["ratio"], pos, color=color, **style, **kw)
        ax.axvline(1, color=INK, lw=1.2)
        ax.set_xscale("log")
        ax.set_xticks([0.7, 1, 1.5, 2, 3, 5])
        percent_axis(ax)
        ax.set_yticks(list(y.values()), [f"{a} {r}" for a, r in y])
        ax.set_ylim(-0.7, len(cells) - 0.3)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Time against the production kernel in the same batch (left is faster)")
        ax.grid(axis="y", visible=False)
    handles = [
        Line2D([], [], ls="", marker="o", mfc="none", mec=MUTED, label="Round 1, MIP"),
        Line2D([], [], ls="", marker="D", color=MUTED, label="Round 1, agent's anchor"),
        Line2D([], [], ls="", marker="x", color=MUTED, label="Round 1, random control"),
        Line2D([], [], ls="", marker="o", color=MUTED, label="Round 2, MIP after revision"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05)
    )
    fig.suptitle("Every measured configuration", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "01_every_configuration.png")
    fig.savefig(out / "01_every_configuration.svg")


def progress(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    stages = ("Agent's anchor", "Round 1 best", "Round 2 best")
    for ax, (task, title) in zip(axes, TASKS.items(), strict=True):
        ends = []
        for arm, rep in cells_of(rows, task):
            mine = [r for r in rows if r["task"] == task and r["arm"] == arm and r["repeat"] == rep]
            anchor = next((r["ratio"] for r in mine if r["role"] == "anchor"), None)
            one = min((r["ratio"] for r in mine if r["round"] == 1), default=None)
            two = min((r["ratio"] for r in mine if r["round"] == 2), default=None)
            pts = [(i, v) for i, v in enumerate((anchor, one, two)) if v]
            if not pts:
                continue
            ls = "-" if rep == 1 else "--"
            ax.plot(*zip(*pts, strict=True), color=COLORS[arm], ls=ls, lw=2, marker="o", ms=5)
            ends.append((pts[-1][1], f"{arm} {rep}", COLORS[arm], pts[-1][0]))
        ends.sort()
        floor = -1.0
        spread = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.045
        for v, label, color, x in ends:
            floor = max(floor + spread, v)
            ax.annotate(
                label,
                (x, v),
                xytext=(2.08, floor),
                color=INK,
                fontsize=9,
                va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
            )
        ax.axhline(1, color=INK, lw=1.2)
        ax.set_xticks(range(3), stages)
        ax.set_xlim(-0.15, 2.55)
        percent_axis(ax, "y")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Time against the same-batch production kernel")
    fig.suptitle(
        "From the agent's guess to the search result (lower is faster)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "02_guess_to_search.png")
    fig.savefig(out / "02_guess_to_search.svg")


def prediction(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (task, title) in zip(axes, TASKS.items(), strict=True):
        mine = [r for r in rows if r["task"] == task]
        hi = max(max(r["predicted_us"], r["measured_us"]) for r in mine) * 1.1
        lo = min(min(r["predicted_us"], r["measured_us"]) for r in mine) * 0.9
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1)
        ax.fill_between(
            [lo, hi], [lo * 0.8, hi * 0.8], [lo * 1.2, hi * 1.2], color=GRID, alpha=0.6, lw=0
        )
        for n, color in ROUNDS.items():
            got = [r for r in mine if r["round"] == n]
            ax.scatter(
                [r["predicted_us"] for r in got],
                [r["measured_us"] for r in got],
                s=30,
                color=color,
                label=f"Round {n}",
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ticks = (
            (2000, 3000, 5000, 10000, 20000)
            if task == "paged_decode"
            else (150, 200, 300, 500, 1000)
        )
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_ticks(ticks)
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Predicted µs, form on compile records")
        ax.set_ylabel("Measured µs")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Prediction against measurement; the band is ±20 %",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "03_prediction.png")
    fig.savefig(out / "03_prediction.svg")


def ranking(rows, out):
    counts = defaultdict(lambda: [0, 0])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["arm"], r["repeat"], r["round"])].append(r)
    for (task, _, _, n), got in groups.items():
        for a, b in itertools.combinations(got, 2):
            if abs(a["measured_us"] / b["measured_us"] - 1) < 0.03:
                continue
            counts[(task, n)][1] += 1
            counts[(task, n)][0] += (a["predicted_us"] < b["predicted_us"]) == (
                a["measured_us"] < b["measured_us"]
            )
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels, y = [], 0
    for task, title in TASKS.items():
        for n in (1, 2):
            good, total = counts[(task, n)]
            if total:
                ax.barh(y, good / total, color=ROUNDS[n], height=0.6)
                ax.text(good / total + 0.01, y, f"{good}/{total}", va="center", color=INK)
            labels.append(f"{title.split(' (')[0]}, round {n}")
            y += 1
    ax.axvline(0.5, color=INK, lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(
        "Pairs at least 3 % apart ordered correctly, within one cell and round; 50 % is chance"
    )
    ax.grid(axis="y", visible=False)
    ax.set_title("Ranking quality of the form", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "04_ranking.png")
    fig.savefig(out / "04_ranking.svg")


def funnel(data, out):
    data = sorted(data, key=lambda r: (r["task"], ARMS.index(r["arm"]), r["repeat"], r["round"]))
    fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.34)))
    y = list(range(len(data)))
    for key, color, h, off, label in (
        ("domain", "#cbd2d9", 0.72, 0, "Search space"),
        ("compiled", "#7b8794", 0.46, 0, "Compiled"),
        ("measured", "#1f2933", 0.22, 0, "Measured on GPU"),
    ):
        ax.barh([i + off for i in y], [r[key] for r in data], color=color, height=h, label=label)
    for i, r in enumerate(data):
        ax.text(
            r["domain"] * 1.06,
            i,
            f"{r['domain']}  ({r['rejected']} rejected at compile)",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{r['task'].split('_')[0]} {r['cell']} r{r['round']}" for r in data])
    ax.invert_yaxis()
    ax.set_xlim(1, 4000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    ax.set_title("Configurations at each stage, log scale", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "05_funnel.png")
    fig.savefig(out / "05_funnel.svg")


def cost(cells, out):
    cells = sorted(cells, key=lambda r: (r["task"], ARMS.index(r["arm"]), r["repeat"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    y = list(range(len(cells)))
    labels = [f"{r['task'].split('_')[0]} {r['cell']}" for r in cells]
    colors = [COLORS[r["arm"]] for r in cells]
    specs = (
        ("model_minutes", 1, "Model session minutes", "{:.0f}"),
        ("output_tokens", 1e3, "Output tokens, thousands", "{:.0f}"),
        ("input_tokens", 1e6, "Input tokens, millions (mostly cached)", "{:.1f}"),
    )
    for ax, (key, div, title, fmt) in zip(axes, specs, strict=True):
        values = [r[key] / div for r in cells]
        ax.barh(y, values, color=colors, height=0.68)
        for i, v in enumerate(values):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.15)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Model cost per cell", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "06_cost.png")
    fig.savefig(out / "06_cost.svg")


def confirmation(windows, out):
    if not windows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, (task, title) in zip(axes, TASKS.items(), strict=True):
        cells = cells_of(windows, task)
        y = {c: len(cells) - 1 - i for i, c in enumerate(cells)}
        for r in windows:
            if r["task"] != task or not r["ratio"]:
                continue
            ax.scatter(
                r["ratio"],
                y[(r["arm"], r["repeat"])],
                s=46,
                marker="o" if r["valid"] else "x",
                color=COLORS[r["arm"]],
                zorder=3,
            )
        ax.axvline(1, color=INK, lw=1.2)
        percent_axis(ax)
        ax.set_yticks(list(y.values()), [f"{a} {r}" for a, r in y])
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Best kernel against production, fresh seeds (left is faster)")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Confirmation: three windows per cell, production measured around each",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "07_confirmation.png")
    fig.savefig(out / "07_confirmation.svg")


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    load = lambda n: json.loads((data / f"{n}.json").read_text())  # noqa: E731
    rows = load("rows")
    all_configs(rows, out)
    progress(rows, out)
    prediction(rows, out)
    ranking(rows, out)
    funnel(load("funnel"), out)
    cost(load("cells"), out)
    confirmation(load("windows"), out)


if __name__ == "__main__":
    main()
