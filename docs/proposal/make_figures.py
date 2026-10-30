"""Figure 2 of the research proposal (preliminary results), from the same results file as the project page.

    python docs/proposal/make_figures.py --results results.json --runs $CVLA_RUNS/final --out docs/proposal/fig_results.pdf
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
import build_page as B  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7.5,
                     "axes.titlesize": 8.5, "pdf.fonttype": 42, "axes.grid": True})
CORE_COLORS = {1: B.TASK_COLORS[1], 2: B.TASK_COLORS[2], None: "#d6e8c4"}


def panel_cores(ax, core):
    """Every unit group of the action expert (rows: layers, columns: 15 heads then 32 MLP blocks), coloured by the core
    that holds it in the seed-0 three-task policy."""
    for layer in range(16):
        heads, blocks, _ = B.layer_owners(core, layer)
        for i, owner in enumerate(heads + blocks):
            x = i + (0.6 if i >= B.HEADS else 0)  # a gap between heads and MLP blocks
            ax.add_patch(Rectangle((x, layer), 0.86, 0.82, color=CORE_COLORS[owner], lw=0))
    ax.set_xlim(-0.3, B.UNITS + 0.9)
    ax.set_ylim(-0.4, 16.2)
    ax.set_xticks([B.HEADS / 2, B.HEADS + 0.6 + B.BLOCKS / 2])
    ax.set_xticklabels(["Heads", "MLP blocks"])
    ax.set_yticks([0.4, 15.4])
    ax.set_yticklabels(["1", "16"])
    ax.set_ylabel("Layer", labelpad=-2)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(color=CORE_COLORS[1], label="Pushing core (5%)"),
                       Patch(color=CORE_COLORS[2], label="Red-on-green core (5%)"),
                       Patch(color=CORE_COLORS[None], label="Free (learns green on red)")],
              loc="upper left", bbox_to_anchor=(-0.02, -0.16), ncol=2, fontsize=6.6, handlelength=1, columnspacing=0.8,
              handletextpad=0.4, borderaxespad=0)
    ax.set_title("Cores after three tasks", loc="left", fontweight="bold")


def panel_forgetting(ax, N):
    """Pushing while the second task is learned (validation scenes, mean of the training seeds)."""
    names = {"seqft": "Fine-tuning", "ewc": "EWC", "er50": "ER-50", "gate_k5c_curve": "Consolidate-and-gate"}
    for m in B.CURVE_METHODS:
        steps, ys = B.mean_curve(B.stage_curves(N, m), B.PUSH)
        ax.plot(steps, ys, "-o", color=B.COLORS[m], lw=1.4, ms=2.6, label=names[m], zorder=3 if m.startswith("gate") else 2)
    ax.set_xscale("symlog", linthresh=50, linscale=0.8)
    ticks = [0, 50, 500, 5000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(-5, 12000)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("Updates on the second task")
    ax.set_ylabel("Pushing success (%)")
    ax.set_title("Pushing during task 2", loc="left", fontweight="bold")


def panel_allocation(ax, N):
    """Three-task average against the share of the network reserved per task."""
    for name, color, k3 in (("Consolidate-and-gate", B.COLORS["gate_k5c"], 2), ("PackNet", B.COLORS["packnet_k5sp"], 4)):
        pts = [(c[0], B.mean_sd(N["three"][c[k3]]["avg"])) for c in B.CURVE if N["three"][c[k3]]["avg"]]
        ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], fmt="-o", color=color, lw=1.4,
                    ms=3.2, capsize=2, label=name, zorder=3)
    ax.axhline(B.mean_sd(N["three"]["joint"]["avg"])[0], color=B.GREY, lw=1, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([c[0] for c in B.CURVE])
    ax.set_xticklabels([f"{c[0]:g}" for c in B.CURVE])
    ax.minorticks_off()
    ax.set_xlim(40, 0.65)  # smaller shares to the right
    ax.set_ylim(20, 85)
    ax.set_xlabel("Share of the network per task (%)")
    ax.set_ylabel("Three-task average (%)")
    ax.set_title("Allocation per task", loc="left", fontweight="bold")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    N = B.collect(json.load(open(args.results)))
    core = json.load(open(os.path.join(args.runs, "s0", "C_gate_rand_k5_c2000", "core.json")))
    fig = plt.figure(figsize=(7.3, 2.2))
    axes = [fig.add_axes(r) for r in ([0.045, 0.33, 0.33, 0.56], [0.465, 0.33, 0.215, 0.56], [0.775, 0.33, 0.215, 0.56])]
    panel_cores(axes[0], core)
    panel_forgetting(axes[1], N)
    panel_allocation(axes[2], N)
    methods = [("Fine-tuning", "seqft"), ("EWC", "ewc"), ("ER-50", "er50"), ("PackNet", "packnet_k5sp"),
               ("Consolidate-and-gate", "gate_k5c")]
    handles = [Line2D([], [], color=B.COLORS[m], marker="o", ms=3, lw=1.4, label=n) for n, m in methods]
    handles.append(Line2D([], [], color=B.GREY, lw=1, ls="--", label="Joint training (reference)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.43, 0.0), ncol=3, fontsize=6.6, handlelength=1.6,
               columnspacing=1.2, handletextpad=0.4, borderaxespad=0)
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.13 if letter == "a" else -0.3, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    main()
