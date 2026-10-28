"""Build the project page as one self-contained HTML file: styles, charts (inline SVG), images and
videos (re-encoded and embedded as data URIs) all live inside the output file.

    python scripts/build_page.py --results results.json --runs $CVLA_RUNS/final --data $CVLA_DATA \
        --repo-url https://github.com/DishengTang/continual_vla --out docs/index.html

The text is in scripts/page_template.html; every number is computed here from the output of
scripts/export_results.py. Rollout clips come from a seed-0 test evaluation that recorded every episode
(scripts/evaluate.py --save-traces --video-all --tag test_videos): for each row the page shows one scene
where every method shows its intended outcome (EXAMPLES below), preferring successes that are still
complete when the episode ends, with the fewest lapses and the earliest completion. Needs numpy, h5py, Pillow, matplotlib and ffmpeg;
no simulator, GPU or model.
"""

import argparse
import base64
import datetime
import glob
import html
import io
import json
import math
import os
import re
import subprocess
import tempfile

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PUSH, STACK, GREEN = "PushCube-v1", "StackCube-v1", "StackCubeGreenOnRed-v1"
NAMES = {"seqft": "Fine-tuning", "ewc": "EWC", "er50": "ER-50", "er500": "ER-500",
         "gate_rand": "Consolidate-and-gate, 25% cores", "gate_k10": "Consolidate-and-gate, 10% cores",
         "gate_k50": "Consolidate-and-gate, 50% cores", "gate_k5": "Consolidate-and-gate, 5% cores, 500 consolidation updates",
         "joint": "Joint training", "routed": "Two checkpoints", "packnet": "PackNet", "packnet_k10": "PackNet, 10% per task, 500 retraining updates, projections pruned",
         "packnet_k5": "PackNet, 5% per task, 500 retraining updates, projections pruned",
         "gate_k5c": "Consolidate-and-gate, 5% cores",
         "gate_k2p5c": "Consolidate-and-gate, 2.5% cores",
         "gate_k1c": "Consolidate-and-gate, 1% cores",
         "gate_k10L": "Consolidate-and-gate, 10% cores, 4&times; stacking consolidation",
         "gate_k5s25": "Consolidate-and-gate, 5% core for pushing, 25% for stacking",
         "packnet_unfrozen": "PackNet checkpoint, task 3 unfrozen",
         "gate_adapter": "Consolidate-and-gate + interface adapter",
         "gate_bundle": "Consolidate-and-gate, attention-aware cores",
         "seqft_plus6k": "Fine-tuning, +6000 stacking updates",
         "er50_plus6k": "ER-50, +6000 stacking updates",
         "er500_plus6k": "ER-500, +6000 stacking updates",
         "seqft_plus24k": "Fine-tuning, +24000 stacking updates",
         "er500_plus24k": "ER-500, +24000 stacking updates",
         "gate_bundle_L24k": "Consolidate-and-gate, attention-aware cores, 24000 stacking consolidation",
         "gate_bundle_L24k_brC": "Consolidate-and-gate, attention-aware cores, 24000 stacking consolidation, task-3 branch",
         "gate_bundle_L24k_brC64": "Consolidate-and-gate, attention-aware cores, 24000 stacking consolidation, task-3 branch (rank 64)",
         "packnet_L24k": "PackNet, 24000 stacking retraining", "packnet_k5sp": "PackNet, 5% per task",
         "packnet_k2p5sp": "PackNet, 2.5% per task", "packnet_k10sp": "PackNet, 10% per task, matched", "packnet_k1sp": "PackNet, 1% per task",
         "gate_k10c": "Consolidate-and-gate, 10% cores, 4&times; consolidation",
         "gate_k5c_curve": "Consolidate-and-gate, 5% cores", "gate_nofreeze_k5c": "Consolidate-and-gate, 5% cores, core not frozen",
         "gate_abl_k5c": "Consolidate-and-gate, 5% cores selected by intervention"}
# Each method is reported at its configuration with the best three-task average on the test scenes, among those run on
# every seed (the same rule for every method). The gate and PackNet reserve a share of the network per task and are
# compared at a nominal 5% per task; other shares are in the budget comparison.
FAMILIES = {"seqft": ("seqft", "seqft_plus6k", "seqft_plus24k"), "ewc": ("ewc",), "er50": ("er50", "er50_plus6k"),
            "er500": ("er500", "er500_plus6k", "er500_plus24k"), "packnet": ("packnet_k5", "packnet_k5sp"),
            "gate": ("gate_k5", "gate_k5c")}
FAMILY_NAMES = {"seqft": "Fine-tuning", "ewc": "EWC", "er50": "ER-50", "er500": "ER-500", "packnet": "PackNet", "gate": "Consolidate-and-gate"}
FAMILY_OF = {"seqft": "seqft", "ewc": "ewc", "er50": "er50", "er500": "er500"}
OURS = set()  # the selected gate configuration (build)
COLORS = {"gate_rand": "#2a78d6", "gate_k10": "#2a78d6", "gate_k50": "#2a78d6", "gate_k5": "#2a78d6", "seqft": "#eb6834", "er50": "#1baf7a",
          "ewc": "#eda100", "er500": "#e87ba4", "packnet": "#6250d6", "packnet_k10": "#6250d6", "packnet_k5": "#6250d6",
          "gate_k5c": "#2a78d6", "gate_k2p5c": "#2a78d6", "gate_k1c": "#2a78d6", "gate_k10L": "#2a78d6", "gate_k5s25": "#2a78d6", "packnet_unfrozen": "#6250d6", "gate_adapter": "#2a78d6", "gate_bundle": "#2a78d6", "seqft_plus6k": "#eb6834", "er50_plus6k": "#1baf7a", "er500_plus6k": "#e87ba4", "seqft_plus24k": "#eb6834", "er500_plus24k": "#e87ba4",
          "gate_bundle_L24k": "#2a78d6", "gate_bundle_L24k_brC": "#2a78d6", "gate_bundle_L24k_brC64": "#2a78d6",
          "packnet_L24k": "#6250d6", "packnet_k5sp": "#6250d6", "packnet_k2p5sp": "#6250d6", "packnet_k10sp": "#6250d6", "packnet_k1sp": "#6250d6",
          "gate_k10c": "#2a78d6", "gate_k5c_curve": "#2a78d6"}
GREY, INK, INK2 = "#8f8e88", "#121211", "#52514e"
TASK_NAMES = {PUSH: "Push the cube onto the target.", STACK: "Stack the red cube on the green cube.",
              GREEN: "Stack the green cube on the red cube."}
VIDEO_SEED, VIDEO_TAG, FIRST_TEST_SEED = "s0", "test_videos", 200000
# Outcome shown per method and instruction: a clean success where the method succeeds often, its typical
# failure where it rarely does ("wrong": built the other stack).
EXAMPLES = {
    "C": [(PUSH, {"seqft": "failure", "er50": "success", "packnet": "success", "gate": "success"}),
          (STACK, {"seqft": "wrong", "er50": "failure", "packnet": "success", "gate": "success"}),
          (GREEN, {"seqft": "success", "er50": "success", "packnet": "success", "gate": "success"})],
}

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b9b8b2",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


# ----------------------------------------------------------------------------------------------- numbers

def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def pm(values):
    if not values:
        return "&mdash;"
    m, s = mean_sd(values)
    return f"{m:.1f} &plusmn; {s:.1f}" if len(values) > 1 else f"{m:.1f}"


def mean(values):
    return f"{mean_sd(values)[0]:.1f}"


def label(m, values, total, ours=True):
    """Method name, marked (ours) for the main method and with the seed count while seeds are missing."""
    name = NAMES[m] + (" (ours)" if m in OURS and ours else "")
    return name + (f" ({len(values)} of {total} seeds)" if 0 < len(values) < total else "")


def per_seed(seeds, fn):
    """fn(seed record) for every seed where it is available (missing runs are skipped)."""
    out = []
    for s in sorted(seeds, key=int):
        try:
            v = fn(seeds[s])
        except (KeyError, TypeError):
            continue
        if v is not None:
            out.append(v)
    return out


def collect(results):
    seeds = results["seeds"]
    two = {}
    for m in ("seqft", "ewc", "er50", "er500", "gate_rand", "joint", "gate_nofreeze", "gate_abl", "gate_k10", "gate_k50",
              "gate_k5", "gate_k2p5", "gate_k1", "gate_k0p5", "packnet", "packnet_k10", "packnet_k5", "gate_k5c", "gate_k2p5c", "gate_k1c", "gate_bundle",
              "packnet_k5sp", "packnet_k1sp", "gate_k5c_curve", "gate_nofreeze_k5c", "gate_abl_k5c"):
        def done(d, m=m):
            c = d["conditions"][m]
            return c if c["done"] and c["stack_after_b"] else None
        two[m] = {k: per_seed(seeds, lambda d, f=f: f(done(d))) for k, f in {
            "push": lambda c: c["push_after_b"]["any"], "stack": lambda c: c["stack_after_b"]["any"],
            "stack_end": lambda c: c["stack_after_b"]["end"],
            "push_before": lambda c: c["push_before_b"]["any"] if c["push_before_b"] else None,
            "avg": lambda c: (c["push_after_b"]["any"] + c["stack_after_b"]["any"]) / 2,
            "gate_off": lambda c: 100 * c["gate_off_push"][-1]["success"] if c["gate_off_push"] else None,
            "curve": lambda c: c["curve"], "curve_early": lambda c: c.get("curve_early")}.items()}
    two["routed"] = {"push": two["seqft"]["push_before"], "stack": two["seqft"]["stack"],
                     "stack_end": two["seqft"]["stack_end"],
                     "avg": [(p + s) / 2 for p, s in zip(two["seqft"]["push_before"], two["seqft"]["stack"])]}

    three = {}
    for m in ("seqft", "ewc", "er50", "er500", "packnet", "gate_rand", "gate_k10", "gate_k5", "gate_k2p5", "gate_k1", "gate_k0p5",
              "gate_k50_r0", "gate_k50_r20", "gate_k50_r50", "gate_k50_r20_d1", "gate_k50_r50_d1", "packnet_k10", "packnet_k5", "gate_k10L", "gate_k5s25", "packnet_unfrozen", "gate_adapter", "gate_bundle", "seqft_plus6k", "er50_plus6k", "er500_plus6k", "seqft_plus24k", "er500_plus24k",
              "gate_k5c", "gate_k2p5c", "gate_k1c", "gate_bundle_L24k", "gate_bundle_L24k_brC", "gate_bundle_L24k_brC64",
              "packnet_L24k", "packnet_k5sp", "packnet_k2p5sp", "packnet_k50_r0", "packnet_k50_r50", "gate_k50_r50rg",
              "gate_k10c", "packnet_k10sp", "joint", "packnet_k1sp"):
        def done3(d, m=m):  # every task scored after the third stage
            r = d["three_task"][m]
            return r if r["done"] and all(r["after_c"][t] for t in (PUSH, STACK, GREEN)) else None
        three[m] = {
            "push": per_seed(seeds, lambda d: done3(d)["after_c"][PUSH]["any"]),
            "rog": per_seed(seeds, lambda d: done3(d)["after_c"][STACK]["any"]),
            "rog_wrong": per_seed(seeds, lambda d: done3(d)["after_c"][STACK]["wrong"]),
            "gor": per_seed(seeds, lambda d: done3(d)["after_c"][GREEN]["any"]),
            "gor_wrong": per_seed(seeds, lambda d: done3(d)["after_c"][GREEN]["wrong"]),
            "rog_before": per_seed(seeds, lambda d: done3(d)["before_c"][STACK]["any"]),
            "push_before": per_seed(seeds, lambda d: done3(d)["before_c"][PUSH]["any"]),
        }
        three[m]["avg"] = [(a + b + c) / 3 for a, b, c in zip(three[m]["push"], three[m]["rog"], three[m]["gor"])]

    cons = {sel: {k: per_seed(seeds, lambda d, k=k: d["consolidation"][sel][k]["any"]) for k in ("0", "500")}
            for sel in ("rand", "abl")}
    traces = {st: per_seed(seeds, lambda d, st=st: d["traces"][st]) for st in ("B_seqft", "B_er50", "B_gate_rand_k5_c2000")}
    core_push = {k: {w: per_seed(seeds, lambda d, k=k, w=w: d["core_push"][k][w]["any"]) for w in ("before", "after")}
                 for k in ("0p5", "1", "2p5", "5", "10", "25", "50", "1c", "2p5c", "5c", "abl5c", "bundle")}
    protected = {k: {w: per_seed(seeds, lambda d, k=k, w=w: d["protected"][k][w]) for w in ("after_1", "after_2")}
                 for k in ("gate_bundle", "packnet", "gate_k10c", "packnet_k10sp", "gate_k5c", "packnet_k5sp", "gate_k2p5c",
                           "packnet_k2p5sp", "gate_k1c", "packnet_k1sp")}
    total = len(seeds)
    sel = {}
    for fam, members in FAMILIES.items():
        full = [m for m in members if len(three[m]["avg"]) == total]
        if full:
            sel[fam] = max(full, key=lambda m: mean_sd(three[m]["avg"])[0])
    if "seqft" in sel:  # one fine-tuning checkpoint per task (fine-tuning's selected chain), chosen by instruction
        f = three[sel["seqft"]]
        three["routed"] = {"push": two["seqft"]["push_before"], "rog": f["rog_before"], "gor": f["gor"]}
        three["routed"]["avg"] = [(a + b + c) / 3 for a, b, c in zip(two["seqft"]["push_before"], f["rog_before"], f["gor"])]
    budget = {k: per_seed(seeds, lambda d, k=k: d["budget_push"][k]["any"])
              for k in ("gate_bundle", "gate_k5c", "packnet", "packnet_k5", "packnet_k5r", "packnet_k5sp", "gate_k2p5c",
                        "packnet_k2p5sp", "gate_k50", "packnet_k50sp", "gate_k10", "packnet_k10", "gate_k10c", "packnet_k10sp",
                        "gate_k1c", "packnet_k1sp")}
    return {"two": two, "three": three, "cons": cons, "traces": traces, "total": total, "core_push": core_push, "protected": protected,
            "core_size": seeds.get("0", {}).get("core_size", {}),
            "exec": seeds.get("0", {}).get("exec_steps", {}), "sel": sel, "budget": budget,
            "sliced": results.get("sliced", {}), "core_json_bytes": results.get("core_json_bytes")}


# ----------------------------------------------------------------------------------------------- charts

def to_svg(fig, salt):
    plt.rcParams["svg.hashsalt"] = salt  # distinct element ids for every inline SVG on the page
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    end = s.index(">")
    head = re.sub(r'\s(width|height)="[^"]*"', "", s[:end])  # size comes from CSS and the viewBox
    return head + s[end:]


def mean_curve(curves, task, last=12000):
    steps = sorted(set.intersection(*[{p["step"] for p in c[task]} for c in curves]))
    steps = [s for s in steps if s < last]
    return steps, [np.mean([next(p["success"] for p in c[task] if p["step"] == s) for c in curves]) for s in steps]


CURVE_METHODS = ("seqft", "ewc", "er50", "gate_k5c_curve")


def stage_curves(N, m):
    """Per-seed stage-B learning curves; before 2000 updates from the run of the stage's first 2000 updates,
    once that run has finished for every seed."""
    curves, early = N["two"][m]["curve"], N["two"][m].get("curve_early", [])
    if len(early) != len(curves) or not all(e[t] and e[t][-1]["step"] == 2000 for e in early for t in (PUSH, STACK)):
        return curves
    return [{t: [p for p in e[t] if p["step"] < 2000] + c[t] for t in (PUSH, STACK)} for c, e in zip(curves, early)]


def has_early(N):
    return all(stage_curves(N, m) is not N["two"][m]["curve"] for m in CURVE_METHODS)


def chart_curves(N):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    early = has_early(N)
    for ax, task, title in zip(axes, (PUSH, STACK), ("Pushing (old task)", "Stacking (new task)")):
        for m in CURVE_METHODS:
            curves = [c for c in (stage_curves(N, m) if early else N["two"][m]["curve"]) if c[task]]
            if not curves:
                continue
            steps, ys = mean_curve(curves, task)
            ax.plot(steps, ys, "-o", color=COLORS[m], lw=2, ms=4.5, label=label(m, curves, N["total"]),
                    zorder=3 if m == "gate_k5c_curve" else 2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("Stage-B updates" + (" (linear up to 50, logarithmic above)" if early else ""))
        if early:  # the first updates matter most: linear up to 50, logarithmic above
            ax.set_xscale("symlog", linthresh=50, linscale=1.0)
            ticks = [0, 10, 20, 30, 50, 100, 250, 500, 2000, 10000]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.minorticks_off()
            ax.set_xlim(-10, 11000)
        else:
            ax.set_xticks([2000, 4000, 6000, 8000, 10000])
        ax.set_ylim(-4, 104)
    axes[0].set_ylabel("Success, validation scenes (%)")
    axes[1].legend(loc="upper left", ncol=2, fontsize=9.5)
    fig.tight_layout(w_pad=2.5)
    return to_svg(fig, "curves")


def chart_three(N):
    tasks = [("push", None, "Pushing"), ("rog", "rog_wrong", "Red on green"), ("gor", "gor_wrong", "Green on red")]
    fams = [f for f in ("seqft", "ewc", "er50", "er500", "packnet", "gate") if f in N["sel"]]
    methods = [N["sel"][f] for f in fams] + (["joint"] if N["three"]["joint"]["push"] else [])  # joint: reference
    color = lambda m: GREY if m == "joint" else COLORS[m]  # noqa: E731
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        for j, (key, wrong, _) in enumerate(tasks):
            x = j + (i - (len(methods) - 1) / 2) * width
            v = N["three"][m][key]
            h = mean_sd(v)[0]
            ax.bar(x, h, width * 0.9, color=color(m), zorder=2)
            ax.scatter([x] * len(v), v, s=10, color="white", edgecolors=color(m), linewidths=0.8, zorder=4)
            if wrong:
                w = mean_sd(N["three"][m][wrong])[0]
                ax.bar(x, w, width * 0.9, bottom=h, color="white", edgecolor=color(m), hatch="////",
                       linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[2] for t in tasks])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success after task 3 (%)")
    ax.grid(axis="x", visible=False)
    handles = [matplotlib.patches.Patch(color=COLORS[m], label=FAMILY_NAMES[f]) for f, m in zip(fams, methods)]
    if "joint" in methods:
        handles.append(matplotlib.patches.Patch(color=GREY, label="Joint training (reference)"))
    handles.append(matplotlib.patches.Patch(facecolor="white", edgecolor=GREY, hatch="////", label="built the other stack"))
    ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
    return to_svg(fig, "three")


def chart_exec(N):
    exec_ = N["exec"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    panels = ((PUSH, "any", "Pushing"), (STACK, "any", "Stacking, any step"), (STACK, "end", "Stacking, final step"))
    for ax, (task, key, title) in zip(axes, panels):
        for stage, m in (("B_seqft", "seqft"), ("B_er50", "er50"), ("B_gate_rand_k5_c2000", "gate_k5c")):
            pts = exec_.get(stage, {})
            ns = sorted(int(n) for n in pts if pts[n].get(task))
            if len(ns) < 2:
                continue
            ax.plot(ns, [pts[str(n)][task][key] for n in ns], "-o", color=COLORS[m], lw=2, ms=4.5, label=NAMES[m])
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 25, 50])
        ax.set_xticklabels(["1", "2", "5", "10", "25", "50"])
        ax.minorticks_off()
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("Actions executed per chunk of 50")
        ax.set_ylim(-4, 104)
    axes[0].set_ylabel("Success, test scenes (%)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout(w_pad=2.5)
    return to_svg(fig, "exec")


def stage3(N, m):
    """Three-task results for a two-task method: its selected three-task configuration (the gate and PackNet: their own)."""
    fam = FAMILY_OF.get(m)
    return N["three"][N["sel"][fam]] if fam in N["sel"] else N["three"].get(m, {})


def chart_name(N, m, values):
    """Legend name: the method alone for the reported gate and PackNet configurations."""
    for fam in ("gate", "packnet"):
        if m == N["sel"].get(fam):
            return FAMILY_NAMES[fam] + (" (ours)" if fam == "gate" else "") + \
                (f" ({len(values)} of {N['total']} seeds)" if len(values) < N["total"] else "")
    return label(m, values, N["total"])


def chart_sequence(N):
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    gate = N["sel"].get("gate")
    for m in ("seqft", "ewc", "er50", "er500", N["sel"].get("packnet"), gate):
        if m is None:
            continue
        two, three = N["two"].get(m, {}), stage3(N, m)
        pts = [(x, v) for x, v in ((1, two.get("push_before")), (2, two.get("avg")), (3, three.get("avg"))) if v]
        if len(pts) < 2:
            continue
        stats = [mean_sd(v) for _, v in pts]
        gap = [x for x, _ in pts] != list(range(1, len(pts) + 1))  # a stage not scored yet: points only, no line through it
        ax.errorbar([x for x, _ in pts], [a for a, _ in stats], yerr=[b for _, b in stats], fmt="o" if gap else "-o",
                    color=COLORS[m], lw=2.6 if m == gate else 1.8, ms=6.5, capsize=3,
                    zorder=3 if m == gate else 2, label=chart_name(N, m, pts[-1][1]))
    joint = [(x, v) for x, v in ((2, N["two"]["joint"]["avg"]), (3, N["three"]["joint"]["avg"])) if v]
    if joint:
        stats = [mean_sd(v) for _, v in joint]
        ax.errorbar([x for x, _ in joint], [a for a, _ in stats], yerr=[b for _, b in stats], fmt="*", ms=13, color=GREY,
                    capsize=3, zorder=4, label="Joint training (reference)")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["after PushCube", "+ red on green", "+ green on red"])
    ax.set_xlim(0.75, 3.25)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Average success over learned tasks (%)")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    return to_svg(fig, "sequence")


def memory_bytes(runs, data):
    """Bytes each method keeps for earlier tasks, after two and after three tasks (seed-0 files)."""
    s0 = os.path.join(runs, "s0")
    def size(*parts):
        path = os.path.join(s0, *parts)
        return os.path.getsize(path) if os.path.exists(path) else None
    def buffer(stage, name):
        path = os.path.join(s0, stage, name + ".json")
        return json.load(open(path))["bytes"] if os.path.exists(path) else None
    core = json.load(open(os.path.join(s0, "B_gate_rand", "core.json")))
    mask = (len(core["core"]) + len(core["silenced"]) + 7) // 8  # one bit per unit group and task
    def examples(task, stage):  # training examples (demonstration steps) of a task
        replay = json.load(open(os.path.join(data, "replay", task, "trajectory.rgb.pd_ee_delta_pos.physx_cpu.json")))
        lengths = {e["episode_id"]: e["elapsed_steps"] for e in replay["episodes"]}
        return sum(lengths[i] for i in json.load(open(os.path.join(s0, stage, "splits.json")))[task]["train"])
    push_examples, stack_examples = examples(PUSH, "A_plus"), examples(STACK, "B_seqft")
    per_example = buffer("A_plus", "memory") / json.load(open(os.path.join(s0, "A_plus", "memory.json")))["n"]
    plus = lambda a, b: a + b if a is not None and b is not None else None  # noqa: E731
    weights = size("A_plus", "trainable.safetensors") // 4  # fp32 trainable weights
    packnet_mask = weights // 8  # PackNet: one bit per weight and task
    records = {m: mask if m.startswith("gate") else packnet_mask for m in NAMES if m.startswith(("gate", "packnet"))}
    return {"two": {"seqft": 0, "ewc": plus(size("A_plus", "fisher.safetensors"), size("A_plus", "trainable.safetensors")),
                    "er50": buffer("A_plus", "memory"), "er500": buffer("A_plus", "memory500"), **records,
                    "joint": per_example * push_examples, "routed": size("A_plus", "trainable.safetensors")},
            "three": {"seqft": 0, "ewc": plus(size("A_plus", "fisher.safetensors"), size("A_plus", "trainable.safetensors")),
                      "er50": plus(buffer("A_plus", "memory"), buffer("B_er50", "memory")),
                      "er500": plus(buffer("A_plus", "memory500"), buffer("B_er500", "memory500")),
                      **{m: 2 * v for m, v in records.items()},
                      "joint": per_example * (push_examples + stack_examples), "routed": 2 * size("A_plus", "trainable.safetensors")}}


def chart_cost(N, mem):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    gate, packnet = N["sel"].get("gate"), N["sel"].get("packnet")
    offsets = {"seqft": (9, 0, "left"), "ewc": (0, -14, "center"), "er50": (0, -14, "center"), "er500": (0, 12, "center"),
               gate: (7, -20, "left"), "joint": (0, 13, "center"), "routed": (9, -3, "left"), packnet: (-8, 10, "right"),
               (packnet, "two"): (0, -16, "center"), ("er50", "two"): (-8, 8, "right"),  # per panel, where labels would collide
               ("joint", "three"): (-11, 0, "right"), ("routed", "three"): (0, 12, "center"), ("er500", "three"): (0, -14, "center")}
    panels = [("After two tasks", "two", ("seqft", "ewc", "er50", "er500", packnet, gate, "joint", "routed")),
              ("After three tasks", "three", ("seqft", "ewc", "er50", "er500", packnet, gate, "joint", "routed"))]
    for ax, (title, key, methods) in zip(axes, panels):
        for m in methods:
            if m is None:
                continue
            values = N["two"].get(m, {}).get("avg") if key == "two" else stage3(N, m).get("avg")
            if not values or mem[key].get(m) is None:
                continue
            x, (mu, sd) = max(mem[key][m], 10), mean_sd(values)
            ref = m in ("joint", "routed")
            marker = "*" if m == "joint" else "D" if m == "routed" else "o"
            ax.errorbar(x, mu, yerr=sd, fmt=marker, ms=13 if m == "joint" else 7 if ref else 9,
                        color=GREY if ref else COLORS[m], mfc="white" if m == "routed" else None, capsize=3, zorder=3)
            dx, dy, ha = offsets.get((m, key), offsets[m])
            name = "Three checkpoints" if (m, key) == ("routed", "three") else chart_name(N, m, values)
            name = name.replace("Consolidate-and-gate ", "Consolidate-and-gate\n", 1)
            ax.annotate(name, (x, mu), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center", fontsize=9.5, color=INK2 if ref else INK,
                        fontweight="bold" if m == gate else "normal")
        ax.set_xscale("log")
        ax.set_xticks([10, 1e3, 1e5, 1e7, 1e9])
        ax.set_xticklabels(["none", "1 kB", "100 kB", "10 MB", "1 GB"])
        ax.minorticks_off()
        ax.set_xlim(4, 4e10)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("Memory kept for earlier tasks")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Average success (%)")
    fig.tight_layout(w_pad=2.5)
    return to_svg(fig, "cost")


def method_svg():
    """Schematic of the cycle every task goes through, drawn directly as SVG: learn, consolidate into a new core
    outside the earlier ones, then learn the next task with all earlier cores frozen (5% cores, three tasks)."""
    rng = np.random.default_rng(3)
    rows, cols, cell, gap = 6, 10, 13, 3
    order = rng.permutation(rows * cols)
    core1, core2 = set(order[:3].tolist()), set(order[3:6].tolist())
    colors = {1: "#2a78d6", 2: "#6250d6"}
    tint = {k: "#dbe5cf" for k in (1, 2, 3)}  # units still learning: one neutral tint, cores keep the task colours
    off = "#f1f0ec"
    def state(panel, i):  # (fill, dashed, lock)
        if panel == 0:
            return tint[1], False, False
        if panel == 1:
            return (colors[1], False, False) if i in core1 else (off, True, False)
        if panel == 2:
            return (colors[1], False, True) if i in core1 else (tint[2], False, False)
        if panel == 3:
            return ((colors[1], False, True) if i in core1 else (colors[2], False, False) if i in core2 else (off, True, False))
        return ((colors[1], False, True) if i in core1 else (colors[2], False, True) if i in core2 else (tint[3], False, False))
    panels = [("Learn task 1", "all units learn"), ("Consolidate task 1", "instruction 1 runs core 1"),
              ("Learn task 2", "core 1 frozen, the rest learn"), ("Consolidate task 2", "instruction 2 runs cores 1 + 2"),
              ("Learn task 3", "cores 1, 2 frozen, the rest learn")]
    gw = cols * (cell + gap) - gap
    step = gw + 42
    parts = ['<svg viewBox="0 0 1000 268" role="img" aria-label="The consolidate, gate and freeze cycle over three tasks">']
    for k, (title, sub) in enumerate(panels):
        x0 = 10 + k * step
        parts.append(f'<text x="{x0}" y="20" font-size="13.5" font-weight="600" fill="{INK}">{title}</text>')
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                fill, dashed, lock = state(k, i)
                x, y = x0 + c * (cell + gap), 34 + r * (cell + gap)
                stroke = ' stroke="#b9b8b2" stroke-dasharray="2 2"' if dashed else ""
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2.5" fill="{fill}"{stroke}/>')
                if lock:
                    parts.append(f'<path d="M{x + 3.5} {y + 6.5}h6v4.5h-6z M{x + 4.5} {y + 6.5}v-1.3a2 2 0 0 1 4 0v1.3" '
                                 f'fill="none" stroke="white" stroke-width="1.1"/>')
        parts.append(f'<text x="{x0}" y="{34 + rows * (cell + gap) + 15}" font-size="11.5" fill="{INK2}">{sub}</text>')
        if k < len(panels) - 1:
            xa = x0 + gw + 10
            parts.append(f'<path d="M{xa} 80h18m-5 -5l5 5l-5 5" fill="none" stroke="{GREY}" stroke-width="1.6"/>')
    parts.append(f'<text x="{10 + 5 * step - 34}" y="84" font-size="20" fill="{GREY}">&#8230;</text>')
    legend = [(colors[1], "core of task 1"), (colors[2], "core of task 2"), (tint[2], "learning the current task"),
              (off, "silenced for that task's instruction")]
    lx = 10
    for color, label in legend:
        stroke = ' stroke="#b9b8b2" stroke-dasharray="2 2"' if color == off else ""
        parts.append(f'<rect x="{lx}" y="190" width="12" height="12" rx="2.5" fill="{color}"{stroke}/>'
                     f'<text x="{lx + 18}" y="200" font-size="12" fill="{INK2}">{label}</text>')
        lx += 30 + 6.6 * len(label)
    parts.append(f'<text x="{lx + 4}" y="200" font-size="12" fill="{INK2}">padlock: frozen</text>')
    parts.append(f'<text x="10" y="232" font-size="12.5" fill="{INK2}">Task <tspan font-style="italic">k</tspan>: learn with '
                 f'cores 1&#8230;<tspan font-style="italic">k</tspan>&#8722;1 frozen, consolidate into a new core among the units no task has '
                 f'reserved, then its instruction runs cores 1&#8230;<tspan font-style="italic">k</tspan>.</text>')
    parts.append(f'<text x="10" y="252" font-size="12.5" fill="{INK2}">Each core is a fixed share of the unit groups (5% here), '
                 f'so the number of tasks is limited by capacity, not by memory.</text>')
    parts.append("</svg>")
    return "".join(parts)


# Two schematics of one action-expert layer, with the seed-0 cores of one layer: the layer drawn like the original
# Transformer with every unit shown, and the same 5% under the gate and under PackNet.
HEADS, BLOCKS, UNITS = 15, 32, 47  # query heads, MLP blocks of 64 neurons and unit groups per layer
SHOW_LAYER = 7  # seed 0: two heads and one MLP block of the pushing core, two heads of the red-on-green core
TASK_COLORS = {1: "#2a78d6", 2: "#6250d6"}
KV_TINTS = {1: "#d4e3f6", 2: "#dfdbf6"}
FREE, OFF, GATE_INK, GATE_TINT = "#dbe5cf", "#f1f0ec", "#b4246a", "#fbe9f1"
EXPERT = "model.vlm_with_expert.lm_expert.layers"


def layer_owners(core, layer):
    """Task (1: pushing, 2: red on green) whose core holds each head, MLP block and key/value head of a layer."""
    owner = {}
    for task, units in ((1, core["earlier"][0]["core"]), (2, core["new"])):
        owner.update({u % UNITS: task for u in units if u // UNITS == layer})
    heads = [owner.get(h) for h in range(HEADS)]
    kv = []  # a key/value head is frozen with the earliest core head that reads it
    for g in range(5):
        readers = [t for t in heads[3 * g:3 * g + 3] if t]
        kv.append(min(readers) if readers else None)
    return heads, [owner.get(HEADS + b) for b in range(BLOCKS)], kv


def read_tensors(path, prefix):
    """The uint8 tensors of a safetensors file whose names start with prefix (named without it and ".weight"),
    read without loading the rest."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        out = {}
        for name in sorted(k for k in header if k.startswith(prefix)):
            begin, end = header[name]["data_offsets"]
            f.seek(8 + n + begin)
            out[name[len(prefix):].removesuffix(".weight")] = np.frombuffer(f.read(end - begin), np.uint8).reshape(
                header[name]["shape"])
    return out


def svg_text(x, y, s, size=12, anchor="start", weight="normal", color=INK2):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{color}">{s}</text>')


def svg_lines(x, y, rows, size=12, color=INK2, weight="normal", lh=1.38):
    spans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else size * lh:.1f}">{r}</tspan>' for i, r in enumerate(rows))
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{spans}</text>'


def padlock(x, y, color):
    return (f'<path d="M{x} {y + 3}h6v4.5h-6z M{x + 1} {y + 3}v-1.3a2 2 0 0 1 4 0v1.3" fill="none" '
            f'stroke="{color}" stroke-width="1.1"/>')


def layer_svg(core):
    """One action-expert layer drawn bottom to top like the original Transformer, with weight shapes [output x input],
    every unit as a square coloured by the seed-0 core that holds it, and our gate hooks."""
    heads, blocks, kv = layer_owners(core, SHOW_LAYER)
    cx, p = 320, ['<svg viewBox="0 0 1000 880" role="img" aria-label="One layer of the action expert with its 15 attention '
                  'heads, 5 key/value heads and 32 MLP blocks, and the gate hooks">',
                  '<defs><marker id="lay-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
                  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{GREY}"/></marker></defs>']
    def arrow(x1, y1, x2, y2, color=GREY, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="{color}" stroke-width="1.4" marker-end="url(#lay-ah)"{d}/>'
    def path(d, head=True):
        end = ' marker-end="url(#lay-ah)"' if head else ""
        return f'<path d="{d}" fill="none" stroke="{GREY}" stroke-width="1.4"{end}/>'
    def box(x, y, w, h, title, shape="", fill="white", stroke="#a9a8a2", color=INK):
        out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        if shape:
            return out + svg_text(x + w / 2, y + h / 2 + 4.5, f'<tspan font-weight="600">{title}</tspan>  {shape}', 12.5,
                                  "middle", color=color)
        return out + svg_text(x + w / 2, y + h / 2 + 4.5, title, 12, "middle", color=color)
    def plus(y, sym):
        return (f'<circle cx="{cx}" cy="{y}" r="10" fill="white" stroke="{GREY}" stroke-width="1.4"/>'
                + svg_text(cx, y + 5, sym, 15, "middle", "bold", INK))
    def tile(x, y, w, h, task, kind="unit"):
        if kind == "kv":
            fill, stroke = (KV_TINTS[task], TASK_COLORS[task]) if task else (OFF, "#c9c8c2")
            out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2.5" fill="{fill}" stroke="{stroke}"/>'
            return out + (padlock(x + w / 2 - 3, y + h / 2 - 4, TASK_COLORS[task]) if task else "")
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2.5" fill="{TASK_COLORS[task] if task else FREE}"/>'

    # frame, input and output of the layer
    p.append('<rect x="84" y="56" width="466" height="736" rx="12" fill="none" stroke="#c9c8c2" stroke-dasharray="6 5"/>')
    p.append(svg_text(46, 430, "&#215;16", 22, "middle", "bold", INK))
    p.append(svg_text(cx, 22, "to the next layer: 50 action tokens &#215; 720", 13, "middle", "600", INK))
    p.append(arrow(cx, 63, cx, 30))
    p.append(svg_text(cx, 840, "x: 50 action tokens &#215; 720 (the residual stream)", 13, "middle", "600", INK))
    # feed-forward sub-block (top)
    p.append(plus(74, "+"))
    p.append(f'<rect x="104" y="96" width="432" height="266" rx="10" fill="#edf2f7" stroke="#b8c9da"/>')
    p.append(box(165, 106, 310, 32, "down_proj", "[720 &#215; 2048]"))
    p.append(arrow(cx, 106, cx, 85))
    p.append(box(165, 152, 310, 26, "ContextGate (ours): &#215; 1 or &#215; 0 per block", fill=GATE_TINT, stroke=GATE_INK,
                 color=GATE_INK))
    p.append(arrow(cx, 152, cx, 138))
    x0, w, g = cx - (BLOCKS * 11 + (BLOCKS - 1) * 2) / 2, 11, 2
    p.extend(tile(x0 + b * (w + g), 194, w, 20, blocks[b]) for b in range(BLOCKS))
    p.append(arrow(cx, 194, cx, 178))
    p.append(svg_text(cx - 8, 234, "2048 neurons = 32 blocks of 64", 11.5, "end"))
    p.append(arrow(cx, 248, cx, 214))
    p.append(plus(258, "&#215;"))
    p.append(box(185, 276, 70, 22, "SiLU"))
    p.append(path(f"M220 276V258H{cx - 11}"))
    p.append(path(f"M420 312V258H{cx + 11}"))
    p.append(box(140, 312, 160, 34, "gate_proj", "[2048 &#215; 720]"))
    p.append(box(340, 312, 160, 34, "up_proj", "[2048 &#215; 720]"))
    p.append(arrow(220, 312, 220, 298))
    p.append(path(f"M{cx} 372V356M220 356H420", head=False))
    p.append(arrow(220, 356, 220, 346))
    p.append(arrow(420, 356, 420, 346))
    p.append(box(165, 372, 310, 24, "RMSNorm"))
    # residual around the feed-forward
    p.append(f'<circle cx="{cx}" cy="406" r="2.6" fill="{GREY}"/>')
    p.append(path(f"M{cx} 406H96V74H{cx - 11}"))
    p.append(arrow(cx, 408, cx, 396))
    # attention sub-block
    p.append(plus(420, "+"))
    p.append(f'<rect x="104" y="440" width="432" height="282" rx="10" fill="#f8f1e7" stroke="#dcc7a6"/>')
    p.append(box(165, 450, 310, 32, "o_proj", "[720 &#215; 960]"))
    p.append(arrow(cx, 450, cx, 431))
    p.append(box(165, 496, 310, 26, "ContextGate (ours): &#215; 1 or &#215; 0 per head", fill=GATE_TINT, stroke=GATE_INK,
                 color=GATE_INK))
    p.append(arrow(cx, 496, cx, 482))
    p.append(f'<rect x="114" y="538" width="412" height="122" rx="7" fill="#f3e2c7" stroke="#d7b98b"/>')
    p.append(arrow(cx, 538, cx, 522))
    p.append(svg_text(cx, 556, "attention: 15 query heads (top) share 5 key/value heads (bottom)", 12, "middle", "600", INK))
    tw, tg, gg = 20, 3, 14
    gw = 3 * tw + 2 * tg
    x0 = cx - (5 * gw + 4 * gg) / 2
    for h in range(HEADS):
        p.append(tile(x0 + (h // 3) * (gw + gg) + (h % 3) * (tw + tg), 566, tw, 24, heads[h]))
    for k in range(5):
        p.append(tile(x0 + k * (gw + gg), 596, gw, 16, kv[k], "kv"))
    sub = lambda c: f'<tspan dy="3" font-size="9">{c}</tspan><tspan dy="-3"> </tspan>'  # noqa: E731
    p.append(svg_text(cx, 634, f"head h: softmax(q{sub('h')}&#183; k{sub('g')}/ &#8730;64) &#183; v{sub('g')},  "
                      "g = &#8970;h / 3&#8971;;  64 values per head", 11.5, "middle"))
    p.append(svg_text(cx, 651, "q and k rotated by position (RoPE)", 11, "middle"))
    p.append(box(125, 672, 120, 32, "q_proj", "[960 &#215; 720]"))
    p.append(box(260, 672, 120, 32, "k_proj", "[320 &#215; 720]*"))
    p.append(box(395, 672, 120, 32, "v_proj", "[320 &#215; 720]*"))
    for xc in (185, 320, 455):
        p.append(arrow(xc, 672, xc, 660))
    p.append(path(f"M{cx} 744V714M185 714H455", head=False))
    for xc in (185, 320, 455):
        p.append(arrow(xc, 714, xc, 704))
    p.append(arrow(612, 644, 528, 644, "#9a6b1f", "5 4"))
    p.append(box(165, 744, 310, 24, "RMSNorm"))
    # residual around the attention
    p.append(f'<circle cx="{cx}" cy="780" r="2.6" fill="{GREY}"/>')
    p.append(path(f"M{cx} 780H96V420H{cx - 11}"))
    p.append(arrow(cx, 826, cx, 768))
    # notes on the right
    ax = 622
    p.append(svg_text(ax, 112, "Feed-forward (SwiGLU)", 13, weight="600", color=INK))
    p.append(svg_lines(ax, 131, ["down_proj(SiLU(gate_proj&#8201;x) &#215; up_proj&#8201;x). A block of", "64 neurons reaches the "
                                 "stream only through its own", "64 columns of down_proj: it is one unit, 32 per layer."]))
    p.append(svg_text(ax, 452, "Attention", 13, weight="600", color=INK))
    p.append(svg_lines(ax, 471, ["A query head reaches the stream only through its", "own 64 columns of o_proj: it is one unit, "
                                 "15 per layer.", "Key/value heads are not units: each serves three", "query heads, and is frozen "
                                 "with any core head that", "reads it (padlock)."]))
    p.append(svg_text(ax, 572, "ContextGate (ours)", 13, weight="600", color=GATE_INK))
    p.append(svg_lines(ax, 591, ["Multiplies each unit's 64 values by 1 or 0, as", "the instruction says. Everything else "
                                 "is SmolVLA."], color=GATE_INK))
    p.append(svg_lines(ax, 648, ["Dashed: keys and values of the VLM's tokens. Even", "layers attend to them together with the "
                                 "action tokens;", "in odd layers k_proj and v_proj (*) read them", "instead of x and are "
                                 "[320 &#215; 320]."], color="#8a5d12"))
    p.append(svg_text(ax, 748, "15 + 32 = 47 units per layer, 752 in 16 layers", 13, weight="600", color=INK))
    # legend
    lx, ly = 22, 868
    for task, label in ((1, "core of task 1 (pushing)"), (2, "core of task 2 (red on green)"), (None, "not reserved: learns task 3")):
        p.append(tile(lx, ly - 11, 13, 13, task) + svg_text(lx + 19, ly, label, 12))
        lx += 34 + 6.3 * len(label)
    p.append(tile(lx, ly - 11, 26, 13, 1, "kv") + svg_text(lx + 32, ly, "key/value head frozen with a core", 12))
    p.append("</svg>")
    return "".join(p)


def packnet_svg(core, mask):
    """The pushing task's 5% in one layer under the gate (whole units) and under PackNet (single weights), seed 0."""
    heads, blocks, _ = layer_owners(core, SHOW_LAYER)
    q, o = mask["self_attn.q_proj"], mask["self_attn.o_proj"]
    gate_w, up, down = mask["mlp.gate_proj"], mask["mlp.up_proj"], mask["mlp.down_proj"]
    head_share = [(q[64 * h:64 * h + 64].sum() + o[:, 64 * h:64 * h + 64].sum()) / (2 * 64 * q.shape[1]) for h in range(HEADS)]
    block_share = [(gate_w[64 * b:64 * b + 64].sum() + up[64 * b:64 * b + 64].sum() + down[:, 64 * b:64 * b + 64].sum())
                   / (3 * 64 * gate_w.shape[1]) for b in range(BLOCKS)]
    n_weights = sum(v.size for v in mask.values())
    rows, cols = 256, 192  # the q_proj rows of heads 0-3 and its first 192 inputs, one pixel per weight
    blue = tuple(int(TASK_COLORS[1][i:i + 2], 16) for i in (1, 3, 5))
    img = np.full((rows, cols, 3), 244, np.uint8)
    img[q[:rows, :cols] == 1] = blue
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG", optimize=True)
    crop = data_uri(buf.getvalue(), "image/png")
    p = ['<svg viewBox="0 0 1000 400" role="img" aria-label="The pushing task\'s 5% of one layer under consolidate-and-gate '
         'and under PackNet">']
    def unit_tile(x, y, w, h, gated, share):
        if gated:
            return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="{TASK_COLORS[1]}"/>' if share else
                    f'<rect x="{x + 0.5}" y="{y + 0.5}" width="{w - 1}" height="{h - 1}" rx="2" fill="{OFF}" '
                    f'stroke="#b9b8b2" stroke-dasharray="2 2"/>')
        bar = max(share * h, 0.8)
        return (f'<rect x="{x + 0.5}" y="{y + 0.5}" width="{w - 1}" height="{h - 1}" rx="2" fill="white" stroke="{GREY}"/>'
                f'<rect x="{x + 0.5}" y="{y + h - 0.5 - bar:.2f}" width="{w - 1}" height="{bar:.2f}" fill="{TASK_COLORS[1]}"/>')
    pct = lambda v: f"{100 * min(v):.1f}&#8211;{100 * max(v):.1f}%"  # noqa: E731
    panels = [("Consolidate-and-gate (ours): whole units", True), ("PackNet: single weights", False)]
    for k, (title, gated) in enumerate(panels):
        x0 = 20 + 500 * k
        p.append(svg_text(x0, 22, title, 14, weight="600", color=INK))
        p.append(svg_text(x0, 41, f"frozen for task 1 (pushing) in layer {SHOW_LAYER} of the action expert, seed 0", 12))
        mx, my = x0 + 48, 58
        if gated:
            p.append(f'<rect x="{mx}" y="{my}" width="{cols}" height="{rows}" fill="rgb(244,244,244)"/>')
            p.extend(f'<rect x="{mx}" y="{my + 64 * h}" width="{cols}" height="64" fill="{TASK_COLORS[1]}"/>'
                     for h in range(4) if heads[h] == 1)
        else:
            p.append(f'<image x="{mx}" y="{my}" width="{cols}" height="{rows}" href="{crop}" style="image-rendering:pixelated"/>')
        p.append(f'<rect x="{mx}" y="{my}" width="{cols}" height="{rows}" fill="none" stroke="{GREY}"/>')
        for h in range(4):
            p.append(svg_text(mx - 6, my + 64 * h + 36, f"head {h}", 11, "end"))
            if h:
                p.append(f'<path d="M{mx - 4} {my + 64 * h}H{mx + cols}" stroke="{GREY}" stroke-width="0.8" stroke-dasharray="3 3"/>')
        p.append(svg_lines(mx, my + rows + 17, [f"q_proj: rows of heads 0&#8211;3, first {cols} of", "720 inputs; one pixel per weight"],
                           size=11))
        rx = x0 + 262
        p.append(svg_text(rx, 70, "Attention heads", 12, weight="600", color=INK))
        p.extend(unit_tile(rx + h * 14, 80, 12, 26, gated, heads[h] == 1 if gated else head_share[h]) for h in range(HEADS))
        on = sum(t == 1 for t in heads)
        p.append(svg_lines(rx, 124, [f"{on} of 15 run for pushing,", "each with all of its weights"] if gated else
                           ["all 15 run for pushing, each", f"with {pct(head_share)} of its weights"]))
        p.append(svg_text(rx, 166, "MLP blocks", 12, weight="600", color=INK))
        for b in range(BLOCKS):
            p.append(unit_tile(rx + (b % 16) * 13, 176 + (b // 16) * 22, 11.5, 19, gated,
                               blocks[b] == 1 if gated else block_share[b]))
        on = sum(t == 1 for t in blocks)
        p.append(svg_lines(rx, 236, [f"{on} of 32 runs for pushing,", "with all of its weights"] if gated else
                           ["all 32 run for pushing, each", f"with {pct(block_share)} of its weights"]))
        p.append(svg_text(rx, 282, "Stored to run the task", 12, weight="600", color=INK))
        p.append(svg_lines(rx, 300, ["one bit per unit and task: 47 bits", "for this layer, 752 for the network"] if gated else
                           ["an owner for every weight: " f"{n_weights / 1e6:.1f}M", "for this layer, 99.9M for the network"]))
    ly = 382
    p.append(f'<rect x="20" y="{ly - 11}" width="13" height="13" rx="2" fill="{TASK_COLORS[1]}"/>'
             + svg_text(39, ly, "frozen for task 1 and used for pushing", 12))
    p.append(unit_tile(300, ly - 11, 13, 13, True, False) + svg_text(319, ly, "switched off for pushing", 12))
    p.append(unit_tile(520, ly - 13, 13, 16, False, 0.3) + svg_text(539, ly, "runs for pushing; blue: share of its weights "
                                                                       "frozen for task 1", 12))
    p.append("</svg>")
    return "".join(p)


# ----------------------------------------------------------------------------------------------- media

def data_uri(raw, mime):
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def video_uri(path, size=240, crf=30):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "clip.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", f"scale={size}:{size}", "-c:v", "libx264",
                        "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-an", out], check=True)
        return data_uri(open(out, "rb").read(), "video/mp4")


def jpeg_uri(array, quality=86):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="JPEG", quality=quality)
    return data_uri(buf.getvalue(), "image/jpeg")


def episode_kinds(stage_dir, task):
    """Per test episode of the recorded evaluation: success (completed, and still completed when the episode
    ends), wrong (built the other stack at some point, never the requested one), failure (neither), the step of first
    success and the number of lapses after it (steps where the success condition stopped holding)."""
    rows = [r for r in map(json.loads, open(os.path.join(stage_dir, f"evals_{VIDEO_TAG}.jsonl"))) if r["task"] == task]
    r = rows[-1]
    opposite = r.get("opposite_episodes", [False] * r["n"])
    opposite_end = r.get("opposite_episodes_end", [False] * r["n"])
    kinds = []
    for i, trace in enumerate(r["traces"]):
        first = trace.find("1")
        kinds.append({"success": r["episodes_end"][i], "wrong": opposite[i] and not r["episodes"][i],
                      "wrong_end": opposite_end[i], "failure": not r["episodes"][i] and not opposite[i],
                      "first": first if first >= 0 else len(trace),
                      "lapses": trace[first:].count("10") if first >= 0 else 0})
    return kinds


def stack_visible(stage_dir, task, index, top_first):
    """Whether the final frame of an episode's video shows both cubes stacked, the expected one on top: red
    and green pixels each above a small count, horizontally aligned, the top cube's pixels higher."""
    path = glob.glob(os.path.join(stage_dir, f"videos_{VIDEO_TAG}", f"{task}_seed{FIRST_TEST_SEED + index}_*.mp4"))[0]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.06", "-i", path, "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, stdout=subprocess.PIPE).stdout
    side = int(round((len(raw) / 3) ** 0.5))  # square frames
    a = np.frombuffer(raw, np.uint8).reshape(side, side, 3).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red, green = (r > 150) & (g < 60) & (b < 60), (g > 120) & (r < 90) & (b < 90)
    if red.sum() < 10 or green.sum() < 10:
        return False
    (red_row, red_col), (green_row, green_col) = np.argwhere(red).mean(0), np.argwhere(green).mean(0)
    if abs(red_col - green_col) > 6:
        return False
    return red_row < green_row if top_first == "red" else green_row < red_row


def choose(runs, task, wants, stages):
    """Episode index per method: one scene satisfying every method's intended outcome if possible, else the
    best episode per method; successes with the fewest lapses and the earliest completion are preferred.
    Returns (indices, same_scene)."""
    kinds = {m: episode_kinds(os.path.join(runs, VIDEO_SEED, stages[m]), task) for m in wants}
    n = len(next(iter(kinds.values())))
    top = {STACK: "red", GREEN: "green"}.get(task)
    hidden = {}
    def unseen(m, i):  # stacking examples: the intended stack (or the wrong one) should be visible at the end
        if top is None or wants[m] not in ("success", "wrong"):
            return 0
        if (m, i) not in hidden:
            other = "green" if top == "red" else "red"
            standing = wants[m] == "success" or kinds[m][i]["wrong_end"]  # a wrong stack still standing at the end
            hidden[m, i] = not (standing and stack_visible(os.path.join(runs, VIDEO_SEED, stages[m]), task, i,
                                                           top if wants[m] == "success" else other))
        return int(hidden[m, i])
    def lateness(i, ms):
        ok = [m for m in ms if wants[m] == "success"]
        return (sum(unseen(m, i) for m in ms), sum(kinds[m][i]["lapses"] for m in ok),
                max([kinds[m][i]["first"] for m in ok] or [0]))
    separate = {m: min((i for i in range(n) if kinds[m][i][w]), key=lambda i: (lateness(i, [m]), i))
                for m, w in wants.items()}
    shared = [i for i in range(n) if all(kinds[m][i][w] for m, w in wants.items())]
    if shared:  # one scene for every method, unless that costs a visible stack that separate scenes give
        i = min(shared, key=lambda i: (lateness(i, wants), i))
        if lateness(i, wants)[0] <= sum(lateness(separate[m], [m])[0] for m in wants):
            return {m: i for m in wants}, True
    return separate, False


def clip(runs, stage, task, index, label, cache, step=""):
    stage_dir = os.path.join(runs, VIDEO_SEED, stage)
    path = glob.glob(os.path.join(stage_dir, f"videos_{VIDEO_TAG}", f"{task}_seed{FIRST_TEST_SEED + index}_*.mp4"))[0]
    if path not in cache:
        cache[path] = video_uri(path)
    kind = episode_kinds(stage_dir, task)[index]
    badge = ('<span class="badge ok">success</span>' if kind["success"] else
             '<span class="badge no">wrong stack</span>' if kind["wrong"] else '<span class="badge no">failure</span>')
    step = f'<div class="step">{step}</div>' if step else ""
    return (f'<div class="clip">{step}<video autoplay muted loop playsinline preload="auto" src="{cache[path]}"></video>'
            f'<div class="label"><b>{label}</b>{badge}</div></div>')


def video_grid(runs, stage_set, stages, names, cache):
    out, separate = [], []
    for task, wants in EXAMPLES[stage_set]:
        wants = {m: w for m, w in wants.items() if m in stages}  # methods whose rollouts are recorded
        chosen, same = choose(runs, task, wants, stages)
        if not same:
            separate.append(TASK_NAMES[task])
        out.append(f'<div class="rowlabel">Instruction: <q>{TASK_NAMES[task]}</q></div>')
        out += [clip(runs, stages[m], task, chosen[m], names.get(m, NAMES.get(m)), cache) for m in wants]
    return "".join(out), separate


C_STAGE = {"gate_k5c": "C_gate_rand_k5_c2000", "gate_k5": "C_gate_rand_k5"}


def recorded(runs, stage):
    """Whether every task's test episodes of a stage are recorded with per-step success (clips are chosen by it)."""
    path = os.path.join(runs, VIDEO_SEED, stage or "", f"evals_{VIDEO_TAG}.jsonl")
    return bool(stage) and os.path.exists(path) and \
        {r["task"] for r in map(json.loads, open(path)) if "traces" in r} >= {PUSH, STACK, GREEN}


def video_stage(runs, m):
    """Task-3 stage whose recorded test episodes show the gate: the reported configuration once recorded, else 25% cores."""
    stage = C_STAGE.get(m)
    return (stage, m) if recorded(runs, stage) else ("C_gate_rand", "gate_rand")


def teaser(runs, cache, gate_stage):
    stage = os.path.join(runs, VIDEO_SEED, gate_stage)
    push = episode_kinds(stage, PUSH)
    red, green = episode_kinds(stage, STACK), episode_kinds(stage, GREEN)
    def best(task, kinds):
        top = {STACK: "red", GREEN: "green"}.get(task)
        visible = lambda i: top is None or stack_visible(stage, task, i, top)  # noqa: E731
        return min((i for i, k in enumerate(kinds) if k["success"]),
                   key=lambda i: (not visible(i), kinds[i]["lapses"], kinds[i]["first"], i))
    clips = [clip(runs, gate_stage, task, best(task, kinds), f"<q>{TASK_NAMES[task]}</q>", cache, step)
             for task, kinds, step in ((PUSH, push, "Task 1 &middot; learned first"), (STACK, red, "Task 2 &middot; learned second"),
                                      (GREEN, green, "Task 3 &middot; learned last"))]
    then = '<div class="then"><b>&rarr;</b>then</div>'
    return then.join(clips)


def task_frames(data):
    def replay(task):
        prefix = os.path.join(data, "replay", task, "trajectory.rgb.pd_ee_delta_pos.physx_cpu")
        episodes = {e["episode_seed"]: e for e in json.load(open(prefix + ".json"))["episodes"] if e["success"]}
        return h5py.File(prefix + ".h5", "r"), episodes
    push, push_eps = replay(PUSH)
    red, red_eps = replay(STACK)
    green, green_eps = replay(GREEN)
    p = push[f"traj_{push_eps[min(push_eps)]['episode_id']}"]["obs/sensor_data/base_camera/rgb"]
    # A few published StackCube demonstrations keep scenes that this ManiSkill version no longer draws from
    # their seed; take the first seed whose two demonstrations start from the same scene (up to rendering noise).
    for seed in sorted(set(red_eps) & set(green_eps)):
        r = red[f"traj_{red_eps[seed]['episode_id']}"]["obs/sensor_data/base_camera/rgb"]
        g = green[f"traj_{green_eps[seed]['episode_id']}"]["obs/sensor_data/base_camera/rgb"]
        if np.abs(r[0].astype(int) - g[0].astype(int)).max() <= 2:
            break
    frames = [(p[0], "PushCube: start"), (p[-1], "PushCube: cube on target"),
              (r[0], "Both stacking tasks: start"), (r[-1], "Red cube on green"), (g[-1], "Green cube on red")]
    return "".join(f'<figure><img src="{jpeg_uri(f)}" alt="{c}"><figcaption>{c}</figcaption></figure>'
                   for f, c in frames)


# ----------------------------------------------------------------------------------------------- tables

def config_name(m):
    """The configuration part of a method name (after the method itself)."""
    parts = NAMES[m].split(", ", 1)
    return parts[1] if len(parts) > 1 else "as in the two-task study"


def two_memory(N):
    """What each method stores for the old task in the two-task study."""
    return {"seqft": "none", "ewc": "Fisher + old weights (0.8 GB)", "er50": "9.4 MiB", "er500": "94 MiB",
            N["sel"].get("packnet"): "an owner per weight", N["sel"].get("gate"): "a unit mask per task",
            "joint": "all PushCube data"}


def two_table(N):
    """Two-task study, one row per method; the gate and PackNet at their reported configuration (5% per task)."""
    memory, rows = two_memory(N), []
    for m in ("seqft", "ewc", "er50", "er500", N["sel"].get("packnet"), N["sel"].get("gate"), "joint"):
        t = N["two"].get(m, {})
        if not m or not t.get("push"):
            continue
        cls = ' class="ours"' if m == N["sel"].get("gate") else ""
        name = (f'<span class="swatch" style="background:{GREY if m == "joint" else COLORS[m]}"></span>'
                + label(m, t["push"], N["total"]))
        name = f"<i>{name} (reference: all data at once, 20500 updates)</i>" if m == "joint" else name
        rows.append(f'<tr{cls}><td>{name}</td><td>{pm(t.get("push_before"))}</td>'
                    f'<td>{pm(t["push"])}</td><td>{pm(t["stack"])}</td><td>{pm(t["avg"])}</td><td class="cfg">{memory[m]}</td></tr>')
    return ('<table><thead><tr><th>Method</th><th>Pushing before<br>stacking</th><th>Pushing after<br>stacking</th><th>Stacking</th>'
            '<th>Average</th><th>Stored for the old task</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def main_three_table(N):
    """One row per method: its selected configuration (best three-task average on the test scenes, all seeds)."""
    rows = []
    for fam in ("seqft", "ewc", "er50", "er500", "packnet", "gate"):
        m = N["sel"].get(fam)
        if m is None:
            continue
        t = N["three"][m]
        cls = ' class="ours"' if fam == "gate" else ""
        rows.append(f'<tr{cls}><td><span class="swatch" style="background:{COLORS[m]}"></span>{FAMILY_NAMES[fam]}'
                    f'{" (ours)" if fam == "gate" else ""}</td><td class="cfg">{config_name(m)}</td><td>{pm(t["push"])}</td>'
                    f'<td>{pm(t["rog"])}</td><td>{pm(t["rog_wrong"])}</td><td>{pm(t["gor"])}</td><td>{pm(t["avg"])}</td></tr>')
    t = N["three"]["joint"]
    if t["push"]:  # reference, not a continual learner
        rows.append(f'<tr><td><i>{label("joint", t["push"], N["total"])} (reference: all data at once)</i></td><td class="cfg">all '
                    f'three tasks mixed in every batch, 32500 updates</td><td>{pm(t["push"])}</td><td>{pm(t["rog"])}</td><td>{pm(t["rog_wrong"])}</td>'
                    f'<td>{pm(t["gor"])}</td><td>{pm(t["avg"])}</td></tr>')
    return ('<table><thead><tr><th>Method</th><th>Selected configuration</th><th>Pushing</th><th>Red on green</th>'
            '<th>&hellip;built green on<br>red instead</th><th>Green on red</th><th>Average<br>of three</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def curves_text(N):
    """Heading and caption of the stage-B learning curves."""
    if not any(c[PUSH] for c in N["two"]["gate_k5c_curve"]["curve"]):
        return "Fine-tuning forgets within the first 2000 updates", ("Pushing collapses for fine-tuning and EWC before the first "
                                                                     "check and declines gradually under replay; the gated policy's curve (5% cores) is running.")
    if not has_early(N):
        return "Fine-tuning forgets within the first 2000 updates", ("Pushing collapses for fine-tuning and EWC before the first "
                                                                     "check and declines gradually under replay; the gated policy's pushing does not move.")
    push = {m: mean_curve(stage_curves(N, m), PUSH) for m in CURVE_METHODS}
    def below(m, level=10):  # first check with pushing below the level
        return next((s for s, y in zip(*push[m]) if y < level), None)
    def span(m):
        ys = push[m][1]
        return f"stays at {ys[0]:.0f}%" if round(min(ys)) == round(max(ys)) else f"ranges from {min(ys):.0f}% to {max(ys):.0f}%"
    def at(points, step):
        return next((p["success"] for p in points if p["step"] == step), None)
    same = all(at(e[t], 2000) == at(c[t], 2000) for m in CURVE_METHODS
               for c, e in zip(N["two"][m]["curve"], N["two"][m]["curve_early"]) for t in (PUSH, STACK))
    ft, ewc = below("seqft"), below("ewc")
    heading = f"Fine-tuning forgets within the first {ft} updates" if ft else "Fine-tuning forgets pushing"
    if ft and ft == ewc:
        lost = f"for fine-tuning and EWC by {ft} updates"
    else:
        lost = " and ".join(f"for {name} by {s} updates" if s else f"never for {name}" for name, s in (("fine-tuning", ft), ("EWC", ewc)))
    warm = f", while the learning rate is still {100 * ft / 500:.0f}% of its peak" if ft and ft < 500 else ""
    steps, ys = push["er50"]
    k = min(range(len(ys)), key=lambda i: ys[i])
    back = next(((s, y) for s, y in zip(steps[k:], ys[k:]) if y >= 0.9 * ys[0]), None)
    replay = (f"under replay (ER-50) it drops to {ys[k]:.0f}% at {steps[k]} updates"
              + (f" and is back at {back[1]:.0f}% by {back[0]}" if back else ""))
    return heading, (f"Before 2000 updates the points come from a second run of each method's first 2000 updates (same seed, settings and GPU "
                     f"type{'; it scores exactly like the original run at 2000 updates in every seed' if same else ''}), "
                     f"scored at 0, 50, 100, 250, 500 and 1000 updates (pushing for fine-tuning, EWC and ER-50 also at 10, 20 and 30); "
                     f"the learning rate warms up over the first 500. Pushing, at {push['seqft'][1][0]:.0f}% before the stage, falls below "
                     f"10% {lost}{warm}; {replay}; for the gated policy it {span('gate_k5c_curve')}.")


def freeze_table(N):
    g, nf = N["two"][N["sel"]["gate"]], N["two"]["gate_nofreeze_k5c"]
    cell = lambda v: mean(v) + "%" + (f" ({len(v)} of {N['total']})" if len(v) < N["total"] else "") if v else "running"  # noqa: E731
    return ('<table class="grid2x2"><thead><tr><th></th><th>Gate on</th><th>Gate off</th></tr></thead><tbody>'
            f'<tr><th>Core frozen during B</th><td class="hi">{cell(g["push"])}</td><td>{cell(g["gate_off"])}</td></tr>'
            f'<tr><th>Core not frozen</th><td>{cell(nf["push"])}</td><td>{cell(nf["gate_off"])}</td></tr></tbody></table>'
            f'<p style="font-size:0.85rem;color:var(--fg2);text-align:center;margin-top:8px">Pushing success after stage B with 5% cores, '
            f'100 test scenes, mean of {N["total"]} seeds unless stated.</p>')


def freeze_text(N):
    g, nf = N["two"][N["sel"]["gate"]], N["two"]["gate_nofreeze_k5c"]
    text = (f"With the core frozen, switching the gate off when pushing is tested drops pushing from {mean(g['push'])}% to "
            f"{mean(g['gate_off'])}%: the units trained for stacking interfere." if g["gate_off"] else "")
    if nf["push"] and nf["gate_off"]:
        n = f" ({len(nf['push'])} of {N['total']} seeds)" if len(nf["push"]) < N["total"] else ""
        text += (f" Without freezing, stage B rewrites the core: pushing is {mean(nf['push'])}% with the gate on and "
                 f"{mean(nf['gate_off'])}% with it off{n}.")
    return text


def next_steps(N):
    """Two preliminary results: a core chosen by intervention instead of at random (5%, two tasks), and releasing part
    of the earlier allocations when two 50% allocations have filled the network."""
    r, a = N["two"][N["sel"]["gate"]], N["two"]["gate_abl_k5c"]
    seeds = lambda v: "/".join(f"{x:.0f}" for x in v)  # noqa: E731
    core = (f"Cores are chosen at random. A 5% core chosen by intervention instead, the unit groups whose silencing hurts pushing "
            f"most, raises the two-task average from {mean(r['avg'])}% to {mean(a['avg'])}% (seeds 0/1/2: {seeds(r['avg'])}% and "
            f"{seeds(a['avg'])}%), with pushing at {mean(a['push'])}% and stacking at {mean(a['stack'])}% against {mean(r['push'])}% "
            f"and {mean(r['stack'])}%. Choosing each task's units by how much the task relies on them could improve the method further."
            if len(a["avg"]) == len(r["avg"]) == N["total"] else "")
    t = {m: N["three"][m] for m in ("gate_k50_r0", "packnet_k50_r0", "gate_k50_r50rg", "packnet_k50_r50")}
    if not all(len(v["avg"]) == N["total"] for v in t.values()):
        return core, ""
    g0, p0, g, pk = t["gate_k50_r0"], t["packnet_k50_r0"], t["gate_k50_r50rg"], t["packnet_k50_r50"]
    capacity = (f"Each task reserves a fixed share, so the network holds a fixed number of tasks. In a first test at 50% per task, "
                f"the first two tasks fill the network: with nothing released, the third task cannot be learned (average over the "
                f"three tasks {mean(g0['avg'])}% for the gate, {mean(p0['avg'])}% for PackNet, which is ahead at large allocations as "
                f"above). Releasing half of each earlier share lets it in: the gate gives up its least important unit groups, "
                f"keeps pushing at {mean(g['push'])}% and learns green on red to {mean(g['gor'])}%; PackNet gives up its smallest "
                f"weights and learns green on red to {mean(pk['gor'])}%, but its pushing falls to {mean(pk['push'])}%. Both give up "
                f"red on green, so a release that keeps the earlier skills, for example by consolidating each one into the units it "
                f"keeps, is a next step.")
    return core, capacity


def next_core_table(N):
    """Random and intervention-selected 5% cores, both consolidated for 2000 updates, after stage B."""
    r, a = N["two"][N["sel"]["gate"]], N["two"]["gate_abl_k5c"]
    if not a["push"]:
        return ""
    return ('<div class="tablewrap" style="max-width:620px;margin:14px auto 0"><table><thead><tr><th>5% core</th><th>Pushing</th>'
            '<th>Stacking</th><th>Average</th></tr></thead><tbody>'
            f'<tr><td>Random (reported)</td><td>{pm(r["push"])}</td><td>{pm(r["stack"])}</td><td>{pm(r["avg"])}</td></tr>'
            f'<tr><td>Selected by intervention</td><td>{pm(a["push"])}</td><td>{pm(a["stack"])}</td><td>{pm(a["avg"])}</td></tr>'
            '</tbody></table><p style="font-size:0.85rem;color:var(--fg2);margin-top:8px">Two-task study, after stage B; 100 test '
            f'scenes, mean &plusmn; SD over {len(a["push"])} seeds.</p></div>')


def next_capacity_table(N):
    """Task 3 after two 50% allocations filled the network, for both methods: nothing released, or half of each earlier
    allocation released (the earlier tasks stop using it)."""
    rows = []
    for m, method, what in (("gate_k50_r0", "Consolidate-and-gate", "nothing"),
                            ("packnet_k50_r0", "PackNet", "nothing"),
                            ("gate_k50_r50rg", "Consolidate-and-gate", "half of each earlier core (its least important groups)"),
                            ("packnet_k50_r50", "PackNet", "half of each earlier task's weights (its smallest)")):
        t = N["three"].get(m)
        if t and t["push"]:
            rows.append(f'<tr{" class=ours" if method.startswith("Consolidate") else ""}><td>{method}</td><td>{what}</td>'
                        f'<td>{pm(t["push"])}</td><td>{pm(t["rog"])}</td><td>{pm(t["gor"])}</td><td>{pm(t["gor_wrong"])}</td>'
                        f'<td>{pm(t["avg"])}</td></tr>')
    if not rows:
        return ""
    return ('<div class="tablewrap" style="max-width:1000px;margin:14px auto 0"><table><thead><tr><th>Method</th>'
            '<th>Released for task 3</th><th>Pushing</th><th>Red on green</th><th>Green on red<br>(task 3)</th>'
            '<th>&hellip;built red on<br>green instead</th><th>Average</th></tr></thead><tbody>' + "".join(rows)
            + '</tbody></table><p style="font-size:0.85rem;color:var(--fg2);margin-top:8px">The first two tasks take half of the '
            'network each (50% per task), so the third finds nothing free. After task 3; 100 test scenes, mean &plusmn; SD over '
            f'{N["total"]} seeds. Released: chosen by each method\'s own measure (the gate: the unit groups whose silencing hurts '
            'their task least; PackNet: the smallest weights), retrained for task 3 and no longer used by the task that gave them up. '
            f'Red on green before task 3: {mean(N["three"]["gate_k50_r0"]["rog_before"])}% for the gate and '
            f'{mean(N["three"]["packnet_k50_r0"]["rog_before"])}% for PackNet.</p></div>')


def seeds_note(*values):
    """' (2 of 3 seeds)' style note for averages over an incomplete set of seeds."""
    counts = sorted({len(v) for v in values})
    return "" if counts == [3] else f" ({' and '.join(str(c) for c in counts)} of 3 seeds)"


def frozen(N, m, when="after_1"):
    v = N["protected"][m][when]
    return f"{mean_sd(v)[0]:.1f}" if v else "&mdash;"


def budget_summary(N):
    """Which method is ahead after three tasks at each share (short), and the numbers (long)."""
    frozen_pairs = ", ".join(f"{frozen(N, g)}% vs {frozen(N, p)}% at {share:g}%" for share, g, _, p, _ in CURVE[1:])
    ahead, parts = {"gate": [], "packnet": []}, []
    for share, _, g3, _, p3 in CURVE:
        g, p = N["three"].get(g3, {}).get("avg", []), N["three"].get(p3, {}).get("avg", [])
        if not (g and p):
            continue
        fam = "gate" if mean_sd(g)[0] > mean_sd(p)[0] else "packnet"
        ahead[fam].append(f"{share:g}%")
        parts.append(f"{share:g}%: {mean(g)}% vs {mean(p)}%{seeds_note(g, p)}")
    def join(xs):
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    short = "; ".join(s for s in (f"the gate ahead at {join(ahead['gate'])} per task" if ahead["gate"] else "",
                                   f"PackNet at {join(ahead['packnet'][::-1])}" if ahead["packnet"] else "") if s)
    long = (f"After three tasks the gate is ahead at {join(ahead['gate'])} per task and PackNet at "
            f"{join(ahead['packnet'][::-1])} (gate vs PackNet, three-task average: {'; '.join(parts)}). Pushing, scored right after "
            f"its own stage, separates the methods most: as the share shrinks from 10% to 1% the gate keeps "
            f"{mean(N['budget']['gate_k10c'])}% &rarr; {mean(N['budget']['gate_k1c'])}% and PackNet {mean(N['budget']['packnet_k10sp'])}% "
            f"&rarr; {mean(N['budget']['packnet_k1sp'])}%. The nominal shares count unit groups for the gate and weights for PackNet, "
            f"and at each share below 25% the gate freezes slightly more weights after task 1 ({frozen_pairs}); the gate at 1% still "
            f"keeps more pushing ({mean(N['budget']['gate_k1c'])}% with {frozen(N, 'gate_k1c')}% of the weights frozen) than PackNet at "
            f"2.5% ({mean(N['budget']['packnet_k2p5sp'])}% with {frozen(N, 'packnet_k2p5sp')}%).")
    return short, long


# (share per task, our budget key, our three-task key, PackNet budget key, PackNet three-task key)
CURVE = [(25, "gate_bundle", "gate_bundle_L24k_brC", "packnet", "packnet_L24k"),
         (10, "gate_k10c", "gate_k10c", "packnet_k10sp", "packnet_k10sp"),
         (5, "gate_k5c", "gate_k5c", "packnet_k5sp", "packnet_k5sp"),
         (2.5, "gate_k2p5c", "gate_k2p5c", "packnet_k2p5sp", "packnet_k2p5sp"),
         (1, "gate_k1c", "gate_k1c", "packnet_k1sp", "packnet_k1sp")]


def chart_budget(N):
    """Pushing right after its own stage, and the three-task average, against the share of the network per task."""
    shares = CURVE
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    series = (("Consolidate-and-gate (ours)", COLORS["gate_k5c"], 1, 2), ("PackNet", COLORS["packnet_k5sp"], 3, 4))
    for ax, (title, which) in zip(axes, (("Pushing, right after its own stage", 0), ("Average after three tasks", 1))):
        for name, color, kb, k3 in series:
            pts = []
            for c in shares:
                v = N["budget"].get(c[kb], []) if which == 0 else N["three"].get(c[k3], {}).get("avg", [])
                if v:
                    pts.append((c[0], v))
            if not pts:
                continue
            xs, stats = [p[0] for p in pts], [mean_sd(p[1]) for p in pts]
            ax.plot(xs, [m for m, _ in stats], "-", color=color, lw=2, zorder=2)
            for (x, v), (m, sd) in zip(pts, stats):
                full = len(v) == N["total"]
                ax.errorbar(x, m, yerr=sd, fmt="o", ms=8, color=color, mfc=color if full else "white", mew=2, capsize=3, zorder=3)
        if which == 1 and N["three"]["joint"]["avg"]:  # reference: all three tasks' data at once
            ax.axhline(mean_sd(N["three"]["joint"]["avg"])[0], color=GREY, lw=1.5, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([c[0] for c in shares])
        ax.set_xticklabels([f"{c[0]:g}%" for c in shares])
        ax.minorticks_off()
        ax.set_xlim(max(c[0] for c in shares) * 1.5, min(c[0] for c in shares) / 1.5)  # smaller shares to the right
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("Share of the network per task (nominal)")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Success, test scenes (%)")
    handles = [matplotlib.lines.Line2D([], [], color=c, marker="o", ms=8, lw=2, label=n) for n, c, _, _ in series]
    if N["three"]["joint"]["avg"]:
        handles.append(matplotlib.lines.Line2D([], [], color=GREY, lw=1.5, ls="--", label="Joint training, three tasks (reference)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.07), fontsize=9.5)
    fig.tight_layout(w_pad=2.5)
    return to_svg(fig, "budget")


def readme_block(N):
    """The README's results table: every method after all three tasks, the gate and PackNet at 5% per task."""
    def md(values):
        return pm(values).replace("&plusmn;", "±")
    rows = ["PushCube, then StackCube (red cube on green), then the opposite stack in the same scenes. Success rate (%) after all "
            f"three tasks on 100 test scenes per task, mean ± SD over {N['total']} training seeds. Consolidate-and-gate and PackNet "
            "reserve 5% of the network per task; each method is shown at its best three-task configuration, and replay stores "
            "examples of each earlier task:", "",
            "| Method | Configuration | Pushing | Red on green | … built green on red instead | Green on red (new) | Average |",
            "|---|---|---|---|---|---|---|"]
    for fam in ("seqft", "ewc", "er50", "er500", "packnet", "gate"):
        m = N["sel"].get(fam)
        if m is None:
            continue
        t = N["three"][m]
        name = {"er50": "Experience replay, 50 + 50 examples", "er500": "Experience replay, 500 + 500 examples"}.get(fam, FAMILY_NAMES[fam])
        cells = [config_name(m).replace("&times;", "×"), md(t["push"]), md(t["rog"]), md(t["rog_wrong"]), md(t["gor"]), mean(t["avg"])]
        rows.append(f"| {'**' + name + '**' if fam == 'gate' else name} | " + " | ".join(cells) + " |")
    t = N["three"]["joint"]
    if t["push"]:  # reference, not a continual learner
        name = label("joint", t["push"], N["total"], ours=False)
        rows.append(f"| *{name} (reference: all data at once)* | all three tasks mixed in every batch, 32500 updates | " + " | ".join(
            [md(t["push"]), md(t["rog"]), md(t["rog_wrong"]), md(t["gor"]), mean(t["avg"])]) + " |")
    return "\n".join(rows)


def update_readme(path, N):
    text = open(path).read()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    assert start in text and end in text, f"{path} needs {start} and {end} markers"
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    new = f"{head}{start}\n{readme_block(N)}\n{end}{tail}"  # built before the file is opened for writing
    with open(path + ".tmp", "w") as f:
        f.write(new)
    os.replace(path + ".tmp", path)
    print(f"updated {path}")


# ----------------------------------------------------------------------------------------------- page

def build(args):
    results = json.load(open(args.results))
    N = collect(results)
    two, three = N["two"], N["three"]
    G, P = N["sel"]["gate"], N["sel"]["packnet"]
    OURS.add(G)
    g3, p3 = three[G], three[P]
    q3, qp3 = three["gate_bundle_L24k_brC"], three["packnet_L24k"]  # a quarter of the network per task
    vstage, vm = video_stage(args.runs, G)
    s3, e3, r3 = three[N["sel"]["seqft"]], three[N["sel"]["er50"]], three[N["sel"]["er500"]]
    sl = N["sliced"].get("", {})
    # seed-0 cores of the reported configuration and PackNet's task-1 weight owners at the same share, for the layer schematics
    core0 = json.load(open(os.path.join(args.runs, "s0", "C_gate_rand_k5_c2000", "core.json")))
    packnet_map = os.path.join(args.runs, "s0", "A_packnet_k5_r2000_sp", "packnet.safetensors")
    cache = {}
    packnet_stage = "C_packnet_k5sp" if P == "packnet_k5sp" and recorded(args.runs, "C_packnet_k5sp") else None
    c_stages = {"seqft": "C_seqft", "er50": "C_er50", **({"packnet": packnet_stage} if packnet_stage else {}), "gate": vstage}
    c_videos, separate = video_grid(args.runs, "C", c_stages, {"gate": FAMILY_NAMES["gate"], "packnet": FAMILY_NAMES["packnet"]},
                                    cache)
    exec_section = ""
    ex_g, ex_e, ex_f = (N["exec"].get(k, {}) for k in ("B_gate_rand_k5_c2000", "B_er50", "B_seqft"))
    if all(str(n) in ex_g for n in (1, 2, 5, 10, 25, 50)) and all(str(n) in ex_e for n in (25, 50)) and ex_f:
        push = lambda d, n: f'{d[str(n)][PUSH]["any"]:.0f}'  # noqa: E731
        near = [ex_g[str(n)][PUSH]["any"] for n in (1, 2, 5, 10)]
        exec_section = ('<div class="prose"><h3 id="exec">Control: how many predicted actions to execute</h3><p>The policy predicts 50 actions and '
                        'executes the first 5 before looking again. Five gives the highest or near-highest success for every method. '
                        'Executing more of each chunk without looking lowers stacking and loses most finished stacks by the end '
                        'of the episode; executing fewer stacks less often but keeps more stacks. The gated policy\'s pushing stays at '
                        f'{min(near):.0f}–{max(near):.0f}% up to 10 actions per chunk and falls to {push(ex_g, 25)}% and {push(ex_g, 50)}% '
                        f'at 25 and 50 (ER-50: {push(ex_e, 25)}% and {push(ex_e, 50)}%); fine-tuning pushes in at most '
                        f'{max(v[PUSH]["any"] for v in ex_f.values()):.0f}% at any setting.</p></div>'
                        f'<figure class="chart">{chart_exec(N)}<figcaption>Seed 0 stage-B checkpoints, 100 test scenes per task '
                        'and setting.</figcaption></figure>')
    values = {
        "authors": html.escape(args.authors), "repo_url": args.repo_url, "repo_name": args.repo_url.rstrip("/").split("/")[-1],
        "n_test": "100", "n_seeds": str(len(two[G]["push"])),
        "seeds_three": f'{len(g3["push"])} seed{"s" if len(g3["push"]) > 1 else ""}',
        "seqft_push": mean(two["seqft"]["push"]), "gate_push": mean(two[G]["push"]),
        "er50_push": mean(two["er50"]["push"]), "gate_push_before": mean(two[G]["push_before"]),
        "gate_stack": mean(two[G]["stack"]), "seqft_stack": mean(two["seqft"]["stack"]),
        "gate_avg": mean(two[G]["avg"]), "er500_avg": mean(two["er500"]["avg"]),
        "three_gate_rog": mean(g3["rog"]), "three_seqft_rog": mean(s3["rog"]),
        "three_er50_rog": mean(e3["rog"]), "three_er500_rog": mean(r3["rog"]), "three_seqft_wrong": mean(s3["rog_wrong"]),
        "three_gate_gor": mean(g3["gor"]),
        "three_gate_avg": mean(g3["avg"]), "three_er50_avg": mean(e3["avg"]),
        "three_seqft_avg": mean(s3["avg"]), "three_gate_push": mean(g3["push"]),
        "three_packnet_avg": mean(p3["avg"]), "three_packnet_rog": mean(p3["rog"]), "three_packnet_gor": mean(p3["gor"]),
        "three_packnet_push": mean(p3["push"]),
        "gate_gap": f'{mean_sd(g3["avg"])[0] - mean_sd(p3["avg"])[0]:.1f}', "three_er500_avg": mean(r3["avg"]),
        "gate_gap_seeds": " / ".join(f"{g - p:+.1f}" for g, p in zip(g3["avg"], p3["avg"])),
        "q_gate_avg": mean(q3["avg"]), "q_packnet_avg": mean(qp3["avg"]),
        "k5_gate_push": mean(N["budget"]["gate_k5c"]), "k5_packnet_push": mean(N["budget"]["packnet_k5sp"]),
        "k5_packnet_pruned": mean(N["budget"]["packnet_k5r"]),
        "main_three_table": main_three_table(N), "two_table": two_table(N),
        "budget_hollow": "; hollow: fewer than three seeds so far" if any(
            0 < len(N["budget"].get(c[k], []) if k in (1, 3) else three.get(c[k], {}).get("avg", [])) < N["total"]
            for c in CURVE for k in (1, 2, 3, 4)) else "",
        "core_json_kb": f'{N["core_json_bytes"] / 1000:.1f}' if N["core_json_bytes"] else "&mdash;",
        "packnet_map_mb": f'{sl["expert_weights"] / 1e6:.1f}' if sl else "&mdash;",
        "er500_three_note": "" if three["ewc"]["push"] else ". EWC was run for two tasks only (its third task is running)",
        "packnet_clause": (", and PackNet, the closest earlier method, which keeps a fixed share of every weight matrix per task "
                           "(at the same 5% per task and the same retraining budget as our consolidation)" if p3["push"] else ""),
        "packnet_cost": (", one bit per weight and task (PackNet)" if p3["push"] else ""),
        "next_core": next_steps(N)[0], "next_capacity": next_steps(N)[1],
        "next_core_table": next_core_table(N), "next_capacity_table": next_capacity_table(N),
        "budget_summary": budget_summary(N)[0], "budget_text": budget_summary(N)[1],
        "k10_gate_avg": mean(three["gate_k10c"]["avg"]) + "%" + seeds_note(three["gate_k10c"]["avg"]),
        "k10_packnet_avg": mean(three["packnet_k10sp"]["avg"]) + "%" + seeds_note(three["packnet_k10sp"]["avg"]),
        "free_c": f'{100 - mean_sd(N["protected"]["gate_k5c"]["after_2"])[0]:.0f}' if N["protected"]["gate_k5c"]["after_2"] else "88",
        # weights frozen at the reported 5% configuration, after task 1 and after tasks 1 and 2
        **{f"p_{fam}_{i}": f'{mean_sd(N["protected"][m][k])[0]:.1f}' if N["protected"][m][k] else "&mdash;"
           for fam, m in (("gate", "gate_k5c"), ("packnet", "packnet_k5sp")) for i, k in ((1, "after_1"), (2, "after_2"))},
        "freeze_text": freeze_text(N),
        "budget_svg": chart_budget(N),
        "sequence_svg": chart_sequence(N), "cost_svg": chart_cost(N, memory_bytes(args.runs, args.data)),
        # pushing with only the 5% core active, before and after its 2000 consolidation updates
        "cons_start": mean(N["core_push"]["5c"]["before"]), "cons_end": mean(N["core_push"]["5c"]["after"]),
        "method_svg": method_svg(), "curves_svg": chart_curves(N),
        "layer_svg": layer_svg(core0), "packnet_svg": packnet_svg(core0, read_tensors(packnet_map, f"{EXPERT}.{SHOW_LAYER}.")),
        "curves_heading": curves_text(N)[0], "curves_text": curves_text(N)[1],
        "three_svg": chart_three(N), "exec_section": exec_section,
        "freeze_table": freeze_table(N),
        "task_frames": task_frames(args.data),
        "teaser_videos": teaser(args.runs, cache, vstage),
        "video_config": config_name(vm), "video_cols": "four" if packnet_stage else "three",
        "video_packnet": "; PackNet: 5% per task" if packnet_stage else "",
        "c_videos": c_videos,
        "video_caption": ("Example test episodes (seed 0) chosen to show each method's usual outcome relative to the others: a clean "
                          "success (completed and still completed at the end of the episode) where the method succeeds often, its "
                          "typical failure where it rarely does. Within a row all methods face the same scene"
                          + (f" (except for the rows {' and '.join(f'<q>{t[:-1]}</q>' for t in separate)})" if separate else "")
                          + ". Success rates over all 100 test scenes are in the table above."),
        "footer": (f'Built {datetime.date.today().isoformat()} from the experiment outputs with '
                   f'<a href="{args.repo_url}/blob/main/scripts/build_page.py">scripts/build_page.py</a>.'),
    }
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "page_template.html")).read()
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", value)
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", page)))
    assert not missing, f"unfilled placeholders: {missing}"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 2**20:.1f} MiB, {len(cache)} clips)")
    if args.readme:
        update_readme(args.readme, N)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="output of scripts/export_results.py")
    p.add_argument("--runs", required=True, help="the final-study run directory ($CVLA_RUNS/final)")
    p.add_argument("--data", required=True, help="$CVLA_DATA (replayed demonstrations, for the task frames)")
    p.add_argument("--repo-url", default="https://github.com/DishengTang/continual_vla")
    p.add_argument("--authors", default="")
    p.add_argument("--out", default="docs/index.html")
    p.add_argument("--readme", default=None, help="also rewrite the results tables between the markers in this README")
    build(p.parse_args())


if __name__ == "__main__":
    main()
