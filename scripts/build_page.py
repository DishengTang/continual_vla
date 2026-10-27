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
         "packnet_k2p5sp": "PackNet, 2.5% per task", "packnet_k10sp": "PackNet, 10% per task, matched",
         "gate_k10c": "Consolidate-and-gate, 10% cores, 4&times; consolidation"}
# Each method is reported at its configuration with the best three-task average on the test scenes, among those run on
# every seed (the same rule for every method); the others are listed in the full table. The gate and PackNet reserve a
# share of the network per task and are compared at 5% per task (room for twenty tasks; a quarter fits only four).
FAMILIES = {"seqft": ("seqft", "seqft_plus6k", "seqft_plus24k"), "er50": ("er50", "er50_plus6k"),
            "er500": ("er500", "er500_plus6k", "er500_plus24k"), "packnet": ("packnet_k5", "packnet_k5sp"),
            "gate": ("gate_k5", "gate_k5c")}
FAMILY_NAMES = {"seqft": "Fine-tuning", "er50": "ER-50", "er500": "ER-500", "packnet": "PackNet", "gate": "Consolidate-and-gate"}
FAMILY_OF = {"seqft": "seqft", "er50": "er50", "er500": "er500"}
OURS = set()  # the selected gate configuration (build)
COLORS = {"gate_rand": "#2a78d6", "gate_k10": "#2a78d6", "gate_k50": "#2a78d6", "gate_k5": "#2a78d6", "seqft": "#eb6834", "er50": "#1baf7a",
          "ewc": "#eda100", "er500": "#e87ba4", "packnet": "#6250d6", "packnet_k10": "#6250d6", "packnet_k5": "#6250d6",
          "gate_k5c": "#2a78d6", "gate_k2p5c": "#2a78d6", "gate_k1c": "#2a78d6", "gate_k10L": "#2a78d6", "gate_k5s25": "#2a78d6", "packnet_unfrozen": "#6250d6", "gate_adapter": "#2a78d6", "gate_bundle": "#2a78d6", "seqft_plus6k": "#eb6834", "er50_plus6k": "#1baf7a", "er500_plus6k": "#e87ba4", "seqft_plus24k": "#eb6834", "er500_plus24k": "#e87ba4",
          "gate_bundle_L24k": "#2a78d6", "gate_bundle_L24k_brC": "#2a78d6", "gate_bundle_L24k_brC64": "#2a78d6",
          "packnet_L24k": "#6250d6", "packnet_k5sp": "#6250d6", "packnet_k2p5sp": "#6250d6", "packnet_k10sp": "#6250d6",
          "gate_k10c": "#2a78d6"}
GREY, INK, INK2 = "#8f8e88", "#121211", "#52514e"
TASK_NAMES = {PUSH: "Push the cube onto the target.", STACK: "Stack the red cube on the green cube.",
              GREEN: "Stack the green cube on the red cube."}
VIDEO_SEED, VIDEO_TAG, FIRST_TEST_SEED = "s0", "test_videos", 200000
# Outcome shown per method and instruction: a clean success where the method succeeds often, its typical
# failure where it rarely does ("wrong": built the other stack).
EXAMPLES = {
    "C": [(PUSH, {"seqft": "failure", "er50": "success", "gate": "success"}),
          (STACK, {"seqft": "wrong", "er50": "failure", "gate": "success"}),
          (GREEN, {"seqft": "success", "er50": "success", "gate": "success"})],
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
              "packnet_k5sp"):
        def done(d, m=m):
            c = d["conditions"][m]
            return c if c["done"] and c["stack_after_b"] else None
        two[m] = {k: per_seed(seeds, lambda d, f=f: f(done(d))) for k, f in {
            "push": lambda c: c["push_after_b"]["any"], "stack": lambda c: c["stack_after_b"]["any"],
            "stack_end": lambda c: c["stack_after_b"]["end"],
            "push_before": lambda c: c["push_before_b"]["any"] if c["push_before_b"] else None,
            "avg": lambda c: (c["push_after_b"]["any"] + c["stack_after_b"]["any"]) / 2,
            "gate_off": lambda c: 100 * c["gate_off_push"][-1]["success"] if c["gate_off_push"] else None,
            "curve": lambda c: c["curve"]}.items()}
    two["routed"] = {"push": two["seqft"]["push_before"], "stack": two["seqft"]["stack"],
                     "stack_end": two["seqft"]["stack_end"],
                     "avg": [(p + s) / 2 for p, s in zip(two["seqft"]["push_before"], two["seqft"]["stack"])]}

    three = {}
    for m in ("seqft", "er50", "er500", "packnet", "gate_rand", "gate_k10", "gate_k5", "gate_k2p5", "gate_k1", "gate_k0p5",
              "gate_k50_r0", "gate_k50_r20", "gate_k50_r50", "gate_k50_r20_d1", "gate_k50_r50_d1", "packnet_k10", "packnet_k5", "gate_k10L", "gate_k5s25", "packnet_unfrozen", "gate_adapter", "gate_bundle", "seqft_plus6k", "er50_plus6k", "er500_plus6k", "seqft_plus24k", "er500_plus24k",
              "gate_k5c", "gate_k2p5c", "gate_k1c", "gate_bundle_L24k", "gate_bundle_L24k_brC", "gate_bundle_L24k_brC64",
              "packnet_L24k", "packnet_k5sp", "packnet_k2p5sp", "packnet_k50_r0", "packnet_k50_r50", "gate_k50_r50rg",
              "gate_k10c", "packnet_k10sp", "joint"):
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
    traces = {st: per_seed(seeds, lambda d, st=st: d["traces"][st]) for st in ("B_seqft", "B_er50", "B_gate_rand")}
    core_push = {k: {w: per_seed(seeds, lambda d, k=k, w=w: d["core_push"][k][w]["any"]) for w in ("before", "after")}
                 for k in ("0p5", "1", "2p5", "5", "10", "25", "50", "1c", "2p5c", "5c", "bundle")}
    total = len(seeds)
    sel = {}
    for fam, members in FAMILIES.items():
        full = [m for m in members if len(three[m]["avg"]) == total]
        if full:
            sel[fam] = max(full, key=lambda m: mean_sd(three[m]["avg"])[0])
    budget = {k: per_seed(seeds, lambda d, k=k: d["budget_push"][k]["any"])
              for k in ("gate_bundle", "gate_k5c", "packnet", "packnet_k5", "packnet_k5r", "packnet_k5sp", "gate_k2p5c",
                        "packnet_k2p5sp", "gate_k50", "packnet_k50sp", "gate_k10", "packnet_k10", "gate_k10c", "packnet_k10sp")}
    return {"two": two, "three": three, "cons": cons, "traces": traces, "total": total, "core_push": core_push,
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


def chart_curves(N):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, task, title in zip(axes, (PUSH, STACK), ("Pushing (old task)", "Stacking (new task)")):
        for m in ("seqft", "ewc", "er50", "gate_rand"):
            curves = N["two"][m]["curve"]
            if not curves:
                continue
            steps, ys = mean_curve(curves, task)
            ax.plot(steps, ys, "-o", color=COLORS[m], lw=2, ms=4.5, label=NAMES[m], zorder=3 if m == "gate_rand" else 2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("Stage-B updates")
        ax.set_xticks([2000, 4000, 6000, 8000, 10000])
        ax.set_ylim(-4, 104)
    axes[0].set_ylabel("Success, validation scenes (%)")
    axes[1].legend(loc="upper left", ncol=2, fontsize=9.5)
    fig.tight_layout(w_pad=2.5)
    return to_svg(fig, "curves")


def chart_three(N):
    tasks = [("push", None, "Pushing"), ("rog", "rog_wrong", "Red on green"), ("gor", "gor_wrong", "Green on red")]
    fams = [f for f in ("seqft", "er50", "er500", "packnet", "gate") if f in N["sel"]]
    methods = [N["sel"][f] for f in fams]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        for j, (key, wrong, _) in enumerate(tasks):
            x = j + (i - (len(methods) - 1) / 2) * width
            v = N["three"][m][key]
            h = mean_sd(v)[0]
            ax.bar(x, h, width * 0.9, color=COLORS[m], zorder=2)
            ax.scatter([x] * len(v), v, s=10, color="white", edgecolors=COLORS[m], linewidths=0.8, zorder=4)
            if wrong:
                w = mean_sd(N["three"][m][wrong])[0]
                ax.bar(x, w, width * 0.9, bottom=h, color="white", edgecolor=COLORS[m], hatch="////",
                       linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[2] for t in tasks])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success after task 3 (%)")
    ax.grid(axis="x", visible=False)
    handles = [matplotlib.patches.Patch(color=COLORS[m], label=FAMILY_NAMES[f]) for f, m in zip(fams, methods)]
    handles.append(matplotlib.patches.Patch(facecolor="white", edgecolor=GREY, hatch="////", label="built the other stack"))
    ax.legend(handles=handles, loc="upper center", ncol=6, fontsize=9.5, bbox_to_anchor=(0.5, 1.13), handlelength=1.2)
    return to_svg(fig, "three")


def chart_kept(N):
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    labels = {"B_seqft": "seqft", "B_er50": "er50", "B_gate_rand": "gate_rand"}
    for stage, m in labels.items():
        curves = [t["curve"] for t in N["traces"][stage]]
        if not curves:
            continue
        y = np.mean(np.array(curves), axis=0)
        t = np.arange(1, len(y) + 1) / 20
        ax.plot(t, y, color=COLORS[m], lw=2, zorder=3 if m == "gate_rand" else 2)
        ax.annotate(NAMES[m], (t[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9.5, color=INK, fontweight="bold" if m == "gate_rand" else "normal")
    ax.set_xlabel("Time in the episode (s)")
    ax.set_ylabel("Episodes stacked (%)")
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, None)
    return to_svg(fig, "kept")


def chart_exec(N):
    exec_ = N["exec"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    panels = ((PUSH, "any", "Pushing"), (STACK, "any", "Stacking, any step"), (STACK, "end", "Stacking, final step"))
    for ax, (task, key, title) in zip(axes, panels):
        for stage, m in (("B_seqft", "seqft"), ("B_er50", "er50"), ("B_gate_rand", "gate_rand")):
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
                    capsize=3, zorder=4, label="Joint training (all data)")
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
    replay = json.load(open(os.path.join(data, "replay", PUSH, "trajectory.rgb.pd_ee_delta_pos.physx_cpu.json")))
    lengths = {e["episode_id"]: e["elapsed_steps"] for e in replay["episodes"]}
    push_examples = sum(lengths[i] for i in json.load(open(os.path.join(s0, "A_plus", "splits.json")))[PUSH]["train"])
    per_example = buffer("A_plus", "memory") / json.load(open(os.path.join(s0, "A_plus", "memory.json")))["n"]
    plus = lambda a, b: a + b if a is not None and b is not None else None  # noqa: E731
    weights = size("A_plus", "trainable.safetensors") // 4  # fp32 trainable weights
    packnet_mask = weights // 8  # PackNet: one bit per weight and task
    records = {m: mask if m.startswith("gate") else packnet_mask for m in NAMES if m.startswith(("gate", "packnet"))}
    return {"two": {"seqft": 0, "ewc": plus(size("A_plus", "fisher.safetensors"), size("A_plus", "trainable.safetensors")),
                    "er50": buffer("A_plus", "memory"), "er500": buffer("A_plus", "memory500"), **records,
                    "joint": per_example * push_examples, "routed": size("A_plus", "trainable.safetensors")},
            "three": {"seqft": 0, "er50": plus(buffer("A_plus", "memory"), buffer("B_er50", "memory")),
                      "er500": plus(buffer("A_plus", "memory500"), buffer("B_er500", "memory500")),
                      **{m: 2 * v for m, v in records.items()}}}


def chart_cost(N, mem):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    gate, packnet = N["sel"].get("gate"), N["sel"].get("packnet")
    offsets = {"seqft": (9, 0, "left"), "ewc": (0, -14, "center"), "er50": (0, -14, "center"), "er500": (0, 12, "center"),
               gate: (7, -20, "left"), "joint": (0, 13, "center"), "routed": (9, -3, "left"), packnet: (-8, 10, "right")}
    panels = [("After two tasks", "two", ("seqft", "ewc", "er50", "er500", packnet, gate, "joint", "routed")),
              ("After three tasks", "three", ("seqft", "er50", "er500", packnet, gate))]
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
            dx, dy, ha = offsets[m]
            name = chart_name(N, m, values).replace("Consolidate-and-gate ", "Consolidate-and-gate\n", 1)
            ax.annotate(name, (x, mu), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center", fontsize=9.5, color=INK2 if ref else INK,
                        fontweight="bold" if m == gate else "normal")
        ax.set_xscale("log")
        ax.set_xticks([10, 1e2, 1e4, 1e6, 1e8, 1e10])
        ax.set_xticklabels(["none", "100 B", "10 kB", "1 MB", "100 MB", "10 GB"])
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
    parts.append(f'<text x="10" y="252" font-size="12.5" fill="{INK2}">Each core is a fixed share of the network (5% here, room for twenty tasks), '
                 f'so the number of tasks is limited by capacity, not by memory.</text>')
    parts.append("</svg>")
    return "".join(parts)


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
        chosen, same = choose(runs, task, wants, stages)
        if not same:
            separate.append(TASK_NAMES[task])
        out.append(f'<div class="rowlabel">Instruction: <q>{TASK_NAMES[task]}</q></div>')
        out += [clip(runs, stages[m], task, chosen[m], names.get(m, NAMES.get(m)), cache) for m in wants]
    return "".join(out), separate


C_STAGE = {"gate_k5c": "C_gate_rand_k5_c2000", "gate_k5": "C_gate_rand_k5"}


def video_stage(runs, m):
    """Task-3 stage whose recorded test episodes show the gate: the reported configuration once recorded, else 25% cores."""
    stage = C_STAGE.get(m)
    if stage and os.path.exists(os.path.join(runs, VIDEO_SEED, stage, f"evals_{VIDEO_TAG}.jsonl")):
        return stage, m
    return "C_gate_rand", "gate_rand"


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


def main_three_table(N):
    """One row per method: its selected configuration (best three-task average on the test scenes, all seeds)."""
    rows = []
    for fam in ("seqft", "er50", "er500", "packnet", "gate"):
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
        rows.append(f'<tr><td><i>{label("joint", t["push"], N["total"])} (all data at once)</i></td><td class="cfg">all three tasks '
                    f'mixed in every batch</td><td>{pm(t["push"])}</td><td>{pm(t["rog"])}</td><td>{pm(t["rog_wrong"])}</td>'
                    f'<td>{pm(t["gor"])}</td><td>{pm(t["avg"])}</td></tr>')
    return ('<table><thead><tr><th>Method</th><th>Selected configuration</th><th>Pushing</th><th>Red on green</th>'
            '<th>&hellip;built green on<br>red instead</th><th>Green on red</th><th>Average<br>of three</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def three_table(N):
    rows = []
    for m in ("seqft", "seqft_plus6k", "seqft_plus24k", "er50", "er50_plus6k", "er500", "er500_plus6k", "er500_plus24k",
              "packnet", "packnet_L24k", "packnet_unfrozen", "packnet_k10", "packnet_k5", "packnet_k5sp",
              "packnet_k2p5sp", "gate_rand",
              "gate_adapter", "gate_bundle", "gate_bundle_L24k", "gate_bundle_L24k_brC", "gate_bundle_L24k_brC64", "gate_k10",
              "gate_k10L", "gate_k5", "gate_k5c", "gate_k2p5c", "gate_k1c", "gate_k5s25"):
        t = N["three"][m]
        if not t["push"]:
            continue
        cls = ' class="ours"' if m == N["sel"].get("gate") else ""
        rows.append(f'<tr{cls}><td><span class="swatch" style="background:{COLORS[m]}"></span>{label(m, t["push"], N["total"])}'
                    f'</td><td>{pm(t["push"])}</td><td>{pm(t["rog_before"])}</td>'
                    f'<td>{pm(t["rog"])}</td><td>{pm(t["rog_wrong"])}</td><td>{pm(t["gor"])}</td><td>{pm(t["avg"])}</td></tr>')
    return ('<table><thead><tr><th>Method</th><th>Pushing</th><th>Red on green<br>before task 3</th><th>Red on green<br>after task 3</th>'
            '<th>&hellip;built green on<br>red instead</th><th>Green on red</th><th>Average<br>of three</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def freeze_table(N):
    g, nf = N["two"]["gate_rand"], N["two"]["gate_nofreeze"]
    cell = lambda v: mean(v) if v else "&mdash;"  # noqa: E731
    return ('<table class="grid2x2"><thead><tr><th></th><th>Gate on</th><th>Gate off</th></tr></thead><tbody>'
            f'<tr><th>Core frozen during B</th><td class="hi">{cell(g["push"])}%</td><td>{cell(g["gate_off"])}%</td></tr>'
            f'<tr><th>Core not frozen</th><td>{cell(nf["push"])}%</td><td>{cell(nf["gate_off"])}%</td></tr></tbody></table>'
            f'<p style="font-size:0.85rem;color:var(--fg2);text-align:center;margin-top:8px">Pushing success after stage B, '
            f'100 test scenes, mean of {len(g["push"])} seeds.</p>')


def core_tables(N):
    sel = (f'<table><thead><tr><th>Core</th><th>Pushing</th><th>Stacking</th><th>Average</th></tr></thead><tbody>'
           f'<tr><td>Random</td><td>{pm(N["two"]["gate_rand"]["push"])}</td><td>{pm(N["two"]["gate_rand"]["stack"])}</td>'
           f'<td>{pm(N["two"]["gate_rand"]["avg"])}</td></tr>'
           f'<tr><td>Selected by intervention</td><td>{pm(N["two"]["gate_abl"]["push"])}</td><td>{pm(N["two"]["gate_abl"]["stack"])}</td>'
           f'<td>{pm(N["two"]["gate_abl"]["avg"])}</td></tr></tbody></table>'
           f'<p style="font-size:0.85rem;color:var(--fg2);margin-top:8px">After stage B; test scenes, {len(N["two"]["gate_abl"]["push"])} seeds.</p>')
    return f'<div class="tablewrap" style="max-width:620px;margin:14px auto 0">{sel}</div>'


def size_section(N):
    """Core-size results on test scenes: pushing with only the core active (before and after consolidation), then
    the full sequence where it has been run."""
    sizes = [("0p5", 0.5, "gate_k0p5"), ("1", 1, "gate_k1"), ("1c", 1, "gate_k1c"), ("2p5", 2.5, "gate_k2p5"),
             ("2p5c", 2.5, "gate_k2p5c"), ("5", 5, "gate_k5"), ("5c", 5, "gate_k5c"), ("10", 10, "gate_k10"),
             ("25", 25, "gate_rand"), ("50", 50, "gate_k50")]
    rows = []
    for key, keep, m in sizes:
        cp, t, th = N["core_push"][key], N["two"][m], N["three"].get(m)
        if not cp["after"] and not t["push"]:
            continue
        groups = int(round(keep / 100 * 752))
        cons = f'{mean(cp["before"])} &rarr; {mean(cp["after"])}' if cp["before"] and cp["after"] else \
            (mean(cp["after"]) if cp["after"] else "&mdash;")
        two = [pm(t["stack"]), pm(t["avg"])] if t["push"] else ["&mdash;", "&mdash;"]
        three = (pm(th["avg"]) + (f' ({len(th["avg"])})' if len(th["avg"]) < N["total"] else "")) if th and th["avg"] else \
            ("not possible" if keep == 50 else "&mdash;")
        n = max(len(cp["after"]), len(t["push"]))
        size = f"{keep:g}%" + (", 4&times; consolidation" if key.endswith("c") else "")
        rows.append(f'<tr{" class=ours" if m == N["sel"].get("gate") else ""}><td>{size}</td><td>{groups}</td><td>{cons}</td>'
                    f'<td>{two[0]}</td><td>{two[1]}</td><td>{three}</td><td>{n}</td></tr>')
    table = ('<table><thead><tr><th>Core size</th><th>Unit<br>groups</th><th>Pushing, core only:<br>before &rarr; after consolidation</th>'
             '<th>Stacking<br>(task 2)</th><th>Average,<br>two tasks</th><th>Average,<br>three tasks</th><th>Seeds</th></tr></thead><tbody>'
             + "".join(rows) + '</tbody></table><p style="font-size:0.85rem;color:var(--fg2);margin-top:8px">Test scenes, mean (&plusmn; SD) over '
             'the seeds finished so far; the three-task seed count is in brackets when it differs. A dash: not run yet. Two 50% cores '
             'would fill the network, so a third task cannot be added.</p>')
    text = "Smaller cores leave room for more tasks, as long as the core can still carry its skill after consolidation."
    return text, f'<div class="tablewrap" style="max-width:960px;margin:14px auto 0">{table}</div>'


def full_section(N):
    """Task 3 after two 50% allocations filled the network, for both methods: nothing released, or half of each earlier
    allocation released (the earlier tasks stop using it), and the gate's variants that keep using the released groups."""
    rows = []
    for m, method, what in (("gate_k50_r0", "Consolidate-and-gate", "nothing: task 3 cannot change any weight"),
                            ("packnet_k50_r0", "PackNet", "nothing: task 3 cannot change any weight"),
                            ("gate_k50_r50rg", "Consolidate-and-gate", "half of each earlier core (least important), silenced for its task"),
                            ("packnet_k50_r50", "PackNet", "half of each earlier task's weights (smallest), masked for its task"),
                            ("gate_k50_r20", "Consolidate-and-gate", "20% of each earlier core, still used by its task"),
                            ("gate_k50_r50", "Consolidate-and-gate", "50% of each earlier core, still used by its task"),
                            ("gate_k50_r20_d1", "Consolidate-and-gate", "20% of each earlier core, still used, distillation"),
                            ("gate_k50_r50_d1", "Consolidate-and-gate", "50% of each earlier core, still used, distillation")):
        t = N["three"].get(m)
        if t and t["push"]:
            rows.append(f'<tr{" class=ours" if method.startswith("Consolidate") else ""}><td>{method}</td><td>{what}</td>'
                        f'<td>{pm(t["push"])}</td><td>{pm(t["rog_before"])}</td><td>{pm(t["rog"])}</td><td>{pm(t["gor"])}</td>'
                        f'<td>{pm(t["gor_wrong"])}</td><td>{pm(t["avg"])}</td><td>{len(t["push"])}</td></tr>')
    if not rows:
        return '<div class="prose"><p><i>Running: two 50% allocations, then task 3, for both methods.</i></p></div>'
    return ('<div class="tablewrap" style="max-width:1100px;margin:14px auto 0"><table><thead><tr><th>Method</th><th>Freed for task 3</th>'
            '<th>Pushing</th><th>Red on green<br>before task 3</th><th>Red on green</th><th>Green on red<br>(task 3)</th>'
            '<th>&hellip;built red on<br>green instead</th><th>Average</th><th>Seeds</th></tr></thead><tbody>' + "".join(rows)
            + '</tbody></table><p style="font-size:0.85rem;color:var(--fg2);margin-top:8px">Each method gives the first two tasks '
            'half of the network each, so the third task finds nothing free. Test scenes, mean &plusmn; SD over seeds. Released: '
            'chosen by each method\'s own measure (the gate: the groups whose silencing hurts their task least, gate on; PackNet: '
            'the smallest weights), zeroed or retrained for task 3 and no longer used by the task that gave them up. The gate\'s '
            'other rows keep using the released groups, so task 3 changes part of the earlier computation; with distillation every '
            'update also shows task-3 images with an earlier instruction and keeps the predicted actions close to those before task 3.</p></div>')


SHOWN_BUDGETS = ("25%", "5%", "2.5%")  # 10% per task is run for both methods but not shown yet


def full_text(N):
    """The release trade for both methods: what pushing keeps when half of each earlier share is given up."""
    g, g0, p, p0 = (N["three"][m]["push"] for m in ("gate_k50_r50rg", "gate_k50_r0", "packnet_k50_r50", "packnet_k50_r0"))
    if not (g and g0 and p and p0):
        return ""
    n = (f" ({len(p)} of {N['total']} seeds)" if len(p) < N["total"] else "")
    return (f"Giving up half of each earlier share costs both methods red on green, but the gate gives up the unit groups that matter "
            f"least and keeps pushing at {mean(g)}% (from {mean(g0)}%), while PackNet's pushing falls to {mean(p)}% (from {mean(p0)}%){n}.")


def budget_rows(N):
    """(description, weights per task, pushing per seed, three-task average) for the per-task budget comparison."""
    three = lambda m: N["three"][m]["avg"] if m in N["three"] else []  # noqa: E731
    rows = [("Consolidate-and-gate, attention-aware cores, 500 consolidation updates", "25%", N["budget"]["gate_bundle"],
             three("gate_bundle_L24k_brC")),
            ("PackNet, 500 retraining updates", "25%", N["budget"]["packnet"], three("packnet_L24k")),
            ("Consolidate-and-gate, random cores, 2000 consolidation updates", "10%", N["budget"]["gate_k10c"], three("gate_k10c")),
            ("PackNet, 2000 retraining updates, state/action/time projections shared (matched)", "10%", N["budget"]["packnet_k10sp"],
             three("packnet_k10sp")),
            ("Consolidate-and-gate, random cores, 500 consolidation updates, 6000 stacking", "10%", N["budget"]["gate_k10"],
             three("gate_k10")),
            ("PackNet, 500 retraining updates, 6000 stacking, projections pruned too", "10%", N["budget"]["packnet_k10"],
             three("packnet_k10")),
            ("Consolidate-and-gate, random cores, 2000 consolidation updates", "5%", N["budget"]["gate_k5c"], three("gate_k5c")),
            ("PackNet, 2000 retraining updates, state/action/time projections shared (matched)", "5%", N["budget"]["packnet_k5sp"],
             three("packnet_k5sp")),
            ("PackNet, 2000 retraining updates, projections pruned too", "5%", N["budget"]["packnet_k5r"], []),
            ("PackNet, 500 retraining updates, projections pruned too", "5%", N["budget"]["packnet_k5"], three("packnet_k5")),
            ("Consolidate-and-gate, random cores, 2000 consolidation updates", "2.5%", N["budget"]["gate_k2p5c"], three("gate_k2p5c")),
            ("PackNet, 2000 retraining updates, state/action/time projections shared (matched)", "2.5%",
             N["budget"]["packnet_k2p5sp"], three("packnet_k2p5sp"))]
    return [r for r in rows if r[1] in SHOWN_BUDGETS]


def packnet_section(N):
    """Where consolidate-and-gate differs from PackNet: small budgets, cut-out sub-networks, task records."""
    rows = []
    for what, size, push, avg in budget_rows(N):
        if not push:
            continue
        seeds = " / ".join(f"{v:.0f}" for v in push) + (" (seed 0 only)" if len(push) == 1 else "")
        cls = ' class="ours"' if what.startswith("Consolidate") else ""
        rows.append(f'<tr{cls}><td>{what}</td><td>{size}</td><td>{seeds}</td><td>{pm(push)}</td>'
                    f'<td>{pm(avg) if avg else "running" if "matched" in what else "&mdash;"}</td></tr>')
    budget = ('<div class="tablewrap" style="max-width:960px;margin:14px auto 0"><table><thead><tr><th>Configuration</th>'
              '<th>Weights<br>per task</th><th>Pushing after its own stage,<br>seeds 0 / 1 / 2</th><th>Mean</th>'
              '<th>Average after<br>three tasks</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
              '<p style="font-size:0.85rem;color:var(--fg2);margin-top:8px">100 test scenes, mean &plusmn; SD over seeds. Pushing is '
              'scored right after its own stage (the core consolidated, or the weights pruned and retrained); both methods freeze it '
              'afterwards. The matched PackNet keeps the 1.6M state/action/time projection weights whole for the first task, as the '
              'gate does (they lie outside the unit groups), so at 5% both give pushing about 6.6M weights. A dash: not run.</p></div>')
    S = N["sliced"]
    if not S.get(""):
        return budget, ""
    rows, paired = [], []
    for task, name in ((PUSH, "Pushing"), (STACK, "Red on green")):
        r = S[""]["tasks"][task]
        # episodes that end differently: won only by the gated network vs only by the cut-out one (exact two-sided sign test)
        d, g = 100 - r["episodes_agree"], round(100 * (r["gated_success"] - r["narrow_success"]))
        a, b = (d + g) // 2, (d - g) // 2
        p = min(1.0, 2 * sum(math.comb(d, k) for k in range(min(a, b) + 1)) / 2 ** d) if d else 1.0
        paired.append(f"{name.lower()}: {r['episodes_agree']} of 100 episodes end the same way"
                      + (f", {a} against {b} of the {d} that differ (exact sign test p = {p:.2f})" if d > 1 else ""))
        lat = [(f'{r["ms_batch1"]["narrow"]:.0f} / {r["ms_batch1"]["packnet"]:.0f}'),
               (f'{r["ms_batch10"]["narrow"]:.0f} / {r["ms_batch10"]["packnet"]:.0f}')]
        b64 = S.get("_b64", {}).get("tasks", {}).get(task, {}).get("ms_batch64")
        cpu = S.get("_cpu20", {}).get("tasks", {}).get(task, {}).get("ms_batch1")
        lat += [f'{b64["narrow"]:.0f} / {b64["packnet"]:.0f}' if b64 else "&mdash;",
                f'{cpu["narrow"]:.0f} / {cpu["packnet"]:.0f}' if cpu else "&mdash;"]
        rows.append(f'<tr><td>{name}</td><td>{100 * r["weights_used"]:.1f}%</td><td>{r["max_abs_diff_fp32"]:.0e}</td>'
                    f'<td>{100 * r["narrow_success"]:.0f} / {100 * r["gated_success"]:.0f} ({r["episodes_agree"]})</td>'
                    + "".join(f"<td>{v}</td>" for v in lat) + "</tr>")
    sliced = ('<div class="tablewrap" style="max-width:1000px;margin:14px auto 0"><table><thead><tr><th>Task</th>'
              '<th>Action-expert<br>weights used</th><th>Largest action<br>difference (fp32)</th>'
              '<th>Success: cut out / gated<br>(same outcome, of 100)</th><th>RTX 4090,<br>batch 1</th><th>Batch 10</th>'
              '<th>Batch 64</th><th>CPU (16 threads),<br>batch 1</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
              '<p style="font-size:0.85rem;color:var(--fg2);margin-top:8px">Seed 0. Latency: milliseconds per 50-action chunk '
              '(image and language prefix plus 10 denoising steps, bf16 autocast on the GPU), median of 50 runs (20 at batch 64 and on '
              'the CPU), cut-out network / the full-size network PackNet runs for every task. The third task uses every unit '
              'group, so it runs at full size. Closed-loop success on the 100 test scenes; the differences come from bf16 rounding '
              f'({"; ".join(paired)}).</p></div>')
    return budget, sliced


def readme_block(N):
    """Markdown results tables for the README (between the results markers)."""
    def md(values):
        return pm(values).replace("&plusmn;", "±")
    gate, packnet = N["sel"].get("gate"), N["sel"].get("packnet")
    memory = {"seqft": "none", "ewc": "Fisher + old weights (0.8 GB)", "er50": "9.4 MiB", "er500": "94 MiB",
              packnet: "an owner per weight", "joint": "all PushCube data"}
    two = ["PushCube, then StackCube. Success rate (%) on 100 test scenes per task, mean ± SD over the training seeds "
           "(3 unless stated). Consolidate-and-gate and PackNet reserve 5% of the network per task:", "",
           "| Method | Pushing after learning to stack | Stacking | Average | Stored for the old task |", "|---|---|---|---|---|"]
    shown = [m for m in ("seqft", "ewc", "er50", "er500", packnet, gate, "joint") if m and N["two"].get(m, {}).get("push")]
    for m in shown:
        t = N["two"][m]
        name = label(m, t["push"], N["total"], ours=False).replace("ER-500", "Experience replay, 500 examples", 1) \
            .replace("ER-50", "Experience replay, 50 examples", 1)
        name = f"**{name}**" if m == gate else f"*{name} (all data at once)*" if m == "joint" else name
        push = f"**{md(t['push'])}**" if m == gate else md(t["push"])
        two.append(f"| {name} | {push} | {md(t['stack'])} | {mean(t['avg'])} | {memory.get(m, 'a unit mask per task')} |")
    before = [mean_sd(N["two"][m]["push_before"])[0] for m in shown if m != "joint" and N["two"][m].get("push_before")]
    three = ["", f"Pushing was {min(before):.1f}–{max(before):.1f}% before StackCube was learned. A third task then asks for the",
             "opposite stack in the same scenes. Each method is shown at its best three-task configuration on the test scenes, the",
             "gate and PackNet at 5% per task (room for twenty tasks); all configurations, including a quarter of the network per",
             "task, are on the project page. Replay stores examples of each earlier task:", "",
             "| Method | Configuration | Pushing | Red on green | … built green on red instead | Green on red (new) | Average |",
             "|---|---|---|---|---|---|---|"]
    for fam in ("seqft", "er50", "er500", "packnet", "gate"):
        m = N["sel"].get(fam)
        if m is None:
            continue
        t = N["three"][m]
        name = {"er50": "Experience replay, 50 + 50 examples", "er500": "Experience replay, 500 + 500 examples"}.get(fam, FAMILY_NAMES[fam])
        cells = [config_name(m).replace("&times;", "×"), md(t["push"]), md(t["rog"]), md(t["rog_wrong"]), md(t["gor"]), mean(t["avg"])]
        three.append(f"| {'**' + name + '**' if fam == 'gate' else name} | " + " | ".join(cells) + " |")
    t = N["three"]["joint"]
    if t["push"]:  # reference, not a continual learner
        name = label("joint", t["push"], N["total"], ours=False)
        three.append(f"| *{name} (all data at once)* | all three tasks mixed in every batch | " + " | ".join(
            [md(t["push"]), md(t["rog"]), md(t["rog_wrong"]), md(t["gor"]), mean(t["avg"])]) + " |")
    budget = ["", "Per-task budget against PackNet. Pushing is scored right after its own stage; both methods freeze it afterwards:",
              "", "| Configuration | Weights per task | Pushing, seeds 0 / 1 / 2 | Mean | Average after three tasks |", "|---|---|---|---|---|"]
    for what, size, push, avg in budget_rows(N):
        if push:
            budget.append(f"| {what} | {size} | {' / '.join(f'{v:.0f}' for v in push)}{' (seed 0 only)' if len(push) == 1 else ''} | {md(push)} | "
                          f"{md(avg) if avg else 'running' if 'matched' in what else '—'} |")
    return "\n".join(two + three + budget)


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
    budget_table, sliced_table = packnet_section(N)
    sl = N["sliced"].get("", {})
    cache = {}
    c_videos, separate = video_grid(args.runs, "C", {"seqft": "C_seqft", "er50": "C_er50", "gate": vstage},
                                    {"gate": FAMILY_NAMES["gate"]}, cache)
    exec_section = ""
    if any(len(v) > 2 for v in N["exec"].values()):
        exec_section = ('<div class="prose"><h3 id="exec">Control: how many predicted actions to execute</h3><p>The policy predicts 50 actions and '
                        'executes the first 5 before looking again. Five gives the highest or near-highest success for every method. '
                        'Executing more of each chunk without looking lowers stacking and loses most finished stacks by the end '
                        'of the episode; executing fewer stacks slightly less often but keeps more stacks. Pushing retention '
                        'and the ranking of the methods hold at every setting.</p></div>'
                        f'<figure class="chart">{chart_exec(N)}<figcaption>Seed 0 stage-B checkpoints, 100 test scenes per task '
                        'and setting.</figcaption></figure>')
    traces_lost = [t for st in N["traces"].values() for t in st if "lost" in t]
    kept = {st: mean([t["kept"] for t in v]) for st, v in N["traces"].items() if v}
    values = {
        "authors": html.escape(args.authors), "repo_url": args.repo_url, "repo_name": args.repo_url.rstrip("/").split("/")[-1],
        "n_test": "100", "n_seeds": str(len(two[G]["push"])),
        "seeds_three": f'{len(g3["push"])} seed{"s" if len(g3["push"]) > 1 else ""}',
        "seqft_push": mean(two["seqft"]["push"]), "gate_push": mean(two[G]["push"]),
        "er50_push": mean(two["er50"]["push"]), "gate_push_before": mean(two[G]["push_before"]),
        "gate_stack": mean(two[G]["stack"]), "seqft_stack": mean(two["seqft"]["stack"]),
        "gate_avg": mean(two[G]["avg"]), "er500_avg": mean(two["er500"]["avg"]),
        "two_packnet_avg": mean(two[P]["avg"]) if two.get(P, {}).get("avg") else "&mdash;",
        "three_gate_rog": mean(g3["rog"]), "three_seqft_rog": mean(s3["rog"]),
        "three_er50_rog": mean(e3["rog"]), "three_er500_rog": mean(r3["rog"]), "three_seqft_wrong": mean(s3["rog_wrong"]),
        "three_gate_gor": mean(g3["gor"]), "three_seqft_gor": mean(s3["gor"]),
        "three_gate_avg": mean(g3["avg"]), "three_er50_avg": mean(e3["avg"]),
        "three_seqft_avg": mean(s3["avg"]), "three_gate_push": mean(g3["push"]),
        "three_packnet_avg": mean(p3["avg"]), "three_packnet_rog": mean(p3["rog"]), "three_packnet_gor": mean(p3["gor"]),
        "three_packnet_push": mean(p3["push"]), "gate_config": config_name(N["sel"]["gate"]),
        "gate_gap": f'{mean_sd(g3["avg"])[0] - mean_sd(p3["avg"])[0]:.1f}', "three_er500_avg": mean(r3["avg"]),
        "gate_gap_seeds": " / ".join(f"{g - p:+.1f}" for g, p in zip(g3["avg"], p3["avg"])),
        "q_gate_avg": mean(q3["avg"]), "q_packnet_avg": mean(qp3["avg"]), "q_gate_rog": mean(q3["rog"]),
        "q_packnet_rog": mean(qp3["rog"]), "q_gap": f'{mean_sd(qp3["avg"])[0] - mean_sd(q3["avg"])[0]:.1f}',
        "packnet_config": config_name(N["sel"]["packnet"]),
        "nobranch_avg": mean(three["gate_bundle_L24k"]["avg"]) if three["gate_bundle_L24k"]["avg"] else "&mdash;",
        "k5_gate_push": mean(N["budget"]["gate_k5c"]), "k5_packnet_push": mean(N["budget"]["packnet_k5sp"]),
        "k5_packnet_pruned": mean(N["budget"]["packnet_k5r"]), "k5_gate_avg": mean(three["gate_k5c"]["avg"]),
        "budget_table": budget_table, "sliced_table": sliced_table, "main_three_table": main_three_table(N),
        "push_weights": f'{100 * sl["tasks"][PUSH]["weights_used"]:.1f}' if sl else "&mdash;",
        "stack_weights": f'{100 * sl["tasks"][STACK]["weights_used"]:.1f}' if sl else "&mdash;",
        "core_json_kb": f'{N["core_json_bytes"] / 1000:.1f}' if N["core_json_bytes"] else "&mdash;",
        "packnet_map_mb": f'{sl["expert_weights"] / 1e6:.1f}' if sl else "&mdash;",
        "er500_three_note": "",
        "three_er500_clause": f', {mean(r3["avg"])}% (replay of 500 examples)',
        "packnet_clause": (", and PackNet, the closest earlier method, which keeps a fixed share of every weight matrix per task "
                           "(at the same 5% per task and the same retraining budget as our consolidation)" if p3["push"] else ""),
        "packnet_cost": (", one bit per weight and task (PackNet)" if p3["push"] else ""),
        "size_text": size_section(N)[0], "size_table": size_section(N)[1], "full_section": full_section(N), "full_text": full_text(N),
        "sequence_svg": chart_sequence(N), "cost_svg": chart_cost(N, memory_bytes(args.runs, args.data)),
        # pushing with only the 5% core active, before and after its 2000 consolidation updates
        "cons_start": mean(N["core_push"]["5c"]["before"]), "cons_end": mean(N["core_push"]["5c"]["after"]),
        "abl_start": "/".join(f"{v:.0f}" for v in N["cons"]["abl"]["0"]),
        "rand_start": "/".join(f"{v:.0f}" for v in N["cons"]["rand"]["0"]),
        "cons_range": "{:.0f}&ndash;{:.0f}".format(min(N["cons"]["rand"]["500"] + N["cons"]["abl"]["500"]),
                                                   max(N["cons"]["rand"]["500"] + N["cons"]["abl"]["500"])),
        "abl_avg": mean(two["gate_abl"]["avg"]), "rand_avg": mean(two["gate_rand"]["avg"]),
        "regrasp_share": f'{100 * sum(t["lost_regrasped"] for t in traces_lost) / sum(t["lost"] for t in traces_lost):.0f}',
        "kept_text": f'fine-tuning {kept["B_seqft"]}%, ER-50 {kept["B_er50"]}%, consolidate-and-gate (25% cores) {kept["B_gate_rand"]}%',
        "method_svg": method_svg(), "curves_svg": chart_curves(N),
        "three_svg": chart_three(N), "kept_svg": chart_kept(N), "exec_section": exec_section,
        "three_table": three_table(N), "freeze_table": freeze_table(N),
        "core_tables": core_tables(N),
        "task_frames": task_frames(args.data),
        "teaser_videos": teaser(args.runs, cache, vstage),
        "video_config": config_name(vm),
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
