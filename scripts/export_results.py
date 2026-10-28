"""Collect every number the report needs from the final-study run directories into one JSON file.

    python scripts/export_results.py --root $CVLA_RUNS/final --ewc B_ewc0.1 [--out results.json]

Per seed: PushCube stage results; for each condition the PushCube result before the StackCube stage,
the StackCube-stage learning curve (validation resets, every 2000 updates; earlier points from a run of the
stage's first 2000 updates) and the final test results
(success at any step and at the final step); consolidation-recovery points for random and
importance-selected cores; frozen/plastic parameter counts; gate-off evaluations. Conditions whose
runs exist only for some seeds (references and ablations) are exported where present. Core-size pilots
(seed 0, validation resets): PushCube after gated consolidation and both tasks after the StackCube stage.
Per-step trace summaries of the StackCube-stage checkpoints (scripts/analyze_traces.py). The three-task
sequence: every task before and after the third task (green cube on red cube) with wrong-order rates,
and its learning curves. Executed actions per chunk (1-50) for the seed-0 StackCube-stage checkpoints.
For the per-task budget comparison: the share of the action expert frozen after task 1 and after tasks 1 and 2.
For stacking tasks, "wrong"/"wrong_end" is the percentage of episodes that built the opposite stack.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_traces import load as load_traces, summarize as summarize_traces  # noqa: E402

PUSH, STACK, GREEN = "PushCube-v1", "StackCube-v1", "StackCubeGreenOnRed-v1"


def rows(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def score(r):
    out = {"any": 100 * r["success"], "end": 100 * r["success_end"], "n": r["n"]}
    if "opposite_success" in r:
        out.update(wrong=100 * r["opposite_success"], wrong_end=100 * r["opposite_success_end"])
    return out


def final_test(run_dir, task):
    """(any-step %, final-step %) of the stage's last test evaluation over the most scenes, else None."""
    evals = [r for r in rows(os.path.join(run_dir, "evals.jsonl")) if r.get("split") == "test" and r["task"] == task]
    evals += [r for r in rows(os.path.join(run_dir, "evals_test_exec5.jsonl")) if r["task"] == task]
    if not evals:
        return None
    return score(max(evals, key=lambda r: (r["n"], r.get("step", 0))))


def tagged(run_dir, task, tag):
    """Last result of a separate evaluation (scripts/evaluate.py, evals_<tag>.jsonl), else None."""
    evals = [r for r in rows(os.path.join(run_dir, f"evals_{tag}.jsonl")) if r["task"] == task]
    return score(evals[-1]) if evals else None


def final_val(run_dir, task, min_episodes=50):
    """(any-step %, final-step %) of the stage's last validation evaluation with at least `min_episodes`."""
    evals = [r for r in rows(os.path.join(run_dir, "evals.jsonl")) if r.get("split") == "val" and r["task"] == task]
    evals += [r for r in rows(os.path.join(run_dir, "evals_val_exec5.jsonl")) if r["task"] == task]
    evals = [r for r in evals if r["n"] >= min_episodes]
    if not evals:
        return None
    last = max(evals, key=lambda r: r.get("step", 0))
    return {"any": 100 * last["success"], "end": 100 * last["success_end"], "n": last["n"]}


def initial_test(run_dir, task):
    evals = [r for r in rows(os.path.join(run_dir, "evals.jsonl"))
             if r.get("split") == "test" and r["task"] == task and r["step"] == 0]
    return {"any": 100 * evals[0]["success"], "end": 100 * evals[0]["success_end"], "n": evals[0]["n"]} if evals else None


def core_initial(root, stage):
    """Pushing with only the core active before consolidation; a longer consolidation run (_c2000) that was not scored
    before takes the score of the 500-update run of the same seed when both drew the same core."""
    r = initial_test(os.path.join(root, stage), PUSH)
    short = stage.replace("_c2000", "")
    if r or short == stage:
        return r
    cores = [os.path.join(root, st, "core.json") for st in (stage, short)]
    if all(os.path.exists(c) for c in cores) and json.load(open(cores[0]))["core"] == json.load(open(cores[1]))["core"]:
        return initial_test(os.path.join(root, short), PUSH)
    return None


def curve(run_dir, tasks=(PUSH, STACK)):
    """Validation-reset success during the stage, by update step."""
    out = {t: [] for t in tasks}
    for r in rows(os.path.join(run_dir, "evals.jsonl")):
        if r.get("split") == "val" and r["task"] in out:
            out[r["task"]].append({"step": r["step"], "success": 100 * r["success"], "n": r["n"],
                                   "val_loss": r.get("val_loss")})
    return {t: sorted(v, key=lambda p: p["step"]) for t, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ewc", default="B_ewc0.1")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    conditions = {"seqft": ("A_plus", "B_seqft"), "er50": ("A_plus", "B_er50"), "ewc": ("A_plus", args.ewc),
                  "gate_rand": ("A_core_rand", "B_gate_rand"),
                  # references and ablations
                  "er500": ("A_plus", "B_er500"), "joint": (None, "joint"),
                  "gate_nofreeze": ("A_core_rand", "B_gate_rand_nofreeze"), "gate_abl": ("A_core_abl", "B_gate_abl"),
                  "gate_k10": ("A_core_rand_k10", "B_gate_rand_k10"), "gate_k50": ("A_core_rand_k50", "B_gate_rand_k50"),
                  "gate_k5": ("A_core_rand_k5", "B_gate_rand_k5"), "gate_k2p5": ("A_core_rand_k2p5", "B_gate_rand_k2p5"),
                  "gate_k1": ("A_core_rand_k1", "B_gate_rand_k1"), "gate_k0p5": ("A_core_rand_k0p5", "B_gate_rand_k0p5"),
                  "packnet": ("A_packnet", "B_packnet"),
                  # the 25% gate chain re-run on the local GPUs (hardware check)
                  "gate_hw": ("A_core_rand_hw", "B_gate_rand_hw"), "packnet_k10": ("A_packnet_k10", "B_packnet_k10"),
                  "packnet_k5": ("A_packnet_k5", "B_packnet_k5"),
                  # attention-aware cores: key/value bundles and MLP blocks with per-layer quotas
                  "gate_bundle": ("A_core_bundle", "B_gate_bundle"),
                  # small cores consolidated for 2000 instead of 500 updates
                  "gate_k5c": ("A_core_rand_k5_c2000", "B_gate_rand_k5_c2000"),
                  "gate_k2p5c": ("A_core_rand_k2p5_c2000", "B_gate_rand_k2p5_c2000"),
                  "gate_k1c": ("A_core_rand_k1_c2000", "B_gate_rand_k1_c2000"),
                  # the analyses of gate_k5c: core not frozen during stage B; a core selected by intervention; the same
                  # stage B scored on validation resets every 2000 updates (learning curve)
                  "gate_nofreeze_k5c": ("A_core_rand_k5_c2000", "B_gate_rand_k5_c2000_nofreeze"),
                  "gate_abl_k5c": ("A_core_abl_k5_c2000", "B_gate_abl_k5_c2000"),
                  "gate_k5c_curve": ("A_core_rand_k5_c2000", "B_gate_rand_k5_c2000_curve"),
                  # PackNet at 5% per task matched to gate_k5c (stacking scored before it is pruned, as for "packnet")
                  "packnet_k5sp": ("A_packnet_k5_r2000_sp", "B_packnet_k5sp"),
                  "packnet_k1sp": ("A_packnet_k1sp", "B_packnet_k1sp")}
    result = {"ewc_run": args.ewc, "seeds": {}}
    for s in args.seeds:
        root = os.path.join(args.root, f"s{s}")
        seed = {"A": {PUSH: final_test(os.path.join(root, "A"), PUSH), STACK: final_test(os.path.join(root, "A"), STACK)},
                "conditions": {}, "consolidation": {}}
        if s == 0:
            seed["base_model"] = {t: initial_test(os.path.join(root, "A"), t) for t in (PUSH, STACK)}
        for name, (prep, b) in conditions.items():
            b_dir = os.path.join(root, b)
            if not os.path.isdir(b_dir):
                continue
            freeze = os.path.join(b_dir, "freeze.json")
            seed["conditions"][name] = {
                "done": os.path.exists(os.path.join(b_dir, "stage.json")),
                "push_before_b": final_test(os.path.join(root, prep), PUSH) if prep else None,
                "push_after_b": final_test(b_dir, PUSH), "stack_after_b": final_test(b_dir, STACK),
                "curve": curve(b_dir),
                # the stage's first 2000 updates run again and scored more often (learning curve before 2000)
                "curve_early": curve(os.path.join(root, b.removesuffix("_curve") + "_early")),
                "freeze": json.load(open(freeze)) if os.path.exists(freeze) else None,
                "gate_off_push": [r for r in rows(os.path.join(b_dir, "evals_test_exec5_nogate.jsonl"))],
            }
        for sel in ("rand", "abl"):
            points = {}
            full = os.path.join(root, f"A_core_{sel}")
            if initial_test(full, PUSH):
                points[0] = initial_test(full, PUSH)
            for n in (50, 200):
                r = final_test(os.path.join(root, "curve", f"A_core_{sel}_c{n}"), PUSH)
                if r:
                    points[n] = r
            if final_test(full, PUSH):
                points[500] = final_test(full, PUSH)
            seed["consolidation"][sel] = points
        sizes = {}
        for keep in (10, 25, 50):
            a_dir = os.path.join(root, "A_core_rand" if keep == 25 else f"A_core_rand_k{keep}")
            b_dir = os.path.join(root, "B_gate_rand" if keep == 25 else f"B_gate_rand_k{keep}")
            freeze = os.path.join(b_dir, "freeze.json")
            if final_val(b_dir, STACK):
                sizes[keep] = {"push_consolidated": final_val(a_dir, PUSH), "push_after_b": final_val(b_dir, PUSH),
                               "stack_after_b": final_val(b_dir, STACK),
                               "freeze": json.load(open(freeze)) if os.path.exists(freeze) else None}
        if sizes:
            seed["core_size"] = sizes
        core_push = {}  # pushing with only the core active, before and after its 500 consolidation updates
        for keep, stage in (("0p5", "A_core_rand_k0p5"), ("1", "A_core_rand_k1"), ("2p5", "A_core_rand_k2p5"),
                            ("5", "A_core_rand_k5"), ("10", "A_core_rand_k10"), ("25", "A_core_rand"), ("50", "A_core_rand_k50"),
                            ("1c", "A_core_rand_k1_c2000"), ("2p5c", "A_core_rand_k2p5_c2000"), ("5c", "A_core_rand_k5_c2000"),
                            ("abl5c", "A_core_abl_k5_c2000"), ("bundle", "A_core_bundle")):
            # a running stage has only its initial test evaluation, which final_test would return
            done = os.path.exists(os.path.join(root, stage, "stage.json"))
            after = final_test(os.path.join(root, stage), PUSH) if done else None
            if after:
                core_push[keep] = {"before": core_initial(root, stage), "after": after}
        if core_push:
            seed["core_push"] = core_push
        budget = {}  # PushCube after its own stage (core consolidated, or PackNet pruned and retrained)
        for name, stage in (("gate_bundle", "A_core_bundle"), ("gate_k5c", "A_core_rand_k5_c2000"), ("packnet", "A_packnet"),
                            ("packnet_k5", "A_packnet_k5"), ("packnet_k5r", "A_packnet_k5_r2000"),
                            ("packnet_k5sp", "A_packnet_k5_r2000_sp"), ("gate_k2p5c", "A_core_rand_k2p5_c2000"),
                            ("packnet_k2p5sp", "A_packnet_k2p5sp"), ("gate_k50", "A_core_rand_k50"),
                            ("packnet_k50sp", "A_packnet_k50sp"), ("gate_k10", "A_core_rand_k10"), ("packnet_k10", "A_packnet_k10"),
                            ("gate_k10c", "A_core_rand_k10_c2000"), ("packnet_k10sp", "A_packnet_k10sp"),
                            ("gate_k1c", "A_core_rand_k1_c2000"), ("packnet_k1sp", "A_packnet_k1sp")):
            r = final_test(os.path.join(root, stage), PUSH)
            if r:
                budget[name] = r
        if budget:
            seed["budget_push"] = budget
        # share of the action expert's weights frozen while task 2 is learned (after task 1) and while task 3 is learned
        # (after tasks 1 and 2), for the per-task budget comparison
        protected = {}
        for name, b, c in (("gate_bundle", "B_gate_bundle", "C_gate_bundle_L24k"), ("packnet", "B_packnet", "C_packnet_L24k"),
                           ("gate_k10c", "B_gate_rand_k10_c2000", "C_gate_rand_k10_c2000"),
                           ("packnet_k10sp", "B_packnet_k10sp", "C_packnet_k10sp"),
                           ("gate_k5c", "B_gate_rand_k5_c2000", "C_gate_rand_k5_c2000"),
                           ("packnet_k5sp", "B_packnet_k5sp", "C_packnet_k5sp"),
                           ("gate_k2p5c", "B_gate_rand_k2p5_c2000", "C_gate_rand_k2p5_c2000"),
                           ("packnet_k2p5sp", "B_packnet_k2p5sp", "C_packnet_k2p5sp"),
                           ("gate_k1c", "B_gate_rand_k1_c2000", "C_gate_rand_k1_c2000"),
                           ("packnet_k1sp", "B_packnet_k1sp", "C_packnet_k1sp")):
            shares = {}
            for key, stage in (("after_1", b), ("after_2", c)):
                path = os.path.join(root, stage, "freeze.json")
                if os.path.exists(path):
                    f = json.load(open(path))
                    shares[key] = 100 * f["frozen"] / (f["frozen"] + f["plastic"])
            if shares:
                protected[name] = shares
        if protected:
            seed["protected"] = protected
        traces = {}
        for stage in ("B_seqft", "B_er50", "B_gate_rand", "B_gate_rand_k5_c2000"):
            t, states = load_traces(os.path.join(root, stage), STACK)
            if t is not None:
                traces[stage] = summarize_traces(t, states)
        if traces:
            seed["traces"] = traces
        three = {}
        for name, (b, c) in {"seqft": ("B_seqft", "C_seqft"), "er50": ("B_er50", "C_er50"), "er500": ("B_er500", "C_er500"),
                             # EWC: the penalty of task 3 uses stage B's Fisher information (red on green) and weights
                             "ewc": (args.ewc, "C_" + args.ewc.split("_", 1)[1]),
                             "gate_rand": ("B_core_rand", "C_gate_rand"), "gate_k10": ("B_core_rand_k10", "C_gate_rand_k10"),
                             "gate_k5": ("B_core_rand_k5", "C_gate_rand_k5"), "gate_k2p5": ("B_core_rand_k2p5", "C_gate_rand_k2p5"),
                             "gate_k1": ("B_core_rand_k1", "C_gate_rand_k1"), "gate_k0p5": ("B_core_rand_k0p5", "C_gate_rand_k0p5"),
                             "packnet": ("B_packnet_pruned", "C_packnet"), "gate_hw": ("B_core_rand_hw", "C_gate_rand_hw"),
                             "packnet_k10": ("B_packnet_k10_pruned", "C_packnet_k10"),
                             "packnet_k5": ("B_packnet_k5_pruned", "C_packnet_k5"),
                             # 10% cores with StackCube consolidated 4x longer (24000 updates)
                             "gate_k10L": ("B_core_rand_k10_L24k", "C_gate_rand_k10_L24k"),
                             # a 5% core for PushCube (2000 updates), a 25% core for StackCube
                             "gate_k5s25": ("B_core_rand_k5_c2000_s25", "C_gate_rand_k5_c2000_s25"),
                             # controls and pilots after the 10-26 audit
                             "packnet_unfrozen": ("B_packnet_pruned", "C_packnet_unfrozen"),
                             "gate_adapter": ("B_core_rand", "C_gate_rand_adapter"),
                             "gate_bundle": ("B_core_bundle", "C_gate_bundle"),
                             "seqft_plus6k": ("B_seqft_plus6k", "C_seqft_plus6k"),
                             "er50_plus6k": ("B_er50_plus6k", "C_er50_plus6k"),
                             "er500_plus6k": ("B_er500_plus6k", "C_er500_plus6k"),
                             "seqft_plus24k": ("B_seqft_plus24k", "C_seqft_plus24k"),
                             "er500_plus24k": ("B_er500_plus24k", "C_er500_plus24k"),
                             "gate_k5c": ("B_core_rand_k5_c2000", "C_gate_rand_k5_c2000"),
                             "gate_k2p5c": ("B_core_rand_k2p5_c2000", "C_gate_rand_k2p5_c2000"),
                             "gate_k1c": ("B_core_rand_k1_c2000", "C_gate_rand_k1_c2000"),
                             # attention-aware cores with StackCube consolidated for 24000 updates; task 3 without or with
                             # an attention branch of its own (rank 16 or 64)
                             "gate_bundle_L24k": ("B_core_bundle_L24k", "C_gate_bundle_L24k"),
                             "gate_bundle_L24k_brC": ("B_core_bundle_L24k", "C_gate_bundle_L24k_brC"),
                             "gate_bundle_L24k_brC64": ("B_core_bundle_L24k", "C_gate_bundle_L24k_brC64"),
                             # PackNet with StackCube retrained for 24000 updates after pruning
                             "packnet_L24k": ("B_packnet_pruned_L24k", "C_packnet_L24k"),
                             # PackNet at 5% per task matched to gate_k5c: PushCube retrained 2000 updates with the
                             # state/action/time projections shared like the 1-D parameters, StackCube retrained 24000
                             "packnet_k5sp": ("B_packnet_k5sp_pruned", "C_packnet_k5sp"),
                             "packnet_k2p5sp": ("B_packnet_k2p5sp_pruned", "C_packnet_k2p5sp"),
                             # the same pair at 10% per task
                             "gate_k10c": ("B_core_rand_k10_c2000", "C_gate_rand_k10_c2000"),
                             # reference: all three tasks trained at once from the pretrained policy
                             "joint": ("joint3", "joint3"),
                             "packnet_k10sp": ("B_packnet_k10sp_pruned", "C_packnet_k10sp"),
                             "packnet_k1sp": ("B_packnet_k1sp_pruned", "C_packnet_k1sp"),
                             # beyond capacity, matched to gate_k50_*: two 50% allocations fill the network; task 3 with
                             # nothing released, or half of each earlier allocation released and no longer used by it
                             "packnet_k50_r0": ("B_packnet_k50sp_pruned", "B_packnet_k50sp_pruned"),
                             "packnet_k50_r50": ("B_packnet_k50sp_pruned", "C_packnet_k50sp_r50"),
                             "gate_k50_r50rg": ("B_core_rand_k50", "C_gate_rand_k50_r50rg"),
                             # a full network (two 50% cores): task 3 without release is the consolidated checkpoint itself
                             "gate_k50_r0": ("B_core_rand_k50", "B_core_rand_k50"),
                             "gate_k50_r20": ("B_core_rand_k50", "C_gate_rand_k50_r20"),
                             "gate_k50_r50": ("B_core_rand_k50", "C_gate_rand_k50_r50"),
                             # released groups protected by distillation from the checkpoint before task 3
                             "gate_k50_r20_d1": ("B_core_rand_k50", "C_gate_rand_k50_r20_d1"),
                             "gate_k50_r50_d1": ("B_core_rand_k50", "C_gate_rand_k50_r50_d1")}.items():
            c_dir = os.path.join(root, c)
            if os.path.isdir(c_dir):
                three[name] = {"done": os.path.exists(os.path.join(c_dir, "stage.json")),
                               "before_c": {t: final_test(os.path.join(root, b), t) for t in (PUSH, STACK)},
                               "after_c": {t: final_test(c_dir, t) for t in (PUSH, STACK, GREEN)},
                               "curve": curve(c_dir, (PUSH, STACK, GREEN))}
        if three:
            seed["three_task"] = three
        sweep = {}
        for stage in ("B_seqft", "B_er50", "B_gate_rand", "B_gate_rand_k5_c2000"):
            points = {n: {t: tagged(os.path.join(root, stage), t, f"test_exec{n}") for t in (PUSH, STACK)}
                      for n in (1, 2, 10, 25, 50)}
            points = {n: v for n, v in points.items() if any(x for x in v.values())}
            if points:
                points[5] = {t: final_test(os.path.join(root, stage), t) for t in (PUSH, STACK)}
                sweep[stage] = dict(sorted(points.items()))
        if sweep:
            seed["exec_steps"] = sweep
        result["seeds"][s] = seed
    # seed 0: each task's gated sub-network cut out as a narrower expert (scripts/benchmark_sliced.py), and task records
    stage = os.path.join(args.root, "s0", "C_gate_bundle_L24k_brC")
    sliced = {tag: json.load(open(os.path.join(stage, f"sliced_benchmark{tag}.json")))
              for tag in ("", "_b64", "_cpu20") if os.path.exists(os.path.join(stage, f"sliced_benchmark{tag}.json"))}
    if sliced:
        result["sliced"] = sliced
    core = os.path.join(stage, "core.json")
    if os.path.exists(core):
        result["core_json_bytes"] = os.path.getsize(core)
    out = args.out or os.path.join(args.root, "results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
