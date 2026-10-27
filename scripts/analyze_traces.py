"""Per-step success traces (evaluate.py --save-traces): when episodes first succeed and whether they stay
successful until the episode ends.

    python scripts/analyze_traces.py --root $CVLA_RUNS/final --stages B_seqft B_er50 B_gate_rand [--out traces.json]

For each stage and seed, over the test resets: success at any step and at the final step, the step of
first success, the fraction of successful episodes still successful at the final step (kept), the
fraction of control steps after first success spent in the success state (held), the number of
success-to-failure transitions per successful episode (losses), and the fraction of episodes in the
success state at every control step. Where the evaluation also recorded per-step states (S success,
H cube on the other cube but still held, M on it and released but moving, O not on it): the final-step
state of the episodes that succeeded at some step but not at the final step (lost), and how many of them
were re-grasped (held again after the last success, before the cube left the other cube).
"""

import argparse
import json
import os

import numpy as np


def load(stage_dir, task):
    path = os.path.join(stage_dir, "evals_test_exec5_traces.jsonl")
    if not os.path.exists(path):
        return None, None
    rows = [r for r in map(json.loads, open(path)) if r["task"] == task]
    if not rows:
        return None, None
    states = np.array([list(t) for t in rows[-1]["states"]]) if "states" in rows[-1] else None
    return np.array([[c == "1" for c in t] for t in rows[-1]["traces"]]), states


def summarize(traces, states=None):
    hit = traces.any(1)
    first = np.argmax(traces, 1)
    held = [traces[i, first[i]:].mean() for i in np.flatnonzero(hit)]
    losses = [int(np.sum(traces[i, :-1] & ~traces[i, 1:])) for i in np.flatnonzero(hit)]
    out = {"n": len(traces), "any": 100 * hit.mean(), "end": 100 * traces[:, -1].mean(),
           "first_step_median": float(np.median(first[hit])) if hit.any() else None,
           "kept": 100 * traces[hit, -1].mean() if hit.any() else None,
           "held": 100 * float(np.mean(held)) if held else None,
           "losses_mean": float(np.mean(losses)) if losses else None,
           "curve": (100 * traces.mean(0)).round(1).tolist()}
    if states is not None:
        lost = hit & ~traces[:, -1]
        out["lost"] = int(lost.sum())
        out["lost_final_state"] = {c: int(np.sum(states[lost, -1] == c)) for c in "HMO"}
        after = ["".join(states[i])["".join(states[i]).rfind("S") + 1:] for i in np.flatnonzero(lost)]
        out["lost_regrasped"] = sum("O" in a and "H" in a.split("O")[0] for a in after)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--stages", nargs="+", default=["B_seqft", "B_er50", "B_gate_rand"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--task", default="StackCube-v1")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    result = {}
    for stage in args.stages:
        for s in args.seeds:
            traces, states = load(os.path.join(args.root, f"s{s}", stage), args.task)
            if traces is None:
                continue
            r = result.setdefault(stage, {})[str(s)] = summarize(traces, states)
            lost = (f", lost {r['lost']} (final state {r['lost_final_state']}, re-grasped {r['lost_regrasped']})"
                    if "lost" in r else "")
            print(f"{stage} s{s}: any {r['any']:.0f}, end {r['end']:.0f}, first success step {r['first_step_median']}, "
                  f"kept {r['kept']:.0f}%, held {r['held']:.0f}%, losses {r['losses_mean']:.2f}{lost}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
