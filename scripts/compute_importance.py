"""Compute intervention-derived importance for a finished stage (and its shuffled control).

    python scripts/compute_importance.py --stage $CVLA_RUNS/dev1/A
    python scripts/compute_importance.py --stage $CVLA_RUNS/final/s0/B_core_rand --task PushCube-v1 --gated

Writes into the stage directory: ablation.safetensors, ablation_shuffled.safetensors (per-parameter
importance, mean 1) and ablation_scores.json (group-level loss increases). With --task (an earlier task
of the stage) and/or --gated (the stage's context gate on for that task's instruction, as the policy runs
it), only the scores are written, to ablation_scores_<task>[_gated].json.
"""

import argparse
import json
import os
import time


def main():
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from continual_vla.data import DemoSet, load_splits
    from continual_vla.importance import (ablation_scores, expert_groups, parameter_importance,
                                          shuffle_within_kind)
    from continual_vla.policy import build_policy, load_trainable, raw_batch

    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--n-batches", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--mlp-block", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", default=None, help="score on this task's training examples (default: the stage's task)")
    p.add_argument("--gated", action="store_true", help="with the stage's context gate on for the task's instruction")
    args = p.parse_args()
    torch.manual_seed(args.seed)  # fixed noise and flow times for every intervention

    with open(os.path.join(args.stage, "config.json")) as f:
        stage_task = json.load(f)["task"]
    task_id = args.task or stage_task
    policy, pre, _ = build_policy()
    load_trainable(policy, os.path.join(args.stage, "trainable.safetensors"))

    # Anchor examples: random training examples of the stage's task (computed at the end of that
    # stage, while its data is still available).
    data = DemoSet(task_id, load_splits(args.stage)[task_id]["train"], policy.config.chunk_size)
    g = torch.Generator().manual_seed(args.seed)
    batches = []
    for _ in range(args.n_batches):
        idx = torch.randint(len(data), (args.batch,), generator=g)
        i, s, a, m = data.get(idx)
        batch = pre(raw_batch(i, s, [data.instruction] * len(idx), a, m))
        noise = policy.model.sample_noise((len(idx), policy.config.chunk_size, policy.config.max_action_dim), "cuda")
        t = policy.model.sample_time(len(idx), "cuda")
        batches.append((batch, noise, t))

    groups = expert_groups(policy, args.mlp_block)
    if args.gated:
        from continual_vla.gating import ContextGate
        with open(os.path.join(args.stage, "core.json")) as f:
            core = json.load(f)
        gate = ContextGate(policy, groups)
        for entry in core.get("earlier", []) + [core]:
            gate.silence(entry["instruction"], entry["silenced"])
        gate.set_context([data.instruction] * args.batch)
    t0 = time.time()
    base, scores = ablation_scores(policy, batches, groups)
    print(f"{len(groups)} groups, base loss {base:.4f}, {time.time() - t0:.0f}s; "
          f"loss increase median {np.median(scores):.4f}, max {scores.max():.4f}, "
          f"negative {np.mean(scores < 0):.2f}")
    if args.task or args.gated:
        name = f"ablation_scores_{task_id}" + ("_gated" if args.gated else "") + ".json"
        with open(os.path.join(args.stage, name), "w") as f:
            json.dump({"base_loss": base, "groups": groups, "scores": scores.tolist(), "task": task_id,
                       "gated": args.gated}, f)
        return
    shuffled = shuffle_within_kind(groups, scores, args.seed)
    save_file({n: t.contiguous() for n, t in parameter_importance(policy, groups, scores).items()},
              os.path.join(args.stage, "ablation.safetensors"))
    save_file({n: t.contiguous() for n, t in parameter_importance(policy, groups, shuffled).items()},
              os.path.join(args.stage, "ablation_shuffled.safetensors"))
    with open(os.path.join(args.stage, "ablation_scores.json"), "w") as f:
        json.dump({"base_loss": base, "groups": groups, "scores": scores.tolist(),
                   "shuffled": shuffled.tolist()}, f)


if __name__ == "__main__":
    main()
