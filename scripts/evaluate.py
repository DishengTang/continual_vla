"""Evaluate a trained stage in closed loop on fixed reset seeds.

    python scripts/evaluate.py --stage $CVLA_RUNS/final/s0/B_seqft --split test --episodes 100
    python scripts/evaluate.py --stage $CVLA_RUNS/final/s0/B_gate_rand --tasks PushCube-v1 --no-gate

A stage trained with core gating is evaluated with its gate restored, unless --no-gate is given.
Results are appended to <stage>/evals_<tag>.jsonl (the training-time evals.jsonl is not touched).
"""

import argparse
import json
import os
import time

from continual_vla.tasks import SEQUENCE, SPLIT_SEEDS


def main():
    import torch

    from continual_vla.evaluation import evaluate
    from continual_vla.stages import load_stage

    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--tasks", nargs="+", default=list(SEQUENCE))
    p.add_argument("--split", choices=list(SPLIT_SEEDS), default="test")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--exec-steps", type=int, default=5)
    p.add_argument("--n-envs", type=int, default=5)
    p.add_argument("--n-video", type=int, default=0)
    p.add_argument("--video-all", action="store_true", help="record a video of every episode (to choose examples)")
    p.add_argument("--no-gate", action="store_true", help="evaluate a gated stage with its gate disabled")
    p.add_argument("--save-traces", action="store_true",
                   help="also record each episode's per-step success (0/1 string per episode)")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    gated = not args.no_gate
    tag = args.tag or (f"{args.split}_exec{args.exec_steps}" + ("" if gated else "_nogate")
                       + ("_traces" if args.save_traces else ""))
    seeds = SPLIT_SEEDS[args.split][:args.episodes]

    policy, pre, post, context_fn, core = load_stage(args.stage, gate=gated)
    gate_state = "none" if core is None else ("on" if gated else "off")
    for task in args.tasks:
        t0 = time.time()
        res = evaluate(policy, pre, post, task, seeds, n_envs=args.n_envs, exec_steps=args.exec_steps,
                       n_video=args.n_video, context_fn=context_fn, record_traces=args.save_traces,
                       video_all=args.video_all,
                       video_dir=os.path.join(args.stage, f"videos_{tag}") if args.n_video or args.video_all else None)
        res.update(split=args.split, exec_steps=args.exec_steps, gate=gate_state, eval_seconds=time.time() - t0)
        with open(os.path.join(args.stage, f"evals_{tag}.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")
        wrong = f", wrong order {res['opposite_success']:.2f}" if "opposite_success" in res else ""
        print(f"{os.path.basename(args.stage.rstrip('/'))} {task} [{tag}, gate {gate_state}]: "
              f"success {res['success']:.2f} (end {res['success_end']:.2f}){wrong}, {res['eval_seconds']:.0f}s",
              flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
