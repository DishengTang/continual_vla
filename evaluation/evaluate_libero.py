"""Evaluate a trained LIBERO stage, or LeRobot's reference pi0 checkpoint, in closed loop.

    python evaluation/evaluate_libero.py --stage $CVLA_RUNS/libero/.../stage_03 --out evals.jsonl [--tasks ...]
    python evaluation/evaluate_libero.py --reference --tasks libero_goal --out reference_goal.jsonl

--tasks takes task names ("libero_goal/3") or whole suites ("libero_goal"); a stage defaults to the tasks it has
learned. --reference evaluates lerobot/pi0_libero_finetuned_v044 (pi0 fine-tuned by LeRobot on the four suites; its
own normalisation statistics), which checks that this evaluation reproduces LeRobot's LIBERO results.
"""

import argparse
import json
import os
import time

from continual_vla.libero import suites


def expand(tasks):
    return [t for name in tasks for t in (suites.suite_tasks(name) if name in suites.SUITES else [name])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--stage")
    src.add_argument("--reference", action="store_true")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--episodes", type=int, default=suites.N_TEST_EPISODES)
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--n-action-steps", type=int, default=None, help="default: the stage's own setting (reference: 10)")
    p.add_argument("--seed", type=int, default=0, help="seed of the policy's noise")
    p.add_argument("--n-video", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from continual_vla.libero.data import task_stats
    from continual_vla.libero.evaluation import evaluate
    from continual_vla.libero.policy import build_policy, build_processors, load_trainable

    if args.reference:
        from huggingface_hub import snapshot_download
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from safetensors.torch import load_file

        from lerobot.configs.policies import PreTrainedConfig

        path = snapshot_download("lerobot/pi0_libero_finetuned_v044")
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.compile_model = False  # its config compiles the model (max-autotune), which only delays evaluation
        cfg.n_action_steps = args.n_action_steps or 10
        policy = PI0Policy.from_pretrained(path, config=cfg)
        norm = load_file(os.path.join(path, "policy_preprocessor_step_5_normalizer_processor.safetensors"))
        stats = {k: {s: norm[f"{k}.{s}"] for s in ("mean", "std")} for k in ("observation.state", "action")}
        pre, post = build_processors(policy.config, "pi0", stats)
        policy.to("cuda")
        tasks = expand(args.tasks or list(suites.SUITES))
        source = {"reference": "lerobot/pi0_libero_finetuned_v044"}
        stats_of = lambda task: None  # noqa: E731 (its own processors normalise; absolute actions)
    else:
        with open(os.path.join(args.stage, "config.json")) as f:
            cfg = json.load(f)
        policy, pre, post = build_policy(cfg["model"], cfg["train"], args.n_action_steps or cfg["n_action_steps"])
        stats_of = lambda task: task_stats(task, policy.config.chunk_size)  # noqa: E731
        load_trainable(policy, os.path.join(args.stage, "trainable.safetensors"))
        with open(os.path.join(args.stage, "stage.json")) as f:
            learned = json.load(f)["tasks"]
        tasks = expand(args.tasks or learned)
        source = {"stage": os.path.abspath(args.stage)}
    policy.eval()
    for task in tasks:
        t0 = time.time()
        res = evaluate(policy, pre, post, task, stats_of(task), n_episodes=args.episodes, split=args.split, n_envs=args.n_envs,
                       seed=args.seed, n_video=args.n_video,
                       video_dir=os.path.join(os.path.dirname(os.path.abspath(args.out)), "videos",
                                              task.replace("/", "_")) if args.n_video else None)
        res.update(source, n_action_steps=policy.config.n_action_steps, eval_seconds=time.time() - t0)
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"{task}: success {res['success']:.2f} ({res['eval_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
