"""Train the policy on one task of the sequence with a continual-learning method.

    python training/train_stage.py --method seqft --task PushCube-v1 --out $CVLA_RUNS/dev/A
    python training/train_stage.py --method er --task StackCube-v1 --init $CVLA_RUNS/dev/A --out $CVLA_RUNS/dev/B_er

A stage trains on one task's demonstrations, from the pretrained policy or from an earlier stage (--init). This file is
the pipeline every method shares: data splits, batches, the flow-matching loss, AdamW with warmup and cosine decay,
closed-loop evaluations, resumable checkpoints and the files a later stage reads. --method picks one file of
src/continual_vla/methods/, which changes the stage through the hooks of methods/base.py:

  seqft             sequential fine-tuning on the new task only
  er                experience replay from bounded buffers of the earlier tasks
  ewc               elastic weight consolidation
  packnet           PackNet: disjoint weights per task, pruned and retrained after each task
  joint             joint training on several tasks at once (the non-sequential reference)
  consolidate_gate  consolidate-and-gate, this project's method: consolidate each task into a core of units, gate the
                    other units off for its instruction, freeze the core while later tasks are learned

Each method's options are listed under its name in --help; an option of another method is an error.

A stage that stops early (e.g. the host crashes) resumes from its last resume.pt (every --ckpt-every updates) when
the same command is run again; the file is removed once the stage has finished.

Outputs in --out: config.json, policy_config.json, splits.json, log.jsonl (training), evals.jsonl
(closed-loop success and validation loss; intermediate evaluations on validation resets, the final
one on --final-split), trainable.safetensors, and for a later stage memory.{safetensors,json}
(replay buffer), fisher.safetensors and stage.json. Methods add their own files (core.json, freeze.json, packnet.*).
"""

import argparse
import dataclasses
import json
import math
import os
import time

from continual_vla.tasks import SEQUENCE, SPLIT_SEEDS, TASKS

# The model stack is imported inside functions: simulator worker processes are spawned and re-import
# this module, and should not load torch models they never use.


def parse_args(argv=None):
    from continual_vla.methods import METHODS

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="seqft", choices=list(METHODS))
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--init", default=None, help="output directory of the previous stage")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=2.5e-6)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--n-train", type=int, default=200, help="training demonstrations of a new task")
    p.add_argument("--n-val", type=int, default=20, help="validation demonstrations of a new task")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-steps", nargs="+", type=int, default=[],
                   help="further updates at which to evaluate like --eval-every (0: before the first update)")
    p.add_argument("--eval-old-steps", nargs="+", type=int, default=[],
                   help="updates at which to evaluate only the earlier tasks (forgetting at a finer step)")
    p.add_argument("--stop-after", type=int, default=0,
                   help="end the stage after this many updates and their evaluation, with the learning-rate schedule "
                        "of --steps; only the evaluations and stage.json are written (learning curves)")
    p.add_argument("--final-episodes", type=int, default=50)
    p.add_argument("--final-split", choices=list(SPLIT_SEEDS), default="val")
    p.add_argument("--final-tasks", nargs="+", default=list(SEQUENCE))
    p.add_argument("--eval-init", action="store_true",
                   help="also evaluate the initial policy on --final-tasks (final split) before training")
    p.add_argument("--memory-size", type=int, default=50, help="examples stored in this stage's replay buffer")
    p.add_argument("--fisher-samples", type=int, default=300, help="examples of this stage's Fisher estimate")
    p.add_argument("--ckpt-every", type=int, default=2000, help="updates between resumable checkpoints (0: none)")
    p.add_argument("--exec-steps", type=int, default=5)
    p.add_argument("--n-envs", type=int, default=5, help="parallel simulator workers (~0.8 GB GPU each)")
    p.add_argument("--n-video", type=int, default=4)
    groups = {}
    for name, method in METHODS.items():
        groups[name] = p.add_argument_group(f"--method {name}")
        method.add_args(groups[name])
    args = p.parse_args(argv)
    for name, group in groups.items():
        used = [a.option_strings[0] for a in group._group_actions if getattr(args, a.dest) != a.default]
        if used and name != args.method:
            p.error(f"{', '.join(used)}: option(s) of --method {name}")
    return args


def lr_factor(step, args):
    if step < args.warmup:
        return (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    floor = args.min_lr / args.lr
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


class Stage:
    """One training stage; the attributes and functions listed in continual_vla/methods/base.py are what a method
    uses."""

    def __init__(self, args):
        import numpy as np
        import torch
        from safetensors.torch import load_file

        from continual_vla.data import DemoSet, load_splits, split_episodes
        from continual_vla.methods import METHODS
        from continual_vla.methods.ewc import Importance
        from continual_vla.policy import build_policy, load_trainable, trainable_named

        self.args = args
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.method = method = METHODS[args.method](self)

        prev_tasks, splits = [], {}
        if args.init:
            with open(os.path.join(args.init, "stage.json")) as f:
                prev_tasks = json.load(f)["tasks"]
            splits = load_splits(args.init)
        prev_tasks = method.earlier_tasks(prev_tasks, splits)
        self.seen = prev_tasks + ([args.task] if args.task not in prev_tasks else [])
        if args.task not in splits:
            train_ids, val_ids = split_episodes(args.task, args.n_train, args.n_val)
            splits[args.task] = {"train": train_ids, "val": val_ids}
        with open(os.path.join(args.out, "splits.json"), "w") as f:
            json.dump(splits, f)
        self.splits = splits

        self.policy, self.pre, self.post = build_policy()
        with open(os.path.join(args.out, "policy_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.policy.config), f, indent=2, default=str)  # audit of the settings used
        if args.init:
            load_trainable(self.policy, os.path.join(args.init, "trainable.safetensors"))
        self.params = trainable_named(self.policy)
        self.chunk = self.policy.config.chunk_size

        self.data = DemoSet(args.task, splits[args.task]["train"], self.chunk)
        self.val_sets = {task: DemoSet(task, splits[task]["val"], self.chunk) for task in self.seen}
        print(f"{args.task}: {len(splits[args.task]['train'])} training episodes ({len(self.data)} examples)", flush=True)
        # Fisher-weighted drift from the --init weights (ewc.py), logged for every method.
        self.importance = None
        if args.init and os.path.exists(os.path.join(args.init, "fisher.safetensors")):
            self.importance = Importance(load_file(os.path.join(args.init, "fisher.safetensors")),
                                         load_file(os.path.join(args.init, "trainable.safetensors")))
        method.setup()

    def make_batch(self, dataset, size=None, extra=(), instruction=None):
        """Current-task examples, plus `extra` = [(replay buffer or joint dataset, count), ...]; `instruction`
        replaces the current task's instruction."""
        import torch

        from continual_vla.policy import raw_batch

        size = size or self.args.batch
        parts = [(dataset, size - sum(n for _, n in extra))] + list(extra)
        images, states, actions, pads, instructions = [], [], [], [], []
        for ds, n in parts:
            i, s, a, m = ds.get(torch.randint(len(ds), (n,)))
            images.append(i), states.append(s), actions.append(a), pads.append(m)
            instructions += [instruction if instruction and ds is dataset else ds.instruction] * n
        if self.method.context_fn is not None:
            self.method.context_fn(instructions)
        return self.pre(raw_batch(torch.cat(images), torch.cat(states), instructions,
                                  torch.cat(actions), torch.cat(pads)))

    def loss_on(self, batch):
        import torch

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = self.policy.forward(batch)
        return loss

    def validation_loss(self, dataset, n_batches=10):
        import numpy as np
        import torch

        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(1234)  # same examples, noise and flow times at every evaluation
            return float(np.mean([self.loss_on(self.make_batch(dataset)).item() for _ in range(n_batches)]))

    def run_eval(self, step, tasks, n_episodes, split="val", video=False, required=False):
        """Closed-loop evaluation. The simulator workers render on the same GPU, so cached training
        memory is released first and a failed evaluation is retried; an intermediate evaluation
        that keeps failing is logged and skipped instead of ending the run."""
        import torch

        from continual_vla.evaluation import evaluate

        args, policy = self.args, self.policy
        policy.eval()
        torch.cuda.empty_cache()
        seeds = SPLIT_SEEDS[split][:n_episodes]
        for task in tasks:
            t0 = time.time()
            for attempt in range(3):
                try:
                    res = evaluate(policy, self.pre, self.post, task, seeds, n_envs=args.n_envs,
                                   exec_steps=args.exec_steps, n_video=args.n_video,
                                   video_dir=os.path.join(args.out, "videos") if video else None,
                                   context_fn=self.method.context_fn)
                    break
                except Exception as exc:  # e.g. renderer out of GPU memory, worker crash
                    print(f"[eval step {step}] {task}: attempt {attempt + 1} failed: {exc!r}", flush=True)
                    torch.cuda.empty_cache()
                    time.sleep(60)
            else:
                if required:
                    raise RuntimeError(f"evaluation of {task} at step {step} failed")
                continue
            if task in self.val_sets:
                res["val_loss"] = self.validation_loss(self.val_sets[task])
            res.update(step=step, split=split, stage_task=args.task, method=args.method,
                       gate="on" if self.method.gated else "none", eval_seconds=time.time() - t0)
            with open(os.path.join(args.out, "evals.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")
            wrong = f", wrong order {res['opposite_success']:.2f}" if "opposite_success" in res else ""
            print(f"[eval step {step} {split}] {task}: success {res['success']:.2f} "
                  f"(end {res['success_end']:.2f}){wrong}, val loss {res.get('val_loss', float('nan')):.4f}, "
                  f"{res['eval_seconds']:.0f}s", flush=True)
        policy.train()

    def run(self):
        import numpy as np
        import torch
        from safetensors.torch import load_file

        from continual_vla.methods import METHODS
        from continual_vla.policy import save_trainable

        args, policy, params, method = self.args, self.policy, self.params, self.method
        cfg = policy.config
        trained = list(params.values()) + list(method.extra_params.values())
        optimizer = torch.optim.AdamW(trained, lr=args.lr, betas=cfg.optimizer_betas,
                                      eps=cfg.optimizer_eps, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, args))

        # Resume an interrupted run of the same command: weights, optimizer, schedule and random state at its last
        # checkpoint; log and evaluation lines written after it are dropped. A fresh start clears a crashed run's logs.
        ckpt_path, first, elapsed = os.path.join(args.out, "resume.pt"), 1, 0.0
        logs = [os.path.join(args.out, f) for f in ("log.jsonl", "evals.jsonl")]
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            paths = ("init", "out", "core_from", "memory_name")  # a run may resume on another host's directory tree
            same = lambda d: {k: v for k, v in d.items() if k not in paths}  # noqa: E731
            # options added after the checkpoint was written count as their defaults
            saved = same({**vars(parse_args(["--task", args.task, "--out", args.out])), **ckpt["args"]})
            now = same(vars(args))
            changed = sorted(k for k in saved.keys() | now.keys() if saved.get(k) != now.get(k))
            assert not changed, f"resume.pt was written by a different command ({', '.join(changed)} differ)"
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(ckpt["params"][n].to(p.device))
                for n, p in method.extra_params.items():
                    p.copy_(ckpt["extra_params"][n].to(p.device))
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            torch.set_rng_state(ckpt["rng_cpu"])
            torch.cuda.set_rng_state(ckpt["rng_cuda"])
            np.random.set_state(ckpt["rng_numpy"])
            first, elapsed = ckpt["step"] + 1, ckpt["elapsed"]
            for path in logs:
                if os.path.exists(path):
                    kept = []
                    for l in open(path):
                        try:  # a line the crash cut off is dropped
                            if json.loads(l)["step"] <= ckpt["step"]:
                                kept.append(l if l.endswith("\n") else l + "\n")
                        except json.JSONDecodeError:
                            pass
                    open(path, "w").writelines(kept)
            print(f"resumed from update {ckpt['step']} of {args.steps}", flush=True)
            del ckpt
        else:
            for path in logs:
                open(path, "w").close()
        if args.eval_init and first == 1:
            self.run_eval(0, args.final_tasks, args.final_episodes, split=args.final_split, required=True)
        if 0 in args.eval_steps and first == 1:
            self.run_eval(0, self.seen, args.eval_episodes)

        policy.train()
        torch.cuda.reset_peak_memory_stats()
        t_start = time.time() - elapsed
        log = open(os.path.join(args.out, "log.jsonl"), "a")
        for step in range(first, args.steps + 1):
            loss, total, logged = method.loss(step)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if method.frozen is not None:
                for n, p in params.items():
                    if p.grad is not None:
                        p.grad[method.frozen[n]] = 0
            grad_norm = torch.nn.utils.clip_grad_norm_(trained, cfg.optimizer_grad_clip_norm)
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                drift, distilled = logged.get("drift"), logged.get("distill")
                if drift is None and self.importance is not None:
                    with torch.no_grad():
                        drift = self.importance.drift(params)
                rec = {"step": step, "loss": loss.item(), "grad_norm": grad_norm.item(),
                       "lr": scheduler.get_last_lr()[0], "elapsed": time.time() - t_start,
                       "drift": None if drift is None else drift.item(),
                       "distill": None if distilled is None else distilled.item(),
                       "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % (args.log_every * 10) == 0:
                    print(json.dumps(rec), flush=True)
            if (step % args.eval_every == 0 or step in args.eval_steps) and step < args.steps:
                self.run_eval(step, self.seen, args.eval_episodes)
            elif step in args.eval_old_steps and step < args.steps:
                self.run_eval(step, [t for t in self.seen if t != args.task], args.eval_episodes)
            if step == args.stop_after:
                with open(os.path.join(args.out, "stage.json"), "w") as f:
                    json.dump({"tasks": self.seen, "stopped_after": step, "train_seconds": time.time() - t_start}, f)
                return
            if args.ckpt_every and step % args.ckpt_every == 0 and step < args.steps:
                method.before_checkpoint()
                state = {"args": vars(args), "step": step, "elapsed": time.time() - t_start,
                         "params": {n: p.detach().cpu() for n, p in params.items()},
                         "extra_params": {n: p.detach().cpu() for n, p in method.extra_params.items()},
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                         "rng_numpy": np.random.get_state()}
                torch.save(state, ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)  # never a half-written checkpoint
                del state

        method.finish()
        save_trainable(policy, os.path.join(args.out, "trainable.safetensors"))
        if method.frozen is not None:
            frozen = method.frozen
            initial = method.reference or load_file(os.path.join(args.init, "trainable.safetensors"))
            with torch.no_grad():
                diff = max(float((p[frozen[n]] - initial[n].to(p.device)[frozen[n]]).abs().max())
                           if frozen[n].any() else 0.0 for n, p in params.items())
            print(f"freeze check: max change of frozen parameters vs init checkpoint = {diff:.3g}", flush=True)
        self.run_eval(args.steps, args.final_tasks, args.final_episodes, split=args.final_split, video=True,
                      required=True)

        # Files for protecting this task in a later stage, whatever that stage's method: a bounded replay buffer of its
        # examples (er.py) and a per-example squared-gradient importance estimate (ewc.py).
        for later in METHODS.values():
            later.save_for_later_stages(self)
        with open(os.path.join(args.out, "stage.json"), "w") as f:
            json.dump({"tasks": self.seen, "train_seconds": time.time() - t_start}, f)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        print("done", flush=True)


def main():
    Stage(parse_args()).run()


if __name__ == "__main__":
    main()
