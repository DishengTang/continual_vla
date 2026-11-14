"""Learn the tasks of a LIBERO suite one after another with a continual-learning method.

    python training/train_libero.py --suite libero_goal --method seqft --model pi0 --train expert --out $CVLA_RUNS/libero/...
    python training/train_libero.py --suite libero_goal --method er --replay-frac 0.5 --memory-size 1000 ... --out ...

The setup of Liu et al. (2026), from their released code (github.com/Continual-VLAs/continual-openpi): the suite's 10
tasks are learned in their order (suites.LIU_ORDERS; or --tasks), each from all of its demonstrations, starting from the
pretrained pi0, 10000 updates of batch 8 per task. One AdamW optimizer (betas 0.9/0.95, gradient clipping 1.0) and one
learning-rate schedule run through the whole sequence, as in their single training run: 1000 warmup updates to
2.5e-5, cosine decay to 2.5e-6 at update 30000 of the sequence, constant after. Each task's states and actions are
normalised with its own statistics, actions relative to the chunk's starting state (libero/policy.py), and training
images get openpi's augmentation. After each task the policy is evaluated on every task learned so far, 50 episodes
on LIBERO's fixed initial states (split "test"), as openpi's LIBERO evaluation (libero/evaluation.py), which gives the
success matrix and its metrics (continual_vla/libero/metrics.py).

Every task is a stage, trained by the same pipeline as training/train_stage.py and changed by the --method hooks of
src/continual_vla/methods/base.py:
  seqft  sequential fine-tuning
  er     1000 random examples of every earlier task (--memory-size); Liu et al. add a batch of 8 replayed examples, drawn
         from all of them together, to the 8 new ones: --batch 16 --replay-frac 0.5 --replay-sampling union (the first
         task, with nothing to replay, is learned with batch 8)
  ewc    online EWC as in LIBERO and Liu et al.'s configuration: the Fisher estimates of the earlier tasks (50 batches
         of 8, unscaled: --fisher-samples 50 --fisher-batch 8 --fisher-raw), decayed by --ewc-gamma 0.9 per task,
         anchored at the previous task's weights
  packnet       PackNet under LIBERO's rule: after --steps updates a task keeps 25% of the free weights of every matrix
                (--packnet-keep-share: a fixed share of every matrix instead), retrains them for as many updates and
                freezes them (methods/packnet.py, SequencePackNet); --packnet-shared-qk: the action expert's query
                and key weights belong to the first task in full, as with motor_memory's --attention-units value
  motor_memory  this project's method: after --steps updates a task's skill moves into a core of the action expert's
                units, sized by need, gated by its instruction and frozen (--train expert; methods/motor_memory.py)

A task's second phase (packnet, motor_memory) runs at the learning rate of its update --steps and does not count in the
schedule; with --phase2-lr restart it has a schedule of its own instead (--phase2-warmup updates to --lr, cosine to
--min-lr at its last update). Its evaluation at the end of the task gives the success matrix; packnet and motor_memory
also evaluate the task just before that phase (the record with step = --steps in evals.jsonl).

Outputs in --out: sequence.json, metrics.json (success matrix and FWT/NBT/AUC/final after every stage) and
stage_<k>/ per task: config.json, log.jsonl, evals.jsonl, trainable.safetensors, optimizer.pt (the optimizer state the
next task continues from; only the latest is kept), the method's files and stage.json.
A finished stage is skipped when the command is run again; an interrupted one resumes from its resume.pt.
"""

import argparse
import json
import math
import os
import time
from types import SimpleNamespace

from continual_vla.libero import suites

SUPPORTED = ("seqft", "er", "ewc", "packnet", "motor_memory")
# Options added after the first runs; at these values they are left out of a run's recorded options, so that the
# earlier runs, which recorded none of them, still resume.
LATER_OPTIONS = {"phase2_lr": "inherit", "phase2_warmup": 500, "core_share": 0.0, "packnet_keep_share": 0.0,
                 "unit_select": "random", "attention_units": "qkv", "packnet_shared_qk": False}


def run_config(args):
    """A run's recorded options (sequence.json, resume.pt): all but --stop-after-tasks and later options at their old
    behaviour."""
    return {k: v for k, v in vars(args).items()
            if k != "stop_after_tasks" and not (k in LATER_OPTIONS and v == LATER_OPTIONS[k])}


def libero_methods():
    """The methods of a sequence: train_stage.py's, with PackNet in its sequence form."""
    from continual_vla.methods import METHODS
    from continual_vla.methods.motor_memory import MotorMemory
    from continual_vla.methods.packnet import SequencePackNet

    return {**METHODS, "packnet": SequencePackNet, "motor_memory": MotorMemory}


def parse_args(argv=None):
    from continual_vla.libero.policy import TRAIN_MODES

    METHODS = libero_methods()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True, choices=suites.SUITES)
    p.add_argument("--tasks", nargs="+", type=int, default=None,
                   help="task indices in the order they are learned (default: Liu et al.'s order)")
    p.add_argument("--method", default="seqft", choices=SUPPORTED)
    p.add_argument("--model", default="pi0", choices=("pi0", "smolvla"))
    p.add_argument("--train", default="expert", choices=TRAIN_MODES, help="trainable parameters (libero/policy.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=10000, help="updates per task")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--min-lr", type=float, default=2.5e-6)
    p.add_argument("--warmup", type=int, default=1000, help="warmup updates at the start of the sequence")
    p.add_argument("--decay-steps", type=int, default=30000,
                   help="update of the sequence at which the cosine decay reaches --min-lr (constant after)")
    p.add_argument("--phase2-lr", default="inherit", choices=("inherit", "restart"),
                   help="learning rate of a task's second phase (packnet, motor_memory): that of its update --steps, or "
                        "a schedule of its own (--phase2-warmup updates to --lr, cosine to --min-lr at its last update)")
    p.add_argument("--phase2-warmup", type=int, default=500, help="warmup updates of the second phase (--phase2-lr restart)")
    p.add_argument("--weight-decay", type=float, default=1e-10)
    p.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.95])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-action-steps", type=int, default=5, help="actions executed from each predicted chunk")
    p.add_argument("--no-augment", action="store_true", help="train without openpi's image augmentation")
    p.add_argument("--eval-episodes", type=int, default=suites.N_TEST_EPISODES)
    p.add_argument("--eval-split", default="test", choices=("test", "val"))
    p.add_argument("--n-envs", type=int, default=10, help="parallel simulator processes")
    p.add_argument("--n-video", type=int, default=0, help="episodes recorded per evaluated task")
    p.add_argument("--frames-in-memory", action="store_true",
                   help="read the task's decoded frames into memory instead of memory-mapping them (network disks)")
    p.add_argument("--memory-size", type=int, default=1000, help="examples stored in each task's replay buffer (er)")
    p.add_argument("--fisher-samples", type=int, default=50, help="backward passes of each task's Fisher estimate (ewc)")
    p.add_argument("--ewc-gamma", type=float, default=0.9, help="decay of earlier tasks' Fisher per task (online EWC)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2500, help="updates between resumable checkpoints (0: none)")
    p.add_argument("--keep-weights", choices=("all", "last"), default="all",
                   help="keep every stage's trainable weights, or only the latest stage's")
    p.add_argument("--stop-after-tasks", type=int, default=0, help="end after this many tasks (0: all)")
    groups = {}
    for name in SUPPORTED:
        groups[name] = p.add_argument_group(f"--method {name}")
        METHODS[name].add_args(groups[name])
    args = p.parse_args(argv)
    if args.tasks is None:
        args.tasks = suites.liu_order(args.suite)
    for name, group in groups.items():
        used = [a.option_strings[0] for a in group._group_actions
                if getattr(args, a.dest) != a.default and a.dest != "memory_name"]
        if used and name != args.method:
            p.error(f"{', '.join(used)}: option(s) of --method {name}")
    return args


def lr_at(update, args):
    """Learning rate at `update` (0-based) of the whole sequence: optax.warmup_cosine_decay_schedule as openpi's
    CosineDecaySchedule builds it (linear warmup from lr / (warmup + 1), cosine to min_lr at decay_steps, then constant)."""
    if update < args.warmup:
        start = args.lr / (args.warmup + 1)
        return start + (args.lr - start) * update / args.warmup
    progress = min(1.0, (update - args.warmup) / max(1, args.decay_steps - args.warmup))
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def stage_lr(s, args, offset, phase2_updates):
    """Learning rate of a task's update s + 1 (s scheduler steps taken): the sequence's schedule, held at its value of
    update --steps in the second phase, or with --phase2-lr restart the second phase's own schedule."""
    if s < args.steps or args.phase2_lr == "inherit":
        return lr_at(offset + min(s, args.steps), args)
    phase2 = SimpleNamespace(lr=args.lr, min_lr=args.min_lr, warmup=args.phase2_warmup, decay_steps=phase2_updates)
    return lr_at(s - args.steps, phase2)


def stage_dir(out, k):
    return os.path.join(out, f"stage_{k:02d}")


class Stage:
    """One task of the sequence; the attributes and functions listed in continual_vla/methods/base.py are what a
    method uses."""

    def __init__(self, args, k):
        import numpy as np
        import torch
        from safetensors.torch import load_file

        from continual_vla.libero.data import LiberoDemos, LiberoReplayBuffer
        from continual_vla.libero.policy import build_policy, load_trainable, trainable_named
        from continual_vla.methods.ewc import Importance

        METHODS = libero_methods()
        self.replay_buffer = LiberoReplayBuffer
        order = [f"{args.suite}/{i}" for i in args.tasks]
        self.task, self.seen = order[k], order[:k + 1]
        out = stage_dir(args.out, k)
        init = stage_dir(args.out, k - 1) if k else None
        # The per-stage options a method reads (train_stage.py's): this task, the previous stage, this stage's directory
        # and, for replay, the buffers of every earlier task.
        self.k = k
        self.args = SimpleNamespace(**{**vars(args), "task": self.task, "init": init, "out": out,
                                       "memory_name": [os.path.join(stage_dir(args.out, j), "memory") for j in range(k)]})
        if args.method == "er" and k == 0:  # nothing to replay yet: the new examples alone, as in Liu et al.
            self.args.batch = args.batch - round(args.batch * args.replay_frac)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=2)
        torch.manual_seed(args.seed * 1000 + k)
        np.random.seed(args.seed * 1000 + k)

        self._stats = {}
        self.augment = not args.no_augment
        self.policy, self.pre, self.post = build_policy(args.model, args.train, args.n_action_steps)
        if init:
            load_trainable(self.policy, os.path.join(init, "trainable.safetensors"))
        self.params = trainable_named(self.policy)
        self.chunk = self.policy.config.chunk_size
        self.splits = {}
        self.data = LiberoDemos(self.task, self.chunk, in_memory=args.frames_in_memory)
        print(f"{self.task} ({self.data.instruction!r}): {len(self.data.episode_ids)} demonstrations, "
              f"{len(self.data)} examples; {sum(p.numel() for p in self.params.values()) / 1e6:.1f}M trainable "
              f"parameters ({args.train})", flush=True)
        # Online EWC importance over the earlier tasks, anchored at the previous stage's weights; its drift is logged
        # for every method whose earlier stages wrote a Fisher estimate.
        self.importance = None
        fishers = [os.path.join(stage_dir(args.out, j), "fisher.safetensors") for j in range(k)]
        if k and all(os.path.exists(f) for f in fishers):
            total = None
            for j, path in enumerate(fishers):
                weight = args.ewc_gamma ** (k - 1 - j)
                f = load_file(path)
                total = {n: weight * v for n, v in f.items()} if total is None else \
                    {n: total[n] + weight * v for n, v in f.items()}
            self.importance = Importance(total, load_file(os.path.join(init, "trainable.safetensors")))
        # The first task has nothing earlier to protect: it is learned by plain fine-tuning unless the method allocates
        # weights to it, and the method's files for the later tasks (its replay buffer, its Fisher estimate) are written
        # at its end.
        self.method = METHODS[args.method if k or METHODS[args.method].from_first_task else "seqft"](self)
        self.method.setup()

    def stats(self, task):
        """Normalisation statistics of a task (its own, as in Liu et al.)."""
        from continual_vla.libero.data import task_stats

        if task not in self._stats:
            self._stats[task] = task_stats(task, self.chunk)
        return self._stats[task]

    def make_batch(self, dataset, size=None, extra=(), instruction=None):
        """Examples of the current task plus `extra` = [(replay buffer, count), ...], each normalised with its own
        task's statistics; training images are augmented (not for the Fisher estimate, as in Liu et al.)."""
        import torch

        from continual_vla.libero.policy import normalise, raw_batch

        size = size or self.args.batch
        parts = [(dataset, size - sum(n for _, n in extra))] + list(extra)
        images, states, actions, pads, instructions = [], [], [], [], []
        for ds, n in parts:
            i, s, a, m = ds.get(torch.randint(len(ds), (n,)))
            s, a = normalise(s, a, self.stats(ds.task_id))
            images.append(i), states.append(s), actions.append(a), pads.append(m)
            instructions += [instruction if instruction and ds is dataset else ds.instruction] * n
        if self.method.context_fn is not None:
            self.method.context_fn(instructions)
        return self.pre(raw_batch(torch.cat(images), torch.cat(states), instructions, torch.cat(actions),
                                  torch.cat(pads), augment_images=self.augment))

    def loss_on(self, batch):
        from continual_vla.libero.policy import autocast, expert_loss

        with autocast():
            if self.args.model == "pi0" and self.args.train == "expert":
                return expert_loss(self.policy, batch)  # same loss without backpropagating into the frozen VLM
            loss, _ = self.policy.forward(batch)
        return loss

    def run_eval(self, step, tasks):
        import torch

        from continual_vla.libero.evaluation import evaluate

        args = self.args
        self.policy.eval()
        torch.cuda.empty_cache()
        for task in tasks:
            t0 = time.time()
            res = evaluate(self.policy, self.pre, self.post, task, self.stats(task), n_episodes=args.eval_episodes,
                           split=args.eval_split, n_envs=args.n_envs, seed=args.seed,
                           context_fn=self.method.context_fn, n_video=args.n_video,
                           video_dir=os.path.join(args.out, "videos", task.replace("/", "_")) if args.n_video else None)
            res.update(step=step, stage_task=self.task, method=args.method, eval_seconds=time.time() - t0)
            with open(os.path.join(args.out, "evals.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"[eval step {step} {res['split']}] {task}: success {res['success']:.2f} "
                  f"({res['eval_seconds']:.0f}s)", flush=True)
        self.policy.train()

    def run(self):
        import numpy as np
        import torch

        from continual_vla.libero.policy import save_trainable

        METHODS = libero_methods()
        args, policy, params, method = self.args, self.policy, self.params, self.method
        trained = list(params.values()) + list(method.extra_params.values())
        optimizer = torch.optim.AdamW(trained, lr=args.lr, betas=tuple(args.betas), eps=1e-8,
                                      weight_decay=args.weight_decay)
        # The optimizer of the whole sequence: its state (moments, update count) continues from the previous task.
        if args.init:
            optimizer.load_state_dict(torch.load(os.path.join(args.init, "optimizer.pt"), map_location="cpu",
                                                 weights_only=False))
        method.prepare_optimizer(optimizer)
        offset = self.k * args.steps  # updates of the sequence before this task
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: stage_lr(s, args, offset, method.extra_updates()) / args.lr)
        last_update = args.steps + method.extra_updates()
        ckpt_path, first, elapsed = os.path.join(args.out, "resume.pt"), 1, 0.0
        logs = [os.path.join(args.out, f) for f in ("log.jsonl", "evals.jsonl")]
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert ckpt["config"] == run_config(args), "resume.pt was written by a different command"
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(ckpt["params"][n].to(p.device))
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            torch.set_rng_state(ckpt["rng_cpu"])
            torch.cuda.set_rng_state(ckpt["rng_cuda"])
            np.random.set_state(ckpt["rng_numpy"])
            first, elapsed = ckpt["step"] + 1, ckpt["elapsed"]
            for path in logs:
                if os.path.exists(path):
                    kept = []
                    for line in open(path):
                        try:
                            if json.loads(line)["step"] <= ckpt["step"]:
                                kept.append(line if line.endswith("\n") else line + "\n")
                        except json.JSONDecodeError:
                            pass
                    open(path, "w").writelines(kept)
            print(f"resumed from update {ckpt['step']} of {last_update}", flush=True)
            method.on_resume(ckpt["step"], optimizer)
            del ckpt
        else:
            for path in logs:
                open(path, "w").close()

        policy.train()
        torch.cuda.reset_peak_memory_stats()
        t_start = time.time() - elapsed
        log = open(os.path.join(args.out, "log.jsonl"), "a")
        for step in range(first, last_update + 1):
            if step == args.steps + 1:
                method.after_learning(optimizer)
            loss, total, logged = method.loss(step)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if method.frozen is not None:
                for n, p in params.items():
                    if p.grad is not None:
                        p.grad[method.frozen[n]] = 0
            grad_norm = torch.nn.utils.clip_grad_norm_(trained, args.grad_clip)
            optimizer.step()
            scheduler.step()
            if step % args.log_every == 0:
                drift = logged.get("drift")
                if drift is None and self.importance is not None:
                    with torch.no_grad():
                        drift = self.importance.drift(params)
                rec = {"step": step, "loss": loss.item(), "grad_norm": grad_norm.item(),
                       "lr": scheduler.get_last_lr()[0], "elapsed": time.time() - t_start,
                       "drift": None if drift is None else float(drift),
                       "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % (args.log_every * 10) == 0:
                    print(json.dumps(rec), flush=True)
            if method.after_update(step, optimizer):
                break
            if args.ckpt_every and step % args.ckpt_every == 0 and step < last_update:
                method.before_checkpoint()
                state = {"config": run_config(args), "step": step,
                         "elapsed": time.time() - t_start,
                         "params": {n: p.detach().cpu() for n, p in params.items()},
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                         "rng_numpy": np.random.get_state()}
                torch.save(state, ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
                del state
        train_seconds = time.time() - t_start
        torch.save(optimizer.state_dict(), os.path.join(args.out, "optimizer.pt.tmp"))
        os.replace(os.path.join(args.out, "optimizer.pt.tmp"), os.path.join(args.out, "optimizer.pt"))
        del optimizer, scheduler
        torch.cuda.empty_cache()

        method.finish()
        save_trainable(policy, os.path.join(args.out, "trainable.safetensors"))
        self.run_eval(step, self.seen)
        # Only this run's method's files for the later tasks (the replay buffer for er, the Fisher estimate for ewc).
        self.augment = False
        METHODS[args.method].save_for_later_stages(self)
        with open(os.path.join(args.out, "stage.json"), "w") as f:
            json.dump({"task": self.task, "tasks": self.seen, "train_seconds": train_seconds,
                       "peak_gib": torch.cuda.max_memory_allocated() / 2**30}, f)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def success_matrix(out, order):
    """r[m][n]: success on task n after the stage that learned task m (NaN where not evaluated yet)."""
    import numpy as np

    r = np.full((len(order), len(order)), np.nan)
    for m in range(len(order)):
        path = os.path.join(stage_dir(out, m), "evals.jsonl")
        if not os.path.exists(os.path.join(stage_dir(out, m), "stage.json")):
            continue
        for line in open(path):
            e = json.loads(line)
            if e["step"] > 0 and e["task"] in order:
                r[m, order.index(e["task"])] = e["success"]
    return r


def main():
    import gc

    import torch

    from continual_vla.libero.metrics import sequence_metrics

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    order = [f"{args.suite}/{i}" for i in args.tasks]
    seq_path = os.path.join(args.out, "sequence.json")
    if os.path.exists(seq_path):
        with open(seq_path) as f:
            saved = json.load(f)
        assert saved["args"] == run_config(args), "--out holds a sequence run with different options"
    else:
        with open(seq_path, "w") as f:
            json.dump({"args": run_config(args), "order": order,
                       "instructions": [suites.instruction(t) for t in order]}, f, indent=2)
    n = len(order) if not args.stop_after_tasks else min(args.stop_after_tasks, len(order))
    for k in range(n):
        if os.path.exists(os.path.join(stage_dir(args.out, k), "stage.json")):
            continue
        stage = Stage(args, k)
        stage.run()
        del stage
        gc.collect()
        torch.cuda.empty_cache()
        if k:  # the optimizer state is only needed by the next task
            prev = os.path.join(stage_dir(args.out, k - 1), "optimizer.pt")
            if os.path.exists(prev):
                os.remove(prev)
        if args.keep_weights == "last" and k:
            prev = os.path.join(stage_dir(args.out, k - 1), "trainable.safetensors")
            if os.path.exists(prev):
                os.remove(prev)
        r = success_matrix(args.out, order)
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump({"order": order, "success_matrix": [[None if math.isnan(x) else x for x in row] for row in r],
                       **sequence_metrics(r)}, f, indent=2)
        print(f"after task {k + 1}/{len(order)}: {json.dumps(sequence_metrics(r))}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
