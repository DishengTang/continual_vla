"""Train the policy on one task of the sequence, optionally protecting the previous task.

    python scripts/train_stage.py --task PushCube-v1 --out $CVLA_RUNS/dev/A
    python scripts/train_stage.py --task StackCube-v1 --init $CVLA_RUNS/dev/A --method er \
        --out $CVLA_RUNS/dev/B_er

Methods for the second stage:
  seqft  sequential fine-tuning on the new task only
  er     experience replay: a fixed fraction of every batch comes from the previous stage's bounded
         replay buffer (memory.safetensors; only those examples are stored and loaded), or is split
         evenly across several buffers (--memory-name memory /path/to/A/memory for a third task)
  ewc    elastic weight consolidation with a diagonal empirical-Fisher-style importance
  packnet  PackNet (continual_vla/packnet.py): a new task trains only weights no earlier task owns; with
         --packnet-prune the task just learned (the one in --init, same --task) keeps --packnet-keep of every
         weight matrix, its largest free weights, which are retrained for --steps updates and then frozen;
         with --packnet-release FRAC a new task first frees FRAC of each earlier task's weights (PackNet.release)

Core gating (consolidate-and-gate: consolidation, stabilisation and context-dependent gating):
  --core-from DIR   while (re)training a task, silence every action-expert unit group outside its
                    core (the --core-keep most important groups by DIR/ablation_scores.json, or a
                    random set with the same numbers of heads and MLP blocks) whenever the policy acts
                    on that task's instruction, so the skill consolidates into the core; writes core.json.
  --freeze-core     in a later stage, keep the core and everything else on the gated computation of
                    the earlier task fixed, so only its silenced groups learn the new task.
  --release FRAC    with --freeze-core, when earlier cores leave too few free groups: unfreeze this fraction of
                    the groups each earlier task added to the cores, its least important ones by
                    ablation_scores_<task>_gated.json in --init (compute_importance.py --task --gated), so the
                    new task can use them at the cost of changing part of the earlier tasks' computation.
  --core-selection bundle  attention-aware random core: in every layer --core-attn-keep of the key/value bundles
                    (a key/value head with the query heads that share it) and --core-keep of the MLP blocks.
  --adapter         with --freeze-core, while learning a new task: give its instruction its own additive copies of
                    the frozen state/action/time projections (continual_vla/adapter.py; stored as adapter.safetensors).
  --release-mode M  train (default): the earlier tasks keep using the released groups, which then change;
                    regate: the released groups are silenced for the earlier tasks too (written to core.json), so
                    they lose that capacity but are not changed by the new task.
  --distill LAMBDA  with --release, protect the released groups without old data: every update also runs the
                    current task's images with an earlier task's instruction (cycling, --distill-batch examples)
                    and adds LAMBDA x the squared difference between the action velocities the policy and the
                    --init checkpoint predict for the same noise and flow time (learning without forgetting).
  --branch-rank R   with core gating: give the current task's instruction its own rank-R attention branch
                    (continual_vla/branches.py: low-rank deltas on every expert attention projection, active only for
                    that instruction), trained in this and later stages of the same task and frozen afterwards;
                    branches in --init are always kept (stored as branches.safetensors).
  --teacher LAMBDA  with --core-from: every update also adds LAMBDA x the squared difference between the action
                    velocities the gated policy and the --init checkpoint with nothing silenced (the skill before
                    consolidation) predict for the same examples, noise and flow time, so the core learns to
                    reproduce the whole network and not only the demonstrations.
  --gate-ramp N     with --core-from: silence the task's non-core groups gradually, scaling their output from 1 to
                    0 over the first N updates, instead of all at once.
A stage whose --init has core.json keeps silencing that task's non-core groups for its instruction.
With --core-from and an --init that already has a core (a later task's consolidation), the new core
is chosen among the groups outside the earlier core, the task's instruction keeps the earlier core
plus its new one, and --freeze-core keeps the earlier task's gated computation fixed meanwhile.

A stage that stops early (e.g. the host crashes) resumes from its last resume.pt (every --ckpt-every updates) when
the same command is run again; the file is removed once the stage has finished.

Outputs in --out: config.json, policy_config.json, splits.json, log.jsonl (training), evals.jsonl
(closed-loop success and validation loss; intermediate evaluations on validation resets, the final
one on --final-split), trainable.safetensors, and for the next stage memory.{safetensors,json}
(replay buffer), fisher.safetensors and stage.json. Gated stages also write core.json/freeze.json.
"""

import argparse
import dataclasses
import json
import math
import os
import time

from continual_vla.tasks import SEQUENCE, SPLIT_SEEDS, TASKS

# The model stack is imported inside main(): simulator worker processes are spawned and re-import
# this module, and should not load torch models they never use.


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--init", default=None, help="output directory of the previous stage")
    p.add_argument("--method", default="seqft", choices=["seqft", "er", "ewc", "packnet"])
    p.add_argument("--packnet-prune", action="store_true", help="prune and retrain the task in --init (see above)")
    p.add_argument("--packnet-keep", type=float, default=0.25, help="fraction of every weight matrix a task keeps")
    p.add_argument("--packnet-share-projections", action="store_true",
                   help="state/action/time projections are shared like the 1-D parameters (as under the context gate)")
    p.add_argument("--packnet-release", type=float, default=0.0,
                   help="learning a new task: free this fraction of each earlier task's weights first (smallest magnitude)")
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
    p.add_argument("--replay-frac", type=float, default=0.25)
    p.add_argument("--memory-size", type=int, default=50, help="examples stored in this stage's replay buffer")
    p.add_argument("--memory-limit", type=int, default=None,
                   help="replay only this many of the previous stage's stored examples (random subset)")
    p.add_argument("--memory-name", nargs="+", default=["memory"],
                   help="replay buffer(s) to use: <name>.safetensors in --init, or an absolute path prefix "
                        "(see make_replay_buffer.py); several buffers share the replayed part of each batch")
    p.add_argument("--joint-task", nargs="+", default=None, choices=list(TASKS),
                   help="joint-training reference: also train on these tasks' full training splits, from the "
                        "pretrained policy, each with its --joint-frac of every batch")
    p.add_argument("--joint-frac", nargs="+", type=float, default=None)
    p.add_argument("--joint-n-train", nargs="+", type=int, default=200,
                   help="training demonstrations of each --joint-task (one value: all of them)")
    p.add_argument("--ewc-lambda", type=float, default=1.0)
    p.add_argument("--importance", default="fisher",
                   help="importance weights for the ewc penalty: <name>.safetensors in --init "
                        "(fisher, ablation, ablation_shuffled)")
    p.add_argument("--fisher-samples", type=int, default=300)
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
    p.add_argument("--core-from", default=None, help="stage directory with ablation_scores.json")
    p.add_argument("--core-keep", type=float, default=0.25, help="fraction of unit groups in the core")
    p.add_argument("--core-selection", default="ablation", choices=["ablation", "random", "bundle"])
    p.add_argument("--core-attn-keep", type=float, default=None, help="bundle selection: key/value bundles per layer")
    p.add_argument("--adapter", action="store_true", help="instruction-specific projection adapter (see above)")
    p.add_argument("--freeze-core", action="store_true")
    p.add_argument("--release", type=float, default=0.0, help="fraction of each earlier core to unfreeze (see above)")
    p.add_argument("--release-mode", choices=["train", "regate"], default="train",
                   help="train: the earlier tasks keep using the released groups; regate: they are silenced for them")
    p.add_argument("--distill", type=float, default=0.0, help="weight of the release distillation (see above)")
    p.add_argument("--distill-batch", type=int, default=8)
    p.add_argument("--branch-rank", type=int, default=0, help="rank of the task's attention branch (see above)")
    p.add_argument("--teacher", type=float, default=0.0, help="weight of the consolidation distillation (see above)")
    p.add_argument("--gate-ramp", type=int, default=0, help="updates over which consolidation silences (see above)")
    p.add_argument("--ckpt-every", type=int, default=2000, help="updates between resumable checkpoints (0: none)")
    p.add_argument("--exec-steps", type=int, default=5)
    p.add_argument("--n-envs", type=int, default=5, help="parallel simulator workers (~0.8 GB GPU each)")
    p.add_argument("--n-video", type=int, default=4)
    return p.parse_args(argv)


def lr_factor(step, args):
    if step < args.warmup:
        return (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    floor = args.min_lr / args.lr
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    from continual_vla.data import DemoSet, ReplayBuffer, load_splits, split_episodes
    from continual_vla.evaluation import evaluate
    from continual_vla.gating import ContextGate, core_trainable_masks, family_counts, select_bundles, select_core
    from continual_vla.importance import expert_groups
    from continual_vla.methods import Importance, fisher_diagonal
    from continual_vla.policy import (build_policy, load_trainable, raw_batch, save_trainable,
                                      trainable_named)

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prev_tasks, splits = [], {}
    if args.init:
        with open(os.path.join(args.init, "stage.json")) as f:
            prev_tasks = json.load(f)["tasks"]
        splits = load_splits(args.init)
    if args.joint_task:
        assert not args.init and args.method == "seqft" and len(args.joint_frac or []) == len(args.joint_task), \
            "joint training starts from the pretrained policy and needs one --joint-frac per --joint-task"
        n_train = args.joint_n_train if isinstance(args.joint_n_train, list) else [args.joint_n_train]
        n_train = n_train * len(args.joint_task) if len(n_train) == 1 else n_train
        for task, n in zip(args.joint_task, n_train):
            train_ids, val_ids = split_episodes(task, n, args.n_val)
            splits[task] = {"train": train_ids, "val": val_ids}
        prev_tasks = list(args.joint_task)
    seen = prev_tasks + ([args.task] if args.task not in prev_tasks else [])
    if args.task not in splits:
        train_ids, val_ids = split_episodes(args.task, args.n_train, args.n_val)
        splits[args.task] = {"train": train_ids, "val": val_ids}
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(splits, f)

    policy, pre, post = build_policy()
    with open(os.path.join(args.out, "policy_config.json"), "w") as f:
        json.dump(dataclasses.asdict(policy.config), f, indent=2, default=str)  # audit of the settings used
    if args.init:
        load_trainable(policy, os.path.join(args.init, "trainable.safetensors"))
    params = trainable_named(policy)
    chunk = policy.config.chunk_size

    # Context-dependent gating of action-expert unit groups (see module docstring).
    gate = core = earlier = frozen = None
    if args.init and os.path.exists(os.path.join(args.init, "core.json")):
        with open(os.path.join(args.init, "core.json")) as f:
            core = json.load(f)
    mlp_block = (core or {}).get("group_layout", {}).get("mlp_block", 64)  # group ids are relative to this layout
    if args.core_from:
        with open(os.path.join(args.core_from, "ablation_scores.json")) as f:
            saved = json.load(f)
        groups = expert_groups(policy, mlp_block)
        assert [list(g) for g in groups] == [list(g) for g in saved["groups"]], "group layout changed"
        earlier, core = core, None
        taken = earlier["core"] if earlier else []
        if args.core_selection == "bundle":
            wrapper = policy.model.vlm_with_expert
            members = select_bundles(groups, args.core_keep, args.core_attn_keep,
                                     wrapper.num_attention_heads // wrapper.num_key_value_heads, args.seed, exclude=taken)
        else:
            members = select_core(groups, saved["scores"], args.core_keep, args.core_selection, args.seed,
                                  exclude=taken)
        allowed = sorted(set(taken) | set(members))
        core = {"task": args.task, "instruction": TASKS[args.task].instruction, "core": allowed,
                "silenced": sorted(set(range(len(groups))) - set(allowed)),
                "selection": args.core_selection, "keep": args.core_keep, "attn_keep": args.core_attn_keep,
                "seed": args.seed, "group_layout": {"mlp_block": mlp_block, "groups": len(groups)}}
        if earlier:
            core.update(new=members, earlier=earlier.get("earlier", []) +
                        [{k: v for k, v in earlier.items() if k != "earlier"}])
    if core is not None:
        groups = expert_groups(policy, mlp_block)
        gate = ContextGate(policy, groups)
        for entry in core.get("earlier", []) + [core]:
            gate.silence(entry["instruction"], entry["silenced"])
            print(f"gating: {len(entry['silenced'])} of {len(groups)} unit groups silenced for "
                  f"'{entry['instruction']}' ({entry['selection']} core)", flush=True)
        with open(os.path.join(args.out, "core.json"), "w") as f:
            json.dump(core, f)
        if args.freeze_core:
            # Frozen entries get zero gradient; with zero weight decay AdamW then never moves them,
            # so no copy of their values is needed (checked against the init checkpoint at the end).
            assert args.weight_decay == 0, "--freeze-core requires --weight-decay 0"
            assert not args.core_from or earlier, "--freeze-core with --core-from needs a core in --init"
            protected = earlier["core"] if args.core_from else core["core"]
            released = []
            if args.release:
                assert not args.core_from, "--release applies when a new task is learned"
                taken = set()
                for entry in core.get("earlier", []) + [core]:  # the groups each task added, least important first
                    own = [g for g in entry.get("new", entry["core"]) if g not in taken]
                    taken |= set(entry["core"])
                    with open(os.path.join(args.init, f"ablation_scores_{entry['task']}_gated.json")) as f:
                        scores = json.load(f)["scores"]
                    released += sorted(own, key=lambda g: scores[g])[:int(round(args.release * len(own)))]
                protected = sorted(set(protected) - set(released))
                if args.release_mode == "regate":  # the earlier tasks stop using the released groups (less capacity, no interference)
                    for entry in core.get("earlier", []) + [core]:
                        entry["silenced"] = sorted(set(entry["silenced"]) | set(released))
                        entry["core"] = sorted(set(entry["core"]) - set(released))
                        gate.silence(entry["instruction"], entry["silenced"])
                    with open(os.path.join(args.out, "core.json"), "w") as f:
                        json.dump(core, f)
            frozen = {n: ~m for n, m in core_trainable_masks(policy, groups, protected).items()}
            counts = {"frozen": sum(int(m.sum()) for m in frozen.values()),
                      "plastic": sum(int((~m).sum()) for m in frozen.values()),
                      "core_groups": len(protected), "groups": len(groups), "released": sorted(released),
                      "by_module": family_counts(params, frozen)}
            with open(os.path.join(args.out, "freeze.json"), "w") as f:
                json.dump(counts, f)
            print(f"freeze-core: {counts['frozen']} frozen and {counts['plastic']} plastic parameters", flush=True)
    context_fn = gate.set_context if gate is not None else None
    adapter = None
    if args.adapter:
        from continual_vla.adapter import InterfaceAdapter
        assert gate is not None and args.freeze_core and not args.core_from, "--adapter is for learning a new task"
        adapter = InterfaceAdapter(policy, TASKS[args.task].instruction)
        print(f"adapter: {adapter.count()} task-specific projection parameters for '{adapter.instruction}'", flush=True)

        def context_fn(instructions, gated=gate.set_context):
            gated(instructions)
            adapter.set_context(instructions)
    branches = None
    if args.branch_rank or (args.init and os.path.exists(os.path.join(args.init, "branches.json"))):
        from continual_vla.branches import AttentionBranches
        assert gate is not None, "attention branches belong to core gating"
        branches = (AttentionBranches.load(args.init, policy) if args.init and
                    os.path.exists(os.path.join(args.init, "branches.json")) else AttentionBranches(policy))
        instruction = TASKS[args.task].instruction
        if args.branch_rank and instruction not in branches.instructions:
            branches.add(instruction, args.branch_rank)
        own = branches.owned(instruction)
        for n, p in branches.params.items():
            p.requires_grad_(n in own)
        print(f"branches: {len(branches.instructions)} ({branches.count(instruction)} trainable parameters for "
              f"'{instruction}')", flush=True)

        def context_fn(instructions, inner=context_fn):
            inner(instructions)
            branches.set_context(instructions)
    if args.distill:
        assert args.release, "--distill protects released groups"
        old_instructions = [entry["instruction"] for entry in core.get("earlier", []) + [core]]
    if args.teacher or args.gate_ramp:
        assert args.core_from and args.gate_ramp <= args.steps, "--teacher and --gate-ramp apply to a consolidation"
    if args.distill or args.teacher:
        teacher = {n: t.to(params[n].device) for n, t in load_file(os.path.join(args.init, "trainable.safetensors")).items()}
        velocity = {}
        policy.model.action_out_proj.register_forward_hook(lambda module, inputs, out: velocity.update(v=out))

    packnet = reference = None
    if args.method == "packnet":
        from continual_vla.packnet import PackNet
        assert gate is None and frozen is None and args.weight_decay == 0, "PackNet replaces the core options"
        packnet = PackNet.load(args.init, params) if args.init else PackNet(params)
        if args.packnet_prune:
            k = packnet.prune(TASKS[args.task].instruction, args.packnet_keep, args.packnet_share_projections)
            packnet.training_task = k
            trainable = packnet.trainable_masks(k)
        else:
            assert TASKS[args.task].instruction not in packnet.tasks, "a PackNet task is learned once"
            if args.packnet_release:
                print(f"packnet: released {packnet.release(args.packnet_release)} weights of earlier tasks", flush=True)
            trainable = packnet.trainable_masks()
        frozen = {n: ~m for n, m in trainable.items()}
        reference = {n: p.detach().cpu().clone() for n, p in params.items()}  # frozen values after pruning
        counts = {"frozen": sum(int(m.sum()) for m in frozen.values()),
                  "plastic": sum(int((~m).sum()) for m in frozen.values()),
                  "owned": {i + 1: packnet.count(i + 1) for i in range(len(packnet.tasks))},
                  "by_module": family_counts(params, frozen)}
        with open(os.path.join(args.out, "freeze.json"), "w") as f:
            json.dump(counts, f)
        print(f"packnet: tasks {packnet.tasks}; {counts['frozen']} frozen and {counts['plastic']} plastic parameters",
              flush=True)
        context_fn = packnet.set_context

    data = DemoSet(args.task, splits[args.task]["train"], chunk)
    val_sets = {task: DemoSet(task, splits[task]["val"], chunk) for task in seen}
    print(f"{args.task}: {len(splits[args.task]['train'])} training episodes ({len(data)} examples)", flush=True)

    memories, joint = [], None
    if args.method == "er":
        for name in args.memory_name:
            memories.append(ReplayBuffer(os.path.join(args.init, name), args.memory_limit, args.seed))
            print(f"replay buffer: {len(memories[-1])} examples of {memories[-1].task_id} "
                  f"({memories[-1].nbytes / 2**20:.1f} MiB)", flush=True)
    if args.joint_task:
        joint = [DemoSet(task, splits[task]["train"], chunk) for task in args.joint_task]
        for ds, task, frac in zip(joint, args.joint_task, args.joint_frac):
            print(f"joint training: {len(ds)} {task} examples, {round(args.batch * frac)} of every {args.batch}", flush=True)

    # Fisher-weighted drift is logged for every method; the ewc penalty uses --importance.
    importance = penalty = None
    if args.init and os.path.exists(os.path.join(args.init, "fisher.safetensors")):
        anchor = load_file(os.path.join(args.init, "trainable.safetensors"))
        importance = Importance(load_file(os.path.join(args.init, "fisher.safetensors")), anchor)
        if args.method == "ewc":
            penalty = importance if args.importance == "fisher" else Importance(
                load_file(os.path.join(args.init, f"{args.importance}.safetensors")), anchor)

    cfg = policy.config
    extra_params = (list(adapter.params.values()) if adapter is not None else []) + (
        [p for p in branches.params.values() if p.requires_grad] if branches is not None else [])
    optimizer = torch.optim.AdamW(list(params.values()) + extra_params, lr=args.lr, betas=cfg.optimizer_betas,
                                  eps=cfg.optimizer_eps, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, args))

    def make_batch(dataset, size=args.batch, extra=(), instruction=None):
        """Current-task examples, plus `extra` = [(replay buffer or joint dataset, count), ...]; `instruction`
        replaces the current task's instruction."""
        parts = [(dataset, size - sum(n for _, n in extra))] + list(extra)
        images, states, actions, pads, instructions = [], [], [], [], []
        for ds, n in parts:
            i, s, a, m = ds.get(torch.randint(len(ds), (n,)))
            images.append(i), states.append(s), actions.append(a), pads.append(m)
            instructions += [instruction if instruction and ds is dataset else ds.instruction] * n
        if context_fn is not None:
            context_fn(instructions)
        return pre(raw_batch(torch.cat(images), torch.cat(states), instructions,
                             torch.cat(actions), torch.cat(pads)))

    def loss_on(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch)
        return loss

    def distill_loss(step):
        """Squared difference of the predicted action velocities from the --init checkpoint's, for current-task
        examples with an earlier task's instruction (gated as that task)."""
        n = args.distill_batch
        batch = make_batch(data, size=n, instruction=old_instructions[step % len(old_instructions)])
        noise = policy.model.sample_noise((n, chunk, cfg.max_action_dim), "cuda")
        flow_time = policy.model.sample_time(n, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.func.functional_call(policy, teacher, (batch,), {"noise": noise, "time": flow_time})
        target = velocity["v"].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy.forward(batch, noise=noise, time=flow_time)
        dims = cfg.action_feature.shape[0]
        sq = (velocity["v"].float() - target)[:, :, :dims] ** 2
        valid = (~batch["action_is_pad"]).unsqueeze(-1).to(sq.dtype)
        return (sq * valid).sum() / (valid.sum() * dims).clamp_min(1)

    def teacher_loss(batch):
        """Flow-matching loss on the batch, and the squared difference of the gated policy's action velocities from
        those of the --init checkpoint with nothing silenced, for the same noise and flow time."""
        n = len(batch["action_is_pad"])
        noise = policy.model.sample_noise((n, chunk, cfg.max_action_dim), "cuda")
        flow_time = policy.model.sample_time(n, "cuda")
        instructions = gate.instructions
        gate.set_context([])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.func.functional_call(policy, teacher, (batch,), {"noise": noise, "time": flow_time})
        target = velocity["v"].float()
        gate.set_context(instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch, noise=noise, time=flow_time)
        dims = cfg.action_feature.shape[0]
        sq = (velocity["v"].float() - target)[:, :, :dims] ** 2
        valid = (~batch["action_is_pad"]).unsqueeze(-1).to(sq.dtype)
        return loss, (sq * valid).sum() / (valid.sum() * dims).clamp_min(1)

    @torch.no_grad()
    def validation_loss(dataset, n_batches=10):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(1234)  # same examples, noise and flow times at every evaluation
            return float(np.mean([loss_on(make_batch(dataset)).item() for _ in range(n_batches)]))

    def run_eval(step, tasks, n_episodes, split="val", video=False, required=False):
        """Closed-loop evaluation. The simulator workers render on the same GPU, so cached training
        memory is released first and a failed evaluation is retried; an intermediate evaluation
        that keeps failing is logged and skipped instead of ending the run."""
        policy.eval()
        torch.cuda.empty_cache()
        seeds = SPLIT_SEEDS[split][:n_episodes]
        for task in tasks:
            t0 = time.time()
            for attempt in range(3):
                try:
                    res = evaluate(policy, pre, post, task, seeds, n_envs=args.n_envs,
                                   exec_steps=args.exec_steps, n_video=args.n_video,
                                   video_dir=os.path.join(args.out, "videos") if video else None,
                                   context_fn=context_fn)
                    break
                except Exception as exc:  # e.g. renderer out of GPU memory, worker crash
                    print(f"[eval step {step}] {task}: attempt {attempt + 1} failed: {exc!r}", flush=True)
                    torch.cuda.empty_cache()
                    time.sleep(60)
            else:
                if required:
                    raise RuntimeError(f"evaluation of {task} at step {step} failed")
                continue
            if task in val_sets:
                res["val_loss"] = validation_loss(val_sets[task])
            res.update(step=step, split=split, stage_task=args.task, method=args.method,
                       gate="none" if gate is None else "on", eval_seconds=time.time() - t0)
            with open(os.path.join(args.out, "evals.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")
            wrong = f", wrong order {res['opposite_success']:.2f}" if "opposite_success" in res else ""
            print(f"[eval step {step} {split}] {task}: success {res['success']:.2f} "
                  f"(end {res['success_end']:.2f}){wrong}, val loss {res.get('val_loss', float('nan')):.4f}, "
                  f"{res['eval_seconds']:.0f}s", flush=True)
        policy.train()

    extra = []
    if memories:
        replayed = round(args.batch * args.replay_frac)
        extra = [(m, replayed // len(memories) + (i < replayed % len(memories))) for i, m in enumerate(memories)]
    if joint is not None:
        extra = [(ds, round(args.batch * frac)) for ds, frac in zip(joint, args.joint_frac)]

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
            for n, p in (adapter.params.items() if adapter is not None else ()):
                p.copy_(ckpt["adapter"][n].to(p.device))
            for n, p in (branches.params.items() if branches is not None else ()):
                p.copy_(ckpt["branches"][n].to(p.device))
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
        run_eval(0, args.final_tasks, args.final_episodes, split=args.final_split, required=True)
    if 0 in args.eval_steps and first == 1:
        run_eval(0, seen, args.eval_episodes)

    policy.train()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time() - elapsed
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    for step in range(first, args.steps + 1):
        if args.gate_ramp:
            gate.scale[core["instruction"]] = max(0.0, 1 - step / args.gate_ramp)
        distilled = distill_loss(step) if args.distill else None  # before the main batch sets the new task's gate
        batch = make_batch(data, extra=extra)
        if args.teacher:
            loss, distilled = teacher_loss(batch)
            total = loss + args.teacher * distilled
        else:
            loss = loss_on(batch)
            total = loss if distilled is None else loss + args.distill * distilled
        drift = None
        if penalty is not None:
            weighted = penalty.drift(params)
            total = loss + 0.5 * args.ewc_lambda * weighted
            if penalty is importance:
                drift = weighted
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if frozen is not None:
            for n, p in params.items():
                if p.grad is not None:
                    p.grad[frozen[n]] = 0
        grad_norm = torch.nn.utils.clip_grad_norm_(list(params.values()) + extra_params, cfg.optimizer_grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            if drift is None and importance is not None:
                with torch.no_grad():
                    drift = importance.drift(params)
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
            run_eval(step, seen, args.eval_episodes)
        elif step in args.eval_old_steps and step < args.steps:
            run_eval(step, [t for t in seen if t != args.task], args.eval_episodes)
        if step == args.stop_after:
            with open(os.path.join(args.out, "stage.json"), "w") as f:
                json.dump({"tasks": seen, "stopped_after": step, "train_seconds": time.time() - t_start}, f)
            return
        if args.ckpt_every and step % args.ckpt_every == 0 and step < args.steps:
            if packnet is not None:
                packnet.restore()
            state = {"args": vars(args), "step": step, "elapsed": time.time() - t_start,
                     "params": {n: p.detach().cpu() for n, p in params.items()},
                     "adapter": {n: p.detach().cpu() for n, p in adapter.params.items()} if adapter is not None else None,
                     "branches": {n: p.detach().cpu() for n, p in branches.params.items()} if branches is not None else None,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                     "rng_numpy": np.random.get_state()}
            torch.save(state, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)  # never a half-written checkpoint
            del state

    if gate is not None:
        gate.scale.clear()  # fully silenced from here on, as the saved core states
    if packnet is not None:
        packnet.restore()
        packnet.save(args.out)
    if adapter is not None:
        adapter.save(args.out)
    if branches is not None:
        branches.save(args.out)
    save_trainable(policy, os.path.join(args.out, "trainable.safetensors"))
    if frozen is not None:
        initial = reference or load_file(os.path.join(args.init, "trainable.safetensors"))
        with torch.no_grad():
            diff = max(float((p[frozen[n]] - initial[n].to(p.device)[frozen[n]]).abs().max())
                       if frozen[n].any() else 0.0 for n, p in params.items())
        print(f"freeze-core check: max change of frozen parameters vs init checkpoint = {diff:.3g}", flush=True)
    run_eval(args.steps, args.final_tasks, args.final_episodes, split=args.final_split, video=True,
             required=True)

    # Artifacts for protecting this task in the next stage: a bounded replay buffer of its examples
    # and a per-example squared-gradient importance estimate.
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(len(data), size=min(args.memory_size, len(data)), replace=False))
    ReplayBuffer.save(os.path.join(args.out, "memory"), data, torch.from_numpy(chosen))
    policy.train()
    fisher = fisher_diagonal(params, lambda: loss_on(make_batch(data, size=1)), args.fisher_samples)
    save_file({n: f.cpu().contiguous() for n, f in fisher.items()}, os.path.join(args.out, "fisher.safetensors"))
    with open(os.path.join(args.out, "stage.json"), "w") as f:
        json.dump({"tasks": seen, "train_seconds": time.time() - t_start}, f)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("done", flush=True)


if __name__ == "__main__":
    main()
