"""PackNet baseline (Mallya & Lazebnik, CVPR 2018): one network, disjoint sets of weights per task.

After a task is learned, each weight matrix keeps its largest-magnitude weights among those no earlier task owns (a
fixed fraction of the matrix per task); the rest are set to zero and freed. The kept weights are retrained briefly and
then frozen. A later task trains only free weights. A task is evaluated with the weights of itself and of earlier tasks;
weights owned by later tasks or still free are zeroed. One-dimensional parameters (biases, norms) are shared: trained
with the first task, then frozen; with share_projections the state/action/time projections are shared the same way, as
under the context gate. Task identity is the instruction, as for the context gate.

A stage of this method either prunes and retrains the task just learned (--packnet-prune, with --init the stage that
learned it and the same --task) or learns a new task on the free weights (optionally after --packnet-release).
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from ..tasks import TASKS
from .base import Method, clear_moments, family_counts

PROJECTIONS = ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp")
EXPERT = "model.paligemma_with_expert.gemma_expert.model.layers"  # as in motor_memory.py


class PackNet:
    def __init__(self, params, owner=None, tasks=()):
        self.params = params  # name -> parameter (the trainable action expert and projections)
        self.owner = owner or {n: torch.zeros_like(p, dtype=torch.uint8) for n, p in params.items()}
        self.tasks = list(tasks)  # instructions in owner order: task i owns weights with owner == i + 1
        self.saved = None  # unmasked weights while an earlier task's weights are in use
        self.active = 0  # 0: all weights (the task being learned)
        self.training_task = None  # the task being retrained after pruning: it uses all (non-free) weights

    @classmethod
    def load(cls, stage_dir, params):
        path = os.path.join(stage_dir, "packnet.safetensors")
        if not os.path.exists(path):
            return cls(params)
        owner = {n: t.to(params[n].device) for n, t in load_file(path).items()}
        with open(os.path.join(stage_dir, "packnet.json")) as f:
            return cls(params, owner, json.load(f)["tasks"])

    def save(self, stage_dir):
        save_file({n: t.cpu().contiguous() for n, t in self.owner.items()}, os.path.join(stage_dir, "packnet.safetensors"))
        with open(os.path.join(stage_dir, "packnet.json"), "w") as f:
            json.dump({"tasks": self.tasks, "owned": {i + 1: self.count(i + 1) for i in range(len(self.tasks))}}, f)

    def count(self, k):
        return sum(int((o == k).sum()) for o in self.owner.values())

    @torch.no_grad()
    def prune(self, instruction, keep, share_projections=False):
        """Assign the task its weights: in every matrix, the `keep` fraction of all entries with the largest magnitude
        among the free ones; the other free entries are zeroed. Returns the task's owner index."""
        self.restore()
        k = len(self.tasks) + 1
        self.tasks.append(instruction)
        for n, p in self.params.items():
            free = self.owner[n] == 0
            if p.dim() < 2 or (share_projections and any(m in n for m in PROJECTIONS)):  # shared: the first task's
                if k == 1:
                    self.owner[n][free] = 1
                continue
            n_keep = min(int(round(keep * p.numel())), int(free.sum()))
            if n_keep:
                scores = torch.where(free, p.abs(), torch.full_like(p, -1.0))
                top = torch.topk(scores.flatten(), n_keep).indices
                self.owner[n].view(-1)[top] = k
            p[self.owner[n] == 0] = 0
        return k

    @torch.no_grad()
    def release(self, fraction):
        """Free, in every matrix, the `fraction` of each earlier task's weights with the smallest magnitude, for a new
        task when too few weights are free; the earlier tasks lose them (zeroed, then trained by the new task). The
        state/action/time projections and the 1-D parameters stay with their owners. Returns the number released."""
        self.restore()
        released = 0
        for n, p in self.params.items():
            if p.dim() < 2 or any(m in n for m in PROJECTIONS):
                continue
            for k in range(1, len(self.tasks) + 1):
                own = self.owner[n] == k
                m = int(round(fraction * int(own.sum())))
                if m:
                    scores = torch.where(own, p.abs(), torch.full_like(p, float("inf")))
                    idx = torch.topk(scores.flatten(), m, largest=False).indices
                    self.owner[n].view(-1)[idx] = 0
                    p.view(-1)[idx] = 0
                    released += m
        return released

    def trainable_masks(self, k=None):
        """True = may change: the weights of task k (retraining after pruning; the first task also trains the shared
        parameters), or with k None the free weights (a new task)."""
        if k is None:
            return {n: (o == 0) & (self.params[n].dim() > 1) for n, o in self.owner.items()}
        return {n: (o == k) for n, o in self.owner.items()}

    @torch.no_grad()
    def restore(self):
        if self.saved is not None:
            for n, p in self.params.items():
                p.copy_(self.saved[n])
            self.saved = None
        self.active = 0

    @torch.no_grad()
    def set_context(self, instructions):
        """Weights for the instructions of the next forward passes (all the same task): an earlier task uses only the
        weights of tasks up to itself; any other instruction (the task being learned) uses every weight."""
        k = self.tasks.index(instructions[0]) + 1 if instructions[0] in self.tasks else 0
        assert all(i == instructions[0] for i in instructions), "PackNet evaluates one task per batch"
        if k == self.training_task:  # newest owner and free weights are zero: masking would change nothing
            k = 0
        if k == self.active:
            return
        self.restore()
        if k:
            self.saved = {n: p.detach().clone() for n, p in self.params.items()}
            for n, p in self.params.items():
                o = self.owner[n]
                p.mul_(((o >= 1) & (o <= k)).to(p.dtype))
            self.active = k


class PackNetMethod(Method):
    @staticmethod
    def add_args(group):
        group.add_argument("--packnet-prune", action="store_true",
                           help="prune the task in --init (same --task) to its --packnet-keep largest free weights "
                                "of every weight matrix, retrain those for --steps updates, then freeze them")
        group.add_argument("--packnet-keep", type=float, default=0.25, help="fraction of every weight matrix a task keeps")
        group.add_argument("--packnet-share-projections", action="store_true",
                           help="state/action/time projections are shared like the 1-D parameters (as under the context gate)")
        group.add_argument("--packnet-release", type=float, default=0.0,
                           help="learning a new task: free this fraction of each earlier task's weights first (smallest magnitude)")

    def setup(self):
        args, params = self.stage.args, self.stage.params
        assert args.weight_decay == 0, "PackNet keeps weights fixed by zeroing their gradients"
        packnet = self.packnet = PackNet.load(args.init, params) if args.init else PackNet(params)
        if args.packnet_prune:
            k = packnet.prune(TASKS[args.task].instruction, args.packnet_keep, args.packnet_share_projections)
            packnet.training_task = k
            trainable = packnet.trainable_masks(k)
        else:
            assert TASKS[args.task].instruction not in packnet.tasks, "a PackNet task is learned once"
            if args.packnet_release:
                print(f"packnet: released {packnet.release(args.packnet_release)} weights of earlier tasks", flush=True)
            trainable = packnet.trainable_masks()
        self.frozen = {n: ~m for n, m in trainable.items()}
        self.reference = {n: p.detach().cpu().clone() for n, p in params.items()}  # frozen values after pruning
        counts = {"frozen": sum(int(m.sum()) for m in self.frozen.values()),
                  "plastic": sum(int((~m).sum()) for m in self.frozen.values()),
                  "owned": {i + 1: packnet.count(i + 1) for i in range(len(packnet.tasks))},
                  "by_module": family_counts(params, self.frozen)}
        with open(os.path.join(args.out, "freeze.json"), "w") as f:
            json.dump(counts, f)
        print(f"packnet: tasks {packnet.tasks}; {counts['frozen']} frozen and {counts['plastic']} plastic parameters",
              flush=True)
        self.context_fn = packnet.set_context

    def before_checkpoint(self):
        self.packnet.restore()

    def finish(self):
        self.packnet.restore()
        self.packnet.save(self.stage.args.out)


class SequenceMasks(PackNet):
    """The weights' owners under LIBERO's pruning rule (libero/lifelong/algos/packnet.py): the task being learned holds
    every free weight, and pruning keeps those above the magnitude of rank prune_perc x (free weights) in each matrix,
    so a task keeps 1 - prune_perc of what is free. Weights that are not kept become free again, at zero."""

    @torch.no_grad()
    def prune(self, instruction, prune_perc, keep_share=0.0, shared=()):
        """With keep_share, a task keeps that share of every matrix instead: its largest-magnitude free weights, at most
        all free ones. The first task keeps every entry of the matrices in `shared`; later tasks find none of them free."""
        self.restore()
        k = len(self.tasks) + 1
        self.tasks.append(instruction)
        for n, p in self.params.items():
            free = self.owner[n] == 0
            if n in shared:
                self.owner[n][free] = k
                continue
            if keep_share:
                n_keep = min(int(round(keep_share * p.numel())), int(free.sum()))
                kept = torch.zeros_like(free)
                if n_keep:
                    scores = torch.where(free, p.abs(), torch.full_like(p, -1.0))
                    kept.view(-1)[torch.topk(scores.flatten(), n_keep).indices] = True
                self.owner[n][kept] = k
                p[self.owner[n] == 0] = 0
                continue
            values = p[free].abs()
            rank = round(prune_perc * values.numel())
            kept = free
            if rank:
                kept = free & ~(p.abs() <= values.view(-1).kthvalue(rank).values)
            self.owner[n][kept] = k
            p[self.owner[n] == 0] = 0
        return k


class SequencePackNet(Method):
    """PackNet in a sequence of tasks (training/train_libero.py), as LIBERO runs it: a task trains the free weights
    for --steps updates, is pruned, and its kept weights are retrained for --packnet-retrain-steps updates, then frozen.
    The weights of Linear and Conv2d layers are pruned and owned; every other parameter (biases, norms, embeddings) never
    trains. A task is evaluated with its own and earlier tasks' weights. With --packnet-shared-qk the action expert's
    query and key weights are not pruned: the first task keeps all of them (trained in both its phases) and later tasks
    leave them frozen, as the motor memory's value attention units do (--attention-units value)."""

    from_first_task = True

    @staticmethod
    def add_args(group):
        group.add_argument("--packnet-prune-perc", type=float, default=0.75,
                           help="fraction of the free weights of every matrix a task gives back (LIBERO's prune_perc)")
        group.add_argument("--packnet-keep-share", type=float, default=0.0,
                           help="a task keeps this share of every matrix instead (its largest-magnitude free weights); "
                                "0: LIBERO's rule (--packnet-prune-perc)")
        group.add_argument("--packnet-shared-qk", action="store_true",
                           help="the action expert's query and key weights belong to the first task in full (not pruned), "
                                "frozen for later tasks")
        group.add_argument("--packnet-retrain-steps", type=int, default=10000,
                           help="updates of the kept weights after pruning (LIBERO: as many as the task's training)")

    def setup(self):
        stage = self.stage
        args, params, policy = stage.args, stage.params, stage.policy
        masked = {f"{name}.weight" for name, m in policy.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))}
        self.masks = SequenceMasks({n: p for n, p in params.items() if n in masked})
        self.shared = {n for n in self.masks.params if n.startswith(EXPERT + ".")
                       and n.endswith((".self_attn.q_proj.weight", ".self_attn.k_proj.weight"))} if args.packnet_shared_qk else set()
        assert not args.packnet_shared_qk or self.shared, "no query/key weights of the action expert among the trainable ones"
        if args.init:
            self.masks = SequenceMasks.load(args.init, self.masks.params)
        self.instruction = stage.data.instruction
        assert self.instruction not in self.masks.tasks, "a PackNet task is learned once"
        self.known = set(self.masks.tasks) | {self.instruction}
        self.frozen = self._frozen(lambda o: o != 0)
        self.frozen_from_start = {n: m.clone() for n, m in self.frozen.items()}
        self.start = {n: p.detach().cpu().clone() for n, p in params.items()}
        self.context_fn = self._context
        self._count(f"tasks so far {len(self.masks.tasks)}")

    def _frozen(self, is_frozen):
        return {n: is_frozen(self.masks.owner[n]) if n in self.masks.owner else torch.ones_like(p, dtype=torch.bool)
                for n, p in self.stage.params.items()}

    def _count(self, when):
        counts = {"frozen": sum(int(m.sum()) for m in self.frozen.values()),
                  "plastic": sum(int((~m).sum()) for m in self.frozen.values()),
                  "owned": {i + 1: self.masks.count(i + 1) for i in range(len(self.masks.tasks))},
                  "by_module": family_counts(self.stage.params, self.frozen)}
        print(f"packnet ({when}): {counts['frozen']} frozen and {counts['plastic']} plastic parameters", flush=True)
        return counts

    def _context(self, instructions):
        assert instructions[0] in self.known, f"instruction not learned: {instructions[0]!r}"
        self.masks.set_context(instructions)

    def extra_updates(self):
        return self.stage.args.packnet_retrain_steps

    def prepare_optimizer(self, optimizer):
        clear_moments(optimizer, self.stage.params, self.frozen)

    def _retrain_phase(self, k, optimizer):
        self.masks.training_task = k
        self.frozen = self._frozen(lambda o: o != k)
        clear_moments(optimizer, self.stage.params, self.frozen)

    def after_learning(self, optimizer):
        args = self.stage.args
        self.stage.run_eval(args.steps, [self.stage.task])  # before pruning; the matrix uses the last evaluation
        k = self.masks.prune(self.instruction, args.packnet_prune_perc, args.packnet_keep_share, self.shared)
        self.masks.save(args.out)
        self._retrain_phase(k, optimizer)
        self._count(f"pruned, {self.masks.count(k)} weights kept")

    def on_resume(self, step, optimizer):
        args = self.stage.args
        if step > args.steps:
            self.masks = SequenceMasks.load(args.out, self.masks.params)
            self._retrain_phase(len(self.masks.tasks), optimizer)

    def before_checkpoint(self):
        self.masks.restore()

    @torch.no_grad()
    def finish(self):
        self.masks.restore()
        args, params = self.stage.args, self.stage.params
        changed = sum(int((params[n].cpu()[m.cpu()] != self.start[n][m.cpu()]).sum())
                      for n, m in self.frozen_from_start.items())
        nonzero_free = sum(int((params[n][o == 0] != 0).sum()) for n, o in self.masks.owner.items())
        assert changed == 0 and nonzero_free == 0, f"{changed} frozen weights changed, {nonzero_free} free weights not zero"
        self.masks.save(args.out)
        with open(os.path.join(args.out, "freeze.json"), "w") as f:
            json.dump({**self._count("end of task"), "frozen_weights_changed": changed}, f)
        del self.start
