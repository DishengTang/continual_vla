"""PackNet baseline (Mallya & Lazebnik, CVPR 2018): one network, disjoint sets of weights per task.

After a task is learned, each weight matrix keeps its largest-magnitude weights among those no earlier task owns (a
fixed fraction of the matrix per task); the rest are set to zero and freed. The kept weights are retrained briefly and
then frozen. A later task trains only free weights. A task is evaluated with the weights of itself and of earlier tasks;
weights owned by later tasks or still free are zeroed. One-dimensional parameters (biases, norms) are shared: trained
with the first task, then frozen; with share_projections the state/action/time projections are shared the same way, as
under the context gate. Task identity is the instruction, as for the context gate.
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file

PROJECTIONS = ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp")


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
