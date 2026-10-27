"""Instruction-specific low-rank attention branches.

A task can own a branch: low-rank additive deltas (B A, with B zero-initialised, so the branch starts as the
unchanged network) on the query, key, value and output projections of every action-expert attention layer,
active only when the policy acts on that task's instruction. The branch is trained while its task is learned and
consolidated and frozen afterwards, so each task gets attention capacity of its own instead of competing for the
few key/value heads that earlier cores hold. Its parameters are task-specific storage, counted in branches.json.
"""

import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


class AttentionBranches:
    def __init__(self, policy):
        self.layers = policy.model.vlm_with_expert.lm_expert.layers
        self.instructions = []  # owner of each branch, in the order the tasks were learned
        self.rank = {}
        self.params = {}  # "<branch>.<layer>.<projection>.<a|b>" -> parameter
        self.context = []
        self.handles = [getattr(layer.self_attn, proj).register_forward_hook(self._hook(i, proj))
                        for i, layer in enumerate(self.layers) for proj in PROJECTIONS]

    def add(self, instruction, rank, state=None):
        k = len(self.instructions)
        self.instructions.append(instruction)
        self.rank[instruction] = rank
        for i, layer in enumerate(self.layers):
            for proj in PROJECTIONS:
                m = getattr(layer.self_attn, proj)
                a = torch.empty(rank, m.in_features, device=m.weight.device)
                torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                self.params[f"{k}.{i}.{proj}.a"] = torch.nn.Parameter(a)
                self.params[f"{k}.{i}.{proj}.b"] = torch.nn.Parameter(
                    torch.zeros(m.out_features, rank, device=m.weight.device))
        if state is not None:
            with torch.no_grad():
                for n, p in self.owned(instruction).items():
                    p.copy_(state[n].to(p.device))

    def owned(self, instruction):
        """Parameters of the instruction's branch."""
        if instruction not in self.instructions:
            return {}
        prefix = f"{self.instructions.index(instruction)}."
        return {n: p for n, p in self.params.items() if n.startswith(prefix)}

    def _hook(self, layer, proj):
        def hook(module, inputs, out):
            for k, instruction in enumerate(self.instructions):
                rows = [b for b, i in enumerate(self.context) if i == instruction]
                if not rows:
                    continue
                x = inputs[0]
                a, b = self.params[f"{k}.{layer}.{proj}.a"], self.params[f"{k}.{layer}.{proj}.b"]
                delta = F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)).to(out.dtype)
                if len(rows) < len(self.context):  # mixed instructions: only this instruction's rows
                    if x.shape[0] != len(self.context):
                        raise ValueError(f"branch context has {len(self.context)} samples, batch has {x.shape[0]}")
                    keep = torch.zeros(x.shape[0], *[1] * (x.dim() - 1), device=x.device, dtype=out.dtype)
                    keep[rows] = 1
                    delta = delta * keep
                out = out + delta
            return out
        return hook

    def set_context(self, instructions):
        self.context = list(instructions)

    def count(self, instruction):
        return sum(p.numel() for p in self.owned(instruction).values())

    def save(self, stage_dir):
        save_file({n: p.detach().cpu().contiguous() for n, p in self.params.items()},
                  os.path.join(stage_dir, "branches.safetensors"))
        with open(os.path.join(stage_dir, "branches.json"), "w") as f:
            json.dump({"instructions": self.instructions, "rank": self.rank, "modules": list(PROJECTIONS),
                       "parameters": {i: self.count(i) for i in self.instructions},
                       "bytes_fp32": {i: 4 * self.count(i) for i in self.instructions}}, f)

    @classmethod
    def load(cls, stage_dir, policy):
        with open(os.path.join(stage_dir, "branches.json")) as f:
            meta = json.load(f)
        state = load_file(os.path.join(stage_dir, "branches.safetensors"))
        branches = cls(policy)
        for instruction in meta["instructions"]:
            branches.add(instruction, meta["rank"][instruction], state)
        return branches
