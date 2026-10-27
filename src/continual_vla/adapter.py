"""Instruction-specific interface adapter.

The core gate freezes the action expert's global state, action and flow-time projections after the first task,
since every task's computation uses them. The adapter gives one instruction its own additive copies of those
weights (zero-initialised deltas, so it starts from the frozen projections), active only when the policy acts on
that instruction; every other instruction's computation is unchanged. Its parameters are task-specific storage
and are counted separately (adapter.json).
"""

import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

PROJECTIONS = ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out")


class InterfaceAdapter:
    def __init__(self, policy, instruction, state=None):
        self.instruction = instruction
        self.active = False
        self.params = {}
        self.handles = []
        for name in PROJECTIONS:
            module = getattr(policy.model, name)
            self.params[f"{name}.weight"] = torch.nn.Parameter(torch.zeros_like(module.weight, dtype=torch.float32))
            if module.bias is not None:
                self.params[f"{name}.bias"] = torch.nn.Parameter(torch.zeros_like(module.bias, dtype=torch.float32))
            self.handles.append(module.register_forward_hook(self._hook(name)))
        if state is not None:
            with torch.no_grad():
                for n, p in self.params.items():
                    p.copy_(state[n].to(p.device))

    def _hook(self, name):
        def hook(module, inputs, out):
            if not self.active:
                return None
            x = inputs[0]
            bias = self.params.get(f"{name}.bias")
            return out + F.linear(x, self.params[f"{name}.weight"].to(x.dtype),
                                  None if bias is None else bias.to(x.dtype)).to(out.dtype)
        return hook

    def set_context(self, instructions):
        self.active = len(instructions) > 0 and all(i == self.instruction for i in instructions)

    def count(self):
        return sum(p.numel() for p in self.params.values())

    def save(self, stage_dir):
        save_file({n: p.detach().cpu().contiguous() for n, p in self.params.items()},
                  os.path.join(stage_dir, "adapter.safetensors"))
        with open(os.path.join(stage_dir, "adapter.json"), "w") as f:
            json.dump({"instruction": self.instruction, "modules": list(PROJECTIONS), "parameters": self.count(),
                       "bytes_fp32": 4 * self.count()}, f)

    @classmethod
    def load(cls, stage_dir, policy):
        with open(os.path.join(stage_dir, "adapter.json")) as f:
            meta = json.load(f)
        return cls(policy, meta["instruction"], load_file(os.path.join(stage_dir, "adapter.safetensors")))
