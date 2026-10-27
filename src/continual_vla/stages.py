"""Rebuild a trained stage from its output directory: policy weights and, if the stage was trained
with core gating (core.json, including the gates of earlier tasks under "earlier"), the
instruction-keyed context gate (and an instruction-specific projection adapter, adapter.safetensors, and per-task
attention branches, branches.safetensors), or with
PackNet (packnet.safetensors), its per-task weight masks."""

import json
import os

from .gating import ContextGate
from .importance import expert_groups
from .policy import build_policy, load_trainable, trainable_named


def load_stage(stage_dir, gate=True, device="cuda"):
    """Returns (policy, pre, post, context_fn, core). With gate=False the stage's gate is not
    installed, so every unit is active for every instruction (used to measure what the gate does)."""
    policy, pre, post = build_policy(device)
    load_trainable(policy, os.path.join(stage_dir, "trainable.safetensors"))
    core = None
    context_fn = None
    core_path = os.path.join(stage_dir, "core.json")
    if os.path.exists(core_path):
        with open(core_path) as f:
            core = json.load(f)
        if gate:
            # group ids refer to the layout the core was built with (64-unit MLP blocks before it was recorded)
            context_gate = ContextGate(policy, expert_groups(policy, core.get("group_layout", {}).get("mlp_block", 64)))
            for entry in core.get("earlier", []) + [core]:
                context_gate.silence(entry["instruction"], entry["silenced"])
            context_fn = context_gate.set_context
            if os.path.exists(os.path.join(stage_dir, "adapter.safetensors")):
                from .adapter import InterfaceAdapter
                adapter = InterfaceAdapter.load(stage_dir, policy)

                def context_fn(instructions, gated=context_gate.set_context):
                    gated(instructions)
                    adapter.set_context(instructions)
            if os.path.exists(os.path.join(stage_dir, "branches.json")):
                from .branches import AttentionBranches
                branches = AttentionBranches.load(stage_dir, policy)

                def context_fn(instructions, inner=context_fn):
                    inner(instructions)
                    branches.set_context(instructions)
    if gate and os.path.exists(os.path.join(stage_dir, "packnet.safetensors")):  # PackNet: per-task weight sets
        from .packnet import PackNet
        context_fn = PackNet.load(stage_dir, trainable_named(policy)).set_context
    return policy, pre, post, context_fn, core
