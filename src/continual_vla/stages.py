"""Rebuild a trained stage from its output directory: policy weights and, if the stage was trained
with consolidate-and-gate (core.json, including the gates of earlier tasks under "earlier"), the
instruction-keyed context gate (and an instruction-specific projection adapter, adapter.safetensors, and per-task
attention branches, branches.safetensors), or with
PackNet (packnet.safetensors), its per-task weight masks."""

import json
import os

from .methods.consolidate_gate import load_gate
from .methods.packnet import PackNet
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
            context_fn = load_gate(stage_dir, policy, core)
    if gate and os.path.exists(os.path.join(stage_dir, "packnet.safetensors")):  # PackNet: per-task weight sets
        context_fn = PackNet.load(stage_dir, trainable_named(policy)).set_context
    return policy, pre, post, context_fn, core
