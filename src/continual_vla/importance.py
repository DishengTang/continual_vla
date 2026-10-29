"""Unit groups of the action expert and their intervention-derived importance.

The action expert's units are grouped into attention heads and blocks of MLP units (expert_groups); consolidate-and-gate
silences, keeps and freezes whole groups. After a task is learned, each group is silenced in turn and the increase of
that task's flow-matching loss is measured on a fixed set of its examples with fixed noise and flow time
(ablation_scores, run by training/compute_importance.py). Silencing is an intervention on the network, so the score
measures how much the task's behaviour depends on the group, not just how sensitive the loss is to small parameter
changes (as the Fisher does). The scores choose importance-selected cores and the order in which cores are released.
"""

import numpy as np
import torch


def expert_groups(policy, mlp_block=64):
    """(kind, layer, start, stop) for every attention head and MLP unit block of the action expert."""
    wrapper = policy.model.vlm_with_expert
    groups = []
    for layer_idx, layer in enumerate(wrapper.lm_expert.layers):
        head_dim = layer.self_attn.o_proj.in_features // wrapper.num_attention_heads
        for h in range(wrapper.num_attention_heads):
            groups.append(("head", layer_idx, h * head_dim, (h + 1) * head_dim))
        inter = layer.mlp.down_proj.in_features
        for start in range(0, inter, mlp_block):
            groups.append(("mlp", layer_idx, start, min(start + mlp_block, inter)))
    return groups


def target_module(policy, kind, layer_idx):
    layer = policy.model.vlm_with_expert.lm_expert.layers[layer_idx]
    return layer.self_attn.o_proj if kind == "head" else layer.mlp.down_proj


@torch.no_grad()
def ablation_scores(policy, batches, groups):
    """Loss increase when each group is silenced. `batches` are (batch, noise, time) triples so that
    every intervention is compared against identical noise and flow time."""
    def mean_loss():
        total = 0.0
        for batch, noise, time in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch, noise=noise, time=time)
            total += loss.item()
        return total / len(batches)

    policy.eval()
    base = mean_loss()
    scores = np.zeros(len(groups))
    for i, (kind, layer_idx, start, stop) in enumerate(groups):
        def silence(module, inputs, start=start, stop=stop):
            x = inputs[0].clone()
            x[..., start:stop] = 0
            return (x,) + tuple(inputs[1:])
        handle = target_module(policy, kind, layer_idx).register_forward_pre_hook(silence)
        try:
            scores[i] = mean_loss() - base
        finally:
            handle.remove()
    return base, scores

