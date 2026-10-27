"""Intervention-derived parameter importance for selective consolidation.

After a task is learned, each group of action-expert units (one attention head, or a block of MLP
units) is silenced in turn, and the increase of that task's flow-matching loss is measured on a
fixed set of its examples with fixed noise and flow time. A group's importance is that loss
increase; every parameter that writes into or reads out of the group inherits it. Silencing is an
intervention on the network, so the score measures how much the task's behaviour depends on the
group, not just how sensitive the loss is to small parameter changes (as the Fisher does).

The matched control permutes the group scores among groups of the same kind (heads with heads,
MLP blocks with MLP blocks), keeping the importance distribution but breaking its assignment.
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


def shuffle_within_kind(groups, scores, seed=0):
    rng = np.random.default_rng(seed)
    shuffled = scores.copy()
    for kind in ("head", "mlp"):
        idx = np.array([i for i, g in enumerate(groups) if g[0] == kind])
        shuffled[idx] = scores[rng.permutation(idx)]
    return shuffled


def parameter_importance(policy, groups, scores):
    """Per-parameter importance from group scores (negative scores clipped to 0), normalised to
    mean 1 over all trainable parameters. Parameters outside any group get the mean group score."""
    wrapper = policy.model.vlm_with_expert
    scores = np.clip(scores, 0, None)
    default = float(scores.mean())
    named = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    imp = {n: torch.full_like(p, default, dtype=torch.float32) for n, p in named.items()}
    prefix = "model.vlm_with_expert.lm_expert.layers"
    kv_groups = wrapper.num_attention_heads // wrapper.num_key_value_heads
    for (kind, layer_idx, start, stop), s in zip(groups, scores):
        base = f"{prefix}.{layer_idx}"
        if kind == "head":
            imp[f"{base}.self_attn.q_proj.weight"][start:stop] = s
            imp[f"{base}.self_attn.o_proj.weight"][:, start:stop] = s
        else:
            imp[f"{base}.mlp.gate_proj.weight"][start:stop] = s
            imp[f"{base}.mlp.up_proj.weight"][start:stop] = s
            imp[f"{base}.mlp.down_proj.weight"][:, start:stop] = s
    # Key/value heads are shared by `kv_groups` query heads: use their mean score.
    for layer_idx, layer in enumerate(wrapper.lm_expert.layers):
        head_scores = np.array([s for (k, l, _, _), s in zip(groups, scores) if k == "head" and l == layer_idx])
        kv_dim = layer.self_attn.k_proj.out_features // wrapper.num_key_value_heads
        for kv in range(wrapper.num_key_value_heads):
            s = float(head_scores[kv * kv_groups:(kv + 1) * kv_groups].mean())
            for proj in ("k_proj", "v_proj"):
                imp[f"{prefix}.{layer_idx}.self_attn.{proj}.weight"][kv * kv_dim:(kv + 1) * kv_dim] = s
    mean = sum(t.sum() for t in imp.values()) / sum(t.numel() for t in imp.values())
    return {n: (t / mean).cpu() for n, t in imp.items()}
