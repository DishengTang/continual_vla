"""A task's gated sub-network as a narrower dense action expert.

Under the context gate a task computes with whole unit groups only: attention heads (with the key/value heads they
share) and 64-unit MLP blocks of the action expert. Removing the silenced rows and columns of those projections gives a
narrower expert that computes the same function with fewer operations, so a task's inference cost follows its
allocation. Per-weight masks (PackNet) leave every matrix at full size. Inference only (cached prefix, then denoising).
"""

import torch
from torch import nn


def _narrow(linear, idx, dim):
    """Copy of `linear` keeping output rows (dim 0) or input columns (dim 1) `idx`."""
    w = linear.weight.data.index_select(dim, idx)
    out = nn.Linear(w.shape[1], w.shape[0], bias=linear.bias is not None, device=w.device, dtype=w.dtype)
    out.weight.data.copy_(w)
    if linear.bias is not None:
        out.bias.data.copy_(linear.bias.data if dim == 1 else linear.bias.data[idx])
    return out


def _span(starts, width, device):
    return torch.cat([torch.arange(s, s + width, device=device) for s in starts])


def expert_size(policy):
    """Weights of the action expert and the state/action/time projections (the trainable part)."""
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


@torch.no_grad()
def slice_expert(policy, groups, silenced):
    """Narrow the action expert of `policy`, in place, to the unit groups not in `silenced`; the policy then runs that
    task without a gate. Heads must come in whole key/value bundles and every layer must keep the same numbers of heads
    (the attention takes one head count)."""
    wrapper = policy.model.vlm_with_expert
    assert policy.config.use_cache, "the narrow expert reads the cached prefix"
    full = (wrapper.num_attention_heads, wrapper.num_key_value_heads)
    kv_group = full[0] // full[1]
    silenced = set(int(i) for i in silenced)
    kept_kv, counts = {}, set()
    for layer_idx, layer in enumerate(wrapper.lm_expert.layers):
        att, mlp = layer.self_attn, layer.mlp
        mine = [(i, g) for i, g in enumerate(groups) if g[1] == layer_idx and i not in silenced]
        heads = sorted(start // att.head_dim for _, (kind, _, start, _) in mine if kind == "head")
        blocks = sorted((start, stop) for _, (kind, _, start, stop) in mine if kind == "mlp")
        kv = sorted({h // kv_group for h in heads})
        assert heads == [k * kv_group + j for k in kv for j in range(kv_group)], "heads must form whole bundles"
        dev = att.q_proj.weight.device
        q = _span([h * att.head_dim for h in heads], att.head_dim, dev)
        k = _span([h * att.head_dim for h in kv], att.head_dim, dev)
        att.q_proj, att.o_proj = _narrow(att.q_proj, q, 0), _narrow(att.o_proj, q, 1)
        att.k_proj, att.v_proj = _narrow(att.k_proj, k, 0), _narrow(att.v_proj, k, 0)
        assert all(stop - start == blocks[0][1] - blocks[0][0] for start, stop in blocks)
        m = _span([start for start, _ in blocks], blocks[0][1] - blocks[0][0], dev)
        mlp.gate_proj, mlp.up_proj = _narrow(mlp.gate_proj, m, 0), _narrow(mlp.up_proj, m, 0)
        mlp.down_proj = _narrow(mlp.down_proj, m, 1)
        kept_kv[layer_idx] = torch.tensor(kv, device=dev)
        counts.add((len(heads), len(kv)))
    assert len(counts) == 1, f"layers keep different numbers of heads: {counts}"
    narrow = counts.pop()
    # expert layers that attend jointly over [prefix; suffix] keys/values (the others cross-attend to the prefix
    # keys/values through their own key/value projections, which take all VLM key/value heads)
    joint = {i for i in range(len(wrapper.lm_expert.layers)) if "cross" not in wrapper.attention_mode
             or (wrapper.self_attn_every_n_layers > 0 and i % wrapper.self_attn_every_n_layers == 0)}
    forward = wrapper.forward

    def narrow_forward(attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                       use_cache=None):
        if inputs_embeds[0] is not None:  # prefix prefill: the frozen VLM, all heads
            assert inputs_embeds[1] is None, "the narrow expert supports inference only"
            outputs, cache = forward(attention_mask=attention_mask, position_ids=position_ids,
                                     past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache)
            for i in joint:  # the expert's queries read only their own key/value heads of the prefix
                cache.layers[i].keys = cache.layers[i].keys[:, kept_kv[i]]
                cache.layers[i].values = cache.layers[i].values[:, kept_kv[i]]
            return outputs, cache
        wrapper.num_attention_heads, wrapper.num_key_value_heads = narrow
        try:
            return forward(attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
                           inputs_embeds=inputs_embeds, use_cache=use_cache)
        finally:
            wrapper.num_attention_heads, wrapper.num_key_value_heads = full

    wrapper.forward = narrow_forward
    return policy
