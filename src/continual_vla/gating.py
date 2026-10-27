"""Instruction-dependent gating of action-expert unit groups (context-dependent gating).

Each instruction can be assigned a set of unit groups (attention heads and MLP unit blocks of the
action expert) that are silenced whenever the policy acts on that instruction. The instruction is a
permitted policy input, so the gate uses no privileged information.
"""

import numpy as np
import torch

from .importance import target_module


def select_bundles(groups, keep, attn_keep, kv_group, seed=0, exclude=()):
    """Attention-aware random core: in every layer, `attn_keep` of the key/value bundles (a key/value
    head with the `kv_group` query heads that share it, kept together so other bundles' key/value
    weights stay free) and `keep` of the MLP blocks, chosen among those no earlier core holds."""
    rng = np.random.default_rng(seed)
    taken = set(exclude)
    core = []
    for layer in sorted({g[1] for g in groups}):
        heads = [i for i, g in enumerate(groups) if g[1] == layer and g[0] == "head"]
        mlps = [i for i, g in enumerate(groups) if g[1] == layer and g[0] == "mlp"]
        bundles = [heads[b:b + kv_group] for b in range(0, len(heads), kv_group)]
        free = [b for b in bundles if not taken & set(b)]
        n_b, n_m = int(round(attn_keep * len(bundles))), int(round(keep * len(mlps)))
        free_mlp = [i for i in mlps if i not in taken]
        assert n_b <= len(free) and n_m <= len(free_mlp), f"layer {layer}: not enough free bundles or MLP blocks"
        for j in rng.choice(len(free), size=n_b, replace=False):
            core += free[j]
        core += [int(i) for i in rng.choice(free_mlp, size=n_m, replace=False)]
    return sorted(int(i) for i in core)


def module_family(name):
    """Coarse module family of an action-expert parameter, for frozen/plastic accounting."""
    for key, family in (("q_proj", "attention q/o"), ("o_proj", "attention q/o"), ("k_proj", "attention k/v"),
                        ("v_proj", "attention k/v"), ("mlp.", "mlp"), ("norm", "norms")):
        if key in name:
            return family
    if any(k in name for k in ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp")):
        return "state/action/time projections"
    return "other"


def family_counts(params, frozen):
    """{family: {"frozen": n, "plastic": n}} for the parameters in `params` under the `frozen` masks."""
    out = {}
    for n, p in params.items():
        c = out.setdefault(module_family(n), {"frozen": 0, "plastic": 0})
        f = int(frozen[n].sum()) if n in frozen else 0
        c["frozen"] += f
        c["plastic"] += p.numel() - f
    return out


def select_core(groups, scores, keep, selection="ablation", seed=0, exclude=()):
    """Indices of the `keep` fraction of unit groups forming a task's core: the most important by
    intervention-derived importance, or (control) the same number of heads and MLP blocks at random.
    Groups in `exclude` (earlier tasks' cores) are never chosen."""
    kinds = np.array([g[0] for g in groups])
    k = int(round(keep * len(groups)))
    available = np.ones(len(groups), dtype=bool)
    available[list(exclude)] = False
    assert k <= available.sum(), "not enough unit groups left for this core"
    core = np.argsort(-np.where(available, np.asarray(scores, dtype=float), -np.inf))[:k]
    if selection == "random":
        rng = np.random.default_rng(seed)
        core = np.concatenate([rng.choice(np.flatnonzero((kinds == kind) & available),
                                          size=int(np.sum(kinds[core] == kind)), replace=False)
                               for kind in ("head", "mlp")])
    elif selection != "ablation":
        raise ValueError(selection)
    return sorted(int(i) for i in core)


def core_trainable_masks(policy, groups, core):
    """Per-parameter masks (True = may change) for learning a new task without altering the gated
    computation of the task that owns `core`: core heads (query rows, output-projection columns and
    the key/value rows of their key/value heads), core MLP blocks and every parameter outside the
    unit groups are frozen; only non-core groups, which are silenced for that task, stay plastic."""
    wrapper = policy.model.vlm_with_expert
    prefix = "model.vlm_with_expert.lm_expert.layers"
    named = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    masks = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in named.items()}
    core = set(core)
    kv_groups = wrapper.num_attention_heads // wrapper.num_key_value_heads
    used_kv = set()
    for i, (kind, layer_idx, start, stop) in enumerate(groups):
        base = f"{prefix}.{layer_idx}"
        if kind == "head":
            head_dim = stop - start
            if i in core:
                used_kv.add((layer_idx, (start // head_dim) // kv_groups))
            else:
                masks[f"{base}.self_attn.q_proj.weight"][start:stop] = True
                masks[f"{base}.self_attn.o_proj.weight"][:, start:stop] = True
        elif i not in core:
            masks[f"{base}.mlp.gate_proj.weight"][start:stop] = True
            masks[f"{base}.mlp.up_proj.weight"][start:stop] = True
            masks[f"{base}.mlp.down_proj.weight"][:, start:stop] = True
    for layer_idx, layer in enumerate(wrapper.lm_expert.layers):
        kv_dim = layer.self_attn.k_proj.out_features // wrapper.num_key_value_heads
        for kv in range(wrapper.num_key_value_heads):
            if (layer_idx, kv) not in used_kv:
                for proj in ("k_proj", "v_proj"):
                    masks[f"{prefix}.{layer_idx}.self_attn.{proj}.weight"][kv * kv_dim:(kv + 1) * kv_dim] = True
    return masks


class ContextGate:
    def __init__(self, policy, groups):
        self.groups = groups
        self.silenced = {}  # instruction -> set of silenced group indices
        self.scale = {}  # instruction -> factor on its silenced groups while they are being silenced (0 when absent)
        self.instructions = None
        self._masks = {}
        self.slices = {}  # module -> [(group index, start, stop)]
        for i, (kind, layer_idx, start, stop) in enumerate(groups):
            self.slices.setdefault(target_module(policy, kind, layer_idx), []).append((i, start, stop))
        self.handles = [m.register_forward_pre_hook(self._hook) for m in self.slices]

    def silence(self, instruction, group_indices):
        self.silenced[instruction] = set(int(i) for i in group_indices)
        self._masks = {}

    def set_context(self, instructions):
        """Instructions of the samples in the next forward passes (one per batch element); none: nothing silenced."""
        self.instructions = list(instructions)
        self._masks = {}

    def _mask(self, module, features, device, dtype):
        if module not in self._masks:
            mask = torch.ones(len(self.instructions), features, device=device, dtype=dtype)
            for b, instruction in enumerate(self.instructions):
                off = self.silenced.get(instruction, ())
                value = self.scale.get(instruction, 0.0)
                for i, start, stop in self.slices[module]:
                    if i in off:
                        mask[b, start:stop] = value
            self._masks[module] = mask
        return self._masks[module]

    def _hook(self, module, inputs):
        if not self.instructions or not any(s in self.silenced for s in self.instructions):
            return None
        x = inputs[0]
        if x.shape[0] != len(self.instructions):
            raise ValueError(f"gate context has {len(self.instructions)} samples, batch has {x.shape[0]}")
        mask = self._mask(module, x.shape[-1], x.device, x.dtype)
        return (x * mask.view(x.shape[0], *([1] * (x.dim() - 2)), x.shape[-1]),) + tuple(inputs[1:])

    def remove(self):
        for h in self.handles:
            h.remove()
