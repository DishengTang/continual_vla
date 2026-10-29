"""Consolidate-and-gate, this project's method: each learned skill is re-expressed in a small core of the action
expert's units, the other units are switched off whenever the policy acts on that skill's instruction, and the core is
frozen while later skills are learned (context-dependent gating with synaptic stabilisation, Masse, Grant & Freedman,
PNAS 2018). The instruction is a permitted policy input, so the gate uses no privileged information.

Units are unit groups of the action expert (importance.expert_groups): in each layer, every attention head and every
block of 64 MLP units. A stage of this method does one of the following, by its options:

  consolidate       --core-from DIR, with --init the stage that learned the task and the same --task: choose the task's
                    core (--core-keep of the groups: the most important by DIR/ablation_scores.json, the same numbers of
                    heads and MLP blocks at random, or attention-aware bundles) and keep training the task with every
                    other group silenced for its instruction, so the skill moves into the core; writes core.json. With a
                    core already in --init (a later task's consolidation), the new core is chosen among the groups
                    outside the earlier core, the task's instruction keeps the earlier core plus its new one, and
                    --freeze-core keeps the earlier task's gated computation fixed meanwhile.
  learn a new task  --init a consolidated stage: the earlier tasks stay gated, and with --freeze-core every weight on
                    their gated computation is frozen, so only groups they never use learn the new task (without it,
                    the ablation "gate on, not frozen").
  over capacity     --release FRAC with --freeze-core, when the earlier cores leave too few free groups: unfreeze the
                    least important part of the groups each earlier task added (--release-mode, --distill).

Options explored on top: an instruction-specific interface adapter (--adapter, InterfaceAdapter) and low-rank
attention branches (--branch-rank, AttentionBranches). slice_expert cuts a task's gated sub-network out as a narrower
dense action expert (evaluation/benchmark_sliced.py).
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from ..importance import expert_groups, target_module
from ..tasks import TASKS
from .base import Method, family_counts


class ContextGate:
    """Instruction-dependent gating of unit groups: each instruction can be assigned a set of groups that are silenced
    whenever the policy acts on it (their inputs to the attention output projection or MLP down projection are
    zeroed), per sample of a batch."""

    def __init__(self, policy, groups):
        self.groups = groups
        self.silenced = {}  # instruction -> set of silenced group indices
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
                for i, start, stop in self.slices[module]:
                    if i in off:
                        mask[b, start:stop] = 0.0
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


def gate_from_core(policy, core):
    """The context gate of a stage's core.json: every task's non-core groups silenced for its instruction (the earlier
    tasks' gates are listed under "earlier"; group ids refer to the layout the core was built with)."""
    gate = ContextGate(policy, expert_groups(policy, core.get("group_layout", {}).get("mlp_block", 64)))
    for entry in core.get("earlier", []) + [core]:
        gate.silence(entry["instruction"], entry["silenced"])
    return gate


def load_gate(stage_dir, policy, core):
    """Context function of a gated stage: its gate, and its adapter and attention branches if it has them."""
    context_fn = gate_from_core(policy, core).set_context
    if os.path.exists(os.path.join(stage_dir, "adapter.safetensors")):
        adapter = InterfaceAdapter.load(stage_dir, policy)

        def context_fn(instructions, gated=context_fn):
            gated(instructions)
            adapter.set_context(instructions)
    if os.path.exists(os.path.join(stage_dir, "branches.json")):
        branches = AttentionBranches.load(stage_dir, policy)

        def context_fn(instructions, inner=context_fn):
            inner(instructions)
            branches.set_context(instructions)
    return context_fn


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


class ConsolidateGate(Method):
    @staticmethod
    def add_args(group):
        group.add_argument("--core-from", default=None,
                           help="consolidate: stage directory with ablation_scores.json (training/compute_importance.py)")
        group.add_argument("--core-keep", type=float, default=0.25, help="fraction of unit groups in the core")
        group.add_argument("--core-selection", default="ablation", choices=["ablation", "random", "bundle"],
                           help="ablation: the most important groups; random: the same numbers of heads and MLP blocks "
                                "at random; bundle: in every layer --core-attn-keep of the key/value bundles (a key/value "
                                "head with the query heads that share it) and --core-keep of the MLP blocks")
        group.add_argument("--core-attn-keep", type=float, default=None, help="bundle selection: key/value bundles per layer")
        group.add_argument("--freeze-core", action="store_true",
                           help="keep the earlier tasks' cores and everything else on their gated computation fixed, "
                                "so only their silenced groups learn")
        group.add_argument("--release", type=float, default=0.0,
                           help="with --freeze-core, learning a new task: unfreeze this fraction of the groups each "
                                "earlier task added to the cores, its least important ones by "
                                "ablation_scores_<task>_gated.json in --init (compute_importance.py --task --gated)")
        group.add_argument("--release-mode", choices=["train", "regate"], default="train",
                           help="train: the earlier tasks keep using the released groups, which then change; regate: "
                                "the released groups are silenced for the earlier tasks too (written to core.json), so "
                                "they lose that capacity but are not changed by the new task")
        group.add_argument("--distill", type=float, default=0.0,
                           help="with --release, protect the released groups without old data: every update also runs "
                                "the current task's images with an earlier task's instruction (cycling, --distill-batch "
                                "examples) and adds this weight x the squared difference between the action velocities "
                                "the policy and the --init checkpoint predict for the same noise and flow time")
        group.add_argument("--distill-batch", type=int, default=8)
        group.add_argument("--adapter", action="store_true",
                           help="with --freeze-core, learning a new task: give its instruction its own additive copies "
                                "of the frozen state/action/time projections (stored as adapter.safetensors)")
        group.add_argument("--branch-rank", type=int, default=0,
                           help="give the current task's instruction its own rank-R attention branch, trained in this "
                                "and later stages of the same task and frozen afterwards; branches in --init are always "
                                "kept (stored as branches.safetensors)")

    def setup(self):
        args, policy, params = self.stage.args, self.stage.policy, self.stage.params
        core = earlier = None
        if args.init and os.path.exists(os.path.join(args.init, "core.json")):
            with open(os.path.join(args.init, "core.json")) as f:
                core = json.load(f)
        assert core is not None or args.core_from, "consolidate_gate needs --core-from or an --init with core.json"
        mlp_block = (core or {}).get("group_layout", {}).get("mlp_block", 64)  # group ids are relative to this layout
        if args.core_from:
            with open(os.path.join(args.core_from, "ablation_scores.json")) as f:
                saved = json.load(f)
            groups = expert_groups(policy, mlp_block)
            assert [list(g) for g in groups] == [list(g) for g in saved["groups"]], "group layout changed"
            earlier, core = core, None
            taken = earlier["core"] if earlier else []
            if args.core_selection == "bundle":
                wrapper = policy.model.vlm_with_expert
                members = select_bundles(groups, args.core_keep, args.core_attn_keep,
                                         wrapper.num_attention_heads // wrapper.num_key_value_heads, args.seed, exclude=taken)
            else:
                members = select_core(groups, saved["scores"], args.core_keep, args.core_selection, args.seed,
                                      exclude=taken)
            allowed = sorted(set(taken) | set(members))
            core = {"task": args.task, "instruction": TASKS[args.task].instruction, "core": allowed,
                    "silenced": sorted(set(range(len(groups))) - set(allowed)),
                    "selection": args.core_selection, "keep": args.core_keep, "attn_keep": args.core_attn_keep,
                    "seed": args.seed, "group_layout": {"mlp_block": mlp_block, "groups": len(groups)}}
            if earlier:
                core.update(new=members, earlier=earlier.get("earlier", []) +
                            [{k: v for k, v in earlier.items() if k != "earlier"}])
        gate = gate_from_core(policy, core)
        groups = gate.groups
        for entry in core.get("earlier", []) + [core]:
            print(f"gating: {len(entry['silenced'])} of {len(groups)} unit groups silenced for "
                  f"'{entry['instruction']}' ({entry['selection']} core)", flush=True)
        with open(os.path.join(args.out, "core.json"), "w") as f:
            json.dump(core, f)
        self.gated = True
        if args.freeze_core:
            # Frozen entries get zero gradient; with zero weight decay AdamW then never moves them,
            # so no copy of their values is needed (checked against the init checkpoint at the end).
            assert args.weight_decay == 0, "--freeze-core requires --weight-decay 0"
            assert not args.core_from or earlier, "--freeze-core with --core-from needs a core in --init"
            protected = earlier["core"] if args.core_from else core["core"]
            released = []
            if args.release:
                assert not args.core_from, "--release applies when a new task is learned"
                taken = set()
                for entry in core.get("earlier", []) + [core]:  # the groups each task added, least important first
                    own = [g for g in entry.get("new", entry["core"]) if g not in taken]
                    taken |= set(entry["core"])
                    with open(os.path.join(args.init, f"ablation_scores_{entry['task']}_gated.json")) as f:
                        scores = json.load(f)["scores"]
                    released += sorted(own, key=lambda g: scores[g])[:int(round(args.release * len(own)))]
                protected = sorted(set(protected) - set(released))
                if args.release_mode == "regate":  # the earlier tasks stop using the released groups (less capacity, no interference)
                    for entry in core.get("earlier", []) + [core]:
                        entry["silenced"] = sorted(set(entry["silenced"]) | set(released))
                        entry["core"] = sorted(set(entry["core"]) - set(released))
                        gate.silence(entry["instruction"], entry["silenced"])
                    with open(os.path.join(args.out, "core.json"), "w") as f:
                        json.dump(core, f)
            self.frozen = {n: ~m for n, m in core_trainable_masks(policy, groups, protected).items()}
            counts = {"frozen": sum(int(m.sum()) for m in self.frozen.values()),
                      "plastic": sum(int((~m).sum()) for m in self.frozen.values()),
                      "core_groups": len(protected), "groups": len(groups), "released": sorted(released),
                      "by_module": family_counts(params, self.frozen)}
            with open(os.path.join(args.out, "freeze.json"), "w") as f:
                json.dump(counts, f)
            print(f"freeze-core: {counts['frozen']} frozen and {counts['plastic']} plastic parameters", flush=True)
        context_fn = gate.set_context
        self.adapter = self.branches = None
        if args.adapter:
            assert args.freeze_core and not args.core_from, "--adapter is for learning a new task"
            adapter = self.adapter = InterfaceAdapter(policy, TASKS[args.task].instruction)
            self.extra_params.update({f"adapter.{n}": p for n, p in adapter.params.items()})
            print(f"adapter: {adapter.count()} task-specific projection parameters for '{adapter.instruction}'", flush=True)

            def context_fn(instructions, gated=gate.set_context):
                gated(instructions)
                adapter.set_context(instructions)
        if args.branch_rank or (args.init and os.path.exists(os.path.join(args.init, "branches.json"))):
            branches = self.branches = (AttentionBranches.load(args.init, policy) if args.init and
                                        os.path.exists(os.path.join(args.init, "branches.json")) else AttentionBranches(policy))
            instruction = TASKS[args.task].instruction
            if args.branch_rank and instruction not in branches.instructions:
                branches.add(instruction, args.branch_rank)
            own = branches.owned(instruction)
            for n, p in branches.params.items():
                p.requires_grad_(n in own)
            self.extra_params.update({f"branches.{n}": p for n, p in branches.params.items() if p.requires_grad})
            print(f"branches: {len(branches.instructions)} ({branches.count(instruction)} trainable parameters for "
                  f"'{instruction}')", flush=True)

            def context_fn(instructions, inner=context_fn):
                inner(instructions)
                branches.set_context(instructions)
        self.context_fn = context_fn
        if args.distill:
            assert args.release, "--distill protects released groups"
            self.old_instructions = [entry["instruction"] for entry in core.get("earlier", []) + [core]]
            self.teacher = {n: t.to(params[n].device) for n, t in load_file(os.path.join(args.init, "trainable.safetensors")).items()}
            self.velocity = {}
            policy.model.action_out_proj.register_forward_hook(lambda module, inputs, out: self.velocity.update(v=out))

    def distill_loss(self, step):
        """Squared difference of the predicted action velocities from the --init checkpoint's, for current-task
        examples with an earlier task's instruction (gated as that task)."""
        stage, policy = self.stage, self.stage.policy
        cfg, n = policy.config, stage.args.distill_batch
        batch = stage.make_batch(stage.data, size=n, instruction=self.old_instructions[step % len(self.old_instructions)])
        noise = policy.model.sample_noise((n, stage.chunk, cfg.max_action_dim), "cuda")
        flow_time = policy.model.sample_time(n, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.func.functional_call(policy, self.teacher, (batch,), {"noise": noise, "time": flow_time})
        target = self.velocity["v"].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy.forward(batch, noise=noise, time=flow_time)
        dims = cfg.action_feature.shape[0]
        sq = (self.velocity["v"].float() - target)[:, :, :dims] ** 2
        valid = (~batch["action_is_pad"]).unsqueeze(-1).to(sq.dtype)
        return (sq * valid).sum() / (valid.sum() * dims).clamp_min(1)

    def loss(self, step):
        if not self.stage.args.distill:
            return super().loss(step)
        distilled = self.distill_loss(step)  # before the main batch sets the new task's gate
        loss, _, logged = super().loss(step)
        return loss, loss + self.stage.args.distill * distilled, {**logged, "distill": distilled}

    def finish(self):
        if self.adapter is not None:
            self.adapter.save(self.stage.args.out)
        if self.branches is not None:
            self.branches.save(self.stage.args.out)


ADAPTER_PROJECTIONS = ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out")


class InterfaceAdapter:
    """Instruction-specific interface adapter.

    The core gate freezes the action expert's global state, action and flow-time projections after the first task,
    since every task's computation uses them. The adapter gives one instruction its own additive copies of those
    weights (zero-initialised deltas, so it starts from the frozen projections), active only when the policy acts on
    that instruction; every other instruction's computation is unchanged. Its parameters are task-specific storage
    and are counted separately (adapter.json)."""

    def __init__(self, policy, instruction, state=None):
        self.instruction = instruction
        self.active = False
        self.params = {}
        self.handles = []
        for name in ADAPTER_PROJECTIONS:
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
            json.dump({"instruction": self.instruction, "modules": list(ADAPTER_PROJECTIONS), "parameters": self.count(),
                       "bytes_fp32": 4 * self.count()}, f)

    @classmethod
    def load(cls, stage_dir, policy):
        with open(os.path.join(stage_dir, "adapter.json")) as f:
            meta = json.load(f)
        return cls(policy, meta["instruction"], load_file(os.path.join(stage_dir, "adapter.safetensors")))


BRANCH_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


class AttentionBranches:
    """Instruction-specific low-rank attention branches.

    A task can own a branch: low-rank additive deltas (B A, with B zero-initialised, so the branch starts as the
    unchanged network) on the query, key, value and output projections of every action-expert attention layer,
    active only when the policy acts on that task's instruction. The branch is trained while its task is learned and
    consolidated and frozen afterwards, so each task gets attention capacity of its own instead of competing for the
    few key/value heads that earlier cores hold. Its parameters are task-specific storage, counted in branches.json."""

    def __init__(self, policy):
        self.layers = policy.model.vlm_with_expert.lm_expert.layers
        self.instructions = []  # owner of each branch, in the order the tasks were learned
        self.rank = {}
        self.params = {}  # "<branch>.<layer>.<projection>.<a|b>" -> parameter
        self.context = []
        self.handles = [getattr(layer.self_attn, proj).register_forward_hook(self._hook(i, proj))
                        for i, layer in enumerate(self.layers) for proj in BRANCH_PROJECTIONS]

    def add(self, instruction, rank, state=None):
        k = len(self.instructions)
        self.instructions.append(instruction)
        self.rank[instruction] = rank
        for i, layer in enumerate(self.layers):
            for proj in BRANCH_PROJECTIONS:
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
            json.dump({"instructions": self.instructions, "rank": self.rank, "modules": list(BRANCH_PROJECTIONS),
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


# A task's gated sub-network as a narrower dense action expert. Under the context gate a task computes with whole unit
# groups only: attention heads (with the key/value heads they share) and 64-unit MLP blocks of the action expert.
# Removing the silenced rows and columns of those projections gives a narrower expert that computes the same function
# with fewer operations, so a task's inference cost follows its allocation. Per-weight masks (PackNet) leave every
# matrix at full size. Inference only (cached prefix, then denoising).

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
