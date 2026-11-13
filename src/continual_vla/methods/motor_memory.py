"""Contextual motor memory: every learned skill is stored in a core of the action expert's units, sized by what the
skill needs, reached only through the skill's instruction and frozen while later skills are learned. It starts from
consolidate-and-gate (consolidate_gate.py; context-dependent gating with synaptic stabilisation, Masse, Grant & Freedman,
PNAS 2018), which gives every skill the same fixed share of whole attention heads and MLP blocks, and changes two things:

  allocation by need   a skill's core starts as a small seed and doubles until the core alone reaches the loss of the
                       full policy on the skill's own demonstrations (same examples, noise and flow time), up to
                       --core-cap of the free units
  finer attention      pi0's action expert has 8 query heads of 256 dimensions that share one key/value head of 256, so
                       whole heads cannot be divided among skills. With --attention-units qkv (the default) an attention
                       unit is 8 of the 256 dimensions, 4 and their partners under the rotary position encoding (which
                       mixes dimension i with i + 128), in the key and value rows and in all 8 query heads' query rows
                       and output-projection columns. With --attention-units value it is 8 of the 256 value dimensions
                       and the matching output-projection columns of all 8 query heads; the query and key projections
                       are shared like the norms (trained with the first task, frozen afterwards), so every instruction
                       sees the same attention scores

A task is learned in two phases. First, --steps updates with every free unit active and the earlier cores frozen. Then the
core phase, up to --memory-steps updates: the task's instruction silences every unit outside the earlier cores and the
task's core, and only the core trains, so the skill moves into the core. Every --core-check-every updates the core's loss
is compared with the full policy's; the phase ends once their ratio is at most --core-tolerance, and the core doubles
every --core-patience updates until then. With --core-share the core has a fixed share of the units from the start of
the phase and does not grow. After a task, its core is frozen and the free units continue from their values at the end
of its first phase.

The core's units are taken in every layer, of each kind, from the free ones (--unit-select): at random (the default), by
magnitude (the squared weights of the unit, PackNet's criterion for whole units) or by importance (the first-order
Taylor estimate of the loss change when the unit is removed, Molchanov et al., CVPR 2019: the squared gradient of the
loss with respect to a gate on the unit, averaged over --core-loss-batches fixed batches of the task, for the full
policy at the end of the first phase). For comparison, the loss of the seed core under every rule is logged as well.

Earlier tasks use earlier cores read-only: task k's instruction activates the cores of tasks 1 to k, all frozen by the
time a later task trains, and silences the rest, so its computation cannot change. The state, action and time
projections and the norms train with the first task and are frozen afterwards. The units are in the action expert only
(--train expert). The instruction is a permitted policy input, so the gate uses no privileged information.

Stored for later tasks: memory.json (each task's instruction and core, about 216 bytes per task as a bit mask). Written
per task: core_log.jsonl (the core's size and its loss ratio at every check) and freeze.json.
"""

import json
import math
import os

import numpy as np
import torch

from .base import Method, clear_moments, family_counts

ATTENTION_PAIRS = 4  # pairs of key/value dimensions per qkv attention unit (i and its rotary partner i + head_dim / 2)
VALUE_UNIT = 8  # value dimensions per value attention unit (the same number of dimensions)
MLP_UNIT = 64  # MLP units per block, as in consolidate_gate.py
EXPERT = "model.paligemma_with_expert.gemma_expert.model.layers"
LOSS_SEEDS = 10 ** 6  # the fixed examples, noise and flow times of the loss comparison
IMPORTANCE_SEEDS = 2 * 10 ** 6  # those of the importance estimate (other examples than the loss comparison's)
RULES = ("random", "magnitude", "importance")


class UnitGate:
    """Silences units of the action expert for the current instruction: a silenced attention unit has its dimensions
    zeroed before the output projection in every head and, for qkv units, its query dimensions zeroed in every head, so
    that with their rotary partners they do not contribute to the attention scores; a silenced MLP block has its inputs
    to the down projection zeroed. Nothing else depends on the weights of a silenced unit."""

    def __init__(self, policy, attention_units="qkv"):
        layers = policy.model.paligemma_with_expert.gemma_expert.model.layers
        self.attention_units = attention_units
        self.units = []  # [(layer, kind, start, stop)] in the key/value head's dimensions (qkv: in its first half)
        self.handles = []
        self.width = {}
        self.silenced = {}  # instruction -> {(layer, kind): bool mask of its dimensions}
        self.current = None
        self.scale = None  # while estimating importance: {(layer, kind): gate of each dimension's unit}
        for i, layer in enumerate(layers):
            attention, mlp = layer.self_attn, layer.mlp
            self.head_dim = attention.k_proj.out_features
            self.heads = attention.q_proj.out_features // self.head_dim
            assert self.heads * self.head_dim == attention.q_proj.out_features == attention.o_proj.in_features
            assert attention.v_proj.out_features == self.head_dim and attention.head_dim == self.head_dim
            self.width[i, "attention"], self.width[i, "mlp"] = attention.q_proj.out_features, mlp.down_proj.in_features
            if attention_units == "qkv":
                self.units += [(i, "attention", s, s + ATTENTION_PAIRS) for s in range(0, self.head_dim // 2, ATTENTION_PAIRS)]
                self.handles.append(attention.q_proj.register_forward_hook(self._after(i)))
            else:
                assert attention_units == "value", attention_units
                self.units += [(i, "attention", s, s + VALUE_UNIT) for s in range(0, self.head_dim, VALUE_UNIT)]
            self.units += [(i, "mlp", s, s + MLP_UNIT) for s in range(0, self.width[i, "mlp"], MLP_UNIT)]
            self.handles += [attention.o_proj.register_forward_pre_hook(self._before(i, "attention")),
                             mlp.down_proj.register_forward_pre_hook(self._before(i, "mlp"))]
        self.dim_unit = {key: torch.full((n,), -1, dtype=torch.long) for key, n in self.width.items()}
        for u, (layer, kind, start, stop) in enumerate(self.units):
            self.dim_unit[layer, kind][self._gated_dims(kind, start, stop)] = u
        assert all((m >= 0).all() for m in self.dim_unit.values()), "every gated dimension belongs to a unit"

    def _gate(self, x, key):
        if self.current is not None:
            x = x.masked_fill(self.current[key], 0)
        if self.scale is not None:
            x = x * self.scale[key].to(x.dtype)
        return x

    def _after(self, layer):
        def hook(module, inputs, output):
            return self._gate(output, (layer, "attention"))
        return hook

    def _before(self, layer, kind):
        def hook(module, inputs):
            return (self._gate(inputs[0], (layer, kind)),) + tuple(inputs[1:])
        return hook

    def _kv_dims(self, start, stop):
        """A unit's dimensions in the key/value head."""
        if self.attention_units == "value":
            return list(range(start, stop))
        half = self.head_dim // 2
        return list(range(start, stop)) + list(range(start + half, stop + half))

    def _head_dims(self, start, stop):
        """The same dimensions in every query head (the query rows and the output projection's columns)."""
        return [h * self.head_dim + d for h in range(self.heads) for d in self._kv_dims(start, stop)]

    def _gated_dims(self, kind, start, stop):
        return self._head_dims(start, stop) if kind == "attention" else list(range(start, stop))

    def silence(self, instruction, active, device):
        """Silence every unit outside `active` (unit indices) for the instruction."""
        masks = {key: torch.ones(n, dtype=torch.bool, device=device) for key, n in self.width.items()}
        for u in active:
            layer, kind, start, stop = self.units[u]
            masks[layer, kind][self._gated_dims(kind, start, stop)] = False
        self.silenced[instruction] = masks

    def set_context(self, instructions):
        assert all(i == instructions[0] for i in instructions), "the gate is set per batch of one task"
        self.current = self.silenced.get(instructions[0])

    def unit_parts(self, u):
        """A unit's weights: [(parameter name, 0 for rows or 1 for columns, indices)]."""
        layer, kind, start, stop = self.units[u]
        base = f"{EXPERT}.{layer}"
        if kind == "mlp":
            rows = list(range(start, stop))
            return [(f"{base}.mlp.gate_proj.weight", 0, rows), (f"{base}.mlp.up_proj.weight", 0, rows),
                    (f"{base}.mlp.down_proj.weight", 1, rows)]
        kv, heads = self._kv_dims(start, stop), self._head_dims(start, stop)
        parts = [(f"{base}.self_attn.v_proj.weight", 0, kv), (f"{base}.self_attn.o_proj.weight", 1, heads)]
        if self.attention_units == "qkv":
            parts += [(f"{base}.self_attn.q_proj.weight", 0, heads), (f"{base}.self_attn.k_proj.weight", 0, kv)]
        return parts

    def unit_entries(self, frozen, u, value):
        """Set a unit's weights in the {parameter name: bool mask} `frozen`."""
        for name, axis, index in self.unit_parts(u):
            if axis == 0:
                frozen[name][index] = value
            else:
                frozen[name][:, index] = value

    @torch.no_grad()
    def magnitude(self, params):
        """Every unit's sum of squared weights."""
        return [sum(float((params[n][i] if axis == 0 else params[n][:, i]).float().pow(2).sum())
                    for n, axis, i in self.unit_parts(u)) for u in range(len(self.units))]

    def taylor(self, loss_fn, n_batches, device):
        """Every unit's mean over batches of (dL/dg)^2, with g a gate (= 1) on all of the unit's gated dimensions:
        loss_fn(b) is the loss on batch b."""
        gates = torch.ones(len(self.units), device=device, requires_grad=True)
        index = {key: m.to(device) for key, m in self.dim_unit.items()}
        score = torch.zeros(len(self.units), device=device)
        try:
            for b in range(n_batches):
                self.scale = {key: gates[i] for key, i in index.items()}
                (grad,) = torch.autograd.grad(loss_fn(b), gates)
                score += grad.float() ** 2
        finally:
            self.scale = None
        return (score / n_batches).tolist()


class MotorMemory(Method):
    from_first_task = True

    @staticmethod
    def add_args(group):
        group.add_argument("--memory-steps", type=int, default=10000, help="most updates of a task's core phase")
        group.add_argument("--core-seed", type=float, default=0.02,
                           help="the seed core: this share of each layer's attention units and MLP blocks (at least one each)")
        group.add_argument("--core-cap", type=float, default=0.25, help="largest core, as a share of the free units")
        group.add_argument("--core-share", type=float, default=0.0,
                           help="a core of fixed size instead: this share of each layer's attention units and MLP blocks "
                                "(rounded, at least one each, at most the free ones) from the start of the core phase, "
                                "without seed or growth; 0: sized by need (--core-seed, --core-cap)")
        group.add_argument("--core-tolerance", type=float, default=1.1,
                           help="the core phase ends when core loss / full policy loss is at most this")
        group.add_argument("--core-check-every", type=int, default=500, help="updates between loss comparisons")
        group.add_argument("--core-patience", type=int, default=2000, help="updates between doublings of the core")
        group.add_argument("--core-loss-batches", type=int, default=32, help="batches of the loss comparison")
        group.add_argument("--unit-select", default="random", choices=RULES,
                           help="how the core's units are taken from the free ones in every layer and kind")
        group.add_argument("--attention-units", default="qkv", choices=("qkv", "value"),
                           help="qkv: query, key, value and output dimensions; value: value and output dimensions, with "
                                "the query and key projections shared (trained with the first task, frozen afterwards)")

    def setup(self):
        stage = self.stage
        args, policy, params = stage.args, stage.policy, stage.params
        assert args.train == "expert", "motor_memory: the units are in the action expert (--train expert)"
        assert args.core_patience % args.core_check_every == 0, "--core-patience: a multiple of --core-check-every"
        self.device = next(iter(params.values())).device
        self.gate = UnitGate(policy, args.attention_units)
        self.instruction = stage.data.instruction
        self.tasks = []
        if args.init:
            with open(os.path.join(args.init, "memory.json")) as f:
                memory = json.load(f)
            self.tasks = memory["tasks"]
            assert memory.get("attention_units", "qkv") == args.attention_units, "--attention-units as for earlier tasks"
        assert self.instruction not in [t["instruction"] for t in self.tasks], "a task is learned once"
        taken = []
        for t in self.tasks:
            taken += t["core"]
            self.gate.silence(t["instruction"], taken, self.device)
        self.earlier = taken
        self.core, self.growth, self.full_loss, self.scores = [], 0, None, {}
        self.free = [u for u in range(len(self.gate.units)) if u not in set(taken)]
        self.known = {t["instruction"] for t in self.tasks} | {self.instruction}
        self.frozen = self._frozen(self.free)
        assert args.init or not any(m.any() for m in self.frozen.values()), "the units do not cover the weight matrices"
        self.frozen_from_start = {n: m.clone() for n, m in self.frozen.items()}
        self.start = {n: p.detach().cpu().clone() for n, p in params.items()}
        self.context_fn = self._context
        self.log_path = os.path.join(args.out, "core_log.jsonl")
        print(f"motor memory: {len(self.tasks)} tasks stored, {len(self.free)} of {len(self.gate.units)} units free "
              f"({args.attention_units} attention units, {args.unit_select} selection)", flush=True)

    def _context(self, instructions):
        assert instructions[0] in self.known, f"instruction not learned: {instructions[0]!r}"
        self.gate.set_context(instructions)

    def _unit_weights(self):
        """Names of the weight matrices that the units divide."""
        modules = ("self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
        if self.gate.attention_units == "qkv":
            modules += ("self_attn.q_proj", "self_attn.k_proj")
        return {f"{EXPERT}.{i}.{m}.weight" for i in {u[0] for u in self.gate.units} for m in modules}

    def _frozen(self, trainable_units):
        """Everything is frozen except `trainable_units`; the first task also trains what is outside the units."""
        params = self.stage.params
        frozen = {n: torch.full_like(p, bool(self.stage.args.init), dtype=torch.bool) for n, p in params.items()}
        for n in self._unit_weights():
            frozen[n][...] = True
        for u in trainable_units:
            self.gate.unit_entries(frozen, u, False)
        return frozen

    def extra_updates(self):
        return self.stage.args.memory_steps

    def prepare_optimizer(self, optimizer):
        clear_moments(optimizer, self.stage.params, self.frozen)

    def _fixed_loss(self, seed):
        """The policy's loss on the fixed examples, noise and flow time of `seed`, without touching the training streams."""
        stage = self.stage
        augment, stage.augment = stage.augment, False
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                return stage.loss_on(stage.make_batch(stage.data, size=8))
        finally:
            stage.augment = augment

    @torch.no_grad()
    def _loss(self):
        """The policy's mean loss on the fixed batches of the loss comparison."""
        n = self.stage.args.core_loss_batches
        return sum(self._fixed_loss(LOSS_SEEDS + b).item() for b in range(n)) / n

    def _score(self):
        """The magnitude and importance of every unit, for the full policy (the task's instruction not yet silenced)."""
        assert self.instruction not in self.gate.silenced
        self.scores = {"magnitude": self.gate.magnitude(self.stage.params),
                       "importance": self.gate.taylor(lambda b: self._fixed_loss(IMPORTANCE_SEEDS + b),
                                                      self.stage.args.core_loss_batches, self.device)}

    def _target(self, layer, kind, growth):
        """The core's units of a kind in a layer at a growth step."""
        total = sum(1 for u in self.gate.units if u[0] == layer and u[1] == kind)
        free = sum(1 for u in self.free if self.gate.units[u][0] == layer and self.gate.units[u][1] == kind)
        if self.stage.args.core_share:
            return min(free, max(1, round(self.stage.args.core_share * total)))
        cap = max(1, int(self.stage.args.core_cap * free))
        return min(cap, max(1, math.ceil(self.stage.args.core_seed * 2 ** growth * total)))

    def _choose(self, growth, rule, core=()):
        """The core at a growth step from `core`, stratified over layers and kinds: random free units (the same choice
        after a restart) or those with the largest score under the rule."""
        rng = np.random.default_rng([self.stage.args.seed, self.stage.k, growth])
        added = []
        current = set(core)
        for layer in sorted({u[0] for u in self.gate.units}):
            for kind in ("attention", "mlp"):
                candidates = [u for u in self.free if self.gate.units[u][:2] == (layer, kind)]
                have = [u for u in candidates if u in current]
                rest = [u for u in candidates if u not in current]
                want = min(self._target(layer, kind, growth) - len(have), len(rest))
                if want > 0:
                    if rule == "random":
                        added += [int(u) for u in rng.choice(rest, size=want, replace=False)]
                    else:
                        added += sorted(rest, key=lambda u: -self.scores[rule][u])[:want]
        return sorted(current | set(added))

    def _grow(self, growth):
        """The core at a growth step under --unit-select. Returns whether it changed."""
        core = self._choose(growth, self.stage.args.unit_select, self.core)
        changed = len(core) > len(self.core)
        self.core, self.growth = core, growth
        return changed

    def _apply_core(self):
        self.gate.silence(self.instruction, sorted(set(self.earlier) | set(self.core)), self.device)
        self.frozen = self._frozen(self.core)

    def _log(self, **record):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        shown = {k: v for k, v in record.items() if k != "scores"}
        print(f"core: {json.dumps(shown)}", flush=True)

    def after_learning(self, optimizer):
        args = self.stage.args
        self.stage.run_eval(args.steps, [self.stage.task])  # the full policy; the matrix uses the last evaluation
        self.full_loss = self._loss()
        self._score()
        rule_loss = {}
        for rule in RULES:  # the seed core under every rule, on the same weights
            self.core = self._choose(0, rule)
            self._apply_core()
            rule_loss[rule] = self._loss()
        self.core = []
        self._grow(0)
        self._apply_core()
        clear_moments(optimizer, self.stage.params, self.frozen)
        open(self.log_path, "w").close()
        self._log(step=args.steps, event="seed", units=len(self.core), full_loss=self.full_loss, core=self.core,
                  core_loss=rule_loss[args.unit_select], rule_core_loss=rule_loss,
                  scores=self.scores.get(args.unit_select))

    def after_update(self, step, optimizer):
        args = self.stage.args
        n = step - args.steps
        if n <= 0 or n % args.core_check_every:
            return False
        loss = self._loss()
        ratio = loss / self.full_loss
        self._log(step=step, event="check", units=len(self.core), core_loss=loss, ratio=ratio)
        if ratio <= args.core_tolerance:
            return True
        if n % args.core_patience == 0 and n < args.memory_steps and self._grow(self.growth + 1):
            self._apply_core()
            self._log(step=step, event="grow", units=len(self.core), core=self.core)
        return False

    def on_resume(self, step, optimizer):
        args = self.stage.args
        if step <= args.steps:
            return
        kept = []
        with open(self.log_path) as f:
            for line in f:
                record = json.loads(line)
                if record["step"] > step:
                    continue
                kept.append(line)
                if record["event"] == "seed":
                    self.full_loss, self.core = record["full_loss"], record["core"]
                    if record.get("scores") is not None:
                        self.scores = {args.unit_select: record["scores"]}
                elif record["event"] == "grow":
                    self.core, self.growth = record["core"], self.growth + 1
        with open(self.log_path, "w") as f:
            f.writelines(kept)
        self._apply_core()

    @torch.no_grad()
    def finish(self):
        args, params = self.stage.args, self.stage.params
        changed = sum(int((params[n].cpu()[m.cpu()] != self.start[n][m.cpu()]).sum())
                      for n, m in self.frozen_from_start.items())
        assert changed == 0, f"{changed} frozen weights changed"
        entry = {"task": self.stage.task, "instruction": self.instruction, "core": self.core}
        expert_weights = self._unit_weights()
        after = {n: torch.full_like(p, n not in expert_weights, dtype=torch.bool) for n, p in params.items()}
        for u in sorted(set(self.earlier) | set(self.core)):
            self.gate.unit_entries(after, u, True)
        counts = {"frozen_after_task": sum(int(m.sum()) for m in after.values()),
                  "parameters": sum(p.numel() for p in params.values()),
                  "core_units": len(self.core), "units": len(self.gate.units), "earlier_units": len(self.earlier),
                  "frozen_weights_changed": changed, "by_module": family_counts(params, after)}
        with open(os.path.join(args.out, "freeze.json"), "w") as f:
            json.dump(counts, f)
        with open(os.path.join(args.out, "memory.json"), "w") as f:
            json.dump({"tasks": self.tasks + [entry], "attention_units": args.attention_units,
                       "attention_pairs": ATTENTION_PAIRS, "value_unit": VALUE_UNIT, "mlp_unit": MLP_UNIT}, f)
        print(f"motor memory: core of {len(self.core)} units; {counts['frozen_after_task']} of {counts['parameters']} "
              f"parameters frozen after this task", flush=True)
        del self.start
