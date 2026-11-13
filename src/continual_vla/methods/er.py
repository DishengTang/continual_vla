"""Experience replay (Rolnick et al., 2019): a fixed fraction of every batch comes from bounded replay buffers of the
earlier tasks.

Every stage, whatever its method, stores --memory-size random examples of its task (memory.safetensors: image, joint
positions, action chunk and padding mask); only those are kept and loaded, not the task's other demonstrations.
training/make_replay_buffer.py writes larger buffers of a finished stage. Several buffers (--memory-name memory
/path/to/A/memory, for a third task) share the replayed part of each batch evenly; each replayed example carries its
own task's instruction.
"""

import os

import numpy as np
import torch

from ..data import ReplayBuffer
from .base import Method


class ExperienceReplay(Method):
    @staticmethod
    def add_args(group):
        group.add_argument("--replay-frac", type=float, default=0.25, help="fraction of every batch that is replayed")
        group.add_argument("--replay-sampling", choices=("equal", "union"), default="equal",
                           help="equal: every buffer gets an equal share of each batch's replayed examples; union: they "
                                "are drawn uniformly from all stored examples together (Liu et al.)")
        group.add_argument("--memory-name", nargs="+", default=["memory"],
                           help="replay buffer(s) to use: <name>.safetensors in --init, or an absolute path prefix "
                                "(see make_replay_buffer.py); several buffers share the replayed part of each batch")

    @staticmethod
    def save_for_later_stages(stage):
        args = stage.args
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(len(stage.data), size=min(args.memory_size, len(stage.data)), replace=False))
        buffer = getattr(stage, "replay_buffer", ReplayBuffer)  # a benchmark's own buffer class (LIBERO)
        buffer.save(os.path.join(args.out, "memory"), stage.data, torch.from_numpy(chosen))

    def setup(self):
        args = self.stage.args
        buffer = getattr(self.stage, "replay_buffer", ReplayBuffer)
        memories = []
        for name in args.memory_name:
            memories.append(buffer(os.path.join(args.init, name)))
            print(f"replay buffer: {len(memories[-1])} examples of {memories[-1].task_id} "
                  f"({memories[-1].nbytes / 2**20:.1f} MiB)", flush=True)
        replayed = round(args.batch * args.replay_frac)
        self.memories, self.remainder = memories, replayed % len(memories)
        self.extra = [(m, replayed // len(memories)) for m in memories]

    def extra_examples(self):
        """Each buffer gets an equal share of the replayed examples; when they do not divide evenly (many earlier
        tasks, small batches), the remaining examples come from buffers drawn at random for every batch. With
        --replay-sampling union, every replayed example comes from a buffer drawn in proportion to its size."""
        if getattr(self.stage.args, "replay_sampling", "equal") == "union":
            sizes = torch.tensor([len(m) for m in self.memories], dtype=torch.float)
            replayed = sum(n for _, n in self.extra) + self.remainder
            counts = torch.bincount(torch.multinomial(sizes, replayed, replacement=True), minlength=len(sizes))
            return [(m, int(n)) for m, n in zip(self.memories, counts) if n]
        if not self.remainder:
            return self.extra
        extra = list(self.extra)
        for i in torch.randperm(len(self.memories))[:self.remainder].tolist():
            extra[i] = (extra[i][0], extra[i][1] + 1)
        return [(m, n) for m, n in extra if n]
