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
        group.add_argument("--memory-name", nargs="+", default=["memory"],
                           help="replay buffer(s) to use: <name>.safetensors in --init, or an absolute path prefix "
                                "(see make_replay_buffer.py); several buffers share the replayed part of each batch")

    @staticmethod
    def save_for_later_stages(stage):
        args = stage.args
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(len(stage.data), size=min(args.memory_size, len(stage.data)), replace=False))
        ReplayBuffer.save(os.path.join(args.out, "memory"), stage.data, torch.from_numpy(chosen))

    def setup(self):
        args = self.stage.args
        memories = []
        for name in args.memory_name:
            memories.append(ReplayBuffer(os.path.join(args.init, name)))
            print(f"replay buffer: {len(memories[-1])} examples of {memories[-1].task_id} "
                  f"({memories[-1].nbytes / 2**20:.1f} MiB)", flush=True)
        replayed = round(args.batch * args.replay_frac)
        self.extra = [(m, replayed // len(memories) + (i < replayed % len(memories))) for i, m in enumerate(memories)]

    def extra_examples(self):
        return self.extra
