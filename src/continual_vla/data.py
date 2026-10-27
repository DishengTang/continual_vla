"""Demonstration episodes as (image, state, action chunk) training examples, frames memory-mapped from a cache."""

import hashlib
import json
import os

import h5py
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .observations import EPISODE_FIELDS
from .tasks import CONTROL_MODE, TASKS

# numpy requests transparent huge pages for large arrays; on a host with fragmented memory every page fault while
# the frames (tens of GB) are read then stalls in kernel compaction, and loading takes an hour instead of minutes.
np._core.multiarray._set_madvise_hugepage(False)


def replay_prefix(task_id):
    return os.path.join(os.environ["CVLA_DATA"], "replay", task_id,
                        f"trajectory.rgb.{CONTROL_MODE}.physx_cpu")


def split_episodes(task_id, n_train, n_val, seed=0):
    """Split successful replayed episodes into disjoint training and validation episodes."""
    with open(replay_prefix(task_id) + ".json") as f:
        episodes = json.load(f)["episodes"]
    ids = np.array([e["episode_id"] for e in episodes if e["success"]])
    np.random.default_rng(seed).shuffle(ids)
    assert len(ids) >= n_train + n_val, f"{task_id}: only {len(ids)} episodes"
    return ids[:n_train].tolist(), ids[n_train:n_train + n_val].tolist()


def load_splits(stage_dir):
    """Training/validation episodes of every task learned up to a stage. Splits are persisted so that
    later stages validate an earlier task on exactly the episodes held out when it was learned."""
    path = os.path.join(stage_dir, "splits.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Stage written before splits were persisted: rebuild them from that stage's own settings.
    with open(os.path.join(stage_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(stage_dir, "stage.json")) as f:
        tasks = json.load(f)["tasks"]
    return {t: dict(zip(("train", "val"), split_episodes(t, cfg["n_train"], cfg["n_val"]))) for t in tasks}


class DemoSet:
    """All timesteps of the given episodes. Example i is the observation at step i and the
    chunk of the next `chunk_size` actions, padded (and masked) past the end of its episode."""

    def __init__(self, task_id, episode_ids, chunk_size):
        self.task_id = task_id
        self.instruction = TASKS[task_id].instruction
        self.episode_ids = list(episode_ids)
        self.chunk_size = chunk_size
        # The frames (tens of GB for large datasets) are written once to a file shared by every run that uses the
        # same episodes and memory-mapped: one copy in the page cache, which the kernel can reclaim under memory
        # pressure, instead of a private copy per run.
        key = hashlib.sha1(json.dumps([task_id, self.episode_ids]).encode()).hexdigest()[:16]
        cache = os.path.join(os.environ["CVLA_DATA"], "frames", f"{task_id}.{key}.npy")
        with h5py.File(replay_prefix(task_id) + ".h5", "r") as f:
            groups = [f[f"traj_{ep}"] for ep in self.episode_ids]
            lengths = [len(g["actions"]) for g in groups]
            n = sum(lengths)
            first = groups[0]
            if not os.path.exists(cache):
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                tmp = f"{cache}.{os.getpid()}.tmp"
                images = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                                   shape=(n, *first[EPISODE_FIELDS["image"]].shape[1:]))
            else:
                images = None
            states = np.empty((n, first[EPISODE_FIELDS["state"]].shape[1]), dtype=np.float32)
            actions = np.empty((n, first["actions"].shape[1]), dtype=np.float32)
            ends = np.empty(n, dtype=np.int64)
            steps = np.empty((n, 2), dtype=np.int64)
            pos = 0
            for ep, g, T in zip(self.episode_ids, groups, lengths):
                rows = np.s_[pos:pos + T]
                if images is not None:
                    g[EPISODE_FIELDS["image"]].read_direct(images, np.s_[0:T], rows)
                g[EPISODE_FIELDS["state"]].read_direct(states, np.s_[0:T], rows)
                g["actions"].read_direct(actions, np.s_[0:T], rows)
                ends[rows] = pos + T
                steps[rows] = np.stack([np.full(T, ep), np.arange(T)], axis=1)
                pos += T
            if images is not None:
                images.flush()
                del images
                os.replace(tmp, cache)  # concurrent writers produce identical files
        images = np.load(cache, mmap_mode="c")  # copy-on-write: shared pages, never written
        assert images.shape[0] == n, f"{cache}: {images.shape[0]} frames, expected {n}"
        self.images = torch.from_numpy(images)
        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)
        self.ends = torch.from_numpy(ends)
        self.steps = steps  # (episode_id, t) of every example

    def __len__(self):
        return len(self.actions)

    def get(self, idx):
        offsets = idx[:, None] + torch.arange(self.chunk_size)
        end = self.ends[idx][:, None]
        is_pad = offsets >= end
        actions = self.actions[torch.minimum(offsets, end - 1)]
        actions[is_pad] = 0.0
        return self.images[idx], self.states[idx], actions, is_pad

    def index_of(self, steps):
        """Example indices of (episode_id, t) pairs."""
        lookup = {(int(e), int(t)): i for i, (e, t) in enumerate(self.steps)}
        return torch.tensor([lookup[(int(e), int(t))] for e, t in steps])


class ReplayBuffer:
    """A bounded store of one task's examples (image, joint state, action chunk, padding mask).
    Only these examples are kept on disk and loaded; the task's other demonstrations are not."""

    def __init__(self, prefix, limit=None, seed=0):
        tensors = load_file(prefix + ".safetensors")
        with open(prefix + ".json") as f:
            self.meta = json.load(f)
        keep = torch.arange(len(tensors["states"]))
        if limit is not None and limit < len(keep):
            keep = torch.randperm(len(keep), generator=torch.Generator().manual_seed(seed))[:limit]
        self.images, self.states, self.actions, self.is_pad = (
            tensors[k][keep].clone() for k in ("images", "states", "actions", "is_pad"))
        self.task_id = self.meta["task"]
        self.instruction = TASKS[self.task_id].instruction

    def __len__(self):
        return len(self.states)

    def get(self, idx):
        return self.images[idx], self.states[idx], self.actions[idx], self.is_pad[idx]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.images, self.states, self.actions, self.is_pad))

    @staticmethod
    def save(prefix, dataset, indices):
        images, states, actions, is_pad = dataset.get(indices)
        tensors = {"images": images, "states": states, "actions": actions, "is_pad": is_pad}
        save_file({k: v.contiguous() for k, v in tensors.items()}, prefix + ".safetensors")
        with open(prefix + ".json", "w") as f:
            json.dump({"task": dataset.task_id, "n": len(indices),
                       "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
                       "steps": dataset.steps[indices.numpy()].tolist()}, f)
