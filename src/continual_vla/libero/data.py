"""LIBERO demonstrations (the four suites in LeRobot format, HuggingFaceVLA/libero: failed and no-op steps removed as
in OpenVLA, 1693 episodes) as (two camera images, state, action chunk) training examples of one task.

LiberoDemos and LiberoReplayBuffer have the interface of continual_vla.data.DemoSet and ReplayBuffer, so the methods
use them unchanged: get(idx) -> (images (B, 2, 256, 256, 3) uint8: third-person then wrist camera, states (B, 8):
end-effector position, axis-angle and gripper fingers, actions (B, chunk, 7): end-effector delta and gripper, is_pad).
The decoded frames of a task are written once to a file shared by every run and memory-mapped, or read into memory
(in_memory=True) where random reads from that file are slow, e.g. on a network disk.
"""

import glob
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from . import suites

DATASET = "HuggingFaceVLA/libero"
IMAGE_COLUMNS = ("observation.images.image", "observation.images.image2")

np._core.multiarray._set_madvise_hugepage(False)  # see continual_vla/data.py


def dataset_root():
    return os.path.join(os.environ["HF_LEROBOT_HOME"], DATASET)


# openpi's LIBERO action transform (DeltaActions(make_bool_mask(6, -1))), kept by Liu et al.: the first six action
# dimensions (end-effector translation and rotation) become relative to the state at the start of the chunk.
DELTA_DIMS = 6


def to_delta(actions, states):
    """(B, chunk, 7) actions relative to the (B, 8) states the chunks start from, in the first DELTA_DIMS dims."""
    out = actions.clone()
    out[..., :DELTA_DIMS] -= states[:, None, :DELTA_DIMS]
    return out


def task_stats(task, chunk_size, root=None):
    """Normalisation statistics of one task, as Liu et al. compute them (openpi's compute_norm_stats on each task's own
    dataset): mean and standard deviation of the state, and of every action of every training chunk after to_delta
    (chunks past an episode's end repeat its last action). Written once to $CVLA_DATA/libero_stats/."""
    path = os.path.join(os.environ["CVLA_DATA"], "libero_stats", f"{task.replace('/', '_')}.chunk{chunk_size}.json")
    if not os.path.exists(path):
        demos = LiberoDemos(task, chunk_size, root=root, load_frames=False)
        _, states, actions, _ = demos.get(torch.arange(len(demos)))
        delta = to_delta(actions, states).reshape(-1, actions.shape[-1]).double()
        stats = {"observation.state": {"mean": states.double().mean(0).tolist(), "std": states.double().std(0, unbiased=False).tolist()},
                 "action": {"mean": delta.mean(0).tolist(), "std": delta.std(0, unbiased=False).tolist()},
                 "n_frames": len(demos), "episodes": demos.episode_ids}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + f".{os.getpid()}.tmp", "w") as f:
            json.dump(stats, f)
        os.replace(path + f".{os.getpid()}.tmp", path)
    with open(path) as f:
        stats = json.load(f)
    return {k: {s: torch.tensor(stats[k][s], dtype=torch.float32) for s in ("mean", "std")}
            for k in ("observation.state", "action")}


def task_episodes(task, root=None):
    """Episode indices of a task's demonstrations, found by its instruction."""
    root = root or dataset_root()
    text = suites.instruction(task)
    episodes = pq.read_table(sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet"))),
                             columns=["episode_index", "tasks"]).to_pylist()
    ids = sorted(e["episode_index"] for e in episodes if list(e["tasks"]) == [text])
    assert ids, f"{task}: no demonstrations with instruction {text!r}"
    return ids


def _decode(png):
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


class LiberoDemos:
    """All timesteps of one task's demonstrations. Example i is the observation at step i and the chunk of the next
    `chunk_size` actions, padded past the end of its episode (see get)."""

    def __init__(self, task, chunk_size, root=None, in_memory=False, load_frames=True):
        root = root or dataset_root()
        self.task_id = task
        self.instruction = suites.instruction(task)
        self.episode_ids = task_episodes(task, root)
        self.chunk_size = chunk_size
        wanted = set(self.episode_ids)
        files = sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet")))
        tables = []
        for path in files:
            meta = pq.read_table(path, columns=["episode_index"]).column("episode_index").to_numpy()
            if np.isin(meta, list(wanted)).any():
                tables.append(path)
        cols = ["index", "episode_index", "frame_index", "observation.state", "action"]
        parts = [pq.read_table(p, columns=cols).to_pandas() for p in tables]
        df = pd.concat(parts)
        df = df[df["episode_index"].isin(wanted)].sort_values("index")
        n = len(df)
        self.states = torch.tensor(np.stack(df["observation.state"].to_numpy()), dtype=torch.float32)
        self.actions = torch.tensor(np.stack(df["action"].to_numpy()), dtype=torch.float32)
        episodes = df["episode_index"].to_numpy()
        frames = df["frame_index"].to_numpy()
        starts = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1]
        ends = np.empty(n, dtype=np.int64)
        for s, e in zip(starts, np.r_[starts[1:], n]):
            assert (frames[s:e] == np.arange(e - s)).all(), f"{task}: episode {episodes[s]} frames out of order"
            ends[s:e] = e
        self.ends = torch.from_numpy(ends)
        self.steps = np.stack([episodes, frames], axis=1)  # (episode, t) of every example

        self.images = None
        if not load_frames:  # states and actions only
            return
        key = hashlib.sha1(json.dumps([task, self.episode_ids, n]).encode()).hexdigest()[:16]
        cache = os.path.join(os.environ["CVLA_DATA"], "frames", "libero", f"{task.replace('/', '_')}.{key}.npy")
        if not os.path.exists(cache):
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            tmp = f"{cache}.{os.getpid()}.tmp"
            images = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, 2, 256, 256, 3))
            position = {int(i): r for r, i in enumerate(df["index"].to_numpy())}
            with ThreadPoolExecutor(16) as pool:
                for path in tables:
                    t = pq.read_table(path, columns=["index", "episode_index", *IMAGE_COLUMNS])
                    keep = np.isin(t.column("episode_index").to_numpy(), list(wanted))
                    if not keep.any():
                        continue
                    t = t.filter(keep)
                    idx = [position[int(i)] for i in t.column("index").to_numpy()]
                    for c, col in enumerate(IMAGE_COLUMNS):
                        pngs = [v["bytes"] for v in t.column(col).to_pylist()]
                        for r, img in zip(idx, pool.map(_decode, pngs)):
                            images[r, c] = img
            images.flush()
            del images
            os.replace(tmp, cache)  # concurrent writers produce identical files
        images = np.load(cache) if in_memory else np.load(cache, mmap_mode="c")
        assert images.shape[0] == n, f"{cache}: {images.shape[0]} frames, expected {n}"
        self.images = torch.from_numpy(images)

    def __len__(self):
        return len(self.actions)

    def get(self, idx):
        """Past the end of its episode a chunk repeats the last action, as in LeRobot datasets: pi0's loss does not
        mask padding, so the padded steps teach it to hold the final command (SmolVLA masks them with is_pad)."""
        offsets = idx[:, None] + torch.arange(self.chunk_size)
        end = self.ends[idx][:, None]
        is_pad = offsets >= end
        actions = self.actions[torch.minimum(offsets, end - 1)]
        return (None if self.images is None else self.images[idx]), self.states[idx], actions, is_pad


class LiberoReplayBuffer:
    """A bounded store of one task's examples; only these are kept on disk and loaded (continual_vla.data.ReplayBuffer
    for LIBERO tasks: the instruction is stored with the examples)."""

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
        self.instruction = self.meta["instruction"]

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
            json.dump({"task": dataset.task_id, "instruction": dataset.instruction, "n": len(indices),
                       "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
                       "steps": dataset.steps[indices.numpy()].tolist()}, f)
