"""Write an additional replay buffer for a finished stage, drawn from that stage's training split.

    python training/make_replay_buffer.py --stage $CVLA_RUNS/final/s0/A_plus --size 500 --name memory500

The buffer is sampled the same way every training stage samples its own (uniformly without replacement, seeded
by the stage's training seed), independently of the stage's existing buffer, and saved as
<stage>/<name>.{safetensors,json}. Use it with train_stage.py --method er --memory-name <name>.
"""

import argparse
import json
import os


def main():
    import numpy as np
    import torch

    from continual_vla.data import DemoSet, ReplayBuffer, load_splits

    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()
    with open(os.path.join(args.stage, "config.json")) as f:
        cfg = json.load(f)
    task = cfg["task"]
    data = DemoSet(task, load_splits(args.stage)[task]["train"], chunk_size=50)
    rng = np.random.default_rng(cfg["seed"])
    chosen = np.sort(rng.choice(len(data), size=min(args.size, len(data)), replace=False))
    prefix = os.path.join(args.stage, args.name)
    ReplayBuffer.save(prefix, data, torch.from_numpy(chosen))
    meta = json.load(open(prefix + ".json"))
    print(f"wrote {prefix}: {meta['n']} examples of {task}, {meta['bytes'] / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
