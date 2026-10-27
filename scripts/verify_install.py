"""Check the GPU, the simulator renderer and the shared task configuration.

    python scripts/verify_install.py [--out DIR]

Prints every observation field the simulator returns for each task, so the policy-input
whitelist can be checked against it, and saves one RGB frame per task camera.
"""

import argparse
import os
import time

import numpy as np
import torch

TASKS = ["PushCube-v1", "StackCube-v1"]
ROBOT = "panda"
CONTROL_MODE = "pd_ee_delta_pos"


def check_torch():
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda})")
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device {name}, capability sm_{cap[0]}{cap[1]}, arch list {torch.cuda.get_arch_list()}")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = a @ a  # warm up cuBLAS before timing
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        b = a @ a
    torch.cuda.synchronize()
    tflops = 20 * 2 * 4096**3 / (time.time() - t0) / 1e12
    print(f"bf16 matmul ok: {tflops:.0f} TFLOP/s, finite={bool(torch.isfinite(b).all())}")


def describe(prefix, obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            describe(f"{prefix}/{k}" if prefix else k, v)
    else:
        shape = tuple(obj.shape) if hasattr(obj, "shape") else type(obj).__name__
        dtype = getattr(obj, "dtype", "")
        print(f"  {prefix}: {shape} {dtype}")


def check_task(env_id, out_dir):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers environments)

    env = gym.make(env_id, obs_mode="rgb", control_mode=CONTROL_MODE, robot_uids=ROBOT,
                   render_mode="rgb_array", num_envs=1)
    obs, _ = env.reset(seed=0)
    print(f"{env_id}: action space {env.action_space}, control freq {env.unwrapped.control_freq} Hz,"
          f" max episode steps {env.spec.max_episode_steps}")
    describe("", obs)
    t0 = time.time()
    n = 50
    for _ in range(n):
        obs, *_ = env.step(env.action_space.sample())
    print(f"  {n} steps with RGB rendering: {n / (time.time() - t0):.0f} steps/s")
    import imageio.v2 as imageio
    for cam, data in obs["sensor_data"].items():
        rgb = data["rgb"][0].cpu().numpy().astype(np.uint8)
        path = os.path.join(out_dir, f"{env_id}_{cam}.png")
        imageio.imwrite(path, rgb)
        print(f"  saved {path}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.environ.get("CVLA_RUNS", "."), "verify"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    check_torch()
    for env_id in TASKS:
        check_task(env_id, args.out)
    print("all checks passed")


if __name__ == "__main__":
    main()
