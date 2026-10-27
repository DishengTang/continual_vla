"""Replay ManiSkill motion-planning demonstrations into the shared observation/action configuration.

    python scripts/replay_demos.py [--tasks PushCube-v1 StackCube-v1] [--count N] [--workers 8]

For each task, the source demonstrations (recorded with joint-position control) are replayed with
the Panda robot, end-effector delta-position control and RGB observations from the base camera
rendered at --image-size. ManiSkill converts the actions by closed-loop tracking of the original
trajectory; only replays that end in task success are kept. Output:
$CVLA_DATA/replay/<task>/trajectory.rgb.pd_ee_delta_pos.physx_cpu.{h5,json}
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

ROBOT = "panda"
CONTROL_MODE = "pd_ee_delta_pos"


def prepare_source(task, image_size):
    src_dir = os.path.join(os.environ["MS_ASSET_DIR"], "demos", task, "motionplanning")
    out_dir = os.path.join(os.environ["CVLA_DATA"], "replay", task)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(src_dir, "trajectory.h5"), out_dir)
    with open(os.path.join(src_dir, "trajectory.json")) as f:
        meta = json.load(f)
    env_kwargs = meta["env_info"]["env_kwargs"]
    env_kwargs["robot_uids"] = ROBOT
    sensor_configs = env_kwargs.get("sensor_configs", {})
    sensor_configs.update(width=image_size, height=image_size)
    env_kwargs["sensor_configs"] = sensor_configs
    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump(meta, f)
    return os.path.join(out_dir, "trajectory.h5")


def summarize(task):
    path = os.path.join(os.environ["CVLA_DATA"], "replay", task,
                        f"trajectory.rgb.{CONTROL_MODE}.physx_cpu.json")
    with open(path) as f:
        episodes = json.load(f)["episodes"]
    lengths = [e["elapsed_steps"] for e in episodes]
    print(f"{task}: kept {len(episodes)} successful replays, "
          f"length {min(lengths)}-{max(lengths)} steps (mean {sum(lengths) / len(lengths):.0f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["PushCube-v1", "StackCube-v1"])
    parser.add_argument("--count", type=int, default=None, help="demonstrations to replay per task")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    for task in args.tasks:
        traj = prepare_source(task, args.image_size)
        cmd = [sys.executable, "-m", "continual_vla.replay_entry",
               "--traj-path", traj, "-o", "rgb", "-c", CONTROL_MODE, "-b", "physx_cpu",
               "--save-traj", "-n", str(args.workers)]
        if args.count is not None:
            cmd += ["--count", str(args.count)]
        subprocess.run(cmd, check=True)
        summarize(task)


if __name__ == "__main__":
    main()
