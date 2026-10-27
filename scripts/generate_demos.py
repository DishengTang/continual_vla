"""Generate motion-planning demonstrations for a stacking task registered by this project.

    PYTHONPATH=src $CVLA_ROOT/envs/mp/bin/python scripts/generate_demos.py --task StackCubeGreenOnRed-v1 --n 1000

Runs in the planner environment from scripts/setup_planner_env.sh (mplib 0.1.1 needs NumPy < 2).

Uses ManiSkill's StackCube motion-planning solver, which moves `cubeA` onto `cubeB` using the simulator's
object poses, exactly as ManiSkill's published StackCube-v1 demonstrations were made. The learned policy
never receives that state. Only successful episodes are kept. Recorded with joint-position control and
no observations (CPU only); scripts/replay_demos.py then converts them like the published demonstrations.
Output: $MS_ASSET_DIR/demos/<task>/motionplanning/trajectory.{h5,json}
"""

import argparse
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor


def worker(task, seeds, out_dir, proc):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.examples.motionplanning.panda.solutions.stack_cube import solve
    from mani_skill.utils.wrappers.record import RecordEpisode

    from continual_vla import custom_tasks  # noqa: F401

    env = gym.make(task, obs_mode="none", control_mode="pd_joint_pos", render_mode="rgb_array",
                   reward_mode="none", sim_backend="physx_cpu", robot_uids="panda")
    env = RecordEpisode(env, output_dir=out_dir, trajectory_name=f"trajectory_{proc}", save_video=False,
                        source_type="motionplanning", source_desc="ManiSkill StackCube motion-planning solver",
                        record_reward=False, save_on_reset=False)
    kept = 0
    for seed in seeds:
        try:
            res = solve(env, seed=seed, debug=False, vis=False)
            ok = res != -1 and bool(res[-1]["success"].item())
        except Exception as exc:  # planner failures are skipped, like ManiSkill's own runner
            print(f"seed {seed}: {exc!r}", flush=True)
            ok = False
        env.flush_trajectory(save=ok)
        kept += ok
    env.close()
    return os.path.join(out_dir, f"trajectory_{proc}.h5"), kept


def main():
    from mani_skill.trajectory.merge_trajectory import merge_trajectories

    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--n", type=int, default=1000, help="reset seeds 0..n-1 to attempt")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    out = os.path.join(os.environ["MS_ASSET_DIR"], "demos", args.task, "motionplanning")
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    jobs = [(args.task, list(range(i, args.n, args.workers)), parts, i) for i in range(args.workers)]
    # Unlike multiprocessing.Pool, the executor raises (BrokenProcessPool) if a worker crashes.
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(worker, *zip(*jobs)))
    files = [path for path, kept in results if kept]
    merge_trajectories(os.path.join(out, "trajectory.h5"), files)
    shutil.rmtree(parts)
    print(f"{args.task}: kept {sum(k for _, k in results)} of {args.n} planned episodes -> {out}/trajectory.h5")


if __name__ == "__main__":
    main()
