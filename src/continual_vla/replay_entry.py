"""ManiSkill's trajectory replay with this project's extra tasks registered.

    python -m continual_vla.replay_entry --traj-path ... (same arguments as mani_skill.trajectory.replay_trajectory)

Replay workers are spawned and re-import this module, so the registration reaches them too.
"""

import multiprocessing as mp

from mani_skill.trajectory.replay_trajectory import main, parse_args

from continual_vla import custom_tasks  # noqa: F401  (registers extra tasks)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())
