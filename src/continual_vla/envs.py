"""Simulator construction (kept free of model imports so evaluation worker processes stay small)."""

import gymnasium as gym

from .tasks import CONTROL_MODE, IMAGE_SIZE, ROBOT


def make_env(env_id, horizon):
    import mani_skill.envs  # noqa: F401  (registers environments)
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    from . import custom_tasks  # noqa: F401  (registers this project's extra tasks)

    env = gym.make(env_id, obs_mode="rgb", control_mode=CONTROL_MODE, robot_uids=ROBOT,
                   sim_backend="physx_cpu", sensor_configs=dict(width=IMAGE_SIZE, height=IMAGE_SIZE),
                   max_episode_steps=horizon)
    if env_id == "StackCube-v1":  # also report whether the opposite stacking order was achieved
        env = custom_tasks.OppositeStackInfo(env)
    return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
