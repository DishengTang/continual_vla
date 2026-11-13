"""Closed-loop evaluation in LIBERO with LeRobot's environment and rollout loop (lerobot.envs.libero,
lerobot.scripts.lerobot_eval.eval_policy), following openpi's LIBERO evaluation as Liu et al. run it: 50 episodes per
task from LIBERO's fixed initial states, a new action chunk every `n_action_steps` (5) actions, openpi's episode
horizons (suites.MAX_STEPS) after 10 settling steps, environment seed 7.

The task's own normalisation statistics are used (as Liu et al.'s policy server loads them per task); the relative
action dimensions are made absolute with the state the chunk was predicted from. Split "test" starts episode i from
LIBERO's fixed initial state i of the task; split "val" draws new initial scenes with seeds VAL_SEEDS (suites.py). The
camera images are rotated by 180 degrees to the dataset's convention (LiberoProcessorStep). Success is read from the
simulator only to score the episode.
"""

import os

import torch

from . import suites
from .policy import STATE_KEY, normalise, unnormalise

ENV_SEED = 7  # openpi's examples/libero/main.py


def _gpu_for_rendering():
    """MuJoCo renders with EGL on the GPU the policy uses (the first of CUDA_VISIBLE_DEVICES)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", visible.split(",")[0])


def _new_chunk(policy):
    """Whether the policy predicts a new action chunk at its next select_action call."""
    queue = getattr(policy, "_action_queue", None)
    if queue is None:
        queue = policy._queues["action"]  # SmolVLA
    return len(queue) == 0


@torch.no_grad()
def evaluate(policy, pre, post, task, stats, n_episodes=suites.N_TEST_EPISODES, split="test", n_envs=10, seed=0,
             context_fn=None, video_dir=None, n_video=0):
    """{"success": fraction of episodes, "episodes": [0/1 per episode], ...} for `task` ("<suite>/<index>"), with the
    task's normalisation `stats` (data.task_stats; None: `pre` and `post` normalise themselves and actions are absolute,
    e.g. LeRobot's reference checkpoint). `context_fn(instructions)` is called before every prediction (e.g. to set a
    context gate); `seed` seeds the policy's noise."""
    _gpu_for_rendering()
    suites._configure()
    from lerobot.envs.configs import LiberoEnv
    from lerobot.scripts.lerobot_eval import eval_policy

    suite, index = suites.parse(task)
    assert split in ("test", "val"), split
    if split == "test":
        assert n_episodes <= suites.N_TEST_EPISODES and suites.N_TEST_EPISODES % n_envs == 0
    env_cfg = LiberoEnv(task=suite, task_ids=[index], observation_height=256, observation_width=256,
                        init_states=split == "test", episode_length=suites.MAX_STEPS[suite])
    env = env_cfg.create_envs(n_envs, use_async_envs=n_envs > 1)[suite][index]
    env_pre, env_post = env_cfg.get_env_processors()
    chunk = {}

    def preprocess(observation):
        if context_fn is not None:
            context_fn(list(observation["task"]))
        if stats is None:
            return pre(observation)
        raw = observation[STATE_KEY]
        if _new_chunk(policy):
            chunk["state"] = raw.clone()  # the state this chunk's relative actions refer to
        return pre({**observation, STATE_KEY: normalise(raw, None, stats)[0]})

    def postprocess(action):
        return post(action) if stats is None else unnormalise(post(action), chunk["state"], stats)

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)  # the policy's noise samples, the same at every evaluation
        try:
            out = eval_policy(env, policy, env_pre, env_post, preprocess, postprocess, n_episodes=n_episodes,
                              max_episodes_rendered=n_video, videos_dir=video_dir,
                              start_seed=suites.VAL_SEEDS[0] if split == "val" else ENV_SEED)
        finally:
            env.close()
    episodes = [int(bool(e["success"])) for e in out["per_episode"]][:n_episodes]
    return {"task": task, "split": split, "episodes": episodes, "success": sum(episodes) / len(episodes),
            "n_episodes": len(episodes), "instruction": suites.instruction(task),
            "n_action_steps": policy.config.n_action_steps}
