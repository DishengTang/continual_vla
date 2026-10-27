"""Closed-loop evaluation in the simulator from fixed reset seeds.

Each episode runs for the task's fixed horizon. The policy predicts a 50-step action chunk and
executes the first `exec_steps` actions before observing again. Success is read from the
simulator only to score the episode; it never reaches the policy.
"""

import functools
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from .envs import make_env
from .observations import from_env
from .policy import raw_batch
from .tasks import CONTROL_FREQ, TASKS


_renderer = None


def _keep_renderer():
    """Keep one render system alive in this process. Every vector environment builds and closes a dummy
    environment here, and the renderer's Vulkan context goes with the last render system; the 535.230.02
    driver allows only four such contexts per process, so later evaluations of a run would fail."""
    global _renderer
    if _renderer is None:
        import sapien
        _renderer = sapien.render.RenderSystem(sapien.Device("cuda"))


@torch.no_grad()
def evaluate(policy, pre, post, task_id, seeds, n_envs=10, exec_steps=5, video_dir=None, n_video=0,
             device="cuda", context_fn=None, record_traces=False, video_all=False):
    """Returns per-episode success (at any step, and at the final step) for the given seeds.
    `context_fn(instructions)` is called before every prediction (e.g. to set a ContextGate).
    With `record_traces`, also returns each episode's per-step success as a string of 0/1 and, for the
    stacking tasks, its per-step state: S success, H cube on the other cube but still held, M on it and
    released but moving, O not on it. Videos go to `video_dir`: the first `n_video` episodes, or every
    episode with `video_all` (to choose examples from the same run that is scored)."""
    task = TASKS[task_id]
    assert len(seeds) % n_envs == 0, "number of seeds must be a multiple of n_envs"
    _keep_renderer()
    envs = gym.vector.AsyncVectorEnv(
        [functools.partial(make_env, task_id, task.horizon) for _ in range(n_envs)], context="spawn")
    policy.eval()
    success_once, success_end, traces, states, opposite = [], [], [], [], {"once": [], "end": []}
    # Every evaluation uses the same flow-matching noise sequence, drawn from a forked RNG so the
    # caller's (training) random streams continue exactly as if no evaluation had happened.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0)
        _rollouts(policy, pre, post, task, task_id, seeds, envs, n_envs, exec_steps, video_dir, n_video,
                  device, context_fn, success_once, success_end, traces, states, opposite, video_all)
    result = {"task": task_id, "n": len(seeds), "success": float(np.mean(success_once)),
              "success_end": float(np.mean(success_end)), "episodes": success_once,
              "episodes_end": success_end}
    if opposite["once"]:  # stacking tasks: the other stacking order (completing the wrong goal)
        result.update(opposite_success=float(np.mean(opposite["once"])),
                      opposite_success_end=float(np.mean(opposite["end"])),
                      opposite_episodes=opposite["once"], opposite_episodes_end=opposite["end"])
    if record_traces:
        result["traces"] = traces
        if states:
            result["states"] = states
    return result


def _rollouts(policy, pre, post, task, task_id, seeds, envs, n_envs, exec_steps, video_dir, n_video,
              device, context_fn, success_once, success_end, traces, states, opposite, video_all=False):
    try:
        for start in range(0, len(seeds), n_envs):
            batch_seeds = seeds[start:start + n_envs]
            obs, _ = envs.reset(seed=list(batch_seeds))
            once = np.zeros(n_envs, dtype=bool)
            opp_once = np.zeros(n_envs, dtype=bool)
            trace = np.zeros((n_envs, task.horizon), dtype=bool)
            state = np.full((n_envs, task.horizon), "O")
            frames = []
            for t in range(task.horizon):
                inputs = from_env(obs)
                if t % exec_steps == 0:
                    if context_fn is not None:
                        context_fn([task.instruction] * n_envs)
                    batch = pre(raw_batch(torch.from_numpy(inputs["image"]), torch.from_numpy(inputs["state"]),
                                          [task.instruction] * n_envs, device=device))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        chunk = policy.predict_action_chunk(batch)
                    chunk = post(chunk).float().cpu().numpy()
                if video_dir is not None and (video_all or start == 0):
                    frames.append(inputs["image"] if video_all else inputs["image"][:n_video])
                obs, _, _, _, info = envs.step(chunk[:, t % exec_steps])
                trace[:, t] = np.asarray(info["success"], dtype=bool)
                once |= trace[:, t]
                if "is_cubeA_on_cubeB" in info:  # stacking tasks: why a step is not a success
                    on = np.asarray(info["is_cubeA_on_cubeB"], dtype=bool)
                    held = np.asarray(info["is_cubeA_grasped"], dtype=bool)
                    state[:, t] = np.where(trace[:, t], "S", np.where(~on, "O", np.where(held, "H", "M")))
                if "opposite_success" in info:
                    opp_once |= np.asarray(info["opposite_success"], dtype=bool)
            end = trace[:, -1]
            success_once.extend(once.tolist())
            success_end.extend(end.tolist())
            if "opposite_success" in info:
                opposite["once"].extend(opp_once.tolist())
                opposite["end"].extend(np.asarray(info["opposite_success"], dtype=bool).tolist())
            traces.extend("".join("1" if v else "0" for v in row) for row in trace)
            if "is_cubeA_on_cubeB" in info:
                states.extend("".join(row) for row in state)
            if frames:
                os.makedirs(video_dir, exist_ok=True)
                frames = np.stack(frames, axis=1)
                for i in range(n_envs if video_all else min(n_video, n_envs)):
                    label = "success" if once[i] else "failure"
                    path = os.path.join(video_dir, f"{task_id}_seed{batch_seeds[i]}_{label}.mp4")
                    imageio.mimsave(path, list(frames[i]), fps=CONTROL_FREQ)
    finally:
        envs.close()
