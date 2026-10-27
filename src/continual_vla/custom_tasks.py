"""Additional simulator tasks: the opposite stacking goal in an identical scene.

StackCubeGreenOnRed-v1 asks for the green cube on the red cube. For the same reset seed its scene is
identical to StackCube-v1 (red on green): the cube colours are swapped and so are the cubes' initial
poses, so the same image and joint state can carry either instruction. Both stacking tasks also report
`opposite_success`, whether the other order was achieved, so completing the wrong goal is visible.
Importing this module registers the task (simulator worker processes import it via envs.make_env).
"""

import gymnasium as gym
import sapien
import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


def stacked(env, top, bottom):
    """StackCube's success test for an arbitrary pair: `top` rests on `bottom`, is static and released."""
    offset = top.pose.p - bottom.pose.p
    xy = torch.linalg.norm(offset[..., :2], axis=1) <= torch.linalg.norm(env.cube_half_size[:2]) + 0.005
    z = torch.abs(offset[..., 2] - env.cube_half_size[..., 2] * 2) <= 0.005
    return xy & z & top.is_static(lin_thresh=1e-2, ang_thresh=0.5) & ~env.agent.is_grasping(top)


@register_env("StackCubeGreenOnRed-v1", max_episode_steps=50)
class StackCubeGreenOnRedEnv(StackCubeEnv):
    """Put the green cube on the red cube; the cube to move (cubeA) is green."""

    def _load_scene(self, options: dict):
        self.cube_half_size = torch.tensor([0.02] * 3, device=self.device)
        from mani_skill.utils.scene_builder.table import TableSceneBuilder

        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.cubeA = actors.build_cube(self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeA",
                                       initial_pose=sapien.Pose(p=[0, 0, 0.1]))
        self.cubeB = actors.build_cube(self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeB",
                                       initial_pose=sapien.Pose(p=[1, 0, 0.1]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Sample exactly as StackCube-v1 does, then exchange the two cubes' poses: the red cube ends up
        # where StackCube-v1's red cube starts, and likewise for green.
        super()._initialize_episode(env_idx, options)
        pose_a = self.cubeA.pose.raw_pose.clone()
        pose_b = self.cubeB.pose.raw_pose.clone()
        self.cubeA.set_pose(Pose.create(pose_b))
        self.cubeB.set_pose(Pose.create(pose_a))

    def evaluate(self):
        info = super().evaluate()
        info["opposite_success"] = stacked(self, self.cubeB, self.cubeA)
        return info


class OppositeStackInfo(gym.Wrapper):
    """Adds `opposite_success` (green on red) to StackCube-v1's step info."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env = self.env.unwrapped
        info["opposite_success"] = stacked(env, env.cubeB, env.cubeA)
        return obs, reward, terminated, truncated, info
