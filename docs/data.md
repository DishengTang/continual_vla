# Data: from ManiSkill demonstrations to training examples

## Source demonstrations

[download_assets.py](../scripts/download_assets.py) fetches ManiSkill's official motion-planning
demonstrations for `PushCube-v1` and `StackCube-v1` (1000 episodes each) into
`$MS_ASSET_DIR/demos/<task>/motionplanning/`. The reversed stacking task, `StackCubeGreenOnRed-v1`
([custom_tasks.py](../src/continual_vla/custom_tasks.py)), has no published demonstrations;
[generate_demos.py](../scripts/generate_demos.py) makes them with ManiSkill's StackCube motion planner
(971 of 1000 attempts succeed). A planner sees the simulator's object poses; the learned policy never does.

An episode is one trial: a reset (the arm at its home pose plus small noise, the cubes and the push
target at positions drawn from the episode's seed), then the planner's motion until shortly after
success. The source files contain numbers only, recorded with `obs_mode: none`:

| `traj_N/...` | Shape | Meaning |
|---|---|---|
| `actions` | T × 8 | the planner's commands: 7 target joint angles (rad) and one gripper command (−1 close, +1 open) |
| `env_states/actors/<object>` | (T+1) × 13 | position (3), orientation quaternion (4), linear (3) and angular (3) velocity of each object |
| `env_states/articulations/panda` | (T+1) × 31 | robot base pose (7) and velocity (6), 9 joint positions, 9 joint velocities |
| `success`, `terminated`, `truncated`, `rewards` | T | per-step flags and ManiSkill's shaped reward |
| `obs` | empty | no observations were recorded |

The JSON next to each file lists every episode's seed, length and success.

## Replay into this project's observations and actions

[replay_demos.py](../scripts/replay_demos.py) runs ManiSkill's trajectory replay for every episode:
two simulators are reset with the episode's seed; one executes the stored joint commands, and the other
follows the same hand path with this project's controller (`pd_ee_delta_pos`: hand displacement x, y, z
plus gripper, 20 Hz) while rendering the 256 × 256 base camera at every step. Episodes whose converted
replay does not succeed are dropped (kept: PushCube 998, StackCube 983, green-on-red 956). The result is
`$CVLA_DATA/replay/<task>/trajectory.rgb.pd_ee_delta_pos.physx_cpu.{h5,json}`:

| `traj_N/...` | Shape | Used by the policy |
|---|---|---|
| `obs/sensor_data/base_camera/rgb` | (T+1) × 256 × 256 × 3, uint8 | **input image** |
| `obs/agent/qpos` | (T+1) × 9 | **input state**: 7 arm joint angles, 2 finger positions |
| `actions` | T × 4 | **training target**: hand displacement (1 = 0.1 m per step) and gripper |
| `obs/agent/qvel`, `obs/extra/tcp_pose`, `obs/sensor_param/*` | | not used |
| `env_states/*` | | not used (object poses: privileged) |
| `success`, `terminated`, `truncated` | T | only the episode-level success in the JSON is used, to keep successful episodes |

Row t of the observations is the state before action t. Demonstration lengths at 20 Hz:

| Task | Episodes | Steps: min / mean / 99th percentile / max | Mean duration |
|---|---|---|---|
| PushCube | 998 | 61 / 69 / 79 / 137 | 3.5 s |
| StackCube (red on green) | 983 | 80 / 109 / 207 / 412 | 5.5 s |
| Green on red | 956 | 81 / 110 / 205 / 806 | 5.5 s |

A few published StackCube demonstrations were recorded in scenes that this ManiSkill version no longer
draws from their seed; their replays keep the recorded scene. This does not affect evaluation, which
always resets fresh scenes from seeds.

## What reaches the policy

[observations.py](../src/continual_vla/observations.py) is the only path from simulator or file to the
policy: it keeps the camera image and the 9 joint positions and drops everything else. The instruction
comes from [tasks.py](../src/continual_vla/tasks.py). States are normalised as (qpos − home pose) / scale
and actions by the controller's range, using constants from the robot specification rather than
statistics of any task's data ([policy.py](../src/continual_vla/policy.py)).

## Training examples and splits

[data.py](../src/continual_vla/data.py): successful episodes are shuffled with a fixed seed and split
into training episodes (200 PushCube, 900 StackCube, 900 green-on-red) and 20 validation episodes per
task; every stage saves the lists in `splits.json`, so later stages validate an earlier task on the
episodes held out when it was learned. Every timestep of a training episode is one example: the image and
joint state at step t, the instruction, and the next 50 actions (padded and masked past the episode end).
That gives 13758 PushCube, 98730 StackCube and 99347 green-on-red examples. A replay buffer stores a fixed
random subset of one task's examples (50 examples = 9.4 MiB).

## Evaluation scenes

Closed-loop evaluation resets the simulator from fixed seeds: 100000+ for validation, 200000+ for test.
Demonstrations come from seeds 0-1025, so no test scene is a training scene. For the same seed both
stacking tasks produce the identical scene; only the instruction differs.
