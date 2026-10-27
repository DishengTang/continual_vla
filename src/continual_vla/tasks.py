"""Task sequence, supported instructions and the shared simulator configuration."""

from dataclasses import dataclass

ROBOT = "panda"
CONTROL_MODE = "pd_ee_delta_pos"  # end-effector XYZ delta + gripper, normalised to [-1, 1]
CAMERA = "base_camera"
IMAGE_SIZE = 256
CONTROL_FREQ = 20  # Hz

# Evaluation resets; demonstrations were generated with seeds 0-1025.
VAL_SEEDS = list(range(100000, 100500))
TEST_SEEDS = list(range(200000, 200500))
# Never used during development (2026-10-26): the final, frozen comparison is evaluated here once.
FRESH_SEEDS = list(range(300000, 300500))
SPLIT_SEEDS = {"val": VAL_SEEDS, "test": TEST_SEEDS, "fresh": FRESH_SEEDS}


@dataclass(frozen=True)
class Task:
    env_id: str
    instruction: str
    horizon: int  # evaluation episode length in control steps (fixed per task)


TASKS = {
    # Demonstrations: 61-137 steps, 99th percentile 79.
    "PushCube-v1": Task("PushCube-v1", "Push the cube onto the target.", 120),
    # Demonstrations: 80-412 steps, 99th percentile 208.
    "StackCube-v1": Task("StackCube-v1", "Stack the red cube on the green cube.", 250),
    # Same scene as StackCube-v1 for the same reset seed; only the requested order differs (custom_tasks.py).
    "StackCubeGreenOnRed-v1": Task("StackCubeGreenOnRed-v1", "Stack the green cube on the red cube.", 250),
}

# The two-task sequence of the main study: the tasks evaluated after a stage unless others are named.
SEQUENCE = ["PushCube-v1", "StackCube-v1"]
