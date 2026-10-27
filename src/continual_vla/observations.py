"""The only path from simulator observations to policy inputs.

The policy sees the base-camera RGB image and the robot's joint positions (arm + gripper fingers).
Everything else the simulator returns (object poses, end-effector pose, camera parameters, rewards,
success flags, environment states) is dropped here.
"""

from .tasks import CAMERA


def from_env(obs):
    """Whitelist a (batched) observation dict returned by the environment."""
    return {"image": obs["sensor_data"][CAMERA]["rgb"], "state": obs["agent"]["qpos"]}


# Whitelisted fields of a replayed demonstration (h5py dataset paths within one episode group).
EPISODE_FIELDS = {"image": f"obs/sensor_data/{CAMERA}/rgb", "state": "obs/agent/qpos"}


def from_episode(group):
    """Whitelist the observations of one replayed demonstration (an h5py group)."""
    return {name: group[path][:] for name, path in EPISODE_FIELDS.items()}
