"""SmolVLA configured for this project's observation and action layout."""

import math

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from safetensors.torch import load_file, save_file

from .tasks import IMAGE_SIZE

BASE = "lerobot/smolvla_base"
IMAGE_KEY = "observation.images.camera1"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_DIM = 9  # 7 arm joints + 2 gripper fingers
ACTION_DIM = 4  # end-effector XYZ delta + gripper

# Normalisation is fixed from the robot specification before any task data is seen, so no later
# task leaks into earlier stages: state = (qpos - rest pose) / scale; the controller already maps
# actions to [-1, 1], where 1 is a 0.1 m end-effector step, and typical steps are ~0.1 of that.
REST_QPOS = torch.tensor([0.0, math.pi / 8, 0.0, -5 * math.pi / 8, 0.0, 3 * math.pi / 4, math.pi / 4,
                          0.04, 0.04])
STATE_SCALE = torch.tensor([0.3] * 7 + [0.02] * 2)
ACTION_SCALE = torch.tensor([0.1, 0.1, 0.1, 1.0])


def build_policy(device="cuda"):
    """Pretrained SmolVLA; the vision-language backbone stays frozen and the action expert and
    projections are trainable, held in fp32 (bf16 weights would round away Adam updates)."""
    cfg = PreTrainedConfig.from_pretrained(BASE)
    cfg.input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
    }
    cfg.output_features = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(BASE, config=cfg)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    policy.to(device)
    stats = {
        STATE_KEY: {"mean": REST_QPOS, "std": STATE_SCALE},
        ACTION_KEY: {"mean": torch.zeros(ACTION_DIM), "std": ACTION_SCALE},
    }
    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=stats)
    return policy, pre, post


def raw_batch(images, states, instructions, actions=None, is_pad=None, device="cuda"):
    """Policy input from uint8 images (B, H, W, 3), joint positions (B, 9) and instructions."""
    images = images.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    batch = {IMAGE_KEY: images, STATE_KEY: states.to(device).float(), "task": list(instructions)}
    if actions is not None:
        batch[ACTION_KEY] = actions.to(device)
        batch["action_is_pad"] = is_pad.to(device)
    return batch


def trainable_named(policy):
    return {n: p for n, p in policy.named_parameters() if p.requires_grad}


def save_trainable(policy, path):
    save_file({n: p.detach().cpu().contiguous() for n, p in trainable_named(policy).items()}, path)


def load_trainable(policy, path):
    state = load_file(path)
    for n, p in trainable_named(policy).items():
        p.data.copy_(state[n].to(p.device))
