"""pi0 and SmolVLA configured for LIBERO: two cameras (third-person and wrist, 256 x 256), the 8-dimensional state and
the 7-dimensional end-effector action. Which parameters train is the `train` setting:

  expert  the action expert and the state, action and time projections; the vision-language model stays frozen
  lora    openpi's low-memory pi0 recipe (pi0_libero_low_mem_finetune, used by Liu et al., 2026): LoRA adapters on the
          attention and MLP layers of the language model (rank 16) and of the action expert (rank 32), alpha = rank,
          while the vision encoder and the state, action and time projections train in full
  full    every parameter

Trainable parameters are held in fp32 (bf16 weights would round away Adam updates) and the forward pass runs under
bf16 autocast. pi0's third camera slot gets an empty image, as in LeRobot's pi0 LIBERO fine-tuning. The two token
output heads (lm_head) of LeRobot's pi0 are never used to predict actions and stay frozen in every mode.

States and actions are normalised as in Liu et al.'s openpi code: each task with its own statistics (data.task_stats),
actions relative to the chunk's starting state in their first six dimensions (data.to_delta), x -> (x - mean) /
(std + 1e-6). normalise() does this per example, so a batch can mix tasks (replay); LeRobot's processors then only
tokenise the instruction and move the batch (their normalisation is the identity). Training images get openpi's
augmentation (augment()).
"""

import os

import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

from ..policy import load_trainable, save_trainable, trainable_named  # noqa: F401 (re-exported)

BASES = {"pi0": "lerobot/pi0_base", "smolvla": "lerobot/smolvla_base"}
TRAIN_MODES = ("expert", "lora", "full")
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_DIM, ACTION_DIM, IMAGE_SIZE = 8, 7, 256
# openpi's gemma_2b_lora and gemma_300m_lora
LORA_RANKS = {"language": 16, "expert": 32}
LORA_LAYERS = r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def tokenizer_dir():
    return os.path.join(os.environ["CVLA_DATA"], "tokenizers", "paligemma")


def _add_lora(policy):
    """openpi's LoRA layout: adapters on the language model's and the action expert's linear layers; every other
    weight of the language model and the expert (embeddings, norms) is frozen, the rest of pi0 trains in full."""
    from peft import LoraConfig, inject_adapter_in_model

    model = policy.model.paligemma_with_expert
    for part, prefix in (("language", r"paligemma\.model\.language_model\."), ("expert", r"gemma_expert\.model\.")):
        rank = LORA_RANKS[part]
        config = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=prefix + LORA_LAYERS,
                            bias="none")
        inject_adapter_in_model(config, model, adapter_name=part)
    for name, p in policy.named_parameters():
        in_llm = ".language_model." in name or ".gemma_expert." in name
        p.requires_grad_(not in_llm or ".lora_" in name)


def build_policy(model="pi0", train="expert", n_action_steps=5, device="cuda"):
    """The pretrained policy with LIBERO's inputs and outputs, `train` deciding the trainable parameters (above).
    Returns (policy, pre, post); pre and post expect normalised states and actions (normalise, unnormalise)."""
    assert train in TRAIN_MODES, train
    cfg = PreTrainedConfig.from_pretrained(BASES[model])
    cfg.input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
                          for k in IMAGE_KEYS}
    cfg.input_features[STATE_KEY] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    cfg.output_features = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    cfg.n_action_steps = n_action_steps
    cfg.device = device
    cfg.train_expert_only = train == "expert"
    if model == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as Policy

        cfg.empty_cameras = 1
        cfg.dtype = "bfloat16"
        cfg.gradient_checkpointing = train != "expert"  # expert: expert_loss never backpropagates into the VLM
    else:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy

        assert train == "expert", "SmolVLA is trained with its vision-language model frozen"
    policy = Policy.from_pretrained(BASES[model], config=cfg)
    if train == "full":
        for p in policy.parameters():
            p.requires_grad_(True)
    elif train == "lora":
        _add_lora(policy)
    for name, p in policy.named_parameters():
        if "lm_head" in name:
            p.requires_grad_(False)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    policy.to(device)
    pre, post = build_processors(policy.config, model, IDENTITY)
    return policy, pre, post


def build_processors(cfg, model, stats):
    if model == "smolvla":
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        return make_smolvla_pre_post_processors(cfg, dataset_stats=stats)
    # make_pi0_pre_post_processors of LeRobot 0.6.1, with the PaliGemma tokenizer loaded from tokenizer_dir() (the
    # official repository is gated; scripts/download_libero.py checks the files against its hashes).
    from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor
    from lerobot.processor import (AbsoluteActionsProcessorStep, RelativeActionsProcessorStep,
                                   TokenizerProcessorStep)
    from lerobot.processor.factory import make_default_policy_processor_steps, make_policy_processor_pipelines
    from transformers import AutoTokenizer

    relative = RelativeActionsProcessorStep(enabled=cfg.use_relative_actions,
                                            exclude_joints=getattr(cfg, "relative_exclude_joints", []),
                                            action_names=getattr(cfg, "action_feature_names", None))
    steps = make_default_policy_processor_steps(cfg, stats)
    tokenizer = TokenizerProcessorStep(tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir()),
                                       max_length=cfg.tokenizer_max_length, padding_side="right", padding="max_length")
    input_steps = [steps.rename_observations, steps.add_batch_dim, Pi0NewLineProcessor(), tokenizer, steps.to_device,
                   relative, steps.normalize]
    output_steps = [steps.unnormalize, AbsoluteActionsProcessorStep(enabled=cfg.use_relative_actions,
                                                                    relative_step=relative), steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


IDENTITY = {STATE_KEY: {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
            ACTION_KEY: {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)}}
EPS = 1e-6  # openpi's Normalize


def normalise(states, actions, stats):
    """Raw (B, 8) states and (B, chunk, 7) actions of one task -> normalised, actions made relative first."""
    from .data import to_delta

    s, a = stats[STATE_KEY], stats[ACTION_KEY]
    states_n = (states - s["mean"]) / (s["std"] + EPS)
    if actions is None:
        return states_n, None
    return states_n, (to_delta(actions, states) - a["mean"]) / (a["std"] + EPS)


def unnormalise(actions, chunk_states, stats):
    """Normalised (B, 7) actions -> commands for the simulator, given the raw states their chunk started from."""
    from .data import DELTA_DIMS

    a = stats[ACTION_KEY]
    out = actions * (a["std"].to(actions) + EPS) + a["mean"].to(actions)
    out[:, :DELTA_DIMS] += chunk_states[:, :DELTA_DIMS].to(actions)
    return out


def _rgb_to_hsv(x):
    r, g, b = x.unbind(1)
    v, _ = x.max(1)
    c = v - x.min(1)[0]
    s = torch.where(v > 0, c / v.clamp_min(1e-8), torch.zeros_like(v))
    cc = c.clamp_min(1e-8)
    h = torch.where(v == r, ((g - b) / cc) % 6, torch.where(v == g, (b - r) / cc + 2, (r - g) / cc + 4)) / 6
    return torch.where(c > 0, h, torch.zeros_like(h)), s, v


def _hsv_to_rgb(h, s, v):
    k = lambda n: (n + h[:, None] * 6) % 6  # noqa: E731
    n = torch.tensor([5.0, 3.0, 1.0], device=h.device).view(1, 3, 1, 1)
    kk = k(n)
    return v[:, None] - v[:, None] * s[:, None] * torch.clamp(torch.minimum(kk, 4 - kk), 0, 1)


def _color_jitter(x):
    """augmax's ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5), as openpi calls it, on (N, 3, H, W) images in
    [0, 1]: with probability 0.5 per image, in HSV: brightness b ~ U(-0.3, 0.3) and an invertible contrast curve
    c ~ U(-0.4, 0.4) on the value channel, the hue shifted by U(-0.1, 0.1) (augmax's default hue); augmax's saturation
    adjustment does not change the image."""
    n, dev = x.shape[0], x.device
    u = lambda lim: ((torch.rand(n, device=dev) * 2 - 1) * lim).view(n, 1, 1)  # noqa: E731
    h, s, v = _rgb_to_hsv(x)
    b = u(0.3)
    v = torch.where(b < 0, v * (1 + b), v * (1 - b) + b)
    slant = torch.tan((u(0.4) + 1) * (torch.pi / 4))
    p1 = (slant - slant ** 2) / (2 * (1 - slant ** 2))
    v = torch.where(v < p1, v / slant, torch.where(v > 1 - p1, v / slant + 1 - 1 / slant, slant * (v - 0.5) + 0.5))
    h = (h + u(0.1)) % 1
    out = _hsv_to_rgb(h, s, v)
    apply = (torch.rand(n, device=dev) < 0.5).view(n, 1, 1, 1)
    return torch.where(apply, out, x)


def augment(images):
    """openpi's training augmentation (preprocess_observation with train=True, used by Liu et al.) on (B, 2, 3, H, W)
    images in [0, 1]: the third-person camera is cropped to a random 95% window, resized back and rotated by
    U(-5, 5) degrees (one affine resampling); both cameras get _color_jitter."""
    b, dev = images.shape[0], images.device
    out = images.clone()
    theta = torch.deg2rad((torch.rand(b, device=dev) * 2 - 1) * 5)
    shift = (torch.rand(b, 2, device=dev) * 2 - 1) * (1 - 0.95)  # crop centre, normalised coordinates
    affine = torch.zeros(b, 2, 3, device=dev)
    affine[:, 0, 0], affine[:, 0, 1], affine[:, 0, 2] = 0.95 * theta.cos(), -0.95 * theta.sin(), shift[:, 0]
    affine[:, 1, 0], affine[:, 1, 1], affine[:, 1, 2] = 0.95 * theta.sin(), 0.95 * theta.cos(), shift[:, 1]
    grid = F.affine_grid(affine, list(out[:, 0].shape), align_corners=False)
    out[:, 0] = F.grid_sample(out[:, 0], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    flat = out.reshape(-1, *out.shape[2:])
    return _color_jitter(flat).clamp(0, 1).reshape(out.shape)


def raw_batch(images, states, instructions, actions=None, is_pad=None, device="cuda", augment_images=False):
    """Policy input from uint8 images (B, 2, H, W, 3), normalised states (B, 8) and actions, and instructions."""
    images = images.to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255.0)
    if augment_images:
        images = augment(images)
    batch = {key: images[:, i] for i, key in enumerate(IMAGE_KEYS)}
    batch[STATE_KEY] = states.to(device).float()
    batch["task"] = list(instructions)
    if actions is not None:
        batch[ACTION_KEY] = actions.to(device)
        batch["action_is_pad"] = is_pad.to(device)
    return batch


def expert_loss(policy, batch):
    """pi0's training loss (LeRobot's PI0Policy.forward) with the vision-language prefix run once without gradients and
    passed to the action expert as a key/value cache, as pi0's own action sampling does. The prefix never attends to
    the action tokens, so the loss and the action expert's gradients are those of the joint forward; the frozen
    vision-language model is just not backpropagated through. For `train` = "expert" only."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    lang_tokens, lang_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    noise = model.sample_noise(actions.shape, actions.device)
    time = model.sample_time(actions.shape[0], actions.device)
    t = time[:, None, None]
    x_t = t * noise + (1 - t) * actions
    u_t = noise - actions
    vlm_bf16 = model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype \
        == torch.bfloat16
    with torch.no_grad():
        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if vlm_bf16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        _, cache = model.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(make_att_2d_masks(prefix_pad, prefix_att)),
            position_ids=torch.cumsum(prefix_pad, dim=1) - 1, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True)
    suffix_embs, suffix_pad, suffix_att, adarms_cond = model.embed_suffix(state, x_t, time)
    if vlm_bf16:
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
    b, n_suffix, n_prefix = suffix_pad.shape[0], suffix_pad.shape[1], prefix_pad.shape[1]
    masks = torch.cat([prefix_pad[:, None, :].expand(b, n_suffix, n_prefix), make_att_2d_masks(suffix_pad, suffix_att)],
                      dim=2)
    position_ids = torch.sum(prefix_pad, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
    (_, suffix_out), _ = model.paligemma_with_expert.forward(
        attention_mask=prepare_attention_masks_4d(masks), position_ids=position_ids, past_key_values=cache,
        inputs_embeds=[None, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond])
    v_t = model.action_out_proj(suffix_out[:, -policy.config.chunk_size:].to(dtype=torch.float32))
    losses = F.mse_loss(u_t, v_t, reduction="none")[:, :, :policy.config.output_features[ACTION_KEY].shape[0]]
    return losses.mean()


def trainable_count(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)
