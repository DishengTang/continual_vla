"""Inference cost of each task's gated sub-network cut out as a narrower action expert (slice_expert in continual_vla/methods/consolidate_gate.py),
against the full-size network that PackNet runs for every task (and that the gate runs before cutting).

    python evaluation/benchmark_sliced.py --stage $CVLA_RUNS/final/s0/C_gate_bundle_L24k_brC \
        --packnet $CVLA_RUNS/final/s0/C_packnet_L24k --rollouts 100

For every task of the stage: the narrow network's actions against the gated network's on the same inputs and noise
(fp32, and bf16 autocast as in evaluation), action-chunk latency at each --batch (median of --repeats after warm-up,
bf16 autocast), split into the VLM prefix and the denoising steps, the share of the action expert's weights used, and
with --rollouts the narrow network's closed-loop test success next to the gated network's recorded episodes.
Writes <stage>/sliced_benchmark.json.
"""

import argparse
import json
import os
import statistics
import time

import torch


def main():
    from lerobot.policies.smolvla import modeling_smolvla as ms
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    from continual_vla.importance import expert_groups
    from continual_vla.policy import REST_QPOS, raw_batch
    from continual_vla.methods.consolidate_gate import expert_size, slice_expert
    from continual_vla.stages import load_stage
    from continual_vla.tasks import IMAGE_SIZE, SEQUENCE, SPLIT_SEEDS, TASKS

    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--packnet", default=None, help="a PackNet stage to time at full size for the same tasks")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 10])
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--rollouts", type=int, default=0, help="test episodes per task for the narrow networks")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = args.device
    cuda = dev.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    ref, pre, post, gate, core = load_stage(args.stage, gate=True, device=dev)
    groups = expert_groups(ref, core.get("group_layout", {}).get("mlp_block", 64))
    silenced = {e["instruction"]: e["silenced"] for e in core.get("earlier", []) + [core]}
    instruction = {TASKS[t].instruction: t for t in SEQUENCE}
    full_size = expert_size(ref)
    narrow = {}
    for ins, off in silenced.items():
        policy = load_stage(args.stage, gate=False, device=dev)[0]
        narrow[instruction[ins]] = slice_expert(policy, groups, off)
    packnet = load_stage(args.packnet, gate=True, device=dev) if args.packnet else None

    def inputs(b, ins):
        g = torch.Generator().manual_seed(0)
        images = torch.randint(0, 256, (b, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8, generator=g)
        states = REST_QPOS + 0.05 * torch.randn(b, len(REST_QPOS), generator=g)
        return pre(raw_batch(images, states, [ins] * b, device=dev))

    def noise(b):
        cfg = ref.config
        return torch.randn(b, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(1)).to(dev)

    @torch.inference_mode()
    def chunk(policy, context, batch, eps, bf16=True):
        if context is not None:
            context([ins] * eps.shape[0])
        policy.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and cuda):
            return policy.predict_action_chunk(batch, noise=eps).float()

    @torch.inference_mode()
    def prefix(policy, batch):
        m = policy.model
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            embs, pad, att = m.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                            batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
            m.vlm_with_expert.forward(attention_mask=ms.make_att_2d_masks(pad, att),
                                      position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
                                      inputs_embeds=[embs, None], use_cache=True)

    def ms_median(fn):
        times = []
        for r in range(args.repeats + 5):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            if r >= 5:
                times.append(1000 * (time.perf_counter() - t0))
        return statistics.median(times)

    results = {"stage": args.stage, "packnet": args.packnet, "device": torch.cuda.get_device_name() if cuda else dev,
               "repeats": args.repeats, "expert_weights": full_size, "tasks": {}}
    for task in SEQUENCE:
        ins = TASKS[task].instruction
        net = narrow.get(task)
        row = {"instruction": ins, "weights_used": (expert_size(net) if net is not None else full_size) / full_size}
        if net is not None:
            for name, bf16 in (("fp32", False), ("bf16", True)):
                b = max(args.batch)
                batch, eps = inputs(b, ins), noise(b)
                a, c = chunk(ref, gate, batch, eps, bf16), chunk(net, None, batch, eps, bf16)
                row[f"max_abs_diff_{name}"] = float((a - c).abs().max())
                row[f"max_abs_action_{name}"] = float(a.abs().max())
        for b in args.batch:
            batch, eps = inputs(b, ins), noise(b)
            timing = {"gated_full": ms_median(lambda: chunk(ref, gate, batch, eps)),
                      "prefix": ms_median(lambda: prefix(ref, batch))}
            if net is not None:
                timing["narrow"] = ms_median(lambda: chunk(net, None, batch, eps))
            if packnet is not None:
                timing["packnet"] = ms_median(lambda: chunk(packnet[0], packnet[3], batch, eps))
            row[f"ms_batch{b}"] = timing
        if net is not None and args.rollouts:
            from continual_vla.evaluation import evaluate
            res = evaluate(net, pre, post, task, SPLIT_SEEDS["test"][:args.rollouts], n_envs=5, exec_steps=5,
                           device=dev)
            row["narrow_success"] = res["success"]
            path = os.path.join(args.stage, "evals_test_exec5.jsonl")
            if os.path.exists(path):
                rec = [r for r in map(json.loads, open(path)) if r["task"] == task and r["n"] == args.rollouts]
                if rec:
                    row["gated_success"] = rec[-1]["success"]
                    row["episodes_agree"] = sum(x == y for x, y in zip(res["episodes"], rec[-1]["episodes"]))
        results["tasks"][task] = row
        print(task, json.dumps(row), flush=True)
    out = args.out or os.path.join(args.stage, "sliced_benchmark.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
