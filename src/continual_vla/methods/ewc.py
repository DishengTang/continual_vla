"""Elastic weight consolidation (Kirkpatrick et al., 2017): a quadratic penalty keeps the parameters near their values
after the earlier tasks, weighted by a diagonal importance estimate.

Every stage, whatever its method, writes fisher.safetensors at its end (the importance of its own task); a later EWC
stage adds (--ewc-lambda / 2) x sum_i F_i (theta_i - theta*_i)^2 to its loss, with theta* the --init weights. The same
Fisher-weighted drift is logged for every method whose --init has a Fisher estimate.
"""

import os

import torch
from safetensors.torch import save_file

from .base import Method


def fisher_diagonal(params, example_loss_fn, n_examples):
    """Diagonal parameter importance: the mean over examples of each example's squared loss gradient
    (one example per backward pass, so gradients do not cancel within a minibatch), normalised to
    mean 1. `example_loss_fn()` returns the flow-matching loss of one random training example, so
    this is an empirical-Fisher-style surrogate, not the exact Fisher of a likelihood."""
    fisher = {n: torch.zeros_like(p) for n, p in params.items()}
    for _ in range(n_examples):
        for p in params.values():
            p.grad = None
        example_loss_fn().backward()
        for n, p in params.items():
            if p.grad is not None:
                fisher[n] += p.grad.detach().float() ** 2
    for p in params.values():
        p.grad = None
    mean = sum(f.sum() for f in fisher.values()) / sum(f.numel() for f in fisher.values())
    return {n: f / mean for n, f in fisher.items()}


class Importance:
    """Importance-weighted drift from the parameters at the end of the previous task:
    drift = sum_i F_i (theta_i - theta*_i)^2, with F normalised to mean 1.
    EWC adds (lambda / 2) * drift to the loss."""

    def __init__(self, fisher, anchor, device="cuda"):
        self.fisher = {n: f.to(device) for n, f in fisher.items()}
        self.anchor = {n: a.to(device).float() for n, a in anchor.items()}

    def drift(self, params):
        total = 0.0
        for n, p in params.items():
            total = total + (self.fisher[n] * (p - self.anchor[n]) ** 2).sum()
        return total


class EWC(Method):
    @staticmethod
    def add_args(group):
        group.add_argument("--ewc-lambda", type=float, default=1.0, help="weight of the penalty")

    @staticmethod
    def save_for_later_stages(stage):
        stage.policy.train()
        fisher = fisher_diagonal(stage.params, lambda: stage.loss_on(stage.make_batch(stage.data, size=1)),
                                 stage.args.fisher_samples)
        save_file({n: f.cpu().contiguous() for n, f in fisher.items()}, os.path.join(stage.args.out, "fisher.safetensors"))

    def setup(self):
        assert self.stage.importance is not None, "ewc needs fisher.safetensors in --init"

    def loss(self, step):
        loss, _, logged = super().loss(step)
        weighted = self.stage.importance.drift(self.stage.params)
        return loss, loss + 0.5 * self.stage.args.ewc_lambda * weighted, {**logged, "drift": weighted}
