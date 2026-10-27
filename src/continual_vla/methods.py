"""Continual-learning methods: parameter-importance estimates and the consolidation penalty."""

import torch


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
