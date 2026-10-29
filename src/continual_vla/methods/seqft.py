"""Sequential fine-tuning: each task is learned on its own demonstrations only, with nothing to protect the earlier
tasks (the lower reference for forgetting). It is the shared pipeline with no change: every hook keeps its default
from base.py, and each other method is this plus the hooks it overrides."""

from .base import Method


class FineTuning(Method):
    pass
