"""Continual-learning methods, one file each, chosen with training/train_stage.py --method (hooks: base.py).

  seqft             sequential fine-tuning (seqft.py)
  er                experience replay (er.py)
  ewc               elastic weight consolidation (ewc.py)
  packnet           PackNet (packnet.py)
  joint             joint training on several tasks at once, the non-sequential reference (joint.py)
  consolidate_gate  consolidate-and-gate, this project's method (consolidate_gate.py)
"""

from .base import Method
from .consolidate_gate import ConsolidateGate
from .er import ExperienceReplay
from .ewc import EWC
from .joint import JointTraining
from .packnet import PackNetMethod
from .seqft import FineTuning

# The order in which every stage writes the files of save_for_later_stages (the replay buffer, then the Fisher).
METHODS = {"seqft": FineTuning, "er": ExperienceReplay, "ewc": EWC, "packnet": PackNetMethod, "joint": JointTraining,
           "consolidate_gate": ConsolidateGate}

__all__ = ["METHODS", "Method"]
