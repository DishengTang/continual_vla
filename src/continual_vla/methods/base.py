"""What a continual-learning method changes in a training stage.

training/train_stage.py runs one pipeline for every method: it draws batches of the current task's demonstrations,
computes the flow-matching loss, takes an AdamW step, evaluates in closed loop, writes resumable checkpoints and saves
the stage. A method is a subclass of Method, registered in methods/__init__.py, that changes this pipeline through the
hooks below; the defaults are plain fine-tuning. A method reaches the running stage through `self.stage`: `args`,
`policy`, `params` (the trainable parameters by name), `data` (the current task's training demonstrations),
`splits`, `chunk`, `importance` (Fisher-weighted drift from --init, or None), `make_batch(dataset, size, extra,
instruction)` and `loss_on(batch)`.
"""


class Method:
    @staticmethod
    def add_args(group):
        """Command-line options of this method (its argument group of train_stage.py)."""

    @staticmethod
    def save_for_later_stages(stage):
        """Files that every stage writes at its end, whatever its method, for a later stage of this method."""

    def __init__(self, stage):
        self.stage = stage
        self.context_fn = None  # called with the instructions of every forward pass, in training and evaluation
        self.frozen = None  # {parameter name: bool mask}: entries that must not change (their gradient is zeroed)
        self.reference = None  # values the frozen entries must keep (default: the --init checkpoint), checked at the end
        self.extra_params = {}  # parameters trained besides the policy's: optimised, clipped and checkpointed with them
        self.gated = False  # a context gate is installed (recorded with every evaluation)

    def earlier_tasks(self, tasks, splits):
        """Tasks this stage keeps evaluating besides its own: those learned before it (from --init) by default."""
        return tasks

    def setup(self):
        """Called once the policy, with the --init weights, and the current task's data are loaded."""

    def extra_examples(self):
        """[(dataset, examples per batch)]: examples of other tasks mixed into every training batch."""
        return []

    def loss(self, step):
        """(flow-matching loss of update `step`'s training batch, total loss to minimise, {log key: value})."""
        batch = self.stage.make_batch(self.stage.data, extra=self.extra_examples())
        loss = self.stage.loss_on(batch)
        return loss, loss, {}

    def before_checkpoint(self):
        """Called before a resumable checkpoint is written."""

    def finish(self):
        """Called after the last update, before the weights are saved: write this method's own files."""


def module_family(name):
    """Coarse module family of an action-expert parameter, for frozen/plastic accounting."""
    for key, family in (("q_proj", "attention q/o"), ("o_proj", "attention q/o"), ("k_proj", "attention k/v"),
                        ("v_proj", "attention k/v"), ("mlp.", "mlp"), ("norm", "norms")):
        if key in name:
            return family
    if any(k in name for k in ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp")):
        return "state/action/time projections"
    return "other"


def family_counts(params, frozen):
    """{family: {"frozen": n, "plastic": n}} for the parameters in `params` under the `frozen` masks."""
    out = {}
    for n, p in params.items():
        c = out.setdefault(module_family(n), {"frozen": 0, "plastic": 0})
        f = int(frozen[n].sum()) if n in frozen else 0
        c["frozen"] += f
        c["plastic"] += p.numel() - f
    return out
