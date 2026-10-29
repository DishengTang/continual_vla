"""Joint training, the non-sequential reference: every batch mixes the current task's demonstrations with the full
training splits of the --joint-task tasks, each with its --joint-frac of the batch, from the pretrained policy."""

from ..data import DemoSet, split_episodes
from ..tasks import TASKS
from .base import Method


class JointTraining(Method):
    @staticmethod
    def add_args(group):
        group.add_argument("--joint-task", nargs="+", default=None, choices=list(TASKS),
                           help="also train on these tasks' full training splits, each with its --joint-frac of every batch")
        group.add_argument("--joint-frac", nargs="+", type=float, default=None)
        group.add_argument("--joint-n-train", nargs="+", type=int, default=200,
                           help="training demonstrations of each --joint-task (one value: all of them)")

    def earlier_tasks(self, tasks, splits):
        args = self.stage.args
        assert not args.init and args.joint_task and len(args.joint_frac or []) == len(args.joint_task), \
            "joint training starts from the pretrained policy and needs one --joint-frac per --joint-task"
        n_train = args.joint_n_train if isinstance(args.joint_n_train, list) else [args.joint_n_train]
        n_train = n_train * len(args.joint_task) if len(n_train) == 1 else n_train
        for task, n in zip(args.joint_task, n_train):
            train_ids, val_ids = split_episodes(task, n, args.n_val)
            splits[task] = {"train": train_ids, "val": val_ids}
        return list(args.joint_task)

    def setup(self):
        args, stage = self.stage.args, self.stage
        joint = [DemoSet(task, stage.splits[task]["train"], stage.chunk) for task in args.joint_task]
        for ds, task, frac in zip(joint, args.joint_task, args.joint_frac):
            print(f"joint training: {len(ds)} {task} examples, {round(args.batch * frac)} of every {args.batch}", flush=True)
        self.extra = [(ds, round(args.batch * frac)) for ds, frac in zip(joint, args.joint_frac)]

    def extra_examples(self):
        return self.extra
