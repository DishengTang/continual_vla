# Consolidate, then Gate

**Continual learning of manipulation skills in a vision-language-action policy.** A pretrained
[SmolVLA](https://arxiv.org/abs/2506.01844) policy learns robot manipulation tasks one after another in
[ManiSkill 3](https://www.maniskill.ai/), from a camera image, its joint angles and a language
instruction. Re-expressing each learned skill in a small core of the action expert, switching the other
units off whenever that skill's instruction is given, and freezing the core keeps every earlier skill
intact while later skills are learned, without storing any old data.

**[Project page](docs/index.html)** · [Reproduce](experiments/README.md) · [Data](docs/data.md)

## Results

<!-- results:start -->
PushCube, then StackCube. Success rate (%) on 100 test scenes per task, mean ± SD over the training seeds (3 unless stated). Consolidate-and-gate and PackNet reserve 5% of the network per task:

| Method | Pushing after learning to stack | Stacking | Average | Stored for the old task |
|---|---|---|---|---|
| Fine-tuning | 1.0 ± 1.0 | 54.7 ± 4.0 | 27.8 | none |
| EWC | 1.3 ± 0.6 | 33.3 ± 1.2 | 17.3 | Fisher + old weights (0.8 GB) |
| Experience replay, 50 examples | 85.7 ± 10.5 | 50.3 ± 3.1 | 68.0 | 9.4 MiB |
| Experience replay, 500 examples | 92.3 ± 4.0 | 52.7 ± 2.5 | 72.5 | 94 MiB |
| PackNet, 5% per task | 85.3 ± 2.3 | 38.3 ± 4.6 | 61.8 | an owner per weight |
| **Consolidate-and-gate, 5% cores** | **93.3 ± 3.8** | 51.0 ± 2.6 | 72.2 | a unit mask per task |
| *Joint training (all data at once)* | 98.0 ± 1.0 | 61.0 ± 3.6 | 79.5 | all PushCube data |

Pushing was 85.3–95.7% before StackCube was learned. A third task then asks for the
opposite stack in the same scenes. Each method is shown at its best three-task configuration on the test scenes, the
gate and PackNet at 5% per task (room for twenty tasks); all configurations, including a quarter of the network per
task, are on the project page. Replay stores examples of each earlier task:

| Method | Configuration | Pushing | Red on green | … built green on red instead | Green on red (new) | Average |
|---|---|---|---|---|---|---|
| Fine-tuning | +24000 stacking updates | 1.7 ± 1.2 | 0.0 ± 0.0 | 63.0 ± 1.7 | 64.7 ± 1.2 | 22.1 |
| Experience replay, 50 + 50 examples | +6000 stacking updates | 80.3 ± 5.0 | 3.0 ± 2.0 | 14.0 ± 14.8 | 60.0 ± 2.0 | 47.8 |
| Experience replay, 500 + 500 examples | +24000 stacking updates | 92.3 ± 3.1 | 28.3 ± 3.1 | 3.7 ± 1.5 | 58.7 ± 3.1 | 59.8 |
| PackNet | 5% per task | 85.3 ± 2.3 | 45.3 ± 3.2 | 0.0 ± 0.0 | 53.7 ± 9.1 | 61.4 |
| **Consolidate-and-gate** | 5% cores | 93.3 ± 3.8 | 43.3 ± 4.2 | 0.0 ± 0.0 | 63.0 ± 7.0 | 66.6 |

Per-task budget against PackNet. Pushing is scored right after its own stage; both methods freeze it afterwards:

| Configuration | Weights per task | Pushing, seeds 0 / 1 / 2 | Mean | Average after three tasks |
|---|---|---|---|---|
| Consolidate-and-gate, attention-aware cores, 500 consolidation updates | 25% | 94 / 99 / 97 | 96.7 ± 2.5 | 72.3 ± 1.5 |
| PackNet, 500 retraining updates | 25% | 94 / 100 / 100 | 98.0 ± 3.5 | 75.4 ± 1.2 |
| Consolidate-and-gate, random cores, 2000 consolidation updates | 5% | 95 / 96 / 89 | 93.3 ± 3.8 | 66.6 ± 2.0 |
| PackNet, 2000 retraining updates, state/action/time projections shared (matched) | 5% | 88 / 84 / 84 | 85.3 ± 2.3 | 61.4 ± 4.9 |
| PackNet, 2000 retraining updates, projections pruned too | 5% | 45 / 44 / 44 | 44.3 ± 0.6 | — |
| PackNet, 500 retraining updates, projections pruned too | 5% | 21 (seed 0 only) | 21.0 | 45.3 |
| Consolidate-and-gate, random cores, 2000 consolidation updates | 2.5% | 93 / 93 / 90 | 92.0 ± 1.7 | 58.9 ± 3.0 |
| PackNet, 2000 retraining updates, state/action/time projections shared (matched) | 2.5% | 36 / 37 / 37 | 36.7 ± 0.6 | 40.7 ± 2.7 |
<!-- results:end -->

Consolidate-and-gate and PackNet keep the second skill (frozen, so unchanged by the third task); fine-tuning
ignores the instruction and builds the newest stack. At 5% of the network per task, room for twenty tasks, the gate
is ahead of PackNet on every seed; at a quarter per task, room for only four, PackNet is ahead. With attention-aware
cores each task's skill can be cut out as a smaller network that acts the same
([sliced.py](src/continual_vla/sliced.py), [benchmark_sliced.py](scripts/benchmark_sliced.py)), and the gate's task
records take kilobytes instead of a per-weight map. The [project page](docs/index.html) adds all configurations,
ablations, per-step analyses and rollout videos.

## How it works

The action expert is divided into 752 unit groups (in each of 16 layers, 15 attention heads and 32 blocks
of 64 MLP units). After a task is learned:

1. **Consolidate**: choose 5% of the groups (38) at random as the task's core and train the task for 2000 more
   updates (24000 for stacking) with the other groups silenced, so the skill is re-expressed in the core.
2. **Gate**: whenever the policy receives that task's instruction, silence the non-core groups.
3. **Freeze**: while later tasks are learned, keep every weight on the task's gated path fixed; only
   groups the task never uses learn.

With a quarter of the network per task (room for four tasks) the best configuration takes whole key/value bundles (a
key/value head with the query heads that share it) and a quarter of the MLP blocks in every layer, and gives the third
task a rank-16 attention branch keyed by its instruction ([branches.py](src/continual_vla/branches.py)), which makes no
measurable difference.

The earlier task's gated computation then uses only frozen weights, so it is preserved exactly. The cost is
plasticity: later tasks are learned with fewer free units. The mechanism follows context-dependent gating
with synaptic stabilisation ([Masse, Grant & Freedman, 2018](https://doi.org/10.1073/pnas.1803839115));
in machine-learning terms it is related to PackNet and HAT, applied to whole heads and MLP blocks and keyed
by the language instruction. Implementation: [gating.py](src/continual_vla/gating.py),
[importance.py](src/continual_vla/importance.py) and the core options of [train_stage.py](scripts/train_stage.py).

## Setup

| | |
|---|---|
| Simulator | ManiSkill 3.0.1 (SAPIEN 3.0.3), Franka Panda, end-effector delta-position control at 20 Hz |
| Tasks | `PushCube-v1` → `StackCube-v1` → `StackCubeGreenOnRed-v1` (this repository: the opposite stack in the same scene) |
| Policy input | one 256 × 256 RGB image from a fixed camera, 9 joint positions, the instruction |
| Policy output | 50 future actions (hand displacement x, y, z and gripper); the first 5 are executed, then it observes again |
| Model | SmolVLA (0.45B parameters); the vision-language backbone is frozen, the 100M-parameter action expert is fine-tuned |
| Data | ManiSkill motion-planning demonstrations re-rendered for this setup: 200 (push), 900 (each stacking task) training episodes; see [docs/data.md](docs/data.md) |
| Training | batch 32, AdamW, learning rate 1e-4 with warmup and cosine decay; 8000 updates (push) and 12000 (each stacking task); consolidation 2000 (push) and 24000 (stacking), 500 ordinary push updates for fine-tuning, EWC and replay |
| Evaluation | 100 fresh test scenes per task (seeds 200000+); success at any step and at the final step; 3 training seeds |

The policy never receives object or target positions, rewards or success signals; only
[observations.py](src/continual_vla/observations.py) passes simulator data to it.

## Repository layout

```
├── src/continual_vla/   library: tasks and custom task, observation filter, policy, data, gating,
│                        importance, continual-learning methods, closed-loop evaluation
├── scripts/             command-line entry points: environment setup, data, training (train_stage.py),
│                        evaluation, analysis, results export and the project page (build_page.py)
├── experiments/         the exact arguments of every reported run, and a runner
└── docs/                project page (index.html, one self-contained file) and guides
```

## Installation and reproduction

Linux with an NVIDIA GPU (tested: Ubuntu 22.04, RTX 5090, driver 580, CUDA 13.0, torch 2.11, LeRobot 0.6.1;
see `requirements.lock.txt`). All large files go under `$CVLA_ROOT`.

```bash
export CVLA_ROOT=/path/to/large/disk
PYTHON=python3.12 bash scripts/setup_env.sh && source scripts/env.sh
python scripts/verify_install.py && python scripts/download_assets.py && python scripts/replay_demos.py
MAX=2 bash experiments/run_jobs.sh experiments/01_main_study.jobs
```

Every run behind a reported number is one line in [experiments/](experiments/), checked against the arguments
that run recorded. [experiments/README.md](experiments/README.md) lists every job file, what it reproduces, its
running time and how to rebuild the results and the project page.

## References

SmolVLA (Shukor et al., 2025); ManiSkill 3 (Tao et al., 2025); context-dependent gating (Masse et al., 2018);
EWC (Kirkpatrick et al., 2017); experience replay (Rolnick et al., 2019); PackNet (Mallya & Lazebnik, 2018);
HAT (Serrà et al., 2018). Model weights and demonstrations keep their upstream licenses.
