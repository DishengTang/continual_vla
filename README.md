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
PushCube, then StackCube (red cube on green), then the opposite stack in the same scenes. Success rate (%) after all three tasks on 100 test scenes per task, mean ± SD over 3 training seeds. Consolidate-and-gate and PackNet reserve 5% of the network per task; each method is shown at its best three-task configuration, and replay stores examples of each earlier task:

| Method | Configuration | Pushing | Red on green | … built green on red instead | Green on red (new) | Average |
|---|---|---|---|---|---|---|
| Fine-tuning | +24000 stacking updates | 1.7 ± 1.2 | 0.0 ± 0.0 | 63.0 ± 1.7 | 64.7 ± 1.2 | 22.1 |
| EWC | as in the two-task study | 6.7 ± 4.7 | 1.3 ± 1.2 | 14.0 ± 2.6 | 13.3 ± 4.6 | 7.1 |
| Experience replay, 50 + 50 examples | +6000 stacking updates | 80.3 ± 5.0 | 3.0 ± 2.0 | 14.0 ± 14.8 | 60.0 ± 2.0 | 47.8 |
| Experience replay, 500 + 500 examples | +24000 stacking updates | 92.3 ± 3.1 | 28.3 ± 3.1 | 3.7 ± 1.5 | 58.7 ± 3.1 | 59.8 |
| PackNet | 5% per task | 85.3 ± 2.3 | 45.3 ± 3.2 | 0.0 ± 0.0 | 53.7 ± 9.1 | 61.4 |
| **Consolidate-and-gate** | 5% cores | 93.3 ± 3.8 | 43.3 ± 4.2 | 0.0 ± 0.0 | 63.0 ± 7.0 | 66.6 |
| *Joint training (reference: all data at once)* | all three tasks mixed in every batch, 32500 updates | 99.0 ± 0.0 | 48.3 ± 4.0 | 1.7 ± 0.6 | 49.0 ± 6.1 | 65.4 |
<!-- results:end -->

Consolidate-and-gate and PackNet keep the second skill (frozen, so unchanged by the third task); fine-tuning
ignores the instruction and builds the newest stack. Brain-inspired, instruction-dependent consolidation preserves the
earlier skills; under the small allocations explored here (5% per task and below) it gives a better trade-off between
performance and task bookkeeping than PackNet, while PackNet remains stronger at larger allocations (10% and 25%). The
gate's task records take kilobytes instead of a per-weight map. The [project page](docs/index.html) adds the two-task
results, the comparison across allocations, analyses of why the method works, rollout videos and hardware details.

## How it works

The action expert is divided into 752 unit groups (in each of 16 layers, 15 attention heads and 32 blocks
of 64 MLP units). After a task is learned:

1. **Consolidate**: choose 5% of the groups (38) at random as the task's core and train the task for 2000 more
   updates (24000 for stacking) with the other groups silenced, so the skill is re-expressed in the core.
2. **Gate**: whenever the policy receives that task's instruction, silence the non-core groups.
3. **Freeze**: while later tasks are learned, keep every weight on the task's gated path fixed; only
   groups the task never uses learn.

The earlier task's gated computation then uses only frozen weights, so it is preserved exactly. The cost is
plasticity: later tasks are learned with fewer free units. The mechanism follows context-dependent gating
with synaptic stabilisation ([Masse, Grant & Freedman, 2018](https://doi.org/10.1073/pnas.1803839115));
in machine-learning terms it is related to PackNet and HAT, applied to whole heads and MLP blocks and keyed
by the language instruction. Implementation: [consolidate_gate.py](src/continual_vla/methods/consolidate_gate.py),
trained with `--method consolidate_gate` by the pipeline every method shares, [train_stage.py](training/train_stage.py).

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
├── src/continual_vla/     library: tasks and the custom task, observation filter, policy, demonstrations,
│   │                      unit groups and their importance, closed-loop evaluation, loading a trained stage
│   └── methods/           one file per continual-learning method, chosen with --method: consolidate_gate.py
│                          (this project's method), seqft.py, er.py, ewc.py, packnet.py, joint.py, and the
│                          hooks they implement (base.py)
├── training/              train_stage.py, the training pipeline every method shares; importance scores and
│                          replay buffers of finished stages
├── evaluation/            closed-loop evaluation of saved stages, per-step traces, cut-out sub-networks
├── scripts/               environment setup, data preparation, results export and the project page
├── experiments/           the exact arguments of every reported run, and a runner
└── docs/                  project page (index.html, one self-contained file) and guides
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
