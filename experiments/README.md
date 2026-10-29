# Reproducing the experiments

Every run behind the reported results is one line in a job file here, with the exact arguments it was
run with. A line reads `<name>|<prerequisite>|<arguments for python>`: the job starts once the
prerequisite path exists (an output of an earlier job). [run_jobs.sh](run_jobs.sh) runs a file, at most
`MAX` jobs at a time, and can be stopped and restarted (started jobs are skipped). For example, the third
task of the reported 5% configuration, seed 0:

```text
s0_C_gate_rand_k5_c2000|$CVLA_RUNS/final/s0/B_core_rand_k5_c2000/stage.json|training/train_stage.py
    --method consolidate_gate --task StackCubeGreenOnRed-v1 --n-train 900 --steps 12000 --eval-every 100000
    --final-tasks PushCube-v1 StackCube-v1 StackCubeGreenOnRed-v1 --final-split test --final-episodes 100
    --freeze-core --init $CVLA_RUNS/final/s0/B_core_rand_k5_c2000 --seed 0 --out $CVLA_RUNS/final/s0/C_gate_rand_k5_c2000
```

(one line in the file). A job's name is its seed and output stage, plus what it does when it only evaluates or
scores a stage; its output goes to `$CVLA_RUNS/final/s<seed>/<stage>/`.

| Job file | Reproduces | Jobs | Time on one RTX 5090 (two jobs at once) |
|---|---|---|---|
| [01_main_study.jobs](01_main_study.jobs) | Two tasks, three seeds: fine-tuning, ER-50, EWC (λ chosen on seed 0), consolidate-and-gate with 25% cores; importance-selected cores; consolidation curves; gate-off evaluations; StackCube-only reference | 49 | about 11 h |
| [02_ablations.jobs](02_ablations.jobs) | Gating without freezing; ER-500; joint training (two tasks); selected vs random core through StackCube; core size 10/25/50%; per-step traces | 36 | about 10 h |
| [03_three_task.jobs](03_three_task.jobs) | Feasibility checks; nested consolidation of StackCube; the third task (green cube on red cube) for fine-tuning, ER and consolidate-and-gate with 25% cores | 28 | about 11 h |
| [04_exec_steps.jobs](04_exec_steps.jobs) | Executed actions per predicted chunk (1-50), seed 0 | 15 | about 2 h |
| [05_core_size.jobs](05_core_size.jobs) | Core size 10%, 5% and 50% with 500 consolidation updates; two 50% cores filling the network (seed 0) | 43 | about 14 h |
| [06_packnet.jobs](06_packnet.jobs) | PackNet at 25% per task, matched to the gate's 500 + 6000 consolidation updates | 12 | about 5 h |
| [07_small_cores.jobs](07_small_cores.jobs) | **The reported configuration: 5% cores** (2000 PushCube and 24000 StackCube consolidation updates); 2.5% and 1% cores; a 5% PushCube core with a 25% StackCube core; PushCube on 0.5-2.5% cores | 48 | about 27 h |
| [08_attention_cores.jobs](08_attention_cores.jobs) | Attention-aware 25% cores, 24000 StackCube consolidation updates, task-3 attention branches; interface adapter; hardware repeat; cut-out sub-networks against PackNet; repeated test evaluations | 47 | about 21 h |
| [09_extra_updates.jobs](09_extra_updates.jobs) | Fine-tuning and replay with +6000 or +24000 StackCube updates before task 3 | 36 | about 19 h |
| [10_packnet_budgets.jobs](10_packnet_budgets.jobs) | PackNet with 24000 StackCube retraining updates (25%); task 3 unfrozen; 10% and 5% with 500 updates; **5% and 2.5% matched to the gate** and its two-task scores | 58 | about 27 h |
| [11_over_capacity.jobs](11_over_capacity.jobs) | Beyond capacity: two 50% shares, then task 3, with nothing or half of each earlier share released, for the gate and PackNet | 30 | about 9 h |
| [12_ten_percent.jobs](12_ten_percent.jobs) | 10% per task for both methods with the 5% recipe; 10% cores with 24000 StackCube consolidation updates | 33 | about 23 h |
| [13_joint_three_task.jobs](13_joint_three_task.jobs) | Joint training on all three tasks at once (reference) | 3 | about 4 h |
| [14_rollout_videos.jobs](14_rollout_videos.jobs) | Every test episode of seed 0 recorded, for the rollout videos on the project page | 7 | about 1 h |
| [15_five_percent_analyses.jobs](15_five_percent_analyses.jobs) | The analyses of the reported 5% configuration: the core not frozen, a core selected by intervention, per-step traces, the stage-B learning curve and its first 2000 updates scored densely for every method, executed actions per chunk | 35 | about 12 h |

Files 02-15 start from checkpoints written by 01 (task 1, its importance scores and the replay buffers) and some
from each other: run them in this order. 03 and later need the green-on-red demonstrations below. Every line was
checked against the run it reproduces: parsed by the current `training/train_stage.py`, it gives back the arguments
recorded in that run's `config.json` (runs recorded before the methods were split into one file each name
consolidate-and-gate and joint training `--method seqft`, and also record four options, unused by any reported run,
that have since been removed).

## 1. Environment and data

Linux with an NVIDIA GPU and a Vulkan-capable driver; all large files go under `$CVLA_ROOT`.

```bash
export CVLA_ROOT=/path/to/large/disk          # Python environment, caches, data and runs
PYTHON=python3.12 bash scripts/setup_env.sh   # INSTALL_APT=1 / HEADLESS_VULKAN=1 if needed, see the script
source scripts/env.sh
python scripts/verify_install.py              # GPU, renderer and task configuration
python scripts/download_assets.py             # SmolVLA weights; ManiSkill PushCube/StackCube demonstrations
python scripts/replay_demos.py                # re-render them with this project's camera and controller
```

The reversed stacking task needs its own demonstrations, made with ManiSkill's motion planner in a small
separate environment (the planner needs NumPy < 2):

```bash
bash scripts/setup_planner_env.sh
PYTHONPATH=src $CVLA_ROOT/envs/mp/bin/python scripts/generate_demos.py --task StackCubeGreenOnRed-v1 --n 1000
python scripts/replay_demos.py --tasks StackCubeGreenOnRed-v1
```

## 2. Runs

```bash
source scripts/env.sh
MAX=2 bash experiments/run_jobs.sh experiments/01_main_study.jobs
MAX=2 bash experiments/run_jobs.sh experiments/02_ablations.jobs
MAX=2 bash experiments/run_jobs.sh experiments/03_three_task.jobs
MAX=2 bash experiments/run_jobs.sh experiments/04_exec_steps.jobs
MAX=2 bash experiments/run_jobs.sh experiments/05_core_size.jobs
# ... and so on through 15_five_percent_analyses.jobs, in order
```

Outputs go to `$CVLA_RUNS/final/s<seed>/<stage>/` (prototype checks to `$CVLA_RUNS/proto/`), logs to
`$CVLA_RUNS/jobs/<job file>/logs/`. Each stage directory holds its arguments (`config.json`), data
splits, weights, gate (`core.json`), frozen-parameter counts (`freeze.json`), every evaluation
(`evals*.jsonl`) and rollout videos. Use a fresh `$CVLA_RUNS`: training appends to existing logs.

## 3. Results and the project page

```bash
python scripts/export_results.py --root $CVLA_RUNS/final --out results.json
python evaluation/analyze_traces.py --root $CVLA_RUNS/final          # per-step success and state traces
python scripts/build_page.py --results results.json --runs $CVLA_RUNS/final --data $CVLA_DATA \
    --repo-url https://github.com/DishengTang/continual_vla --out docs/index.html
```

Evaluation uses fixed reset seeds and a fixed noise stream, so re-evaluating a saved stage reproduces its
numbers up to GPU nondeterminism (in our repeats: exactly, or within one episode in a hundred).
