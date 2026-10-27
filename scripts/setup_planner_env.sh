#!/usr/bin/env bash
# Create the small, CPU-only environment used by scripts/generate_demos.py.
#
#   PYTHON=python3.12 [PIP_INDEX=<mirror>] bash scripts/setup_planner_env.sh
#
# ManiSkill's motion-planning solvers need mplib 0.1.1, whose binary wheels crash (segfault while
# loading the robot model) under NumPy 2, which the main environment requires. This environment pins
# ManiSkill's own versions with NumPy < 2 and is used only to plan demonstrations; replay, training
# and evaluation stay in the main environment.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/scripts/env.sh"

PYTHON=${PYTHON:-python3.12}
PIP_INDEX=${PIP_INDEX:-https://pypi.org/simple}
PLANNER_ENV=${PLANNER_ENV:-$CVLA_ROOT/envs/mp}

if [ ! -f "$PLANNER_ENV/bin/activate" ]; then
    "$PYTHON" -m venv "$PLANNER_ENV"
fi
"$PLANNER_ENV/bin/pip" install -q -i "$PIP_INDEX" --upgrade pip uv
"$PLANNER_ENV/bin/uv" pip install --python "$PLANNER_ENV/bin/python" --index-url "$PIP_INDEX" \
    "mani_skill==3.0.1" "sapien==3.0.3" "mplib==0.1.1" "numpy<2" "gymnasium==1.4.0" "torch==2.5.1"
echo "planner environment: $PLANNER_ENV"
echo "use: PYTHONPATH=$REPO/src $PLANNER_ENV/bin/python scripts/generate_demos.py --task StackCubeGreenOnRed-v1"
