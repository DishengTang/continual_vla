# Source this file before running anything:  source scripts/env.sh
# All heavy state (Python env, caches, datasets, checkpoints) lives under CVLA_ROOT,
# so nothing large is written to the system disk.

: "${CVLA_ROOT:?set CVLA_ROOT to a directory on a large disk before sourcing scripts/env.sh}"
export CVLA_ROOT
export CVLA_ENV=${CVLA_ENV:-$CVLA_ROOT/envs/vla}
export CVLA_DATA=${CVLA_DATA:-$CVLA_ROOT/data}
export CVLA_RUNS=${CVLA_RUNS:-$CVLA_ROOT/runs}

# Caches
export XDG_CACHE_HOME=$CVLA_ROOT/cache
export PIP_CACHE_DIR=$CVLA_ROOT/cache/pip
export UV_CACHE_DIR=$CVLA_ROOT/cache/uv
export UV_HTTP_TIMEOUT=300
export HF_HOME=$CVLA_ROOT/cache/huggingface
export TORCH_HOME=$CVLA_ROOT/cache/torch
export TRITON_CACHE_DIR=$CVLA_ROOT/cache/triton
export TMPDIR=$CVLA_ROOT/tmp

# Simulator assets/demonstrations and LeRobot datasets
export MS_ASSET_DIR=$CVLA_DATA/maniskill
export HF_LEROBOT_HOME=$CVLA_DATA/lerobot

# Keep BLAS/OpenMP pools small; torch otherwise sizes them from the host core count.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# Optional Hugging Face mirror, e.g. HF_ENDPOINT=https://hf-mirror.com (set before sourcing).
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
if [ "$HF_ENDPOINT" != "https://huggingface.co" ]; then
    # Xet storage fetches file chunks from huggingface.co directly, bypassing the mirror.
    export HF_HUB_DISABLE_XET=1
fi

# Headless Vulkan ICD written by setup_env.sh (HEADLESS_VULKAN=1)
if [ -f "$CVLA_ROOT/vulkan/nvidia_egl_icd.json" ]; then
    export VK_ICD_FILENAMES=$CVLA_ROOT/vulkan/nvidia_egl_icd.json
fi

mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TMPDIR" \
         "$MS_ASSET_DIR" "$HF_LEROBOT_HOME" "$CVLA_RUNS"

if [ -f "$CVLA_ENV/bin/activate" ]; then
    source "$CVLA_ENV/bin/activate"
fi
