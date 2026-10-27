#!/usr/bin/env bash
# Create the project Python environment (Linux + NVIDIA GPU).
#
#   PYTHON=python3.12 [PIP_INDEX=<mirror>] [INSTALL_APT=1] [HEADLESS_VULKAN=1] bash scripts/setup_env.sh
#
# INSTALL_APT=1 installs the Vulkan loader (needed by SAPIEN rendering) and FFmpeg
# (needed by LeRobot video datasets); it requires root.
# HEADLESS_VULKAN=1 is for containers without an X server, where the driver's default
# GLX-based Vulkan ICD fails (vulkaninfo lists only llvmpipe). It writes an ICD manifest
# that uses the EGL library instead; scripts/env.sh selects it via VK_ICD_FILENAMES.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/scripts/env.sh"

PYTHON=${PYTHON:-python3.12}
PIP_INDEX=${PIP_INDEX:-https://pypi.org/simple}

if [ "${INSTALL_APT:-0}" = "1" ]; then
    apt-get update -q
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q libvulkan1 vulkan-tools ffmpeg
fi

if [ "${HEADLESS_VULKAN:-0}" = "1" ]; then
    mkdir -p "$CVLA_ROOT/vulkan"
    cat > "$CVLA_ROOT/vulkan/nvidia_egl_icd.json" <<'EOF'
{ "file_format_version": "1.0.1", "ICD": { "library_path": "libEGL_nvidia.so.0", "api_version": "1.3.0" } }
EOF
fi

if [ ! -f "$CVLA_ENV/bin/activate" ]; then
    "$PYTHON" -m venv "$CVLA_ENV"
fi
source "$CVLA_ENV/bin/activate"

# uv downloads wheels in parallel, which matters on slow or throttled links.
pip install -q -i "$PIP_INDEX" --upgrade pip uv
uv pip install --index-url "$PIP_INDEX" -r "$REPO/requirements.txt"
uv pip install --no-deps -e "$REPO"

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    print("arch list", torch.cuda.get_arch_list())
EOF
