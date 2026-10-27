"""Download the pretrained policy and the ManiSkill demonstrations.

    source scripts/env.sh && python scripts/download_assets.py

Models go to the Hugging Face cache ($HF_HOME); demonstrations go to $MS_ASSET_DIR/demos/<task>/.
Downloads resume after interruptions.
"""

import os
import time

from huggingface_hub import snapshot_download

MODELS = ["lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]
DEMO_PATTERNS = [
    "demos/PushCube-v1/motionplanning/*",
    "demos/StackCube-v1/motionplanning/*",
]


def download(retries=20, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return snapshot_download(max_workers=8, **kwargs)
        except Exception as exc:  # network errors; partial files resume on retry
            print(f"attempt {attempt} failed: {exc!r}; retrying")
            time.sleep(10)
    raise RuntimeError(f"download failed: {kwargs}")


def main():
    for repo in MODELS:
        path = download(repo_id=repo, ignore_patterns=["*.gif", "*.ipynb", "onnx/*"])
        print(f"{repo} -> {path}")
    path = download(repo_id="haosulab/ManiSkill_Demonstrations", repo_type="dataset",
                    allow_patterns=DEMO_PATTERNS, local_dir=os.environ["MS_ASSET_DIR"])
    print(f"demonstrations -> {path}/demos")


if __name__ == "__main__":
    main()
