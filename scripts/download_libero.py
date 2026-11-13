"""Download the pretrained policies, the PaliGemma tokenizer and the LIBERO demonstrations.

    source scripts/env.sh && python scripts/download_libero.py [--skip-dataset] [--reference]

Models go to the Hugging Face cache ($HF_HOME). The LIBERO demonstrations (the four 10-task suites in LeRobot format,
with failed and no-op steps removed as in OpenVLA) go to $HF_LEROBOT_HOME/HuggingFaceVLA/libero. Downloads resume
after interruptions.

pi0 tokenises instructions with the PaliGemma tokenizer, whose official repository is gated. Its files are taken from
an ungated copy and accepted only if every file has the official file hash, so the tokenizer is the one pi0 was
pretrained with. They go to $CVLA_DATA/tokenizers/paligemma.

LIBERO's simulation assets (object meshes, scenes, textures) are not part of the hf-libero package; they are downloaded
(at a fixed revision) to $CVLA_DATA/libero/assets and copied into the installed package, where LIBERO looks first.

--reference also downloads LeRobot's pi0 fine-tuned on all four suites (REFERENCE), used only to check that
evaluation/evaluate_libero.py reproduces a published LIBERO success rate.
"""

import argparse
import hashlib
import importlib.util
import os
import shutil
import time

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = ["lerobot/pi0_base", "lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]
DATASET = "HuggingFaceVLA/libero"
REFERENCE = "lerobot/pi0_libero_finetuned_v044"
ASSETS, ASSETS_REVISION = "lerobot/libero-assets", "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
TOKENIZER_COPY = "leo009/paligemma-3b-pt-224"
# Hashes of the files in google/paligemma-3b-pt-224: SHA-256 for files stored with Git LFS, Git blob SHA-1 otherwise.
TOKENIZER_FILES = {
    "tokenizer.model": ("sha256", "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"),
    "tokenizer.json": ("sha256", "ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563"),
    "tokenizer_config.json": ("blob", "960dcec0bccc6e594d865eceb63c8078e164cfb2"),
    "special_tokens_map.json": ("blob", "0c18fdd94629ea652e2ef36ed9a422474ad0a8e8"),
    "added_tokens.json": ("blob", "ff9246e420971b10c510b39a0de289132c2bf23f"),
}


def retry(fn, retries=50, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return fn(**kwargs)
        except Exception as exc:  # network errors; partial files resume on retry
            print(f"attempt {attempt} failed: {exc!r}; retrying", flush=True)
            time.sleep(15)
    raise RuntimeError(f"download failed: {kwargs}")


def file_hash(path, kind):
    data = open(path, "rb").read()
    if kind == "sha256":
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download_tokenizer(out):
    os.makedirs(out, exist_ok=True)
    for name, (kind, expected) in TOKENIZER_FILES.items():
        path = retry(hf_hub_download, repo_id=TOKENIZER_COPY, filename=name, local_dir=out)
        got = file_hash(path, kind)
        if got != expected:
            os.remove(path)
            raise RuntimeError(f"{name}: hash {got} differs from the official {expected}")
    print(f"tokenizer -> {out} (all {len(TOKENIZER_FILES)} files match the official hashes)", flush=True)


def install_assets():
    out = os.path.join(os.environ["CVLA_DATA"], "libero", "assets")
    retry(snapshot_download, repo_id=ASSETS, repo_type="dataset", revision=ASSETS_REVISION, local_dir=out,
          max_workers=8)
    spec = importlib.util.find_spec("libero")
    if spec is None:
        print(f"LIBERO assets -> {out}; hf-libero is not installed in this environment, run again from one that has it",
              flush=True)
        return
    package = os.path.join(spec.submodule_search_locations[0], "libero", "assets")
    if not os.path.exists(package):
        shutil.copytree(out, package, ignore=shutil.ignore_patterns(".cache", ".gitattributes"))
    print(f"LIBERO assets -> {out}, installed in {package}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--reference", action="store_true")
    args = p.parse_args()
    download_tokenizer(os.path.join(os.environ["CVLA_DATA"], "tokenizers", "paligemma"))
    install_assets()
    for repo in MODELS + ([REFERENCE] if args.reference else []):
        path = retry(snapshot_download, repo_id=repo, max_workers=8, ignore_patterns=["*.gif", "*.ipynb", "onnx/*"])
        print(f"{repo} -> {path}", flush=True)
    if not args.skip_dataset:
        out = os.path.join(os.environ["HF_LEROBOT_HOME"], DATASET)
        path = retry(snapshot_download, repo_id=DATASET, repo_type="dataset", local_dir=out, max_workers=8)
        print(f"{DATASET} -> {path}", flush=True)


if __name__ == "__main__":
    main()
