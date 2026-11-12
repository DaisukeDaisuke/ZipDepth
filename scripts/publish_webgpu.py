"""Publish every build to the same user-owned Hugging Face model repository."""
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi


def main():
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    account = api.whoami()["name"]
    repo_id = f"{account}/ZipDepth-onnx"
    # The name is stable across runs. No run number, date or SHA in repo_id.
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    folder = Path("release")
    shutil.copyfile("LICENSE", folder / "LICENSE")
    manifest = json.loads((folder / "zipdepth-base-768-fp32.json").read_text(encoding="utf-8"))
    card = """---
license: mit
library_name: onnx
pipeline_tag: depth-estimation
tags:
- onnx
- webgpu
- zipdepth
---
# ZipDepth ONNX
FP32 ONNX conversion of [Fabio Tosi et al.'s ZipDepth](https://github.com/fabiotosi92/ZipDepth), with the original learned attention and convex upsampling preserved. No quantization or retraining is applied.
## License and provenance
The upstream repository distributes its code and checkpoints under its MIT LICENSE; the original copyright and permission notice is included in LICENSE. This conversion is not an official upstream release. ZipDepth was distilled from Depth Anything V2 Large; this card does not relicense the teacher or its original datasets, which are not included here.
## Input and output
Input `image` is FP32 RGB divided by 255, NCHW `[1,3,768,768]`. Preserve the source aspect ratio when resizing and pad the remaining area with black. Ignore padded regions when displaying or exporting scene points.
Output `inverse_depth` is FP32 `[1,1,768,768]`, affine-invariant inverse depth: larger means nearer. It is not a distance in meters. Calibrated distance requires a scale and shift in inverse-depth space. A relative preview may use `z = scale / (raw + shift)` with positive denominators, while retaining the raw output unchanged.
## Validation
The JSON manifest contains checkpoint and ONNX SHA-256 hashes and numerical comparisons with the original PyTorch model on a public image, RGB noise and black input. Browser latency and WebGPU operator support must be measured separately; CPU conversion checks are not browser benchmarks.
## Build
Source: https://github.com/DaisukeDaisuke/ZipDepth
Every successful build updates this same repository. Use a Hugging Face commit revision to pin an immutable model download.
"""
    card += f"Source commit: `{manifest['source_commit']}`\nONNX SHA-256: `{manifest['sha256']}`\n"
    (folder / "README.md").write_text(card, encoding="utf-8")
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder),
                               commit_message=f"Update FP32 ONNX from {manifest['source_commit'][:12]}")
    result = {"repo_id": repo_id, "revision": commit.oid,
              "url": f"https://huggingface.co/{repo_id}",
              "model_url": f"https://huggingface.co/{repo_id}/resolve/{commit.oid}/{manifest['filename']}",
              "sha256": manifest["sha256"]}
    print(json.dumps(result), flush=True)
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(
        f"Model: {result['url']}\nRevision: `{commit.oid}`\nSHA-256: `{manifest['sha256']}`\n", encoding="utf-8")


if __name__ == "__main__":
    main()
