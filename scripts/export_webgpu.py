"""Export the original FP32 ZipDepth predictor and verify numerical equivalence."""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zipdepth.model.architecture import create_model
from zipdepth.utils.model_utils import fuse_remaining_conv_bn, strip_state_dict_prefixes


def sha256(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()


def letterbox(path, height, width):
    image = Image.open(path).convert("RGB")
    scale = min(width / image.width, height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(image.resize(size, Image.Resampling.BILINEAR),
                 ((width - size[0]) // 2, (height - size[1]) // 2))
    return np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0


def check(actual, expected, label):
    if not np.isfinite(actual).all():
        raise RuntimeError(f"Nonfinite output: {label}")
    diff = np.abs(actual - expected)
    result = {"label": label, "max_absolute_error": float(diff.max()),
              "mean_absolute_error": float(diff.mean()),
              "output_min": float(actual.min()), "output_max": float(actual.max())}
    print(json.dumps(result), flush=True)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4, err_msg=label)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/zipdepth_base.pth")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--output", default="release/zipdepth-base-fp32.onnx")
    args = parser.parse_args()
    if min(args.height, args.width) < 32 or args.height % 32 or args.width % 32:
        raise ValueError("Input dimensions must be positive multiples of 32")
    torch.set_num_threads(2)
    torch.manual_seed(20261112)
    model = create_model(variant="base", global_mode="balanced", upsample_unfold=True).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = strip_state_dict_prefixes(checkpoint.get("model_state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        raise RuntimeError(f"Missing trained weights: {missing}")
    print(f"Unused checkpoint keys: {unexpected}", flush=True)
    shape = (1, 3, args.height, args.width)
    rng = np.random.default_rng(20261112)
    samples = [("public-image", letterbox("assets/examples/im0.jpg", args.height, args.width)),
               ("random-rgb", rng.random(shape, dtype=np.float32)),
               ("black", np.zeros(shape, dtype=np.float32))]
    with torch.inference_mode():
        reference = [model(torch.from_numpy(x)).numpy().copy() for _, x in samples]
    model.fuse_for_inference()
    fuse_remaining_conv_bn(model)
    model.eval()
    checks = []
    with torch.inference_mode():
        for (name, x), expected in zip(samples, reference):
            checks.append(check(model(torch.from_numpy(x)).numpy(), expected, "fusion/" + name))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the trained softmax-weighted GlobalContextBlock. Upstream export.py
    # substitutes average pooling here, which is not numerically equivalent.
    with torch.inference_mode():
        torch.onnx.export(model, torch.from_numpy(samples[0][1]), str(destination),
                          input_names=["image"], output_names=["inverse_depth"],
                          opset_version=17, do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    props = {"model": "ZipDepth-base", "precision": "float32",
             "output_semantics": "affine-invariant inverse depth; larger means nearer",
             "preprocess": "RGB / 255, NCHW; aspect-preserving resize and black padding",
             "checkpoint_sha256": sha256(args.checkpoint),
             "source_commit": os.environ.get("GITHUB_SHA", "local")}
    onnx.helper.set_model_props(graph, props)
    onnx.save(graph, str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(destination), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    for (name, x), expected in zip(samples, reference):
        actual = session.run(["inverse_depth"], {"image": x})[0]
        checks.append(check(actual, expected, "onnx/" + name))
    manifest = {**props, "filename": destination.name, "sha256": sha256(destination),
                "bytes": destination.stat().st_size, "input_name": "image",
                "output_name": "inverse_depth", "input_shape": list(shape),
                "output_shape": list(reference[0].shape), "opset": 17,
                "operators": sorted({node.op_type for node in graph.graph.node}),
                "validation": checks, "checkpoint_unused_keys": unexpected,
                "torch_version": torch.__version__, "onnxruntime_version": ort.__version__}
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "sha256": manifest["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
