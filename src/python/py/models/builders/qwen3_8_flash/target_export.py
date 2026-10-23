# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Export raw NVFP4 target weights and a BF16-cache graph for fresh calibration.

This intermediate export is not a runnable GenAI/DFlash package. The final S4
graph must be emitted only after calibration supplies its INT8 cache scales.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import onnx
import torch

from .graph import build_target_graph
from .quantization import SafetensorsCheckpoint
from .weights import TargetWeights, bind_weights, tensor_bytes

WEIGHT_RECIPE = "qwen38-procedural-v3-interleaved-swiglu-partial-rope"


def export_target(checkpoint, output, *, workers=8, threads=1, calibration_capacity=4096):
    checkpoint = Path(checkpoint).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((checkpoint / "config.json").read_text())
    _, runtime_weights = build_target_graph(config)
    calibration, calibration_weights = build_target_graph(
        config, calibration=True, calibration_capacity=calibration_capacity
    )
    if not 1 <= threads <= 32 or not 1 <= workers <= 32:
        raise ValueError("CPU thread and worker counts must each be in [1, 32]")
    # Final target/PLE data is roughly 90 GiB; do not start an expensive
    # conversion when it cannot fit, and never remove existing assets to fit it.
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 100 * 1024**3:
        raise RuntimeError("Raw target export requires at least 100 GiB free output disk space")
    torch.set_num_threads(threads)
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {
        "stage": "target_weight_conversion",
        "complete": False,
        "genai_package_ready": False,
        "checkpoint": str(checkpoint),
        "public_context_length": 262144,
        "calibration_capacity": calibration_capacity,
        "workers": workers,
        "threads_per_worker": threads,
        "weight_recipe": WEIGHT_RECIPE,
    }
    report_path = output / "target_export.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    def progress(record):
        record = {**record, "elapsed_seconds": time.monotonic() - started}
        print(json.dumps(record), flush=True)
        with (output / "target_conversion_progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    with SafetensorsCheckpoint(checkpoint) as source, TargetWeights(source, output, config, workers=workers) as writer:
        writer.materialize(runtime_weights, progress)
        writer.write_ple()
        bind_weights(calibration, calibration_weights, writer.entries)
        onnx.save(calibration, output / "target.calibration.onnx")
    onnx.checker.check_model(str(output / "target.calibration.onnx"))
    report.update(
        stage="awaiting_fresh_calibration",
        complete=True,
        elapsed_seconds=time.monotonic() - started,
        target_data_bytes=(output / "model.onnx.data").stat().st_size,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def finalize_target(output):
    output = Path(output)
    destination = output / "model.onnx"
    if destination.exists():
        raise FileExistsError(destination)
    profile = json.loads((output / "kv_calibration.json").read_text())
    export = json.loads((output / "target_export.json").read_text())
    if export.get("weight_recipe") != WEIGHT_RECIPE or profile.get("weight_recipe") != WEIGHT_RECIPE:
        raise ValueError("Fresh KV calibration and target weights must use the current procedural weight recipe")
    graph_hash = hashlib.sha256((output / "target.calibration.onnx").read_bytes()).hexdigest()
    if profile.get("method") != "fresh-bf16-per-channel-amax" or profile.get("graph_sha256") != graph_hash:
        raise ValueError("KV profile must come from fresh calibration of this target graph")
    model, specs = build_target_graph(json.loads((output / "config.json").read_text()))
    entries = json.loads((output / "target_weights.json").read_text())
    cache_specs = [spec for spec in specs if spec.transform.startswith("cache_scale:")]
    expected = {spec.transform.removeprefix("cache_scale:").replace(":", ".") for spec in cache_specs}
    if set(profile.get("scales", {})) != expected:
        raise ValueError("KV profile must contain exactly the 24 target per-channel scale vectors")
    values = {}
    for key in expected:
        value = torch.tensor(profile["scales"][key], dtype=torch.float32)
        if value.shape != (512,) or not torch.isfinite(value).all() or not (value > 0).all():
            raise ValueError(f"KV profile scales must be 512 finite positive values: {key}")
        values[key] = value
    bind_weights(model, [spec for spec in specs if spec not in cache_specs], entries)
    with (output / "model.onnx.data").open("ab") as stream:
        for spec in cache_specs:
            key = spec.transform.removeprefix("cache_scale:").replace(":", ".")
            stream.write(bytes((-stream.tell()) % 4096))
            offset = stream.tell()
            payload = tensor_bytes(values[key])
            stream.write(payload)
            entries[spec.name] = dict(offset=offset, length=len(payload), dtype=spec.dtype, shape=list(spec.shape))
    bind_weights(model, cache_specs, entries)
    onnx.save(model, destination)
    onnx.checker.check_model(str(destination))
    (output / "target_weights.json").write_text(json.dumps(entries, indent=2) + "\n")
    lines = ["qwen38_weights_pool 1"]
    for index, spec in enumerate(specs):
        entry = entries[spec.name]
        lines.append(
            " ".join(
                map(
                    str,
                    [
                        "pool",
                        index,
                        spec.dtype,
                        entry["offset"],
                        entry["length"],
                        len(spec.shape),
                        *spec.shape,
                        "model.onnx.data",
                    ],
                )
            )
        )
    lines.extend(f"session target {spec.name} {index}" for index, spec in enumerate(specs))
    (output / "weights_pool.txt").write_text("\n".join(lines) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--calibration-capacity", type=int, default=4096)
    args = parser.parse_args()
    export_target(
        args.checkpoint,
        args.output,
        workers=args.workers,
        threads=args.threads,
        calibration_capacity=args.calibration_capacity,
    )


if __name__ == "__main__":
    main()
