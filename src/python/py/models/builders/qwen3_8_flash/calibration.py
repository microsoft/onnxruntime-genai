# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Collect fresh per-channel KV maxima from the raw-exported BF16 target."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto as T

from .quantization import SafetensorsCheckpoint
from .target_export import WEIGHT_RECIPE


def ple_hash(tokens, multipliers, vocab, offsets, eos, heads_per_ngram=8):
    tokens = np.asarray(tokens, dtype=np.int64)
    multipliers = np.asarray(multipliers, dtype=np.uint64)
    vocab = np.asarray(vocab, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if tokens.ndim != 1 or tokens.size == 0 or np.any(tokens < 0):
        raise ValueError("PLE hash requires a nonempty vector of nonnegative token IDs")
    if multipliers.shape != (3,) or vocab.shape != (16,) or offsets.shape != (16,) or np.any(vocab <= 0):
        raise ValueError("Unsupported PLE hash metadata")
    context = np.concatenate((np.array([eos, eos], dtype=np.int64), tokens))
    previous = context[1:-1]
    older = np.where(previous == eos, eos, context[:-2])
    # The CUDA contract wraps UINT64 products/XOR, then takes signed INT64
    # remainder and normalizes it to a nonnegative row index.
    with np.errstate(over="ignore"):
        mixed = tokens.astype(np.uint64) * multipliers[0]
        mixed ^= previous.astype(np.uint64) * multipliers[1]
        longer = mixed ^ (older.astype(np.uint64) * multipliers[2])
    result = np.empty((tokens.size, 16), dtype=np.int64)
    for head in range(16):
        value = mixed if head < heads_per_ngram else longer
        result[:, head] = value.view(np.int64) % vocab[head] + offsets[head]
    return result


class CheckpointPle:
    def __init__(self, checkpoint, config):
        self.checkpoint = checkpoint
        self.eos = config["text_config"]["eos_token_id"]
        self.prefix = "model.language_model.layers.1.ple.ple_embedding."
        self.multipliers = checkpoint.tensor(self.prefix + "layer_multipliers").numpy()
        self.vocab = checkpoint.tensor(self.prefix + "ngram_heads_vocab_sizes").numpy()
        self.offsets = checkpoint.tensor(self.prefix + "ngram_heads_offsets").numpy()
        self.scale = checkpoint.tensor(self.prefix + "ngram_embedding.weight_scale").float().item()
        self.shards = [
            checkpoint.tensor(self.prefix + f"ngram_embedding.shard_{index}.weight")
            for index in range(config["text_config"]["split_ngram_parts"])
        ]
        self.ends = np.cumsum([shard.shape[0] for shard in self.shards])

    def embeddings(self, tokens):
        rows = ple_hash(tokens, self.multipliers, self.vocab, self.offsets, self.eos)
        shard_ids = np.searchsorted(self.ends, rows, side="right")
        if np.any(shard_ids >= len(self.shards)):
            raise ValueError("PLE hash row exceeds checkpoint table")
        output = torch.empty((*rows.shape, 160), dtype=torch.uint8)
        for index in np.unique(shard_ids):
            positions = np.nonzero(shard_ids == index)
            local_rows = rows[positions] - (self.ends[index - 1] if index else 0)
            values = self.shards[index].view(torch.uint8)[torch.from_numpy(local_rows)]
            output[torch.from_numpy(positions[0]), torch.from_numpy(positions[1])] = values
        return (output.view(torch.float8_e4m3fn).float() * self.scale).to(torch.bfloat16).reshape(1, -1, 2560)


def read_samples(path, checkpoint, capacity):
    from tokenizers import Tokenizer  # noqa: PLC0415

    tokenizer = None
    samples = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Calibration line {number} must be an object with text or input_ids")
        if "input_ids" in record:
            ids = record["input_ids"]
        elif isinstance(record.get("text"), str):
            if tokenizer is None:
                tokenizer = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
            ids = tokenizer.encode(record["text"]).ids
        else:
            raise ValueError(f"Calibration line {number} requires text or input_ids")
        if not isinstance(ids, list) or not all(type(value) is int and 0 <= value < 248320 for value in ids):
            raise ValueError(f"Calibration line {number} has invalid token IDs")
        if not 4 <= len(ids) <= capacity:
            raise ValueError(f"Calibration line {number} needs 4..{capacity} tokens, got {len(ids)}")
        samples.append(ids)
    if not samples:
        raise ValueError("Calibration input contains no samples")
    return samples


def build_runner(ort_home, output):
    ort_home = Path(ort_home).resolve(strict=True)
    source = Path(__file__).with_name("calibration_runner.cc")
    include = ort_home / "include"
    library = ort_home / "lib"
    for required in (include / "onnxruntime_cxx_api.h", library / "libonnxruntime.so"):
        if not required.is_file():
            raise FileNotFoundError(required)
    library = (library / "libonnxruntime.so").resolve(strict=True).parent
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = (
        source.read_bytes()
        + str(ort_home).encode()
        + str(library).encode()
        + str((library / "libonnxruntime.so").stat()).encode()
    )
    executable = output / ("calibrate_target-" + hashlib.sha256(identity).hexdigest()[:16])
    if not executable.exists():
        temporary = executable.with_suffix(".tmp")
        subprocess.run(
            [
                os.environ.get("CXX", "c++"),
                "-std=c++17",
                "-O2",
                str(source),
                "-I",
                str(include),
                "-L",
                str(library),
                "-Wl,-rpath," + str(library),
                "-lonnxruntime",
                "-o",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(executable)
    return executable


def calibration_arrays(model, ids, ple, capacity):
    rows = len(ids)
    special = {
        "input_ids": np.array([ids], dtype=np.int64),
        "position_ids": np.arange(rows, dtype=np.int64).reshape(1, -1),
        "attention_mask": np.ones((1, rows), dtype=np.int64),
        "past_position_ids": np.zeros((1, capacity), dtype=np.int64),
        "logits_indices": np.array([min(rows, 4) - 1], dtype=np.int64),
        "committed_length": np.zeros(1, dtype=np.int32),
        "sequence_length_minus_one": np.array([rows - 1], dtype=np.int32),
        "cumulative_sequence_lengths": np.array([0, rows], dtype=np.int32),
        "prefill_numerics": np.ones(1, dtype=np.int32),
        "past_key_values.1.ple_context": np.full((1, 2), ple.eos, dtype=np.int64),
        "ple_embeddings.1": ple.embeddings(ids).view(torch.uint16).numpy(),
    }
    feeds = {}
    for value in model.graph.input:
        tensor_type = value.type.tensor_type
        if value.name in special:
            array = special[value.name]
        else:
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.dim_param == "prefill_cache_capacity":
                    shape.append(capacity)
                elif dim.dim_param == "prefill_scoring_capacity":
                    shape.append((capacity + 3) // 4)
                else:
                    raise ValueError(f"Unsupported dynamic calibration input dimension: {value.name}/{dim.dim_param}")
            dtype = {T.BFLOAT16: np.uint16, T.FLOAT: np.float32, T.INT64: np.int64, T.INT32: np.int32}.get(
                tensor_type.elem_type
            )
            if dtype is None:
                raise ValueError(f"Unsupported calibration input type: {value.name}/{tensor_type.elem_type}")
            array = np.zeros(shape, dtype=dtype)
        feeds[value.name] = array
    return feeds


def write_request(directory, model, arrays, output_names):
    directory = Path(directory)
    directory.mkdir()
    types = {value.name: value.type.tensor_type.elem_type for value in model.graph.input}
    lines = ["qwen38_calibration_request 1", str(len(arrays))]
    for index, (name, array) in enumerate(arrays.items()):
        path = directory / f"input.{index}.bin"
        array.tofile(path)
        lines.append(
            f"{json.dumps(name)} {types[name]} {array.ndim} "
            + " ".join(map(str, array.shape))
            + " "
            + json.dumps(str(path))
        )
    lines.append(str(len(output_names)))
    outputs = {}
    for index, name in enumerate(output_names):
        path = directory / f"output.{index}.bin"
        lines.append(json.dumps(name) + " " + json.dumps(str(path)))
        outputs[name] = path
    manifest = directory / "request.txt"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest, outputs


def calibrate_target(target, checkpoint, prompts, provider, ort_home, *, margin=1.05):
    target, checkpoint, prompts = map(Path, (target, checkpoint, prompts))
    export = json.loads((target / "target_export.json").read_text())
    if export["stage"] != "awaiting_fresh_calibration" or not export["complete"]:
        raise ValueError("Target weight conversion must finish before fresh calibration")
    if export.get("weight_recipe") != WEIGHT_RECIPE:
        raise ValueError("Fresh calibration requires the current procedural target weight recipe")
    if str(checkpoint.resolve(strict=True)) != export["checkpoint"]:
        raise ValueError("Calibration checkpoint must match the completed raw-target export")
    if not np.isfinite(margin) or not 1.0 <= margin <= 4.0:
        raise ValueError("Calibration scale margin must be finite and in [1, 4]")
    destination = target / "kv_calibration.json"
    if destination.exists():
        raise FileExistsError(destination)
    capacity = export["calibration_capacity"]
    samples = read_samples(prompts, checkpoint, capacity)
    config = json.loads((checkpoint / "config.json").read_text())
    executable = build_runner(ort_home, target / ".build")
    graph_path = target / "target.calibration.onnx"
    model = onnx.load(graph_path, load_external_data=False)
    output_names = [f"calibration.{layer}.{kind}.amax" for layer in range(3, 48, 4) for kind in ("key", "value")]
    maxima = {name: np.zeros(512, dtype=np.float32) for name in output_names}
    with (
        SafetensorsCheckpoint(checkpoint) as source,
        tempfile.TemporaryDirectory(prefix="calibration-", dir=target) as scratch,
    ):
        ple = CheckpointPle(source, config)
        requests, sample_outputs = [], []
        for index, ids in enumerate(samples):
            manifest, outputs = write_request(
                Path(scratch) / str(index),
                model,
                calibration_arrays(model, ids, ple, capacity),
                [*output_names, "validation.aggregate"],
            )
            requests.append(json.dumps(str(manifest)))
            sample_outputs.append(outputs)
        request_list = Path(scratch) / "requests.txt"
        request_list.write_text("\n".join(requests) + "\n")
        subprocess.run(
            [str(executable), str(graph_path), str(Path(provider).resolve(strict=True)), str(request_list)], check=True
        )
        for index, (ids, outputs) in enumerate(zip(samples, sample_outputs, strict=True)):
            status = np.fromfile(outputs["validation.aggregate"], dtype=np.int32)
            if status.shape != (24,) or np.any(status):
                raise RuntimeError(f"Target calibration validation failed at sample {index}: {status.tolist()}")
            for name in output_names:
                array = np.fromfile(outputs[name], dtype=np.float32)
                if array.shape != (512,) or not np.isfinite(array).all() or np.any(array < 0):
                    raise RuntimeError(f"Invalid KV calibration maxima: sample={index}, output={name}")
                maxima[name] = np.maximum(maxima[name], array)
            print(
                json.dumps({"stage": "calibration", "sample": index + 1, "samples": len(samples), "tokens": len(ids)}),
                flush=True,
            )
    for name, value in maxima.items():
        if np.any(value <= 0):
            raise RuntimeError(f"Calibration did not observe positive values in every KV channel: {name}")
    profile = {
        "format": "qwen38-kv-calibration-v1",
        "method": "fresh-bf16-per-channel-amax",
        "margin": margin,
        "samples": len(samples),
        "tokens": sum(map(len, samples)),
        "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
        "graph_sha256": hashlib.sha256((target / "target.calibration.onnx").read_bytes()).hexdigest(),
        "weight_recipe": export["weight_recipe"],
        "scales": {
            name.removeprefix("calibration.").removesuffix(".amax"): (values * margin / 127.0).tolist()
            for name, values in maxima.items()
        },
    }
    destination.write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--ort-home", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=1.05)
    args = parser.parse_args()
    calibrate_target(args.target, args.checkpoint, args.prompts, args.provider, args.ort_home, margin=args.margin)


if __name__ == "__main__":
    main()
