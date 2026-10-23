# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Stream procedural target weights into one ONNX external-data file."""

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from onnx import TensorProto as T
from onnx import external_data_helper

from .quantization import prepack_int2, quantize_expert


def tensor_bytes(value):
    return value.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()


def quantize_fp8(value, block):
    if value.ndim != 2 or value.shape[1] % block:
        raise ValueError(f"FP8 conversion requires a matrix with K divisible by {block}")
    values = value.float().reshape(value.shape[0], -1, block)
    if not torch.isfinite(values).all():
        raise ValueError("FP8 conversion requires finite weights")
    scales = values.abs().amax(-1).clamp_min(1e-12) / 448.0
    packed = (values / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return packed.reshape_as(value), scales


def rotary_cache(rows, width, kind):
    if rows <= 0 or width not in (32, 64) or kind not in ("cos", "sin"):
        raise ValueError("Unsupported Qwen rotary cache dimensions or function")
    # The packed indexer repeats the model's 32 partial-rotary frequencies.
    inv_frequency = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2).float() / 64))
    frequency = torch.arange(rows).float().unsqueeze(-1) * inv_frequency
    value = (frequency.cos() if kind == "cos" else frequency.sin()).to(torch.bfloat16)
    return value.repeat(1, 2) if width == 64 else value


def external_tensor(spec, location, offset, length):
    tensor = T(name=spec.name, data_type=spec.dtype, dims=spec.shape, raw_data=b"\0")
    external_data_helper.set_external_data(tensor, location, offset=offset, length=length)
    tensor.ClearField("raw_data")
    return tensor


class TargetWeights:
    def __init__(self, checkpoint, output, config, *, workers=1):
        self.checkpoint = checkpoint
        self.output = Path(output)
        self.config = config
        self.path = self.output / "model.onnx.data"
        self.index_path = self.output / "target_weights.json"
        self.entries = {}
        self.stream = None
        if not 1 <= workers <= 32:
            raise ValueError("Target weight conversion requires 1 to 32 workers")
        self.workers = workers

    def __enter__(self):
        self.stream = self.path.open("xb")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.close()

    def align(self):
        padding = (-self.stream.tell()) % 4096
        self.stream.write(bytes(padding))
        return self.stream.tell()

    def write(self, spec, value):
        expected_dtype = {
            T.BFLOAT16: torch.bfloat16,
            T.FLOAT: torch.float32,
            T.FLOAT8E4M3FN: torch.float8_e4m3fn,
            T.UINT8: torch.uint8,
        }[spec.dtype]
        if tuple(value.shape) != spec.shape or value.dtype != expected_dtype:
            raise ValueError(
                f"Invalid target weight {spec.name}: expected {spec.shape}/{expected_dtype}, "
                f"got {tuple(value.shape)}/{value.dtype}"
            )
        offset = self.align()
        payload = tensor_bytes(value)
        self.stream.write(payload)
        self.entries[spec.name] = dict(offset=offset, length=len(payload), dtype=spec.dtype, shape=list(spec.shape))

    def checkpoint_value(self, spec):
        value = self.checkpoint.tensor(spec.source)
        if spec.transform in ("norm", "norm_float"):
            value = value.float() + 1.0
        elif spec.transform == "transpose":
            if value.ndim != 2:
                raise ValueError(f"Cannot transpose non-matrix checkpoint tensor: {spec.source}")
            value = value.T
        elif spec.transform != "identity":
            raise ValueError(f"Unsupported checkpoint weight transform: {spec.transform}")
        return value.to(torch.float32 if spec.dtype == T.FLOAT else torch.bfloat16)

    def experts(self, specs, layer, progress):
        by_name = {spec.transform: spec for spec in specs}
        required = {f"experts:{field}:{suffix}" for field in ("fc1", "fc2") for suffix in ("experts_weights", "scales")}
        if set(by_name) != required:
            raise ValueError(f"Expected four INT2 tensor declarations for target layer {layer}")
        for spec in specs:
            offset = self.align()
            length = int(np.prod(spec.shape)) * (2 if spec.dtype == T.BFLOAT16 else 1)
            self.entries[spec.name] = dict(offset=offset, length=length, dtype=spec.dtype, shape=list(spec.shape))
            self.stream.seek(length - 1, 1)
            self.stream.write(b"\0")
        end = self.stream.tell()
        error = energy = 0.0

        def convert(expert):
            return {
                field: quantize_expert(self.checkpoint, layer, expert, field)
                for field in ("gate_proj", "up_proj", "down_proj")
            }

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            pending = deque()
            next_expert = 0
            for expert in range(512):
                while next_expert < 512 and len(pending) < 2 * self.workers:
                    pending.append(executor.submit(convert, next_expert))
                    next_expert += 1
                converted = pending.popleft().result()
                self.write_expert(by_name, expert, converted)
                error += sum(value.squared_error for value in converted.values())
                energy += sum(value.squared_norm for value in converted.values())
                if progress and (expert + 1) % 64 == 0:
                    progress(dict(stage="int2", layer=layer, experts_completed=expert + 1))
        self.stream.seek(end)
        if progress:
            progress(
                dict(
                    stage="int2",
                    layer=layer,
                    experts_completed=512,
                    relative_weight_mse=error / energy if energy else 0.0,
                )
            )

    def write_expert(self, by_name, expert, converted):
        for field, projections in (("fc1", ("gate_proj", "up_proj")), ("fc2", ("down_proj",))):
            if field == "fc1":
                # Fused SwiGLU consumes adjacent gate/up output rows.
                packed = torch.stack([converted[p].packed for p in projections], dim=1).flatten(0, 1)
                scales = torch.stack([converted[p].scales for p in projections], dim=1).flatten(0, 1)
            else:
                packed, scales = converted["down_proj"].packed, converted["down_proj"].scales
            packed = prepack_int2(packed)
            for suffix, value in (("experts_weights", packed), ("scales", scales)):
                spec = by_name[f"experts:{field}:{suffix}"]
                payload = tensor_bytes(value)
                entry = self.entries[spec.name]
                if len(payload) * 512 != entry["length"]:
                    raise ValueError(f"Incorrect expert payload length: {spec.name}")
                self.stream.seek(entry["offset"] + expert * len(payload))
                self.stream.write(payload)

    def materialize(self, specs, progress=None):
        pending = {spec.name: spec for spec in specs}
        for spec in specs:
            if spec.name not in pending:
                continue
            transform = spec.transform
            if transform.startswith("cache_scale:"):
                continue
            if transform.startswith("experts:"):
                group = [s for s in pending.values() if s.source == spec.source and s.transform.startswith("experts:")]
                self.experts(group, int(spec.source), progress)
                for member in group:
                    pending.pop(member.name)
            elif transform.startswith(("fp8:", "fp8_scale:")):
                block = int(transform.split(":")[1])
                value = self.checkpoint.tensor(spec.source)
                # Large embedding/head matrices are quantized in bounded row chunks.
                siblings = [
                    s
                    for s in pending.values()
                    if s.source == spec.source and s.transform in (f"fp8:{block}", f"fp8_scale:{block}")
                ]
                if len(siblings) != 2:
                    raise ValueError(f"Expected paired FP8 weights/scales: {spec.source}")
                for sibling in siblings:
                    offset = self.align()
                    length = int(np.prod(sibling.shape)) * (4 if sibling.dtype == T.FLOAT else 1)
                    self.entries[sibling.name] = dict(
                        offset=offset, length=length, dtype=sibling.dtype, shape=list(sibling.shape)
                    )
                    self.stream.seek(length - 1, 1)
                    self.stream.write(b"\0")
                end = self.stream.tell()
                for start in range(0, value.shape[0], 1024):
                    quantized, scales = quantize_fp8(value[start : start + 1024], block)
                    for sibling in siblings:
                        result = scales if sibling.dtype == T.FLOAT else quantized
                        entry = self.entries[sibling.name]
                        stride = entry["length"] // value.shape[0]
                        self.stream.seek(entry["offset"] + start * stride)
                        self.stream.write(tensor_bytes(result))
                self.stream.seek(end)
                for sibling in siblings:
                    pending.pop(sibling.name)
            elif transform.startswith("rope:"):
                _, kind, width = transform.split(":")
                self.write(spec, rotary_cache(spec.shape[0], int(width), kind))
                pending.pop(spec.name)
            else:
                self.write(spec, self.checkpoint_value(spec))
                pending.pop(spec.name)
            self.stream.flush()
            self.index_path.write_text(json.dumps(self.entries, indent=2) + "\n")
        if progress:
            progress(dict(stage="target_weights", tensor_count=len(self.entries), bytes=self.stream.tell()))

    def write_ple(self):
        text = self.config["text_config"]
        prefix = "model.language_model.layers.1.ple.ple_embedding."
        n_heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
        head_dim = text["ple_embed_dim"] // n_heads
        scale = self.checkpoint.tensor(prefix + "ngram_embedding.weight_scale").float()
        if scale.numel() != 1 or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("PLE embedding requires one finite positive scale")
        multipliers = self.checkpoint.tensor(prefix + "layer_multipliers").reshape(-1).tolist()
        vocab = self.checkpoint.tensor(prefix + "ngram_heads_vocab_sizes").reshape(-1).tolist()
        offsets = self.checkpoint.tensor(prefix + "ngram_heads_offsets").reshape(-1).tolist()
        if len(multipliers) != 3 or len(vocab) != n_heads or len(offsets) != n_heads:
            raise ValueError("PLE checkpoint hash metadata does not match target architecture")
        lines = [
            "qwen38_ple_table 1",
            f"n_heads {n_heads}",
            f"head_dim {head_dim}",
            f"heads_per_ngram {text['heads_per_ngram']}",
            f"eos {text['eos_token_id']}",
            f"scale {scale.item():.17g}",
            "mult " + " ".join(map(str, [*multipliers, 0])),
            "vocab " + " ".join(map(str, vocab)),
            "offset " + " ".join(map(str, offsets)),
        ]
        total_rows = 0
        for index in range(text["split_ngram_parts"]):
            value = self.checkpoint.tensor(prefix + f"ngram_embedding.shard_{index}.weight")
            if value.ndim != 2 or value.shape[1] != head_dim or value.dtype != torch.float8_e4m3fn:
                raise ValueError(f"Unexpected PLE shard {index} shape/dtype")
            offset = self.align()
            # Avoid a second full-shard bytes copy.
            for start in range(0, value.shape[0], 65536):
                self.stream.write(tensor_bytes(value[start : start + 65536]))
            total_rows += value.shape[0]
            lines.append(f"shard {self.path.name} {offset} {value.shape[0]}")
        if total_rows < offsets[-1] + vocab[-1]:
            raise ValueError("PLE shards do not cover the checkpoint hash vocabulary")
        self.stream.flush()
        (self.output / "ple_table.txt").write_text("\n".join(lines) + "\n")


def bind_weights(model, specs, entries, location="model.onnx.data"):
    for spec in specs:
        entry = entries[spec.name]
        shape_matches = list(spec.shape) == entry["shape"]
        if spec.transform.startswith("rope:") and spec.shape[1:] == tuple(entry["shape"][1:]):
            shape_matches = spec.shape[0] <= entry["shape"][0]
        if not shape_matches or spec.dtype != entry["dtype"]:
            raise ValueError(f"External-data entry does not match graph weight: {spec.name}")
        length = entry["length"]
        if spec.transform.startswith("rope:"):
            length = int(np.prod(spec.shape)) * 2
        model.graph.initializer.append(external_tensor(spec, location, entry["offset"], length))
    return model
