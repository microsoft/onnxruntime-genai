# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Restore a consolidated Qwen/DFlash package's native 262144-token context."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import torch

from consolidate_dflash_package import graphs, tensors


CONTEXT = 262144
GUARD = 4


def rotary_rows(start, stop, width, theta, numpy_precision):
    positions = np.arange(start, stop, dtype=np.int64)
    if numpy_precision:
        inverse = 1.0 / (theta ** (np.arange(0, width * 2, 2, dtype=np.float64) / (width * 2)))
        angles = positions[:, None] * inverse[None, :]
        return [torch.from_numpy(fn(angles).astype(np.float32)).to(torch.bfloat16).view(torch.uint16).numpy()
                for fn in (np.cos, np.sin)]
    inverse = 1.0 / (theta ** (torch.arange(0, width * 2, 2, dtype=torch.float32) / (width * 2)))
    angles = torch.from_numpy(positions).float()[:, None] * inverse[None, :]
    return [fn(angles).to(torch.bfloat16).view(torch.uint16).numpy() for fn in (torch.cos, torch.sin)]


def read_tensor(root, tensor):
    metadata = {entry.key: entry.value for entry in tensor.external_data}
    if tensor.data_type != TensorProto.BFLOAT16 or len(tensor.dims) != 2:
        raise ValueError(f"Expected BF16 rotary matrix: {tensor.name}")
    path = (root / metadata["location"]).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Rotary data escapes package: {path}")
    return np.memmap(path, mode="r", dtype=np.uint16, offset=int(metadata["offset"]), shape=tuple(tensor.dims))


def extend_rotary(root, model, output_root, draft):
    caches = [tensor for tensor in tensors(model.graph)
              if tensor.name.endswith(("cos_cache", "sin_cache", "cos_cache.folded", "sin_cache.folded"))]
    expected_names = {"dflash2.cos_cache", "dflash2.sin_cache"} if draft else {
        "model.rotary_emb.cos_cache", "model.rotary_emb.sin_cache",
        "capture.indexer_cos_cache.folded", "capture.indexer_sin_cache.folded",
        "head.mtp.rotary_emb.cos_cache", "head.mtp.rotary_emb.sin_cache"}
    if {tensor.name for tensor in caches} != expected_names:
        raise ValueError("Unexpected rotary-cache names; refusing an unqualified extension")
    data_name = "dflash2.onnx.data" if draft else "model.onnx.data"
    with (output_root / data_name).open("ab") as stream:
        for tensor in caches:
            old = read_tensor(root, tensor)
            rows, width = old.shape
            if rows > CONTEXT + GUARD:
                raise ValueError(f"Unexpected existing rotary length: {tensor.name}")
            base_width = width if draft else 32
            theta = 10000.0 if draft else 10000000.0
            indices = sorted({0, 1, 2, min(1024, rows - 1), rows // 2, rows - 1})
            trig = 0 if "cos_cache" in tensor.name else 1
            for position in indices:
                expected = rotary_rows(position, position + 1, base_width, theta, draft)[trig]
                if width != base_width:
                    expected = np.tile(expected, (1, width // base_width))
                if not np.array_equal(old[position:position + 1], expected):
                    raise ValueError(f"RoPE formula does not match original {tensor.name} at {position}")
            stream.write(bytes((-stream.tell()) % 4096))
            offset = stream.tell()
            stream.write(old.tobytes())
            extra = rotary_rows(rows, CONTEXT + GUARD, base_width, theta, draft)[trig]
            if width != base_width:
                extra = np.tile(extra, (1, width // base_width))
            stream.write(extra.tobytes())
            tensor.dims[0] = CONTEXT + GUARD
            tensor.ClearField("external_data")
            for key, value in (("location", data_name), ("offset", str(offset)),
                               ("length", str((CONTEXT + GUARD) * width * 2))):
                item = tensor.external_data.add()
                item.key, item.value = key, value
            stream.flush()
            packed = read_tensor(output_root, tensor)
            if not np.array_equal(packed[:rows], old) or not np.array_equal(packed[rows:], extra):
                raise ValueError(f"Rotary readback mismatch: {tensor.name}")


def resize_target(model, old_capacity):
    capacity = CONTEXT + GUARD
    old_state = (old_capacity + 3) // 4
    new_state = (capacity + 3) // 4
    counts = {"cache_dimensions": 0, "index_dimensions": 0, "index_attributes": 0,
              "fold_attributes": 0, "length_constants": 0}
    for graph in graphs(model.graph):
        for value in (*graph.input, *graph.output, *graph.value_info):
            for dimension in value.type.tensor_type.shape.dim:
                if dimension.dim_value == old_capacity:
                    dimension.dim_value = capacity
                    counts["cache_dimensions"] += 1
                elif dimension.dim_value == old_state:
                    dimension.dim_value = new_state
                    counts["index_dimensions"] += 1
        for tensor in graph.initializer:
            if tensor.name == "generic_guarded_dsa_total_sequence_length":
                original = numpy_helper.to_array(tensor)
                if original.tolist() != [old_capacity]:
                    raise ValueError("Unexpected guarded DSA total sequence length")
                tensor.CopyFrom(numpy_helper.from_array(np.asarray([capacity], dtype=original.dtype), tensor.name))
                counts["length_constants"] += 1
        for node in graph.node:
            for attribute in node.attribute:
                if node.op_type == "PackedSparseAttentionIndexer" and attribute.name == "state_capacity":
                    if attribute.i != old_state:
                        raise ValueError("Unexpected index state capacity")
                    attribute.i = new_state
                    counts["index_attributes"] += 1
                elif node.op_type == "FoldMid2" and attribute.name == "head_capacity":
                    if attribute.i != old_capacity:
                        raise ValueError("Unexpected fold capacity")
                    attribute.i = capacity
                    counts["fold_attributes"] += 1
    if counts["index_attributes"] != 12 or counts["fold_attributes"] != 1 or counts["length_constants"] != 1:
        raise ValueError(f"Unexpected target graph capacity contracts: {counts}")
    return counts


def update_rotary_pool(path, target):
    initializers = {tensor.name: tensor for tensor in target.graph.initializer}
    lines = [line.split() for line in path.read_text().splitlines()]
    replacements = {}
    for fields in lines:
        if fields and fields[0] == "session" and fields[1] == "target":
            tensor = initializers[fields[2]]
            if "rotary_emb." in tensor.name and tensor.dims[0] == CONTEXT + GUARD:
                replacements[fields[3]] = tensor
    for fields in lines:
        if fields and fields[0] == "pool" and fields[1] in replacements:
            tensor = replacements[fields[1]]
            metadata = {item.key: item.value for item in tensor.external_data}
            fields[3], fields[4] = metadata["offset"], metadata["length"]
            fields[6:8] = [str(value) for value in tensor.dims]
            fields[-1] = metadata["location"]
    path.write_text("\n".join(" ".join(fields) for fields in lines) + "\n")


def extend(source, output):
    source = source.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(source):
        raise ValueError("Output must not be inside source")
    config = json.loads((source / "genai_config.json").read_text())
    model_config = config["model"]
    if model_config["decoder"]["filename"] != "model.onnx" or model_config["dflash2"]["filename"] != "dflash2.onnx":
        raise ValueError("Expected consolidated package with separate DFlash graph")
    old_context = model_config["context_length"]
    old_capacity = model_config["qwen38_graph"]["kv_capacity"]
    if old_context != 131072 or old_capacity != old_context + GUARD:
        raise ValueError("Expected original 128K package with four internal guard slots")
    output.mkdir(parents=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_dir():
            (output / relative).mkdir(exist_ok=True)
        elif path.name in ("model.onnx.data", "dflash2.onnx.data"):
            subprocess.run(["cp", "--reflink=auto", "--sparse=always", str(path), str(output / relative)], check=True)
        else:
            shutil.copy2(path, output / relative)
    target = onnx.load(source / "model.onnx", load_external_data=False)
    draft = onnx.load(source / "dflash2.onnx", load_external_data=False)
    changes = resize_target(target, old_capacity)
    extend_rotary(source, target, output, False)
    extend_rotary(source, draft, output, True)
    update_rotary_pool(output / model_config["qwen38_graph"]["prefill"]["weights_pool"], target)
    onnx.save_model(target, output / "model.onnx")
    onnx.save_model(draft, output / "dflash2.onnx")
    model_config["context_length"] = CONTEXT
    model_config["qwen38_graph"]["kv_capacity"] = CONTEXT + GUARD
    config["search"]["max_length"] = CONTEXT
    (output / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
    onnx.checker.check_model(str(output / "model.onnx"))
    return {"context_length": CONTEXT, "kv_capacity": CONTEXT + GUARD, "graph_changes": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(extend(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
