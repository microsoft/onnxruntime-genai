# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pack an active unified-prefill DFlash package into model.onnx and model.onnx.data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import onnx
from onnx import AttributeProto, TensorProto


def graphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                yield from graphs(attribute.g)
            elif attribute.type == AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from graphs(nested)


def tensors(graph):
    for nested in graphs(graph):
        if nested.sparse_initializer:
            raise ValueError("Sparse initializers are not supported")
        yield from nested.initializer
        for node in nested.node:
            for attribute in node.attribute:
                if attribute.type == AttributeProto.TENSOR:
                    yield attribute.t
                elif attribute.type == AttributeProto.TENSORS:
                    yield from attribute.tensors


def deduplicate_initializers(graph):
    removed = 0
    for nested in graphs(graph):
        unique = {}
        for tensor in nested.initializer:
            if tensor.name in unique:
                if tensor != unique[tensor.name]:
                    raise ValueError(f"Conflicting initializers named {tensor.name}")
                removed += 1
            else:
                unique[tensor.name] = tensor
        values = list(unique.values())
        del nested.initializer[:]
        nested.initializer.extend(values)
    return removed


class DataWriter:
    def __init__(self, source, destination):
        self.source = source.resolve(strict=True)
        self.destination = destination
        self.stream = destination.open("xb")
        self.ranges = {}
        self.records = []

    def source_path(self, name):
        path = (self.source / name).resolve(strict=True)
        if not path.is_relative_to(self.source) or not path.is_file():
            raise ValueError(f"Data reference escapes package or is not a file: {name}")
        return path

    def append(self, data):
        padding = (-self.stream.tell()) % 4096
        self.stream.write(bytes(padding))
        offset = self.stream.tell()
        self.stream.write(data)
        self.records.append({"offset": offset, "length": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return offset

    def copy_range(self, name, offset, length):
        path = self.source_path(name)
        if offset < 0 or length <= 0 or offset + length > path.stat().st_size:
            raise ValueError(f"Invalid external data range: {name} offset={offset} length={length}")
        key = (path, offset, length)
        if key in self.ranges:
            return self.ranges[key]
        self.stream.write(bytes((-self.stream.tell()) % 4096))
        start = self.stream.tell()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            source.seek(offset)
            remaining = length
            while remaining:
                block = source.read(min(8 << 20, remaining))
                if not block:
                    raise EOFError(f"Short read of {path}")
                self.stream.write(block)
                digest.update(block)
                remaining -= len(block)
        self.ranges[key] = start
        self.records.append({"source": name, "source_offset": offset, "offset": start,
                             "length": length, "sha256": digest.hexdigest()})
        return start

    def verify(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        with self.destination.open("rb") as stream:
            for record in self.records:
                stream.seek(record["offset"])
                digest = hashlib.sha256()
                remaining = record["length"]
                while remaining:
                    block = stream.read(min(8 << 20, remaining))
                    if not block:
                        raise EOFError(f"Short read of packed data at {record['offset']}")
                    digest.update(block)
                    remaining -= len(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Packed data checksum mismatch at {record['offset']}")

    def close(self):
        self.stream.close()


def rewrite_pool(path, writer):
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "qwen38_weights_pool 1":
        raise ValueError("Unsupported weight-pool header")
    result = [lines[0]]
    count = 0
    for line in lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "pool":
            rank = int(fields[5])
            if len(fields) != 7 + rank or int(fields[1]) != count:
                raise ValueError(f"Invalid pool entry: {line}")
            fields[3] = str(writer.copy_range(fields[-1], int(fields[3]), int(fields[4])))
            fields[-1] = writer.destination.name
            count += 1
        elif fields[0] == "session":
            if len(fields) != 4 or fields[1] != "target" or not 0 <= int(fields[3]) < count:
                raise ValueError(f"Only target-session pool bindings are supported: {line}")
        else:
            raise ValueError(f"Unknown weight-pool record: {line}")
        result.append(" ".join(fields))
    return "\n".join(result) + "\n"


def rewrite_ple(path, writer):
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "qwen38_ple_table 1":
        raise ValueError("Unsupported PLE table header")
    head_dim = None
    shards = 0
    result = []
    for line in lines:
        fields = line.split()
        if fields and fields[0] == "head_dim":
            head_dim = int(fields[1])
            if head_dim <= 0:
                raise ValueError("PLE head_dim must be positive")
        if fields and fields[0] == "shard":
            if len(fields) != 4 or head_dim is None or int(fields[3]) <= 0:
                raise ValueError(f"Invalid PLE shard: {line}")
            offset = writer.copy_range(fields[1], int(fields[2]), int(fields[3]) * head_dim)
            line = f"shard {writer.destination.name} {offset} {fields[3]}"
            shards += 1
        result.append(line)
    if not shards:
        raise ValueError("PLE table has no shards")
    return "\n".join(result) + "\n"


def consolidate(source, destination):
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(source):
        raise ValueError("Destination must not be inside the source package")
    config = json.loads((source / "genai_config.json").read_text())
    model_config = config["model"]
    graph_config = model_config["qwen38_graph"]
    prefill = graph_config["prefill"]
    if (not model_config.get("dflash2", {}).get("filename") or not graph_config["fold"]["enabled"] or
            not prefill["enabled"] or not prefill["unified"] or
            graph_config["mtp"].get("prefill_head", True) or prefill.get("head_kv", {}).get("filename")):
        raise ValueError("Requires DFlash, folded target, unified prefill, and no standalone head execution")
    if "custom_ops_librar" in json.dumps(model_config):
        raise ValueError("Migrate custom ONNX operators before consolidating")
    target_name = model_config["decoder"]["filename"]
    destination.mkdir(parents=True)
    writer = DataWriter(source, destination / "model.onnx.data")
    try:
        target = onnx.load(writer.source_path(target_name), load_external_data=False)
        duplicates = deduplicate_initializers(target.graph)
        for tensor in tensors(target.graph):
            if tensor.data_location == TensorProto.EXTERNAL:
                metadata = {item.key: item.value for item in tensor.external_data}
                length = int(metadata["length"])
                offset = writer.copy_range(metadata["location"], int(metadata.get("offset", 0)), length)
            # ORT shape inference needs embedded axes/shapes and other small constants.
            elif (len(tensor.raw_data) >= 1024 and tensor.data_type in
                  (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE)):
                length = len(tensor.raw_data)
                offset = writer.append(tensor.raw_data)
            else:
                continue
            tensor.ClearField("external_data")
            tensor.ClearField("raw_data")
            tensor.data_location = TensorProto.EXTERNAL
            for key, value in (("location", writer.destination.name), ("offset", str(offset)), ("length", str(length))):
                item = tensor.external_data.add()
                item.key, item.value = key, value
        onnx.save_model(target, destination / "model.onnx")
        if prefill.get("weights_pool"):
            pool = rewrite_pool(writer.source_path(prefill["weights_pool"]), writer)
            (destination / "weights_pool.txt").write_text(pool)
            prefill["weights_pool"] = "weights_pool.txt"
        ple = rewrite_ple(writer.source_path(graph_config["ple_table"]), writer)
        (destination / "ple_table.txt").write_text(ple)
        graph_config["ple_table"] = "ple_table.txt"
        draft_name = model_config["dflash2"]["filename"]
        draft = onnx.load(writer.source_path(draft_name), load_external_data=False)
        locations = {dict((item.key, item.value) for item in tensor.external_data)["location"]
                     for tensor in tensors(draft.graph) if tensor.data_location == TensorProto.EXTERNAL}
        for name in (draft_name, *sorted(locations)):
            path = writer.source_path(name)
            output = destination / path.relative_to(source)
            if output.exists():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Retained assets are immutable; hardlinks avoid copying the separate drafter weights.
            os.link(path, output)
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
            path = source / name
            if path.is_file():
                shutil.copy2(path, destination / name)
        model_config["decoder"]["filename"] = "model.onnx"
        graph_config["fold"].pop("head_r2", None)
        prefill.pop("dense", None)
        prefill.pop("head_kv", None)
        prefill.pop("constants", None)
        graph_config.pop("head", None)
        graph_config.pop("plane_library", None)
        (destination / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
        writer.verify()
        return {"source": str(source), "destination": str(destination), "duplicate_initializers_removed": duplicates,
                "data_bytes": writer.stream.tell(), "verified_ranges": writer.records}
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = consolidate(args.source, args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "verified_ranges"}))


if __name__ == "__main__":
    main()
