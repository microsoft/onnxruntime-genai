# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Create a bulk-prefill companion graph for a native Nemotron Bolt package.

The companion is the package's fixed-R8 target graph with:
  * input_ids widened to [1, R] and aux_hidden_states to [1, R, width],
  * every Mamba2Verify node replaced by Mamba2Prefill (same inputs/outputs; terminal state in slot 0),
  * the LM head applied only to the final row, so logits are [1, 1, vocab] and pred_ids [1, 1].

Packed projections, MoE and attention already accept R rows. Weights are not copied: package files
are hardlinked into a new directory and every external initializer is listed in
model.decoder.shared_initializers, so the target and prefill sessions share one device copy.
Requires an ONNX Runtime build with Mamba2Prefill and session.gqa.fixed_query_chunk_size.

Example:
  python create_bolt_prefill_model.py --source /models/bolt --output /models/bolt-prefill-2048 --rows 2048
  # Ordinary target-only packages have a dynamic-row graph; pass the fixed-R8 graph explicitly:
  python create_bolt_prefill_model.py --source /models/bolt --graph /models/bolt-k7/model.onnx ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


def _external(graph: onnx.GraphProto, base: Path) -> dict[str, tuple[str, int, int]]:
    result = {}
    for init in graph.initializer:
        if init.data_location != TensorProto.EXTERNAL:
            continue
        info = {entry.key: entry.value for entry in init.external_data}
        result[init.name] = (os.path.realpath(base / info["location"]), int(info.get("offset", 0)),
                             int(info["length"]))
    return result


def _rank(graph: onnx.GraphProto, name: str) -> int:
    for value in list(graph.value_info) + list(graph.output) + list(graph.input):
        if value.name == name and value.type.tensor_type.HasField("shape"):
            return len(value.type.tensor_type.shape.dim)
    producer = next((node for node in graph.node if name in node.output), None)
    if producer is None:
        raise ValueError(f"cannot determine the rank of {name}")
    if producer.op_type == "Reshape":
        target = producer.input[1]
        for node in graph.node:
            if node.op_type == "Constant" and node.output[0] == target:
                return len(numpy_helper.to_array(node.attribute[0].t))
        for init in graph.initializer:
            if init.name == target:
                return int(init.dims[0])
    if producer.op_type in {"SkipSimplifiedLayerNormalization", "SimplifiedLayerNormalization", "Cast", "Identity"}:
        return _rank(graph, producer.input[0])
    raise ValueError(f"cannot determine the rank of {name} produced by {producer.op_type}")


def _set_shape(value: onnx.ValueInfoProto, dims: list[int]) -> None:
    shape = value.type.tensor_type.shape
    del shape.dim[:]
    for dim in dims:
        shape.dim.add().dim_value = dim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="native Bolt package directory")
    parser.add_argument("--graph", type=Path, help="fixed-R8 target graph; defaults to the package decoder graph")
    parser.add_argument("--output", type=Path, required=True, help="new package directory (must not exist)")
    parser.add_argument("--rows", type=int, default=2048, help="prefill chunk rows R, 8 < R <= 2048, R % 8 == 0")
    args = parser.parse_args()
    if not (8 < args.rows <= 2048 and args.rows % 8 == 0):
        parser.error("--rows must be a multiple of eight in [16, 2048]")
    source = args.source.resolve()
    if args.output.exists():
        parser.error("--output must be a new directory")
    config = json.loads((source / "genai_config.json").read_text())
    if config["model"]["type"] != "nemotron_bolt":
        parser.error("the source package is not a nemotron_bolt package")
    decoder = config["model"]["decoder"]
    decoder_path = source / decoder["filename"]
    graph_path = (args.graph or decoder_path).resolve()
    filename = f"prefill-r{args.rows}.onnx"
    if (source / filename).exists():
        parser.error(f"the source package already contains {filename}")

    model = onnx.load(graph_path, load_external_data=False)
    graph = model.graph
    inputs = {value.name: value for value in graph.input}
    outputs = {value.name: value for value in graph.output}
    ids = [dim.dim_value if dim.HasField("dim_value") else None for dim in inputs["input_ids"].type.tensor_type.shape.dim]
    if ids != [1, 8]:
        parser.error(f"{graph_path} must be a fixed-R8 target graph with input_ids [1,8]; got {ids}")

    # Every weight must be byte-identical to, and shareable with, the package's decoder graph.
    # Packages may hardlink the same data file, so compare file identity rather than paths.
    weights = _external(graph, graph_path.parent)

    def same(a: tuple[str, int, int] | None, b: tuple[str, int, int]) -> bool:
        return a is not None and a[1:] == b[1:] and os.path.samefile(a[0], b[0])

    if graph_path != decoder_path.resolve():
        decoder_weights = _external(onnx.load(decoder_path, load_external_data=False).graph, decoder_path.parent)
        different = [name for name, ref in weights.items() if not same(decoder_weights.get(name), ref)]
        if different:
            raise ValueError(f"{len(different)} initializers differ from the decoder graph, e.g. {different[:3]}")
    for location, _, _ in weights.values():
        packaged = source / os.path.basename(location)
        if not (packaged.is_file() and os.path.samefile(packaged, location)):
            raise ValueError(f"external data {location} is not a file of the source package")

    replaced = 0
    for node in graph.node:
        if node.op_type == "Mamba2Verify" and node.domain == "com.microsoft":
            node.op_type = "Mamba2Prefill"
            replaced += 1
    if replaced == 0:
        raise ValueError("the target graph has no Mamba2Verify nodes")

    logits_producer = next(node for node in graph.node if "logits" in node.output)
    head_output = logits_producer.input[0] if logits_producer.op_type == "Identity" else "logits"
    head = next(node for node in graph.node if head_output in node.output)
    if head.op_type != "MatMulPacked":
        raise ValueError(f"expected a MatMulPacked LM head, found {head.op_type}")
    rank = _rank(graph, head.input[0])
    if rank != 3:
        raise ValueError(f"expected a rank-3 [1,R,hidden] LM-head input, found rank {rank}")
    last = helper.make_node("Constant", [], ["bolt_prefill_last_row"], name="bolt_prefill_last_row",
                            value=helper.make_tensor("value", TensorProto.INT64, [1], [-1]))
    terminal = helper.make_node("Gather", [head.input[0], "bolt_prefill_last_row"], ["bolt_prefill_terminal_hidden"],
                                name="bolt_prefill_terminal_hidden", axis=1)
    head.input[0] = "bolt_prefill_terminal_hidden"
    position = list(graph.node).index(head)
    graph.node.insert(position, terminal)
    graph.node.insert(position, last)

    vocab = config["model"]["vocab_size"]
    width = outputs["aux_hidden_states"].type.tensor_type.shape.dim[-1].dim_value
    _set_shape(inputs["input_ids"], [1, args.rows])
    _set_shape(outputs["aux_hidden_states"], [1, args.rows, width])
    _set_shape(outputs["logits"], [1, 1, vocab])
    _set_shape(outputs["pred_ids"], [1, 1])
    del graph.value_info[:]

    args.output.mkdir(parents=True)
    output = args.output.resolve()
    for path in source.iterdir():
        if path.is_file() and path.name != "genai_config.json":
            os.link(path.resolve(), output / path.name)
    onnx.save(model, output / filename)

    shared = {entry["name"]: entry for entry in decoder.get("shared_initializers", [])}
    by_name = {init.name: init for init in graph.initializer}
    for name, (location, offset, length) in weights.items():
        if name in shared:
            continue
        init = by_name[name]
        shared[name] = {"name": name, "data_file": os.path.basename(location), "offset": str(offset),
                        "length": str(length), "data_type": int(init.data_type), "shape": [int(d) for d in init.dims]}
    decoder["prefill_filename"] = filename
    decoder["prefill_sequence_length"] = args.rows
    decoder["shared_initializers"] = list(shared.values())
    (output / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
    digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
    print(f"{output / filename}: rows={args.rows}, Mamba2Prefill={replaced}, shared_initializers={len(shared)}, "
          f"sha256={digest}")


if __name__ == "__main__":
    main()
