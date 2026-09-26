# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Split a packed decoder/DFlash embedding into one CPU session without re-exporting weights.

python -m onnxruntime_genai.models.split_cpu_embedding --input MODEL --output NEW_MODEL
The output hard-links the original external data and tokenizer files (same filesystem).
"""

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def split_graph(model, input_name, hidden_size, output_name="inputs_embeds"):
    """Return (embedding graph, removed initializer names), modifying model in place."""
    ids = next((i for i in model.graph.input if i.name == input_name), None)
    if ids is None or ids.type.tensor_type.elem_type != TensorProto.INT64:
        raise ValueError("Expected an int64 token input")
    dims = ids.type.tensor_type.shape.dim
    if len(dims) != 1 or not dims[0].dim_param:
        raise ValueError("Only packed dynamic token inputs are supported")
    token_dimension = dims[0].dim_param
    candidates = [
        n
        for n in model.graph.node
        if n.op_type in ("Gather", "GatherBlockQuantized") and len(n.input) >= 2 and n.input[1] == input_name
    ]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one token embedding Gather or GatherBlockQuantized")
    node = candidates[0]
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if attributes.get("gather_axis", attributes.get("axis", 0)) != 0:
        raise ValueError("Embedding gather axis must be zero")
    initializers = {i.name: i for i in model.graph.initializer}
    weight_names = {name for index, name in enumerate(node.input) if index != 1 and name}
    if not weight_names <= initializers.keys():
        raise ValueError("Embedding parameters must be initializers")
    weight = initializers[node.input[0]]
    if len(weight.dims) != 2 or weight.dims[1] != hidden_size:
        raise ValueError("Embedding weight does not match hidden_size (packed UINT8 weights are not supported)")
    if node.op_type == "GatherBlockQuantized" and (len(node.input) < 3 or node.input[2] not in initializers):
        raise ValueError("Quantized embedding scales must be an initializer")
    dtype = initializers[node.input[2]].data_type if node.op_type == "GatherBlockQuantized" else weight.data_type
    if dtype not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16):
        raise ValueError("CPU embedding output must be float32, float16, or bfloat16")
    old_output = node.output[0]
    if (
        output_name in {i.name for i in model.graph.input}
        or output_name in initializers
        or any(output_name in other.output for other in model.graph.node if other is not node)
    ):
        raise ValueError(f"Tensor name already exists: {output_name}")
    lookup = copy.deepcopy(node)
    lookup.input[1] = "input_ids"
    lookup.output[0] = "inputs_embeds"
    embedding = helper.make_model(
        helper.make_graph(
            [lookup],
            "cpu_embedding",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"])],
            [helper.make_tensor_value_info("inputs_embeds", dtype, ["num_tokens", hidden_size])],
            [copy.deepcopy(initializers[name]) for name in sorted(weight_names)],
        ),
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
    )
    model.graph.node.remove(node)
    for consumer in model.graph.node:
        for index, name in enumerate(consumer.input):
            if name == old_output:
                consumer.input[index] = output_name
    if any(o.name == old_output for o in model.graph.output):
        raise ValueError("Embedding output is also a graph output")
    used = {name for n in model.graph.node for name in n.input}
    if weight_names & used:
        raise ValueError("Embedding weights have other consumers; cannot offload them safely")
    for name in weight_names:
        model.graph.initializer.remove(initializers[name])
    # DFlash's selector still consumes token IDs after the lookup has been removed.
    if input_name not in used:
        model.graph.input.remove(ids)
    model.graph.input.append(helper.make_tensor_value_info(output_name, dtype, [token_dimension, hidden_size]))
    kept_info = [v for v in model.graph.value_info if v.name != old_output and v.name not in weight_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_info)
    return embedding, weight_names


def lookup_signature(model, source_dir):
    """Compare the operator and exact shared storage, ignoring graph-local tensor names."""
    node = model.graph.node[0]
    weights = []
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    for index, name in enumerate(node.input):
        if index == 1 or not name:
            continue
        value = copy.deepcopy(initializers[name])
        value.ClearField("name")
        for entry in value.external_data:
            if entry.key == "location":
                entry.value = str((source_dir / entry.value).resolve())
        weights.append(value.SerializeToString())
    return node.op_type, node.domain, sorted(a.SerializeToString() for a in node.attribute), weights


def convert(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Output directory must not exist; the input model is never modified")
    config = json.loads((source / "genai_config.json").read_text())
    model_config = config["model"]
    if model_config.get("embedding", {}).get("filename"):
        raise ValueError("Model already has an embedding session")
    if not config.get("engine", {}).get("dynamic_batching"):
        raise ValueError("CPU embedding requires a dynamic-batching Engine model")
    if any(path.is_dir() for path in source.iterdir()):
        raise ValueError("Converter requires a flat model directory")
    if model_config.get("mtp", {}).get("filename"):
        raise ValueError("MTP is not supported by this converter")
    hidden_size = model_config["decoder"]["hidden_size"]
    graphs = {}
    embedding = None
    for component in ("decoder", "dflash2", "dspark"):
        settings = model_config.get(component)
        if not settings or not settings.get("filename"):
            continue
        filename = settings["filename"]
        if Path(filename).name != filename:
            raise ValueError("Converter requires graph files in the model directory")
        graph = onnx.load(source / filename, load_external_data=False)
        inputs = settings.setdefault("inputs", {})
        extracted, removed = split_graph(graph, inputs.get("input_ids", "input_ids"), hidden_size)
        if embedding is None:
            embedding = extracted
        elif lookup_signature(embedding, source) != lookup_signature(extracted, source):
            raise ValueError("Target and drafter must share identical embedding weights and lookup attributes")
        inputs["inputs_embeds"] = "inputs_embeds"
        settings["shared_initializers"] = [
            i for i in settings.get("shared_initializers", []) if i["name"] not in removed
        ]
        graphs[filename] = graph
    embedding_filename = "embedding.onnx"
    suffix = 0
    while embedding_filename in graphs or (source / embedding_filename).exists():
        suffix += 1
        embedding_filename = f"embedding_{suffix}.onnx"
    model_config["embedding"] = {
        "filename": embedding_filename,
        "session_options": {"intra_op_num_threads": 1},
        "inputs": {"input_ids": "input_ids"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
    }
    graphs[embedding_filename] = embedding
    # ORT rejects external-data symlinks that resolve outside the model directory.
    for graph in graphs.values():
        for tensor in graph.graph.initializer:
            for entry in tensor.external_data:
                if entry.key == "location" and (
                    Path(entry.value).name != entry.value or not (source / entry.value).is_file()
                ):
                    raise ValueError(f"Unsupported external data location: {entry.value}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".", dir=destination.parent))
    try:
        for path in source.iterdir():
            if path.name not in graphs and path.name != "genai_config.json":
                os.link(path.resolve(), staging / path.name)
        for name, graph in graphs.items():
            # Serialization preserves external offsets and does not load or rewrite multi-GB weights.
            (staging / name).write_bytes(graph.SerializeToString())
        (staging / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(convert(args.input, args.output))
