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

import onnx_ir as ir


def split_graph(model, input_name, hidden_size, output_name="inputs_embeds"):
    """Split an IR model or ModelProto in place, returning an embedding model of the same type."""
    if not isinstance(model, ir.Model):
        # Preserve the public ModelProto contract at the serialization boundary.
        converted = ir.serde.deserialize_model(model)
        embedding, removed = _split_graph(converted, input_name, hidden_size, output_name)
        model.CopyFrom(ir.serde.serialize_model(converted))
        return ir.serde.serialize_model(embedding), removed
    return _split_graph(model, input_name, hidden_size, output_name)


def _graphs(graph):
    yield graph
    for node in graph:
        for attribute in node.attributes.values():
            if attribute.type == ir.AttributeType.GRAPH:
                yield from _graphs(attribute.as_graph())
            elif attribute.type == ir.AttributeType.GRAPHS:
                for subgraph in attribute.as_graphs():
                    yield from _graphs(subgraph)


def _split_graph(model, input_name, hidden_size, output_name):
    graph = model.graph
    ids = next((i for i in graph.inputs if i.name == input_name), None)
    if ids is None or ids.dtype != ir.DataType.INT64:
        raise ValueError("Expected an int64 token input")
    dims = ids.shape
    if dims is None or len(dims) != 1 or not isinstance(dims[0], ir.SymbolicDim) or not dims[0].value:
        raise ValueError("Only packed dynamic token inputs are supported")
    token_dimension = dims[0]
    candidates = [
        n
        for n in graph
        if n.op_type in ("Gather", "GatherBlockQuantized") and len(n.inputs) >= 2 and n.inputs[1] is ids
    ]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one token embedding Gather or GatherBlockQuantized")
    node = candidates[0]
    attributes = {name: a.value for name, a in node.attributes.items()}
    if attributes.get("gather_axis", attributes.get("axis", 0)) != 0:
        raise ValueError("Embedding gather axis must be zero")
    initializers = graph.initializers
    weight_names = {value.name for index, value in enumerate(node.inputs) if index != 1 and value is not None}
    if not weight_names <= initializers.keys():
        raise ValueError("Embedding parameters must be initializers")
    weight = node.inputs[0]
    if weight.shape is None or len(weight.shape) != 2 or weight.shape[1] != hidden_size:
        raise ValueError("Embedding weight does not match hidden_size (packed UINT8 weights are not supported)")
    if node.op_type == "GatherBlockQuantized" and (
        len(node.inputs) < 3 or node.inputs[2] is None or node.inputs[2].name not in initializers
    ):
        raise ValueError("Quantized embedding scales must be an initializer")
    dtype = node.inputs[2].dtype if node.op_type == "GatherBlockQuantized" else weight.dtype
    if dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
        raise ValueError("CPU embedding output must be float32, float16, or bfloat16")
    old_output = node.outputs[0]
    if (
        output_name in {i.name for i in graph.inputs}
        or output_name in initializers
        or any(output_name == value.name for other in graph if other is not node for value in other.outputs)
    ):
        raise ValueError(f"Tensor name already exists: {output_name}")
    if any(old_output is output for subgraph in _graphs(graph) for output in subgraph.outputs):
        raise ValueError("Embedding output is also a graph output")
    weights = [initializers[name] for name in sorted(weight_names)]
    if any(
        any(usage.node is not node for usage in value.uses())
        or any(value is output for subgraph in _graphs(graph) for output in subgraph.outputs)
        for value in weights
    ):
        raise ValueError("Embedding weights have other consumers; cannot offload them safely")
    embedding_ids = ir.Value(name="input_ids", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape(["num_tokens"]))
    embedding_output = ir.Value(
        name="inputs_embeds", type=ir.TensorType(dtype), shape=ir.Shape(["num_tokens", hidden_size])
    )
    embedding_weights = {}
    for value in weights:
        copied = ir.Value(
            name=value.name,
            type=copy.deepcopy(value.type),
            shape=copy.deepcopy(value.shape),
            const_value=value.const_value,
            doc_string=value.doc_string,
        )
        copied.metadata_props.update(value.metadata_props)
        embedding_weights[value.name] = copied
    lookup = ir.Node(
        node.domain,
        node.op_type,
        [
            embedding_ids if index == 1 else embedding_weights[value.name] if value is not None else None
            for index, value in enumerate(node.inputs)
        ],
        attributes=copy.deepcopy(node.attributes),
        outputs=[embedding_output],
        name=node.name,
        doc_string=node.doc_string,
        metadata_props=node.metadata_props,
    )
    embedding = ir.Model(
        ir.Graph(
            [embedding_ids],
            [embedding_output],
            nodes=[lookup],
            name="cpu_embedding",
            initializers=list(embedding_weights.values()),
            opset_imports=dict(graph.opset_imports),
        ),
        ir_version=model.ir_version,
    )
    replacement = ir.Value(
        name=output_name,
        type=ir.TensorType(dtype),
        shape=ir.Shape([token_dimension, hidden_size]),
        doc_string=old_output.doc_string,
    )
    replacement.metadata_props.update(old_output.metadata_props)
    for usage in tuple(old_output.uses()):
        usage.node.replace_input_with(usage.idx, replacement)
    graph.remove(node, safe=True)
    for name in weight_names:
        del initializers[name]
    # DFlash's selector still consumes token IDs after the lookup has been removed.
    if not ids.uses() and not any(ids is output for subgraph in _graphs(graph) for output in subgraph.outputs):
        graph.inputs.remove(ids)
    graph.inputs.append(replacement)
    return embedding, weight_names


def lookup_signature(model, source_dir):
    """Compare the operator and exact shared storage, ignoring graph-local tensor names."""
    if not isinstance(model, ir.Model):
        model = ir.serde.deserialize_model(model)
    node = model.graph[0]
    weights = []
    for index, value in enumerate(node.inputs):
        if index == 1 or value is None:
            continue
        tensor = value.const_value
        if isinstance(tensor, ir.ExternalTensor):
            storage = (str((source_dir / tensor.location).resolve()), tensor.offset, tensor.length)
        else:
            storage = tensor.tobytes()
        weights.append((tensor.dtype, tuple(tensor.shape), tensor.doc_string, dict(tensor.metadata_props), storage))
    return node.op_type, node.domain, sorted(
        ir.serde.serialize_attribute(a).SerializeToString() for a in node.attributes.values()
    ), weights


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
        graph = ir.load(source / filename)
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
        for subgraph in _graphs(graph.graph):
            for value in subgraph.initializers.values():
                tensor = value.const_value
                if isinstance(tensor, ir.ExternalTensor) and (
                    Path(tensor.location).name != tensor.location or not (source / tensor.location).is_file()
                ):
                    raise ValueError(f"Unsupported external data location: {tensor.location}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".", dir=destination.parent))
    try:
        for path in source.iterdir():
            if path.name not in graphs and path.name != "genai_config.json":
                os.link(path.resolve(), staging / path.name)
        for name, graph in graphs.items():
            # Serialization preserves external offsets and does not load or rewrite multi-GB weights.
            ir.save(graph, staging / name)
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
