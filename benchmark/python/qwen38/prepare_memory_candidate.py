# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import onnx
from onnx import helper


def transform_graph(model, expected_indexers, preserve_context=False):
    initializers = [tensor.SerializeToString() for tensor in model.graph.initializer]
    heads = [node for node in model.graph.node if node.name == "/lm_head/MatMul"]
    if len(heads) != 1 or heads[0].op_type != "MatMul" or list(heads[0].output) != ["logits"]:
        raise ValueError("Expected one named LM-head MatMul producing logits")
    if any(value.name == "logits_indices" for value in model.graph.input):
        raise ValueError("Source already selects logits; use the original model package")
    head = heads[0]
    selected = "/lm_head/selected_hidden_states"
    if any(selected in node.output for node in model.graph.node):
        raise ValueError("Selected-hidden-state output name already exists")
    nodes = list(model.graph.node)
    position = next(index for index, node in enumerate(nodes) if node.name == head.name)
    nodes.insert(
        position,
        helper.make_node(
            "Gather",
            [head.input[0], "logits_indices"],
            [selected],
            name="/lm_head/selected_hidden_states/Gather",
            axis=0,
        ),
    )
    head.input[0] = selected
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.input.append(
        helper.make_tensor_value_info(
            "logits_indices",
            onnx.TensorProto.INT32,
            ["num_logits"],
        )
    )
    for value in list(model.graph.value_info) + list(model.graph.output):
        if value.name == "logits":
            value.type.tensor_type.shape.dim[0].dim_param = "num_logits"
    indexers = [node for node in model.graph.node if node.op_type == "PackedSparseAttentionIndexer"]
    if len(indexers) != expected_indexers:
        raise ValueError(f"Expected {expected_indexers} indexers, found {len(indexers)}")
    for node in indexers:
        attrs = {attr.name: attr for attr in node.attribute}
        if attrs["compress_ratio"].i != 4 or attrs["state_capacity"].i != 65536:
            raise ValueError(f"Unsupported indexer geometry: {node.name}")
        if not preserve_context:
            attrs["state_capacity"].i = 2560
    updated = 0
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.name.endswith(".indexer_key"):
            dims = value.type.tensor_type.shape.dim
            if len(dims) != 3 or dims[1].dim_value != 65536 or dims[2].dim_value != 128:
                raise ValueError(f"Unsupported indexer tensor: {value.name}")
            if not preserve_context:
                dims[1].dim_value = 2560
            updated += 1
    if updated != 2 * expected_indexers:
        raise ValueError("Indexer input/output shape count does not match nodes")
    if initializers != [tensor.SerializeToString() for tensor in model.graph.initializer]:
        raise ValueError("Initializer bytes or metadata changed")


def candidate_config(config, preserve_context=False):
    if preserve_context:
        for component in ("decoder", "mtp"):
            config["model"][component]["inputs"]["logits_indices"] = "logits_indices"
        return config
    config["model"]["context_length"] = 10240
    config["search"]["max_length"] = 10240
    for component in ("decoder", "mtp"):
        section = config["model"][component]
        section["inputs"]["logits_indices"] = "logits_indices"
        options = section["session_options"]
        options["enable_mem_pattern"] = False
        options["ep.cuda.enable_gemm_auto_tune"] = "1"
    providers = config["model"]["decoder"]["session_options"]["provider_options"]
    cuda_options = [provider["cuda"] for provider in providers if "cuda" in provider]
    if len(cuda_options) != 1:
        raise ValueError("Expected exactly one decoder CUDA provider")
    cuda_options[0].update(
        {
            "enable_cuda_graph": "1",
            "arena_extend_strategy": "1",
            "max_dead_bytes_per_chunk": "0",
        }
    )
    config["model"]["mtp"]["enabled"] = True
    config["engine"]["dynamic_batching"].update(
        {
            "block_size": 256,
            "num_blocks": 40,
            "max_batch_size": 1,
            "max_scheduled_tokens": 4096,
            "gpu_utilization_factor": 0.05,
        }
    )
    config["speculative"]["max_draft_tokens"] = 2
    return config


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Prepare experimental Qwen selected-logits graphs")
    parser.add_argument("--source", type=Path, required=True, help="Original NVFP4 Engine package")
    parser.add_argument("--destination", type=Path, required=True, help="New model directory (must not exist)")
    parser.add_argument(
        "--preserve-context",
        action="store_true",
        help="Select logits only; preserve original indexer capacity and all other configuration settings",
    )
    args = parser.parse_args()
    if not args.source.is_dir() or args.destination.exists():
        parser.error("Source must exist and destination must not exist")
    if args.destination.resolve().is_relative_to(args.source.resolve()):
        parser.error("Destination must not be inside the source package")
    config = json.loads((args.source / "genai_config.json").read_text())
    models = {}
    for component, count in (("decoder", 12), ("mtp", 1)):
        filename = config["model"][component]["filename"]
        model = onnx.load(args.source / filename, load_external_data=False)
        transform_graph(model, count, preserve_context=args.preserve_context)
        models[filename] = model
    config = candidate_config(config, preserve_context=args.preserve_context)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-aL", "--reflink=auto", str(args.source), str(args.destination)], check=True)
    weights = {}
    for source in args.source.glob("*.data"):
        destination = args.destination / source.name
        if destination.is_symlink() or destination.stat().st_nlink != 1:
            raise RuntimeError(f"Weight copy must be an independent regular file: {source.name}")
        expected = file_hash(source)
        if file_hash(destination) != expected:
            raise RuntimeError(f"Copied weight bytes differ: {source.name}")
        weights[source.name] = expected
    for filename, model in models.items():
        onnx.save_model(model, args.destination / filename)
    (args.destination / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "status": "experimental; unqualified on actual 80 GiB hardware",
        "source": str(args.source.resolve()),
        "transformation": "selected logits only" if args.preserve_context else "selected logits and bounded indexers",
        "configured_context_tokens": config["model"]["context_length"],
        "observed_engine_request_ceiling_tokens": None if args.preserve_context else 9217,
        "required_environment": {} if args.preserve_context else {"ORT_QMOE_NVFP4_PACKED_PREFILL": "1"},
        "weight_sha256": weights,
        "limitations": [
            "Long-context inference and performance are unverified; Engine capacity depends on the cache configuration.",
            "Source MTP uses SimplifiedLayerNormalization opset 22 unsupported by the stock ONNX checker.",
        ]
        if args.preserve_context
        else [
            "Only batch-one 8192-input/512-output workload was measured.",
            "Measured 99.79 E2E TPS is not a strict at-least-100 guarantee.",
            "Sampled 79.95 GiB peak leaves only about 56 MiB; not a hard cap.",
            "Does not support original 262K context; Engine ceiling depends on runtime.",
            "Source MTP uses SimplifiedLayerNormalization opset 22 unsupported by the stock ONNX checker.",
        ],
    }
    (args.destination / "memory_candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created {args.destination}; load with matching ORT/GenAI builds and validate before deployment")


if __name__ == "__main__":
    main()
