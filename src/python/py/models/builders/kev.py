# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""jaredpalmer/kev-4b PEFT adapter and decision-head export."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnx_ir as ir
import torch


def kev_policy() -> dict:
    """Return immutable KEV artifact and graph policy."""
    return {
        "adapter_revision": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
        "base_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "temperature": 2.406050072164233,
        "head_keys": ("q.weight", "q.bias", "k.weight", "k.bias"),
        "base_model": "Qwen/Qwen3.5-4B-Base",
        "checkpoint_keys": {
            "args", "suite_sha256", "init_source", "temperature_fit",
            "base", "head", "base_revision", "lora", "head_dim",
            "option_isolation", "special_embeddings", "weights_dtype",
            "temperature", "holdout", "lora_placement",
        },
    }


def is_kev_artifact(source: str | Path) -> bool:
    """Return whether a directory has the KEV PEFT adapter/head file layout."""
    source = Path(source)
    return (
        source.is_dir()
        and (source / "adapter_config.json").is_file()
        and (source / "adapter_model.safetensors").is_file()
        and (source / "head.pt").is_file()
    )


def load_kev_checkpoint(source: str | Path) -> dict:
    """Safely load and validate the released KEV adapter and head metadata.

    ``source`` must contain ``adapter_config.json`` and ``head.pt``. The
    returned object contains the exact Q/K state, head dimension, and
    temperature used by graph construction. Unknown metadata, base/pin drift,
    option-isolation changes, tensor-name drift, or shape drift raise
    ``ValueError``; pickle execution is disabled with ``weights_only=True``.
    """
    source = Path(source)
    policy = kev_policy()
    if not is_kev_artifact(source):
        raise ValueError("KEV artifact requires adapter_config.json and head.pt")
    try:
        adapter_config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("KEV adapter_config.json must be valid JSON") from error
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("KEV adapter_config.json peft_type must be LORA")
    if adapter_config.get("base_model_name_or_path") != policy["base_model"]:
        raise ValueError(
            "KEV adapter base_model_name_or_path must be "
            f"{policy['base_model']}"
        )
    try:
        checkpoint = torch.load(source / "head.pt", map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("failed to safely load KEV head.pt") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("KEV head.pt must be an object")
    if set(checkpoint) != policy["checkpoint_keys"]:
        raise ValueError(
            "KEV head.pt keys must be exactly "
            f"{sorted(policy['checkpoint_keys'])}"
        )
    state = checkpoint["head"]
    if not isinstance(state, dict) or set(state) != set(policy["head_keys"]):
        raise ValueError("KEV head state must contain exactly q.weight/q.bias/k.weight/k.bias")
    if checkpoint.get("base") != policy["base_model"]:
        raise ValueError(f"KEV base must be {policy['base_model']}")
    if checkpoint.get("base_revision") != policy["base_revision"]:
        raise ValueError(f"KEV base_revision must be {policy['base_revision']}")
    if checkpoint.get("head_dim") != 256:
        raise ValueError("KEV head_dim must be 256")
    if checkpoint.get("option_isolation") is not False:
        raise ValueError("KEV option_isolation must be false")
    if not isinstance(checkpoint.get("suite_sha256"), str) or not checkpoint["suite_sha256"]:
        raise ValueError("KEV suite_sha256 must be a non-empty string")
    if not isinstance(checkpoint.get("weights_dtype"), str) or not checkpoint["weights_dtype"]:
        raise ValueError("KEV weights_dtype must be a non-empty string")
    temperature = checkpoint.get("temperature")
    if not isinstance(temperature, (int, float)) or not math.isclose(
        float(temperature), policy["temperature"], rel_tol=0, abs_tol=1e-15
    ):
        raise ValueError(f"KEV temperature must be {policy['temperature']}")
    for name in ("q.weight", "k.weight"):
        if not torch.is_tensor(state[name]) or tuple(state[name].shape) != (256, 2560):
            raise ValueError(f"KEV {name} must have shape (256, 2560)")
    for name in ("q.bias", "k.bias"):
        if not torch.is_tensor(state[name]) or tuple(state[name].shape) != (256,):
            raise ValueError(f"KEV {name} must have shape (256,)")
    return {"state": state, "head_dim": 256, "temperature": float(temperature)}


def build_kev_model(checkpoint: dict) -> ir.Model:
    """Build KEV's grouped option-scoring IR model.

    ``checkpoint`` is the normalized result of ``load_kev_checkpoint`` (or an
    equivalent validated synthetic fixture). The graph gathers caller-selected
    decision and box-end token states, projects Q/K, scales their dot products,
    masks invalid options, and returns raw scores plus per-question
    probabilities.
    """
    state = checkpoint["state"]
    policy = kev_policy()
    head_dim, hidden_size = state["q.weight"].shape
    initializers = {}

    def initializer(array, name):
        value = ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.from_numpy(array.dtype)),
            shape=ir.Shape(array.shape),
            const_value=ir.tensor(array, name=name),
        )
        initializers[name] = value
        return value

    for name in policy["head_keys"]:
        array = state[name].detach().cpu().float().numpy()
        if name.endswith(".weight"):
            array = array.T
        initializer(array, name)
    scale = np.asarray(math.sqrt(head_dim) * checkpoint["temperature"], dtype=np.float32)
    denominator = initializer(scale, "score_denominator")
    masked_score = initializer(np.asarray(-3.4028235e38, dtype=np.float32), "masked_score")
    zero = initializer(np.asarray(0.0, dtype=np.float32), "zero_probability")
    unsqueeze_axis = initializer(np.asarray([1], dtype=np.int64), "unsqueeze_axis")
    query_axes = initializer(np.asarray([1, 2], dtype=np.int64), "query_unsqueeze_axes")
    option_axis = initializer(np.asarray([2], dtype=np.int64), "option_unsqueeze_axis")
    query_suffix = initializer(np.asarray([1, hidden_size], dtype=np.int64), "query_shape_suffix")
    option_suffix = initializer(np.asarray([hidden_size], dtype=np.int64), "option_shape_suffix")
    reduce_axis = initializer(np.asarray([-1], dtype=np.int64), "reduce_axis")
    inputs = [
        ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))
        for name, dtype, shape in (
            ("hidden_states", ir.DataType.FLOAT, [None, None, hidden_size]),
            ("decide_indices", ir.DataType.INT64, [None]),
            ("option_indices", ir.DataType.INT64, [None, None]),
            ("option_mask", ir.DataType.BOOL, [None, None]),
        )
    ]
    hidden_states, decide_indices, option_indices, option_mask = inputs
    nodes = []

    def emit(op_type, inputs, output_name, *, name=None, **attributes):
        node = ir.Node(
            "", op_type, inputs, ir.convenience.convert_attributes(attributes),
            outputs=[ir.Value(name=output_name)], name=name,
        )
        nodes.append(node)
        return node.outputs[0]

    # GatherElements selects sequence positions independently for each
    # batch row while retaining the hidden dimension.
    query_batch_shape = emit("Shape", [decide_indices], "query_batch_shape")
    query_gather_shape = emit("Concat", [query_batch_shape, query_suffix], "query_gather_shape", axis=0)
    query_indices = emit("Unsqueeze", [decide_indices, query_axes], "query_indices")
    expanded_query_indices = emit("Expand", [query_indices, query_gather_shape], "expanded_query_indices")
    grouped_decide_states = emit(
        "GatherElements", [hidden_states, expanded_query_indices], "grouped_decide_states", axis=1
    )
    decide_states = emit("Squeeze", [grouped_decide_states, unsqueeze_axis], "decide_states")
    option_batch_shape = emit("Shape", [option_indices], "option_batch_shape")
    option_gather_shape = emit("Concat", [option_batch_shape, option_suffix], "option_gather_shape", axis=0)
    grouped_option_indices = emit("Unsqueeze", [option_indices, option_axis], "grouped_option_indices")
    expanded_option_indices = emit(
        "Expand", [grouped_option_indices, option_gather_shape], "expanded_option_indices"
    )
    option_states = emit("GatherElements", [hidden_states, expanded_option_indices], "option_states", axis=1)
    q = emit("Gemm", [decide_states, initializers["q.weight"], initializers["q.bias"]], "q", name="q/Gemm")
    k_linear = emit("MatMul", [option_states, initializers["k.weight"]], "k_linear", name="k/MatMul")
    k = emit("Add", [k_linear, initializers["k.bias"]], "k", name="k/Add")
    q_grouped = emit("Unsqueeze", [q, unsqueeze_axis], "q_grouped")
    qk = emit("Mul", [q_grouped, k], "qk")
    dot = emit("ReduceSum", [qk, reduce_axis], "dot", keepdims=0)
    scores = emit("Div", [dot, denominator], "scores")
    # Mask before the final axis softmax so normalization occurs separately
    # within every question and padded options receive no probability mass.
    masked_scores = emit("Where", [option_mask, scores, masked_score], "masked_scores")
    raw_probabilities = emit("Softmax", [masked_scores], "raw_probabilities", axis=-1)
    probabilities = emit("Where", [option_mask, raw_probabilities, zero], "probabilities")
    for output in (probabilities, scores):
        output.type = ir.TensorType(ir.DataType.FLOAT)
        output.shape = ir.Shape([None, None])
    graph = ir.Graph(
        inputs, [probabilities, scores], nodes=nodes, name="kev_head",
        initializers=list(initializers.values()), opset_imports={"": 18},
    )
    return ir.Model(
        graph, ir_version=10, producer_name="onnxruntime-genai",
        metadata_props={
            "model_type": "jaredpalmer/kev-4b",
            "head_dim": str(head_dim),
            "temperature": repr(policy["temperature"]),
            "pooling": "gather decide_indices and option_indices (box_end)",
            "grouping": "option_indices and option_mask are [question, option]",
            "adapter_revision": policy["adapter_revision"],
            "base_revision": policy["base_revision"],
        },
    )


def export_kev_components(options, output_root: Path) -> dict:
    """Validate KEV pins, export ``kev_head.onnx``, and return its manifest.

    The adapter/base revisions must match the released contract and
    ``options.model_source`` must pass full PEFT/head validation. Failures raise
    ``ValueError``. The manifest records base-tokenizer provenance and leaves
    token-index discovery and request orchestration to the caller.
    """
    policy = kev_policy()
    if options.artifact_revision != policy["adapter_revision"]:
        raise ValueError(
            f"KEV artifact_revision must be {policy['adapter_revision']}"
        )
    if options.base_revision != policy["base_revision"]:
        raise ValueError(f"KEV base_revision must be {policy['base_revision']}")
    checkpoint = load_kev_checkpoint(options.model_source)
    filename = "kev_head.onnx"
    from .base import Model  # noqa: PLC0415

    Model.save_onnx_model(build_kev_model(checkpoint), output_root / filename)
    return {
        "schema_version": 1,
        "model_type": "jaredpalmer/kev-4b",
        "provenance": {
            "adapter_revision": options.artifact_revision,
            "base_model": policy["base_model"],
            "base_revision": options.base_revision,
            "tokenizer": "base",
            "adapter_tokenizer": "ignored",
        },
        "orchestration": {
            "decide_indices": "caller-provided token indices",
            "option_indices": "caller-provided box_end token indices grouped per question",
            "owned_by": "caller",
        },
        "components": {
            "backbone": {
                "role": "backbone",
                "filename": options.backbone_filename,
                "outputs": {"hidden_states": "hidden_states"},
            },
            "kev_head": {
                "role": "head",
                "filename": filename,
                "inputs": {
                    "hidden_states": "hidden_states",
                    "decide_indices": "decide_indices",
                    "option_indices": "option_indices",
                    "option_mask": "option_mask",
                },
                "outputs": {"probabilities": "probabilities", "scores": "scores"},
            },
        },
    }
