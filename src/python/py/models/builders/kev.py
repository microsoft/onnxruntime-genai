# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""jaredpalmer/kev-4b PEFT adapter and decision-head export."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


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
    return source.is_dir() and (source / "adapter_config.json").is_file() and (source / "head.pt").is_file()


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
    if adapter_config.get("peft_type") is None:
        raise ValueError("KEV adapter_config.json must describe a PEFT adapter")
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


def build_kev_model(checkpoint: dict) -> onnx.ModelProto:
    """Build KEV's grouped option-scoring ONNX graph.

    ``checkpoint`` is the normalized result of ``load_kev_checkpoint`` (or an
    equivalent validated synthetic fixture). The graph gathers caller-selected
    decision and box-end token states, projects Q/K, scales their dot products,
    masks invalid options, and returns raw scores plus per-question
    probabilities. ONNX checker failures propagate to the caller.
    """
    state = checkpoint["state"]
    policy = kev_policy()
    head_dim, hidden_size = state["q.weight"].shape
    initializers = []
    for name in policy["head_keys"]:
        array = state[name].detach().cpu().float().numpy()
        if name.endswith(".weight"):
            array = array.T
        initializers.append(numpy_helper.from_array(array, name))
    scale = np.asarray(math.sqrt(head_dim) * checkpoint["temperature"], dtype=np.float32)
    initializers.extend(
        [
            numpy_helper.from_array(scale, "score_denominator"),
            numpy_helper.from_array(np.asarray(-3.4028235e38, dtype=np.float32), "masked_score"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "unsqueeze_axis"),
            numpy_helper.from_array(np.asarray([1, 2], dtype=np.int64), "query_unsqueeze_axes"),
            numpy_helper.from_array(np.asarray([2], dtype=np.int64), "option_unsqueeze_axis"),
            numpy_helper.from_array(np.asarray([1, hidden_size], dtype=np.int64), "query_shape_suffix"),
            numpy_helper.from_array(np.asarray([hidden_size], dtype=np.int64), "option_shape_suffix"),
            numpy_helper.from_array(np.asarray([-1], dtype=np.int64), "reduce_axis"),
        ]
    )
    nodes = [
        # GatherElements selects sequence positions independently for each
        # batch row while retaining the hidden dimension.
        helper.make_node("Shape", ["decide_indices"], ["query_batch_shape"]),
        helper.make_node(
            "Concat", ["query_batch_shape", "query_shape_suffix"],
            ["query_gather_shape"], axis=0,
        ),
        helper.make_node(
            "Unsqueeze", ["decide_indices", "query_unsqueeze_axes"],
            ["query_indices"],
        ),
        helper.make_node(
            "Expand", ["query_indices", "query_gather_shape"],
            ["expanded_query_indices"],
        ),
        helper.make_node(
            "GatherElements", ["hidden_states", "expanded_query_indices"],
            ["grouped_decide_states"], axis=1,
        ),
        helper.make_node(
            "Squeeze", ["grouped_decide_states", "unsqueeze_axis"],
            ["decide_states"],
        ),
        helper.make_node("Shape", ["option_indices"], ["option_batch_shape"]),
        helper.make_node(
            "Concat", ["option_batch_shape", "option_shape_suffix"],
            ["option_gather_shape"], axis=0,
        ),
        helper.make_node(
            "Unsqueeze", ["option_indices", "option_unsqueeze_axis"],
            ["grouped_option_indices"],
        ),
        helper.make_node(
            "Expand", ["grouped_option_indices", "option_gather_shape"],
            ["expanded_option_indices"],
        ),
        helper.make_node(
            "GatherElements", ["hidden_states", "expanded_option_indices"],
            ["option_states"], axis=1,
        ),
        helper.make_node("Gemm", ["decide_states", "q.weight", "q.bias"], ["q"], name="q/Gemm"),
        helper.make_node("MatMul", ["option_states", "k.weight"], ["k_linear"], name="k/MatMul"),
        helper.make_node("Add", ["k_linear", "k.bias"], ["k"], name="k/Add"),
        helper.make_node("Unsqueeze", ["q", "unsqueeze_axis"], ["q_grouped"]),
        helper.make_node("Mul", ["q_grouped", "k"], ["qk"]),
        helper.make_node("ReduceSum", ["qk", "reduce_axis"], ["dot"], keepdims=0),
        helper.make_node("Div", ["dot", "score_denominator"], ["scores"]),
        # Mask before the final axis softmax so normalization occurs separately
        # within every question and padded options receive no probability mass.
        helper.make_node("Where", ["option_mask", "scores", "masked_score"], ["masked_scores"]),
        helper.make_node("Softmax", ["masked_scores"], ["probabilities"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes,
        "kev_head",
        [
            helper.make_tensor_value_info(
                "hidden_states", TensorProto.FLOAT,
                [None, None, hidden_size],
            ),
            helper.make_tensor_value_info("decide_indices", TensorProto.INT64, [None]),
            helper.make_tensor_value_info("option_indices", TensorProto.INT64, [None, None]),
            helper.make_tensor_value_info("option_mask", TensorProto.BOOL, [None, None]),
        ],
        [
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, None]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, None]),
        ],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.producer_name = "onnxruntime-genai"
    helper.set_model_props(
        model,
        {
            "model_type": "jaredpalmer/kev-4b",
            "head_dim": str(head_dim),
            "temperature": repr(policy["temperature"]),
            "pooling": "gather decide_indices and option_indices (box_end)",
            "grouping": "option_indices and option_mask are [question, option]",
            "adapter_revision": policy["adapter_revision"],
            "base_revision": policy["base_revision"],
        },
    )
    onnx.checker.check_model(model)
    return model


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
    onnx.save(build_kev_model(checkpoint), output_root / filename)
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
