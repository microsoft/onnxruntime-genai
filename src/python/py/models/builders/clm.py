# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""CLM-v0.1-8B contrastive head export."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx_ir as ir
import torch

from .base import Model


def clm_artifact_revision() -> str:
    """Return the immutable released CLM artifact revision."""
    return "e939398d4556fcd9400c76fa8c5a513202f42b0a"


def checkpoint_path(source: Path) -> Path | None:
    """Return the sole CLM checkpoint candidate, excluding KEV's ``head.pt``."""
    candidates = [path for path in sorted(source.glob("*.pt")) + sorted(source.glob("*.pth")) if path.name != "head.pt"]
    return candidates[0] if len(candidates) == 1 else None


def is_clm_artifact(source: str | Path) -> bool:
    """Return whether ``source`` safely loads with the required CLM top-level keys.

    Detection uses ``weights_only=True`` and returns ``False`` for unreadable,
    ambiguous, or non-CLM artifacts rather than leaking loader exceptions.
    """
    source = Path(source)
    required_keys = {
        "state_head",
        "action_head",
        "logit_scale",
        "cfg",
        "hidden_size",
        "projection_dim",
    }
    checkpoint = checkpoint_path(source) if source.is_dir() else source
    if checkpoint is None or not checkpoint.is_file() or checkpoint.name == "head.pt":
        return False
    try:
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return isinstance(data, dict) and required_keys.issubset(data)


def load_clm_checkpoint(source: str | Path) -> dict:
    """Safely load and validate the released CLM checkpoint contract.

    ``source`` is a checkpoint or directory containing exactly one candidate.
    The returned dictionary retains the two head state dictionaries and scalar
    scale. Unsafe loads, unknown keys, cfg drift, dimension drift, or a
    non-scalar scale raise ``ValueError``.
    """
    source = Path(source)
    required_keys = {
        "state_head",
        "action_head",
        "logit_scale",
        "cfg",
        "hidden_size",
        "projection_dim",
    }
    cfg_keys = {
        "model",
        "hidden_size",
        "width",
        "depth",
        "projection_dim",
        "activation",
        "layernorm",
        "residual",
    }
    checkpoint = checkpoint_path(source) if source.is_dir() else source
    if checkpoint is None:
        raise ValueError("CLM artifact must contain exactly one .pt or .pth checkpoint")
    try:
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"failed to safely load CLM checkpoint {checkpoint}") from error
    if not isinstance(data, dict) or set(data) != required_keys:
        raise ValueError(f"CLM checkpoint keys must be exactly {sorted(required_keys)}")
    cfg = data["cfg"]
    expected_cfg = {
        "model": "Qwen/Qwen3-8B",
        "hidden_size": 4096,
        "width": 1536,
        "depth": 3,
        "projection_dim": 512,
        "activation": "gelu",
        "layernorm": True,
        "residual": False,
    }
    if not isinstance(cfg, dict):
        raise ValueError("CLM cfg must be an object")
    if set(cfg) != cfg_keys:
        raise ValueError(f"CLM cfg keys must be exactly {sorted(cfg_keys)}")
    normalized = {str(k).lower(): (str(v).lower() if isinstance(v, str) else v) for k, v in cfg.items()}
    for key, expected in expected_cfg.items():
        normalized_expected = expected.lower() if isinstance(expected, str) else expected
        if normalized.get(key) != normalized_expected:
            raise ValueError(f"CLM cfg.{key} must be {expected!r}")
    if data["hidden_size"] != 4096 or data["projection_dim"] != 512:
        raise ValueError("CLM hidden_size/projection_dim must be 4096/512")
    if not torch.is_tensor(data["logit_scale"]) or data["logit_scale"].numel() != 1:
        raise ValueError("CLM logit_scale must be a scalar tensor")
    if not math.isfinite(float(data["logit_scale"].item())):
        raise ValueError("CLM logit_scale must be finite")
    for name in ("state_head", "action_head"):
        validate_head(
            name,
            data[name],
            int(data["hidden_size"]),
            int(cfg["width"]),
            int(data["projection_dim"]),
            int(cfg["depth"]),
        )
    return data


def validate_head(name: str, state: dict, hidden_size: int, width: int, projection_dim: int, depth: int):
    """Validate exact CLM tensor names, tensor types, and architecture-derived shapes."""
    if not isinstance(state, dict) or any(not torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"CLM {name} must be a tensor state dictionary")
    expected = {
        "inp.weight": (width, hidden_size),
        "inp.bias": (width,),
        "out.weight": (projection_dim, width),
        "out.bias": (projection_dim,),
    }
    for index in range(depth - 2):
        expected[f"hidden.{index}.weight"] = (width, width)
        expected[f"hidden.{index}.bias"] = (width,)
        expected[f"norms.{index}.weight"] = (width,)
        expected[f"norms.{index}.bias"] = (width,)
    if set(state) != set(expected):
        raise ValueError(f"CLM {name} tensor names do not match inp/hidden/norms/out schema")
    for tensor_name, shape in expected.items():
        if tuple(state[tensor_name].shape) != shape:
            raise ValueError(f"CLM {name}.{tensor_name} must have shape {shape}")


class CLMModel(Model):
    """Build the released contrastive heads with the shared Model emitters."""

    def __init__(self, checkpoint=None):
        super().__init__(graph_name="clm_heads", graph_opset=20)
        if checkpoint is not None:
            self.make_model(checkpoint)

    def make_linear(self, value, state, prefix, node_prefix):
        weight_name = f"{node_prefix}.{prefix}.weight"
        bias_name = f"{node_prefix}.{prefix}.bias"
        self.make_initializer(state[f"{prefix}.weight"].detach().cpu().float().numpy().T, weight_name)
        self.make_initializer(state[f"{prefix}.bias"].detach().cpu().float().numpy(), bias_name)
        output = f"{node_prefix}/{prefix}"
        self.make_node("Gemm", [value, weight_name, bias_name], [output], name=f"{output}/Gemm")
        return output

    def make_head(self, input_name, output_name, state, prefix, hidden_size, width, projection_dim, depth):
        """Emit input normalization, projection blocks, and output normalization."""
        validate_head(prefix, state, hidden_size, width, projection_dim, depth)
        input_axes = f"{prefix}.input_l2_axes"
        input_epsilon = f"{prefix}.input_l2_epsilon"
        self.make_initializer(np.asarray([-1], dtype=np.int64), input_axes)
        self.make_initializer(np.asarray(1e-12, dtype=np.float32), input_epsilon)
        input_norm = f"{prefix}/input_l2_norm"
        self.make_node(
            "ReduceL2",
            [input_name, input_axes],
            [input_norm],
            name=f"{prefix}/InputReduceL2",
            keepdims=1,
        )
        input_denominator = f"{prefix}/input_l2_denominator"
        self.make_node("Max", [input_norm, input_epsilon], [input_denominator], name=f"{prefix}/InputL2Epsilon")
        value = f"{prefix}/normalized_input"
        self.make_node("Div", [input_name, input_denominator], [value], name=f"{prefix}/InputL2Normalize")
        linear = self.make_linear(value, state, "inp", prefix)
        value = f"{prefix}/inp_gelu"
        self.make_node("Gelu", [linear], [value], name=f"{prefix}/inp/Gelu")
        # Depth counts the input and output layers as well as hidden blocks.
        for index in range(depth - 2):
            linear = self.make_linear(value, state, f"hidden.{index}", prefix)
            scale_name = f"{prefix}.norms.{index}.weight"
            bias_name = f"{prefix}.norms.{index}.bias"
            self.make_initializer(state[f"norms.{index}.weight"].detach().cpu().float().numpy(), scale_name)
            self.make_initializer(state[f"norms.{index}.bias"].detach().cpu().float().numpy(), bias_name)
            norm = f"{prefix}/norms.{index}"
            self.make_node(
                "LayerNormalization",
                [linear, scale_name, bias_name],
                [norm],
                name=f"{norm}/LayerNormalization",
                axis=-1,
                epsilon=1e-5,
            )
            value = f"{prefix}/hidden.{index}_gelu"
            self.make_node("Gelu", [norm], [value], name=f"{prefix}/hidden.{index}/Gelu")
        value = self.make_linear(value, state, "out", prefix)
        axes = f"{prefix}.l2_axes"
        epsilon = f"{prefix}.l2_epsilon"
        self.make_initializer(np.asarray([-1], dtype=np.int64), axes)
        self.make_initializer(np.asarray(1e-12, dtype=np.float32), epsilon)
        norm = f"{prefix}/l2_norm"
        self.make_node("ReduceL2", [value, axes], [norm], name=f"{prefix}/ReduceL2", keepdims=1)
        denominator = f"{prefix}/l2_denominator"
        # F.normalize uses max(norm, eps), including for a zero projection.
        self.make_node("Max", [norm, epsilon], [denominator], name=f"{prefix}/L2Epsilon")
        self.make_node("Div", [value, denominator], [output_name], name=f"{prefix}/L2Normalize")

    def make_model(self, checkpoint):
        hidden_size = int(checkpoint["hidden_size"])
        projection_dim = int(checkpoint["projection_dim"])
        width = int(checkpoint["cfg"]["width"])
        depth = int(checkpoint["cfg"]["depth"])
        for input_name, output_name, key, prefix in (
            ("state_hidden_states", "state_embedding", "state_head", "state"),
            ("action_hidden_states", "action_embedding", "action_head", "action"),
        ):
            self.model.graph.inputs.append(self.make_value(input_name, ir.DataType.FLOAT, [None, hidden_size]))
            self.make_head(input_name, output_name, checkpoint[key], prefix, hidden_size, width, projection_dim, depth)
            self.model.graph.outputs.append(self.make_value(output_name, ir.DataType.FLOAT, [None, projection_dim]))
        logit_scale = float(checkpoint["logit_scale"].item())
        if not math.isfinite(logit_scale):
            raise ValueError("CLM logit_scale must be finite")
        scale = math.exp(min(logit_scale, math.log(100.0)))
        self.make_initializer(np.asarray(scale, dtype=np.float32), "effective_logit_scale")
        self.model.graph.outputs.append(self.make_value("effective_logit_scale", ir.DataType.FLOAT, []))
        self.model.metadata_props.update(
            {
                "model_type": "clm-v0.1-8b",
                "pooling": "last_token",
                "scoring": "effective_logit_scale * dot(state_embedding, action_embedding)",
                "artifact_revision": clm_artifact_revision(),
            }
        )


def append_linear(nodes, initializers, value, state, prefix, node_prefix):
    """Compatibility wrapper emitting one affine layer through Model."""
    builder = CLMModel()
    builder.values[value.name] = value
    output = builder.make_linear(value.name, state, prefix, node_prefix)
    emitted_nodes = list(builder.model.graph)
    builder.model.graph.remove(emitted_nodes)
    nodes.extend(emitted_nodes)
    initializers.extend(builder.model.graph.initializers.values())
    builder.model.graph.initializers.clear()
    return builder.make_value(output)


def build_head_nodes(input_value, output_name, state, prefix, hidden_size, width, projection_dim, depth):
    """Compatibility wrapper returning a head's inherited-emitter nodes and weights."""
    builder = CLMModel()
    builder.values[input_value.name] = input_value
    builder.make_head(input_value.name, output_name, state, prefix, hidden_size, width, projection_dim, depth)
    nodes = list(builder.model.graph)
    builder.model.graph.remove(nodes)
    initializers = list(builder.model.graph.initializers.values())
    builder.model.graph.initializers.clear()
    return nodes, initializers


def build_clm_model(checkpoint: dict) -> ir.Model:
    """Build the two-head CLM IR model from an already loaded checkpoint.

    Inputs are last-token state/action hidden vectors; outputs are normalized
    embeddings and the capped effective logit scale. Invalid head schemas raise
    ``ValueError``.
    """
    return CLMModel(checkpoint).model


def export_clm_components(options, output_root: Path) -> dict:
    """Validate pins, export ``clm_heads.onnx``, and return manifest metadata.

    ``options.model_source`` is loaded under the released CLM schema. An
    incorrect artifact revision or checkpoint fails with ``ValueError``. The
    manifest leaves last-token pooling and final scoring to the caller.
    """
    revision = clm_artifact_revision()
    if options.artifact_revision != revision:
        raise ValueError(f"CLM artifact_revision must be {revision}")
    checkpoint = load_clm_checkpoint(options.model_source)
    filename = "clm_heads.onnx"
    builder = CLMModel(checkpoint)
    builder.save_onnx_model(builder.model, output_root / filename)
    return {
        "schema_version": 1,
        "model_type": "clm-v0.1-8b",
        "provenance": {
            "artifact_revision": options.artifact_revision,
            "base_model": "Qwen/Qwen3-8B",
            "base_revision": options.base_revision,
        },
        "orchestration": {
            "pooling": "last_token",
            "scoring": "effective_logit_scale * dot(state_embedding, action_embedding)",
            "owned_by": "caller",
        },
        "components": {
            "backbone": {
                "role": "backbone",
                "filename": options.backbone_filename,
                "outputs": {"hidden_states": "hidden_states"},
            },
            "clm_heads": {
                "role": "head",
                "filename": filename,
                "inputs": {
                    "state_hidden_states": "state_hidden_states",
                    "action_hidden_states": "action_hidden_states",
                },
                "outputs": {
                    "state_embedding": "state_embedding",
                    "action_embedding": "action_embedding",
                    "effective_logit_scale": "effective_logit_scale",
                },
            },
        },
    }
