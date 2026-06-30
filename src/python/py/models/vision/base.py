# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
VisionModel base class for exporting ViT-family vision encoders to ONNX.

This module mirrors the structure of builders/base.py but is adapted for
vision encoders (ViT, SigLIP, DINOv2, CLIP, etc.) rather than LLM decoders.

Key differences from builders/base.py:
  - Inputs:  pixel_values [B, C, H, W]  (no token IDs, no KV cache)
  - Outputs: image_features [B, N, D]   (no logits, no KV cache)
  - Layers:  EncoderLayer (bidirectional MHA, no RoPE, no KV cache)
  - Patch embedding: Conv2d → reshape → optional CLS prepend → optional pos embed
  - MLP: FC1/GELU/FC2 by default (not SwiGLU)
  - LayerNorm: standard LayerNorm with bias (not RMSNorm)
"""
from __future__ import annotations

import ast
import json
import math
import os
from collections.abc import Sequence

import numpy as np
import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import TorchTensor, to_torch_dtype
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, QuantFormat
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel, GenerationConfig


def parse_hf_token(hf_token):
    if hf_token.lower() in {"false", "0"}:
        return None
    if hf_token.lower() in {"true", "1"}:
        return True
    return hf_token


class VisionModel:
    """
    Base class for exporting ViT-family vision encoders to ONNX.

    Subclasses override attribute dicts in __init__ (before or after super())
    and/or override make_* methods to handle model-specific differences.

    The export pipeline (called by builder.py) is:
        make_model(input_path)   → build ONNX graph in memory
        save_model(output_dir)   → quantize (INT4) + write .onnx + .onnx.data
        make_genai_config(...)   → write vision section of genai_config.json
        save_processing(...)     → write preprocessor_config.json
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # ── Dimensions ────────────────────────────────────────────────────
        self.image_size = getattr(config, "image_size", 224)
        self.patch_size = getattr(config, "patch_size", 16)
        # num_channels: standard ViT uses "num_channels", Qwen-VL uses "in_channels"
        self.num_channels = (
            getattr(config, "num_channels", None)
            or getattr(config, "in_channels", 3)
        )
        self.hidden_size = config.hidden_size
        # num_attention_heads: standard ViT uses "num_attention_heads", Qwen-VL uses "num_heads"
        self.num_attn_heads = (
            getattr(config, "num_attention_heads", None)
            or getattr(config, "num_heads", None)
        )
        if self.num_attn_heads is None:
            raise AttributeError(
                f"Cannot find num_attention_heads or num_heads in vision config: {type(config).__name__}"
            )
        self.num_kv_heads = (
            getattr(config, "num_key_value_heads", None)
            or self.num_attn_heads
        )
        self.head_size = (
            config.head_dim
            if hasattr(config, "head_dim") and config.head_dim is not None
            else self.hidden_size // self.num_attn_heads
        )
        # num_hidden_layers: standard ViT uses "num_hidden_layers", Qwen-VL uses "depth"
        self.num_layers = (
            int(extra_options["num_hidden_layers"])
            if "num_hidden_layers" in extra_options
            else getattr(config, "num_hidden_layers", None)
            or getattr(config, "depth", None)
        )
        if self.num_layers is None:
            raise AttributeError(
                f"Cannot find num_hidden_layers or depth in vision config: {type(config).__name__}"
            )
        # intermediate_size: standard ViT/SigLIP uses "intermediate_size"
        # DINOv2 uses "mlp_ratio" (intermediate_size = hidden_size * mlp_ratio)
        self.intermediate_size = (
            getattr(config, "intermediate_size", None)
            or int(self.hidden_size * getattr(config, "mlp_ratio", 4))
        )
        self.activation = getattr(config, "hidden_act", "gelu")

        # Number of patches (spatial grid)
        if isinstance(self.image_size, (list, tuple)):
            h, w = self.image_size[0], self.image_size[1]
        else:
            h = w = self.image_size
        self.num_patches = (h // self.patch_size) * (w // self.patch_size)

        # ── Model metadata ─────────────────────────────────────────────────
        self.model_name_or_path = config._name_or_path
        self.model_type = config.architectures[0] if hasattr(config, "architectures") and config.architectures else config.model_type
        self.io_dtype = ir.DataType(io_dtype)
        self.onnx_dtype = ir.DataType(onnx_dtype)
        self.ep = ep
        self.cache_dir = cache_dir
        self.filename = extra_options.get("filename", "vision.onnx")
        self.hf_token = parse_hf_token(extra_options.get("hf_token", "true"))
        self.hf_remote = extra_options.get("hf_remote", True)
        self.extra_options = extra_options

        # ── ONNX IR graph ──────────────────────────────────────────────────
        self.graph = ir.Graph(
            inputs=(),
            outputs=(),
            nodes=(),
            opset_imports={"": 21, "com.microsoft": 1},
            name="vision_graph",
        )
        self.model = ir.Model(self.graph, ir_version=10, producer_name="onnxruntime-genai")
        self.values = {}
        self.node_names = set()

        # ── EP attributes ──────────────────────────────────────────────────
        self.ep_attrs = {
            "cpu": {},
            "cuda": {"enable_cuda_graph": "0"},
            "dml": {},
            "webgpu": {},
            "trt-rtx": {},
        }

        # ── Patch embedding attributes ─────────────────────────────────────
        # Subclasses override these to customize patch embedding behaviour.
        self.patch_embed_attrs = {
            "has_cls_token": True,       # Prepend CLS token (ViT, DINOv2, CLIP)
            "has_pos_embed": True,       # Add learned absolute positional embedding
            "pos_embed_type": "absolute",  # "absolute" | "2d_rope" | "none"
        }

        # ── LayerNorm attributes ───────────────────────────────────────────
        # ViT uses standard LayerNorm (with bias), not RMSNorm.
        epsilon = getattr(config, "layer_norm_eps", getattr(config, "rms_norm_eps", 1e-6))
        self.layernorm_attrs = {
            "simple": False,            # False = LayerNorm with bias (ViT default)
            "first_layernorm": True,    # First layer uses LayerNorm; rest use SkipLayerNorm
            "last_layernorm": False,
            "root_input": "",           # Current residual stream tensor name
            "skip_input": "",           # New contribution (attn/MLP output)
            "output_0": "",             # Normalized output
            "output_3": "",             # Raw residual sum (for SkipLayerNorm)
            "add_offset": 0,
            "epsilon": epsilon,
            "cast": {
                "use_fp32": False,
                "root_input": False,
                "skip_input": False,
                "output_0": False,
                "output_3": False,
            },
        }

        # ── Attention attributes ───────────────────────────────────────────
        # Vision encoders always use full bidirectional MHA — no GQA, no RoPE,
        # no KV cache.
        self.attention_attrs = {
            "op_type": "MultiHeadAttention",
            "use_rope_in_attn": False,      # No RoPE in vision attention
            "use_packed_matmul": True,      # Pack Q/K/V into one MatMul when possible
            "q_norm": False,                # QK-norm (SigLIP2, Qwen3-VL ViT)
            "k_norm": False,
            "scale": 1.0 / math.sqrt(self.head_size),
            "unidirectional": False,        # Bidirectional (full self-attention)
            "q_path": "",
            "k_path": "",
            "v_path": "",
        }

        # ── MLP attributes ─────────────────────────────────────────────────
        # ViT default: FC1 → GELU → FC2  (use_fc=True)
        # SigLIP2 / Qwen-VL ViT: gate_proj + up_proj → SwiGLU → down_proj (use_proj=True)
        self.mlp_attrs = {
            "use_proj": False,   # SwiGLU style
            "use_fc": True,      # FC1/GELU/FC2 style (ViT default)
            "output_0": "",
        }

        # ── Quantization attributes ────────────────────────────────────────
        self.int4_block_size = extra_options.get("int4_block_size", 32)
        self.quant_attrs = {
            "int4": {
                "accuracy_level": int(extra_options.get("int4_accuracy_level", 4 if ep in ["cpu", "webgpu"] else 0)),
                "qdq_block_size": int(self.int4_block_size),
                "is_symmetric": extra_options.get("int4_is_symmetric", True),
                "op_types_to_quantize": extra_options.get("int4_op_types_to_quantize", ("MatMul",)),
                "nodes_to_exclude": extra_options.get("int4_nodes_to_exclude", []),
                "algo_config": None,  # default RTN
            },
            "use_qdq": extra_options.get("use_qdq", False),
        }

        # ── I/O definitions ────────────────────────────────────────────────
        # Base: pixel_values in, image_features out.
        # Subclasses add extra inputs (e.g. image_grid_thw for Qwen-VL).
        self.input_names = {
            "pixel_values": "pixel_values",
        }
        self.input_types = {
            "pixel_values": self.io_dtype,
        }
        self.input_shapes = {
            "pixel_values": ["batch_size", self.num_channels, h, w],
        }

        # seq_len = num_patches + 1 (CLS) if has_cls_token, else num_patches
        # This is updated after patch_embed_attrs is set by subclass.
        # We use a symbolic dim here; concrete models override if needed.
        self.output_names = {
            "image_features": "image_features",
        }
        self.output_types = {
            "image_features": self.io_dtype,
        }
        self.output_shapes = {
            "image_features": ["batch_size", "num_image_tokens", self.hidden_size],
        }

    # ── Utility: dtype string ──────────────────────────────────────────────

    def to_str_dtype(self, dtype: ir.DataType) -> str:
        return dtype.name

    # ── Low-level graph primitives (mirrors builders/base.py) ─────────────

    def make_value(self, name, dtype=None, shape=None) -> ir.Value:
        if name == "":
            return ir.Value(name="")
        value = self.values.setdefault(name, ir.Value(name=name))
        if dtype is not None:
            value.dtype = ir.DataType(dtype)
        if shape is not None:
            value.shape = ir.Shape(shape)
        return value

    def make_node(self, op_type, inputs: Sequence[str], outputs: Sequence[str], *, name: str, domain="", **kwargs):
        assert name, "Node name must be provided"
        if name in self.node_names:
            return
        for input_name in inputs:
            if input_name.startswith("/model/constants") and input_name not in self.node_names:
                self.make_constant(input_name)
        input_values = [self.make_value(n) for n in inputs]
        output_values = [self.make_value(n) for n in outputs]
        node = ir.node(op_type, inputs=input_values, attributes=kwargs, domain=domain, outputs=output_values, name=name)
        self.model.graph.append(node)
        self.node_names.add(name)

    def make_constant(self, name):
        """Make a scalar/vector constant node. Name format: /model/constants/{dtype}/{value}"""
        path = name.split("/")
        onnx_dtype = ir.DataType[path[-2]]
        num = ast.literal_eval(path[-1])
        assert isinstance(num, (int, float, list, tuple)), f"Invalid constant value: {num}"
        tensor = ir.tensor(num, dtype=onnx_dtype, name=name)
        node_name = name.replace("constants", "constant_nodes")
        self.make_node("Constant", inputs=[], outputs=[name], name=node_name, value=tensor)
        self.make_value(name, onnx_dtype, shape=[])

    def make_initializer(self, tensor, /, name: str, to: ir.DataType | None = None):
        if to is not None:
            def tensor_func():
                nonlocal tensor
                tensor = tensor.to(to_torch_dtype(to))
                return TorchTensor(tensor, name=name)
            ir_tensor = ir.LazyTensor(tensor_func, dtype=to, shape=ir.Shape(tensor.shape), name=name)
        elif isinstance(tensor, torch.nn.parameter.Parameter):
            ir_tensor = TorchTensor(tensor, name=name)
        else:
            ir_tensor = ir.tensor(tensor, name=name)
        value = self.make_value(name, ir_tensor.dtype, ir_tensor.shape)
        value.const_value = ir_tensor
        self.model.graph.register_initializer(value)

    # ── Primitive op builders ──────────────────────────────────────────────

    def make_gather(self, name, inputs, dtype, shape, axis=0):
        output = f"{name}/output_0"
        self.make_node("Gather", inputs=inputs, outputs=[output], name=name, axis=axis)
        self.make_value(output, dtype, shape=shape)

    def make_reshape(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Reshape", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_shape(self, name, root_input, shape):
        output = f"{name}/output_0"
        self.make_node("Shape", inputs=[root_input], outputs=[output], name=name)
        self.make_value(output, ir.DataType.INT64, shape=shape)

    def make_unsqueeze(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Unsqueeze", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_squeeze(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Squeeze", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_concat(self, name, inputs, dtype, shape, axis=0):
        output = f"{name}/output_0"
        self.make_node("Concat", inputs=inputs, outputs=[output], name=name, axis=axis)
        self.make_value(output, dtype, shape=shape)

    def make_cast(self, name, root_input, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Cast", inputs=[root_input], outputs=[output], name=name, to=dtype)
        self.make_value(output, dtype, shape=shape)

    def make_add(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Add", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_mul(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Mul", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_transpose(self, name, root_input, dtype, shape, perm):
        output = f"{name}/output_0"
        self.make_node("Transpose", inputs=[root_input], outputs=[output], name=name, perm=perm)
        self.make_value(output, dtype, shape=shape)

    def make_conv(self, name, inputs, dtype, shape, **kwargs):
        output = f"{name}/output_0"
        self.make_node("Conv", inputs=inputs, outputs=[output], name=name, **kwargs)
        self.make_value(output, dtype, shape=shape)

    def make_expand(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Expand", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_flatten(self, name, root_input, dtype, shape, axis=1):
        output = f"{name}/output_0"
        self.make_node("Flatten", inputs=[root_input], outputs=[output], name=name, axis=axis)
        self.make_value(output, dtype, shape=shape)

    def make_cos(self, name, root_input, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Cos", inputs=[root_input], outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_sin(self, name, root_input, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Sin", inputs=[root_input], outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    # ── MatMul builder ─────────────────────────────────────────────────────

    def make_matmul(self, matmul, basename, root_input, **kwargs):
        """Emit a MatMul (float) or MatMulNBits (int4) node for a Linear layer."""
        if self.onnx_dtype in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT}:
            return self._make_matmul_float(matmul, basename, root_input, **kwargs)
        elif self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4}:
            # INT4 is quantized post-hoc by to_int4() in save_model().
            # Emit float MatMul now; MatMulNBitsQuantizer will replace it.
            return self._make_matmul_float(matmul, basename, root_input, **kwargs)
        else:
            raise NotImplementedError(f"Unsupported onnx_dtype: {self.onnx_dtype}")

    def _make_matmul_float(self, matmul, name, root_input, **kwargs):
        weight_name = name[1:].replace("/", ".") + ".weight"
        self.make_initializer(matmul.weight.T, weight_name, to=self.io_dtype)

        last_dim = matmul.weight.shape[0]
        seq_dim = kwargs.get("seq_dim", "num_image_tokens")

        output = f"{name}/output_0"
        self.make_node("MatMul", inputs=[root_input, weight_name], outputs=[output], name=name)
        self.make_value(output, self.io_dtype, shape=["batch_size", seq_dim, last_dim])

        if matmul.bias is not None and torch.count_nonzero(matmul.bias) > 0:
            bias_name = name[1:].replace("/", ".") + ".bias"
            self.make_initializer(matmul.bias, bias_name, to=self.io_dtype)
            add_name = f"{name}/Add"
            add_output = f"{add_name}/output_0"
            self.make_node("Add", inputs=[output, bias_name], outputs=[add_output], name=add_name)
            self.make_value(add_output, self.io_dtype, shape=["batch_size", seq_dim, last_dim])
            return add_name

        return name

    def make_add_bias(self, bias, name, root_input):
        """Add a bias tensor to root_input."""
        bias_name = name[1:].replace("/", ".") + ".bias"
        self.make_initializer(bias, bias_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node("Add", inputs=[root_input, bias_name], outputs=[output], name=name)
        self.make_value(output, self.io_dtype, shape=["batch_size", "num_image_tokens", self.hidden_size])

    # ── LayerNorm builder ──────────────────────────────────────────────────

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        """
        Emit a LayerNorm or SkipLayerNorm node.

        For vision encoders:
          - simple=False  → standard LayerNorm with bias
          - skip=False    → first layer (no residual yet)
          - skip=True     → subsequent layers (fuse residual add + norm)
        """
        root_input = self.layernorm_attrs["root_input"]
        skip_input = self.layernorm_attrs["skip_input"]

        # Weight and bias initializers
        weight_name = f"vision_model.encoder.layers.{layer_id}.{location}_layernorm.weight"
        bias_name = f"vision_model.encoder.layers.{layer_id}.{location}_layernorm.bias"
        self.make_initializer(layernorm.weight, weight_name, to=self.io_dtype)
        if not simple and hasattr(layernorm, "bias") and layernorm.bias is not None:
            self.make_initializer(layernorm.bias, bias_name, to=self.io_dtype)

        # Build inputs list
        if skip:
            inputs = [root_input, skip_input, weight_name]
        else:
            inputs = [root_input, weight_name]
        if not simple and hasattr(layernorm, "bias") and layernorm.bias is not None:
            inputs.append(bias_name)

        # Op type
        op_type = f"{'Skip' if skip else ''}{'Simplified' if simple else ''}LayerNormalization"
        name = f"/vision_model/encoder/layers.{layer_id}/{location}_layernorm/{'Skip' if skip else ''}LayerNorm"
        kwargs = {"epsilon": self.layernorm_attrs["epsilon"]}
        if not skip:
            kwargs.update({"axis": -1, "stash_type": 1})

        # Output names
        output_0 = f"/vision_model/encoder/layers.{layer_id}/{location}_layernorm/output_0"
        output_3 = f"/vision_model/encoder/layers.{layer_id}/{location}_layernorm/output_3"
        outputs = [output_0, "", "", output_3] if skip and not self.layernorm_attrs["last_layernorm"] else [output_0]

        self.make_node(op_type, inputs=inputs, outputs=outputs, name=name,
                       domain=("com.microsoft" if skip else None), **kwargs)
        self.make_value(output_0, self.io_dtype, shape=["batch_size", "num_image_tokens", self.hidden_size])
        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.make_value(output_3, self.io_dtype, shape=["batch_size", "num_image_tokens", self.hidden_size])

        # Update attrs
        self.layernorm_attrs["output_0"] = output_0
        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.layernorm_attrs["output_3"] = output_3
            self.layernorm_attrs["root_input"] = output_3

    # ── Attention builder ──────────────────────────────────────────────────

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """
        Emit the full attention subgraph for one ViT encoder layer.

        Graph:
            root_input
           /     |     \\
        Q_MatMul K_MatMul V_MatMul
           |        |        |
         [Q_Add] [K_Add]  [V_Add]   (bias, if present)
           |        |        |
        [Q_Norm] [K_Norm]           (QK-norm, if q_norm/k_norm=True)
           \\       |       /
            MultiHeadAttention      (com.microsoft, no KV cache)
                   |
               O_MatMul
                   |
               [O_Add]
        """
        self._make_attention_qkv(layer_id, attention, root_input)
        self._make_attention_qk_norm(layer_id, attention)
        self._make_attention_op(layer_id, attention, **kwargs)
        self._make_attention_output(layer_id, attention)

    def _make_attention_qkv(self, layer_id, attention, root_input):
        """Emit Q, K, V projection MatMuls."""
        if self.attention_attrs["use_packed_matmul"]:
            # Pack Q/K/V into one MatMul for efficiency
            qkv_name = f"/vision_model/encoder/layers.{layer_id}/attn/qkv_proj/MatMul"
            q_size = self.num_attn_heads * self.head_size
            kv_size = self.num_kv_heads * self.head_size

            # Stack weights: [q_size + kv_size + kv_size, hidden_size]
            qkv_weight = torch.cat([
                attention.query.weight if hasattr(attention, "query") else attention.q_proj.weight,
                attention.key.weight if hasattr(attention, "key") else attention.k_proj.weight,
                attention.value.weight if hasattr(attention, "value") else attention.v_proj.weight,
            ], dim=0)
            weight_name = qkv_name[1:].replace("/", ".") + ".weight"
            self.make_initializer(qkv_weight.T, weight_name, to=self.io_dtype)

            output = f"{qkv_name}/output_0"
            self.make_node("MatMul", inputs=[root_input, weight_name], outputs=[output], name=qkv_name)
            self.make_value(output, self.io_dtype,
                            shape=["batch_size", "num_image_tokens", q_size + kv_size + kv_size])

            # Handle packed bias
            q_proj = attention.query if hasattr(attention, "query") else attention.q_proj
            k_proj = attention.key if hasattr(attention, "key") else attention.k_proj
            v_proj = attention.value if hasattr(attention, "value") else attention.v_proj

            q_bias = q_proj.bias
            k_bias = k_proj.bias
            v_bias = v_proj.bias
            any_bias = any(
                b is not None and torch.count_nonzero(b) > 0
                for b in [q_bias, k_bias, v_bias]
            )
            if any_bias:
                # Pad missing biases with zeros
                def _bias_or_zeros(b, size):
                    if b is None or torch.count_nonzero(b) == 0:
                        return torch.zeros(size, dtype=torch.float32)
                    return b.float()
                qkv_bias = torch.cat([
                    _bias_or_zeros(q_bias, q_size),
                    _bias_or_zeros(k_bias, kv_size),
                    _bias_or_zeros(v_bias, kv_size),
                ])
                bias_name = qkv_name[1:].replace("/", ".") + ".bias"
                self.make_initializer(qkv_bias, bias_name, to=self.io_dtype)
                add_name = f"{qkv_name}/Add"
                add_output = f"{add_name}/output_0"
                self.make_node("Add", inputs=[output, bias_name], outputs=[add_output], name=add_name)
                self.make_value(add_output, self.io_dtype,
                                shape=["batch_size", "num_image_tokens", q_size + kv_size + kv_size])
                self.attention_attrs["q_path"] = add_output
            else:
                self.attention_attrs["q_path"] = output

            # k_path and v_path are empty when packed (MHA op handles splitting)
            self.attention_attrs["k_path"] = ""
            self.attention_attrs["v_path"] = ""
        else:
            # Separate Q, K, V MatMuls
            q_proj = attention.query if hasattr(attention, "query") else attention.q_proj
            k_proj = attention.key if hasattr(attention, "key") else attention.k_proj
            v_proj = attention.value if hasattr(attention, "value") else attention.v_proj

            q_name = self.make_matmul(q_proj, f"/vision_model/encoder/layers.{layer_id}/attn/q_proj/MatMul", root_input)
            k_name = self.make_matmul(k_proj, f"/vision_model/encoder/layers.{layer_id}/attn/k_proj/MatMul", root_input)
            v_name = self.make_matmul(v_proj, f"/vision_model/encoder/layers.{layer_id}/attn/v_proj/MatMul", root_input)
            self.attention_attrs["q_path"] = f"{q_name}/output_0"
            self.attention_attrs["k_path"] = f"{k_name}/output_0"
            self.attention_attrs["v_path"] = f"{v_name}/output_0"

    def _make_attention_qk_norm(self, layer_id, attention):
        """Emit optional QK-norm (RMSNorm) after Q and K projections."""
        if not (self.attention_attrs["q_norm"] or self.attention_attrs["k_norm"]):
            return

        q_size = self.num_attn_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size

        if self.attention_attrs["q_norm"] and hasattr(attention, "q_norm"):
            q_norm_weight = f"vision_model.encoder.layers.{layer_id}.attn.q_norm.weight"
            self.make_initializer(attention.q_norm.weight, q_norm_weight, to=self.io_dtype)
            q_norm_name = f"/vision_model/encoder/layers.{layer_id}/attn/q_norm/SimplifiedLayerNorm"
            q_norm_output = f"{q_norm_name}/output_0"
            self.make_node(
                "SimplifiedLayerNormalization",
                inputs=[self.attention_attrs["q_path"], q_norm_weight],
                outputs=[q_norm_output],
                name=q_norm_name,
                axis=-1,
                epsilon=1e-6,
                stash_type=1,
            )
            self.make_value(q_norm_output, self.io_dtype, shape=["batch_size", "num_image_tokens", q_size])
            self.attention_attrs["q_path"] = q_norm_output

        if self.attention_attrs["k_norm"] and hasattr(attention, "k_norm"):
            k_norm_weight = f"vision_model.encoder.layers.{layer_id}.attn.k_norm.weight"
            self.make_initializer(attention.k_norm.weight, k_norm_weight, to=self.io_dtype)
            k_norm_name = f"/vision_model/encoder/layers.{layer_id}/attn/k_norm/SimplifiedLayerNorm"
            k_norm_output = f"{k_norm_name}/output_0"
            self.make_node(
                "SimplifiedLayerNormalization",
                inputs=[self.attention_attrs["k_path"], k_norm_weight],
                outputs=[k_norm_output],
                name=k_norm_name,
                axis=-1,
                epsilon=1e-6,
                stash_type=1,
            )
            self.make_value(k_norm_output, self.io_dtype, shape=["batch_size", "num_image_tokens", kv_size])
            self.attention_attrs["k_path"] = k_norm_output

    def _make_attention_op(self, layer_id, attention):
        """Emit the MultiHeadAttention op (com.microsoft)."""
        attn_name = f"/vision_model/encoder/layers.{layer_id}/attn/MultiHeadAttention"
        inputs = [
            self.attention_attrs["q_path"],
            self.attention_attrs["k_path"],
            self.attention_attrs["v_path"],
            "",   # bias (already added to Q/K/V above)
            "",   # attn_mask (none for vision — full bidirectional)
            "",   # add_qk
            "",   # past_k (no KV cache)
            "",   # past_v (no KV cache)
        ]
        output = f"{attn_name}/output_0"
        outputs = [output, "", ""]  # no present_k, present_v

        self.make_node(
            "MultiHeadAttention",
            inputs=inputs,
            outputs=outputs,
            name=attn_name,
            domain="com.microsoft",
            num_heads=self.num_attn_heads,
            scale=self.attention_attrs["scale"],
            unidirectional=0,  # bidirectional
        )
        self.make_value(output, self.io_dtype,
                        shape=["batch_size", "num_image_tokens", self.num_attn_heads * self.head_size])
        self.attention_attrs["attn_output"] = output

    def _make_attention_output(self, layer_id, attention):
        """
        Emit the output projection MatMul (and optional bias).

        Uses self._attn_out_module (set by make_layer) which may be:
          - ViTSelfOutput  (HF ViT): has .dense
          - SiglipAttention (SigLIP): has .out_proj
          - CLIPAttention (CLIP): has .out_proj
          - Dinov2SelfAttention (DINOv2): has .out_proj or .projection
        """
        # Use the output module stored by make_layer (handles HF ViT nested structure)
        out_module = getattr(self, "_attn_out_module", attention)

        out_proj = (
            getattr(out_module, "dense", None)        # HF ViT: ViTSelfOutput.dense
            or getattr(out_module, "out_proj", None)  # SigLIP/CLIP/DINOv2
            or getattr(out_module, "projection", None)
            or getattr(out_module, "o_proj", None)
        )
        if out_proj is None:
            raise AttributeError(
                f"Cannot find output projection in attention layer {layer_id}. "
                f"Module type: {type(out_module).__name__}, "
                f"attrs: {[n for n, _ in out_module.named_children()]}"
            )
        o_name = self.make_matmul(
            out_proj,
            f"/vision_model/encoder/layers.{layer_id}/attn/out_proj/MatMul",
            self.attention_attrs["attn_output"],
        )
        # The output of the attention subgraph becomes the skip_input for the next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{o_name}/output_0"

    # ── MLP builder ────────────────────────────────────────────────────────

    def make_mlp(self, layer_id, mlp, root_input):
        """Dispatch to FC1/FC2 or SwiGLU MLP based on mlp_attrs."""
        if self.mlp_attrs["use_fc"]:
            self._make_mlp_fc(layer_id, mlp, root_input)
        elif self.mlp_attrs["use_proj"]:
            self._make_mlp_proj(layer_id, mlp, root_input)
        else:
            raise NotImplementedError("mlp_attrs must have use_fc=True or use_proj=True")

    def _make_mlp_fc(self, layer_id, mlp, root_input):
        """
        FC1 → activation → FC2  (standard ViT MLP)

        HF attribute names vary:
          ViT:    intermediate.dense (fc1) + output.dense (fc2) — split across two modules
          SigLIP: mlp.fc1, mlp.fc2
          CLIP:   mlp.fc1, mlp.fc2
          DINOv2: mlp.fc1, mlp.fc2

        For HF ViT, make_layer() stores the fc2 module in self._mlp_output_module.
        """
        # fc1: try standard names first, then HF ViT's intermediate.dense
        fc1 = (
            getattr(mlp, "fc1", None)
            or getattr(mlp, "linear_fc1", None)   # Qwen-VL ViT: mlp.linear_fc1
            or getattr(mlp, "dense", None)         # HF ViT: ViTIntermediate.dense
        )

        # fc2: try standard names first, then use the stored output module
        mlp_out = getattr(self, "_mlp_output_module", None)
        fc2 = (
            getattr(mlp, "fc2", None)
            or getattr(mlp, "linear_fc2", None)    # Qwen-VL ViT: mlp.linear_fc2
            or (getattr(mlp_out, "dense", None) if mlp_out is not None else None)  # HF ViT: ViTOutput.dense
        )

        if fc1 is None:
            raise AttributeError(f"Cannot find fc1/dense in MLP module: {type(mlp).__name__}")
        if fc2 is None:
            raise AttributeError(
                f"Cannot find fc2 in MLP module: {type(mlp).__name__}. "
                f"_mlp_output_module={type(mlp_out).__name__ if mlp_out else None}"
            )

        fc1_name = self.make_matmul(fc1, f"/vision_model/encoder/layers.{layer_id}/mlp/fc1/MatMul", root_input)
        act_output = self._make_mlp_activation(layer_id, f"{fc1_name}/output_0")
        fc2_name = self.make_matmul(fc2, f"/vision_model/encoder/layers.{layer_id}/mlp/fc2/MatMul", act_output)
        self.layernorm_attrs["skip_input"] = f"{fc2_name}/output_0"

    def _make_mlp_proj(self, layer_id, mlp, root_input):
        """
        gate_proj + up_proj → SwiGLU → down_proj  (SigLIP2, Qwen-VL ViT)
        """
        gate_name = self.make_matmul(mlp.gate_proj, f"/vision_model/encoder/layers.{layer_id}/mlp/gate_proj/MatMul", root_input)
        up_name = self.make_matmul(mlp.up_proj, f"/vision_model/encoder/layers.{layer_id}/mlp/up_proj/MatMul", root_input)

        # SiLU activation on gate
        act_name = f"/vision_model/encoder/layers.{layer_id}/mlp/act_fn/Silu"
        act_output = f"{act_name}/output_0"
        self.make_node("Silu", inputs=[f"{gate_name}/output_0"], outputs=[act_output], name=act_name, domain="com.microsoft")
        self.make_value(act_output, self.io_dtype, shape=["batch_size", "num_image_tokens", self.intermediate_size])

        # Mul(act, up)
        mul_name = f"/vision_model/encoder/layers.{layer_id}/mlp/Mul"
        mul_output = f"{mul_name}/output_0"
        self.make_node("Mul", inputs=[act_output, f"{up_name}/output_0"], outputs=[mul_output], name=mul_name)
        self.make_value(mul_output, self.io_dtype, shape=["batch_size", "num_image_tokens", self.intermediate_size])

        down_name = self.make_matmul(mlp.down_proj, f"/vision_model/encoder/layers.{layer_id}/mlp/down_proj/MatMul", mul_output)
        self.layernorm_attrs["skip_input"] = f"{down_name}/output_0"

    def _make_mlp_activation(self, layer_id, root_input):
        """Emit the activation function for FC-style MLP. Returns output tensor name."""
        act = self.activation.lower()
        act_name = f"/vision_model/encoder/layers.{layer_id}/mlp/act_fn"

        if act in {"gelu", "gelu_new", "gelu_pytorch_tanh"}:
            approx = "tanh" if act in {"gelu_new", "gelu_pytorch_tanh"} else "none"
            output = f"{act_name}/output_0"
            self.make_node("Gelu", inputs=[root_input], outputs=[output], name=act_name, approximate=approx)
            self.make_value(output, self.io_dtype, shape=["batch_size", "num_image_tokens", self.intermediate_size])
            return output
        elif act == "quick_gelu":
            output = f"{act_name}/output_0"
            self.make_node("QuickGelu", inputs=[root_input], outputs=[output], name=act_name, domain="com.microsoft")
            self.make_value(output, self.io_dtype, shape=["batch_size", "num_image_tokens", self.intermediate_size])
            return output
        elif act == "relu":
            output = f"{act_name}/output_0"
            self.make_node("Relu", inputs=[root_input], outputs=[output], name=act_name)
            self.make_value(output, self.io_dtype, shape=["batch_size", "num_image_tokens", self.intermediate_size])
            return output
        else:
            # Fallback: use standard ONNX Gelu
            output = f"{act_name}/output_0"
            self.make_node("Gelu", inputs=[root_input], outputs=[output], name=act_name, approximate="none")
            self.make_value(output, self.io_dtype, shape=["batch_size", "num_image_tokens", self.intermediate_size])
            return output

    # ── Patch embedding builder ────────────────────────────────────────────

    def make_patch_embedding(self, patch_embed):
        """
        Emit the patch embedding subgraph:
            pixel_values [B, C, H, W]
                ↓ Conv2d (patchify)
            [B, D, h, w]
                ↓ Transpose + Reshape
            [B, N, D]
                ↓ Concat CLS token (if has_cls_token)
            [B, N+1, D]
                ↓ Add positional embedding (if has_pos_embed and pos_embed_type == "absolute")
            [B, N(+1), D]  → layernorm_attrs["root_input"]

        Subclasses override this for non-standard patch embeddings (e.g. 2D RoPE).
        """
        basename = "/vision_model/patch_embedding"

        # ── 1. Conv2d patchify ─────────────────────────────────────────────
        # Attribute name varies across models:
        #   ViT:    patch_embed.projection  (ViTPatchEmbeddings)
        #   SigLIP: patch_embed.patch_embedding  (SiglipVisionEmbeddings — Conv2d directly)
        #   CLIP:   patch_embed.patch_embedding  (CLIPVisionEmbeddings)
        #   DINOv2: patch_embed.projection  (Dinov2PatchEmbeddings)
        proj = (
            getattr(patch_embed, "projection", None)
            or getattr(patch_embed, "proj", None)
            or getattr(patch_embed, "patch_embedding", None)  # SigLIP/CLIP: Conv2d directly
        )
        if proj is None:
            raise AttributeError(
                f"Cannot find projection/proj/patch_embedding in patch_embed: {type(patch_embed).__name__}. "
                f"Children: {[n for n, _ in patch_embed.named_children()]}"
            )

        conv_weight_name = "vision_model.patch_embedding.projection.weight"
        self.make_initializer(proj.weight, conv_weight_name, to=self.io_dtype)

        conv_inputs = ["pixel_values", conv_weight_name]
        if proj.bias is not None and torch.count_nonzero(proj.bias) > 0:
            conv_bias_name = "vision_model.patch_embedding.projection.bias"
            self.make_initializer(proj.bias, conv_bias_name, to=self.io_dtype)
            conv_inputs.append(conv_bias_name)

        if isinstance(self.image_size, (list, tuple)):
            h, w = self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size
        else:
            h = w = self.image_size // self.patch_size

        self.make_conv(
            f"{basename}/Conv",
            conv_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", self.hidden_size, h, w],
            kernel_shape=[self.patch_size, self.patch_size],
            strides=[self.patch_size, self.patch_size],
        )

        # ── 2. Transpose [B, D, h, w] → [B, h, w, D] → Reshape → [B, N, D] ──
        self.make_transpose(
            f"{basename}/Transpose",
            f"{basename}/Conv/output_0",
            dtype=self.io_dtype,
            shape=["batch_size", h, w, self.hidden_size],
            perm=[0, 2, 3, 1],
        )
        self.make_reshape(
            f"{basename}/Reshape",
            [f"{basename}/Transpose/output_0",
             f"/model/constants/INT64/[0, {self.num_patches}, {self.hidden_size}]"],
            dtype=self.io_dtype,
            shape=["batch_size", self.num_patches, self.hidden_size],
        )
        current = f"{basename}/Reshape/output_0"

        # ── 3. Prepend CLS token ───────────────────────────────────────────
        if self.patch_embed_attrs["has_cls_token"] and hasattr(patch_embed, "cls_token"):
            cls_name = "vision_model.patch_embedding.cls_token"
            self.make_initializer(patch_embed.cls_token, cls_name, to=self.io_dtype)
            # Expand CLS token to batch: [1, 1, D] → [B, 1, D]
            expand_shape_name = f"{basename}/cls_token/expand_shape"
            # Use Expand with shape [batch_size, 1, hidden_size]
            # We build the shape tensor dynamically
            shape_of_patches = f"{basename}/cls_token/shape_of_patches"
            self.make_shape(shape_of_patches, current, [3])
            gather_batch = f"{basename}/cls_token/gather_batch"
            self.make_gather(gather_batch, [shape_of_patches, "/model/constants/INT64/0"],
                             ir.DataType.INT64, [1], axis=0)
            concat_shape = f"{basename}/cls_token/concat_shape"
            self.make_concat(
                concat_shape,
                [f"{gather_batch}/output_0",
                 "/model/constants/INT64/[1]",
                 f"/model/constants/INT64/[{self.hidden_size}]"],
                ir.DataType.INT64, [3], axis=0,
            )
            expand_cls = f"{basename}/cls_token/Expand"
            self.make_expand(expand_cls, [cls_name, f"{concat_shape}/output_0"],
                             self.io_dtype, ["batch_size", 1, self.hidden_size])
            concat_cls = f"{basename}/cls_token/Concat"
            self.make_concat(
                concat_cls,
                [f"{expand_cls}/output_0", current],
                self.io_dtype,
                ["batch_size", self.num_patches + 1, self.hidden_size],
                axis=1,
            )
            current = f"{concat_cls}/output_0"
            seq_len = self.num_patches + 1
        else:
            seq_len = self.num_patches

        # ── 4. Add positional embedding ────────────────────────────────────
        if self.patch_embed_attrs["has_pos_embed"] and self.patch_embed_attrs["pos_embed_type"] == "absolute":
            pos_embed_module = (
                getattr(patch_embed, "position_embedding", None)
                or getattr(patch_embed, "pos_embed", None)
            )
            if pos_embed_module is not None:
                pos_weight = (
                    pos_embed_module.weight
                    if hasattr(pos_embed_module, "weight")
                    else pos_embed_module
                )
                pos_name = "vision_model.patch_embedding.position_embedding.weight"
                self.make_initializer(pos_weight, pos_name, to=self.io_dtype)
                add_pos = f"{basename}/pos_embed/Add"
                self.make_add(
                    add_pos,
                    [current, pos_name],
                    self.io_dtype,
                    ["batch_size", seq_len, self.hidden_size],
                )
                current = f"{add_pos}/output_0"

        # ── 5. Update residual stream pointer ─────────────────────────────
        self.layernorm_attrs["root_input"] = current
        self.layernorm_attrs["skip_input"] = current

    # ── Final norm and projection head ────────────────────────────────────

    def make_final_norm(self, norm_module):
        """Emit the final LayerNorm after all encoder layers."""
        weight_name = "vision_model.post_layernorm.weight"
        bias_name = "vision_model.post_layernorm.bias"
        self.make_initializer(norm_module.weight, weight_name, to=self.io_dtype)

        root_input = self.layernorm_attrs["root_input"]
        skip_input = self.layernorm_attrs["skip_input"]

        # Use SkipLayerNorm to fuse the last residual add + norm
        inputs = [root_input, skip_input, weight_name]
        if hasattr(norm_module, "bias") and norm_module.bias is not None:
            self.make_initializer(norm_module.bias, bias_name, to=self.io_dtype)
            inputs.append(bias_name)

        name = "/vision_model/post_layernorm/SkipLayerNorm"
        output_0 = "/vision_model/post_layernorm/output_0"
        self.make_node(
            "SkipLayerNormalization",
            inputs=inputs,
            outputs=[output_0],
            name=name,
            domain="com.microsoft",
            epsilon=self.layernorm_attrs["epsilon"],
        )
        self.make_value(output_0, self.io_dtype, shape=["batch_size", "num_image_tokens", self.hidden_size])
        self.layernorm_attrs["output_0"] = output_0
        self.layernorm_attrs["root_input"] = output_0

    def make_projection_head(self, proj_module):
        """
        Emit the final projection head (if present).
        Some models (e.g. CLIP) have a final linear projection after the encoder.
        Output is connected to the graph output "image_features".
        """
        root_input = self.layernorm_attrs["root_input"]
        proj_name = "/vision_model/projection/MatMul"
        weight_name = "vision_model.visual_projection.weight"
        self.make_initializer(proj_module.weight.T, weight_name, to=self.io_dtype)
        output = "image_features"
        self.make_node("MatMul", inputs=[root_input, weight_name], outputs=[output], name=proj_name)
        self.make_value(output, self.io_dtype, shape=["batch_size", "num_image_tokens", proj_module.weight.shape[0]])

    # ── Module detection ───────────────────────────────────────────────────

    def is_patch_embedding(self, module) -> bool:
        """Detect the patch embedding module."""
        name = module.__class__.__name__
        return name in {
            "ViTPatchEmbeddings",
            "SiglipVisionEmbeddings",
            "CLIPVisionEmbeddings",
            "Dinov2PatchEmbeddings",
            "PatchEmbedding",
            "VisionPatchEmbed",
        }

    def is_layer(self, module) -> bool:
        """Detect a ViT encoder layer."""
        name = module.__class__.__name__
        return (
            name.endswith("EncoderLayer")
            or name.endswith("Block")
            or name.endswith("ViTLayer")
            or name.endswith("SiglipEncoderLayer")
            or name.endswith("CLIPEncoderLayer")
            or name.endswith("Dinov2Layer")
        )

    def has_final_norm(self, module, orig_model) -> bool:
        """Detect the final LayerNorm after all encoder layers."""
        # HF ViT: model.vit.layernorm  or  model.vision_model.post_layernorm
        checks = [
            hasattr(orig_model, "vit") and hasattr(orig_model.vit, "layernorm") and module == orig_model.vit.layernorm,
            hasattr(orig_model, "vision_model") and hasattr(orig_model.vision_model, "post_layernorm") and module == orig_model.vision_model.post_layernorm,
            hasattr(orig_model, "vision_model") and hasattr(orig_model.vision_model, "encoder") and hasattr(orig_model.vision_model.encoder, "layer_norm") and module == orig_model.vision_model.encoder.layer_norm,
            hasattr(orig_model, "layernorm") and module == orig_model.layernorm,
            hasattr(orig_model, "norm") and module == orig_model.norm,
            hasattr(orig_model, "post_layernorm") and module == orig_model.post_layernorm,
        ]
        return any(checks)

    def has_projection_head(self, module, orig_model) -> bool:
        """Detect the final linear projection head (e.g. CLIP visual_projection)."""
        checks = [
            hasattr(orig_model, "visual_projection") and module == orig_model.visual_projection,
            hasattr(orig_model, "vision_projection") and module == orig_model.vision_projection,
        ]
        return any(checks)

    # ── I/O graph declaration ──────────────────────────────────────────────

    def make_inputs_and_outputs(self):
        """Declare graph inputs and outputs in the ONNX IR."""
        inputs = self.model.graph.inputs
        for key in self.input_names:
            name = self.input_names[key]
            dtype = self.input_types[key]
            shape = self.input_shapes[key]
            inputs.append(self.make_value(name, dtype=dtype, shape=shape))

        outputs = self.model.graph.outputs
        for key in self.output_names:
            name = self.output_names[key]
            dtype = self.output_types[key]
            shape = self.output_shapes[key]
            outputs.append(self.make_value(name, dtype=dtype, shape=shape))

    # ── Weight loading ─────────────────────────────────────────────────────

    def load_vision_weights(self, input_path):
        """Load the full VLM or standalone vision model from HF.

        Uses input_path (the actual local directory or HF model name passed to
        make_model) rather than self.model_name_or_path, which may be empty
        when a vision sub-config is passed (e.g. SiglipModel.vision_config).
        """
        # Prefer input_path if it's a non-empty local directory or HF model name.
        # Fall back to self.model_name_or_path for cases where input_path is empty.
        load_path = input_path if input_path else self.model_name_or_path
        extra_kwargs = {"num_hidden_layers": self.num_layers} if "num_hidden_layers" in self.extra_options else {}
        model = AutoModel.from_pretrained(
            load_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            **extra_kwargs,
        )
        return model

    # ── Main export pipeline ───────────────────────────────────────────────

    def _extract_vision_submodel(self, model):
        """
        Extract the vision encoder sub-model from a full VLM or standalone vision model.

        For standalone vision models (ViT, DINOv2), returns the model itself.
        For full VLMs (SiglipModel, CLIPModel), extracts the vision_model sub-module
        so that module iteration only walks the vision encoder, not the text encoder.

        This is critical because SiglipModel.modules() yields text encoder layers
        before vision encoder layers, causing the patch embedding to be processed
        after the encoder layers.
        """
        # SigLIP / CLIP: full model with vision_model + text_model
        if hasattr(model, "vision_model"):
            return model.vision_model
        # Already a standalone vision model
        return model

    def make_model(self, input_path):
        """
        Build the ONNX graph in memory.

        Iterates over the vision encoder's modules and dispatches each to the
        appropriate make_* method, mirroring the LLM make_model() loop.
        """
        self.make_inputs_and_outputs()
        self.weights = self.load_vision_weights(input_path)

        # Extract vision sub-model to avoid iterating text encoder modules
        # (e.g. SiglipModel has text_model + vision_model; we only want vision_model)
        vision_weights = self._extract_vision_submodel(self.weights)

        self.layer_id = 0
        for module in vision_weights.modules():
            if self.is_patch_embedding(module):
                print("Reading patch embedding")
                self.make_patch_embedding(module)

            elif self.is_layer(module) and self.layer_id < self.num_layers:
                print(f"Reading vision encoder layer {self.layer_id}")
                self.make_layer(self.layer_id, module)
                self.layer_id += 1

            elif self.layer_id == self.num_layers and self.has_final_norm(module, vision_weights):
                print("Reading final norm")
                self.make_final_norm(module)

            elif self.layer_id == self.num_layers and self.has_projection_head(module, vision_weights):
                print("Reading projection head")
                self.make_projection_head(module)

        # If no projection head was found, connect final norm output to graph output
        if self.layernorm_attrs["root_input"] != "image_features":
            final_output = self.layernorm_attrs["root_input"]
            # Rename the final tensor to "image_features" by adding an Identity node
            identity_name = "/vision_model/output/Identity"
            self.make_node("Identity", inputs=[final_output], outputs=["image_features"], name=identity_name)
            self.make_value("image_features", self.io_dtype, shape=["batch_size", "num_image_tokens", self.hidden_size])

        del self.weights

    def make_layer(self, layer_id, layer):
        """
        Emit one ViT encoder block:
            layernorm_before → attention → layernorm_after → mlp

        Mirrors LLM make_layer() but:
          - No KV cache
          - No RoPE
          - Bidirectional attention
          - Standard LayerNorm (not RMSNorm)
        """
        # ── Detect sub-module names (vary across HF implementations) ──────
        ln_before = (
            getattr(layer, "layernorm_before", None)
            or getattr(layer, "layer_norm1", None)
            or getattr(layer, "norm1", None)
        )
        attn_wrapper = (
            getattr(layer, "attention", None)
            or getattr(layer, "self_attn", None)
            or getattr(layer, "attn", None)
        )
        ln_after = (
            getattr(layer, "layernorm_after", None)
            or getattr(layer, "layer_norm2", None)
            or getattr(layer, "norm2", None)
        )

        if ln_before is None or attn_wrapper is None or ln_after is None:
            raise AttributeError(
                f"Cannot find expected sub-modules in layer {layer_id} ({type(layer).__name__}). "
                f"Found: {[n for n, _ in layer.named_children()]}"
            )

        # ── Resolve the actual Q/K/V projection module and output projection ──
        #
        # HF ViT structure:
        #   layer.attention (ViTAttention)
        #     ├── .attention (ViTSelfAttention)  ← Q/K/V projections
        #     └── .output (ViTSelfOutput)         ← output projection (.dense)
        #
        # HF SigLIP/CLIP structure:
        #   layer.self_attn (SiglipAttention)    ← Q/K/V + out_proj all here
        #
        # We need to pass the Q/K/V module to _make_attention_qkv and the
        # output projection module to _make_attention_output separately.

        if hasattr(attn_wrapper, "attention") and hasattr(attn_wrapper, "output"):
            # HF ViT-style: nested ViTAttention wrapper
            attn_qkv = attn_wrapper.attention   # ViTSelfAttention: query, key, value
            attn_out = attn_wrapper.output      # ViTSelfOutput: dense
        elif hasattr(attn_wrapper, "self_attn"):
            # Some models wrap self_attn one level deeper
            attn_qkv = attn_wrapper.self_attn
            attn_out = attn_wrapper
        else:
            # Direct: SigLIP, CLIP, DINOv2 — all projections on the same module
            attn_qkv = attn_wrapper
            attn_out = attn_wrapper

        # Store the output module so _make_attention_output can find it
        self._attn_out_module = attn_out

        # ── Resolve MLP modules ────────────────────────────────────────────
        #
        # HF ViT structure:
        #   layer.intermediate (ViTIntermediate) ← fc1 (.dense)
        #   layer.output       (ViTOutput)       ← fc2 (.dense)
        #
        # HF SigLIP/CLIP/DINOv2 structure:
        #   layer.mlp (SiglipMLP)                ← fc1, fc2 directly
        #
        mlp_module = getattr(layer, "mlp", None) or getattr(layer, "ffn", None)
        mlp_output_module = None

        if mlp_module is None:
            # HF ViT: intermediate + output are separate siblings
            mlp_module = getattr(layer, "intermediate", None)
            mlp_output_module = getattr(layer, "output", None)
            if mlp_module is None:
                raise AttributeError(
                    f"Cannot find MLP sub-module in layer {layer_id} ({type(layer).__name__}). "
                    f"Found: {[n for n, _ in layer.named_children()]}"
                )

        # If intermediate has .dense but no .fc1, it's HF ViT style
        if (mlp_output_module is None
                and hasattr(mlp_module, "dense")
                and not hasattr(mlp_module, "fc1")
                and not hasattr(mlp_module, "fc2")):
            # HF ViT: intermediate.dense = fc1, layer.output.dense = fc2
            mlp_output_module = getattr(layer, "output", None)

        # Store for _make_mlp_fc to use
        self._mlp_output_module = mlp_output_module

        # ── Build the layer graph ──────────────────────────────────────────
        self.make_layernorm(layer_id, ln_before,
                            skip=not self.layernorm_attrs["first_layernorm"],
                            simple=self.layernorm_attrs["simple"],
                            location="input")
        self.make_attention(layer_id, attn_qkv, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(layer_id, ln_after,
                            skip=True,
                            simple=self.layernorm_attrs["simple"],
                            location="post_attention")
        self.make_mlp(layer_id, mlp_module, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    # ── Save pipeline ──────────────────────────────────────────────────────

    def to_int4(self) -> ir.Model:
        """Post-hoc INT4 quantization using MatMulNBitsQuantizer."""
        quant = MatMulNBitsQuantizer(
            model=ir.to_proto(self.model),
            block_size=self.quant_attrs["int4"]["qdq_block_size"],
            is_symmetric=self.quant_attrs["int4"]["is_symmetric"],
            accuracy_level=self.quant_attrs["int4"]["accuracy_level"],
            nodes_to_exclude=self.quant_attrs["int4"]["nodes_to_exclude"],
            quant_format=QuantFormat.QDQ if self.quant_attrs["use_qdq"] else QuantFormat.QOperator,
            op_types_to_quantize=self.quant_attrs["int4"]["op_types_to_quantize"],
            algo_config=self.quant_attrs["int4"]["algo_config"],
        )
        quant.process()
        return ir.from_proto(quant.model.model)

    def save_model(self, out_dir):
        """Quantize (if INT4) and serialize the ONNX model to disk."""
        print(f"Saving vision ONNX model in {out_dir}")

        if self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4}:
            model = self.to_int4()
        else:
            model = self.model

        model.graph.sort()

        out_path = os.path.join(out_dir, self.filename)
        data_path = os.path.join(out_dir, os.path.basename(out_path) + ".data")
        if os.path.exists(out_path):
            print(f"Overwriting {out_path}")
            os.remove(out_path)
        if os.path.exists(data_path):
            print(f"Overwriting {data_path}")
            os.remove(data_path)

        with tqdm() as pbar:
            total_set = False

            def callback(tensor: ir.TensorProtocol, metadata: dict):
                nonlocal total_set
                if not total_set:
                    pbar.total = metadata.total
                    total_set = True
                pbar.update()
                pbar.set_description(f"Saving {tensor.name} ({tensor.dtype.short_name()}, {tensor.shape})")

            ir.save(model, out_path, external_data=os.path.basename(data_path),
                    size_threshold_bytes=0, callback=callback)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """
        Write (or update) genai_config.json with the vision encoder section.

        If genai_config.json already exists (written by the text model), this
        method merges the vision section into it. Otherwise it creates a new file.
        """
        config_path = os.path.join(out_dir, "genai_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                genai_config = json.load(f)
        else:
            genai_config = {"model": {}}

        vision_section = {
            "filename": self.filename,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_channels": self.num_channels,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attn_heads,
            "num_hidden_layers": self.num_layers,
            "inputs": {k: v for k, v in self.input_names.items()},
            "outputs": {k: v for k, v in self.output_names.items()},
        }

        if self.ep != "cpu":
            ep_name = self.ep.replace("trt-rtx", "NvTensorRtRtx")
            vision_section["session_options"] = {
                "provider_options": [{ep_name: self.ep_attrs[self.ep]}]
            }

        self.update_genai_config(vision_section)
        genai_config["model"]["vision"] = vision_section

        print(f"Saving GenAI config (vision section) in {out_dir}")
        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)

    def update_genai_config(self, vision_section: dict):
        """Override in subclasses to add model-specific fields to the vision config section."""
        pass

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        """Save the image processor config (preprocessor_config.json)."""
        try:
            processor = AutoImageProcessor.from_pretrained(
                model_name_or_path,
                token=self.hf_token,
                trust_remote_code=self.hf_remote,
                **extra_kwargs,
            )
            print(f"Saving image processor files in {out_dir}")
            processor.save_pretrained(out_dir)
        except Exception as e:
            print(f"Warning: Could not save image processor: {e}")
