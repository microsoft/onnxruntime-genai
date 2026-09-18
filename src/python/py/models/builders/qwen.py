# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# ------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.

import copy
import json
import os

import numpy as np
import onnx_ir as ir
import torch
from transformers import Qwen2ForCausalLM

from .base import Model
from .expansions import Qwen38
from .mtp import MTPModel


class QwenModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Qwen3Model(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self, config):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init(config)


class Qwen25VLTextModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Compute LayerNorms in FP32 for better accuracy
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = True
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = True

        # Compute RoPE in FP32 for better accuracy
        self.rope_attrs["cast"]["use_fp32"] = True
        self.rope_attrs["cast"]["root_input"] = True
        self.rope_attrs["cast"]["output_0"] = True

    def is_packed_matmul_supported(self):
        # We need separate Q, K, V tensors to apply MRoPE manually.
        return False

    def is_fused_rope_supported(self):
        # Qwen 2.5 VL applies MRoPE manually before attention, not fused in the op
        return False

    def make_inputs_and_outputs(self):
        # Qwen2.5-VL uses 3D position_ids
        self.input_shapes["position_ids"] = (
            [3, "num_tokens"] if self.use_paged_attention else [3, "batch_size", "sequence_length"]
        )
        super().make_inputs_and_outputs()


class Qwen3VLTextModel(Qwen25VLTextModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Avoid duplicate Cast nodes that form a SkipLayerNorm --> Cast --> Cast --> SkipLayerNorm pattern
        self.layernorm_attrs["cast"]["output_3"] = False

        # Qwen3-VL uses QK norms whose outputs will have already been casted to FP32
        self.rope_attrs["cast"]["root_input"] = False

        # Qwen3 attention uses QK normalization
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True

        # Qwen3-VL uses the Interleaved MRotaryEmbedding layout.
        self.rope_attrs["mrope_layout"] = 1

    def make_qk_norm(self, layer_id, attention):
        # Before: SimplifiedLayerNorm --> Cast from FP32 to io_dtype --> Reshape --> Cast from io_dtype to FP32 --> MRotaryEmbedding
        # After:  SimplifiedLayerNorm --> Reshape --> MRotaryEmbedding
        # This allows both LayerNorm and MRoPE to be computed in FP32. Reshape is not affected by the dtype.

        self.layernorm_attrs["cast"]["output_0"] = False
        super().make_qk_norm(layer_id, attention)

        # Update dtypes for QK-norm reshapes to stay as FP32 and not cast to self.io_dtype
        self.values[self.attention_attrs["q_path"]].dtype = ir.DataType.FLOAT
        self.values[self.attention_attrs["k_path"]].dtype = ir.DataType.FLOAT

        self.layernorm_attrs["cast"]["output_0"] = True


class VideoChatFlashQwenModel(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def load_weights(self, input_path):
        # Load the standard Qwen2 backbone without importing the checkpoint's
        # custom video modeling code and its optional dependencies.
        extra_kwargs = {} if os.path.isdir(self.model_name_or_path) else {"cache_dir": self.cache_dir}
        return Qwen2ForCausalLM.from_pretrained(
            self.model_name_or_path,
            token=self.hf_token,
            **extra_kwargs,
        )


class Qwen35TextModel(Model):
    def validate_gated_delta_net_options(self, use_paged_attention, linear_attn_op, state_window, ep):
        uses_gated_delta_net = use_paged_attention or linear_attn_op == "gated_delta_net"
        if uses_gated_delta_net and ep != "cuda":
            raise ValueError("GatedDeltaNet exports require the CUDA execution provider")
        if uses_gated_delta_net and state_window:
            raise ValueError("GatedDeltaNet exports commit an unwindowed recurrent state and require state_window=0")

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.linear_attn_op = str(extra_options.get("linear_attn_op", "linear_attention")).lower()
        if self.linear_attn_op not in ("linear_attention", "gated_delta_net"):
            raise ValueError("linear_attn_op must be one of: linear_attention, gated_delta_net")
        self.configure_gated_delta_net_io()

        # OffsetRMSNorm: Qwen3.5 uses (1 + weight) * RMSNorm(x).
        # Pre-bake the +1 into the weight initializer so the base class's
        # SkipSimplifiedLayerNormalization can be used directly.
        self.layernorm_attrs["add_offset"] = 1

        # Qwen-3.5 uses interleaved, partial MRoPE for both text and multimodal inputs.
        self.rope_attrs["mrope_layout"] = 1
        self.rope_attrs["cast"]["use_fp32"] = True
        self.rope_attrs["cast"]["root_input"] = True
        self.rope_attrs["cast"]["output_0"] = True

    def configure_gated_delta_net_io(self):
        """Declare every linear-attention graph binding, so emitters never respell a name or shape."""
        linear_layers = [
            layer_id for layer_id, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"
        ]
        if not linear_layers:
            # Without a linear-attention layer nothing below is emitted, so the capacity has no bindings.
            self.context_length_attrs["state_update_capacity"] = 0
            return

        self.validate_gated_delta_net_options(
            self.use_paged_attention,
            self.linear_attn_op,
            self.context_length_attrs["state_window"],
            self.ep,
        )

        if self.use_paged_attention:
            conv_shape = ["batch_size", self.linear_conv_dim, self.linear_conv_kernel_dim - 1]
            self.input_shapes["past.conv"] = conv_shape
            self.output_shapes["present.conv"] = conv_shape

        if self.use_paged_attention or self.linear_attn_op == "gated_delta_net":
            recurrent_shape = [
                "batch_size",
                self.linear_num_value_heads,
                self.linear_value_head_dim,
                self.linear_key_head_dim,
            ]
            self.input_types["past.recurrent"] = ir.DataType.FLOAT
            self.input_shapes["past.recurrent"] = recurrent_shape
            self.output_types["present.recurrent"] = ir.DataType.FLOAT
            self.output_shapes["present.recurrent"] = recurrent_shape

        capacity = self.context_length_attrs["state_update_capacity"]
        if not capacity:
            return

        self.input_names["state_update.capture_count"] = "state_update_capture_count"
        self.input_types["state_update.capture_count"] = ir.DataType.INT32
        self.input_shapes["state_update.capture_count"] = ["batch_size"]
        self.input_names["state_update.active"] = "state_update_active"
        self.input_types["state_update.active"] = ir.DataType.INT32
        self.input_shapes["state_update.active"] = [1]

        self.output_names["state_update.conv_value"] = {
            layer_id: f"state_update.{layer_id}.conv_value" for layer_id in linear_layers
        }
        self.output_types["state_update.conv_value"] = self.io_dtype
        self.output_shapes["state_update.conv_value"] = ["batch_size", capacity, self.linear_conv_dim]

        # One capsule packs each captured token's decay gates, key row, and value row back to back.
        capsule_width = capacity * (
            self.linear_num_value_heads
            + self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
        )
        self.output_names["state_update.recurrent_capsule"] = {
            layer_id: f"state_update.{layer_id}.recurrent_capsule" for layer_id in linear_layers
        }
        self.output_types["state_update.recurrent_capsule"] = ir.DataType.FLOAT
        self.output_shapes["state_update.recurrent_capsule"] = ["batch_size", capsule_width]

    def make_inputs_and_outputs(self):
        # Qwen-3.5 uses 3D position_ids
        self.input_shapes["position_ids"] = (
            [3, "num_tokens"] if self.use_paged_attention else [3, "batch_size", "sequence_length"]
        )
        super().make_inputs_and_outputs()

    def is_packed_matmul_supported(self):
        # Qwen-3.5 needs a separate Q projection to split its per-head Q and gate values.
        return False

    def is_packed_attn_supported(self):
        return False

    def make_attention_init(self, config):
        # Set QK norm before the base class selects packed or paged attention paths.
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init(config)

    def is_fused_rope_supported(self):
        # Qwen-3.5 applies MRoPE manually before attention, not fused in the op
        return False

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Dispatch to full attention or GatedDeltaNet based on layer type."""
        if self.layer_types[layer_id] == "linear_attention":
            self.make_qwen_gated_delta_net(layer_id, attention, root_input)
        else:
            super().make_attention(layer_id, attention, root_input, **kwargs)

    def get_attn_module(self, layer_id, layer):
        return layer.linear_attn if self.layer_types[layer_id] == "linear_attention" else layer.self_attn

    def make_attention_input_proj(self, layer_id, attention, root_input, **kwargs):
        """Split Qwen3.5's doubled, per-head Q projection into Q and gate."""
        super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)

        q_size = self.num_attn_heads * self.head_size
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        q_gate_reshape = [0, self.num_attn_heads, self.head_size * 2]
        q_reshape = [0, q_size]
        if not self.use_paged_attention:
            q_gate_reshape.insert(1, 0)
            q_reshape.insert(1, 0)

        rs_qg_name = f"/model/layers.{layer_id}/attn/q_gate/Reshape"
        rs_qg_output = f"{rs_qg_name}/output_0"
        self.make_reshape(
            rs_qg_name,
            [self.attention_attrs["q_path"], f"/model/constants/INT64/{q_gate_reshape}"],
            self.io_dtype,
            [*token_shape, self.num_attn_heads, self.head_size * 2],
        )

        split_name = f"/model/layers.{layer_id}/attn/q_gate/Split"
        q_4d_output = f"{split_name}/output_0"
        gate_4d_output = f"{split_name}/output_1"
        q_gate_shape = [*token_shape, self.num_attn_heads, self.head_size]
        self.make_split(
            split_name,
            inputs=[rs_qg_output, f"/model/constants/INT64/[{self.head_size}, {self.head_size}]"],
            outputs=[q_4d_output, gate_4d_output],
            dtypes=[self.io_dtype, self.io_dtype],
            shapes=[q_gate_shape, q_gate_shape],
            axis=-1,
        )

        rs_q_name = f"/model/layers.{layer_id}/attn/q_proj/Reshape"
        self.make_reshape(
            rs_q_name,
            [q_4d_output, f"/model/constants/INT64/{q_reshape}"],
            self.io_dtype,
            [*token_shape, q_size],
        )

        rs_g_name = f"/model/layers.{layer_id}/attn/gate/Reshape"
        self.make_reshape(
            rs_g_name,
            [gate_4d_output, f"/model/constants/INT64/{q_reshape}"],
            self.io_dtype,
            [*token_shape, q_size],
        )

        self.attention_attrs["q_path"] = f"{rs_q_name}/output_0"
        self.attention_attrs["gate_path"] = f"{rs_g_name}/output_0"

    def make_attention_output_proj(self, layer_id, attention, root_input, **kwargs):
        """Apply Qwen3.5's attention output gate before the shared output projection."""
        q_size = self.num_attn_heads * self.head_size
        output_shape = self.make_hidden_state_shape(last_dim=q_size)
        sigmoid_name = f"/model/layers.{layer_id}/attn/gate/Sigmoid"
        self.make_sigmoid(
            sigmoid_name,
            self.attention_attrs["gate_path"],
            self.io_dtype,
            output_shape,
        )

        gated_name = f"/model/layers.{layer_id}/attn/gate/Mul"
        self.make_mul(
            gated_name,
            [self.attention_attrs["o_path"], f"{sigmoid_name}/output_0"],
            self.io_dtype,
            output_shape,
        )
        self.attention_attrs["o_path"] = f"{gated_name}/output_0"

        super().make_attention_output_proj(layer_id, attention, root_input, **kwargs)

    def make_qwen_gated_delta_net(self, layer_id, linear_attn, root_input):
        """Build the Qwen linear-attention layer for dense or packed token layouts.

        Uses com.microsoft contrib ops:
        - CausalConvWithState / VarlenCausalConvWithState
        - LinearAttention / GatedDeltaNet
        """
        basename = f"/model/layers.{layer_id}/linear_attn"

        z_name, b_name, a_name, conv_input, conv_weight_name = self.make_linear_attention_input_proj(
            layer_id, linear_attn, root_input
        )

        conv_bias_name = f"model.layers.{layer_id}.linear_attn.conv1d.bias"
        self.make_initializer(torch.zeros(self.linear_conv_dim, dtype=torch.float32), conv_bias_name, to=self.io_dtype)

        if self.use_paged_attention:
            conv_op_name = f"{basename}/VarlenCausalConvWithState"
            self.make_varlen_causal_conv_with_state(
                conv_op_name,
                root_input=conv_input,
                weight=conv_weight_name,
                bias=conv_bias_name,
                cumulative_sequence_length=self.input_names["cumulative_sequence_lengths"],
                past_conv_state=self.input_names["past.conv"][layer_id],
                present_conv_state=self.output_names["present.conv"][layer_id],
                output_shape=["num_tokens", self.linear_conv_dim],
                present_conv_shape=self.output_shapes["present.conv"],
                **self.make_conv_state_update_kwargs(layer_id),
            )
            linear_output = self.make_gated_delta_net_layer(
                layer_id,
                linear_attn,
                f"{conv_op_name}/output_0",
                b_name,
                a_name,
            )
            self.make_linear_attention_output_proj(layer_id, linear_attn, linear_output, z_name)
            return

        conv_op_name = f"{basename}/CausalConvWithState"
        self.make_causal_conv_with_state(
            conv_op_name,
            root_input=conv_input,
            weight=conv_weight_name,
            bias=conv_bias_name,
            past_conv_state=self.input_names["past.conv"][layer_id],
            present_conv_state=self.output_names["present.conv"][layer_id],
            channels=self.linear_conv_dim,
        )
        conv_out_t_name = f"{basename}/conv_out/Transpose"
        conv_out_t_output = f"{conv_out_t_name}/output_0"
        self.make_transpose(
            conv_out_t_name,
            f"{conv_op_name}/output_0",
            self.io_dtype,
            ["batch_size", "sequence_length", self.linear_conv_dim],
            [0, 2, 1],
        )

        if self.linear_attn_op == "gated_delta_net":
            linear_output = self.make_gated_delta_net_layer(
                layer_id,
                linear_attn,
                conv_out_t_output,
                b_name,
                a_name,
            )
            self.make_linear_attention_output_proj(layer_id, linear_attn, linear_output, z_name)
            return

        q_scaled_output, k_norm_out, v_out, g_output, beta_output = self.make_linear_attention_normalize_and_gate(
            layer_id,
            linear_attn,
            conv_out_t_output,
            b_name,
            a_name,
        )

        # --- Fused recurrence: LinearAttention (com.microsoft) ---
        la_op_name = f"{basename}/LinearAttention"
        self.make_linear_attention(
            la_op_name,
            q_path=q_scaled_output,
            k_path=k_norm_out,
            v_path=v_out,
            past_recurrent_state=self.input_names["past.recurrent"][layer_id],
            present_recurrent_state=self.output_names["present.recurrent"][layer_id],
            decay=g_output,
            beta=beta_output,
            q_num_heads=self.linear_num_key_heads,
            kv_num_heads=self.linear_num_value_heads,
            update_rule="gated_delta",
            scale=1.0,  # Q is already pre-scaled by 1/sqrt(d_k)
        )
        la_output = f"{la_op_name}/output_0"

        # Gated RMSNorm + output projection
        self.make_linear_attention_output_proj(layer_id, linear_attn, la_output, z_name)

    def make_conv_state_update_kwargs(self, layer_id):
        """Compact convolution-capture bindings for this layer, or nothing when capture is disabled."""
        capacity = self.context_length_attrs["state_update_capacity"]
        if not capacity:
            return {}
        return {
            "state_update_capacity": capacity,
            "state_update_capture_count": self.input_names["state_update.capture_count"],
            "state_update_value": self.output_names["state_update.conv_value"][layer_id],
            "state_update_value_shape": self.output_shapes["state_update.conv_value"],
        }

    def make_recurrent_state_update_kwargs(self, layer_id):
        """Compact recurrent-capture bindings for this layer, or nothing when capture is disabled."""
        capacity = self.context_length_attrs["state_update_capacity"]
        if not capacity:
            return {}
        return {
            "state_update_capacity": capacity,
            "state_update_capture_count": self.input_names["state_update.capture_count"],
            "state_update_active": self.input_names["state_update.active"],
            "state_update_capsule": self.output_names["state_update.recurrent_capsule"][layer_id],
            "state_update_capsule_shape": self.output_shapes["state_update.recurrent_capsule"],
        }

    def make_gated_delta_net_layer(self, layer_id, linear_attn, conv_output, b_name, a_name):
        """Split the conv output into per-head Q/K/V and run GatedDeltaNet over dense or packed tokens."""
        basename = f"/model/layers.{layer_id}/linear_attn"
        packed = self.use_paged_attention
        token_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        # Reshape constants keep every token axis, so packed layouts carry one leading 0 and dense two.
        kept_axes = "0" if packed else "0, 0"
        key_heads, key_head_dim = self.linear_num_key_heads, self.linear_key_head_dim
        value_heads, value_head_dim = self.linear_num_value_heads, self.linear_value_head_dim
        key_dim, value_dim = self.linear_key_dim, self.linear_value_dim

        split_name = f"{basename}/split_qkv/Split"
        split_outputs = [f"{split_name}/output_{index}" for index in range(3)]
        self.make_split(
            split_name,
            inputs=[conv_output, f"/model/constants/INT64/[{key_dim}, {key_dim}, {value_dim}]"],
            outputs=split_outputs,
            dtypes=[self.io_dtype] * 3,
            shapes=[[*token_shape, key_dim], [*token_shape, key_dim], [*token_shape, value_dim]],
            axis=-1,
        )

        head_paths = []
        for tag, split_output, num_heads, head_dim in (
            ("q", split_outputs[0], key_heads, key_head_dim),
            ("k", split_outputs[1], key_heads, key_head_dim),
            ("v", split_outputs[2], value_heads, value_head_dim),
        ):
            reshape_name = f"{basename}/{tag}_heads/Reshape"
            self.make_reshape(
                reshape_name,
                [split_output, f"/model/constants/INT64/[{kept_axes}, {num_heads}, {head_dim}]"],
                self.io_dtype,
                [*token_shape, num_heads, head_dim],
            )
            head_paths.append(f"{reshape_name}/output_0")

        # The kernel applies Qwen's own gate arithmetic, so the raw checkpoint tensors are exported as-is.
        a_log_name = f"model.layers.{layer_id}.linear_attn.A_log"
        self.make_initializer(linear_attn.A_log, a_log_name, to=ir.DataType.FLOAT)
        dt_bias_name = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(linear_attn.dt_bias, dt_bias_name, to=ir.DataType.FLOAT)

        op_name = f"{basename}/GatedDeltaNet"
        recurrent_shape = self.output_shapes["present.recurrent"]
        shared_kwargs = {
            "q_path": head_paths[0],
            "k_path": head_paths[1],
            "v_path": head_paths[2],
            "decay": f"{a_name}/output_0",
            "beta": f"{b_name}/output_0",
            "a_log": a_log_name,
            "dt_bias": dt_bias_name,
            "gate_shape": [*token_shape, value_heads],
            "gate_activation": "qwen",
            "beta_activation": "sigmoid",
            "qk_l2_norm": 1,
            "update_rule": "gated_delta",
            "scale": 0.0,
            "output_shape": [*token_shape, value_heads, value_head_dim],
        }
        if packed:
            self.make_varlen_gated_delta_net(
                op_name,
                cumulative_sequence_length=self.input_names["cumulative_sequence_lengths"],
                past_recurrent_state=self.input_names["past.recurrent"][layer_id],
                present_recurrent_state=self.output_names["present.recurrent"][layer_id],
                present_recurrent_shape=recurrent_shape,
                **self.make_recurrent_state_update_kwargs(layer_id),
                **shared_kwargs,
            )
        else:
            self.make_gated_delta_net(
                op_name,
                initial_state=self.input_names["past.recurrent"][layer_id],
                final_state=self.output_names["present.recurrent"][layer_id],
                state_shape=recurrent_shape,
                **shared_kwargs,
            )

        reshape_name = f"{basename}/gdn_out/Reshape"
        self.make_reshape(
            reshape_name,
            [f"{op_name}/output_0", f"/model/constants/INT64/[{kept_axes}, {value_dim}]"],
            self.io_dtype,
            [*token_shape, value_dim],
        )
        return f"{reshape_name}/output_0"

    def make_linear_attention_input_proj(self, layer_id, attention, root_input):
        """Build linear projections, conv weight initializer, and QKV transpose.

        Returns:
            (z_name, b_name, a_name, qkv_t_output, conv_weight_name)
        """
        basename = f"/model/layers.{layer_id}/linear_attn"

        qkv_name = f"{basename}/qkv_proj/MatMul"
        self.make_matmul(attention.in_proj_qkv, qkv_name, root_input)

        z_name = f"{basename}/z_proj/MatMul"
        self.make_matmul(attention.in_proj_z, z_name, root_input)

        # The decay and beta gates drive the GatedDeltaNet recurrence, and their weights are
        # ~0.1% of the model, so they stay dense regardless of which loader supplied them.
        b_name = f"{basename}/b_proj/MatMul"
        self.require_dense_linear_attention_gate(attention.in_proj_b, b_name)
        self.exclude_node_from_quantization(b_name)
        self.make_matmul(attention.in_proj_b, b_name, root_input)

        a_name = f"{basename}/a_proj/MatMul"
        self.require_dense_linear_attention_gate(attention.in_proj_a, a_name)
        self.exclude_node_from_quantization(a_name)
        self.make_matmul(attention.in_proj_a, a_name, root_input)

        conv_input = f"{qkv_name}/output_0"
        if not self.use_paged_attention:
            qkv_t_name = f"{basename}/qkv_proj/Transpose"
            conv_input = f"{qkv_t_name}/output_0"
            self.make_transpose(
                qkv_t_name,
                f"{qkv_name}/output_0",
                self.io_dtype,
                ["batch_size", self.linear_conv_dim, "sequence_length"],
                [0, 2, 1],
            )

        conv_weight_name = f"model.layers.{layer_id}.linear_attn.conv1d.weight"
        self.make_initializer(attention.conv1d.weight, conv_weight_name, to=self.io_dtype)

        return z_name, b_name, a_name, conv_input, conv_weight_name

    def require_dense_linear_attention_gate(self, projection, name):
        if hasattr(projection, "qweight") or getattr(projection, "quant_type", "none") != "none":
            raise ValueError(
                f"Linear-attention gate '{name}' must remain dense, but the checkpoint supplies "
                "pre-quantized weights that its loader did not dequantize."
            )

    def make_linear_attention_normalize_and_gate(self, layer_id, attention, conv_out_3d, b_name, a_name):
        """Split QKV, per-head L2 norm, Q scale, and compute decay/beta gates.

        Args:
            conv_out_3d: Conv output transposed to [B, S, linear_conv_dim].
            b_name: Name of the beta projection MatMul node.
            a_name: Name of the alpha projection MatMul node.

        Returns:
            (q_scaled_output, k_norm_out, v_out, g_output, beta_output)
        """
        basename = f"/model/layers.{layer_id}/linear_attn"

        # Split into Q, K, V
        split_qkv_name = f"{basename}/split_qkv/Split"
        q_out = f"{split_qkv_name}/output_0"
        k_out = f"{split_qkv_name}/output_1"
        v_out = f"{split_qkv_name}/output_2"
        self.make_split(
            split_qkv_name,
            inputs=[
                conv_out_3d,
                f"/model/constants/INT64/[{self.linear_key_dim}, {self.linear_key_dim}, {self.linear_value_dim}]",
            ],
            outputs=[q_out, k_out, v_out],
            dtypes=[self.io_dtype] * 3,
            shapes=[
                ["batch_size", "sequence_length", self.linear_key_dim],
                ["batch_size", "sequence_length", self.linear_key_dim],
                ["batch_size", "sequence_length", self.linear_value_dim],
            ],
            axis=-1,
        )

        # Per-head L2 normalize Q and K
        q_norm_out = self.make_l2_normalize(f"{basename}/q_l2norm", q_out)
        k_norm_out = self.make_l2_normalize(f"{basename}/k_l2norm", k_out)

        # Scale Q by 1/sqrt(head_k_dim)
        scale_name = f"/model/constants/{self.io_dtype}/{float(1.0 / np.sqrt(self.linear_key_head_dim))}"
        q_scaled_name = f"{basename}/q_scaled/Mul"
        self.make_mul(
            q_scaled_name,
            [q_norm_out, scale_name],
            self.io_dtype,
            ["batch_size", "sequence_length", self.linear_key_dim],
        )
        q_scaled_output = f"{q_scaled_name}/output_0"

        # g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)
        dt_bias_init = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(attention.dt_bias, dt_bias_init, to=ir.DataType.FLOAT)

        neg_exp_a_name = f"model.layers.{layer_id}.linear_attn.neg_exp_A"
        neg_exp_a = (-attention.A_log.data.exp()).detach()
        self.make_initializer(neg_exp_a, neg_exp_a_name, to=ir.DataType.FLOAT)

        gate_name = f"{basename}/LinearAttentionGate"
        gate_shape = ["batch_size", "sequence_length", self.linear_num_value_heads]
        self.make_linear_attention_gate(
            gate_name,
            a=f"{a_name}/output_0",
            dt_bias=dt_bias_init,
            decay_scale=neg_exp_a_name,
            b=f"{b_name}/output_0",
            shape=gate_shape,
        )
        g_output = f"{gate_name}/output_0"
        beta_output = f"{gate_name}/output_1"

        return q_scaled_output, k_norm_out, v_out, g_output, beta_output

    def make_linear_attention_output_proj(self, layer_id, attention, attn_output_3d, z_name):
        """Build gated RMSNorm and output projection.

        Args:
            attn_output_3d: Attention output [B, S, linear_value_dim] (3D packed).
            z_name: Name of the z-gate projection MatMul node.
        """
        basename = f"/model/layers.{layer_id}/linear_attn"
        output_shape = (
            ["num_tokens", self.linear_value_dim]
            if self.use_paged_attention
            else ["batch_size", "sequence_length", self.linear_value_dim]
        )
        norm_weight = f"model.layers.{layer_id}.linear_attn.norm.weight"
        self.make_initializer(attention.norm.weight, norm_weight, to=self.io_dtype)

        gated_norm_name = f"{basename}/GatedRMSNorm"
        self.make_gated_rms_norm(
            gated_norm_name,
            root_input=attn_output_3d,
            scale=norm_weight,
            gate=f"{z_name}/output_0",
            shape=output_shape,
            epsilon=self.layernorm_attrs["epsilon"],
        )

        o_name = f"{basename}/out_proj/MatMul"
        self.make_matmul(attention.out_proj, o_name, f"{gated_norm_name}/output_0")

        self.layernorm_attrs["skip_input"] = f"{o_name}/output_0"

    def make_l2_normalize(self, basename, root_input):
        """Per-head L2 normalize: reshape [B, S, N*H] -> [B, S, N, H], norm, reshape back.

        Uses [0, 0, N, H] / [0, 0, N*H] reshape targets so all dims are
        constants or copied from the 3D/4D input, avoiding Shape ops that
        would run on CPU and block CUDA graph capture.
        """
        total_dim = self.linear_num_key_heads * self.linear_key_head_dim

        # Reshape to [B, S, N, H] for per-head normalization
        flat_name = f"{basename}/flat/Reshape"
        flat_out = f"{flat_name}/output_0"
        self.make_reshape(
            flat_name,
            [root_input, f"/model/constants/INT64/[0, 0, {self.linear_num_key_heads}, {self.linear_key_head_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", self.linear_num_key_heads, self.linear_key_head_dim],
        )

        norm_name = f"{basename}/LpNormalization"
        self.make_lp_normalization(
            norm_name,
            flat_out,
            self.io_dtype,
            ["batch_size", "sequence_length", self.linear_num_key_heads, self.linear_key_head_dim],
            axis=-1,
            p=2,
        )

        # Reshape back to [B, S, N*H]
        unflat_name = f"{basename}/unflat/Reshape"
        unflat_out = f"{unflat_name}/output_0"
        self.make_reshape(
            unflat_name,
            [f"{norm_name}/output_0", f"/model/constants/INT64/[0, 0, {total_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", total_dim],
        )
        return unflat_out

    def make_decoder_state_groups(self, inputs, outputs):
        if not self.use_paged_attention:
            return []

        full_attention_layers = [
            layer_id
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type in {"full_attention", "sliding_attention"}
        ]
        conv_layers = [
            layer_id
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type in {"conv", "linear_attention"}
        ]
        linear_attention_layers = [
            layer_id for layer_id, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"
        ]
        state_groups = []
        if full_attention_layers:
            state_groups.append(self.make_paged_key_value_state_group(full_attention_layers))
        if not linear_attention_layers:
            return state_groups

        state_update_capacity = (
            self.context_length_attrs["state_update_capacity"] if "state_update_capture_count" in inputs else 0
        )

        for state_name, layer_ids in (
            ("conv", conv_layers),
            ("recurrent", linear_attention_layers),
        ):
            group = {
                "kind": f"fixed_{state_name}",
                "layer_ids": layer_ids,
            }
            if state_update_capacity:
                state_update = {
                    "capacity": state_update_capacity,
                }
                if state_name == "recurrent":
                    state_update["key_head_count"] = self.linear_num_key_heads
                group["state_update"] = state_update
            state_groups.append(group)

        return state_groups


class Qwen35MoETextModel(Qwen35TextModel):
    """Qwen3.5 MoE hybrid model builder.

    Extends ``Qwen35TextModel`` with Mixture-of-Experts MLP layers.
    Each decoder layer replaces the dense MLP with:
    - A router that selects top-k experts from ``num_experts`` candidates
    - Packed routed expert weights (gate_up_proj + down_proj)
    - A shared expert (always-active) with its own gating signal

    The attention side (GatedDeltaNet linear + gated full) is inherited
    unchanged from the parent class.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # MoE attributes specific to Qwen-3.5 MoE
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["swiglu_fusion"] = 1
        self.moe_attrs["normalize_routing_weights"] = True
        if self.moe_attrs.get("swiglu_limit") is None and self.ep == "trt-rtx":
            # TRT-RTX EP builds currently require QMoE swiglu_limit to be present;
            # use +inf to preserve the "no clamp" behavior when the model omits it.
            self.moe_attrs["swiglu_limit"] = float("inf")

        self.moe_intermediate_size = getattr(config, "moe_intermediate_size", 512)
        self.shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", self.moe_intermediate_size
        )

    def get_moe_module(self, layer_id, layer):
        return layer.mlp

    def make_moe_preprocessing(self, layer_id, moe, root_input):
        gate_up_proj_bias = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        down_proj_bias = f"model.layers.{layer_id}.moe.experts.down_proj.bias"

        gate_up_weight = None
        down_weight = None
        if getattr(moe.experts, "gate_up_proj", None) is not None:
            # Repack HF concatenated [gate|up] to ORT interleaved [g0,u0,g1,u1,...].
            raw_gate_up = moe.experts.gate_up_proj
            half = raw_gate_up.shape[1] // 2
            gate_up_weight = torch.stack([raw_gate_up[:, :half, :], raw_gate_up[:, half:, :]], dim=2).reshape_as(
                raw_gate_up
            )
            down_weight = moe.experts.down_proj
        self.make_moe_expert_initializers(layer_id, moe.experts, gate_up_weight, down_weight)

        num_e = self.moe_attrs["num_experts"]
        self.make_initializer(torch.zeros(num_e, 2 * self.moe_intermediate_size), gate_up_proj_bias, to=self.io_dtype)
        self.make_initializer(torch.zeros(num_e, self.hidden_size), down_proj_bias, to=self.io_dtype)

    def make_moe_router(self, layer_id, moe, root_input):
        basename = f"/model/layers.{layer_id}/moe"
        router_basename = f"{basename}/router/MatMul"
        router_matmul_name = self.make_matmul(moe.gate, router_basename, root_input)
        router_reshape_name = f"{basename}/router/Reshape"
        self.make_reshape(
            router_reshape_name,
            [
                f"{router_matmul_name}/output_0",
                f"/model/constants/INT64/{[-1, self.moe_attrs['num_experts']]}",
            ],
            dtype=self.io_dtype,
            shape=["batch_size * sequence_length", self.moe_attrs["num_experts"]],
        )

    def make_moe_subgraph(self, layer_id, moe, root_input):
        basename = f"/model/layers.{layer_id}/moe"
        op_type = self.moe_attrs["op_type"]
        moe_weight_type = f"{'q' if op_type == 'QMoE' else ''}weight"
        gate_up_proj_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.{moe_weight_type}"
        gate_up_proj_scales = f"model.layers.{layer_id}.moe.experts.gate_up_proj.scales"
        gate_up_proj_bias = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.{moe_weight_type}"
        down_proj_scales = f"model.layers.{layer_id}.moe.experts.down_proj.scales"
        down_proj_bias = f"model.layers.{layer_id}.moe.experts.down_proj.bias"
        gate_up_proj_global_scales, down_proj_global_scales = self.moe_attrs.get("global_scale_names", {}).get(
            layer_id, ("", "")
        )

        moe_name = f"{basename}/{op_type}"
        self.make_moe_op(
            moe_name,
            root_input=root_input,
            router_probs=f"{basename}/router/Reshape/output_0",
            weight1=gate_up_proj_weight,
            scales1=gate_up_proj_scales if op_type == "QMoE" else "",
            bias1=gate_up_proj_bias,
            weight2=down_proj_weight,
            scales2=down_proj_scales if op_type == "QMoE" else "",
            bias2=down_proj_bias,
            global_scales1=gate_up_proj_global_scales,
            global_scales2=down_proj_global_scales,
        )

        shared_output, shared_gate = self.make_shared_expert(
            layer_id, moe.shared_expert, moe.shared_expert_gate, root_input
        )
        combine_name = f"{basename}/GatedAdd"
        self.make_gated_add(
            combine_name,
            root_input=f"{moe_name}/output_0",
            scaled_input=shared_output,
            gate=shared_gate,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )
        return f"{combine_name}/output_0"

    def make_shared_expert(self, layer_id, shared_expert, shared_expert_gate, root_input):
        basename = f"/model/layers.{layer_id}/shared_expert"

        # Temporarily set new intermediate size from shared experts
        intermediate_size = self.intermediate_size
        self.intermediate_size = self.shared_expert_intermediate_size
        self.make_mlp_proj(layer_id, shared_expert, root_input)
        self.intermediate_size = intermediate_size
        shared_output = self.mlp_attrs["output_0"]

        gate_matmul_name = self.make_matmul(shared_expert_gate, f"{basename}_gate/MatMul", root_input)
        gate_sigmoid_name = f"{basename}_gate/Sigmoid"
        self.make_sigmoid(
            gate_sigmoid_name, f"{gate_matmul_name}/output_0", self.io_dtype, shape=["batch_size", "sequence_length", 1]
        )

        return shared_output, f"{gate_sigmoid_name}/output_0"


class Qwen4ExpTextModel(Qwen35MoETextModel, Qwen38):
    """Qwen4-Exp decoder builder using external token/vision embeddings."""

    CPU_EMBEDDING_ANNOTATION = "cpu_embedding"

    def is_packed_matmul_supported(self):
        return Model.is_packed_matmul_supported(self)

    def make_attention_init(self, config):
        super().make_attention_init(config)
        self.attention_attrs["use_packed_matmul"] = self.is_packed_matmul_supported()

    def make_attention_input_proj(self, layer_id, attention, root_input, **kwargs):
        if not self.attention_attrs["use_packed_matmul"]:
            return super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)

        q_size = self.q_size
        self.q_size = 2 * q_size
        try:
            super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)
        finally:
            self.q_size = q_size

    def make_moe_router(self, layer_id, moe, root_input):
        basename = f"/model/layers.{layer_id}/moe/gate_up_router"
        packed_matmul = self.make_packed_matmul(
            moe.shared_expert.gate_proj,
            moe.shared_expert.up_proj,
            moe.gate,
            f"{basename}/MatMul",
            root_input,
        )
        packed_output = f"{packed_matmul}/output_0"

        projections = (moe.shared_expert.gate_proj, moe.shared_expert.up_proj, moe.gate)
        biases = [getattr(projection, "bias", None) for projection in projections]
        if any(bias is not None and torch.count_nonzero(bias) > 0 for bias in biases):
            bias_template = next(bias for bias in biases if bias is not None)
            packed_bias = torch.cat(
                [
                    bias
                    if bias is not None
                    else torch.zeros(
                        projection.out_features,
                        dtype=bias_template.dtype,
                        device=bias_template.device,
                    )
                    for projection, bias in zip(projections, biases, strict=True)
                ]
            )
            packed_add = f"{basename}/Add"
            self.make_add_bias(packed_bias, packed_add, packed_output)
            packed_output = f"{packed_add}/output_0"

        intermediate_size = self.shared_expert_intermediate_size
        num_experts = self.moe_attrs["num_experts"]
        split_outputs = [f"{basename}/Split/output_{index}" for index in range(3)]
        self.make_split(
            f"{basename}/Split",
            [packed_output, f"/model/constants/INT64/[{intermediate_size}, {intermediate_size}, {num_experts}]"],
            split_outputs,
            [self.io_dtype] * 3,
            [
                self.make_hidden_state_shape(last_dim=intermediate_size),
                self.make_hidden_state_shape(last_dim=intermediate_size),
                self.make_hidden_state_shape(last_dim=num_experts),
            ],
            axis=-1,
        )
        self.moe_attrs.setdefault("shared_expert_paths", {})[layer_id] = tuple(split_outputs[:2])

        router_reshape_name = f"/model/layers.{layer_id}/moe/router/Reshape"
        self.make_reshape(
            router_reshape_name,
            [split_outputs[2], f"/model/constants/INT64/{[-1, num_experts]}"],
            dtype=self.io_dtype,
            shape=["batch_size * sequence_length", num_experts],
        )

    def make_shared_expert(self, layer_id, shared_expert, shared_expert_gate, root_input):
        gate_path, up_path = self.moe_attrs["shared_expert_paths"].pop(layer_id)
        intermediate_size = self.intermediate_size
        self.intermediate_size = self.shared_expert_intermediate_size
        try:
            activation = self.make_activation(layer_id, gate_path)
            mul_name = f"/model/layers.{layer_id}/mlp/Mul"
            self.make_mul(
                mul_name,
                [f"{activation}/output_0", up_path],
                self.io_dtype,
                self.make_hidden_state_shape(last_dim=self.intermediate_size),
            )
            down_name = self.make_matmul(
                shared_expert.down_proj,
                f"/model/layers.{layer_id}/mlp/down_proj/MatMul",
                f"{mul_name}/output_0",
            )
            if (
                shared_expert.down_proj.bias is not None
                and torch.count_nonzero(shared_expert.down_proj.bias) > 0
            ):
                down_add = f"/model/layers.{layer_id}/mlp/down_proj/Add"
                self.make_add_bias(shared_expert.down_proj.bias, down_add, f"{down_name}/output_0")
                down_name = down_add
        finally:
            self.intermediate_size = intermediate_size

        gate_matmul_name = self.make_matmul(
            shared_expert_gate,
            f"/model/layers.{layer_id}/shared_expert_gate/MatMul",
            root_input,
        )
        gate_sigmoid_name = f"/model/layers.{layer_id}/shared_expert_gate/Sigmoid"
        self.make_sigmoid(
            gate_sigmoid_name,
            f"{gate_matmul_name}/output_0",
            self.io_dtype,
            shape=self.make_hidden_state_shape(last_dim=1),
        )
        return f"{down_name}/output_0", f"{gate_sigmoid_name}/output_0"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        extra_options = copy.deepcopy(extra_options)
        text_only = extra_options.get("text_only", False)
        extra_options["exclude_embeds"] = not text_only
        extra_options.setdefault("filename", "model.onnx" if text_only else "text.onnx")
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.use_cpu_embedding_gather = text_only
        self.model.metadata_props["qwen4_exp.past_indexer_names"] = "past.%d.indexer_key"
        self.model.metadata_props["qwen4_exp.present_indexer_names"] = "present.%d.indexer_key"
        self.model.metadata_props["qwen4_exp.past_ple_token_names"] = "past.%d.ple_tokens"
        self.model.metadata_props["qwen4_exp.present_ple_token_names"] = "present.%d.ple_tokens"
        self.model.metadata_props["qwen4_exp.past_ple_conv_names"] = "past.%d.ple_conv"
        self.model.metadata_props["qwen4_exp.present_ple_conv_names"] = "present.%d.ple_conv"

        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hc_count * self.hidden_size
        self.ple_layer_ids = {layer_id - 1 for layer_id in config.ple_layer_ids}
        self.ple_embed_dim = config.ple_embed_dim
        self.ple_conv_kernel_size = config.ple_conv_kernel_size
        self.ple_conv_dilation = config.ngram_size
        self.ngram_size = config.ngram_size
        self.ple_token_pad_id = config.eos_token_id
        self.rope_attrs["cast"]["use_fp32"] = False
        self.heads_per_ngram = config.heads_per_ngram
        self.indexer_num_heads = config.indexer_n_heads
        self.indexer_kv_heads = config.indexer_kv_heads
        self.indexer_head_dim = config.indexer_head_dim
        self.indexer_budget = config.indexer_budget
        self.indexer_compress_ratio = config.indexer_compress_ratio
        self.output_gate_type = config.output_gate_type or config.hidden_act
        self.tile_first_hidden_state = True
        self.emit_pre_final_hidden_states = False

        qsa_layers = {
            layer_id: f"past.{layer_id}.indexer_key"
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type == "qwen_sparse_attention"
        }
        qsa_outputs = {
            layer_id: f"present.{layer_id}.indexer_key"
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type == "qwen_sparse_attention"
        }
        self.input_names["past_key_values.key"] = self.make_cache_names(
            ["qwen_sparse_attention"], "past_key_values.key"
        )
        self.input_names["past_key_values.value"] = self.make_cache_names(
            ["qwen_sparse_attention"], "past_key_values.value"
        )
        self.output_names["present.key"] = self.make_cache_names(["qwen_sparse_attention"], "present.key")
        self.output_names["present.value"] = self.make_cache_names(["qwen_sparse_attention"], "present.value")
        self.input_names["past.indexer"] = qsa_layers
        self.input_types["past.indexer"] = self.io_dtype
        self.input_shapes["past.indexer"] = ["batch_size", "past_sequence_length", self.indexer_head_dim]
        self.output_names["present.indexer"] = qsa_outputs
        self.output_types["present.indexer"] = self.io_dtype
        self.output_shapes["present.indexer"] = ["batch_size", "total_sequence_length", self.indexer_head_dim]
        if self.use_paged_attention:
            capacity = self.indexer_budget + self.indexer_compress_ratio - 1
            self.model.metadata_props["qwen4_exp.selected_index_names"] = "sparse_attention.%d.selected_indices"
            self.model.metadata_props["qwen4_exp.selected_count_names"] = "sparse_attention.%d.selected_counts"
            self.input_names["sparse_attention.selected_indices"] = {
                layer_id: f"sparse_attention.{layer_id}.selected_indices" for layer_id in qsa_layers
            }
            self.input_types["sparse_attention.selected_indices"] = ir.DataType.INT32
            self.input_shapes["sparse_attention.selected_indices"] = ["num_tokens", capacity]
            self.input_names["sparse_attention.selected_counts"] = {
                layer_id: f"sparse_attention.{layer_id}.selected_counts" for layer_id in qsa_layers
            }
            self.input_types["sparse_attention.selected_counts"] = ir.DataType.INT32
            self.input_shapes["sparse_attention.selected_counts"] = ["num_tokens"]
            del self.input_names["past.indexer"]
            del self.output_names["present.indexer"]
            del self.model.metadata_props["qwen4_exp.past_indexer_names"]
            del self.model.metadata_props["qwen4_exp.present_indexer_names"]
            self.input_shapes["attention_metadata"] = [5]

        ple_token_state = {layer_id: f"past.{layer_id}.ple_tokens" for layer_id in self.ple_layer_ids}
        ple_conv_state = {layer_id: f"past.{layer_id}.ple_conv" for layer_id in self.ple_layer_ids}
        present_ple_tokens = {layer_id: f"present.{layer_id}.ple_tokens" for layer_id in self.ple_layer_ids}
        present_ple_conv = {layer_id: f"present.{layer_id}.ple_conv" for layer_id in self.ple_layer_ids}
        self.input_names["past.ple_tokens"] = ple_token_state
        self.input_types["past.ple_tokens"] = ir.DataType.INT64
        self.input_shapes["past.ple_tokens"] = ["batch_size", self.ngram_size - 1]
        self.input_names["past.ple_conv"] = ple_conv_state
        self.input_types["past.ple_conv"] = self.io_dtype
        self.input_shapes["past.ple_conv"] = [
            "batch_size",
            self.ple_conv_dilation * (self.ple_conv_kernel_size - 1),
            self.hc_hidden_size,
        ]
        self.output_names["present.ple_tokens"] = present_ple_tokens
        self.output_types["present.ple_tokens"] = ir.DataType.INT64
        self.output_shapes["present.ple_tokens"] = ["batch_size", self.ngram_size - 1]
        self.output_names["present.ple_conv"] = present_ple_conv
        self.output_types["present.ple_conv"] = self.io_dtype
        self.output_shapes["present.ple_conv"] = self.input_shapes["past.ple_conv"]

        self.input_names["input_ids"] = "input_ids"
        self.input_types["input_ids"] = ir.DataType.INT64
        self.input_shapes["input_ids"] = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]

    @staticmethod
    def prepare_engram_embedding(table):
        weight = table.weight.detach().cpu()
        scale = getattr(table, "weight_scale", None)
        if weight.dtype == torch.float8_e4m3fn:
            if scale is None:
                scale = torch.ones(1, dtype=torch.float32)
            else:
                scale = torch.as_tensor(scale).detach().cpu()
            if scale.numel() != 1:
                raise ValueError(f"Engram FP8 weight scale must be scalar, got shape {tuple(scale.shape)}.")
            return weight, scale

        if not weight.is_floating_point():
            raise ValueError(f"Engram embedding weight must be floating point, got {weight.dtype}.")
        max_abs = weight.abs().max().float()
        if not torch.isfinite(max_abs):
            raise ValueError("Engram embedding weight contains non-finite values.")
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = max_abs / fp8_max if max_abs > 0 else torch.ones((), dtype=torch.float32)
        quantized_weight = (weight / scale).clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
        return quantized_weight, scale.reshape(1)

    def update_genai_config(self, genai_config):
        super().update_genai_config(genai_config)
        decoder = genai_config["model"]["decoder"]
        decoder["inputs"]["past_ple_token_names"] = "past.%d.ple_tokens"
        decoder["inputs"]["past_ple_conv_names"] = "past.%d.ple_conv"
        decoder["inputs"]["past_indexer_names"] = "past.%d.indexer_key"
        decoder["outputs"]["present_ple_token_names"] = "present.%d.ple_tokens"
        decoder["outputs"]["present_ple_conv_names"] = "present.%d.ple_conv"
        decoder["outputs"]["present_indexer_names"] = "present.%d.indexer_key"
        decoder["ple_token_pad_id"] = self.ple_token_pad_id
        if self.ep != "cpu":
            session_options = decoder["session_options"]
            session_options["session.layer_assignment_settings"] = (
                f"cpu(={self.CPU_EMBEDDING_ANNOTATION})"
            )

    def make_gated_rms_norm(self, name, root_input, scale, gate, shape, epsilon=1e-5):
        output = f"{name}/output_0"
        self.make_node(
            "GatedRMSNorm",
            inputs=[root_input, scale, gate],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            epsilon=epsilon,
            activation=self.output_gate_type,
        )
        self.make_value(output, self.io_dtype, shape=shape)

    def make_branchwise_rms_norm(self, name, root_input, norm, hidden_size):
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        output = f"{name}/output_0"
        scale_name = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(norm.weight + 1.0, scale_name, to=self.io_dtype)
        self.make_node(
            "BranchwiseRMSNorm",
            inputs=[root_input, scale_name],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            epsilon=self.layernorm_attrs["epsilon"],
            num_branches=self.hc_count,
        )
        self.make_value(output, self.io_dtype, [*token_shape, self.hc_count * hidden_size])
        return output

    def make_scaled_silu(self, name, root_input, shape):
        output = f"{name}/output_0"
        self.make_node(
            "ScaledSiLU",
            inputs=[root_input],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            alpha=1.0 / self.hc_count,
        )
        self.make_value(output, self.io_dtype, shape)
        return output

    def make_hyper_connection_pre_mix(self, name, streams, pre_mix, token_shape):
        output = f"{name}/output_0"
        self.make_node(
            "HyperConnectionPreMix",
            inputs=[streams, pre_mix],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            num_branches=self.hc_count,
            reduction_scale=1.0 / self.hc_count,
        )
        self.make_value(output, self.io_dtype, [*token_shape, self.hidden_size])
        return output

    def make_hyper_connection_post_mix(self, name, streams, block_output, post_mix, token_shape):
        output = f"{name}/output_0"
        self.make_node(
            "HyperConnectionPostMix",
            inputs=[streams, block_output, post_mix],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            num_branches=self.hc_count,
        )
        self.make_value(output, self.io_dtype, [*token_shape, self.hc_hidden_size])
        return output

    def make_hyper_connection_mix(self, layer_id, hyper_connection, root_input, location, combine=True):
        basename = f"/model/layers.{layer_id}/{location}_hyper_connection"
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        normalized = self.make_branchwise_rms_norm(
            f"{basename}/hc_norm", root_input, hyper_connection.hc_norm, self.hidden_size
        )
        down_name = self.make_matmul(
            hyper_connection.input_mix_weight_down, f"{basename}/input_mix_weight_down/MatMul", normalized
        )
        silu_shape = [*token_shape, hyper_connection.input_mix_weight_down.out_features]
        silu_name = f"{basename}/input_mix_weight_down/SiLU"
        silu_output = self.make_scaled_silu(silu_name, f"{down_name}/output_0", silu_shape)
        up_name = self.make_matmul(
            hyper_connection.input_mix_weight_up,
            f"{basename}/input_mix_weight_up/MatMul",
            silu_output,
        )
        mix_sigmoid_shape = [*token_shape, self.hc_hidden_size]
        mix_sigmoid = f"{basename}/input_mix_weight_up/Sigmoid"
        self.make_sigmoid(
            mix_sigmoid,
            f"{up_name}/output_0",
            self.io_dtype,
            mix_sigmoid_shape,
        )
        mixed_name = f"{basename}/mixed/Mean"
        mixed_output = self.make_hyper_connection_pre_mix(
            mixed_name,
            normalized,
            f"{mix_sigmoid}/output_0",
            token_shape,
        )
        if not combine:
            return mixed_output

        inject_name = self.make_matmul(
            hyper_connection.block_inject_weight, f"{basename}/block_inject_weight/MatMul", normalized
        )
        inject_div = f"{basename}/block_inject_weight/Div"
        inject_shape = [*token_shape, self.hc_count]
        self.make_div(
            inject_div,
            [f"{inject_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.hc_count}"],
            self.io_dtype,
            inject_shape,
        )
        inject_sigmoid = f"{basename}/block_inject_weight/Sigmoid"
        self.make_sigmoid(inject_sigmoid, f"{inject_div}/output_0", self.io_dtype, inject_shape)
        inject_scale = f"{basename}/block_inject_weight/Mul"
        self.make_mul(
            inject_scale,
            [f"{inject_sigmoid}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/2"],
            self.io_dtype,
            inject_shape,
        )
        return mixed_output, root_input, f"{inject_scale}/output_0"

    def make_hyper_connection_injection(self, layer_id, block_output, hyper_input, injection_weights, location):
        basename = f"/model/layers.{layer_id}/{location}_hyper_connection/injection"
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        return self.make_hyper_connection_post_mix(
            basename, hyper_input, block_output, injection_weights, token_shape
        )

    def make_ple(self, layer_id, ple, root_input):
        basename = f"/model/layers.{layer_id}/ple"
        embedding = ple.ple_embedding
        multipliers = f"model.layers.{layer_id}.ple.layer_multipliers"
        vocab_sizes = f"model.layers.{layer_id}.ple.head_vocab_sizes"
        offsets = f"model.layers.{layer_id}.ple.head_offsets"
        eos = f"model.layers.{layer_id}.ple.eos_token_id"
        self.make_initializer(embedding.layer_multipliers, multipliers)
        self.make_initializer(embedding.ngram_heads_vocab_sizes, vocab_sizes)
        self.make_initializer(embedding.ngram_heads_offsets, offsets)
        self.make_initializer(torch.tensor(embedding.eos_token_id, dtype=torch.int64), eos)
        ngram_op_type = "VarlenNGramHashMapping" if self.use_paged_attention else "NGramHashMapping"
        ngram_name = f"{basename}/{ngram_op_type}"
        ngram_ids = f"{ngram_name}/output_0"
        ngram_inputs = [
            self.input_names["input_ids"],
            multipliers,
            vocab_sizes,
        ]
        if self.use_paged_attention:
            ngram_inputs.append(self.input_names["cumulative_sequence_lengths"])
        ngram_inputs.extend(
            [
                self.input_names["past.ple_tokens"][layer_id],
                offsets,
                eos,
            ]
        )
        self.make_node(
            ngram_op_type,
            inputs=ngram_inputs,
            outputs=[ngram_ids, self.output_names["present.ple_tokens"][layer_id]],
            name=ngram_name,
            domain="com.microsoft",
            max_ngram_size=self.ngram_size,
            n_head_per_ngram=self.heads_per_ngram,
            pad_id=embedding.eos_token_id,
            reset_on_eos=1,
        )
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.make_value(
            ngram_ids,
            ir.DataType.INT64,
            ["num_tokens", ngram_heads]
            if self.use_paged_attention
            else ["batch_size", "sequence_length", ngram_heads],
        )
        table_name = "model.ple.ngram_embedding.weight"
        table = embedding.ngram_embedding
        if table_name not in self.values:
            quantized_weight, weight_scale = self.prepare_engram_embedding(table)
            self.make_initializer(quantized_weight, table_name)
            self.make_initializer(
                weight_scale.reshape(1, 1),
                "model.ple.ngram_embedding.weight_scale",
                to=self.io_dtype,
            )
        if not hasattr(self, "external_data_files"):
            self.external_data_files = {}
        self.external_data_files[table_name] = "engram.data"
        gather_name = f"{basename}/ngram_embedding/GatherBlockQuantized"
        head_dim = self.ple_embed_dim // ngram_heads
        gather_shape = (
            ["num_tokens", ngram_heads, head_dim]
            if self.use_paged_attention
            else ["batch_size", "sequence_length", ngram_heads, head_dim]
        )
        self.make_node(
            "GatherBlockQuantized",
            inputs=[table_name, ngram_ids, "model.ple.ngram_embedding.weight_scale"],
            outputs=[f"{gather_name}/output_0"],
            name=gather_name,
            domain="com.microsoft",
            metadata_props={"layer_ann": self.CPU_EMBEDDING_ANNOTATION},
            gather_axis=0,
            quantize_axis=1,
            block_size=0,
        )
        self.make_value(f"{gather_name}/output_0", self.io_dtype, gather_shape)
        flatten_name = f"{basename}/ngram_embedding/Reshape"
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        flatten_dims = [-1, self.ple_embed_dim] if self.use_paged_attention else [0, 0, self.ple_embed_dim]
        self.make_reshape(
            flatten_name,
            [f"{gather_name}/output_0", f"/model/constants/INT64/{flatten_dims}"],
            self.io_dtype,
            [*token_shape, self.ple_embed_dim],
        )

        key_scale = f"model.layers.{layer_id}.ple.key_norm_scale"
        query_scale = f"model.layers.{layer_id}.ple.query_norm_scale"
        conv_scale = f"model.layers.{layer_id}.ple.conv_norm_scale"
        self.make_initializer((ple.norm_key.weight + 1).reshape(self.hc_count, self.hidden_size), key_scale, to=self.io_dtype)
        self.make_initializer((ple.norm_query.weight + 1).reshape(self.hc_count, self.hidden_size), query_scale, to=self.io_dtype)
        self.make_initializer((ple.norm_conv.weight + 1).reshape(self.hc_count, self.hidden_size), conv_scale, to=self.io_dtype)
        key_matmul = self.make_matmul(ple.key_proj, f"{basename}/key_proj/MatMul", f"{flatten_name}/output_0")
        value_matmul = self.make_matmul(ple.value_proj, f"{basename}/value_proj/MatMul", f"{flatten_name}/output_0")
        grouped_shape = [*token_shape, self.hc_count, self.hidden_size]
        key_reshape = f"{basename}/key_proj/Reshape"
        query_reshape = f"{basename}/query/Reshape"
        grouped_dims = [-1, self.hc_count, self.hidden_size] if self.use_paged_attention else [0, 0, self.hc_count, self.hidden_size]
        self.make_reshape(
            key_reshape,
            [f"{key_matmul}/output_0", f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        self.make_reshape(
            query_reshape,
            [root_input, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        gate_name = f"{basename}/EngramGate"
        gated_value = f"{gate_name}/output_0"
        gated_value_normed = f"{gate_name}/output_1"
        self.make_node(
            "EngramGate",
            inputs=[
                f"{key_reshape}/output_0",
                f"{query_reshape}/output_0",
                f"{value_matmul}/output_0",
                key_scale,
                query_scale,
                conv_scale,
            ],
            outputs=[gated_value, gated_value_normed],
            name=gate_name,
            domain="com.microsoft",
            epsilon=self.layernorm_attrs["epsilon"],
        )
        self.make_value(gated_value, self.io_dtype, grouped_shape)
        self.make_value(gated_value_normed, self.io_dtype, grouped_shape)
        ple_shape = [*token_shape, self.hc_hidden_size]
        gated_value_flat = f"{gate_name}/Flatten"
        gated_value_normed_flat = f"{gate_name}/FlattenNormed"
        self.make_reshape(
            gated_value_flat,
            [gated_value, f"/model/constants/INT64/{flatten_dims[:-1] + [self.hc_hidden_size]}"],
            self.io_dtype,
            ple_shape,
        )
        self.make_reshape(
            gated_value_normed_flat,
            [gated_value_normed, f"/model/constants/INT64/{flatten_dims[:-1] + [self.hc_hidden_size]}"],
            self.io_dtype,
            ple_shape,
        )

        conv_weight = f"model.layers.{layer_id}.ple.conv1d.weight"
        self.make_initializer(ple.conv1d.weight, conv_weight, to=self.io_dtype)
        conv_name = f"{basename}/CausalConvWithState"
        conv_output = f"{conv_name}/output_0"
        self.make_node(
            "CausalConvWithState",
            inputs=[
                f"{gated_value_normed_flat}/output_0",
                conv_weight,
                "",
                self.input_names["past.ple_conv"][layer_id],
            ],
            outputs=[conv_output, self.output_names["present.ple_conv"][layer_id]],
            name=conv_name,
            domain="com.microsoft",
            ndim=1,
            dilation=self.ple_conv_dilation,
            channels_last=1,
            activation="silu",
        )
        self.make_value(conv_output, self.io_dtype, ple_shape)
        add_name = f"{basename}/Add"
        self.make_add(add_name, [f"{gated_value_flat}/output_0", conv_output], self.io_dtype, ple_shape)
        return f"{add_name}/output_0"

    def make_qwen_sparse_attention(self, layer_id, attention, root_input):
        self.make_attention_input_proj(layer_id, attention, root_input)
        q_norm_weight, k_norm_weight = self.get_qk_norm_weight_names(layer_id)
        self.make_initializer(attention.q_norm.weight + 1, q_norm_weight, to=self.io_dtype)
        self.make_initializer(attention.k_norm.weight + 1, k_norm_weight, to=self.io_dtype)
        cos_cache, sin_cache = self.make_rotary_embedding_caches()
        past_k, past_v, present_k, present_v = self.make_key_value_cache_names(layer_id)
        capacity = self.indexer_budget + self.indexer_compress_ratio - 1

        if self.use_paged_attention:
            selected_indices = self.input_names["sparse_attention.selected_indices"][layer_id]
            selected_counts = self.input_names["sparse_attention.selected_counts"][layer_id]
        else:
            index_q_size = self.indexer_num_heads * self.indexer_head_dim
            index_k_size = self.indexer_kv_heads * self.indexer_head_dim
            index_matmul = self.make_matmul(
                attention.indexer.index_qk_proj,
                f"/model/layers.{layer_id}/attn/indexer/index_qk_proj/MatMul",
                root_input,
            )
            index_q = f"/model/layers.{layer_id}/attn/indexer/query"
            index_k = f"/model/layers.{layer_id}/attn/indexer/key"
            self.make_split(
                f"/model/layers.{layer_id}/attn/indexer/Split",
                [f"{index_matmul}/output_0", f"/model/constants/INT64/[{index_q_size}, {index_k_size}]"],
                [index_q, index_k],
                [self.io_dtype, self.io_dtype],
                [
                    ["batch_size", "sequence_length", index_q_size],
                    ["batch_size", "sequence_length", index_k_size],
                ],
            )
            index_q_4d = f"/model/layers.{layer_id}/attn/indexer/query/Reshape"
            self.make_reshape(
                index_q_4d,
                [index_q, f"/model/constants/INT64/[0, 0, {self.indexer_num_heads}, {self.indexer_head_dim}]"],
                self.io_dtype,
                ["batch_size", "sequence_length", self.indexer_num_heads, self.indexer_head_dim],
            )
            index_q_scale = f"model.layers.{layer_id}.attn.indexer.q_norm.weight"
            index_k_scale = f"model.layers.{layer_id}.attn.indexer.k_norm.weight"
            self.make_initializer(attention.indexer.q_layernorm.weight + 1, index_q_scale, to=self.io_dtype)
            self.make_initializer(attention.indexer.k_layernorm.weight + 1, index_k_scale, to=self.io_dtype)
            index_q_norm = f"/model/layers.{layer_id}/attn/indexer/query/SimplifiedLayerNormalization"
            self.make_node(
                "SimplifiedLayerNormalization",
                inputs=[f"{index_q_4d}/output_0", index_q_scale],
                outputs=[f"{index_q_norm}/output_0"],
                name=index_q_norm,
                axis=-1,
                epsilon=self.layernorm_attrs["epsilon"],
                stash_type=1,
            )
            self.make_value(
                f"{index_q_norm}/output_0",
                self.io_dtype,
                ["batch_size", "sequence_length", self.indexer_num_heads, self.indexer_head_dim],
            )
            index_cos, index_sin = self.make_qsa_rotary_caches(
                layer_id, root_input, cos_cache, sin_cache
            )
            visibility_mask = self.make_qsa_visibility_mask(layer_id, root_input)
            indexer_name = f"/model/layers.{layer_id}/attn/SparseAttentionIndexer"
            selected_indices = f"{indexer_name}/output_0"
            self.make_node(
                "SparseAttentionIndexer",
                inputs=[
                    f"{index_q_norm}/output_0",
                    index_k,
                    index_k_scale,
                    index_cos,
                    index_sin,
                    visibility_mask,
                    self.input_names["past.indexer"][layer_id],
                ],
                outputs=[selected_indices, self.output_names["present.indexer"][layer_id]],
                name=indexer_name,
                domain="com.microsoft",
                policy_mode="qsa",
                compress_ratio=self.indexer_compress_ratio,
                token_budget=self.indexer_budget,
                epsilon=self.layernorm_attrs["epsilon"],
                scale=self.indexer_head_dim**-0.5,
            )
            self.make_value(
                selected_indices,
                ir.DataType.INT32,
                ["batch_size", "sequence_length", capacity],
            )
            selected_counts = self.make_selected_counts(layer_id, selected_indices, capacity, packed=False)
            selected_indices_flat = f"{indexer_name}/Flatten"
            selected_counts_flat = f"{indexer_name}/CountsFlatten"
            self.make_reshape(
                selected_indices_flat,
                [selected_indices, f"/model/constants/INT64/[-1, {capacity}]"],
                ir.DataType.INT32,
                ["batch_size * sequence_length", capacity],
            )
            self.make_reshape(
                selected_counts_flat,
                [selected_counts, "/model/constants/INT64/[-1]"],
                ir.DataType.INT32,
                ["batch_size * sequence_length"],
            )
            selected_indices = f"{selected_indices_flat}/output_0"
            selected_counts = f"{selected_counts_flat}/output_0"

        op_type = "SparsePagedAttention" if self.use_paged_attention else "DynamicSparseAttention"
        name = f"/model/layers.{layer_id}/attn/{op_type}"
        if self.use_paged_attention:
            inputs = [
                self.attention_attrs["q_path"],
                self.attention_attrs["k_path"],
                self.attention_attrs["v_path"],
                past_k,
                past_v,
                self.input_names["cumulative_sequence_lengths"],
                self.input_names["past_sequence_lengths"],
                self.input_names["block_table"],
                "",
                selected_indices,
                selected_counts,
                "",
                "",
                "",
                cos_cache,
                sin_cache,
                "",
                q_norm_weight,
                k_norm_weight,
                "",
                "",
                self.input_names["attention_metadata"],
            ]
        else:
            attention_position_ids = f"{name}/position_ids/Gather"
            self.make_gather(
                attention_position_ids,
                [self.input_names["position_ids"], "/model/constants/INT64/0"],
                ir.DataType.INT64,
                ["batch_size", "sequence_length"],
                axis=0,
            )
            inputs = [
                self.attention_attrs["q_path"],
                self.attention_attrs["k_path"],
                self.attention_attrs["v_path"],
                past_k,
                past_v,
                "",
                "",
                selected_indices,
                selected_counts,
                f"{self.mask_attrs['seqlens_k']}/output_0",
                f"{self.mask_attrs['total_seq_len']}/output_0",
                cos_cache,
                sin_cache,
                f"{attention_position_ids}/output_0",
                q_norm_weight,
                k_norm_weight,
                "",
            ]
        outputs = [
            f"{name}/output_0",
            present_k,
            present_v,
        ]
        attributes = dict(
            num_heads=self.num_attn_heads,
            kv_num_heads=self.num_kv_heads,
            scale=self.attention_attrs["scale"],
            is_causal=1,
            attention_mode="selected_only",
            selected_kv_source="main",
            do_rotary=1,
            rotary_interleaved=self.rope_attrs["interleaved"],
            qk_norm_epsilon=self.attention_attrs["qk_norm_epsilon"],
        )
        if self.use_paged_attention and self.attention_attrs["softcap"] is not None:
            attributes["softcap"] = self.attention_attrs["softcap"]
        self.make_node(
            op_type,
            inputs=inputs,
            outputs=outputs,
            name=name,
            domain="com.microsoft",
            **attributes,
        )
        self.make_value(
            f"{name}/output_0",
            self.io_dtype,
            self.make_hidden_state_shape(last_dim=self.num_attn_heads * self.head_size),
        )
        self.attention_attrs["o_path"] = f"{name}/output_0"
        self.make_attention_output_proj(layer_id, attention, root_input)

    def make_layer(self, layer_id, layer):
        if layer_id == 0 and self.tile_first_hidden_state:
            tile_name = "/model/hyper_connection/Tile"
            tile_repeats = [1, self.hc_count] if self.use_paged_attention else [1, 1, self.hc_count]
            self.make_tile(
                tile_name,
                [self.layernorm_attrs["root_input"], f"/model/constants/INT64/{tile_repeats}"],
                self.io_dtype,
                self.make_hidden_state_shape(last_dim=self.hc_hidden_size),
            )
            self.layernorm_attrs["root_input"] = f"{tile_name}/output_0"

        hyper_states = self.layernorm_attrs["root_input"]
        if layer_id in self.ple_layer_ids:
            ple_output = self.make_ple(layer_id, layer.ple, hyper_states)
            ple_add = f"/model/layers.{layer_id}/ple/residual/Add"
            self.make_add(
                ple_add,
                [hyper_states, ple_output],
                self.io_dtype,
                self.make_hidden_state_shape(last_dim=self.hc_hidden_size),
            )
            hyper_states = f"{ple_add}/output_0"

        mixed, residual, injection = self.make_hyper_connection_mix(
            layer_id, layer.attn_hyper_connection, hyper_states, "attn"
        )
        attention = self.get_attn_module(layer_id, layer)
        if self.layer_types[layer_id] == "linear_attention":
            self.make_qwen_gated_delta_net(layer_id, attention, mixed)
        else:
            self.make_qwen_sparse_attention(layer_id, attention, mixed)
        hyper_states = self.make_hyper_connection_injection(
            layer_id, self.layernorm_attrs["skip_input"], residual, injection, "attn"
        )

        mixed, residual, injection = self.make_hyper_connection_mix(
            layer_id, layer.mlp_hyper_connection, hyper_states, "mlp"
        )
        moe = self.get_moe_module(layer_id, layer)
        self.make_moe_preprocessing(layer_id, moe, mixed)
        self.make_moe_router(layer_id, moe, mixed)
        moe_output = self.make_moe_subgraph(layer_id, moe, mixed)
        hyper_states = self.make_hyper_connection_injection(
            layer_id, moe_output, residual, injection, "mlp"
        )
        self.layernorm_attrs["root_input"] = hyper_states
        self.layernorm_attrs["skip_input"] = hyper_states

        if layer_id == self.num_layers - 1:
            final_output = self.make_hyper_connection_mix(
                self.num_layers,
                self.get_final_hyper_connection_mixer(),
                hyper_states,
                "final",
                combine=False,
            )
            if self.include_hidden_states or self.exclude_lm_head:
                self.make_node(
                    "Identity",
                    inputs=[hyper_states if self.emit_pre_final_hidden_states else final_output],
                    outputs=[self.output_names["hidden_states"]],
                    name="/model/final_hidden_states/Identity",
                )
                if not self.emit_pre_final_hidden_states:
                    final_output = self.output_names["hidden_states"]
            self.layernorm_attrs["output_0"] = final_output

    def get_final_hyper_connection_mixer(self):
        return self.weights.model.language_model.hyper_connection_mixer


class _Qwen4ExpGraphModel(Model):
    def __init__(self, io_dtype, filename, graph_name):
        self.io_dtype = ir.DataType(io_dtype)
        self.filename = filename
        self.graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 22}, name=graph_name)
        self.model = ir.Model(self.graph, ir_version=10, producer_name="onnxruntime-genai")
        self.values = {}
        self.node_names = set()

    def save_model(self, output_dir):
        ir.save(
            self.model,
            os.path.join(output_dir, self.filename),
            external_data=f"{self.filename}.data",
            size_threshold_bytes=0,
        )

    def make_linear(self, name, linear, root_input, shape, output=None):
        weight_name = f"{name}.weight"
        self.make_initializer(linear.weight.T, weight_name, to=self.io_dtype)
        matmul_name = f"/{name.replace('.', '/')}/MatMul"
        matmul_output = f"{matmul_name}/output_0"
        self.make_node("MatMul", [root_input, weight_name], [matmul_output], name=matmul_name)
        self.make_value(matmul_output, self.io_dtype, shape)
        if linear.bias is None:
            if output is not None:
                self.make_node("Identity", [matmul_output], [output], name=f"/{name.replace('.', '/')}/Identity")
                self.make_value(output, self.io_dtype, shape)
                return output
            return matmul_output
        bias_name = f"{name}.bias"
        self.make_initializer(linear.bias, bias_name, to=self.io_dtype)
        add_name = f"/{name.replace('.', '/')}/Add"
        add_output = output or f"{add_name}/output_0"
        self.make_node("Add", [matmul_output, bias_name], [add_output], name=add_name)
        self.make_value(add_output, self.io_dtype, shape)
        return add_output

    def make_layer_norm(self, name, layer_norm, root_input, shape):
        scale_name = f"{name}.weight"
        bias_name = f"{name}.bias"
        self.make_initializer(layer_norm.weight, scale_name, to=self.io_dtype)
        self.make_initializer(layer_norm.bias, bias_name, to=self.io_dtype)
        node_name = f"/{name.replace('.', '/')}/LayerNormalization"
        output = f"{node_name}/output_0"
        self.make_node(
            "LayerNormalization",
            [root_input, scale_name, bias_name],
            [output],
            name=node_name,
            axis=-1,
            epsilon=layer_norm.eps,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape)
        return output

    def make_binary(self, op_type, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node(op_type, inputs, [output], name=name)
        self.make_value(output, dtype, shape)
        return output

    def make_cast(self, name, root_input, dtype, shape):
        super().make_cast(name, root_input, dtype, shape)
        return f"{name}/output_0"


class Qwen4ExpEmbeddingModel(_Qwen4ExpGraphModel):
    def __init__(self, config, embedding_weight, io_dtype):
        super().__init__(io_dtype, "embedding.onnx", "qwen4_exp_embedding")
        hidden_size = embedding_weight.shape[1]
        input_ids = self.make_value("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        image_features = self.make_value("image_features", self.io_dtype, ["num_image_tokens", hidden_size])
        inputs_embeds = self.make_value(
            "inputs_embeds", self.io_dtype, ["batch_size", "sequence_length", hidden_size]
        )
        self.graph.inputs.extend([input_ids, image_features])
        self.graph.outputs.append(inputs_embeds)

        weight_name = "model.embed_tokens.weight"
        self.make_initializer(embedding_weight, weight_name, to=self.io_dtype)
        image_token = "image_token_id"
        video_token = "video_token_id"
        self.make_initializer(torch.tensor(config.image_token_id, dtype=torch.int64), image_token)
        self.make_initializer(torch.tensor(config.video_token_id, dtype=torch.int64), video_token)
        gathered = "/model/embed_tokens/Gather/output_0"
        self.make_node(
            "Gather", [weight_name, "input_ids"], [gathered], name="/model/embed_tokens/Gather", axis=0
        )
        self.make_value(gathered, self.io_dtype, ["batch_size", "sequence_length", hidden_size])
        image_mask = self.make_binary(
            "Equal",
            "/model/image_mask/Equal",
            ["input_ids", image_token],
            ir.DataType.BOOL,
            ["batch_size", "sequence_length"],
        )
        video_mask = self.make_binary(
            "Equal",
            "/model/video_mask/Equal",
            ["input_ids", video_token],
            ir.DataType.BOOL,
            ["batch_size", "sequence_length"],
        )
        multimodal_mask = self.make_binary(
            "Or",
            "/model/multimodal_mask/Or",
            [image_mask, video_mask],
            ir.DataType.BOOL,
            ["batch_size", "sequence_length"],
        )
        indices = "/model/multimodal_indices/NonZero/output_0"
        self.make_node("NonZero", [multimodal_mask], [indices], name="/model/multimodal_indices/NonZero")
        self.make_value(indices, ir.DataType.INT64, [2, "num_image_tokens"])
        scatter_indices = "/model/multimodal_indices/Transpose/output_0"
        self.make_node(
            "Transpose",
            [indices],
            [scatter_indices],
            name="/model/multimodal_indices/Transpose",
            perm=[1, 0],
        )
        self.make_value(scatter_indices, ir.DataType.INT64, ["num_image_tokens", 2])
        self.make_node(
            "ScatterND",
            [gathered, scatter_indices, "image_features"],
            ["inputs_embeds"],
            name="/model/merge_embeddings/ScatterND",
        )


class Qwen4ExpVisionModel(_Qwen4ExpGraphModel):
    def __init__(self, config, visual, io_dtype):
        super().__init__(io_dtype, "vision.onnx", "qwen4_exp_vision")
        self.config = config
        self.visual = visual
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads
        self.merge_size = config.spatial_merge_size
        patch_size = config.patch_size[0] if isinstance(config.patch_size, (list, tuple)) else config.patch_size
        temporal_size = (
            config.temporal_patch_size[0]
            if isinstance(config.temporal_patch_size, (list, tuple))
            else config.temporal_patch_size
        )
        self.patch_dim = config.in_channels * temporal_size * patch_size * patch_size
        self.make_model()

    def make_unary(self, op_type, name, root_input, dtype, shape, **attributes):
        output = f"{name}/output_0"
        self.make_node(op_type, [root_input], [output], name=name, **attributes)
        self.make_value(output, dtype, shape)
        return output

    def make_grid_values(self):
        flat = "/vision/grid/Reshape/output_0"
        self.make_reshape(
            "/vision/grid/Reshape",
            ["image_grid_thw", "/model/constants/INT64/[-1]"],
            ir.DataType.INT64,
            [3],
        )
        values = []
        for index, label in enumerate(("t", "h", "w")):
            name = f"/vision/grid/{label}/Gather"
            self.make_gather(
                name,
                [flat, f"/model/constants/INT64/{index}"],
                ir.DataType.INT64,
                [],
                axis=0,
            )
            values.append(f"{name}/output_0")
        return values

    def make_patch_positions(self, num_patches, height, width):
        positions = "/vision/positions/Range/output_0"
        self.make_node(
            "Range",
            ["/model/constants/INT64/0", num_patches, "/model/constants/INT64/1"],
            [positions],
            name="/vision/positions/Range",
        )
        self.make_value(positions, ir.DataType.INT64, ["num_patches"])
        frame_size = self.make_binary(
            "Mul", "/vision/grid/frame_size/Mul", [height, width], ir.DataType.INT64, []
        )
        within = self.make_binary(
            "Mod", "/vision/positions/within/Mod", [positions, frame_size], ir.DataType.INT64, ["num_patches"]
        )
        merge = f"/model/constants/INT64/{self.merge_size}"
        merge_sq = f"/model/constants/INT64/{self.merge_size * self.merge_size}"
        blocks_w = self.make_binary("Div", "/vision/grid/blocks_w/Div", [width, merge], ir.DataType.INT64, [])
        in_col = self.make_binary(
            "Mod", "/vision/positions/in_col/Mod", [within, merge], ir.DataType.INT64, ["num_patches"]
        )
        within_div_merge = self.make_binary(
            "Div",
            "/vision/positions/within_div_merge/Div",
            [within, merge],
            ir.DataType.INT64,
            ["num_patches"],
        )
        in_row = self.make_binary(
            "Mod",
            "/vision/positions/in_row/Mod",
            [within_div_merge, merge],
            ir.DataType.INT64,
            ["num_patches"],
        )
        within_div_block = self.make_binary(
            "Div",
            "/vision/positions/within_div_block/Div",
            [within, merge_sq],
            ir.DataType.INT64,
            ["num_patches"],
        )
        block_col = self.make_binary(
            "Mod",
            "/vision/positions/block_col/Mod",
            [within_div_block, blocks_w],
            ir.DataType.INT64,
            ["num_patches"],
        )
        row_denominator = self.make_binary(
            "Mul", "/vision/positions/row_denominator/Mul", [blocks_w, merge_sq], ir.DataType.INT64, []
        )
        block_row = self.make_binary(
            "Div",
            "/vision/positions/block_row/Div",
            [within, row_denominator],
            ir.DataType.INT64,
            ["num_patches"],
        )
        row_base = self.make_binary(
            "Mul", "/vision/positions/row_base/Mul", [block_row, merge], ir.DataType.INT64, ["num_patches"]
        )
        col_base = self.make_binary(
            "Mul", "/vision/positions/col_base/Mul", [block_col, merge], ir.DataType.INT64, ["num_patches"]
        )
        row = self.make_binary(
            "Add", "/vision/positions/row/Add", [row_base, in_row], ir.DataType.INT64, ["num_patches"]
        )
        col = self.make_binary(
            "Add", "/vision/positions/col/Add", [col_base, in_col], ir.DataType.INT64, ["num_patches"]
        )
        return row, col

    def make_axis_interpolation(self, label, position, size):
        position_float = self.make_cast(
            f"/vision/interpolation/{label}/position/Cast", position, ir.DataType.FLOAT, ["num_patches"]
        )
        size_minus_one = self.make_binary(
            "Sub",
            f"/vision/interpolation/{label}/size_minus_one/Sub",
            [size, "/model/constants/INT64/1"],
            ir.DataType.INT64,
            [],
        )
        denominator = self.make_binary(
            "Max",
            f"/vision/interpolation/{label}/denominator/Max",
            [size_minus_one, "/model/constants/INT64/1"],
            ir.DataType.INT64,
            [],
        )
        denominator_float = self.make_cast(
            f"/vision/interpolation/{label}/denominator/Cast", denominator, ir.DataType.FLOAT, []
        )
        scaled = self.make_binary(
            "Mul",
            f"/vision/interpolation/{label}/scaled/Mul",
            [position_float, f"/model/constants/FLOAT/{self.visual.num_grid_per_side - 1}.0"],
            ir.DataType.FLOAT,
            ["num_patches"],
        )
        source = self.make_binary(
            "Div",
            f"/vision/interpolation/{label}/source/Div",
            [scaled, denominator_float],
            ir.DataType.FLOAT,
            ["num_patches"],
        )
        floor_float = self.make_unary(
            "Floor", f"/vision/interpolation/{label}/floor/Floor", source, ir.DataType.FLOAT, ["num_patches"]
        )
        floor_int = self.make_cast(
            f"/vision/interpolation/{label}/floor/Cast", floor_float, ir.DataType.INT64, ["num_patches"]
        )
        ceil_int = self.make_binary(
            "Add",
            f"/vision/interpolation/{label}/ceil/Add",
            [floor_int, "/model/constants/INT64/1"],
            ir.DataType.INT64,
            ["num_patches"],
        )
        taps = []
        for tap_name, tap in (("floor", floor_int), ("ceil", ceil_int)):
            clipped = f"/vision/interpolation/{label}/{tap_name}/Clip/output_0"
            self.make_node(
                "Clip",
                [
                    tap,
                    "/model/constants/INT64/0",
                    f"/model/constants/INT64/{self.visual.num_grid_per_side - 1}",
                ],
                [clipped],
                name=f"/vision/interpolation/{label}/{tap_name}/Clip",
            )
            self.make_value(clipped, ir.DataType.INT64, ["num_patches"])
            unsqueezed = f"/vision/interpolation/{label}/{tap_name}/Unsqueeze/output_0"
            self.make_unsqueeze(
                f"/vision/interpolation/{label}/{tap_name}/Unsqueeze",
                [clipped, "/model/constants/INT64/[1]"],
                ir.DataType.INT64,
                ["num_patches", 1],
            )
            taps.append(unsqueezed)
        tap_indices = f"/vision/interpolation/{label}/taps/Concat/output_0"
        self.make_concat(
            f"/vision/interpolation/{label}/taps/Concat",
            taps,
            ir.DataType.INT64,
            ["num_patches", 2],
            axis=1,
        )
        fraction = self.make_binary(
            "Sub",
            f"/vision/interpolation/{label}/fraction/Sub",
            [source, floor_float],
            ir.DataType.FLOAT,
            ["num_patches"],
        )
        lower_weight = self.make_binary(
            "Sub",
            f"/vision/interpolation/{label}/lower_weight/Sub",
            ["/model/constants/FLOAT/1.0", fraction],
            ir.DataType.FLOAT,
            ["num_patches"],
        )
        weight_parts = []
        for weight_name, weight in (("lower", lower_weight), ("upper", fraction)):
            unsqueezed = f"/vision/interpolation/{label}/{weight_name}_weight/Unsqueeze/output_0"
            self.make_unsqueeze(
                f"/vision/interpolation/{label}/{weight_name}_weight/Unsqueeze",
                [weight, "/model/constants/INT64/[1]"],
                ir.DataType.FLOAT,
                ["num_patches", 1],
            )
            weight_parts.append(unsqueezed)
        tap_weights = f"/vision/interpolation/{label}/weights/Concat/output_0"
        self.make_concat(
            f"/vision/interpolation/{label}/weights/Concat",
            weight_parts,
            ir.DataType.FLOAT,
            ["num_patches", 2],
            axis=1,
        )
        return tap_indices, tap_weights

    def make_position_embeddings(self, row, col, height, width):
        h_taps, h_weights = self.make_axis_interpolation("h", row, height)
        w_taps, w_weights = self.make_axis_interpolation("w", col, width)
        h_taps_3d = "/vision/interpolation/h/taps/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/h/taps/Unsqueeze",
            [h_taps, "/model/constants/INT64/[2]"],
            ir.DataType.INT64,
            ["num_patches", 2, 1],
        )
        w_taps_3d = "/vision/interpolation/w/taps/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/w/taps/Unsqueeze",
            [w_taps, "/model/constants/INT64/[1]"],
            ir.DataType.INT64,
            ["num_patches", 1, 2],
        )
        h_offset = self.make_binary(
            "Mul",
            "/vision/interpolation/h_offset/Mul",
            [h_taps_3d, f"/model/constants/INT64/{self.visual.num_grid_per_side}"],
            ir.DataType.INT64,
            ["num_patches", 2, 1],
        )
        indices_3d = self.make_binary(
            "Add",
            "/vision/interpolation/indices/Add",
            [h_offset, w_taps_3d],
            ir.DataType.INT64,
            ["num_patches", 2, 2],
        )
        indices = "/vision/interpolation/indices/Reshape/output_0"
        self.make_reshape(
            "/vision/interpolation/indices/Reshape",
            [indices_3d, "/model/constants/INT64/[-1, 4]"],
            ir.DataType.INT64,
            ["num_patches", 4],
        )
        h_weights_3d = "/vision/interpolation/h/weights/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/h/weights/Unsqueeze",
            [h_weights, "/model/constants/INT64/[2]"],
            ir.DataType.FLOAT,
            ["num_patches", 2, 1],
        )
        w_weights_3d = "/vision/interpolation/w/weights/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/w/weights/Unsqueeze",
            [w_weights, "/model/constants/INT64/[1]"],
            ir.DataType.FLOAT,
            ["num_patches", 1, 2],
        )
        weights_3d = self.make_binary(
            "Mul",
            "/vision/interpolation/weights/Mul",
            [h_weights_3d, w_weights_3d],
            ir.DataType.FLOAT,
            ["num_patches", 2, 2],
        )
        weights = "/vision/interpolation/weights/Reshape/output_0"
        self.make_reshape(
            "/vision/interpolation/weights/Reshape",
            [weights_3d, "/model/constants/INT64/[-1, 4]"],
            ir.DataType.FLOAT,
            ["num_patches", 4],
        )
        table_name = "visual.pos_embed.weight"
        self.make_initializer(self.visual.pos_embed.weight, table_name, to=self.io_dtype)
        gathered = "/vision/pos_embed/Gather/output_0"
        self.make_gather(
            "/vision/pos_embed/Gather",
            [table_name, indices],
            self.io_dtype,
            ["num_patches", 4, self.hidden_size],
            axis=0,
        )
        if self.io_dtype != ir.DataType.FLOAT:
            weights = self.make_cast(
                "/vision/interpolation/weights/Cast",
                weights,
                self.io_dtype,
                ["num_patches", 4],
            )
        weights_expanded = "/vision/interpolation/weights/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/interpolation/weights/Unsqueeze",
            [weights, "/model/constants/INT64/[-1]"],
            self.io_dtype,
            ["num_patches", 4, 1],
        )
        weighted = self.make_binary(
            "Mul",
            "/vision/pos_embed/Mul",
            [gathered, weights_expanded],
            self.io_dtype,
            ["num_patches", 4, self.hidden_size],
        )
        output = "/vision/pos_embed/ReduceSum/output_0"
        self.make_reduce_sum(
            "/vision/pos_embed/ReduceSum",
            [weighted, "/model/constants/INT64/[1]"],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        return output

    def make_rotary_embeddings(self, row, col):
        row_float = self.make_cast("/vision/rotary/row/Cast", row, ir.DataType.FLOAT, ["num_patches"])
        col_float = self.make_cast("/vision/rotary/col/Cast", col, ir.DataType.FLOAT, ["num_patches"])
        row_2d = "/vision/rotary/row/Unsqueeze/output_0"
        col_2d = "/vision/rotary/col/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/rotary/row/Unsqueeze",
            [row_float, "/model/constants/INT64/[1]"],
            ir.DataType.FLOAT,
            ["num_patches", 1],
        )
        self.make_unsqueeze(
            "/vision/rotary/col/Unsqueeze",
            [col_float, "/model/constants/INT64/[1]"],
            ir.DataType.FLOAT,
            ["num_patches", 1],
        )
        position_ids = "/vision/rotary/position_ids/Concat/output_0"
        self.make_concat(
            "/vision/rotary/position_ids/Concat",
            [row_2d, col_2d],
            ir.DataType.FLOAT,
            ["num_patches", 2],
            axis=1,
        )
        position_ids_3d = "/vision/rotary/position_ids/Unsqueeze/output_0"
        self.make_unsqueeze(
            "/vision/rotary/position_ids/Unsqueeze",
            [position_ids, "/model/constants/INT64/[-1]"],
            ir.DataType.FLOAT,
            ["num_patches", 2, 1],
        )
        inv_freq_name = "visual.rotary_pos_emb.inv_freq"
        self.make_initializer(self.visual.rotary_pos_emb.inv_freq, inv_freq_name, to=ir.DataType.FLOAT)
        frequencies = self.make_binary(
            "Mul",
            "/vision/rotary/frequencies/Mul",
            [position_ids_3d, inv_freq_name],
            ir.DataType.FLOAT,
            ["num_patches", 2, self.head_size // 4],
        )
        flattened = "/vision/rotary/frequencies/Reshape/output_0"
        self.make_reshape(
            "/vision/rotary/frequencies/Reshape",
            [frequencies, f"/model/constants/INT64/[-1, {self.head_size // 2}]"],
            ir.DataType.FLOAT,
            ["num_patches", self.head_size // 2],
        )
        full = "/vision/rotary/frequencies/Concat/output_0"
        self.make_concat(
            "/vision/rotary/frequencies/Concat",
            [flattened, flattened],
            ir.DataType.FLOAT,
            ["num_patches", self.head_size],
            axis=1,
        )
        cos = self.make_unary("Cos", "/vision/rotary/Cos", full, ir.DataType.FLOAT, ["num_patches", self.head_size])
        sin = self.make_unary("Sin", "/vision/rotary/Sin", full, ir.DataType.FLOAT, ["num_patches", self.head_size])
        if self.io_dtype != ir.DataType.FLOAT:
            cos = self.make_cast("/vision/rotary/cos/Cast", cos, self.io_dtype, ["num_patches", self.head_size])
            sin = self.make_cast("/vision/rotary/sin/Cast", sin, self.io_dtype, ["num_patches", self.head_size])
        return cos, sin

    def apply_rotary(self, layer_id, label, tensor, cos, sin):
        basename = f"/visual/blocks/{layer_id}/attn/{label}_rotary"
        first = f"{basename}/Split/output_0"
        second = f"{basename}/Split/output_1"
        self.make_split(
            f"{basename}/Split",
            [tensor, f"/model/constants/INT64/[{self.head_size // 2}, {self.head_size // 2}]"],
            [first, second],
            [self.io_dtype, self.io_dtype],
            [["num_patches", self.num_heads, self.head_size // 2]] * 2,
            axis=-1,
        )
        negative = self.make_unary(
            "Neg", f"{basename}/Neg", second, self.io_dtype, ["num_patches", self.num_heads, self.head_size // 2]
        )
        rotated = f"{basename}/Concat/output_0"
        self.make_concat(
            f"{basename}/Concat",
            [negative, first],
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
            axis=-1,
        )
        cos_3d = f"{basename}/cos/Unsqueeze/output_0"
        sin_3d = f"{basename}/sin/Unsqueeze/output_0"
        self.make_unsqueeze(
            f"{basename}/cos/Unsqueeze",
            [cos, "/model/constants/INT64/[1]"],
            self.io_dtype,
            ["num_patches", 1, self.head_size],
        )
        self.make_unsqueeze(
            f"{basename}/sin/Unsqueeze",
            [sin, "/model/constants/INT64/[1]"],
            self.io_dtype,
            ["num_patches", 1, self.head_size],
        )
        direct = self.make_binary(
            "Mul",
            f"{basename}/direct/Mul",
            [tensor, cos_3d],
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
        )
        crossed = self.make_binary(
            "Mul",
            f"{basename}/crossed/Mul",
            [rotated, sin_3d],
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
        )
        return self.make_binary(
            "Add",
            f"{basename}/Add",
            [direct, crossed],
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
        )

    def make_attention(self, layer_id, attention, root_input, cos, sin):
        qkv = self.make_linear(
            f"visual.blocks.{layer_id}.attn.qkv",
            attention.qkv,
            root_input,
            ["num_patches", 3 * self.hidden_size],
        )
        qkv_4d = f"/visual/blocks/{layer_id}/attn/qkv/Reshape/output_0"
        self.make_reshape(
            f"/visual/blocks/{layer_id}/attn/qkv/Reshape",
            [qkv, f"/model/constants/INT64/[-1, 3, {self.num_heads}, {self.head_size}]"],
            self.io_dtype,
            ["num_patches", 3, self.num_heads, self.head_size],
        )
        split_outputs = [f"/visual/blocks/{layer_id}/attn/qkv/Split/output_{index}" for index in range(3)]
        self.make_split(
            f"/visual/blocks/{layer_id}/attn/qkv/Split",
            [qkv_4d, "/model/constants/INT64/[1, 1, 1]"],
            split_outputs,
            [self.io_dtype] * 3,
            [["num_patches", 1, self.num_heads, self.head_size]] * 3,
            axis=1,
        )
        squeezed = []
        for label, value in zip(("q", "k", "v"), split_outputs, strict=True):
            output = f"/visual/blocks/{layer_id}/attn/{label}/Squeeze/output_0"
            self.make_squeeze(
                f"/visual/blocks/{layer_id}/attn/{label}/Squeeze",
                [value, "/model/constants/INT64/[1]"],
                self.io_dtype,
                ["num_patches", self.num_heads, self.head_size],
            )
            squeezed.append(output)
        query = self.apply_rotary(layer_id, "q", squeezed[0], cos, sin)
        key = self.apply_rotary(layer_id, "k", squeezed[1], cos, sin)
        query_t = f"/visual/blocks/{layer_id}/attn/q/Transpose/output_0"
        key_t = f"/visual/blocks/{layer_id}/attn/k/Transpose/output_0"
        value_t = f"/visual/blocks/{layer_id}/attn/v/Transpose/output_0"
        self.make_transpose(
            f"/visual/blocks/{layer_id}/attn/q/Transpose",
            query,
            self.io_dtype,
            [self.num_heads, "num_patches", self.head_size],
            [1, 0, 2],
        )
        self.make_transpose(
            f"/visual/blocks/{layer_id}/attn/k/Transpose",
            key,
            self.io_dtype,
            [self.num_heads, self.head_size, "num_patches"],
            [1, 2, 0],
        )
        self.make_transpose(
            f"/visual/blocks/{layer_id}/attn/v/Transpose",
            squeezed[2],
            self.io_dtype,
            [self.num_heads, "num_patches", self.head_size],
            [1, 0, 2],
        )
        scores = self.make_binary(
            "MatMul",
            f"/visual/blocks/{layer_id}/attn/scores/MatMul",
            [query_t, key_t],
            self.io_dtype,
            [self.num_heads, "num_patches", "num_patches"],
        )
        scaled = self.make_binary(
            "Mul",
            f"/visual/blocks/{layer_id}/attn/scores/Mul",
            [scores, f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{attention.scaling}"],
            self.io_dtype,
            [self.num_heads, "num_patches", "num_patches"],
        )
        probabilities = f"/visual/blocks/{layer_id}/attn/Softmax/output_0"
        self.make_softmax(
            f"/visual/blocks/{layer_id}/attn/Softmax",
            scaled,
            self.io_dtype,
            [self.num_heads, "num_patches", "num_patches"],
            axis=-1,
        )
        context = self.make_binary(
            "MatMul",
            f"/visual/blocks/{layer_id}/attn/context/MatMul",
            [probabilities, value_t],
            self.io_dtype,
            [self.num_heads, "num_patches", self.head_size],
        )
        context_t = f"/visual/blocks/{layer_id}/attn/context/Transpose/output_0"
        self.make_transpose(
            f"/visual/blocks/{layer_id}/attn/context/Transpose",
            context,
            self.io_dtype,
            ["num_patches", self.num_heads, self.head_size],
            [1, 0, 2],
        )
        context_flat = f"/visual/blocks/{layer_id}/attn/context/Reshape/output_0"
        self.make_reshape(
            f"/visual/blocks/{layer_id}/attn/context/Reshape",
            [context_t, f"/model/constants/INT64/[-1, {self.hidden_size}]"],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        return self.make_linear(
            f"visual.blocks.{layer_id}.attn.proj",
            attention.proj,
            context_flat,
            ["num_patches", self.hidden_size],
        )

    def make_gelu(self, name, root_input, shape, approximate="none"):
        output = f"{name}/output_0"
        self.make_node("Gelu", [root_input], [output], name=name, approximate=approximate)
        self.make_value(output, self.io_dtype, shape)
        return output

    def make_model(self):
        pixel_values = self.make_value("pixel_values", self.io_dtype, ["num_patches", self.patch_dim])
        image_grid = self.make_value("image_grid_thw", ir.DataType.INT64, [1, 3])
        image_features = self.make_value("image_features", self.io_dtype, ["num_image_tokens", self.config.out_hidden_size])
        self.graph.inputs.extend([pixel_values, image_grid])
        self.graph.outputs.append(image_features)

        shape = "/vision/pixel_values/Shape/output_0"
        self.make_shape("/vision/pixel_values/Shape", "pixel_values", [2])
        self.make_gather(
            "/vision/pixel_values/num_patches/Gather",
            [shape, "/model/constants/INT64/0"],
            ir.DataType.INT64,
            [],
            axis=0,
        )
        num_patches = "/vision/pixel_values/num_patches/Gather/output_0"
        _, height, width = self.make_grid_values()
        row, col = self.make_patch_positions(num_patches, height, width)

        patch_weight = self.visual.patch_embed.proj.weight.reshape(self.hidden_size, -1).T
        patch_weight_name = "visual.patch_embed.proj.weight"
        patch_bias_name = "visual.patch_embed.proj.bias"
        self.make_initializer(patch_weight, patch_weight_name, to=self.io_dtype)
        self.make_initializer(self.visual.patch_embed.proj.bias, patch_bias_name, to=self.io_dtype)
        patch_matmul = self.make_binary(
            "MatMul",
            "/visual/patch_embed/proj/MatMul",
            ["pixel_values", patch_weight_name],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        hidden_states = self.make_binary(
            "Add",
            "/visual/patch_embed/proj/Add",
            [patch_matmul, patch_bias_name],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        position_embeddings = self.make_position_embeddings(row, col, height, width)
        hidden_states = self.make_binary(
            "Add",
            "/visual/patch_embed/add_position/Add",
            [hidden_states, position_embeddings],
            self.io_dtype,
            ["num_patches", self.hidden_size],
        )
        cos, sin = self.make_rotary_embeddings(row, col)

        for layer_id, block in enumerate(self.visual.blocks):
            norm1 = self.make_layer_norm(
                f"visual.blocks.{layer_id}.norm1", block.norm1, hidden_states, ["num_patches", self.hidden_size]
            )
            attention = self.make_attention(layer_id, block.attn, norm1, cos, sin)
            hidden_states = self.make_binary(
                "Add",
                f"/visual/blocks/{layer_id}/attn/residual/Add",
                [hidden_states, attention],
                self.io_dtype,
                ["num_patches", self.hidden_size],
            )
            norm2 = self.make_layer_norm(
                f"visual.blocks.{layer_id}.norm2", block.norm2, hidden_states, ["num_patches", self.hidden_size]
            )
            fc1 = self.make_linear(
                f"visual.blocks.{layer_id}.mlp.linear_fc1",
                block.mlp.linear_fc1,
                norm2,
                ["num_patches", self.config.intermediate_size],
            )
            activated = self.make_gelu(
                f"/visual/blocks/{layer_id}/mlp/Gelu",
                fc1,
                ["num_patches", self.config.intermediate_size],
                approximate="tanh",
            )
            fc2 = self.make_linear(
                f"visual.blocks.{layer_id}.mlp.linear_fc2",
                block.mlp.linear_fc2,
                activated,
                ["num_patches", self.hidden_size],
            )
            hidden_states = self.make_binary(
                "Add",
                f"/visual/blocks/{layer_id}/mlp/residual/Add",
                [hidden_states, fc2],
                self.io_dtype,
                ["num_patches", self.hidden_size],
            )

        merger = self.visual.merger
        merged_norm = self.make_layer_norm(
            "visual.merger.norm", merger.norm, hidden_states, ["num_patches", self.hidden_size]
        )
        merged_input = "/visual/merger/Reshape/output_0"
        merged_hidden_size = self.hidden_size * self.merge_size * self.merge_size
        self.make_reshape(
            "/visual/merger/Reshape",
            [merged_norm, f"/model/constants/INT64/[-1, {merged_hidden_size}]"],
            self.io_dtype,
            ["num_image_tokens", merged_hidden_size],
        )
        merger_fc1 = self.make_linear(
            "visual.merger.linear_fc1",
            merger.linear_fc1,
            merged_input,
            ["num_image_tokens", merged_hidden_size],
        )
        merger_activated = self.make_gelu(
            "/visual/merger/Gelu", merger_fc1, ["num_image_tokens", merged_hidden_size]
        )
        self.make_linear(
            "visual.merger.linear_fc2",
            merger.linear_fc2,
            merger_activated,
            ["num_image_tokens", self.config.out_hidden_size],
            output="image_features",
        )


class Qwen4ExpModel(MTPModel):
    """Composite builder that emits Qwen4-Exp vision, embedding, and text graphs."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__()
        self.config = config
        self.extra_options = copy.deepcopy(extra_options)
        decoder_options = self.make_mtp_init(config, self.extra_options)
        self.decoder = Qwen4ExpTextModel(
            copy.deepcopy(config), io_dtype, onnx_dtype, ep, cache_dir, decoder_options
        )
        self.decoder.model_type = "qwen3_5"
        self.mtp = None
        if self.mtp_attrs["build"]:
            self.decoder.emit_pre_final_hidden_states = True
            self.decoder.output_shapes["hidden_states"] = self.decoder.make_hidden_state_shape(
                last_dim=self.decoder.hc_hidden_size
            )
            self.make_mtp_model(config, io_dtype, onnx_dtype, ep, cache_dir, decoder_options)
        self.input_path = None

        text_config = config.text_config
        self.bos_token_id = text_config.bos_token_id
        self.eos_token_id = text_config.eos_token_id
        self.pad_token_id = text_config.pad_token_id
        self.vocab_size = self.decoder.vocab_size
        self.hf_token = self.decoder.hf_token
        self.hf_remote = self.decoder.hf_remote
        self.context_length = self.decoder.context_length
        self.exclude_embeds = self.decoder.exclude_embeds
        self.model_type = "qwen3_5"

    def make_mtp_init(self, config, extra_options):
        decoder_options = super().make_mtp_init(config, extra_options)
        num_mtp_layers = getattr(config.text_config, "mtp_num_hidden_layers", 0) or 0
        self.mtp_attrs["build"] = num_mtp_layers > 0 and not extra_options.get("exclude_mtp", False)
        self.mtp_attrs["shared_initializer_prefixes"] = ("lm_head.MatMul.",)
        if not self.mtp_attrs["build"]:
            return decoder_options
        if num_mtp_layers != 1:
            raise ValueError(f"Qwen4-Exp MTP export requires exactly one MTP layer, got {num_mtp_layers}.")
        incompatible_options = [
            option for option in ("exclude_lm_head", "prune_lm_head") if extra_options.get(option, False)
        ]
        if incompatible_options:
            raise ValueError("Qwen4-Exp MTP export cannot be combined with " + ", ".join(incompatible_options) + ".")
        decoder_options["include_hidden_states"] = True
        return decoder_options

    def make_mtp_model(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.mtp_attrs["io_dtype"] = io_dtype
        self.mtp_attrs["onnx_dtype"] = onnx_dtype
        self.mtp_attrs["extra_options"] = copy.deepcopy(extra_options)
        self.resolve_mtp_model_config(extra_options)
        mtp_options = self.mtp_attrs["extra_options"]
        mtp_options["text_only"] = True
        mtp_options["filename"] = "mtp.onnx"
        mtp_options.pop("include_hidden_states", None)
        mtp_options.pop("exclude_lm_head", None)
        self.mtp = Qwen4ExpMTPTextModel(
            copy.deepcopy(config),
            self.mtp_attrs["io_dtype"],
            self.mtp_attrs["onnx_dtype"],
            ep,
            cache_dir,
            mtp_options,
        )

    def make_model(self, input_path):
        self.input_path = input_path
        self.decoder.make_model(input_path)
        if self.mtp is not None:
            print("Building Qwen4-Exp MTP (multi-token prediction) head -> mtp.onnx")
            self.mtp.make_model(input_path)

    def save_model(self, output_dir):
        self.decoder.save_model(output_dir)
        if self.mtp is not None:
            self.mtp.save_model(output_dir)
            self.mtp_attrs["shared_initializers"] = self.share_initializers(
                output_dir, self.decoder.filename, self.mtp.filename
            )
        if self.input_path is None:
            raise RuntimeError("make_model must be called before save_model.")
        weights = self.decoder.load_weights(self.input_path)
        language_model = weights.model.language_model
        embedding_model = Qwen4ExpEmbeddingModel(
            self.config, language_model.embed_tokens.weight.detach().cpu(), self.decoder.io_dtype
        )
        embedding_model.save_model(output_dir)
        vision_config = self.config.vision_config
        if vision_config.out_hidden_size != self.decoder.hidden_size:
            raise ValueError(
                "Qwen4-Exp vision output size must match the text embedding size: "
                f"{vision_config.out_hidden_size} != {self.decoder.hidden_size}."
            )
        vision_model = Qwen4ExpVisionModel(vision_config, weights.model.visual, self.decoder.io_dtype)
        vision_model.save_model(output_dir)
        del weights

    def make_genai_config(self, config, extra_kwargs, out_dir):
        self.decoder.make_genai_config(config.text_config, extra_kwargs, out_dir)
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        model_config = genai_config["model"]
        model_config["type"] = "qwen3_5"
        model_config["image_token_id"] = config.image_token_id
        model_config["vision_start_token_id"] = config.vision_start_token_id
        decoder_inputs = model_config["decoder"]["inputs"]
        decoder_inputs["inputs_embeds"] = "inputs_embeds"
        decoder_inputs["input_ids"] = "input_ids"
        model_config["embedding"] = {
            "filename": "embedding.onnx",
            "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
        }
        model_config["vision"] = {
            "filename": "vision.onnx",
            "spatial_merge_size": config.vision_config.spatial_merge_size,
            "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
            "outputs": {"image_features": "image_features"},
        }
        with open(config_path, "w") as config_file:
            json.dump(genai_config, config_file, indent=4)
        if self.mtp is not None:
            self.add_mtp_to_genai_config(out_dir)

    def add_mtp_to_genai_config(self, out_dir):
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        decoder_outputs = genai_config["model"]["decoder"].setdefault("outputs", {})
        decoder_outputs["hidden_states"] = "hidden_states"
        genai_config["model"]["mtp"] = {
            "filename": "mtp.onnx",
            "num_hidden_layers": 1,
            "num_key_value_heads": self.decoder.num_kv_heads,
            "head_size": self.decoder.head_size,
            "main_hidden_states": "hidden_states",
            "inputs": {
                "input_ids": "input_ids",
                "hidden_states": "hidden_states",
                "attention_mask": "attention_mask",
                "position_ids": "position_ids",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value",
                "past_indexer_names": "past_key_values.%d.indexer_key",
            },
            "outputs": {
                "logits": "logits",
                "hidden_states": "hidden_states_out",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
                "present_indexer_names": "present.%d.indexer_key",
            },
        }
        self.add_shared_initializers_to_genai_config(genai_config)
        with open(config_path, "w") as config_file:
            json.dump(genai_config, config_file, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.decoder.save_processing(model_name_or_path, extra_kwargs, out_dir)


class Qwen4ExpMTPTextModel(Qwen4ExpTextModel):
    """Qwen4-Exp one-layer self-speculative MTP head builder."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        config = copy.deepcopy(config)
        config.text_config.num_hidden_layers = 1
        config.text_config.layer_types = ["qwen_sparse_attention"]
        config.text_config.ple_layer_ids = []
        config.num_hidden_layers = 1
        config.layer_types = ["qwen_sparse_attention"]

        extra_options = copy.deepcopy(extra_options)
        extra_options["num_hidden_layers"] = 1
        extra_options["text_only"] = True
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.tile_first_hidden_state = False
        self.emit_pre_final_hidden_states = True
        self.include_hidden_states = True
        self.output_names["hidden_states"] = "hidden_states_out"
        self.output_shapes["hidden_states"] = self.make_hidden_state_shape(last_dim=self.hc_hidden_size)
        self.input_names["hidden_states"] = "hidden_states"
        self.input_types["hidden_states"] = self.io_dtype
        self.input_shapes["hidden_states"] = self.make_hidden_state_shape(last_dim=self.hc_hidden_size)

    def get_final_hyper_connection_mixer(self):
        return self.mtp_weights.hyper_connection_mixer

    def make_offset_rmsnorm(self, name, root_input, weight_tensor):
        weight_name = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(weight_tensor + self.layernorm_attrs["add_offset"], weight_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, weight_name],
            outputs=[output],
            name=name,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape=self.make_hidden_state_shape())
        return output

    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        self.load_mtp_weights(input_path)
        self.make_preprocessing_nodes()

        projected = self.make_mtp_input_projection()
        self.layernorm_attrs["root_input"] = projected
        self.layernorm_attrs["skip_input"] = projected
        self.layernorm_attrs["first_layernorm"] = True
        self.make_layer(0, self.mtp_weights.layers[0])
        self.make_lm_head(self.mtp_weights.lm_head)

        self.make_postprocessing_nodes()
        del self.mtp_weights

    def load_mtp_weights(self, input_path):
        model_dir = input_path if input_path and os.path.isdir(input_path) else self.model_name_or_path
        if not os.path.isdir(model_dir):
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            model_dir = snapshot_download(
                repo_id=model_dir,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                allow_patterns=["*.safetensors"],
                local_files_only=True,
            )
        try:
            from loaders.qwen import Qwen4ExpMTPModel  # noqa: PLC0415
        except ImportError:
            from onnxruntime_genai.models.loaders.qwen import Qwen4ExpMTPModel  # noqa: PLC0415

        self.mtp_weights = Qwen4ExpMTPModel.from_pretrained(
            self.quant_type,
            input_path,
            model_dir,
            self.hf_load_config.text_config,
            preserve_quantization=False,
            load_quantized_model=self.load_weights,
        )

    def make_mtp_input_projection(self):
        basename = "/model/mtp"
        embed_weight = "model.embed_tokens.weight"
        self.make_initializer(self.mtp_weights.embedding.weight, embed_weight, to=self.io_dtype)
        embed_gather = f"{basename}/embed_tokens/Gather"
        self.make_node(
            "Gather",
            inputs=[embed_weight, self.input_names["input_ids"]],
            outputs=[f"{embed_gather}/output_0"],
            name=embed_gather,
        )
        self.make_value(f"{embed_gather}/output_0", self.io_dtype, self.make_hidden_state_shape())

        embedding_norm = self.make_offset_rmsnorm(
            f"{basename}/pre_fc_norm_embedding",
            f"{embed_gather}/output_0",
            self.mtp_weights.pre_fc_norm_embedding.weight,
        )
        hidden_norm = self.make_branchwise_rms_norm(
            f"{basename}/pre_fc_norm_hidden",
            self.input_names["hidden_states"],
            self.mtp_weights.pre_fc_norm_hidden,
            self.hidden_size,
        )
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        grouped_shape = [*token_shape, self.hc_count, self.hidden_size]
        grouped_dims = [-1, self.hc_count, self.hidden_size] if self.use_paged_attention else [0, 0, self.hc_count, self.hidden_size]
        hidden_grouped = f"{basename}/hidden/Reshape"
        self.make_reshape(
            hidden_grouped,
            [hidden_norm, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        hidden_proj = self.make_matmul(
            self.mtp_weights.fc_hidden,
            f"{basename}/fc_hidden/MatMul",
            f"{hidden_grouped}/output_0",
            output_shape=grouped_shape,
        )
        embedding_proj = self.make_matmul(
            self.mtp_weights.fc_embedding,
            f"{basename}/fc_embedding/MatMul",
            embedding_norm,
        )
        embedding_grouped = f"{basename}/fc_embedding/Unsqueeze"
        self.make_unsqueeze(
            embedding_grouped,
            [f"{embedding_proj}/output_0", "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [*token_shape, 1, self.hidden_size],
        )
        fused = f"{basename}/input_fusion/Add"
        self.make_add(
            fused,
            [f"{hidden_proj}/output_0", f"{embedding_grouped}/output_0"],
            self.io_dtype,
            grouped_shape,
        )
        flatten_dims = [-1, self.hc_hidden_size] if self.use_paged_attention else [0, 0, self.hc_hidden_size]
        flattened = f"{basename}/input_fusion/Reshape"
        self.make_reshape(
            flattened,
            [f"{fused}/output_0", f"/model/constants/INT64/{flatten_dims}"],
            self.io_dtype,
            self.make_hidden_state_shape(last_dim=self.hc_hidden_size),
        )
        return f"{flattened}/output_0"

class Qwen35MoEModel(MTPModel):
    """Composite Qwen3.5 MoE builder for the decoder and optional MTP graph."""

    # Extra options naming a block drafter. The Engine drives one drafter per model, so any of
    # these supersedes the MTP head rather than shipping beside it.
    block_drafter_options = ("dflash2_path", "dspark_path")

    def requested_block_drafter(self, extra_options):
        requested = [name for name in self.block_drafter_options if extra_options.get(name)]
        if len(requested) > 1:
            raise ValueError("Block drafter options are mutually exclusive: " + ", ".join(requested) + ".")
        return requested[0] if requested else None

    def get_decoder_model_class(self):
        return Qwen35MoETextModel

    def get_mtp_model_class(self):
        return Qwen35MTPModel

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__()
        decoder_options = self.make_mtp_init(config, extra_options)
        self.decoder = self.get_decoder_model_class()(
            copy.deepcopy(config), io_dtype, onnx_dtype, ep, cache_dir, decoder_options
        )
        self.mtp = None
        if self.mtp_attrs["build"]:
            self.make_mtp_model(config, io_dtype, onnx_dtype, ep, cache_dir, decoder_options)

        self.dflash2 = None
        self.dflash2_shared_initializers = []
        self.make_dflash2_init(io_dtype, extra_options)

        self.dspark = None
        self.dspark_shared_initializers = []
        self.make_dspark_init(io_dtype, extra_options)

        self.vocab_size = self.decoder.vocab_size
        self.hf_token = self.decoder.hf_token
        self.hf_remote = self.decoder.hf_remote
        self.context_length = self.decoder.context_length
        self.exclude_embeds = self.decoder.exclude_embeds
        self.model_type = self.decoder.model_type

    def make_mtp_init(self, config, extra_options):
        decoder_options = super().make_mtp_init(config, extra_options)
        text_config = getattr(config, "text_config", config)
        num_mtp_layers = getattr(text_config, "mtp_num_hidden_layers", None)
        if num_mtp_layers is None:
            num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 0)
        self.mtp_attrs["build"] = (num_mtp_layers or 0) > 0
        self.mtp_attrs["shared_initializer_names"] = {"model.embed_tokens.weight"}
        self.mtp_attrs["shared_initializer_prefixes"] = ("lm_head.MatMul.",)

        block_drafter = self.requested_block_drafter(extra_options)
        if self.mtp_attrs["build"] and block_drafter:
            print(f"Skipping the MTP head: {block_drafter} supersedes it.")
            self.mtp_attrs["build"] = False

        if self.mtp_attrs["build"] and extra_options.get("exclude_mtp", False):
            print("Skipping the MTP head: exclude_mtp is set.")
            self.mtp_attrs["build"] = False

        if not self.mtp_attrs["build"]:
            return decoder_options

        incompatible_options = [
            option for option in ("exclude_lm_head", "prune_lm_head") if extra_options.get(option, False)
        ]
        if incompatible_options:
            raise ValueError("Qwen3.5 MTP export cannot be combined with " + ", ".join(incompatible_options) + ".")
        decoder_options["include_hidden_states"] = True
        return decoder_options

    def make_mtp_model(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.mtp_attrs["io_dtype"] = io_dtype
        self.mtp_attrs["onnx_dtype"] = onnx_dtype
        self.mtp_attrs["extra_options"] = copy.deepcopy(extra_options)
        self.resolve_mtp_model_config(extra_options)

        mtp_options = self.mtp_attrs["extra_options"]
        self.drop_unusable_mtp_kv_scales(mtp_options)
        mtp_options["exclude_embeds"] = False
        mtp_options["filename"] = "mtp.onnx"
        mtp_options.pop("include_hidden_states", None)
        mtp_options.pop("exclude_lm_head", None)
        # The head is one layer deep and drafts for itself, so it never taps the target's
        # residual streams; inheriting the target's tap set would fail its layer-range check.
        mtp_options.pop("aux_hidden_state_layers", None)
        # A block drafter reads the target's aux hidden states, so it is never nested in the head.
        mtp_options.pop("dflash2_path", None)
        mtp_options.pop("dflash2_num_draft_tokens", None)
        mtp_options.pop("dspark_path", None)
        mtp_options.pop("dspark_num_draft_tokens", None)
        mtp_options.pop("dspark_top_k", None)
        self.mtp = self.get_mtp_model_class()(
            copy.deepcopy(config),
            self.mtp_attrs["io_dtype"],
            self.mtp_attrs["onnx_dtype"],
            ep,
            cache_dir,
            mtp_options,
        )

    def drop_unusable_mtp_kv_scales(self, mtp_options):
        scale_file = mtp_options.get("kv_cache_scale_file")
        if not scale_file:
            return
        try:
            with open(scale_file, encoding="utf-8") as handle:
                has_mtp_section = "mtp" in json.load(handle)
        except (OSError, ValueError):
            return
        if not has_mtp_section:
            mtp_options.pop("kv_cache_quant_scheme", None)
            mtp_options.pop("kv_cache_scale_file", None)

    def make_model(self, input_path):
        self.decoder.make_model(input_path)
        if self.mtp is not None:
            print("Building MTP (multi-token prediction) head -> mtp.onnx")
            self.mtp.make_model(input_path)
        self.make_dflash2_model(input_path)
        self.make_dspark_model(input_path)

    def save_model(self, output_dir):
        self.decoder.save_model(output_dir)
        if self.mtp is not None:
            self.mtp.save_model(output_dir)
            self.mtp_attrs["shared_initializers"] = self.share_initializers(
                output_dir, self.decoder.filename, self.mtp.filename
            )
        self.save_dflash2_model(output_dir)
        self.save_dspark_model(output_dir)

    def make_genai_config(self, config, extra_kwargs, out_dir):
        self.decoder.model_type = self.model_type
        self.decoder.make_genai_config(config, extra_kwargs, out_dir)
        if self.mtp is not None:
            self.add_mtp_to_genai_config(out_dir)
        if self.dflash2 is not None:
            self.add_dflash2_to_genai_config(out_dir)
        if self.dspark is not None:
            self.add_dspark_to_genai_config(out_dir)
        if self.dflash2 is not None or self.dspark is not None:
            self.make_block_drafter_search_defaults(out_dir)

    def make_block_drafter_search_defaults(self, out_dir):
        """Ship greedy search defaults alongside a block drafter.

        ``Engine::PrepareDflash2Feeds`` sets ``wants_drafts = greedy && ...``, so a checkpoint
        whose ``generation_config.json`` asks for sampling would decode with zero drafts and no
        error. Only ``do_sample`` is cleared; ``top_k``/``top_p``/``temperature`` stay as the
        checkpoint declared them, so a caller who opts back into sampling per turn still gets them.
        """
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        if not genai_config["search"].get("do_sample", False):
            return
        genai_config["search"]["do_sample"] = False
        with open(config_path, "w") as config_file:
            json.dump(genai_config, config_file, indent=4)
        print("Set search.do_sample to false: a block drafter only proposes drafts for greedy turns.")

    def add_mtp_to_genai_config(self, out_dir):
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        decoder_outputs = genai_config["model"]["decoder"].setdefault("outputs", {})
        decoder_outputs.setdefault("hidden_states", "hidden_states")
        genai_config["model"]["mtp"] = {
            "filename": "mtp.onnx",
            "num_hidden_layers": 1,
            "num_key_value_heads": self.decoder.num_kv_heads,
            "head_size": self.decoder.head_size,
            "main_hidden_states": "hidden_states",
            "inputs": {
                "input_ids": "input_ids",
                "hidden_states": "hidden_states",
                "attention_mask": "attention_mask",
                "position_ids": "position_ids",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value",
            },
            "outputs": {
                "logits": "logits",
                "hidden_states": "hidden_states_out",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
            },
        }
        self.add_shared_initializers_to_genai_config(genai_config)

        with open(config_path, "w") as config_file:
            json.dump(genai_config, config_file, indent=4)
        print("Added 'mtp' section to genai_config.json")

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.decoder.save_processing(model_name_or_path, extra_kwargs, out_dir)

    def require_specforge_aux_taps(self, target_layer_ids, drafter_name):
        if not target_layer_ids:
            raise ValueError(f"The {drafter_name} checkpoint must define at least one target_layer_ids entry.")

        aux_layers = [layer_id + 1 for layer_id in target_layer_ids]
        untappable = [layer_id - 1 for layer_id in aux_layers if not 1 <= layer_id < self.decoder.num_layers]
        if untappable:
            raise ValueError(
                f"The {drafter_name} checkpoint targets decoder layers {untappable}, whose outputs the exporter "
                f"cannot expose; target_layer_ids must lie in [0, {self.decoder.num_layers - 1})."
            )

        expected = ",".join(str(layer_id) for layer_id in aux_layers)
        actual = ",".join(str(layer_id) for layer_id in self.decoder.aux_hidden_state_layers)
        if actual != expected:
            raise ValueError(
                f"The {drafter_name} drafter needs aux_hidden_state_layers={expected} on the main model, "
                f"got '{actual}'."
            )

    def block_drafter_precision(self, extra_options, option_name):
        precision = str(extra_options.get(option_name, "bf16")).lower()
        allowed = {"bf16", "int4", "int8"}
        if precision not in allowed:
            raise ValueError(f"{option_name} must be one of {sorted(allowed)}, got '{precision}'.")
        return precision

    def block_drafter_quant(self, precision):
        """Resolve weight-only quantization for a block drafter, or ``None`` to keep it dense.

        The drafter's LM head *is* the target's, so quantizing it the same way lets
        ``share_initializers`` fold the two into one copy. Only the symmetric/``default``
        naming convention is reproducible here, so any other algorithm leaves the head dense
        rather than writing a second copy under a name that could never match.
        """
        if precision == "bf16":
            return None
        bits = 4 if precision == "int4" else 8
        block_size = int(self.decoder.quant_attrs["matmul_block_size"])
        prepack = int(self.decoder.matmul_attrs["weights_prepacked"])
        quant = {"bits": bits, "block_size": block_size, "prepack": prepack, "lm_head": None}

        if self.decoder.exclude_lm_head or not self.decoder.is_lm_head_quantized():
            return quant
        head_bits, weight_name, scales_name, zero_point_name = self.decoder.make_tied_quantized_embedding_input_names()
        shareable = (
            weight_name == f"lm_head.MatMul.weight_Q{head_bits}"
            and scales_name == "lm_head.MatMul.weight_scales"
            and not zero_point_name
        )
        if not shareable:
            print(
                f"Leaving the block drafter's LM head dense: the target writes '{weight_name}', "
                "which this exporter cannot reproduce byte-for-byte to share."
            )
            return quant
        quant["lm_head"] = {"bits": head_bits, "block_size": block_size, "prepack": prepack}
        return quant

    def make_dflash2_init(self, io_dtype, extra_options):
        """DFlash 2 block drafter, exported as an auxiliary ``dflash2.onnx``.

        ``dflash2_path`` points at the draft checkpoint. The drafter has no embedding and no
        LM head of its own, so both come from the target and are shared on disk. SpecForge taps
        the output of each ``target_layer_ids`` entry, which is the residual stream entering the
        following layer.
        """
        self.dflash2_path = extra_options.get("dflash2_path")
        if not self.dflash2_path:
            return
        if not self.decoder.use_paged_attention:
            raise ValueError("dflash2_path requires use_paged_attention=true.")

        num_draft_tokens = None
        if "dflash2_num_draft_tokens" in extra_options:
            try:
                num_draft_tokens = int(extra_options["dflash2_num_draft_tokens"])
            except (TypeError, ValueError) as e:
                raise ValueError("dflash2_num_draft_tokens must be a positive integer.") from e
            if num_draft_tokens < 1:
                raise ValueError("dflash2_num_draft_tokens must be a positive integer.")

        fuse_gate_up = str(extra_options.get("dflash2_fuse_gate_up", False)).lower()
        if fuse_gate_up not in ("true", "false"):
            raise ValueError("dflash2_fuse_gate_up must be true or false.")
        self.dflash2_attrs = {
            "io_dtype": io_dtype,
            "num_draft_tokens": num_draft_tokens,
            "precision": self.block_drafter_precision(extra_options, "dflash2_precision"),
            "fuse_gate_up": fuse_gate_up == "true",
        }

        with open(os.path.join(self.dflash2_path, "config.json"), encoding="utf-8") as handle:
            draft_config = json.load(handle)
        dflash_config = draft_config["dflash_config"]
        checkpoint_draft_limit = int(dflash_config["block_size"]) - 1
        if num_draft_tokens is not None and num_draft_tokens > checkpoint_draft_limit:
            raise ValueError(
                f"dflash2_num_draft_tokens must not exceed the drafter checkpoint limit ({checkpoint_draft_limit})."
            )
        target_layer_ids = dflash_config["target_layer_ids"]
        self.require_specforge_aux_taps(target_layer_ids, "DFlash 2")

    def make_dflash2_model(self, input_path):
        if not self.dflash2_path:
            return
        from .dflash2 import DFlash2Builder  # noqa: PLC0415

        print("Building DFlash 2 draft model -> dflash2.onnx")
        target_dir = input_path if input_path and os.path.isdir(input_path) else self.decoder.model_name_or_path
        self.dflash2 = DFlash2Builder(
            self.dflash2_path,
            target_dir,
            self.dflash2_attrs["io_dtype"],
            self.decoder.attention_attrs["paged_block_size"],
            self.decoder.context_length,
            num_draft_tokens=self.dflash2_attrs["num_draft_tokens"],
            quant=self.block_drafter_quant(self.dflash2_attrs["precision"]),
            fuse_gate_up=self.dflash2_attrs["fuse_gate_up"],
        )
        self.dflash2.make_model()

    def save_dflash2_model(self, output_dir):
        if self.dflash2 is None:
            return
        self.dflash2.save_model(output_dir)
        self.dflash2_shared_initializers = self.share_initializers(
            output_dir, self.decoder.filename, self.dflash2.filename
        )
        self.warn_unshared_lm_head(self.dflash2, self.dflash2_shared_initializers, "DFlash 2")

    def warn_unshared_lm_head(self, drafter, shared, drafter_name):
        """Report a drafter head that stayed a separate copy instead of folding onto the target's.

        The drafter head is already much smaller than the dense one it replaces, so this is a
        missed saving rather than a failure. It happens when this exporter's blockwise
        quantizer and the target's MLAS pass round a block differently, which leaves the
        bytes unequal even though both encode the same tensor the same way.
        """
        head = getattr(drafter, "lm_head_quant", None)
        if head is None:
            return
        weight_name = f"lm_head.MatMul.weight_Q{head['bits']}"
        if any(entry["name"] == weight_name for entry in shared):
            return
        print(
            f"Note: the {drafter_name} LM head is quantized but did not match the target's "
            f"'{weight_name}' byte-for-byte, so it remains a separate (still quantized) copy."
        )

    def add_dflash2_to_genai_config(self, out_dir):
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        decoder = genai_config["model"]["decoder"]
        decoder.setdefault("outputs", {}).setdefault("aux_hidden_states", "aux_hidden_states")

        section = self.dflash2.genai_config_section()
        section["aux_hidden_state_layers"] = list(self.decoder.aux_hidden_state_layers)
        if self.dflash2_shared_initializers:
            existing = decoder.get("shared_initializers", [])
            known = {json.dumps(entry, sort_keys=True) for entry in existing}
            for entry in self.dflash2_shared_initializers:
                if json.dumps(entry, sort_keys=True) not in known:
                    existing.append(entry)
            decoder["shared_initializers"] = existing
            section["shared_initializers"] = self.dflash2_shared_initializers
        genai_config["model"]["dflash2"] = section

        with open(config_path, "w") as config_file:
            json.dump(genai_config, config_file, indent=4)
        print("Added 'dflash2' section to genai_config.json")

    def make_dspark_init(self, io_dtype, extra_options):
        """DSpark block drafter, exported as an auxiliary ``dspark.onnx``.

        ``dspark_path`` points at the draft checkpoint. SpecForge taps the *output* of each
        ``target_layer_ids`` entry (``hidden_states[layer_id + 1]``), which is the residual stream
        entering layer ``layer_id + 1`` -- the tensor aux_hidden_state_layers names. Getting that
        off by one leaves acceptance at exactly 1.0.
        """
        self.dspark_path = extra_options.get("dspark_path")
        if not self.dspark_path:
            return
        if self.dflash2_path:
            raise ValueError("dspark_path and dflash2_path are mutually exclusive.")
        if not self.decoder.use_paged_attention:
            raise ValueError("dspark_path requires use_paged_attention=true.")

        num_draft_tokens = None
        if "dspark_num_draft_tokens" in extra_options:
            try:
                num_draft_tokens = int(extra_options["dspark_num_draft_tokens"])
            except (TypeError, ValueError) as error:
                raise ValueError("dspark_num_draft_tokens must be between 2 and the checkpoint block size.") from error

        try:
            top_k = int(extra_options.get("dspark_top_k", 16))
        except (TypeError, ValueError) as error:
            raise ValueError("dspark_top_k must be a positive integer.") from error
        if top_k < 1:
            raise ValueError("dspark_top_k must be a positive integer.")

        with open(os.path.join(self.dspark_path, "config.json"), encoding="utf-8") as handle:
            draft_config = json.load(handle)
        checkpoint_block_size = int(draft_config["block_size"])
        if num_draft_tokens is not None and not 2 <= num_draft_tokens <= checkpoint_block_size:
            raise ValueError(
                "dspark_num_draft_tokens must be between 2 and the drafter checkpoint block size "
                f"({checkpoint_block_size})."
            )
        vocab_size = int(draft_config["vocab_size"])
        if top_k > vocab_size:
            raise ValueError(f"dspark_top_k must not exceed the drafter vocabulary size ({vocab_size}).")

        self.dspark_attrs = {
            "io_dtype": io_dtype,
            "num_draft_tokens": num_draft_tokens,
            "top_k": top_k,
        }

        target_layer_ids = draft_config["dflash_config"]["target_layer_ids"]
        self.require_specforge_aux_taps(target_layer_ids, "DSpark")

    def make_dspark_model(self, input_path):
        if not self.dspark_path:
            return
        from .dspark import DSparkBuilder  # noqa: PLC0415

        print("Building DSpark draft model -> dspark.onnx")
        target_dir = input_path if input_path and os.path.isdir(input_path) else self.decoder.model_name_or_path
        self.dspark = DSparkBuilder(
            self.dspark_path,
            target_dir,
            self.dspark_attrs["io_dtype"],
            self.decoder.attention_attrs["paged_block_size"],
            self.decoder.context_length,
            num_draft_tokens=self.dspark_attrs["num_draft_tokens"],
            top_k=self.dspark_attrs["top_k"],
        )
        self.dspark.make_model()

    def save_dspark_model(self, output_dir):
        if self.dspark is None:
            return
        self.dspark.save_model(output_dir)
        self.dspark_shared_initializers = self.share_initializers(
            output_dir, self.decoder.filename, self.dspark.filename
        )

    def add_dspark_to_genai_config(self, out_dir):
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        decoder = genai_config["model"]["decoder"]
        decoder.setdefault("outputs", {}).setdefault("aux_hidden_states", "aux_hidden_states")

        section = self.dspark.genai_config_section()
        section["aux_hidden_state_layers"] = list(self.decoder.aux_hidden_state_layers)
        if self.dspark_shared_initializers:
            existing = decoder.get("shared_initializers", [])
            known = {json.dumps(entry, sort_keys=True) for entry in existing}
            for entry in self.dspark_shared_initializers:
                if json.dumps(entry, sort_keys=True) not in known:
                    existing.append(entry)
            decoder["shared_initializers"] = existing
            section["shared_initializers"] = self.dspark_shared_initializers
        genai_config["model"]["dspark"] = section

        with open(config_path, "w") as config_file:
            json.dump(genai_config, config_file, indent=4)
        print("Added 'dspark' section to genai_config.json")


class Qwen35MTPModel(Qwen35MoETextModel):
    """Qwen3.6 multi-token-prediction self-speculative head builder."""

    is_moe_mtp = True

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.is_mtp_head = True

        config = copy.deepcopy(config)
        text_config = getattr(config, "text_config", config)
        text_config.num_hidden_layers = 1
        text_config.layer_types = ["full_attention"]
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]

        self.mtp_layer_config = copy.deepcopy(text_config)
        self.mtp_layer_config.layer_types = ["full_attention"]
        self.mtp_layer_config.num_hidden_layers = 1

        extra_options = copy.deepcopy(extra_options)
        extra_options["num_hidden_layers"] = 1
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.preserve_mtp_quantization = "_quant_config" not in extra_options
        self.input_names["hidden_states"] = "hidden_states"
        self.input_types["hidden_states"] = self.io_dtype
        self.input_shapes["hidden_states"] = self.make_hidden_state_shape()

    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        self.load_mtp_weights(input_path)
        self.make_preprocessing_nodes()

        projected = self.make_mtp_input_projection()
        self.layernorm_attrs["root_input"] = projected
        self.layernorm_attrs["skip_input"] = projected
        self.layernorm_attrs["first_layernorm"] = True

        self.make_layer(0, self.mtp_weights.layers[0])
        self.make_layernorm(1, self.mtp_weights.norm, skip=True, simple=True, location="final_norm")
        mtp_norm_output = self.layernorm_attrs["output_0"]
        self.make_lm_head(self.mtp_weights.lm_head)

        hidden_states_output = "hidden_states_out"
        self.make_node(
            "Identity",
            inputs=[mtp_norm_output],
            outputs=[hidden_states_output],
            name="/model/mtp/hidden_states_out/Identity",
        )
        hidden_states_value = self.make_value(
            hidden_states_output,
            self.io_dtype,
            shape=self.make_hidden_state_shape(),
        )
        self.model.graph.outputs.append(hidden_states_value)

        self.make_postprocessing_nodes()
        del self.mtp_weights

    def load_mtp_weights(self, input_path):
        model_dir = input_path if input_path and os.path.isdir(input_path) else self.model_name_or_path
        try:
            from loaders.qwen import QwenMTPModel  # noqa: PLC0415
        except ImportError:
            from onnxruntime_genai.models.loaders.qwen import QwenMTPModel  # noqa: PLC0415

        self.mtp_weights = QwenMTPModel.from_pretrained(
            self.quant_type,
            input_path,
            model_dir,
            self.mtp_layer_config,
            preserve_quantization=self.preserve_mtp_quantization,
            load_quantized_model=self.load_weights,
            is_moe=self.is_moe_mtp,
        )

    def make_offset_rmsnorm(self, name, root_input, weight_tensor):
        weight_name = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(weight_tensor + self.layernorm_attrs["add_offset"], weight_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, weight_name],
            outputs=[output],
            name=name,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape=self.make_hidden_state_shape())
        return output

    def make_mtp_input_projection(self):
        basename = "/model/mtp"

        embed_weight = "model.embed_tokens.weight"
        self.make_initializer(self.mtp_weights.embedding.weight, embed_weight, to=self.io_dtype)
        embed_gather = f"{basename}/embed_tokens/Gather"
        embed_output = f"{embed_gather}/output_0"
        self.make_node(
            "Gather",
            inputs=[embed_weight, self.input_names["input_ids"]],
            outputs=[embed_output],
            name=embed_gather,
        )
        self.make_value(embed_output, self.io_dtype, shape=self.make_hidden_state_shape())

        embedding_norm = self.make_offset_rmsnorm(
            f"{basename}/pre_fc_norm_embedding", embed_output, self.mtp_weights.pre_fc_norm_embedding.weight
        )
        hidden_states_norm = self.make_offset_rmsnorm(
            f"{basename}/pre_fc_norm_hidden",
            self.input_names["hidden_states"],
            self.mtp_weights.pre_fc_norm_hidden.weight,
        )

        concat_name = f"{basename}/fc/Concat"
        self.make_concat(
            concat_name,
            [embedding_norm, hidden_states_norm],
            self.io_dtype,
            self.make_hidden_state_shape(last_dim=2 * self.hidden_size),
            axis=-1,
        )

        fc_name = self.make_matmul(self.mtp_weights.fc, f"{basename}/fc/MatMul", f"{concat_name}/output_0")
        return f"{fc_name}/output_0"


class Qwen35DenseMTPModel(Qwen35MTPModel):
    """Dense Qwen3.5/Qwen3.8 MTP head with one full-attention decoder layer."""

    is_moe_mtp = False

    def make_layer(self, layer_id, layer):
        return Qwen35TextModel.make_layer(self, layer_id, layer)


class Qwen35Model(Qwen35MoEModel):
    """Composite dense Qwen3.5/Qwen3.8 builder with an optional MTP graph."""

    def get_decoder_model_class(self):
        return Qwen35TextModel

    def get_mtp_model_class(self):
        return Qwen35DenseMTPModel
