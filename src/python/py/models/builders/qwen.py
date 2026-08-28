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
    def validate_gated_delta_net_options(self, use_paged_attention, linear_attn_op, state_window, ep, io_dtype):
        uses_gated_delta_net = use_paged_attention or linear_attn_op == "gated_delta_net"
        if use_paged_attention and ep != "cuda":
            raise ValueError("GatedDeltaNet paged exports require the CUDA execution provider")
        if uses_gated_delta_net and state_window:
            raise ValueError("GatedDeltaNet exports commit an unwindowed recurrent state and require state_window=0")
        if uses_gated_delta_net and ir.DataType(io_dtype) == ir.DataType.BFLOAT16:
            raise ValueError("GatedDeltaNet does not support bfloat16 model I/O")

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
            self.io_dtype,
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
        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
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
            [f"{attn_name}/output_0", f"{sigmoid_name}/output_0"],
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

        b_name = f"{basename}/b_proj/MatMul"
        self.make_matmul(attention.in_proj_b, b_name, root_input)

        a_name = f"{basename}/a_proj/MatMul"
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
            inputs=[conv_out_3d, f"/model/constants/INT64/[{self.linear_key_dim}, {self.linear_key_dim}, {self.linear_value_dim}]"],
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
        self.make_mul(q_scaled_name, [q_norm_out, scale_name], self.io_dtype, ["batch_size", "sequence_length", self.linear_key_dim])
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


class Qwen35MoEModel(MTPModel):
    """Composite Qwen3.5 MoE builder for the decoder and optional MTP graph."""

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

        self.bos_token_id = self.decoder.bos_token_id
        self.eos_token_id = self.decoder.eos_token_id
        self.pad_token_id = self.decoder.pad_token_id
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

    def save_model(self, output_dir):
        self.decoder.save_model(output_dir)
        if self.mtp is not None:
            self.mtp.save_model(output_dir)
            self.mtp_attrs["shared_initializers"] = self.share_initializers(
                output_dir, self.decoder.filename, self.mtp.filename
            )

    def make_genai_config(self, config, extra_kwargs, out_dir):
        self.decoder.model_type = self.model_type
        self.decoder.make_genai_config(config, extra_kwargs, out_dir)
        if self.mtp is not None:
            self.add_mtp_to_genai_config(out_dir)

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
