# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import math

import onnx_ir as ir
import torch

from ..base import Model
from ..expansions.standard import Standard


class Qwen35TextModel(Model):
    make_l2_normalize = Standard.make_l2_normalize

    def validate_gated_delta_net_options(self, state_window, ep):
        if self.use_paged_attention and ep not in {"cuda", "webgpu"}:
            raise ValueError("Paged GatedDeltaNet exports require the CUDA or WebGPU execution provider")
        if ep == "webgpu":
            supported_dtypes = (
                (ir.DataType.FLOAT16,) if self.use_paged_attention else (ir.DataType.FLOAT16, ir.DataType.FLOAT)
            )
            if self.io_dtype not in supported_dtypes:
                raise ValueError(
                    "GatedDeltaNet exports on WebGPU require FP16 for paged attention "
                    "or FP16/FP32 for non-paged attention"
                )
        if state_window == 1:
            raise ValueError("GatedDeltaNet state_window must be 0 or at least 2")
        if state_window > 9:
            raise ValueError("GatedDeltaNet state_window must not exceed 9 (state_update_capacity <= 8)")
        if state_window and self.context_length_attrs["state_update_capacity"] not in (0, state_window - 1):
            raise ValueError("state_update_capacity must equal state_window - 1 when both are specified")
        if ep not in {"cuda", "webgpu"} and self.context_length_attrs["state_update_capacity"]:
            raise ValueError("Compact recurrent state capture requires the CUDA or WebGPU execution provider")

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        if "linear_attn_op" in extra_options:
            raise ValueError("linear_attn_op is selected automatically by execution provider; remove this option. Dense CPU/DML/TRT-RTX exports retain LinearAttention.")
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

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
            self.context_length_attrs["state_window"],
            self.ep,
        )
        if self.context_length_attrs["state_window"] and self.ep in {"cuda", "webgpu"}:
            self.context_length_attrs["state_update_capacity"] = self.context_length_attrs["state_window"] - 1

        if self.use_paged_attention:
            conv_shape = ["batch_size", self.linear_conv_dim, self.linear_conv_kernel_dim - 1]
            self.input_shapes["past.conv"] = conv_shape
            self.output_shapes["present.conv"] = conv_shape

        recurrent_shape = [
            "batch_size",
            self.linear_num_value_heads,
            self.linear_value_head_dim,
            self.linear_key_head_dim,
        ]
        if self.ep in {"cuda", "webgpu"}:
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

        if self.use_paged_attention:
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

    def is_packed_attn_supported(self):
        return False

    def make_attention_init(self, config):
        # Set QK norm before the base class selects packed or paged attention paths.
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init(config)
        if self.use_paged_attention:
            self.attention_attrs["use_packed_matmul"] = self.is_packed_matmul_supported()

    def is_qkv_projection_packable(self, layer_id, attention):
        v_name = f"/model/layers.{layer_id}/attn/v_proj/MatMul"
        return v_name not in self.int4_customized_weight_config and super().is_qkv_projection_packable(
            layer_id, attention
        )

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

        self.split_attention_query_gate(layer_id)

    def split_attention_query_gate(self, layer_id):
        """Split the doubled per-head query projection into query and output gate."""

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

    def make_gated_delta_net_expansion(self, layer_id, attention, conv_out_3d, b_name, a_name):
        """Expand dense GatedDeltaNet into gate preprocessing plus LinearAttention."""
        q_path, k_path, v_path, decay, beta = self.make_gated_delta_net_gates_expansion(
            layer_id, attention, conv_out_3d, b_name, a_name
        )
        name = f"/model/layers.{layer_id}/linear_attn/LinearAttention"
        self.make_linear_attention(
            name,
            q_path=q_path,
            k_path=k_path,
            v_path=v_path,
            past_recurrent_state=self.input_names["past.recurrent"][layer_id],
            present_recurrent_state=self.output_names["present.recurrent"][layer_id],
            decay=decay,
            beta=beta,
            q_num_heads=self.linear_num_key_heads,
            kv_num_heads=self.linear_num_value_heads,
            update_rule="gated_delta",
            scale=1.0,
        )
        return f"{name}/output_0"
    def make_gated_delta_net_gates_expansion(self, layer_id, attention, conv_out_3d, b_name, a_name):
        """Expand Qwen gate arithmetic and Q/K normalization around LinearAttention."""
        basename = f"/model/layers.{layer_id}/linear_attn"
        split_name = f"{basename}/split_qkv/Split"
        q_path = f"{split_name}/output_0"
        k_path = f"{split_name}/output_1"
        v_path = f"{split_name}/output_2"
        self.make_split(
            split_name,
            inputs=[
                conv_out_3d,
                f"/model/constants/INT64/[{self.linear_key_dim}, {self.linear_key_dim}, {self.linear_value_dim}]",
            ],
            outputs=[q_path, k_path, v_path],
            dtypes=[self.io_dtype] * 3,
            shapes=[
                ["batch_size", "sequence_length", self.linear_key_dim],
                ["batch_size", "sequence_length", self.linear_key_dim],
                ["batch_size", "sequence_length", self.linear_value_dim],
            ],
            axis=-1,
        )

        q_path = self.make_l2_normalize(f"{basename}/q_l2norm", q_path)
        k_path = self.make_l2_normalize(f"{basename}/k_l2norm", k_path)
        scale = 1.0 / math.sqrt(self.linear_key_head_dim)
        q_scaled = f"{basename}/q_scaled/Mul"
        self.make_mul(
            q_scaled,
            [q_path, f"/model/constants/{self.io_dtype}/{scale}"],
            self.io_dtype,
            ["batch_size", "sequence_length", self.linear_key_dim],
        )

        dt_bias = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(attention.dt_bias, dt_bias, to=ir.DataType.FLOAT)
        decay_scale = f"model.layers.{layer_id}.linear_attn.neg_exp_A"
        self.make_initializer((-attention.A_log.data.exp()).detach(), decay_scale, to=ir.DataType.FLOAT)
        gate_name = f"{basename}/LinearAttentionGate"
        self.make_linear_attention_gate(
            gate_name,
            a=a_name,
            dt_bias=dt_bias,
            decay_scale=decay_scale,
            b=b_name,
            shape=["batch_size", "sequence_length", self.linear_num_value_heads],
        )
        return (
            f"{q_scaled}/output_0",
            k_path,
            v_path,
            f"{gate_name}/output_0",
            f"{gate_name}/output_1",
        )
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

        linear_output = self.make_gated_delta_net_layer(
            layer_id,
            linear_attn,
            conv_out_t_output,
            b_name,
            a_name,
        )
        self.make_linear_attention_output_proj(layer_id, linear_attn, linear_output, z_name)
    def make_conv_state_update_kwargs(self, layer_id):
        """Compact convolution-capture bindings for this layer, or nothing when capture is disabled."""
        capacity = self.context_length_attrs["state_update_capacity"]
        if not capacity:
            return {}
        return {
            "state_update_capacity": capacity,
            "state_update_capture_count": self.input_names["state_update.capture_count"],
            "state_update_active": self.input_names["state_update.active"],
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
        """Run GatedDeltaNet directly from the packed convolution QKV output."""
        if not self.use_paged_attention and self.ep not in {"cuda", "webgpu"}:
            return self.make_gated_delta_net_expansion(layer_id, linear_attn, conv_output, b_name, a_name)
        basename = f"/model/layers.{layer_id}/linear_attn"
        packed = self.use_paged_attention
        token_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        # Reshape constants keep every token axis, so packed layouts carry one leading 0 and dense two.
        kept_axes = "0" if packed else "0, 0"
        value_heads, value_head_dim = self.linear_num_value_heads, self.linear_value_head_dim
        value_dim = self.linear_value_dim

        # The kernel applies Qwen's own gate arithmetic, so the raw checkpoint tensors are exported as-is.
        a_log_name = f"model.layers.{layer_id}.linear_attn.A_log"
        self.make_initializer(linear_attn.A_log, a_log_name, to=ir.DataType.FLOAT)
        dt_bias_name = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(linear_attn.dt_bias, dt_bias_name, to=ir.DataType.FLOAT)

        op_name = f"{basename}/GatedDeltaNet"
        recurrent_shape = self.output_shapes["present.recurrent"]
        shared_kwargs = {
            "q_path": conv_output,
            "k_path": "",
            "v_path": "",
            "decay": a_name,
            "beta": b_name,
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
                **self.make_recurrent_state_update_kwargs(layer_id),
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

        qkv_name, z_name = self.make_linear_attention_qkv_z_proj(layer_id, attention, root_input)

        b_name, a_name = self.make_linear_attention_a_b_proj(layer_id, attention, root_input)

        conv_input = qkv_name
        if not self.use_paged_attention:
            qkv_t_name = f"{basename}/qkv_proj/Transpose"
            conv_input = f"{qkv_t_name}/output_0"
            self.make_transpose(
                qkv_t_name,
                qkv_name,
                self.io_dtype,
                ["batch_size", self.linear_conv_dim, "sequence_length"],
                [0, 2, 1],
            )

        conv_weight_name = f"model.layers.{layer_id}.linear_attn.conv1d.weight"
        self.make_initializer(attention.conv1d.weight, conv_weight_name, to=self.io_dtype)

        return z_name, b_name, a_name, conv_input, conv_weight_name
    def make_linear_attention_qkv_z_proj(self, layer_id, attention, root_input):
        basename = f"/model/layers.{layer_id}/linear_attn"
        qkv_name = self.make_matmul(attention.in_proj_qkv, f"{basename}/qkv_proj/MatMul", root_input)
        z_name = self.make_matmul(attention.in_proj_z, f"{basename}/z_proj/MatMul", root_input)
        return f"{qkv_name}/output_0", f"{z_name}/output_0"
    def make_linear_attention_a_b_proj(self, layer_id, attention, root_input):
        basename = f"/model/layers.{layer_id}/linear_attn"
        # The decay and beta gates drive the GatedDeltaNet recurrence, and their weights are
        # ~0.1% of the model, so they stay dense regardless of which loader supplied them.
        b_name = f"{basename}/b_proj/MatMul"
        self.require_dense_linear_attention_gate(attention.in_proj_b, b_name)
        self.exclude_node_from_quantization(b_name)
        b_name = self.make_matmul(attention.in_proj_b, b_name, root_input)

        a_name = f"{basename}/a_proj/MatMul"
        self.require_dense_linear_attention_gate(attention.in_proj_a, a_name)
        self.exclude_node_from_quantization(a_name)
        a_name = self.make_matmul(attention.in_proj_a, a_name, root_input)
        return f"{b_name}/output_0", f"{a_name}/output_0"
    def require_dense_linear_attention_gate(self, projection, name):
        if hasattr(projection, "qweight") or getattr(projection, "quant_type", "none") != "none":
            raise ValueError(
                f"Linear-attention gate '{name}' must remain dense, but the checkpoint supplies "
                "pre-quantized weights that its loader did not dequantize."
            )
    def make_linear_attention_output_proj(self, layer_id, attention, attn_output_3d, z_name):
        """Build gated RMSNorm and output projection.

        Args:
            attn_output_3d: Attention output [B, S, linear_value_dim] (3D packed).
            z_name: Name of the z-gate projection output.
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
            gate=z_name,
            shape=output_shape,
            epsilon=self.layernorm_attrs["epsilon"],
        )

        o_name = self.make_matmul(attention.out_proj, f"{basename}/out_proj/MatMul", f"{gated_norm_name}/output_0")

        self.layernorm_attrs["skip_input"] = f"{o_name}/output_0"

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

        self.moe_intermediate_size = getattr(config, "moe_intermediate_size", 512)
        self.shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", self.moe_intermediate_size
        )

    def get_moe_module(self, layer_id, layer):
        return layer.mlp

    def make_moe_preprocessing(self, layer_id, moe, root_input):
        self.make_interleaved_swiglu_moe_preprocessing(layer_id, moe)
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
            shape=self.make_moe_router_shape(),
        )
    def make_moe_subgraph(self, layer_id, moe, root_input):
        basename = f"/model/layers.{layer_id}/moe"
        op_type = self.moe_attrs["op_type"]
        names = self.make_moe_expert_names(layer_id)
        gate_up_proj_global_scales, down_proj_global_scales = self.moe_attrs.get("global_scale_names", {}).get(
            layer_id, ("", "")
        )

        moe_name = f"{basename}/{op_type}"
        self.make_moe_op(
            moe_name,
            root_input=root_input,
            router_probs=f"{basename}/router/Reshape/output_0",
            weight1=names["gate_up_weight"],
            scales1=names["gate_up_scales"] if op_type == "QMoE" else "",
            bias1=names["gate_up_bias"],
            weight2=names["down_weight"],
            scales2=names["down_scales"] if op_type == "QMoE" else "",
            bias2=names["down_bias"],
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
            shape=self.make_hidden_state_shape(),
        )
        self.layernorm_attrs["skip_input"] = f"{combine_name}/output_0"
        return f"{combine_name}/output_0"
    def make_shared_expert(self, layer_id, shared_expert, shared_expert_gate, root_input):
        basename = f"/model/layers.{layer_id}/shared_expert"

        # Temporarily set new intermediate size from shared experts
        intermediate_size = self.intermediate_size
        try:
            self.intermediate_size = self.shared_expert_intermediate_size
            self.make_mlp_proj(layer_id, shared_expert, root_input)
        finally:
            self.intermediate_size = intermediate_size
        shared_output = self.mlp_attrs["output_0"]

        gate_matmul_name = self.make_matmul(shared_expert_gate, f"{basename}_gate/MatMul", root_input)
        gate_sigmoid_name = f"{basename}_gate/Sigmoid"
        self.make_sigmoid(
            gate_sigmoid_name,
            f"{gate_matmul_name}/output_0",
            self.io_dtype,
            shape=self.make_hidden_state_shape(last_dim=1),
        )

        return shared_output, f"{gate_sigmoid_name}/output_0"
