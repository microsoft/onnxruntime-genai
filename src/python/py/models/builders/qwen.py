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
from quantization import QuantConfig


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
        self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]
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
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # ModelOpt's FP8 KV-cache metadata maps to the generic fp8_per_tensor path in builder.py.
        # Without a calibration file, preserve the checkpoint's shared unit-scale convention.
        quantization_config = getattr(config, "quantization_config", {})
        self.fp8_kv_cache = extra_options.get("kv_cache_quant_type", "none") == "fp8_per_tensor"
        self._legacy_fp8_kv_cache = (
            quantization_config.get("quant_method") == "modelopt"
            and self.fp8_kv_cache
            and not extra_options.get("kv_cache_scale_file", None)
        )
        self._kv_cache_scale_created = False

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # OffsetRMSNorm: Qwen3.5 uses (1 + weight) * RMSNorm(x).
        # Pre-bake the +1 into the weight initializer so the base class's
        # SkipSimplifiedLayerNormalization can be used directly.
        self.layernorm_attrs["add_offset"] = 1

        # Qwen-3.5 uses interleaved, partial MRoPE for both text and multimodal inputs.
        self.rope_attrs["mrope_layout"] = 1
        self.rope_attrs["cast"]["use_fp32"] = True
        self.rope_attrs["cast"]["root_input"] = True
        self.rope_attrs["cast"]["output_0"] = True

        # Optionally widen the recurrent/conv state I/O into a window of the last W per-position
        # states (`state_window=W`): past/present_key_values.%d.{conv,recurrent}_state become
        # [W, B, ...] instead of [B, ...], right-aligned, with slot W-1 holding the state after the
        # final token of the forward (i.e. the unwindowed state) and being the only slot the op
        # reads back. This lets a multi-token (num_speculative_tokens>1) MTP self-speculative loop
        # CROP the recurrent state to the accepted prefix on partial accept -- copying slot `a`
        # into slot W-1 -- instead of running a full-cost main-model replay forward.
        #
        # W must be at least num_speculative_tokens+1 (the length of a verify forward).
        # 0 (the default) disables the window entirely and produces
        # the legacy unwindowed state I/O (no cropping, so MTP falls back to snapshot + replay).
        # Requires ORT kernels that understand the `state_window` attribute.
        self._state_window = int(extra_options.get("state_window", 0))
        if self._state_window < 0:
            raise ValueError("state_window must be >= 0")
        # Leading-axis window extent to splice into the state shapes, or none when unwindowed.
        self._state_window_dims = [self._state_window] if self._state_window else []

    def get_kv_cache_scale_inputs(self, **kwargs):
        # ModelOpt compatibility mode: every layer shares ONE unit PER_TENSOR scale initializer
        # named `kv_cache_scale`, created lazily at the first GroupQueryAttention node. The
        # ModelOpt checkpoint exports no calibrated k/v scale, so this is a straight E4M3
        # round-trip of the KV cache. Keeping the shared name and the lazy creation point keeps
        # the exported graph (and the external-data layout) identical to the released RC model.
        if self._legacy_fp8_kv_cache:
            if not self._kv_cache_scale_created:
                self.make_initializer(torch.tensor([1.0], dtype=torch.float32), "kv_cache_scale", to=ir.DataType.FLOAT)
                self._kv_cache_scale_created = True
            return "kv_cache_scale", "kv_cache_scale"
        return super().get_kv_cache_scale_inputs(**kwargs)

    def extend_with_optional_inputs(self, inputs, optional_inputs):
        # The ModelOpt compatibility export emits all four trailing optional GroupQueryAttention
        # inputs (k_scale, v_scale, q_norm_weight, k_norm_weight), including empty placeholders,
        # rather than trimming the unused trailing ones. Reproduce that byte-for-byte.
        if self._legacy_fp8_kv_cache and any(optional_inputs):
            inputs.extend(optional_inputs)
            return
        super().extend_with_optional_inputs(inputs, optional_inputs)

    def make_kv_cache_scale_initializers(self):
        """Emit KV cache quantization scales only for the layers that own a KV cache.

        Qwen3.5/3.6 is a hybrid stack, so only ``full_attention`` layers run
        GroupQueryAttention. The calibration file may therefore be indexed either by absolute
        layer id (``num_layers`` entries) or by full-attention order (one entry per KV layer).
        """
        if self._legacy_fp8_kv_cache:
            # The single shared scale is created on demand in `get_kv_cache_scale_inputs`.
            return

        kv_layers = [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]
        per_channel = self.kv_cache_attrs["quant_mode"] == "PER_CHANNEL"
        scale_size = self.num_kv_heads * self.head_size if per_channel else 1

        scale_file = self.kv_cache_attrs["scales_path"]
        if not scale_file:
            raise ValueError(
                "Quantized KV cache requires calibrated scales; provide them via extra_options['kv_cache_scale_file']."
            )

        with open(scale_file, encoding="utf-8") as file:
            scale_data = json.load(file)
        # The MTP head is a separate graph with its own single KV-cache layer whose activation
        # distribution differs from the main stack, so it carries its own calibrated scales in an
        # optional `mtp` section of the same file. This keeps one `kv_cache_scale_file` covering
        # both `text.onnx` and `mtp.onnx`, which is all the builder CLI accepts.
        if getattr(self, "is_mtp_head", False) and "mtp" in scale_data:
            scale_data = scale_data["mtp"]
        try:
            k_scales = scale_data["scales"]["k_scales"]
            v_scales = scale_data["scales"]["v_scales"]
        except (KeyError, TypeError) as error:
            raise ValueError("kv_cache_scale_file must contain scales.k_scales and scales.v_scales.") from error
        if len(k_scales) != len(v_scales) or len(k_scales) not in (self.num_layers, len(kv_layers)):
            raise ValueError(
                f"kv_cache_scale_file must provide {self.num_layers} (per layer) or "
                f"{len(kv_layers)} (per KV layer) scales, got k={len(k_scales)} v={len(v_scales)}"
            )
        # Absolute layer ids when the file covers every layer, else full-attention order.
        by_layer_id = len(k_scales) == self.num_layers
        scale_shape = (self.num_kv_heads, 1, self.head_size) if (per_channel and self.use_paged_attention) else (-1,)

        def make_scale(per_layer, index, layer_id):
            scale = np.asarray(per_layer[index], dtype=np.float32).reshape(-1)
            if scale.size != scale_size:
                raise ValueError(f"kv_cache scale for layer {layer_id} has size {scale.size}, expected {scale_size}")
            if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError(f"kv_cache scale for layer {layer_id} must contain finite positive values")
            return scale.reshape(scale_shape)

        for order, layer_id in enumerate(kv_layers):
            index = layer_id if by_layer_id else order
            k_scale_name, v_scale_name = self.get_kv_cache_scale_names(layer_id)
            self.make_initializer(make_scale(k_scales, index, layer_id), k_scale_name)
            self.make_initializer(make_scale(v_scales, index, layer_id), v_scale_name)

    def make_inputs_and_outputs(self):
        # Qwen-3.5 uses 3D position_ids
        self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]
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
            self.make_gated_delta_net(layer_id, attention, root_input)
        else:
            super().make_attention(layer_id, attention, root_input, **kwargs)

    def get_attn_module(self, layer_id, layer):
        return layer.linear_attn if self.layer_types[layer_id] == "linear_attention" else layer.self_attn

    def make_attention_input_proj(self, layer_id, attention, root_input, **kwargs):
        """Split Qwen3.5's doubled, per-head Q projection into Q and gate."""
        super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)

        q_size = self.num_attn_heads * self.head_size
        rs_qg_name = f"/model/layers.{layer_id}/attn/q_gate/Reshape"
        rs_qg_output = f"{rs_qg_name}/output_0"
        self.make_reshape(
            rs_qg_name,
            [self.attention_attrs["q_path"], f"/model/constants/INT64/[0, 0, {self.num_attn_heads}, {self.head_size * 2}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", self.num_attn_heads, self.head_size * 2],
        )

        split_name = f"/model/layers.{layer_id}/attn/q_gate/Split"
        q_4d_output = f"{split_name}/output_0"
        gate_4d_output = f"{split_name}/output_1"
        q_gate_shape = ["batch_size", "sequence_length", self.num_attn_heads, self.head_size]
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
            [q_4d_output, f"/model/constants/INT64/[0, 0, {q_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", q_size],
        )

        rs_g_name = f"/model/layers.{layer_id}/attn/gate/Reshape"
        self.make_reshape(
            rs_g_name,
            [gate_4d_output, f"/model/constants/INT64/[0, 0, {q_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", q_size],
        )

        self.attention_attrs["q_path"] = f"{rs_q_name}/output_0"
        self.attention_attrs["gate_path"] = f"{rs_g_name}/output_0"

    def make_attention_output_proj(self, layer_id, attention, root_input, **kwargs):
        """Apply Qwen3.5's attention output gate before the shared output projection."""
        q_size = self.num_attn_heads * self.head_size
        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        sigmoid_name = f"/model/layers.{layer_id}/attn/gate/Sigmoid"
        self.make_sigmoid(
            sigmoid_name,
            self.attention_attrs["gate_path"],
            self.io_dtype,
            ["batch_size", "sequence_length", q_size],
        )

        gated_name = f"/model/layers.{layer_id}/attn/gate/Mul"
        self.make_mul(
            gated_name,
            [f"{attn_name}/output_0", f"{sigmoid_name}/output_0"],
            self.io_dtype,
            ["batch_size", "sequence_length", q_size],
        )
        self.attention_attrs["o_path"] = f"{gated_name}/output_0"

        super().make_attention_output_proj(layer_id, attention, root_input, **kwargs)

    def make_gated_delta_net(self, layer_id, linear_attn, root_input):
        """Build GatedDeltaNet using fused CausalConvWithState + LinearAttention ops.

        Uses com.microsoft contrib ops:
        - CausalConvWithState: fused depthwise conv1d + SiLU + carry state
        - LinearAttention: fused 3D-packed linear attention with GQA
        """
        basename = f"/model/layers.{layer_id}/linear_attn"

        # Projections, conv weight init, QKV transpose
        z_name, b_name, a_name, qkv_t_output, conv_weight_name = self.make_linear_attention_input_proj(
            layer_id, linear_attn, root_input
        )

        # --- Fused conv: CausalConvWithState (com.microsoft) ---
        conv_bias_name = f"model.layers.{layer_id}.linear_attn.conv1d.bias"
        self.make_initializer(torch.zeros(self.linear_conv_dim, dtype=torch.float32), conv_bias_name, to=self.io_dtype)

        conv_op_name = f"{basename}/CausalConvWithState"
        self.make_causal_conv_with_state(
            conv_op_name,
            root_input=qkv_t_output,
            weight=conv_weight_name,
            bias=conv_bias_name,
            past_conv_state=self.input_names["past.conv"][layer_id],
            present_conv_state=self.output_names["present.conv"][layer_id],
            channels=self.linear_conv_dim,
            state_window=self._state_window,
        )
        conv_output = f"{conv_op_name}/output_0"

        conv_out_t_name = f"{basename}/conv_out/Transpose"
        conv_out_t_output = f"{conv_out_t_name}/output_0"
        self.make_transpose(
            conv_out_t_name,
            conv_output,
            self.io_dtype,
            ["batch_size", "sequence_length", self.linear_conv_dim],
            [0, 2, 1],
        )

        # Split QKV, L2 norm, gates
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

        qkv_t_name = f"{basename}/qkv_proj/Transpose"
        qkv_t_output = f"{qkv_t_name}/output_0"
        self.make_transpose(
            qkv_t_name,
            f"{qkv_name}/output_0",
            self.io_dtype,
            ["batch_size", self.linear_conv_dim, "sequence_length"],
            [0, 2, 1],
        )

        conv_weight_name = f"model.layers.{layer_id}.linear_attn.conv1d.weight"
        self.make_initializer(attention.conv1d.weight, conv_weight_name, to=self.io_dtype)

        return z_name, b_name, a_name, qkv_t_output, conv_weight_name

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
        norm_weight = f"model.layers.{layer_id}.linear_attn.norm.weight"
        self.make_initializer(attention.norm.weight, norm_weight, to=self.io_dtype)

        gated_norm_name = f"{basename}/GatedRMSNorm"
        self.make_gated_rms_norm(
            gated_norm_name,
            root_input=attn_output_3d,
            scale=norm_weight,
            gate=f"{z_name}/output_0",
            shape=["batch_size", "sequence_length", self.linear_value_dim],
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

    def make_genai_config(self, config, extra_kwargs, out_dir):
        super().make_genai_config(config, extra_kwargs, out_dir)

        # When the MTP head is exported, advertise it (and the main model's
        # hidden-states output it consumes) in genai_config.json so the runtime
        # can load mtp.onnx for self-speculative decoding.
        if getattr(self, "enable_mtp", False):
            self.add_mtp_to_genai_config(out_dir)

    def add_mtp_to_genai_config(self, out_dir):
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        # Expose the main decoder's hidden-states output (the MTP head's input).
        decoder_outputs = genai_config["model"]["decoder"].setdefault("outputs", {})
        decoder_outputs.setdefault("hidden_states", "hidden_states")

        genai_config["model"]["mtp"] = {
            "filename": "mtp.onnx",
            "num_hidden_layers": 1,
            "num_key_value_heads": self.num_kv_heads,
            "head_size": self.head_size,
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

        if self.mtp_shared_initializers:
            genai_config["model"]["decoder"]["shared_initializers"] = self.mtp_shared_initializers
            genai_config["model"]["mtp"]["shared_initializers"] = self.mtp_shared_initializers

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)
        print("Added 'mtp' section to genai_config.json")


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

        # Keep the checkpoint's original FP8 (E4M3) weights instead of dequantizing them
        # to fp16 and re-quantizing to int4/int8. Both the self-attention q/k/v/o projections
        # and the GatedDeltaNet (linear-attention) ``in_proj_qkv`` / ``in_proj_z`` / ``out_proj``
        # projections are emitted as the weight-only ``MatMulBlockQuantizedFp8Weight`` contrib op.
        # ModelOpt FP8 KV-cache metadata is mapped onto the generic
        # `kv_cache_quant_type=fp8_per_tensor` machinery before this model is initialized.

        # Keep the checkpoint's original NVFP4 (E2M1) *dense* weights instead of dequantizing
        # them to fp16 and re-quantizing to int4/int8. The shared-expert MLP and lm_head
        # projections are emitted as the weight-only ``MatMulBlockQuantizedFp4Weight`` contrib op straight from
        # the ModelOpt tensors (E2M1 codes + E4M3 block scale + fp32 global scale). NOTE: the
        # NVFP4 *routed MoE experts* are controlled separately by ``moe_quant_type=nvfp4``
        # (native NVFP4 QMoE); this flag only covers the dense NVFP4 modules.
        # MoE attributes specific to Qwen3.5-MoE

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

        # MoE layers use MoE/QMoE ops instead of individual MatMul nodes,
        # so remove any /mlp/ MatMul overrides that don't apply.
        algo_config = self.quant_attrs.get("algo_config")
        if algo_config is not None and hasattr(algo_config, "customized_weight_config"):
            keys_to_remove = [k for k in algo_config.customized_weight_config if "/mlp/" in k]
            for k in keys_to_remove:
                del algo_config.customized_weight_config[k]

        # Keep the routing-critical projections out of INT4 quantization.
        # The MoE router selects the top-k experts and the shared-expert gate
        # scales the always-on expert. Both are tiny matmuls, but 4-bit rounding
        # of their weights perturbs the routing logits enough to flip top-k
        # expert selection (measured ~1.4 of 8 experts change per token), which
        # injects a large error into every MoE layer. Excluding them costs only
        # a few MB but materially improves quantized-model accuracy.
        if self.onnx_dtype in {ir.DataType.INT4, ir.DataType.INT8}:
            nodes_to_exclude = self.quant_attrs.setdefault("nodes_to_exclude", [])
            for i in range(self.num_layers):
                router_node = f"/model/layers.{i}/moe/router/MatMul"
                shared_gate_node = f"/model/layers.{i}/shared_expert_gate/MatMul"
                if router_node not in nodes_to_exclude:
                    nodes_to_exclude.append(router_node)
                if shared_gate_node not in nodes_to_exclude:
                    nodes_to_exclude.append(shared_gate_node)
                # When keeping original FP8 weights, keep the GatedDeltaNet (linear-attention)
                # projections out of int4/int8 quantization. ``in_proj_a`` / ``in_proj_b`` are
                # BF16 in the checkpoint (and only 32 elements wide), so they stay fp16. The
                # remaining projections are replaced by ``MatMulBlockQuantizedFp8Weight`` and
                # never reach the int4/int8 quantizer.
                if self.quant_type == "modelopt":
                    for proj in ("in_proj_a", "in_proj_b"):
                        linear_node = f"/model/layers.{i}/linear_attn/{proj}/MatMul"
                        if linear_node not in nodes_to_exclude:
                            nodes_to_exclude.append(linear_node)

        # MTP (multi-token prediction) self-speculative head.
        # When ``enable_mtp`` is set, an auxiliary ``mtp.onnx`` model is exported
        # alongside the main model (see ``Qwen35MtpHead``). It is disabled for the
        # MTP head itself (``is_mtp_head``) to avoid infinite recursion.
        self.mtp_head = None
        self.mtp_shared_initializers = []
        self.enable_mtp = str(extra_options.get("enable_mtp", "false")).lower() in ("1", "true", "yes")
        self.enable_mtp = self.enable_mtp and not getattr(self, "is_mtp_head", False)
        if self.enable_mtp:
            include_hidden_states = str(extra_options.get("include_hidden_states", "false")).lower() in (
                "1",
                "true",
                "yes",
            )
            exclude_lm_head = str(extra_options.get("exclude_lm_head", "false")).lower() in ("1", "true", "yes")
            if not include_hidden_states:
                raise ValueError("enable_mtp requires include_hidden_states=true on the main model.")
            if exclude_lm_head:
                raise ValueError("enable_mtp cannot be combined with exclude_lm_head=true.")
            # Stash the constructor arguments so the MTP head can be built from a
            # pristine config after the main model has been generated.
            self._mtp_config = copy.deepcopy(config)
            self._mtp_io_dtype = io_dtype
            self._mtp_onnx_dtype = onnx_dtype
            self._mtp_ep = ep
            self._mtp_cache_dir = cache_dir
            self._mtp_extra_options = copy.deepcopy(extra_options)
            self._resolve_mtp_model_config(extra_options)

    def _resolve_mtp_model_config(self, extra_options):
        """Resolve an independent MTP model configuration.

        Without an MTP-specific option, the head retains the main model's settings and native
        ModelOpt tensor formats. ``mtp_quant_config`` accepts the structured ``QuantConfig``
        JSON schema.
        """
        mtp_quant_config_value = extra_options.get("mtp_quant_config")
        if mtp_quant_config_value is None:
            return

        inherited_options = {
            key: copy.deepcopy(extra_options[key]) for key in ("hf_token", "hf_remote") if key in extra_options
        }
        quant_config = (
            copy.deepcopy(mtp_quant_config_value)
            if isinstance(mtp_quant_config_value, QuantConfig)
            else QuantConfig.from_json(mtp_quant_config_value)
        )

        from .qwen_mtp import mtp_dtypes_from_quant_config  # noqa: PLC0415

        self._mtp_io_dtype, self._mtp_onnx_dtype = mtp_dtypes_from_quant_config(quant_config)
        inherited_options["_quant_config"] = quant_config
        self._mtp_extra_options = inherited_options

    def make_model(self, input_path):
        super().make_model(input_path)

        # Then build the auxiliary MTP head (separate ONNX graph + file).
        if self.enable_mtp:
            from .qwen_mtp import Qwen35MtpHead  # noqa: PLC0415

            print("Building MTP (multi-token prediction) head -> mtp.onnx")
            mtp_extra_options = self._mtp_extra_options
            mtp_extra_options.pop("enable_mtp", None)  # prevent recursion
            mtp_extra_options["exclude_embeds"] = False  # MTP head embeds input_ids
            mtp_extra_options["filename"] = "mtp.onnx"
            # The MTP head is a leaf model whose decoder outputs are logits and its
            # recurrent hidden state. It must not
            # inherit the main model's hidden-states/lm-head output options, which
            # would make the final-norm output double as a graph output and feed the
            # lm_head, creating a graph cycle.
            mtp_extra_options.pop("include_hidden_states", None)
            mtp_extra_options.pop("exclude_lm_head", None)
            self.mtp_head = Qwen35MtpHead(
                self._mtp_config,
                self._mtp_io_dtype,
                self._mtp_onnx_dtype,
                self._mtp_ep,
                self._mtp_cache_dir,
                mtp_extra_options,
            )
            self.mtp_head.make_model(input_path)

    def save_model(self, out_dir):
        super().save_model(out_dir)
        if self.mtp_head is not None:
            self.mtp_head.save_model(out_dir)
            # Deduplicate the embedding + lm_head weights, which the MTP head shares
            # bit-identically with the main model: redirect mtp.onnx's copies to the
            # main model's external data file and pack them out of mtp.onnx.data
            # (~2 GB on disk; the two sessions then mmap the same bytes on the host).
            self.mtp_shared_initializers = self._share_mtp_embedding_lm_head(
                out_dir, self.filename, self.mtp_head.filename
            )

    @staticmethod
    def _share_mtp_embedding_lm_head(out_dir, main_file, mtp_file="mtp.onnx"):
        """Redirect mtp.onnx's embed_tokens/lm_head external data to model.onnx.data
        and remove the duplicated bytes from mtp.onnx.data.

        Only tensors that are byte-identical (same name/dtype/shape and matching
        bytes) are shared; anything that differs (e.g. a quantized main
        lm_head vs an fp16 MTP lm_head) is left untouched. Failures are non-fatal —
        the exported models remain valid (just larger) if sharing is skipped.
        """
        import onnx  # noqa: PLC0415

        main_onnx = os.path.join(out_dir, main_file)
        mtp_onnx = os.path.join(out_dir, mtp_file)
        main_data_name = main_file + ".data"
        mtp_data_name = mtp_file + ".data"
        main_data = os.path.join(out_dir, main_data_name)
        mtp_data = os.path.join(out_dir, mtp_data_name)
        if not (
            os.path.exists(main_onnx)
            and os.path.exists(mtp_onnx)
            and os.path.exists(main_data)
            and os.path.exists(mtp_data)
        ):
            return []

        def ext_info(tensor):
            d = {e.key: e.value for e in tensor.external_data}
            return d.get("location"), int(d.get("offset", 0)), int(d.get("length", 0))

        def set_ext(tensor, location, offset, length):
            del tensor.external_data[:]
            tensor.data_location = onnx.TensorProto.EXTERNAL
            for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
                entry = tensor.external_data.add()
                entry.key, entry.value = k, str(v)

        def external_data_equal(path_a, off_a, path_b, off_b, length, chunk_size=1 << 22):
            with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
                fa.seek(off_a)
                fb.seek(off_b)
                remaining = length
                while remaining:
                    read_size = min(chunk_size, remaining)
                    data_a = fa.read(read_size)
                    data_b = fb.read(read_size)
                    if len(data_a) != read_size or data_a != data_b:
                        return False
                    remaining -= read_size
            return True

        tmp_data = mtp_data + ".tmp"
        tmp_onnx = mtp_onnx + ".tmp"
        try:
            main = onnx.load(main_onnx, load_external_data=False)
            main_info = {}
            for t in main.graph.initializer:
                loc, off, ln = ext_info(t)
                if loc == main_data_name and (
                    t.name == "model.embed_tokens.weight" or t.name.startswith("lm_head.MatMul.")
                ):
                    main_info[t.name] = (t.data_type, tuple(t.dims), off, ln)

            mtp = onnx.load(mtp_onnx, load_external_data=False)
            mtp_inits = {t.name: t for t in mtp.graph.initializer}

            redirect, remove = {}, set()
            for name, (m_dt, m_dims, m_off, m_len) in main_info.items():
                if name not in mtp_inits:
                    continue
                t = mtp_inits[name]
                loc, off, ln = ext_info(t)
                if loc != mtp_data_name or t.data_type != m_dt or tuple(t.dims) != m_dims or ln != m_len:
                    continue
                if not external_data_equal(main_data, m_off, mtp_data, off, ln):
                    continue
                redirect[name] = (m_off, m_len)
                remove.add((off, ln))

            if not redirect:
                return []

            # Rebuild mtp.onnx.data with the redirected tensors packed out, in
            # ascending-offset order, assigning tight new offsets.
            kept = []
            for t in mtp.graph.initializer:
                loc, off, ln = ext_info(t)
                if loc != mtp_data_name or (t.name in redirect and (off, ln) in remove):
                    continue
                kept.append((off, ln, t))
            kept.sort(key=lambda x: x[0])

            with open(mtp_data, "rb") as fin, open(tmp_data, "wb") as fout:
                new_off = 0
                for old_off, ln, t in kept:
                    fin.seek(old_off)
                    remaining = ln
                    while remaining:
                        read_size = min(1 << 22, remaining)
                        buf = fin.read(read_size)
                        if len(buf) != read_size:
                            raise EOFError(f"Unexpected end of {mtp_data_name} while copying initializer '{t.name}'.")
                        fout.write(buf)
                        remaining -= len(buf)
                    set_ext(t, mtp_data_name, new_off, ln)
                    new_off += ln
            for name, (m_off, m_len) in redirect.items():
                set_ext(mtp_inits[name], main_data_name, m_off, m_len)

            # Stage both outputs before replacing either original. Saving this
            # metadata-only model preserves its external-data references.
            onnx.save(mtp, tmp_onnx)
        except Exception as exc:
            for tmp_path in (tmp_data, tmp_onnx):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            print(
                f"Warning: could not share MTP embedding/lm_head weights ({exc}); "
                f"the duplicated copies remain in {mtp_data_name}."
            )
            return []

        # Keep the original pair recoverable until both staged files are installed.
        backup_data = mtp_data + ".bak"
        backup_onnx = mtp_onnx + ".bak"
        try:
            os.replace(mtp_data, backup_data)
            os.replace(mtp_onnx, backup_onnx)
            os.replace(tmp_data, mtp_data)
            os.replace(tmp_onnx, mtp_onnx)
        except Exception as exc:
            rollback_errors = []
            for backup_path, original_path in ((backup_data, mtp_data), (backup_onnx, mtp_onnx)):
                if os.path.exists(backup_path):
                    try:
                        os.replace(backup_path, original_path)
                    except Exception as rollback_exc:
                        rollback_errors.append(rollback_exc)
            for tmp_path in (tmp_data, tmp_onnx):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            if rollback_errors:
                raise RuntimeError("Failed to restore MTP files after replacement failure.") from exc
            print(
                f"Warning: could not commit shared MTP embedding/lm_head weights ({exc}); "
                f"the duplicated copies remain in {mtp_data_name}."
            )
            return []
        os.remove(backup_data)
        os.remove(backup_onnx)
        saved_mb = sum(ln for _, ln in redirect.values()) / 1e6
        print(f"Shared MTP embedding + lm_head with the main model (saved {saved_mb:.0f} MB from {mtp_data_name})")
        return [
            {
                "name": name,
                "data_file": main_data_name,
                "offset": str(offset),
                "length": str(length),
                "data_type": main_info[name][0],
                "shape": list(main_info[name][1]),
            }
            for name, (offset, length) in redirect.items()
        ]

    def get_moe_module(self, layer_id, layer):
        return layer.mlp

    def is_native_nvfp4_moe(self, moe):
        if self.moe_attrs.get("quant_type") != "nvfp4":
            return False
        first_expert = next(iter(moe.experts), None)
        return first_expert is not None and getattr(first_expert.gate_proj, "weight_scale_2", None) is not None

    def make_moe_preprocessing(self, layer_id, moe, root_input):
        op_type = self.moe_attrs["op_type"]
        moe_weight_type = f"{'q' if op_type == 'QMoE' else ''}weight"

        gate_up_proj_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.{moe_weight_type}"
        gate_up_proj_scales = f"model.layers.{layer_id}.moe.experts.gate_up_proj.scales"
        gate_up_proj_bias = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.{moe_weight_type}"
        down_proj_scales = f"model.layers.{layer_id}.moe.experts.down_proj.scales"
        down_proj_bias = f"model.layers.{layer_id}.moe.experts.down_proj.bias"

        # Repack HF concatenated [gate|up] to ORT interleaved [g0,u0,g1,u1,...] for swiglu_fusion=1
        # A ModelOpt checkpoint can contain native NVFP4 expert tensors. Preserve
        # those when their scale metadata is present; otherwise quantize the
        # checkpoint's dense expert tensors through the regular QMoE path.
        is_nvfp4 = self.is_native_nvfp4_moe(moe)
        gate_up_proj_global_scales = ""
        down_proj_global_scales = ""

        if is_nvfp4:
            # Consume the Model Optimizer NVFP4 experts directly (block-16 E2M1 weights,
            # FP8-E4M3 block scales, per-expert FP32 global scale). No re-quantization.
            self.moe_attrs["block_size"] = 16
            gate_up_proj_global_scales = f"model.layers.{layer_id}.moe.experts.gate_up_proj.global_scales"
            down_proj_global_scales = f"model.layers.{layer_id}.moe.experts.down_proj.global_scales"
            self.make_nvfp4_moe_initializers(
                moe.experts,
                gate_up_proj_weight,
                gate_up_proj_scales,
                gate_up_proj_global_scales,
                down_proj_weight,
                down_proj_scales,
                down_proj_global_scales,
            )
        elif op_type == "MoE":
            raw_gate_up = moe.experts.gate_up_proj
            half = raw_gate_up.shape[1] // 2
            interleaved = torch.stack([raw_gate_up[:, :half, :], raw_gate_up[:, half:, :]], dim=2).reshape_as(
                raw_gate_up
            )
            self.make_initializer(interleaved, gate_up_proj_weight, to=self.io_dtype)
            self.make_initializer(moe.experts.down_proj, down_proj_weight, to=self.io_dtype)
        else:
            raw_gate_up = moe.experts.gate_up_proj
            half = raw_gate_up.shape[1] // 2
            interleaved = torch.stack([raw_gate_up[:, :half, :], raw_gate_up[:, half:, :]], dim=2).reshape_as(
                raw_gate_up
            )
            gate_up_qw_list, gate_up_sc_list = [], []
            down_qw_list, down_sc_list = [], []
            for i in range(self.moe_attrs["num_experts"]):
                qw1, sc1 = self.make_qmoe_weights(interleaved[i])
                gate_up_qw_list.append(qw1)
                gate_up_sc_list.append(sc1)
                qw2, sc2 = self.make_qmoe_weights(moe.experts.down_proj[i])
                down_qw_list.append(qw2)
                down_sc_list.append(sc2)
            self.make_initializer(torch.stack(gate_up_qw_list, dim=0).to(torch.uint8), gate_up_proj_weight)
            self.make_initializer(torch.stack(down_qw_list, dim=0).to(torch.uint8), down_proj_weight)
            self.make_initializer(torch.stack(gate_up_sc_list, dim=0), gate_up_proj_scales, to=self.io_dtype)
            self.make_initializer(torch.stack(down_sc_list, dim=0), down_proj_scales, to=self.io_dtype)

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
        is_nvfp4 = self.is_native_nvfp4_moe(moe)
        gate_up_proj_global_scales = (
            f"model.layers.{layer_id}.moe.experts.gate_up_proj.global_scales" if is_nvfp4 else ""
        )
        down_proj_global_scales = (
            f"model.layers.{layer_id}.moe.experts.down_proj.global_scales" if is_nvfp4 else ""
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

    def make_nvfp4_moe_initializers(
        self,
        experts,
        gate_up_weight_name,
        gate_up_scales_name,
        gate_up_global_name,
        down_weight_name,
        down_scales_name,
        down_global_name,
    ):
        """Emit QMoE NVFP4 initializers for all routed experts of one layer.

        Reads the Model Optimizer per-expert tensors (``weight`` uint8 ``[N, K/2]``,
        ``weight_scale`` e4m3 ``[N, K/16]``, ``weight_scale_2`` f32 scalar), repacks the
        E2M1 codes into the CUDA QMoE ``[E, K, N/2]`` layout, and interleaves gate/up
        along N for ``swiglu_fusion=1``. gate and up share one per-expert global scale.
        """
        gate_up_qw, gate_up_sc, gate_up_g = [], [], []
        down_qw, down_sc, down_g = [], [], []
        for expert_id, expert in enumerate(experts):
            gate_prefix = f"expert.{expert_id}.gate_proj"
            up_prefix = f"expert.{expert_id}.up_proj"
            down_prefix = f"expert.{expert_id}.down_proj"
            g_codes = self.repack_modelopt_nvfp4_weight_codes(expert.gate_proj.weight)
            u_codes = self.repack_modelopt_nvfp4_weight_codes(expert.up_proj.weight)
            if g_codes.shape != u_codes.shape:
                raise ValueError(
                    f"ModelOpt expert {expert_id} gate/up weights must have matching shapes, "
                    f"got {tuple(g_codes.shape)} and {tuple(u_codes.shape)}."
                )
            inter = g_codes.shape[0]
            fused_codes = torch.stack([g_codes, u_codes], dim=1).reshape(2 * inter, -1)  # [2*inter, K]
            gate_up_qw.append(self.pack_nvfp4_codes_for_qmoe(fused_codes))  # [K, inter]

            scale_shape = (inter, g_codes.shape[1] // 16)
            g_sc = self.modelopt_e4m3_bytes(expert.gate_proj.weight_scale, f"{gate_prefix}.weight_scale", scale_shape)
            u_sc = self.modelopt_e4m3_bytes(expert.up_proj.weight_scale, f"{up_prefix}.weight_scale", scale_shape)
            gate_up_sc.append(torch.stack([g_sc, u_sc], dim=1).reshape(2 * inter, -1))  # [2*inter, K/16] e4m3 bytes
            gate_global = self.modelopt_positive_scalar(
                expert.gate_proj.weight_scale_2, f"{gate_prefix}.weight_scale_2"
            )
            up_global = self.modelopt_positive_scalar(expert.up_proj.weight_scale_2, f"{up_prefix}.weight_scale_2")
            if gate_global != up_global:
                raise ValueError(
                    f"ModelOpt expert {expert_id} gate/up global scales must match for fused QMoE, "
                    f"got {gate_global} and {up_global}."
                )
            gate_up_g.append(gate_global)

            d_codes = self.repack_modelopt_nvfp4_weight_codes(expert.down_proj.weight)
            down_qw.append(self.pack_nvfp4_codes_for_qmoe(d_codes))  # [inter, hidden/2]
            down_scale_shape = (d_codes.shape[0], d_codes.shape[1] // 16)
            down_sc.append(
                self.modelopt_e4m3_bytes(
                    expert.down_proj.weight_scale,
                    f"{down_prefix}.weight_scale",
                    down_scale_shape,
                )
            )
            down_g.append(
                self.modelopt_positive_scalar(expert.down_proj.weight_scale_2, f"{down_prefix}.weight_scale_2")
            )

        self.make_initializer(torch.stack(gate_up_qw, dim=0).to(torch.uint8), gate_up_weight_name)
        self.make_initializer(torch.stack(down_qw, dim=0).to(torch.uint8), down_weight_name)
        self.make_fp8e4m3_initializer(torch.stack(gate_up_sc, dim=0), gate_up_scales_name)
        self.make_fp8e4m3_initializer(torch.stack(down_sc, dim=0), down_scales_name)
        self.make_initializer(torch.tensor(gate_up_g, dtype=torch.float32), gate_up_global_name)
        self.make_initializer(torch.tensor(down_g, dtype=torch.float32), down_global_name)

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
