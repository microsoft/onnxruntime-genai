# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# ------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.

import os
import numpy as np
import onnx_ir as ir
import torch
from transformers import (
    AutoConfig,
    Qwen2ForCausalLM,
)

from .base import Model


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
        # self.layernorm_attrs["cast"]["skip_input"] = False
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


# TODO: figure out why anything is needed in this class
# Nothing should be necessary technically
class VideoChatFlashQwenModel(QwenModel):
    """
    Builder for OpenGVLab/VideoChat-Flash models (VideoChatFlashQwenForCausalLM).

    The language model backbone is standard Qwen2.5-7B with flat config and
    standard weight keys (model.layers.*, lm_head.*). The model uses standard
    2D RoPE (rope_scaling=None) and GQA (28 query heads, 4 KV heads).

    This builder exports only the text decoder component. It sets exclude_embeds=True
    so the decoder receives inputs_embeds from the embedding merger model, which
    fuses the InternVideo2 visual tokens with text embeddings.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Override model_type for the C++ runtime registration in model.cpp
        # and genai_config.json. Same pattern as Qwen3VLTextModel.
        # Base class transforms this to "videochat_flash_qwen" via:
        #   model_type[:model_type.find("For")].lower()
        self.model_type = "VideoChat_Flash_QwenForCausalLM"

    def load_weights(self, input_path):
        extra_kwargs = {} if os.path.isdir(self.model_name_or_path) else {"cache_dir": self.cache_dir}
        return Qwen2ForCausalLM.from_pretrained(
            self.model_name_or_path,
            token=self.hf_token,
            **extra_kwargs,
        )


class Qwen35TextModel(Model):
    """Qwen3.5 hybrid model builder.

    Qwen3.5 uses a hybrid architecture with two layer types:
    - ``full_attention``: Attention with doubled Q projection (Q + output gate),
      per-head QK RMSNorm, partial rotary embeddings, and output gating
    - ``linear_attention``: GatedDeltaNet recurrent layer with depthwise
      causal conv1d, L2-normalised Q/K, and linear attention recurrence

    The layer type pattern is controlled by ``config.layer_types`` (or
    derived from ``config.full_attention_interval``).

    Both layer types use OffsetRMSNorm (the ``1 + weight`` variant).
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Qwen3.5 is a VL model. The decoder takes inputs_embeds.
        # When exclude_embeds is explicitly set to False, build as a standalone LLM.
        self.is_text_only = extra_options.get("exclude_embeds", None) is False
        if "exclude_embeds" not in extra_options:
            extra_options["exclude_embeds"] = True
            print("Setting exclude_embeds=True for Qwen3.5 VL decoder.")

        # Qwen3.5 is a multimodal model whose HF config nests text config
        # under text_config. Flatten text_config attributes onto config so
        # the base Model init finds them where it expects.
        if hasattr(config, "text_config"):
            text_config = config.text_config
            for key in text_config:
                if not hasattr(config, key) or getattr(config, key) is None:
                    setattr(config, key, getattr(text_config, key))

        # rope parameters contain the actual rope_theta for Qwen3.5
        rope_params = self.get_rope_parameters(config)
        if rope_params is not None:
            if "rope_theta" in rope_params:
                config.rope_theta = rope_params["rope_theta"]
            if "partial_rotary_factor" in rope_params:
                config.partial_rotary_factor = rope_params["partial_rotary_factor"]

        # Parse layer types before super().__init__() because
        # make_quant_init() (via make_matmul_mixed_precision) is called from the base class init
        # and needs self.layer_types to identify linear attention layers.
        # Mirror base class logic: prefer extra_options["num_hidden_layers"] when present.
        text_config = getattr(config, "text_config", config)
        num_layers = extra_options.get("num_hidden_layers", getattr(text_config, "num_hidden_layers", 0))
        if hasattr(config, "layer_types") and config.layer_types is not None:
            self.layer_types = list(config.layer_types)
        elif hasattr(config, "full_attention_interval") and config.full_attention_interval is not None:
            interval = config.full_attention_interval
            self.layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(num_layers)
            ]
        else:
            self.layer_types = ["full_attention"] * num_layers

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.model_type = "Qwen3_5_textForCausalLM" if self.is_text_only else "Qwen3_5ForConditionalGeneration"

        # OffsetRMSNorm: Qwen3.5 uses (1 + weight) * RMSNorm(x).
        # Pre-bake the +1 into the weight initializer so the base class's
        # SkipSimplifiedLayerNormalization can be used directly.
        self.layernorm_attrs["add_offset"] = 1

        # Position IDs input.
        # In text-only mode the runtime provides standard 2D [B, S] position_ids.
        # We expand them to 3D [3, B, S] inside the graph so mRoPE works unchanged.
        # In VL mode the pipeline provides 3D position_ids directly.
        if self.is_text_only:
            self.input_shapes["position_ids"] = ["batch_size", "sequence_length"]
        else:
            self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]
        self.input_names["position_ids"] = "position_ids"

        # mRoPE config
        self.rope_attrs["mrope_section"] = self.rope_attrs.get("mrope", {}).get("sections", [])
        if not self.rope_attrs["mrope_section"]:
            raise ValueError("MRoPE sections not found in text_config rope_parameters/rope_scaling mrope_section")
        if len(self.rope_attrs["mrope_section"]) != 3:
            raise ValueError(
                f"Expected 3 MRoPE sections [T, H, W], got {len(self.rope_attrs['mrope_section'])}: {self.rope_attrs['mrope_section']}"
            )
        self.mrope_rotary_dim = int(self.rope_attrs["partial_rotary_factor"] * self.head_size)

        # Force RoPE computation in float32 for numerical stability
        self.rope_attrs["cast_to_fp32"] = True

        # Pre-compute cos/sin cache tables and interleaving masks for mRoPE
        self._make_rotary_caches()

        # Store linear attention config
        self.linear_key_head_dim = getattr(config, "linear_key_head_dim", 128)
        self.linear_value_head_dim = getattr(config, "linear_value_head_dim", 128)
        self.linear_num_key_heads = getattr(config, "linear_num_key_heads", 16)
        self.linear_num_value_heads = getattr(config, "linear_num_value_heads", 16)
        self.linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)

        # Derived dimensions for GatedDeltaNet
        self.linear_key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        self.linear_conv_dim = self.linear_key_dim * 2 + self.linear_value_dim

        # Full attention uses QK norm and output gating
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        # Disable fused RoPE in attention op - we apply mRoPE manually
        self.attention_attrs["use_rope_in_attn"] = False

        # Replace standard KV cache I/O with hybrid cache I/O
        self._setup_hybrid_cache_io()

    def _setup_hybrid_cache_io(self):
        """Set up hybrid cache I/O: KV cache for attention layers,
        conv_state + recurrent_state for linear attention layers."""

        # The base class creates KV cache entries for all num_layers.
        # We rebuild the lists: keep KV entries only for full-attention layers,
        # and add conv/recurrent state entries for linear-attention layers.
        kv_key_inputs = self.input_names["past_key_values.key"]
        kv_value_inputs = self.input_names["past_key_values.value"]
        kv_key_outputs = self.output_names["present.key"]
        kv_value_outputs = self.output_names["present.value"]

        filtered_key_inputs = []
        filtered_value_inputs = []
        filtered_key_outputs = []
        filtered_value_outputs = []

        for i, lt in enumerate(self.layer_types):
            if lt == "full_attention":
                filtered_key_inputs.append(kv_key_inputs[i])
                filtered_value_inputs.append(kv_value_inputs[i])
                filtered_key_outputs.append(kv_key_outputs[i])
                filtered_value_outputs.append(kv_value_outputs[i])
            else:
                # Fused CausalConvWithState + LinearAttention ops use same dtype as activations.
                state_dtype = self.io_dtype

                # linear_attention: add conv_state + recurrent_state
                self.input_names[f"past_state.{i}.conv"] = f"past_key_values.{i}.conv_state"
                self.input_types[f"past_state.{i}.conv"] = state_dtype
                self.input_shapes[f"past_state.{i}.conv"] = [
                    "batch_size",
                    self.linear_conv_dim,
                    self.linear_conv_kernel_dim - 1,
                ]

                self.input_names[f"past_state.{i}.recurrent"] = f"past_key_values.{i}.recurrent_state"
                self.input_types[f"past_state.{i}.recurrent"] = state_dtype
                self.input_shapes[f"past_state.{i}.recurrent"] = [
                    "batch_size",
                    self.linear_num_value_heads,
                    self.linear_key_head_dim,
                    self.linear_value_head_dim,
                ]

                self.output_names[f"present_state.{i}.conv"] = f"present.{i}.conv_state"
                self.output_types[f"present_state.{i}.conv"] = state_dtype
                self.output_shapes[f"present_state.{i}.conv"] = [
                    "batch_size",
                    self.linear_conv_dim,
                    self.linear_conv_kernel_dim - 1,
                ]

                self.output_names[f"present_state.{i}.recurrent"] = f"present.{i}.recurrent_state"
                self.output_types[f"present_state.{i}.recurrent"] = state_dtype
                self.output_shapes[f"present_state.{i}.recurrent"] = [
                    "batch_size",
                    self.linear_num_value_heads,
                    self.linear_key_head_dim,
                    self.linear_value_head_dim,
                ]

        self.input_names["past_key_values.key"] = filtered_key_inputs
        self.input_names["past_key_values.value"] = filtered_value_inputs
        self.output_names["present.key"] = filtered_key_outputs
        self.output_names["present.value"] = filtered_value_outputs

    def make_position_ids_reformatting(self):
        if self.is_text_only:
            # The graph input is 2D position_ids [B, S].
            # Expand to 3D [3, B, S] for mRoPE by stacking 3 copies.
            pos_2d = "position_ids"
            unsq_name = "/model/position_ids_expand/Unsqueeze"
            unsq_output = f"{unsq_name}/output_0"
            self.make_unsqueeze(
                unsq_name,
                [pos_2d, "/model/constants/INT64/[0]"],
                ir.DataType.INT64,
                [1, "batch_size", "sequence_length"],
            )
            tile_name = "/model/position_ids_expand/Tile"
            tile_output = f"{tile_name}/output_0"
            self.make_tile(
                tile_name,
                [unsq_output, "/model/constants/INT64/[3, 1, 1]"],
                ir.DataType.INT64,
                [3, "batch_size", "sequence_length"],
            )
            return tile_output
        return self.input_names["position_ids"]

    def make_preprocessing_nodes(self):
        super().make_preprocessing_nodes()
        self.position_ids_reformatted = self.make_position_ids_reformatting()

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Dispatch to full attention or GatedDeltaNet based on layer type."""
        if self.layer_types[layer_id] == "linear_attention":
            self._make_linear_attention(layer_id, attention, root_input)
        else:
            self._make_full_attention(layer_id, attention, root_input)

    def make_layer(self, layer_id, layer):
        """Override to pass ``linear_attn`` instead of ``self_attn`` for
        linear-attention layers (the base class assumes ``self_attn``)."""
        attn_module = layer.linear_attn if self.layer_types[layer_id] == "linear_attention" else layer.self_attn
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, attn_module, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    def _make_full_attention(self, layer_id, attn, root_input):
        """Build full attention with output gating.

        Qwen3.5 full attention has a doubled Q projection that produces both
        Q and a gating signal. After attention, the output is multiplied by
        sigmoid(gate) before the output projection.
        """
        # 1. Q projection (doubled: outputs Q and gate)
        q_matmul_name = f"/model/layers.{layer_id}/attn/q_proj/MatMul"
        self.make_matmul(attn.q_proj, q_matmul_name, root_input)
        q_gate_path = f"{q_matmul_name}/output_0"

        # Split Q and gate PER-HEAD: reshape [B,S,N*2H] -> [B,S,N,2H] -> split -> [B,S,N,H] each -> reshape back
        q_size = self.num_attn_heads * self.head_size

        # Reshape to [B, S, N, 2*H]
        rs_qg_name = f"/model/layers.{layer_id}/attn/q_gate_reshape/Reshape"
        rs_qg_output = f"{rs_qg_name}/output_0"
        self.make_reshape(
            rs_qg_name,
            [q_gate_path, f"/model/constants/INT64/[0, 0, {self.num_attn_heads}, {self.head_size * 2}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", self.num_attn_heads, self.head_size * 2],
        )

        # Split per-head: [B, S, N, 2H] -> [B, S, N, H] + [B, S, N, H]
        split_name = f"/model/layers.{layer_id}/attn/q_gate_split/Split"
        q_4d_output = f"{split_name}/output_0"
        gate_4d_output = f"{split_name}/output_1"
        self.make_node(
            "Split",
            [rs_qg_output, f"/model/constants/INT64/[{self.head_size}, {self.head_size}]"],
            [q_4d_output, gate_4d_output],
            name=split_name,
            axis=-1,
        )
        self.make_value(
            q_4d_output, self.io_dtype, ["batch_size", "sequence_length", self.num_attn_heads, self.head_size]
        )
        self.make_value(
            gate_4d_output, self.io_dtype, ["batch_size", "sequence_length", self.num_attn_heads, self.head_size]
        )

        # Reshape Q back to [B, S, N*H]
        rs_q_name = f"/model/layers.{layer_id}/attn/q_reshape/Reshape"
        q_output = f"{rs_q_name}/output_0"
        self.make_reshape(
            rs_q_name,
            [q_4d_output, f"/model/constants/INT64/[0, 0, {q_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", q_size],
        )

        # Reshape gate back to [B, S, N*H]
        rs_g_name = f"/model/layers.{layer_id}/attn/gate_reshape/Reshape"
        gate_output = f"{rs_g_name}/output_0"
        self.make_reshape(
            rs_g_name,
            [gate_4d_output, f"/model/constants/INT64/[0, 0, {q_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", q_size],
        )

        self.attention_attrs["q_path"] = q_output

        # 3. K and V projections
        k_matmul_name = f"/model/layers.{layer_id}/attn/k_proj/MatMul"
        self.make_matmul(attn.k_proj, k_matmul_name, root_input)
        self.attention_attrs["k_path"] = f"{k_matmul_name}/output_0"

        v_matmul_name = f"/model/layers.{layer_id}/attn/v_proj/MatMul"
        self.make_matmul(attn.v_proj, v_matmul_name, root_input)
        self.attention_attrs["v_path"] = f"{v_matmul_name}/output_0"

        # 4. Per-head QK RMSNorm (on Q and K separately)
        #    The base class's make_qk_norm uses SimplifiedLayerNormalization
        #    and pre-bakes the +1 offset via layernorm_attrs["add_offset"].
        self.make_qk_norm(layer_id, attn)

        # 5. Apply interleaved mRoPE to Q and K
        if self.attention_attrs["rope"]:
            q_shape = ["batch_size", "sequence_length", self.num_attn_heads * self.head_size]
            k_shape = ["batch_size", "sequence_length", self.num_kv_heads * self.head_size]

            # Build interleaved cos/sin from pre-computed cache + 3D position_ids
            cos_dyn, sin_dyn = self._make_mrope_cos_sin("/model/rotary_emb")

            # Apply mRoPE rotation to Q
            self.attention_attrs["q_path"] = self._apply_mrope_rotation(
                layer_id,
                self.attention_attrs["q_path"],
                q_shape,
                cos_dyn,
                sin_dyn,
                self.num_attn_heads,
                f"/model/layers.{layer_id}/attn/q_mrope",
            )

            # Apply mRoPE rotation to K
            self.attention_attrs["k_path"] = self._apply_mrope_rotation(
                layer_id,
                self.attention_attrs["k_path"],
                k_shape,
                cos_dyn,
                sin_dyn,
                self.num_kv_heads,
                f"/model/layers.{layer_id}/attn/k_mrope",
            )

        # 6. GroupQueryAttention with per-layer KV cache
        past_k = f"past_key_values.{layer_id}.key"
        past_v = f"past_key_values.{layer_id}.value"
        present_k = f"present.{layer_id}.key"
        present_v = f"present.{layer_id}.value"

        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        self.make_attention_op(
            attn_name,
            layer_id=layer_id,
            q_path=self.attention_attrs["q_path"],
            k_path=self.attention_attrs["k_path"],
            v_path=self.attention_attrs["v_path"],
            past_k=past_k,
            past_v=past_v,
            present_k=present_k,
            present_v=present_v,
            cos_cache="",
            sin_cache="",
        )
        attn_output = f"{attn_name}/output_0"

        # 7. Output gating: attn_output * sigmoid(gate)
        sigmoid_name = f"/model/layers.{layer_id}/attn/gate/Sigmoid"
        self.make_sigmoid(sigmoid_name, gate_output, self.io_dtype, ["batch_size", "sequence_length", q_size])
        sigmoid_output = f"{sigmoid_name}/output_0"

        gated_name = f"/model/layers.{layer_id}/attn/gate/Mul"
        self.make_mul(
            gated_name, [attn_output, sigmoid_output], self.io_dtype, ["batch_size", "sequence_length", q_size]
        )
        gated_output = f"{gated_name}/output_0"

        # 8. Output projection
        o_matmul_name = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        self.make_matmul(attn.o_proj, o_matmul_name, gated_output)
        self.layernorm_attrs["skip_input"] = f"{o_matmul_name}/output_0"

    def _make_rotary_caches(self):
        """Pre-compute cos/sin cache table and h/w interleaving masks.

        Matches the reference model's approach:
        - cos_cache [max_len, rdim_half]: pre-computed cos(pos * inv_freq)
        - sin_cache [max_len, rdim_half]: pre-computed sin(pos * inv_freq)
        - h_mask [rdim_half]: bool mask for height positions
        - w_mask [rdim_half]: bool mask for width positions
        """
        rdim = self.mrope_rotary_dim
        rdim_half = rdim // 2
        max_len = self.context_length

        inv_freq = 1.0 / (
            self.rope_attrs["rescale_factors"]
            * (self.rope_attrs["theta"] ** (torch.arange(0, rdim, 2, dtype=torch.int64).float() / rdim))
        )

        positions = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_len, rdim_half]
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        self.make_initializer(cos_cache, "model.rotary_emb.cos_cache", to=ir.DataType.FLOAT)
        self.make_initializer(sin_cache, "model.rotary_emb.sin_cache", to=ir.DataType.FLOAT)

        # Build interleaving masks
        dim_assignments = [0] * rdim_half
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.rope_attrs["mrope"]["sections"][dim_idx] * 3
            for i in range(offset, length, 3):
                if i < rdim_half:
                    dim_assignments[i] = dim_idx

        h_mask = torch.tensor([d == 1 for d in dim_assignments], dtype=torch.bool)
        w_mask = torch.tensor([d == 2 for d in dim_assignments], dtype=torch.bool)

        self.make_initializer(h_mask, "model.rotary_emb.h_mask", to=ir.DataType.BOOL)
        self.make_initializer(w_mask, "model.rotary_emb.w_mask", to=ir.DataType.BOOL)
        print(f"Created rotary caches [{max_len}, {rdim_half}] + h/w masks [{rdim_half}].")

    def _get_shared_q_scale(self, head_dim):
        """Return the name of a shared 1/sqrt(head_dim) constant (created once)."""
        name = "model.constants.q_scale"
        scale_val = float(1.0 / np.sqrt(head_dim))
        self.make_initializer(
            torch.tensor([scale_val], dtype=torch.float32),
            name,
            to=self.io_dtype,
        )
        return name

    def _get_shared_l2_eps(self):
        """Return the name of a shared L2 epsilon constant (created once)."""
        name = "model.constants.l2_eps"
        self.make_initializer(
            torch.tensor([1e-6], dtype=torch.float32),
            name,
            to=self.io_dtype,
        )
        return name

    def _make_mrope_cos_sin(self, basename):
        """Build interleaved mRoPE cos/sin from pre-computed cache + position_ids.

        Input: position_ids [3, B, S] (from self.position_ids_reformatted)
        Output: cos [B, S, rdim_half], sin [B, S, rdim_half]
        """
        pos_ids = self.position_ids_reformatted
        cos_cache = "model.rotary_emb.cos_cache"
        sin_cache = "model.rotary_emb.sin_cache"
        h_mask = "model.rotary_emb.h_mask"
        w_mask = "model.rotary_emb.w_mask"
        rdim_half = self.mrope_rotary_dim // 2

        def gather_dim(dim_idx, cache_name, suffix):
            g_name = f"{basename}/{suffix}/dim{dim_idx}/pos/Gather"
            self.make_gather(
                g_name,
                [pos_ids, f"/model/constants/INT64/[{dim_idx}]"],
                ir.DataType.INT64,
                [1, "batch_size", "sequence_length"],
                axis=0,
            )
            sq_name = f"{basename}/{suffix}/dim{dim_idx}/Squeeze"
            self.make_squeeze(
                sq_name,
                [f"{g_name}/output_0", "/model/constants/INT64/[0]"],
                ir.DataType.INT64,
                ["batch_size", "sequence_length"],
            )
            gc_name = f"{basename}/{suffix}/dim{dim_idx}/cache/Gather"
            self.make_gather(
                gc_name,
                [cache_name, f"{sq_name}/output_0"],
                ir.DataType.FLOAT,
                ["batch_size", "sequence_length", rdim_half],
                axis=0,
            )
            return f"{gc_name}/output_0"

        def interleave(suffix, cache_name):
            t = gather_dim(0, cache_name, suffix)
            h = gather_dim(1, cache_name, suffix)
            w = gather_dim(2, cache_name, suffix)
            ww_name = f"{basename}/{suffix}/w/Where"
            self.make_where(ww_name, [w_mask, w, t], ir.DataType.FLOAT, ["batch_size", "sequence_length", rdim_half])
            ww_out = f"{ww_name}/output_0"
            hh_name = f"{basename}/{suffix}/h/Where"
            self.make_where(
                hh_name, [h_mask, h, ww_out], ir.DataType.FLOAT, ["batch_size", "sequence_length", rdim_half]
            )
            hh_out = f"{hh_name}/output_0"
            return hh_out

        return interleave("cos", cos_cache), interleave("sin", sin_cache)

    def _make_synthetic_position_ids(self):
        """Build synthetic position_ids [B, S] with values 0 .. B*S-1.

        Derives B and S from the ``position_ids`` model input ``[3, B, S]``
        instead of using Shape on intermediate Q/K tensors.  This avoids a
        data-dependency on Q/K computation.

        B*S is obtained by reshaping position_ids to ``[3, -1]`` and reading
        the inferred dimension from the shape.  This lets the runtime compute
        the product implicitly (Reshape is metadata-only) and avoids an
        explicit INT64 Mul that would fall back to CPU on WebGPU.

        Uses a fixed basename so ``make_node`` dedup ensures nodes are
        created once and reused across all layers and Q/K calls.
        """
        basename = "/model/attn/synthetic_pos_ids"
        pos_ids_input = self.position_ids_reformatted

        # Shape(position_ids) → [3, B, S]
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, root_input=pos_ids_input, shape=[3])

        # Slice shape[1:3] → [B, S] (used as reshape target at the end)
        bs_shape_name = f"{basename}/bs_shape/Slice"
        self.make_slice(
            bs_shape_name,
            inputs=[
                f"{shape_name}/output_0",
                "/model/constants/INT64/[1]",
                "/model/constants/INT64/[3]",
                "/model/constants/INT64/[0]",
            ],
            dtype=ir.DataType.INT64,
            shape=[2],
        )

        # Reshape position_ids [3, B, S] → [3, -1] to get B*S implicitly
        flat_name = f"{basename}/flat/Reshape"
        self.make_reshape(
            flat_name,
            inputs=[pos_ids_input, "/model/constants/INT64/[3, -1]"],
            dtype=ir.DataType.INT64,
            shape=[3, "batch_seq"],
        )

        # Shape([3, B*S]) → [3, B*S], Gather scalar index 1 → scalar B*S
        shape2_name = f"{basename}/Shape2"
        self.make_shape(shape2_name, root_input=f"{flat_name}/output_0", shape=[2])

        total_name = f"{basename}/total/Gather"
        self.make_gather(
            total_name,
            inputs=[f"{shape2_name}/output_0", "/model/constants/INT64/1"],
            dtype=ir.DataType.INT64,
            shape=[],
            axis=0,
        )

        # Range(0, B*S, 1)
        range_name = f"{basename}/range/Range"
        self.make_range(
            range_name,
            inputs=["/model/constants/INT64/0", f"{total_name}/output_0", "/model/constants/INT64/1"],
            dtype=ir.DataType.INT64,
            shape=["batch_seq"],
        )

        # Reshape to [B, S]
        pos_ids_name = f"{basename}/Reshape"
        self.make_reshape(
            pos_ids_name,
            inputs=[f"{range_name}/output_0", f"{bs_shape_name}/output_0"],
            dtype=ir.DataType.INT64,
            shape=["batch_size", "sequence_length"],
        )

        return f"{pos_ids_name}/output_0"

    def _apply_mrope_rotation(self, layer_id, qk_path, qk_shape, dyn_cos, dyn_sin, num_heads, basename):
        """Apply mRoPE via com.microsoft.RotaryEmbedding (4-input variant).

        cos/sin are pre-gathered [B, S, rdim_half].  We flatten them to
        [B*S, rdim_half] and create synthetic linear position_ids [B, S]
        so the kernel simply gathers row-by-row from the flat cache.

        cos/sin caches are always float32. When io_dtype differs (fp16/bf16),
        cast Q/K to float32 before rotation, then cast back — preserving
        numerical precision in the RoPE computation.
        """
        force_fp32 = self.rope_attrs.get("cast_to_fp32", False)
        compute_dtype = ir.DataType.FLOAT if force_fp32 else self.io_dtype
        rdim_half = self.mrope_rotary_dim // 2

        # --- Flatten cos/sin to [B*S, rdim_half] ---
        flat_cos_name = f"{basename}/cos_flat/Reshape"
        self.make_reshape(
            flat_cos_name,
            [dyn_cos, f"/model/constants/INT64/[-1, {rdim_half}]"],
            ir.DataType.FLOAT,
            ["batch_seq", rdim_half],
        )
        flat_cos = f"{flat_cos_name}/output_0"

        flat_sin_name = f"{basename}/sin_flat/Reshape"
        self.make_reshape(
            flat_sin_name,
            [dyn_sin, f"/model/constants/INT64/[-1, {rdim_half}]"],
            ir.DataType.FLOAT,
            ["batch_seq", rdim_half],
        )
        flat_sin = f"{flat_sin_name}/output_0"

        # Cast flat cos/sin to compute dtype if needed
        rope_cos = flat_cos
        rope_sin = flat_sin
        if compute_dtype != ir.DataType.FLOAT:
            cos_cast_name = f"{basename}/cos/Cast"
            self.make_cast(cos_cast_name, flat_cos, compute_dtype, ["batch_seq", rdim_half])
            rope_cos = f"{cos_cast_name}/output_0"

            sin_cast_name = f"{basename}/sin/Cast"
            self.make_cast(sin_cast_name, flat_sin, compute_dtype, ["batch_seq", rdim_half])
            rope_sin = f"{sin_cast_name}/output_0"

        # --- Build synthetic position_ids [B, S] = Range(0, B*S).reshape(B, S) ---
        # Derive B and S from the position_ids input [3, B, S] instead of
        # using Shape on intermediate Q/K tensors.  Shared across all layers
        # and Q/K calls via make_node dedup.
        pos_ids = self._make_synthetic_position_ids()

        # --- Reshape Q/K to [B, N, S, H] for com.microsoft.RotaryEmbedding ---
        head_size = qk_shape[-1] // num_heads if isinstance(qk_shape[-1], int) else self.head_size
        bnsh_shape = ["batch_size", num_heads, "sequence_length", head_size]

        reshape_in_name = f"{basename}/reshape_in/Reshape"
        self.make_reshape(
            reshape_in_name,
            [qk_path, f"/model/constants/INT64/[0, 0, {num_heads}, {head_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", num_heads, head_size],
        )
        transpose_in_name = f"{basename}/transpose_in/Transpose"
        self.make_transpose(
            transpose_in_name,
            f"{reshape_in_name}/output_0",
            self.io_dtype,
            bnsh_shape,
            perm=[0, 2, 1, 3],
        )

        rope_input = f"{transpose_in_name}/output_0"
        if compute_dtype != self.io_dtype:
            cast_in_name = f"{basename}/input/Cast"
            self.make_cast(cast_in_name, rope_input, compute_dtype, bnsh_shape)
            rope_input = f"{cast_in_name}/output_0"

        # --- com.microsoft.RotaryEmbedding ---
        rope_name = f"{basename}/RotaryEmbedding"
        rope_out = f"{rope_name}/output_0"
        self.make_node(
            "RotaryEmbedding",
            [rope_input, pos_ids, rope_cos, rope_sin],
            [rope_out],
            name=rope_name,
            domain="com.microsoft",
            num_heads=num_heads,
            rotary_embedding_dim=self.mrope_rotary_dim,
            interleaved=0,
        )
        self.make_value(rope_out, compute_dtype, bnsh_shape)

        # --- Reshape back to [B, S, N*H] ---
        final = rope_out
        if compute_dtype != self.io_dtype:
            cast_out_name = f"{basename}/output/Cast"
            self.make_cast(cast_out_name, rope_out, self.io_dtype, bnsh_shape)
            final = f"{cast_out_name}/output_0"

        transpose_out_name = f"{basename}/transpose_out/Transpose"
        bsnhv_shape = ["batch_size", "sequence_length", num_heads, head_size]
        self.make_transpose(
            transpose_out_name,
            final,
            self.io_dtype,
            bsnhv_shape,
            perm=[0, 2, 1, 3],
        )

        reshape_out_name = f"{basename}/reshape_out/Reshape"
        total_dim = num_heads * head_size
        self.make_reshape(
            reshape_out_name,
            [f"{transpose_out_name}/output_0", f"/model/constants/INT64/[0, 0, {total_dim}]"],
            self.io_dtype,
            qk_shape,
        )

        return f"{reshape_out_name}/output_0"

    def _make_linear_attention(self, layer_id, linear_attn, root_input):
        """Build GatedDeltaNet using fused CausalConvWithState + LinearAttention ops.

        Uses com.microsoft contrib ops:
        - CausalConvWithState: fused depthwise conv1d + SiLU + carry state
        - LinearAttention: fused 3D-packed linear attention with GQA
        """
        basename = f"/model/layers.{layer_id}/linear_attn"
        conv_dim = self.linear_conv_dim
        v_dim = self.linear_value_dim
        n_kv = self.linear_num_value_heads
        n_k = self.linear_num_key_heads
        hk = self.linear_key_head_dim
        hv = self.linear_value_head_dim
        kernel_size = self.linear_conv_kernel_dim

        # Projections, conv weight init, QKV transpose
        z_name, b_name, a_name, qkv_t_output, conv_weight_name = self._make_linear_attention_projections(
            layer_id, linear_attn, root_input
        )

        # --- Fused conv: CausalConvWithState (com.microsoft) ---
        conv_bias_name = f"model.layers.{layer_id}.linear_attn.conv1d.bias"
        self.make_initializer(torch.zeros(conv_dim, dtype=torch.float32), conv_bias_name, to=self.io_dtype)

        past_conv = f"past_key_values.{layer_id}.conv_state"
        present_conv = f"present.{layer_id}.conv_state"

        conv_op_name = f"{basename}/CausalConvWithState"
        self.make_causal_conv_with_state(
            conv_op_name,
            root_input=qkv_t_output,
            weight=conv_weight_name,
            bias=conv_bias_name,
            past_conv_state=past_conv,
            present_conv_state=present_conv,
            output_shape=["batch_size", conv_dim, "sequence_length"],
            present_conv_shape=["batch_size", conv_dim, kernel_size - 1],
        )
        silu_output = f"{conv_op_name}/output_0"

        conv_out_t_name = f"{basename}/conv_out/Transpose"
        conv_out_t_output = f"{conv_out_t_name}/output_0"
        self.make_transpose(
            conv_out_t_name,
            silu_output,
            self.io_dtype,
            ["batch_size", "sequence_length", conv_dim],
            [0, 2, 1],
        )

        # Split QKV, L2 norm, gates
        q_scaled_output, k_norm_out, v_out, g_output, beta_output = self._make_linear_attention_normalize_and_gate(
            layer_id,
            linear_attn,
            conv_out_t_output,
            b_name,
            a_name,
        )

        # --- Fused recurrence: LinearAttention (com.microsoft) ---
        past_recurrent = f"past_key_values.{layer_id}.recurrent_state"
        present_recurrent = f"present.{layer_id}.recurrent_state"

        la_op_name = f"{basename}/LinearAttention"
        self.make_linear_attention(
            la_op_name,
            q_path=q_scaled_output,
            k_path=k_norm_out,
            v_path=v_out,
            past_recurrent_state=past_recurrent,
            present_recurrent_state=present_recurrent,
            decay=g_output,
            beta=beta_output,
            q_num_heads=n_k,
            kv_num_heads=n_kv,
            update_rule="gated_delta",
            scale=1.0,  # Q is already pre-scaled by 1/sqrt(d_k)
            output_shape=["batch_size", "sequence_length", v_dim],
            present_recurrent_shape=["batch_size", n_kv, hk, hv],
        )
        la_output = f"{la_op_name}/output_0"

        # Gated RMSNorm + output projection
        self._make_linear_attention_output(
            layer_id,
            linear_attn,
            la_output,
            z_name,
        )

    def _make_linear_attention_projections(self, layer_id, linear_attn, root_input):
        """Build linear projections, conv weight initializer, and QKV transpose.

        Returns:
            (z_name, b_name, a_name, qkv_t_output, conv_weight_name)
        """
        basename = f"/model/layers.{layer_id}/linear_attn"
        conv_dim = self.linear_conv_dim

        qkv_name = f"{basename}/in_proj_qkv/MatMul"
        self.make_matmul(linear_attn.in_proj_qkv, qkv_name, root_input)

        z_name = f"{basename}/in_proj_z/MatMul"
        self.make_matmul(linear_attn.in_proj_z, z_name, root_input)

        b_name = f"{basename}/in_proj_b/MatMul"
        self.make_matmul(linear_attn.in_proj_b, b_name, root_input)

        a_name = f"{basename}/in_proj_a/MatMul"
        self.make_matmul(linear_attn.in_proj_a, a_name, root_input)

        qkv_t_name = f"{basename}/qkv_transpose/Transpose"
        qkv_t_output = f"{qkv_t_name}/output_0"
        self.make_transpose(
            qkv_t_name,
            f"{qkv_name}/output_0",
            self.io_dtype,
            ["batch_size", conv_dim, "sequence_length"],
            [0, 2, 1],
        )

        conv_weight_name = f"model.layers.{layer_id}.linear_attn.conv1d.weight"
        self.make_initializer(linear_attn.conv1d.weight, conv_weight_name, to=self.io_dtype)

        return z_name, b_name, a_name, qkv_t_output, conv_weight_name

    def _make_linear_attention_normalize_and_gate(
        self,
        layer_id,
        linear_attn,
        conv_out_3d,
        b_name,
        a_name,
    ):
        """Split QKV, per-head L2 norm, Q scale, and compute decay/beta gates.

        Args:
            conv_out_3d: Conv output transposed to [B, S, conv_dim].
            b_name: Name of the beta projection MatMul node.
            a_name: Name of the alpha projection MatMul node.

        Returns:
            (q_scaled_output, k_norm_out, v_out, g_output, beta_output)
        """
        basename = f"/model/layers.{layer_id}/linear_attn"
        k_dim = self.linear_key_dim
        v_dim = self.linear_value_dim
        n_kv = self.linear_num_value_heads
        n_k = self.linear_num_key_heads
        hk = self.linear_key_head_dim

        # Split into Q, K, V
        split_qkv_name = f"{basename}/split_qkv/Split"
        q_out = f"{split_qkv_name}/output_0"
        k_out = f"{split_qkv_name}/output_1"
        v_out = f"{split_qkv_name}/output_2"
        self.make_node(
            "Split",
            [conv_out_3d, f"/model/constants/INT64/[{k_dim}, {k_dim}, {v_dim}]"],
            [q_out, k_out, v_out],
            name=split_qkv_name,
            axis=-1,
        )
        self.make_value(q_out, self.io_dtype, ["batch_size", "sequence_length", k_dim])
        self.make_value(k_out, self.io_dtype, ["batch_size", "sequence_length", k_dim])
        self.make_value(v_out, self.io_dtype, ["batch_size", "sequence_length", v_dim])

        # Per-head L2 normalize Q and K
        q_norm_out = self._make_per_head_l2_normalize(f"{basename}/q_l2norm", q_out, n_k, hk)
        k_norm_out = self._make_per_head_l2_normalize(f"{basename}/k_l2norm", k_out, n_k, hk)

        # Scale Q by 1/sqrt(head_k_dim)
        scale_name = self._get_shared_q_scale(hk)
        q_scaled_name = f"{basename}/q_scaled/Mul"
        self.make_mul(q_scaled_name, [q_norm_out, scale_name], self.io_dtype, ["batch_size", "sequence_length", k_dim])
        q_scaled_output = f"{q_scaled_name}/output_0"

        # beta = sigmoid(b)
        beta_name = f"{basename}/beta/Sigmoid"
        self.make_sigmoid(beta_name, f"{b_name}/output_0", self.io_dtype, ["batch_size", "sequence_length", n_kv])
        beta_output = f"{beta_name}/output_0"

        # g = -exp(A_log) * softplus(a + dt_bias)
        # The reference model computes this entirely in float32 to prevent
        # precision loss that is exponentially amplified by exp(g) in the
        # recurrence.  Cast inputs to fp32, compute, then cast result back.
        dt_bias_init = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(linear_attn.dt_bias, dt_bias_init, to=ir.DataType.FLOAT)

        neg_exp_a_name = f"model.layers.{layer_id}.linear_attn.neg_exp_A"
        neg_exp_a = (-linear_attn.A_log.data.exp()).detach()
        self.make_initializer(neg_exp_a, neg_exp_a_name, to=ir.DataType.FLOAT)

        # Cast a projection output to fp32
        a_cast_name = f"{basename}/decay/a_cast/Cast"
        self.make_cast(a_cast_name, f"{a_name}/output_0", ir.DataType.FLOAT, ["batch_size", "sequence_length", n_kv])

        a_plus_dt_name = f"{basename}/decay/Add"
        self.make_add(
            a_plus_dt_name, [f"{a_cast_name}/output_0", dt_bias_init], ir.DataType.FLOAT, ["batch_size", "sequence_length", n_kv]
        )
        a_plus_dt_output = f"{a_plus_dt_name}/output_0"

        softplus_name = f"{basename}/decay/Softplus"
        self.make_softplus(softplus_name, a_plus_dt_output, ir.DataType.FLOAT, ["batch_size", "sequence_length", n_kv])
        softplus_output = f"{softplus_name}/output_0"

        g_fp32_name = f"{basename}/decay/Mul"
        self.make_mul(g_fp32_name, [neg_exp_a_name, softplus_output], ir.DataType.FLOAT, ["batch_size", "sequence_length", n_kv])
        g_fp32_output = f"{g_fp32_name}/output_0"

        # Cast decay back to io_dtype for the kernel
        g_cast_name = f"{basename}/decay/g_cast/Cast"
        self.make_cast(g_cast_name, g_fp32_output, self.io_dtype, ["batch_size", "sequence_length", n_kv])
        g_output = f"{g_cast_name}/output_0"

        return q_scaled_output, k_norm_out, v_out, g_output, beta_output

    def _make_linear_attention_output(
        self,
        layer_id,
        linear_attn,
        attn_output_3d,
        z_name,
    ):
        """Build gated RMSNorm and output projection.

        Args:
            attn_output_3d: Attention output [B, S, v_dim] (3D packed).
            z_name: Name of the z-gate projection MatMul node.
        """
        basename = f"/model/layers.{layer_id}/linear_attn"
        z_output = f"{z_name}/output_0"

        gated_norm_output = self._make_gated_rms_norm(
            f"{basename}/gated_norm",
            attn_output_3d,
            z_output,
            linear_attn.norm,
            layer_id,
        )

        o_name = f"{basename}/out_proj/MatMul"
        self.make_matmul(linear_attn.out_proj, o_name, gated_norm_output)
        self.layernorm_attrs["skip_input"] = f"{o_name}/output_0"

    def _make_per_head_l2_normalize(self, basename, input_name, n_heads, head_dim):
        """Per-head L2 normalize: reshape [B, S, N*H] -> [B, S, N, H], norm, reshape back.

        Uses [0, 0, N, H] / [0, 0, N*H] reshape targets so all dims are
        constants or copied from the 3D/4D input, avoiding Shape ops that
        would run on CPU and block CUDA graph capture.
        """
        total_dim = n_heads * head_dim

        # Reshape to [B, S, N, H] for per-head normalization
        flat_name = f"{basename}/flat/Reshape"
        flat_out = f"{flat_name}/output_0"
        self.make_reshape(
            flat_name,
            [input_name, f"/model/constants/INT64/[0, 0, {n_heads}, {head_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", n_heads, head_dim],
        )

        # L2 normalize along last dim (head_dim) — input is 4D [B, S, N, H]
        norm_out = self._make_l2_normalize(
            basename, flat_out, head_dim, leading_dims=["batch_size", "sequence_length", n_heads]
        )

        # Reshape back to [B, S, N*H]
        unflat_name = f"{basename}/unflat/Reshape"
        unflat_out = f"{unflat_name}/output_0"
        self.make_reshape(
            unflat_name,
            [norm_out, f"/model/constants/INT64/[0, 0, {total_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", total_dim],
        )
        return unflat_out

    def _make_l2_normalize(self, basename, input_name, last_dim, leading_dims=None):
        """L2-normalize along last dimension via ORT's LpNormalization(p=2).

        Replaces the 5-node Square+ReduceSum+Add(eps)+Rsqrt+Mul subgraph with
        a single LpNormalization op (natively supported by the WebGPU EP and
        others). The +eps term is dropped; q/k come from RMSNorm+Proj so
        magnitudes far exceed 1e-6, keeping any divergence within fp16 noise.
        """
        if leading_dims is None:
            leading_dims = ["batch_size", "sequence_length"]
        full_shape = [*leading_dims, last_dim]

        node_name = f"{basename}/LpNormalization"
        self.make_lp_normalization(node_name, input_name, self.io_dtype, full_shape, axis=-1, p=2)
        return f"{node_name}/output_0"

    def _make_gated_rms_norm(self, basename, input_name, gate_name, norm_module, layer_id):
        """Gated RMSNorm: RMSNorm(x) * SiLU(z).

        The norm weight is per-head (shape [head_v_dim]).
        Input and gate are [B, S, v_dim]. We reshape to per-head,
        apply per-head norm, gate, and reshape back.
        """
        v_dim = self.linear_value_dim
        hv = self.linear_value_head_dim
        nv = self.linear_num_value_heads

        # Reshape input to [B, S, N, H] for per-head norm (avoids Shape ops
        # that would run on CPU and block CUDA graph capture)
        flat_name = f"{basename}/input_flat/Reshape"
        flat_output = f"{flat_name}/output_0"
        self.make_reshape(
            flat_name,
            [input_name, f"/model/constants/INT64/[0, 0, {nv}, {hv}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", nv, hv],
        )

        # Norm weight (NO offset — Qwen3_5RMSNormGated uses raw weight, not 1+w)
        norm_weight = f"model.layers.{layer_id}.linear_attn.norm.weight"
        self.make_initializer(norm_module.weight, norm_weight, to=self.io_dtype)

        # SimplifiedLayerNormalization (com.microsoft, no offset for gated norm)
        norm_name = f"{basename}/SimplifiedLayerNormalization"
        norm_output = f"{norm_name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            [flat_output, norm_weight],
            [norm_output],
            name=norm_name,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(norm_output, self.io_dtype, ["batch_size", "sequence_length", nv, hv])

        # Reshape back to [B, S, v_dim]
        unflat_name = f"{basename}/norm_unflat/Reshape"
        unflat_output = f"{unflat_name}/output_0"
        self.make_reshape(
            unflat_name,
            [norm_output, f"/model/constants/INT64/[0, 0, {v_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", v_dim],
        )

        # SiLU(z) — computed in float32 as in the reference model to preserve
        # gate precision (F.silu(gate.to(torch.float32))).
        z_cast_name = f"{basename}/z_cast/Cast"
        self.make_cast(z_cast_name, gate_name, ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim])
        z_fp32 = f"{z_cast_name}/output_0"

        z_sigmoid_name = f"{basename}/z_sigmoid/Sigmoid"
        self.make_sigmoid(z_sigmoid_name, z_fp32, ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim])
        z_sigmoid_output = f"{z_sigmoid_name}/output_0"

        z_silu_name = f"{basename}/z_silu/Mul"
        self.make_mul(
            z_silu_name, [z_fp32, z_sigmoid_output], ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim]
        )
        z_silu_output = f"{z_silu_name}/output_0"

        # Cast norm output to fp32 for the multiplication, then cast result back
        norm_cast_name = f"{basename}/norm_cast/Cast"
        self.make_cast(norm_cast_name, unflat_output, ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim])

        # output = norm * silu(z) in fp32
        gated_fp32_name = f"{basename}/gated_fp32/Mul"
        self.make_mul(
            gated_fp32_name, [f"{norm_cast_name}/output_0", z_silu_output], ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim]
        )

        # Cast back to io_dtype
        gated_name = f"{basename}/gated/Cast"
        self.make_cast(gated_name, f"{gated_fp32_name}/output_0", self.io_dtype, ["batch_size", "sequence_length", v_dim])
        gated_output = f"{gated_name}/output_0"

        return gated_output

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """Generate genai_config.json for the decoder (text-only) model.

        Temporarily adjusts attributes so the base class produces the correct
        config for Qwen3.5's hybrid architecture (sparse KV cache, nested
        token IDs in ``text_config``).
        """
        # Flatten text_config token IDs onto the HF config so the base class
        # can access them.  Save to out_dir so AutoConfig.from_pretrained
        # picks up the patched version.
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs
        )
        text_cfg = getattr(hf_config, "text_config", hf_config)
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
            val = getattr(text_cfg, attr, None)
            if val is not None:
                setattr(hf_config, attr, val)
        hf_config.save_pretrained(out_dir)

        # Temporarily restore the KV cache template keys and adjust attributes
        # so the base class generates the right entries.
        saved = {
            "num_layers": self.num_layers,
            "model_type": self.model_type,
        }
        self.num_layers = len(self.layer_types)
        self.input_names["past_key_values.key"] = "past_key_values.%d.key"
        self.input_names["past_key_values.value"] = "past_key_values.%d.value"
        self.output_names["present.key"] = "present.%d.key"
        self.output_names["present.value"] = "present.%d.value"

        super().make_genai_config(out_dir, {}, out_dir)

        # Restore
        self.num_layers = saved["num_layers"]
        self.model_type = saved["model_type"]
        del self.input_names["past_key_values.key"]
        del self.input_names["past_key_values.value"]
        del self.output_names["present.key"]
        del self.output_names["present.value"]


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
        # Map Qwen3.5-MoE config attributes to what the base class expects.
        if hasattr(config, "text_config"):
            tc = config.text_config
            # Base class reads num_local_experts; MoE config uses num_experts
            if hasattr(tc, "num_experts") and not hasattr(tc, "num_local_experts"):
                tc.num_local_experts = tc.num_experts
            # Base class reads intermediate_size; MoE has moe_intermediate_size
            if not hasattr(tc, "intermediate_size") and hasattr(tc, "moe_intermediate_size"):
                tc.intermediate_size = tc.moe_intermediate_size

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # The base builder derives the GenAI model.type by stripping the suffix
        # after "For" and lowercasing, matching Qwen3.5 text-only export.
        self.model_type = (
            "Qwen3_5_Moe_textForCausalLM"
            if self.is_text_only
            else "Qwen3_5_MoeForConditionalGeneration"
        )

        # MoE attributes specific to Qwen3.5-MoE
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["swiglu_fusion"] = 1
        self.moe_attrs["normalize_routing_weights"] = True
        if self.moe_attrs.get("swiglu_limit") is None and self.ep == "trt-rtx":
            # TRT-RTX EP builds currently require QMoE swiglu_limit to be present;
            # use +inf to preserve the "no clamp" behavior when the model omits it.
            self.moe_attrs["swiglu_limit"] = float("inf")

        self.moe_intermediate_size = getattr(config, "moe_intermediate_size", 512)
        self.shared_expert_intermediate_size = getattr(config, "shared_expert_intermediate_size", self.moe_intermediate_size)

        # MoE layers use MoE/QMoE ops instead of individual MatMul nodes,
        # so remove any /mlp/ MatMul overrides that don't apply.
        algo_config = self.quant_attrs.get("algo_config")
        if algo_config is not None and hasattr(algo_config, "customized_weight_config"):
            keys_to_remove = [k for k in algo_config.customized_weight_config if "/mlp/" in k]
            for k in keys_to_remove:
                del algo_config.customized_weight_config[k]

    def make_layer(self, layer_id, layer):
        """Override to use MoE instead of dense MLP."""
        attn_module = layer.linear_attn if self.layer_types[layer_id] == "linear_attention" else layer.self_attn
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, attn_module, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        self.make_moe(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    def make_moe(self, layer_id, mlp, root_input):
        """Build MoE + shared expert subgraph for one decoder layer."""
        basename = f"/model/layers.{layer_id}/moe"
        op_type = self.moe_attrs["op_type"]
        moe_weight_type = f"{'q' if op_type == 'QMoE' else ''}weight"

        # --- Router (bias-free gate) ---
        router_basename = f"{basename}/router/MatMul"
        router_matmul_name = self.make_matmul(mlp.gate, router_basename, root_input)
        router_reshape_name = f"{basename}/router/Reshape"
        self.make_reshape(
            router_reshape_name,
            [f"{router_matmul_name}/output_0",
             f"/model/constants/INT64/{[-1, self.moe_attrs['num_experts']]}"],
            dtype=self.io_dtype,
            shape=["batch_size * sequence_length", self.moe_attrs["num_experts"]],
        )

        # --- Routed expert weights ---
        gate_up_proj_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.{moe_weight_type}"
        gate_up_proj_scales = f"model.layers.{layer_id}.moe.experts.gate_up_proj.scales"
        gate_up_proj_bias = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.{moe_weight_type}"
        down_proj_scales = f"model.layers.{layer_id}.moe.experts.down_proj.scales"
        down_proj_bias = f"model.layers.{layer_id}.moe.experts.down_proj.bias"

        # Repack HF concatenated [gate|up] to ORT interleaved [g0,u0,g1,u1,...] for swiglu_fusion=1
        raw_gate_up = mlp.experts.gate_up_proj
        half = raw_gate_up.shape[1] // 2
        interleaved = torch.stack([raw_gate_up[:, :half, :], raw_gate_up[:, half:, :]], dim=2).reshape_as(raw_gate_up)

        if op_type == "MoE":
            self.make_initializer(interleaved, gate_up_proj_weight, to=self.io_dtype)
            self.make_initializer(mlp.experts.down_proj, down_proj_weight, to=self.io_dtype)
        else:
            gate_up_qw_list, gate_up_sc_list = [], []
            down_qw_list, down_sc_list = [], []
            for i in range(self.moe_attrs["num_experts"]):
                qw1, sc1 = self.make_qmoe_weights(interleaved[i])
                gate_up_qw_list.append(qw1)
                gate_up_sc_list.append(sc1)
                qw2, sc2 = self.make_qmoe_weights(mlp.experts.down_proj[i])
                down_qw_list.append(qw2)
                down_sc_list.append(sc2)
            self.make_initializer(torch.stack(gate_up_qw_list, dim=0).to(torch.uint8), gate_up_proj_weight)
            self.make_initializer(torch.stack(down_qw_list, dim=0).to(torch.uint8), down_proj_weight)
            self.make_initializer(torch.stack(gate_up_sc_list, dim=0), gate_up_proj_scales, to=self.io_dtype)
            self.make_initializer(torch.stack(down_sc_list, dim=0), down_proj_scales, to=self.io_dtype)

        num_e = self.moe_attrs["num_experts"]
        self.make_initializer(torch.zeros(num_e, 2 * self.moe_intermediate_size), gate_up_proj_bias, to=self.io_dtype)
        self.make_initializer(torch.zeros(num_e, self.hidden_size), down_proj_bias, to=self.io_dtype)

        # --- MoE/QMoE op ---
        moe_name = f"{basename}/{op_type}"
        self.make_moe_op(
            moe_name,
            root_input=root_input,
            router_probs=f"{router_reshape_name}/output_0",
            weight1=gate_up_proj_weight,
            scales1=gate_up_proj_scales if op_type == "QMoE" else "",
            bias1=gate_up_proj_bias,
            weight2=down_proj_weight,
            scales2=down_proj_scales if op_type == "QMoE" else "",
            bias2=down_proj_bias,
        )

        # --- Shared expert ---
        shared_output = self.make_shared_expert(layer_id, mlp.shared_expert, mlp.shared_expert_gate, root_input)
        combine_name = f"{basename}/Add"
        self.make_add(
            combine_name,
            [f"{moe_name}/output_0", shared_output],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )
        self.layernorm_attrs["skip_input"] = f"{combine_name}/output_0"

    def make_shared_expert(self, layer_id, shared_expert, shared_expert_gate, root_input):
        """Build shared expert SiLU-MLP with sigmoid gating."""
        basename = f"/model/layers.{layer_id}/shared_expert"

        gate_matmul = self.make_matmul(shared_expert.gate_proj, f"{basename}/gate_proj/MatMul", root_input)
        up_matmul = self.make_matmul(shared_expert.up_proj, f"{basename}/up_proj/MatMul", root_input)

        silu_sigmoid_name = f"{basename}/gate_proj/Sigmoid"
        self.make_sigmoid(silu_sigmoid_name, f"{gate_matmul}/output_0", self.io_dtype,
                          shape=["batch_size", "sequence_length", self.shared_expert_intermediate_size])

        silu_mul_name = f"{basename}/gate_proj/Mul"
        self.make_mul(silu_mul_name,
                      [f"{gate_matmul}/output_0", f"{silu_sigmoid_name}/output_0"],
                      dtype=self.io_dtype,
                      shape=["batch_size", "sequence_length", self.shared_expert_intermediate_size])

        gate_up_mul_name = f"{basename}/Mul"
        self.make_mul(gate_up_mul_name,
                      [f"{silu_mul_name}/output_0", f"{up_matmul}/output_0"],
                      dtype=self.io_dtype,
                      shape=["batch_size", "sequence_length", self.shared_expert_intermediate_size])

        down_matmul = self.make_matmul(shared_expert.down_proj, f"{basename}/down_proj/MatMul",
                                       f"{gate_up_mul_name}/output_0")

        gate_matmul_name = self.make_matmul(shared_expert_gate, f"{basename}_gate/MatMul", root_input)
        gate_sigmoid_name = f"{basename}_gate/Sigmoid"
        self.make_sigmoid(gate_sigmoid_name, f"{gate_matmul_name}/output_0", self.io_dtype,
                          shape=["batch_size", "sequence_length", 1])

        gated_mul_name = f"{basename}/Mul"
        self.make_mul(gated_mul_name,
                      [f"{down_matmul}/output_0", f"{gate_sigmoid_name}/output_0"],
                      dtype=self.io_dtype,
                      shape=["batch_size", "sequence_length", self.hidden_size])
        return f"{gated_mul_name}/output_0"
