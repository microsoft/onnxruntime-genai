# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import math
import os

import onnx_ir as ir
from loaders.bailing import BailingModel

from .base import Model
from .expansions.standard import Standard
from .qwen3_5.text import Qwen35TextModel


class BailingMoeV3Model(Qwen35TextModel):
    """Builder for inclusionAI/Ling-3.0-tiny (BailingMoeV3ForCausalLM, model_type='bailing_hybrid')."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.config = config
        for name in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"):
            limits = getattr(config, name, None)
            if limits is not None:
                if len(limits) != config.num_hidden_layers:
                    raise ValueError(f"{name} must have one value per decoder layer")
                if any(not math.isfinite(limit) or limit < 0 for limit in limits):
                    raise ValueError(f"{name} must contain finite nonnegative limits")
        layer_group_size = getattr(config, "layer_group_size", 1)
        if isinstance(layer_group_size, bool) or not isinstance(layer_group_size, int) or layer_group_size <= 0:
            raise ValueError("BailingMoeV3 layer_group_size must be a positive integer; per-layer lists are not supported")
        full_attention_tail = config.num_hidden_layers // layer_group_size * layer_group_size
        config.layer_types = [
            "full_attention"
            if (i + 1) % layer_group_size == 0 or i >= full_attention_tail
            else "linear_attention"
            for i in range(config.num_hidden_layers)
        ]
        self.mla_attrs = {
            "qk_nope_head_dim": getattr(config, "qk_nope_head_dim", 128),  # Non-rotary query/key channels per MLA head.
            "qk_rope_head_dim": getattr(config, "qk_rope_head_dim", 64),  # Rotary query/key channels per MLA head.
            "v_head_dim": getattr(config, "v_head_dim", 128),  # Value channels retained after padded attention.
            "q_lora_rank": getattr(config, "q_lora_rank", 256),  # Compressed query projection width.
            "kv_lora_rank": getattr(config, "kv_lora_rank", 512),  # Compressed key/value projection width.
        }
        kda_head_dim = getattr(config, "head_dim", 128)
        self.kda_attrs = {
            "gate_lower_bound": None,  # Negative sigmoid-decay bound, used only for safe KDA gating.
            "bf16_precision": extra_options.get("gated_delta_net_bf16_precision", "bf16"),  # CUDA BF16 paged prefill tiles.
        }
        if not isinstance(self.kda_attrs["bf16_precision"], str) or self.kda_attrs["bf16_precision"] not in (
            "fp32", "fp16", "bf16"
        ):
            raise ValueError("gated_delta_net_bf16_precision must be one of fp32, fp16, bf16")
        if getattr(config, "kda_safe_gate", False):
            lower_bound = getattr(config, "kda_lower_bound", None)
            if (
                isinstance(lower_bound, bool)
                or not isinstance(lower_bound, (int, float))
                or not math.isfinite(lower_bound)
                or lower_bound >= 0
            ):
                raise ValueError("BailingMoeV3 kda_safe_gate=true requires a finite negative kda_lower_bound")
            self.kda_attrs["gate_lower_bound"] = float(lower_bound)

        # MLA uses qk_head_dim = 192 (128 nope + 64 rope) and pads v from 128 to 192 so PagedAttention /
        # GroupQueryAttention run with uniform head_size = 192 across all 16 heads.
        config.head_dim = self.mla_attrs["qk_nope_head_dim"] + self.mla_attrs["qk_rope_head_dim"]
        config.num_key_value_heads = config.num_attention_heads
        config.linear_num_key_heads = config.num_attention_heads
        config.linear_num_value_heads = config.num_attention_heads
        config.linear_key_head_dim = kda_head_dim
        config.linear_value_head_dim = kda_head_dim
        config.linear_conv_kernel_dim = getattr(config, "short_conv_kernel_size", 4)

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        if "gated_delta_net_bf16_precision" in extra_options and not (
            self.ep == "cuda" and self.use_paged_attention and self.io_dtype == ir.DataType.BFLOAT16
        ):
            raise ValueError("gated_delta_net_bf16_precision applies only to CUDA BF16 paged-attention exports")

        # Standard RMSNorm (not Qwen's (1 + weight) OffsetRMSNorm) and standard 2D position_ids.
        self.layernorm_attrs["add_offset"] = 0
        self.rope_attrs["mrope_layout"] = 0
        self.rope_attrs["op_type"] = "RotaryEmbedding"
        self.rope_attrs["interleaved"] = int(getattr(config, "rope_interleave", True))
        self.rope_attrs["rotary_embedding_dim"] = self.mla_attrs["qk_rope_head_dim"]
        self.rope_attrs["partial_rotary_factor"] = float(self.mla_attrs["qk_rope_head_dim"]) / float(self.head_size)
        self.rope_attrs["cast"]["use_fp32"] = False
        self.attention_attrs["use_rope_in_attn"] = 0
        self.attention_attrs["scale"] = 1.0 / math.sqrt(float(self.head_size))

        # MoE configuration
        self.moe_attrs["dense_intermediate_size"] = getattr(config, "intermediate_size", 4608)  # Early dense MLP width.
        self.moe_attrs["first_k_dense_replace"] = getattr(config, "first_k_dense_replace", 1)  # First routed-expert layer.
        self.moe_attrs["shared_expert_intermediate_size"] = getattr(
            config, "moe_shared_expert_intermediate_size", self.intermediate_size
        ) * getattr(config, "num_shared_experts", 1)  # Combined width of the shared experts.
        self.moe_attrs["num_experts"] = getattr(config, "num_experts", 128)
        self.moe_attrs["top_k"] = getattr(config, "num_experts_per_tok", 8)
        self.moe_attrs["n_group"] = getattr(config, "n_group", 8)
        self.moe_attrs["topk_group"] = getattr(config, "topk_group", 4)
        self.moe_attrs["group_topk"] = getattr(config, "group_topk", 2)
        group_topk = self.moe_attrs["group_topk"]
        if isinstance(group_topk, bool) or not isinstance(group_topk, int) or group_topk <= 0:
            raise ValueError("BailingMoeV3 group_topk must be a positive integer")
        self.moe_attrs["routed_scaling_factor"] = float(getattr(config, "routed_scaling_factor", 1.0))
        self.moe_attrs["router_sentinel"] = -10000.0
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["swiglu_fusion"] = 1
        self.moe_attrs["normalize_routing_weights"] = getattr(config, "norm_topk_prob", False)

        if self.ep == "cuda":
            # Router projection and expert selection require FP32, not TF32.
            self.ep_attrs["cuda"]["use_tf32"] = "0"

        if self.ep == "cuda" and self.use_paged_attention:
            self.ep_attrs["cuda"]["enable_cuda_graph"] = (
                "1" if str(extra_options.get("enable_cuda_graph", "1")).lower() not in ("0", "false") else "0"
            )
            self.extra_options.setdefault("max_scheduled_tokens", 8192)

        # Check if checkpoint is block-FP8 and we are not re-quantizing to INT4/INT8
        quant_cfg = getattr(config, "quantization_config", None) or {}
        target_is_int = self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4, ir.DataType.INT8, ir.DataType.UINT8}
        if isinstance(quant_cfg, dict) and quant_cfg.get("quant_method") == "fp8" and not target_is_int:
            self.moe_attrs["op_type"] = "QMoE"
            self.moe_attrs["quant_type"] = "fp8"
            self.moe_attrs["expert_weight_bits"] = 8
            self.moe_attrs["block_size"] = 128
            self.moe_attrs["swiglu_fusion"] = 2

    def validate_gated_delta_net_options(self, state_window, ep):
        if self.use_paged_attention and ep not in {"cuda", "webgpu", "cpu"}:
            raise ValueError("Paged GatedDeltaNet exports require the CUDA, WebGPU, or CPU execution provider")
        if self.context_length_attrs["state_update_capacity"] or state_window:
            raise ValueError(
                "BailingMoeV3 does not support state_update_capacity or state_window: "
                "compact state capture and replay do not support per-key decay"
            )

    def make_inputs_and_outputs(self):
        # Use 1D [num_tokens] position_ids for paged attention and 2D [batch_size, sequence_length] otherwise.
        self.input_shapes["position_ids"] = (
            ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        )
        if self.rope_attrs["op_type"] == "MRotaryEmbedding":
            self.input_shapes["position_ids"] = [3, "num_tokens"]
        if self.ep == "cpu":
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
            if not self.use_paged_attention:
                conv_shape = ["batch_size", self.linear_conv_dim, self.linear_conv_kernel_dim - 1]
                self.input_shapes["past.conv"] = conv_shape
                self.output_shapes["present.conv"] = conv_shape
        Model.make_inputs_and_outputs(self)

    def load_weights(self, input_path):
        if self.adapter_path:
            raise ValueError("BailingMoeV3 does not support adapter_path with the streaming weight loader")
        target_is_int = self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4, ir.DataType.INT8, ir.DataType.UINT8}
        weights = BailingModel.from_pretrained(
            input_path or self.model_name_or_path,
            self.config,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            revision=self.revision,
            target_is_int=target_is_int,
            num_layers=self.num_layers,
        )
        if target_is_int:
            # The loader emits dequantized weights; save_model must still run to_nbits for QDQ.
            self.quant_type = None
        return weights

    def has_final_norm(self, module, orig_model):
        return module == getattr(orig_model, "final_norm", None)

    def get_moe_module(self, layer_id, layer):
        return layer.mlp

    def make_gated_delta_net_layer(self, layer_id, linear_attn, conv_output, b_name, a_name):
        basename = f"/model/layers.{layer_id}/linear_attn"
        packed = self.use_paged_attention
        token_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        kept_axes = "0" if packed else "0, 0"
        value_heads, value_head_dim = self.linear_num_value_heads, self.linear_value_head_dim
        key_head_dim = self.linear_key_head_dim
        value_dim = self.linear_value_dim

        a_log_name = f"model.layers.{layer_id}.linear_attn.A_log"
        self.make_initializer(linear_attn.A_log, a_log_name, to=ir.DataType.FLOAT)
        dt_bias_name = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(linear_attn.dt_bias, dt_bias_name, to=ir.DataType.FLOAT)

        a_reshape_name = f"{basename}/a_proj/Reshape"
        self.make_reshape(
            a_reshape_name,
            [a_name, f"/model/constants/INT64/[{kept_axes}, {value_heads}, {key_head_dim}]"],
            self.io_dtype,
            [*token_shape, value_heads, key_head_dim],
        )

        op_name = f"{basename}/GatedDeltaNet"
        recurrent_shape = self.output_shapes["present.recurrent"]
        shared_kwargs = {
            "q_path": conv_output,
            "k_path": "",
            "v_path": "",
            "decay": f"{a_reshape_name}/output_0",
            "beta": b_name,
            "a_log": a_log_name,
            "dt_bias": dt_bias_name,
            "decay_shape": [*token_shape, value_heads, key_head_dim],
            "beta_shape": [*token_shape, value_heads],
            "gate_activation": "sigmoid" if self.kda_attrs["gate_lower_bound"] is not None else "softplus",
            "beta_activation": "sigmoid",
            "qk_l2_norm": 1,
            "update_rule": "gated_delta",
            "scale": 0.0,
            "output_shape": [*token_shape, value_heads, value_head_dim],
        }
        if self.kda_attrs["gate_lower_bound"] is not None:
            shared_kwargs["gate_lower_bound"] = self.kda_attrs["gate_lower_bound"]
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

    def make_gated_delta_net_gates(self, name, kwargs):
        return kwargs["decay"], kwargs["beta"]

    def make_linear_attention_a_b_proj(self, layer_id, attention, root_input):
        basename = f"/model/layers.{layer_id}/linear_attn"
        b_name = f"{basename}/b_proj/MatMul"
        self.require_dense_linear_attention_gate(attention.in_proj_b, b_name)
        self.exclude_node_from_quantization(b_name)
        b_name = self.make_matmul(attention.in_proj_b, b_name, root_input)

        # In KDA, f_proj (in_proj_a) has full head dimension (num_heads * head_dim = 2048),
        # so it uses the model's quantized linear representation (e.g. FP8 or INT8).
        a_name = self.make_matmul(attention.in_proj_a, f"{basename}/a_proj/MatMul", root_input)
        return f"{b_name}/output_0", f"{a_name}/output_0"

    def make_linear_attention_qkv_z_proj(self, layer_id, attention, root_input):
        basename = f"/model/layers.{layer_id}/linear_attn"
        qkv_name = self.make_matmul(attention.in_proj_qkv, f"{basename}/qkv_proj/MatMul", root_input)
        if attention.in_proj_z is not None:
            z_name = self.make_matmul(attention.in_proj_z, f"{basename}/z_proj/MatMul", root_input)
        else:
            za_name = self.make_matmul(attention.in_proj_z_a, f"{basename}/z_proj_a/MatMul", root_input)
            z_name = self.make_matmul(attention.in_proj_z_b, f"{basename}/z_proj_b/MatMul", f"{za_name}/output_0")
        return f"{qkv_name}/output_0", f"{z_name}/output_0"

    def make_linear_attention_output_proj(self, layer_id, attention, attn_output_3d, z_name):
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
            activation="sigmoid",
        )
        o_name = self.make_matmul(attention.out_proj, f"{basename}/out_proj/MatMul", f"{gated_norm_name}/output_0")
        self.layernorm_attrs["skip_input"] = f"{o_name}/output_0"

    def make_mla_rope(self, name: str, root_input: str, num_heads: int, token_shape: list):
        cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches()
        dim = num_heads * self.mla_attrs["qk_rope_head_dim"]
        pos_ids = self.input_names["position_ids"]
        rope_in = root_input
        if self.use_paged_attention:
            u_in = f"{name}/Unsqueeze_in"
            self.make_unsqueeze(u_in, [root_input, "/model/constants/INT64/[0]"], self.io_dtype, [1, "num_tokens", dim])
            rope_in = f"{u_in}/output_0"
            u_pos = f"{name}/Unsqueeze_pos"
            self.make_unsqueeze(u_pos, [pos_ids, "/model/constants/INT64/[0]"], ir.DataType.INT64, [1, "num_tokens"])
            pos_ids = f"{u_pos}/output_0"

        rope_dtype = ir.DataType.FLOAT if self.rope_attrs["cast"]["use_fp32"] else self.io_dtype
        rope_node_in = rope_in
        if rope_dtype != self.io_dtype:
            cast_in = f"{name}/Cast_in"
            self.make_cast(
                cast_in,
                rope_in,
                rope_dtype,
                [1, "num_tokens", dim] if self.use_paged_attention else [*token_shape, dim],
            )
            rope_node_in = f"{cast_in}/output_0"

        rope_out = f"{name}/rope_raw"
        self.make_node(
            "RotaryEmbedding",
            inputs=[rope_node_in, pos_ids, cos_cache_name, sin_cache_name],
            outputs=[rope_out],
            name=name,
            domain="com.microsoft",
            interleaved=self.rope_attrs["interleaved"],
            num_heads=num_heads,
            rotary_embedding_dim=self.mla_attrs["qk_rope_head_dim"],
        )
        self.make_value(
            rope_out,
            rope_dtype,
            [1, "num_tokens", dim] if self.use_paged_attention else [*token_shape, dim],
        )
        out = rope_out
        if rope_dtype != self.io_dtype:
            cast_out = f"{name}/Cast_out"
            self.make_cast(
                cast_out,
                rope_out,
                self.io_dtype,
                [1, "num_tokens", dim] if self.use_paged_attention else [*token_shape, dim],
            )
            out = f"{cast_out}/output_0"

        if self.use_paged_attention:
            sq_out = f"{name}/Squeeze_out"
            self.make_squeeze(sq_out, [out, "/model/constants/INT64/[0]"], self.io_dtype, ["num_tokens", dim])
            out = f"{sq_out}/output_0"
        return out

    def make_mla_query(self, layer_id, attention, root_input):
        basename = f"/model/layers.{layer_id}/attn"
        if self.mla_attrs["q_lora_rank"] is None:
            return self.make_matmul(attention.q_proj, f"{basename}/q_proj/MatMul", root_input)
        qa_matmul = self.make_matmul(attention.q_a_proj, f"{basename}/q_a_proj/MatMul", root_input)
        qa_norm = self.make_mla_projection_norm(layer_id, attention.q_a_layernorm, f"{qa_matmul}/output_0", "q")
        return self.make_matmul(attention.q_b_proj, f"{basename}/q_b_proj/MatMul", qa_norm)

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        if self.layer_types[layer_id] == "linear_attention":
            self.make_qwen_gated_delta_net(layer_id, attention, root_input)
        else:
            self.make_mla_attention(layer_id, attention, root_input, **kwargs)

    def make_mla_attention(self, layer_id, attention, root_input, **kwargs):
        """Query/KV projections -> attention -> value slice -> sigmoid gate -> output projection."""
        q_path = self.make_mla_query_projection(layer_id, attention, root_input)
        k_path, v_path = self.make_mla_key_value_projection(layer_id, attention, root_input)
        attn_op_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        past_k, past_v, present_k, present_v = self.make_key_value_cache_names(layer_id)
        self.make_attention_op(
            attn_op_name,
            layer_id=layer_id,
            root_input=root_input,
            q_path=q_path,
            k_path=k_path,
            v_path=v_path,
            past_k=past_k,
            past_v=past_v,
            present_k=present_k,
            present_v=present_v,
            cos_cache="",
            sin_cache="",
            sinks="",
            q_norm_weight="",
            k_norm_weight="",
            qk_norm_epsilon=self.attention_attrs["qk_norm_epsilon"],
            **kwargs,
        )
        self.make_mla_output_projection(layer_id, attention, root_input, f"{attn_op_name}/output_0")

    def make_mla_projection_norm(self, layer_id, layernorm, root_input, location):
        weight = f"model.layers.{layer_id}.attn.{location}_a_layernorm.weight"
        self.make_initializer(layernorm.weight, weight, to=self.io_dtype)
        name = f"/model/layers.{layer_id}/attn/{location}_a_layernorm/LayerNorm"
        output = f"{name}/output_0"
        self.make_layernorm_op(
            name,
            op_type="SimplifiedLayerNormalization",
            inputs=[root_input, weight],
            outputs=[output],
            skip=False,
            new_io_dtype=self.io_dtype,
            axis=-1,
            epsilon=self.layernorm_attrs["epsilon"],
            stash_type=1,
        )
        self.make_value(
            output, self.io_dtype, self.make_hidden_state_shape(last_dim=self.mla_attrs[f"{location}_lora_rank"])
        )
        return output

    def make_mla_query_projection(self, layer_id, attention, root_input):
        """Query LoRA projection -> RMSNorm -> up projection -> split -> RoPE -> packed heads."""
        basename = f"/model/layers.{layer_id}/attn"
        packed = self.use_paged_attention
        token_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        kept_axes = "0" if packed else "0, 0"
        num_heads = self.num_attn_heads
        nope_dim = self.mla_attrs["qk_nope_head_dim"]
        rope_dim = self.mla_attrs["qk_rope_head_dim"]
        qk_dim = self.head_size

        qb_matmul = self.make_mla_query(layer_id, attention, root_input)
        q_reshape = f"{qb_matmul.removesuffix('/MatMul')}/Reshape"
        self.make_reshape(
            q_reshape,
            [f"{qb_matmul}/output_0", f"/model/constants/INT64/[{kept_axes}, {num_heads}, {qk_dim}]"],
            self.io_dtype,
            [*token_shape, num_heads, qk_dim],
        )
        q_split = f"{basename}/q_split/Split"
        q_nope = f"{q_split}/output_0"
        q_pe_4d = f"{q_split}/output_1"
        self.make_split(
            q_split,
            inputs=[f"{q_reshape}/output_0", f"/model/constants/INT64/[{nope_dim}, {rope_dim}]"],
            outputs=[q_nope, q_pe_4d],
            dtypes=[self.io_dtype, self.io_dtype],
            shapes=[[*token_shape, num_heads, nope_dim], [*token_shape, num_heads, rope_dim]],
            axis=-1,
        )
        q_pe_flat = f"{basename}/q_pe_flat/Reshape"
        self.make_reshape(
            q_pe_flat,
            [q_pe_4d, f"/model/constants/INT64/[{kept_axes}, {num_heads * rope_dim}]"],
            self.io_dtype,
            [*token_shape, num_heads * rope_dim],
        )
        q_pe_rot_flat = self.make_mla_rope(f"{basename}/q_rotary", f"{q_pe_flat}/output_0", num_heads, token_shape)
        q_pe_rot = f"{basename}/q_pe_rot/Reshape"
        self.make_reshape(
            q_pe_rot,
            [q_pe_rot_flat, f"/model/constants/INT64/[{kept_axes}, {num_heads}, {rope_dim}]"],
            self.io_dtype,
            [*token_shape, num_heads, rope_dim],
        )
        q_concat = f"{basename}/q_concat/Concat"
        self.make_concat(
            q_concat,
            [q_nope, f"{q_pe_rot}/output_0"],
            self.io_dtype,
            [*token_shape, num_heads, qk_dim],
            axis=-1,
        )
        q_final = f"{basename}/q_final/Reshape"
        self.make_reshape(
            q_final,
            [f"{q_concat}/output_0", f"/model/constants/INT64/[{kept_axes}, {num_heads * qk_dim}]"],
            self.io_dtype,
            [*token_shape, num_heads * qk_dim],
        )
        return f"{q_final}/output_0"

    def make_mla_kv_projection(self, layer_id, attention, root_input):
        """KV LoRA projection -> split -> RMSNorm -> up projection -> key/value split."""
        basename = f"/model/layers.{layer_id}/attn"
        packed = self.use_paged_attention
        token_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        kept_axes = "0" if packed else "0, 0"
        num_heads = self.num_attn_heads
        nope_dim = self.mla_attrs["qk_nope_head_dim"]
        rope_dim = self.mla_attrs["qk_rope_head_dim"]
        v_dim = self.mla_attrs["v_head_dim"]
        kva_matmul = self.make_matmul(attention.kv_a_proj_with_mqa, f"{basename}/kv_a_proj/MatMul", root_input)
        kva_split = f"{basename}/kv_a_split/Split"
        kv_comp = f"{kva_split}/output_0"
        k_pe_flat = f"{kva_split}/output_1"
        self.make_split(
            kva_split,
            inputs=[f"{kva_matmul}/output_0", f"/model/constants/INT64/[{self.mla_attrs['kv_lora_rank']}, {rope_dim}]"],
            outputs=[kv_comp, k_pe_flat],
            dtypes=[self.io_dtype, self.io_dtype],
            shapes=[[*token_shape, self.mla_attrs["kv_lora_rank"]], [*token_shape, rope_dim]],
            axis=-1,
        )
        kva_norm = self.make_mla_projection_norm(layer_id, attention.kv_a_layernorm, kv_comp, "kv")
        kvb_matmul = self.make_matmul(attention.kv_b_proj, f"{basename}/kv_b_proj/MatMul", kva_norm)
        kvb_reshape = f"{basename}/kv_b_proj/Reshape"
        self.make_reshape(
            kvb_reshape,
            [f"{kvb_matmul}/output_0", f"/model/constants/INT64/[{kept_axes}, {num_heads}, {nope_dim + v_dim}]"],
            self.io_dtype,
            [*token_shape, num_heads, nope_dim + v_dim],
        )
        kvb_split = f"{basename}/kv_b_split/Split"
        k_nope = f"{kvb_split}/output_0"
        v_states = f"{kvb_split}/output_1"
        self.make_split(
            kvb_split,
            inputs=[f"{kvb_reshape}/output_0", f"/model/constants/INT64/[{nope_dim}, {v_dim}]"],
            outputs=[k_nope, v_states],
            dtypes=[self.io_dtype, self.io_dtype],
            shapes=[[*token_shape, num_heads, nope_dim], [*token_shape, num_heads, v_dim]],
            axis=-1,
        )
        return k_nope, k_pe_flat, v_states

    def make_mla_key_value_projection(self, layer_id, attention, root_input):
        """KV projection -> single-head RoPE -> key broadcast -> value padding -> packed heads."""
        k_nope, k_pe_flat, v_states = self.make_mla_kv_projection(layer_id, attention, root_input)
        basename = f"/model/layers.{layer_id}/attn"
        packed = self.use_paged_attention
        token_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        kept_axes = "0" if packed else "0, 0"
        num_heads = self.num_attn_heads
        rope_dim = self.mla_attrs["qk_rope_head_dim"]
        qk_dim = self.head_size
        k_pe_rot_flat = self.make_mla_rope(f"{basename}/k_rotary", k_pe_flat, 1, token_shape)
        k_pe_unsq = f"{basename}/k_pe_rot/Unsqueeze"
        self.make_unsqueeze(
            k_pe_unsq,
            [k_pe_rot_flat, "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [*token_shape, 1, rope_dim],
        )
        k_pe_tiles = f"{basename}/k_pe_rot/Concat_heads"
        self.make_concat(
            k_pe_tiles,
            [f"{k_pe_unsq}/output_0"] * num_heads,
            self.io_dtype,
            [*token_shape, num_heads, rope_dim],
            axis=-2,
        )
        k_concat = f"{basename}/k_concat/Concat"
        self.make_concat(
            k_concat,
            [k_nope, f"{k_pe_tiles}/output_0"],
            self.io_dtype,
            [*token_shape, num_heads, qk_dim],
            axis=-1,
        )
        k_final = f"{basename}/k_final/Reshape"
        self.make_reshape(
            k_final,
            [f"{k_concat}/output_0", f"/model/constants/INT64/[{kept_axes}, {num_heads * qk_dim}]"],
            self.io_dtype,
            [*token_shape, num_heads * qk_dim],
        )
        v_path = self.make_mla_value_padding(layer_id, v_states, f"{k_pe_tiles}/output_0")
        return f"{k_final}/output_0", v_path

    def make_mla_value_padding(self, layer_id, v_states, rotary_keys):
        basename = f"/model/layers.{layer_id}/attn"
        packed = self.use_paged_attention
        token_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        kept_axes = "0" if packed else "0, 0"
        num_heads = self.num_attn_heads
        rope_dim = self.mla_attrs["qk_rope_head_dim"]
        qk_dim = self.head_size
        # Pad v_states from 128 to 192 per head using GPU Mul(0) + Concat so all nodes run on CUDA.
        v_zeros = f"{basename}/v_pad/Mul_zero"
        self.make_mul(
            v_zeros,
            [rotary_keys, f"/model/constants/{self.io_dtype}/0.0"],
            self.io_dtype,
            [*token_shape, num_heads, rope_dim],
        )
        v_pad = f"{basename}/v_pad/Concat"
        self.make_concat(
            v_pad,
            [v_states, f"{v_zeros}/output_0"],
            self.io_dtype,
            [*token_shape, num_heads, qk_dim],
            axis=-1,
        )
        v_final = f"{basename}/v_final/Reshape"
        self.make_reshape(
            v_final,
            [f"{v_pad}/output_0", f"/model/constants/INT64/[{kept_axes}, {num_heads * qk_dim}]"],
            self.io_dtype,
            [*token_shape, num_heads * qk_dim],
        )
        return f"{v_final}/output_0"

    def make_mla_output_projection(self, layer_id, attention, root_input, attn_output):
        """Slice padded value channels -> per-head sigmoid gate -> output projection."""
        basename = f"/model/layers.{layer_id}/attn"
        packed = self.use_paged_attention
        token_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        kept_axes = "0" if packed else "0, 0"
        num_heads = self.num_attn_heads
        qk_dim = self.head_size
        v_dim = self.mla_attrs["v_head_dim"]
        attn_4d = f"{basename}/attn_out/Reshape"
        self.make_reshape(
            attn_4d,
            [attn_output, f"/model/constants/INT64/[{kept_axes}, {num_heads}, {qk_dim}]"],
            self.io_dtype,
            [*token_shape, num_heads, qk_dim],
        )
        attn_slice = f"{basename}/attn_out/Slice"
        self.make_slice(
            attn_slice,
            [
                f"{attn_4d}/output_0",
                "/model/constants/INT64/[0]",
                f"/model/constants/INT64/[{v_dim}]",
                "/model/constants/INT64/[-1]",
            ],
            self.io_dtype,
            [*token_shape, num_heads, v_dim],
        )
        g_matmul = self.make_matmul(attention.g_proj, f"{basename}/g_proj/MatMul", root_input)
        g_sig = f"{basename}/g_proj/Sigmoid"
        self.make_sigmoid(g_sig, f"{g_matmul}/output_0", self.io_dtype, [*token_shape, num_heads])
        g_unsq = f"{basename}/g_proj/Unsqueeze"
        self.make_unsqueeze(
            g_unsq,
            [f"{g_sig}/output_0", "/model/constants/INT64/[-1]"],
            self.io_dtype,
            [*token_shape, num_heads, 1],
        )
        g_mul = f"{basename}/g_proj/Mul"
        self.make_mul(
            g_mul,
            [f"{attn_slice}/output_0", f"{g_unsq}/output_0"],
            self.io_dtype,
            [*token_shape, num_heads, v_dim],
        )
        attn_flat = f"{basename}/attn_gated/Reshape"
        self.make_reshape(
            attn_flat,
            [f"{g_mul}/output_0", f"/model/constants/INT64/[{kept_axes}, {num_heads * v_dim}]"],
            self.io_dtype,
            [*token_shape, num_heads * v_dim],
        )
        o_matmul = self.make_matmul(attention.dense, f"{basename}/o_proj/MatMul", f"{attn_flat}/output_0")
        self.layernorm_attrs["skip_input"] = f"{o_matmul}/output_0"

    def make_moe(self, layer_id, moe, root_input):
        if layer_id < self.moe_attrs["first_k_dense_replace"]:
            prev_inter = self.intermediate_size
            try:
                self.intermediate_size = self.moe_attrs["dense_intermediate_size"]
                self.make_mlp(layer_id, moe, root_input=root_input)
            finally:
                self.intermediate_size = prev_inter
            return
        self.make_moe_preprocessing(layer_id, moe, root_input)
        router_probs, output_scale = self.make_moe_router(layer_id, moe, root_input)
        self.make_moe_subgraph(layer_id, moe, root_input, router_probs, output_scale)

    def make_moe_preprocessing(self, layer_id, moe, root_input):
        moe.gate.exclude_from_quantization = True
        self.make_interleaved_swiglu_moe_preprocessing(layer_id, moe, self.intermediate_size)

    def make_moe_router(self, layer_id, moe, root_input):
        """Emit BailingMoeV3's fused MoEGroupRouter (noaux_tc group-limited sigmoid + expert_bias routing)."""
        basename = f"/model/layers.{layer_id}/moe/router"
        top_k = self.moe_attrs["top_k"]
        n_group = self.moe_attrs["n_group"]
        topk_group = self.moe_attrs["topk_group"]
        logits_shape = self.make_moe_router_shape()

        router_input = root_input
        if self.io_dtype != ir.DataType.FLOAT:
            cast_name = f"{basename}/Cast_input"
            self.make_cast(cast_name, root_input, ir.DataType.FLOAT, self.make_hidden_state_shape())
            router_input = f"{cast_name}/output_0"
        matmul_name = f"{basename}/MatMul"
        weight_name = f"model.layers.{layer_id}.moe.router.weight"
        self.make_initializer(moe.gate.weight.T, weight_name, to=ir.DataType.FLOAT)
        self.make_node(
            "MatMul",
            inputs=[router_input, weight_name],
            outputs=[f"{matmul_name}/output_0"],
            name=matmul_name,
        )
        self.make_value(
            f"{matmul_name}/output_0",
            ir.DataType.FLOAT,
            shape=self.make_hidden_state_shape(last_dim=self.moe_attrs["num_experts"]),
        )
        self.exclude_node_from_quantization(matmul_name)
        expert_bias_name = f"model.layers.{layer_id}.moe.expert_bias"
        self.make_initializer(moe.expert_bias, expert_bias_name, to=ir.DataType.FLOAT)

        router_name = f"{basename}/MoEGroupRouter"
        router_probs_name = f"{router_name}/output_0"
        self.make_node(
            "MoEGroupRouter",
            inputs=[f"{matmul_name}/output_0", expert_bias_name],
            outputs=[router_probs_name],
            name=router_name,
            domain="com.microsoft",
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            group_topk=self.moe_attrs["group_topk"],
            router_sentinel=float(self.moe_attrs["router_sentinel"]),
        )
        self.make_value(router_probs_name, ir.DataType.FLOAT, shape=logits_shape)
        if self.io_dtype != ir.DataType.FLOAT:
            cast_name = f"{basename}/Cast_scores"
            self.make_cast(cast_name, router_probs_name, self.io_dtype, logits_shape)
            router_probs_name = f"{cast_name}/output_0"

        routed_scale = self.moe_attrs["routed_scaling_factor"]
        scale_const = f"/model/constants/{self.io_dtype}/{routed_scale}"
        if not self.moe_attrs["normalize_routing_weights"]:
            # MoE softmax normalizes the selected log-scores even without top-k renormalization.
            scores_name = f"{basename}/Exp"
            self.make_exp(scores_name, router_probs_name, self.io_dtype, logits_shape)
            sum_name = f"{basename}/ReduceSum"
            self.make_reduce_sum(
                sum_name,
                [f"{scores_name}/output_0", "/model/constants/INT64/[-1]"],
                self.io_dtype,
                [logits_shape[0], 1],
                keepdims=True,
            )
            sum_output = f"{sum_name}/output_0"
            if not self.use_paged_attention:
                shape_name = f"{basename}/token_shape/Shape"
                self.make_shape(shape_name, root_input, [2], end=2)
                concat_name = f"{basename}/token_shape/Concat"
                self.make_concat(
                    concat_name,
                    [f"{shape_name}/output_0", "/model/constants/INT64/[1]"],
                    ir.DataType.INT64,
                    [3],
                    axis=0,
                )
                reshape_name = f"{basename}/weight_sum/Reshape"
                self.make_reshape(
                    reshape_name,
                    [sum_output, f"{concat_name}/output_0"],
                    self.io_dtype,
                    self.make_hidden_state_shape(last_dim=1),
                )
                sum_output = f"{reshape_name}/output_0"
            scale_name = f"{basename}/output_scale/Mul"
            self.make_mul(
                scale_name,
                [sum_output, scale_const],
                self.io_dtype,
                self.make_hidden_state_shape(last_dim=1),
            )
            scale_const = f"{scale_name}/output_0"
        return router_probs_name, scale_const

    def make_moe_subgraph(self, layer_id, moe, root_input, router_probs=None, output_scale=None):
        basename = f"/model/layers.{layer_id}/moe"
        op_type = self.moe_attrs["op_type"]
        names = self.make_moe_expert_names(layer_id)
        moe_name = f"{basename}/{op_type}"
        limits = getattr(self.config, "expert_swiglu_limit_list", None)
        limit = float(limits[layer_id]) if limits else 0
        activation_attrs = {"swiglu_limit": limit, "swiglu_limit_after_activation": 1} if limit > 0 else {}
        self.make_moe_op(
            moe_name,
            root_input=root_input,
            router_probs=router_probs,
            weight1=names["gate_up_weight"],
            scales1=names["gate_up_scales"] if op_type == "QMoE" else "",
            bias1="",
            weight2=names["down_weight"],
            scales2=names["down_scales"] if op_type == "QMoE" else "",
            bias2="",
            **activation_attrs,
        )
        mul_name = f"{basename}/Mul_scale"
        self.make_mul(
            mul_name,
            [f"{moe_name}/output_0", output_scale],
            dtype=self.io_dtype,
            shape=self.make_hidden_state_shape(),
        )

        # Shared expert MLP
        intermediate_size = self.intermediate_size
        try:
            self.intermediate_size = self.moe_attrs["shared_expert_intermediate_size"]
            self.make_mlp_proj(layer_id, moe.shared_experts, root_input)
        finally:
            self.intermediate_size = intermediate_size
        shared_output = self.mlp_attrs["output_0"]

        add_name = f"{basename}/Add_shared"
        self.make_add(
            add_name,
            [f"{mul_name}/output_0", shared_output],
            dtype=self.io_dtype,
            shape=self.make_hidden_state_shape(),
        )
        self.layernorm_attrs["skip_input"] = f"{add_name}/output_0"

    def update_genai_config(self, genai_config):
        super().update_genai_config(genai_config)
        if self.ep == "cuda" and self.use_paged_attention and self.io_dtype == ir.DataType.BFLOAT16:
            genai_config["model"]["decoder"]["session_options"]["ep.cuda.gated_delta_net_bf16_precision"] = (
                self.kda_attrs["bf16_precision"]
            )

    def make_mlp_proj(self, layer_id, mlp, root_input):
        limits = getattr(self.config, "share_expert_swiglu_limit_list", None)
        limit = limits[layer_id] if limits else 0
        if layer_id < self.moe_attrs["first_k_dense_replace"] or limit <= 0:
            super().make_mlp_proj(layer_id, mlp, root_input)
            return
        basename = f"/model/layers.{layer_id}/mlp"
        names = []
        inputs = []
        for kind in ("gate", "up"):
            linear = getattr(mlp, f"{kind}_proj")
            name = self.make_matmul(linear, f"{basename}/{kind}_proj/MatMul", root_input)
            names.append(name)
            if linear.bias is not None:
                add = f"{basename}/{kind}_proj/Add"
                self.make_add_bias(linear.bias, add, root_input=f"{name}/output_0")
                name = add
            inputs.append(f"{name}/output_0")
        self.make_mlp_proj_fused(layer_id, *names)
        activated = Standard.make_clipped_swiglu(
            self,
            f"{basename}/clipped_swiglu",
            *inputs,
            limit,
            self.make_hidden_state_shape(last_dim=self.intermediate_size),
        )
        down = self.make_matmul(mlp.down_proj, f"{basename}/down_proj/MatMul", activated)
        if mlp.down_proj.bias is not None:
            add = f"{basename}/down_proj/Add"
            self.make_add_bias(mlp.down_proj.bias, add, root_input=f"{down}/output_0")
            down = add
        self.mlp_attrs["output_0"] = f"{down}/output_0"

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        super().save_processing(model_name_or_path, extra_kwargs, out_dir)
        tokenizer_json_path = os.path.join(out_dir, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, encoding="utf-8") as f:
                tok_data = json.load(f)
            compatible_ling_regex = (
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| "
                r"?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            )
            pre_tok = tok_data.get("pre_tokenizer", {})
            for entry in pre_tok.get("pretokenizers", []):
                pattern = entry.get("pattern", {})
                if isinstance(pattern, dict) and "Regex" in pattern:
                    if "?+" in pattern["Regex"] or "++" in pattern["Regex"]:
                        pattern["Regex"] = compatible_ling_regex
            with open(tokenizer_json_path, "w", encoding="utf-8") as f:
                json.dump(tok_data, f, ensure_ascii=False, indent=2)
