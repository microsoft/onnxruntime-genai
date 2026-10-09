import copy
import json
import os

import onnx_ir as ir
import torch
from builder_config import serialize_genai_config
from huggingface_hub import hf_hub_download, snapshot_download

from .base import Model
from .expansions import Qwen38
from .mtp import MTPModel
from .qwen3_5 import Qwen35MoETextModel


class Qwen4ExpTextModel(Qwen35MoETextModel, Qwen38):
    """Qwen4-Exp decoder builder using external token/vision embeddings."""

    CPU_EMBEDDING_ANNOTATION = "cpu_embedding"

    def make_moe_preprocessing(self, layer_id, moe, root_input):
        if getattr(moe.experts, "quant_type", None) != "fp8_block":
            return super().make_moe_preprocessing(layer_id, moe, root_input)
        if self.ep not in {"cuda", "webgpu"}:
            raise ValueError("Native block-FP8 Qwen3.8 experts require the CUDA or WebGPU execution provider.")
        self.moe_attrs.update(
            op_type="QMoE",
            quant_type="fp8",
            expert_weight_bits=8,
            block_size=128,
            activation_type="silu",
            swiglu_fusion=0,
            weights_prepacked=-1,
        )
        names = self.make_moe_expert_names(layer_id)
        projections = [getattr(moe.experts, str(index)) for index in range(self.moe_attrs["num_experts"])]
        gate = torch.stack([expert.gate_proj.weight for expert in projections])
        gate_scales = torch.stack([expert.gate_proj.weight_scale_inv for expert in projections])
        up = torch.stack([expert.up_proj.weight for expert in projections])
        up_scales = torch.stack([expert.up_proj.weight_scale_inv for expert in projections])
        down = torch.stack([expert.down_proj.weight for expert in projections])
        down_scales = torch.stack([expert.down_proj.weight_scale_inv for expert in projections])
        scale_dtype = ir.DataType.FLOAT if self.ep == "webgpu" else None
        self.make_initializer(gate, names["gate_up_weight"])
        self.make_initializer(gate_scales, names["gate_up_scales"], to=scale_dtype)
        prefix = f"model.layers.{layer_id}.moe.experts.up_proj"
        self.moe_attrs["up_projection_names"] = (f"{prefix}.qweight", f"{prefix}.scales")
        self.make_initializer(up, f"{prefix}.qweight")
        self.make_initializer(up_scales, f"{prefix}.scales", to=scale_dtype)
        self.make_initializer(down, names["down_weight"])
        self.make_initializer(down_scales, names["down_scales"], to=scale_dtype)

    def make_moe_expert_names(self, layer_id):
        names = super().make_moe_expert_names(layer_id)
        if self.moe_attrs.get("quant_type") == "fp8" and self.moe_attrs.get("block_size", 0) > 0:
            for name in ("gate_up_weight", "gate_up_scales"):
                names[name] = names[name].replace("gate_up_proj", "gate_proj")
            names["gate_up_bias"] = names["down_bias"] = ""
        return names

    def make_moe_op(self, name, **kwargs):
        if self.moe_attrs.get("quant_type") == "fp8" and self.moe_attrs.get("block_size", 0) > 0:
            kwargs["weight3"], kwargs["scales3"] = self.moe_attrs["up_projection_names"]
        return super().make_moe_op(name, **kwargs)

    def is_packed_matmul_supported(self):
        return Model.is_packed_matmul_supported(self)

    def make_attention_init(self, config):
        super().make_attention_init(config)
        self.attention_attrs["use_packed_matmul"] = self.is_packed_matmul_supported()

    def select_projection_outputs(self, projection, indices):
        """Clone a projection while selecting output rows and their quantization metadata."""
        selected = copy.copy(projection)
        if hasattr(projection, "_parameters"):
            selected._parameters = projection._parameters.copy()
        if hasattr(projection, "_buffers"):
            selected._buffers = projection._buffers.copy()
        out_features = getattr(projection, "out_features", None)
        if out_features is None:
            out_features = projection.weight.shape[0]
        for attribute in ("weight", "bias", "qweight", "scales", "qzeros", "weight_scale"):
            tensor = getattr(projection, attribute, None)
            if tensor is None or tensor.ndim == 0:
                continue
            index = indices.to(tensor.device)
            if tensor.shape[0] == out_features:
                value = tensor.index_select(0, index).contiguous()
            elif tensor.ndim > 1 and tensor.shape[1] == out_features:
                value = tensor.index_select(1, index).contiguous()
            else:
                continue
            if isinstance(tensor, torch.nn.Parameter):
                value = torch.nn.Parameter(value, requires_grad=False)
            setattr(selected, attribute, value)
        selected.out_features = indices.numel()
        return selected

    def make_attention_input_proj(self, layer_id, attention, root_input, **kwargs):
        indexer_proj = kwargs.pop("indexer_proj", None)
        if indexer_proj is not None and self.attention_attrs["use_packed_matmul"]:
            q_size = self.q_size
            q_gate_rows = torch.arange(2 * q_size).reshape(self.num_attn_heads, 2, self.head_size)
            q_proj = self.select_projection_outputs(attention.q_proj, q_gate_rows[:, 0, :].reshape(-1))
            gate_proj = self.select_projection_outputs(attention.q_proj, q_gate_rows[:, 1, :].reshape(-1))

            qkv_projections = (q_proj, attention.k_proj, attention.v_proj)
            packed_qkv = self.make_packed_matmul_class(*qkv_projections)
            packed_qkv_name = self.make_matmul(
                packed_qkv,
                f"/model/layers.{layer_id}/attn/qkv_proj/MatMul",
                root_input,
            )
            packed_qkv_output = f"{packed_qkv_name}/output_0"
            biases = [getattr(projection, "bias", None) for projection in qkv_projections]
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
                        for projection, bias in zip(qkv_projections, biases, strict=True)
                    ]
                )
                packed_add = f"/model/layers.{layer_id}/attn/qkv_proj/Add"
                self.make_add_bias(packed_bias, packed_add, packed_qkv_output)
                packed_qkv_output = f"{packed_add}/output_0"

            gate_name = self.make_matmul(
                gate_proj,
                f"/model/layers.{layer_id}/attn/gate_proj/MatMul",
                root_input,
            )
            gate_output = f"{gate_name}/output_0"
            gate_bias = getattr(gate_proj, "bias", None)
            if gate_bias is not None and torch.count_nonzero(gate_bias) > 0:
                gate_add = f"/model/layers.{layer_id}/attn/gate_proj/Add"
                self.make_add_bias(gate_bias, gate_add, gate_output)
                gate_output = f"{gate_add}/output_0"

            indexer_name = self.make_matmul(
                indexer_proj,
                f"/model/layers.{layer_id}/attn/indexer/index_qk_proj/MatMul",
                root_input,
            )
            indexer_output = f"{indexer_name}/output_0"
            indexer_bias = getattr(indexer_proj, "bias", None)
            if indexer_bias is not None and torch.count_nonzero(indexer_bias) > 0:
                indexer_add = f"/model/layers.{layer_id}/attn/indexer/index_qk_proj/Add"
                self.make_add_bias(indexer_bias, indexer_add, indexer_output)
                indexer_output = f"{indexer_add}/output_0"

            self.attention_attrs["q_path"] = packed_qkv_output
            self.attention_attrs["k_path"] = ""
            self.attention_attrs["v_path"] = ""
            self.attention_attrs["gate_path"] = gate_output
            self.attention_attrs["indexer_qk_path"] = indexer_output
            return

        if not self.attention_attrs["use_packed_matmul"]:
            return super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)

        q_size = self.q_size
        self.q_size = 2 * q_size
        try:
            super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)
        finally:
            self.q_size = q_size

    def make_linear_attention_qkv_z_proj(self, layer_id, attention, root_input):
        qkv = attention.in_proj_qkv
        z = attention.in_proj_z
        if any(getattr(projection, "quant_type", "none") != "none" for projection in (qkv, z)):
            return super().make_linear_attention_qkv_z_proj(layer_id, attention, root_input)

        basename = f"/model/layers.{layer_id}/linear_attn"
        packed_name = f"{basename}/qkv_z_proj/MatMul"

        if hasattr(qkv, "qweight") and hasattr(z, "qweight"):
            if qkv.bits != z.bits or qkv.group_size != z.group_size or qkv.in_features != z.in_features:
                raise ValueError("QKV and Z must use compatible quantization to share a packed MatMul.")
            if hasattr(qkv, "qzeros") != hasattr(z, "qzeros"):
                raise ValueError("QKV and Z must use the same zero-point representation to share a packed MatMul.")

            class PackedQkvZ:
                qweight = torch.cat([qkv.qweight, z.qweight], dim=0)
                scales = torch.cat([qkv.scales, z.scales], dim=0)
                qzeros = torch.cat([qkv.qzeros, z.qzeros], dim=0) if hasattr(qkv, "qzeros") else None
                g_idx = getattr(qkv, "g_idx", None)
                in_features = qkv.in_features
                out_features = qkv.out_features + z.out_features
                bits = qkv.bits
                group_size = qkv.group_size

            packed_matmul = self.make_matmul_nbits(PackedQkvZ(), packed_name, root_input)
            qkv_size = qkv.out_features
            z_size = z.out_features
        elif hasattr(qkv, "weight") and hasattr(z, "weight"):

            class PackedQkvZ:
                weight = torch.cat([qkv.weight, z.weight], dim=0)

            packed_matmul = self.make_matmul(PackedQkvZ(), packed_name, root_input)
            qkv_size = qkv.weight.shape[0]
            z_size = z.weight.shape[0]
        else:
            raise ValueError("QKV and Z must use the same weight representation to share a packed MatMul.")

        qkv_name = f"{basename}/qkv_proj/MatMul"
        z_name = f"{basename}/z_proj/MatMul"
        self.make_split(
            f"{basename}/qkv_z_proj/Split",
            [f"{packed_matmul}/output_0", f"/model/constants/INT64/[{qkv_size}, {z_size}]"],
            [f"{qkv_name}/output_0", f"{z_name}/output_0"],
            [self.io_dtype] * 2,
            [
                self.make_hidden_state_shape(last_dim=qkv_size),
                self.make_hidden_state_shape(last_dim=z_size),
            ],
            axis=-1,
        )
        return qkv_name, z_name

    def make_linear_attention_a_b_proj(self, layer_id, attention, root_input):
        basename = f"/model/layers.{layer_id}/linear_attn"
        a = attention.in_proj_a
        b = attention.in_proj_b
        a_name = f"{basename}/a_proj/MatMul"
        b_name = f"{basename}/b_proj/MatMul"
        self.require_dense_linear_attention_gate(a, a_name)
        self.require_dense_linear_attention_gate(b, b_name)

        packed_name = f"{basename}/a_b_proj/MatMul"

        class PackedAB:
            weight = torch.cat([a.weight, b.weight], dim=0)

        packed_matmul = self.make_matmul(PackedAB(), packed_name, root_input)
        a_size = a.weight.shape[0]
        b_size = b.weight.shape[0]
        self.make_split(
            f"{basename}/a_b_proj/Split",
            [f"{packed_matmul}/output_0", f"/model/constants/INT64/[{a_size}, {b_size}]"],
            [f"{a_name}/output_0", f"{b_name}/output_0"],
            [self.io_dtype] * 2,
            [
                self.make_hidden_state_shape(last_dim=a_size),
                self.make_hidden_state_shape(last_dim=b_size),
            ],
            axis=-1,
        )
        return b_name, a_name

    def make_moe_router(self, layer_id, moe, root_input):
        return super().make_moe_router(layer_id, moe, root_input)

    def make_shared_expert(self, layer_id, shared_expert, shared_expert_gate, root_input):
        return super().make_shared_expert(layer_id, shared_expert, shared_expert_gate, root_input)

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        extra_options = copy.deepcopy(extra_options)
        text_only = extra_options.get("text_only", False)
        extra_options["exclude_embeds"] = not text_only
        extra_options.setdefault("filename", "model.onnx" if text_only else "text.onnx")
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        if not hasattr(self, "context_length_attrs"):
            self.context_length_attrs = {"state_window": 0, "state_window_dims": []}
        self.use_cpu_embedding_gather = False
        if self.use_paged_attention:
            self.input_names.pop("position_ids", None)
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
        self.external_engram = extra_options.get("external_engram", False)
        if getattr(self, "external_engram", False):
            self.input_names["engram_embeddings"] = "engram_embeddings"
            self.input_types["engram_embeddings"] = self.io_dtype
            self.input_shapes["engram_embeddings"] = self.make_hidden_state_shape(last_dim=self.ple_embed_dim)

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
        self.fixed_indexer_cache = not self.use_paged_attention and self.ep == "cuda"
        self.output_names["present.indexer"] = qsa_outputs
        self.output_types["present.indexer"] = self.io_dtype
        if self.use_paged_attention:
            self.indexer_state_capacity = (
                self.context_length + self.indexer_compress_ratio - 1
            ) // self.indexer_compress_ratio
            indexer_state_shape = ["batch_size", self.indexer_state_capacity, self.indexer_head_dim]
            indexer_buffer_shape = [
                "batch_size",
                2 * self.indexer_compress_ratio - 1,
                self.indexer_head_dim,
            ]
            indexer_lengths_shape = ["batch_size", 2]
            self.input_shapes["past.indexer"] = indexer_state_shape
            self.output_shapes["present.indexer"] = indexer_state_shape
            self.input_names["past.indexer_kv_buffer"] = {
                layer_id: f"past.{layer_id}.indexer_kv_buffer" for layer_id in qsa_layers
            }
            self.input_types["past.indexer_kv_buffer"] = self.io_dtype
            self.input_shapes["past.indexer_kv_buffer"] = indexer_buffer_shape
            self.output_names["present.indexer_kv_buffer"] = {
                layer_id: f"present.{layer_id}.indexer_kv_buffer" for layer_id in qsa_layers
            }
            self.output_types["present.indexer_kv_buffer"] = self.io_dtype
            self.output_shapes["present.indexer_kv_buffer"] = indexer_buffer_shape
            self.input_names["past.indexer_state_lengths"] = {
                layer_id: f"past.{layer_id}.indexer_state_lengths" for layer_id in qsa_layers
            }
            self.input_types["past.indexer_state_lengths"] = ir.DataType.INT32
            self.input_shapes["past.indexer_state_lengths"] = indexer_lengths_shape
            self.output_names["present.indexer_state_lengths"] = {
                layer_id: f"present.{layer_id}.indexer_state_lengths" for layer_id in qsa_layers
            }
            self.output_types["present.indexer_state_lengths"] = ir.DataType.INT32
            self.output_shapes["present.indexer_state_lengths"] = indexer_lengths_shape
            self.input_shapes["attention_metadata"] = [5]
        else:
            self.input_shapes["past.indexer"] = ["batch_size", "past_sequence_length", self.indexer_head_dim]
            self.output_shapes["present.indexer"] = ["batch_size", "total_sequence_length", self.indexer_head_dim]
        if self.fixed_indexer_cache:
            self.input_names["past_sequence_length"] = "past_sequence_length"
            self.input_types["past_sequence_length"] = ir.DataType.INT32
            self.input_shapes["past_sequence_length"] = [1]

        ple_token_state = {layer_id: f"past.{layer_id}.ple_tokens" for layer_id in self.ple_layer_ids}
        ple_conv_state = {layer_id: f"past.{layer_id}.ple_conv" for layer_id in self.ple_layer_ids}
        present_ple_tokens = {layer_id: f"present.{layer_id}.ple_tokens" for layer_id in self.ple_layer_ids}
        present_ple_conv = {layer_id: f"present.{layer_id}.ple_conv" for layer_id in self.ple_layer_ids}
        self.input_names["past.ple_tokens"] = ple_token_state
        self.input_types["past.ple_tokens"] = ir.DataType.INT64
        self.input_shapes["past.ple_tokens"] = ["batch_size", self.ngram_size - 1]
        self.input_names["past.ple_conv"] = ple_conv_state
        self.input_types["past.ple_conv"] = self.io_dtype
        ple_conv_state_length = self.ple_conv_dilation * (self.ple_conv_kernel_size - 1)
        self.input_shapes["past.ple_conv"] = (
            ["batch_size", self.hc_hidden_size, ple_conv_state_length]
            if self.use_paged_attention
            else [
                *self.context_length_attrs["state_window_dims"],
                "batch_size",
                ple_conv_state_length,
                self.hc_hidden_size,
            ]
        )
        self.output_names["present.ple_tokens"] = present_ple_tokens
        self.output_types["present.ple_tokens"] = ir.DataType.INT64
        self.output_shapes["present.ple_tokens"] = ["batch_size", self.ngram_size - 1]
        self.output_names["present.ple_conv"] = present_ple_conv
        self.output_types["present.ple_conv"] = self.io_dtype
        self.output_shapes["present.ple_conv"] = self.input_shapes["past.ple_conv"]

        state_update_capacity = int(extra_options.get("state_update_capacity", 0)) if self.use_paged_attention else 0
        if state_update_capacity and (self.ple_layer_ids or qsa_layers):
            self.context_length_attrs["state_update_capacity"] = state_update_capacity
            self.input_names["state_update.capture_count"] = "state_update_capture_count"
            self.input_types["state_update.capture_count"] = ir.DataType.INT32
            self.input_shapes["state_update.capture_count"] = ["batch_size"]
            self.input_names["state_update.active"] = "state_update_active"
            self.input_types["state_update.active"] = ir.DataType.INT32
            self.input_shapes["state_update.active"] = [1]
            self.output_names["state_update.ple_tokens"] = {
                layer_id: f"state_update.{layer_id}.ple_tokens" for layer_id in self.ple_layer_ids
            }
            self.output_types["state_update.ple_tokens"] = ir.DataType.INT64
            self.output_shapes["state_update.ple_tokens"] = [
                "batch_size",
                state_update_capacity,
                self.ngram_size - 1,
            ]
            self.output_names["state_update.ple_conv_value"] = {
                layer_id: f"state_update.{layer_id}.ple_conv_value" for layer_id in self.ple_layer_ids
            }
            self.output_types["state_update.ple_conv_value"] = self.io_dtype
            self.output_shapes["state_update.ple_conv_value"] = [
                "batch_size",
                state_update_capacity,
                self.hc_hidden_size,
            ]
            self.output_names["state_update.indexer"] = {
                layer_id: f"state_update.{layer_id}.indexer" for layer_id in qsa_layers
            }
            self.output_types["state_update.indexer"] = self.io_dtype
            self.output_shapes["state_update.indexer"] = [
                "batch_size",
                state_update_capacity,
                self.indexer_head_dim,
            ]

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
        decoder["inputs"]["past_indexer_kv_buffer_names"] = "past.%d.indexer_kv_buffer"
        decoder["inputs"]["past_indexer_state_lengths_names"] = "past.%d.indexer_state_lengths"
        if getattr(self, "fixed_indexer_cache", False):
            decoder["inputs"]["past_sequence_length"] = self.input_names["past_sequence_length"]
        decoder["outputs"]["present_ple_token_names"] = "present.%d.ple_tokens"
        decoder["outputs"]["present_ple_conv_names"] = "present.%d.ple_conv"
        decoder["outputs"]["present_indexer_names"] = "present.%d.indexer_key"
        decoder["outputs"]["present_indexer_kv_buffer_names"] = "present.%d.indexer_kv_buffer"
        decoder["outputs"]["present_indexer_state_lengths_names"] = "present.%d.indexer_state_lengths"
        if getattr(self, "context_length_attrs", {}).get("state_window", 0):
            decoder["outputs"]["state_update_indexer_value_names"] = (
                "state_update.%d.indexer_value"
            )
            decoder["outputs"]["state_update_indexer_row_names"] = (
                "state_update.%d.indexer_row"
            )
        decoder["ple_token_pad_id"] = self.ple_token_pad_id
        if self.ep != "cpu" and not getattr(self, "external_engram", False):
            session_options = decoder["session_options"]
            session_options["session.layer_assignment_settings"] = (
                f"cpu(={self.CPU_EMBEDDING_ANNOTATION})"
            )

    def make_decoder_state_groups(self, inputs, outputs):
        state_groups = super().make_decoder_state_groups(inputs, outputs)
        if not self.use_paged_attention:
            return state_groups

        qsa_layers = [
            layer_id
            for layer_id, layer_type in enumerate(self.layer_types)
            if layer_type == "qwen_sparse_attention"
        ]
        if qsa_layers:
            paged_group = next((group for group in state_groups if group["kind"] == "paged_kv"), None)
            if paged_group is None:
                state_groups.insert(0, self.make_paged_key_value_state_group(qsa_layers))
            else:
                paged_group["layer_ids"] = sorted(set(paged_group["layer_ids"] + qsa_layers))

        if self.ple_layer_ids:
            group = {
                "kind": "fixed_ple",
                "layer_ids": sorted(self.ple_layer_ids),
            }
            if self.context_length_attrs["state_update_capacity"]:
                group["state_update"] = {"capacity": self.context_length_attrs["state_update_capacity"]}
            state_groups.append(group)

        indexer_bindings = {
            "past.indexer",
            "past.indexer_kv_buffer",
            "past.indexer_state_lengths",
        }
        if indexer_bindings.issubset(self.input_names):
            if qsa_layers:
                group = {
                    "kind": "fixed_indexer",
                    "layer_ids": qsa_layers,
                }
                if self.context_length_attrs["state_update_capacity"]:
                    group["state_update"] = {
                        "capacity": self.context_length_attrs["state_update_capacity"],
                        "compress_ratio": self.indexer_compress_ratio,
                    }
                state_groups.append(group)
        return state_groups

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

    def make_hyper_connection_post_mix(self, name, streams, block_output, post_mix, token_shape, output_name=None):
        output = f"{name}/output_0" if output_name is None else output_name
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
        if combine:
            down_name, inject_name = self.make_hyper_connection_down_inject_proj(
                basename, hyper_connection, normalized, token_shape
            )
        else:
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

    def make_hyper_connection_down_inject_proj(self, basename, hyper_connection, root_input, token_shape):
        down = hyper_connection.input_mix_weight_down
        inject = hyper_connection.block_inject_weight
        if any(
            hasattr(projection, "qweight") or getattr(projection, "quant_type", "none") != "none"
            for projection in (down, inject)
        ):
            raise ValueError("Input-mix-down and block-inject must use dense weights before graph quantization.")

        class PackedDownInject:
            weight = torch.cat([down.weight, inject.weight], dim=0)

        packed_name = f"{basename}/input_mix_down_block_inject/MatMul"
        packed_matmul = self.make_matmul(PackedDownInject(), packed_name, root_input)
        down_name = f"{basename}/input_mix_weight_down/MatMul"
        inject_name = f"{basename}/block_inject_weight/MatMul"
        down_size = down.out_features
        inject_size = inject.out_features
        self.make_split(
            f"{basename}/input_mix_down_block_inject/Split",
            [f"{packed_matmul}/output_0", f"/model/constants/INT64/[{down_size}, {inject_size}]"],
            [f"{down_name}/output_0", f"{inject_name}/output_0"],
            [self.io_dtype] * 2,
            [
                [*token_shape, down_size],
                [*token_shape, inject_size],
            ],
            axis=-1,
        )
        return down_name, inject_name

    def make_hyper_connection_injection(self, layer_id, block_output, hyper_input, injection_weights, location, output_name=None):
        basename = f"/model/layers.{layer_id}/{location}_hyper_connection/injection"
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        return self.make_hyper_connection_post_mix(
            basename, hyper_input, block_output, injection_weights, token_shape, output_name=output_name
        )

    def make_ple(self, layer_id, ple, root_input):
        basename = f"/model/layers.{layer_id}/ple"
        embedding = ple.ple_embedding
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        state_update_capacity = (
            getattr(self, "context_length_attrs", {}).get("state_update_capacity", 0)
            if self.use_paged_attention
            else 0
        )
        flatten_dims = [-1, self.ple_embed_dim] if self.use_paged_attention else [0, 0, self.ple_embed_dim]
        if not getattr(self, "external_engram", False) or self.use_paged_attention:
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
            ngram_inputs = [self.input_names["input_ids"], multipliers, vocab_sizes]
            if self.use_paged_attention:
                ngram_inputs.append(self.input_names["cumulative_sequence_lengths"])
            ngram_inputs.extend(
                [self.input_names["past.ple_tokens"][layer_id], offsets, eos]
            )
            if self.use_paged_attention and state_update_capacity:
                ngram_inputs.extend(["", "", self.input_names["state_update.capture_count"]])
            ngram_outputs = [
                ngram_ids,
                self.output_names["present.ple_tokens"][layer_id],
            ]
            if self.use_paged_attention and state_update_capacity:
                ngram_outputs.extend(["", self.output_names["state_update.ple_tokens"][layer_id]])
            self.make_node(
                ngram_op_type,
                inputs=ngram_inputs,
                outputs=ngram_outputs,
                name=ngram_name,
                domain="com.microsoft",
                max_ngram_size=self.ngram_size,
                n_head_per_ngram=self.heads_per_ngram,
                pad_id=embedding.eos_token_id,
                reset_on_eos=1,
                **(
                    {"state_update_capacity": state_update_capacity}
                    if self.use_paged_attention and state_update_capacity
                    else {}
                ),
            )
            ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
            self.make_value(
                ngram_ids,
                ir.DataType.INT64,
                ["num_tokens", ngram_heads]
                if self.use_paged_attention
                else ["batch_size", "sequence_length", ngram_heads],
            )
            if getattr(self, "external_engram", False):
                self.make_value(
                    ngram_outputs[1], ir.DataType.INT64, ["batch_size", self.ngram_size - 1]
                )
        if getattr(self, "external_engram", False):
            engram_embeddings = self.input_names["engram_embeddings"]
            if not self.use_paged_attention:
                self.make_node(
                    "Identity",
                    inputs=[self.input_names["past.ple_tokens"][layer_id]],
                    outputs=[self.output_names["present.ple_tokens"][layer_id]],
                    name=f"{basename}/token_state/Identity",
                )
        else:
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
            self.external_data_files[table_name] = "engram.onnx.data"
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
            flatten_dims = [-1, self.ple_embed_dim] if self.use_paged_attention else [0, 0, self.ple_embed_dim]
            self.make_reshape(
                flatten_name,
                [f"{gather_name}/output_0", f"/model/constants/INT64/{flatten_dims}"],
                self.io_dtype,
                [*token_shape, self.ple_embed_dim],
            )
            engram_embeddings = f"{flatten_name}/output_0"

        key_scale = f"model.layers.{layer_id}.ple.key_norm_scale"
        query_scale = f"model.layers.{layer_id}.ple.query_norm_scale"
        conv_scale = f"model.layers.{layer_id}.ple.conv_norm_scale"
        self.make_initializer((ple.norm_key.weight + 1).reshape(self.hc_count, self.hidden_size), key_scale, to=self.io_dtype)
        self.make_initializer((ple.norm_query.weight + 1).reshape(self.hc_count, self.hidden_size), query_scale, to=self.io_dtype)
        self.make_initializer((ple.norm_conv.weight + 1).reshape(self.hc_count, self.hidden_size), conv_scale, to=self.io_dtype)
        key_matmul = self.make_matmul(ple.key_proj, f"{basename}/key_proj/MatMul", engram_embeddings)
        value_matmul = self.make_matmul(ple.value_proj, f"{basename}/value_proj/MatMul", engram_embeddings)
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
            [gated_value, f"/model/constants/INT64/{[*flatten_dims[:-1], self.hc_hidden_size]}"],
            self.io_dtype,
            ple_shape,
        )
        self.make_reshape(
            gated_value_normed_flat,
            [gated_value_normed, f"/model/constants/INT64/{[*flatten_dims[:-1], self.hc_hidden_size]}"],
            self.io_dtype,
            ple_shape,
        )

        conv_weight = f"model.layers.{layer_id}.ple.conv1d.weight"
        self.make_initializer(ple.conv1d.weight, conv_weight, to=self.io_dtype)
        conv_op_type = "VarlenCausalConvWithState" if self.use_paged_attention else "CausalConvWithState"
        conv_name = f"{basename}/{conv_op_type}"
        conv_output = f"{conv_name}/output_0"
        if self.use_paged_attention:
            self.make_varlen_causal_conv_with_state(
                conv_name,
                root_input=f"{gated_value_normed_flat}/output_0",
                weight=conv_weight,
                cumulative_sequence_length=self.input_names["cumulative_sequence_lengths"],
                bias="",
                past_conv_state=self.input_names["past.ple_conv"][layer_id],
                present_conv_state=self.output_names["present.ple_conv"][layer_id],
                output_shape=ple_shape,
                present_conv_shape=[
                    "batch_size",
                    self.hc_hidden_size,
                    self.ple_conv_dilation * (self.ple_conv_kernel_size - 1),
                ],
                dilation=self.ple_conv_dilation,
                **(
                    {
                        "state_update_capacity": state_update_capacity,
                        "state_update_capture_count": self.input_names["state_update.capture_count"],
                        "state_update_active": self.input_names["state_update.active"],
                        "state_update_value": self.output_names["state_update.ple_conv_value"][layer_id],
                        "state_update_value_shape": self.output_shapes["state_update.ple_conv_value"],
                    }
                    if state_update_capacity
                    else {}
                ),
            )
        else:
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
                state_window=getattr(self, "context_length_attrs", {}).get("state_window", 0),
            )
            self.make_value(conv_output, self.io_dtype, ple_shape)
        add_name = f"{basename}/Add"
        self.make_add(add_name, [f"{gated_value_flat}/output_0", conv_output], self.io_dtype, ple_shape)
        return f"{add_name}/output_0"

    def make_qwen_sparse_attention(self, layer_id, attention, root_input):
        self.make_attention_input_proj(
            layer_id,
            attention,
            root_input,
            indexer_proj=attention.indexer.index_qk_proj,
        )
        q_norm_weight, k_norm_weight = self.get_qk_norm_weight_names(layer_id)
        self.make_initializer(attention.q_norm.weight + 1, q_norm_weight, to=self.io_dtype)
        self.make_initializer(attention.k_norm.weight + 1, k_norm_weight, to=self.io_dtype)
        cos_cache, sin_cache = self.make_rotary_embedding_caches()
        past_k, past_v, present_k, present_v = self.make_key_value_cache_names(layer_id)
        capacity = self.indexer_budget + self.indexer_compress_ratio - 1
        index_qk_path = self.attention_attrs.pop("indexer_qk_path", None)
        if index_qk_path is None:
            index_matmul = self.make_matmul(
                attention.indexer.index_qk_proj,
                f"/model/layers.{layer_id}/attn/indexer/index_qk_proj/MatMul",
                root_input,
            )
            index_qk_path = f"{index_matmul}/output_0"
        index_q_scale = f"model.layers.{layer_id}.attn.indexer.q_norm.weight"
        index_k_scale = f"model.layers.{layer_id}.attn.indexer.k_norm.weight"
        self.make_initializer(attention.indexer.q_layernorm.weight + 1, index_q_scale, to=self.io_dtype)
        self.make_initializer(attention.indexer.k_layernorm.weight + 1, index_k_scale, to=self.io_dtype)

        if self.use_paged_attention:
            if self.indexer_kv_heads != 1:
                raise ValueError("PackedSparseAttentionIndexer requires indexer_kv_heads=1 for QSA.")
            indexer_name = f"/model/layers.{layer_id}/attn/PackedSparseAttentionIndexer"
            index_query = f"{indexer_name}/query"
            index_key = f"{indexer_name}/key"
            query_size = self.indexer_num_heads * self.indexer_head_dim
            key_size = self.indexer_head_dim
            self.make_split(
                f"{indexer_name}/Split",
                [index_qk_path, f"/model/constants/INT64/[{query_size}, {key_size}]"],
                [index_query, index_key],
                [self.io_dtype] * 2,
                [["num_tokens", query_size], ["num_tokens", key_size]],
                axis=-1,
            )
            selected_indices = f"{indexer_name}/output_0"
            selected_counts = f"{indexer_name}/output_1"
            self.make_node(
                "PackedSparseAttentionIndexer",
                inputs=[
                    index_query,
                    index_key,
                    index_q_scale,
                    index_k_scale,
                    cos_cache,
                    sin_cache,
                    self.input_names["cumulative_sequence_lengths"],
                    self.input_names["past_sequence_lengths"],
                    "",
                    "",
                    "",
                    "",
                    self.input_names["past.indexer"][layer_id],
                    self.input_names["past.indexer_kv_buffer"][layer_id],
                    "",
                    self.input_names["past.indexer_state_lengths"][layer_id],
                    *(
                        [
                            self.input_names["state_update.capture_count"],
                            self.input_names["state_update.active"],
                        ]
                        if getattr(self, "context_length_attrs", {}).get("state_update_capacity", 0)
                        else []
                    ),
                ],
                outputs=[
                    selected_indices,
                    selected_counts,
                    self.output_names["present.indexer"][layer_id],
                    self.output_names["present.indexer_kv_buffer"][layer_id],
                    "",
                    self.output_names["present.indexer_state_lengths"][layer_id],
                    *(
                        [self.output_names["state_update.indexer"][layer_id]]
                        if getattr(self, "context_length_attrs", {}).get("state_update_capacity", 0)
                        else []
                    ),
                ],
                name=indexer_name,
                domain="com.microsoft",
                policy_mode="qsa",
                compress_ratio=self.indexer_compress_ratio,
                state_capacity=self.indexer_state_capacity,
                token_budget=self.indexer_budget,
                epsilon=self.layernorm_attrs["epsilon"],
                scale=self.indexer_head_dim**-0.5,
                **(
                    {"state_update_capacity": self.context_length_attrs["state_update_capacity"]}
                    if getattr(self, "context_length_attrs", {}).get("state_update_capacity", 0)
                    else {}
                ),
            )
            self.make_value(selected_indices, ir.DataType.INT32, ["num_tokens", capacity])
            self.make_value(selected_counts, ir.DataType.INT32, ["num_tokens"])
        else:
            indexer_name = f"/model/layers.{layer_id}/attn/SparseAttentionIndexer"
            selected_indices = f"{indexer_name}/output_0"
            if self.fixed_indexer_cache:
                indexer_inputs = [
                    index_qk_path,
                    "",
                    index_q_scale,
                    index_k_scale,
                    cos_cache,
                    sin_cache,
                    "",
                    self.input_names["past.indexer"][layer_id],
                    "",
                    "",
                    "",
                    "",
                    self.input_names["past_sequence_length"],
                ]
            else:
                indexer_inputs = [
                    index_qk_path,
                    "",
                    index_q_scale,
                    index_k_scale,
                    cos_cache,
                    sin_cache,
                    self.input_names["attention_mask"],
                    self.input_names["past.indexer"][layer_id],
                ]
            capture_indexer_updates = (
                getattr(self, "context_length_attrs", {}).get("state_window", 0)
                > 0
            )
            self.make_node(
                "SparseAttentionIndexer",
                inputs=indexer_inputs,
                outputs=(
                    [
                        selected_indices,
                        self.output_names["present.indexer"][layer_id],
                        f"state_update.{layer_id}.indexer_value",
                        f"state_update.{layer_id}.indexer_row",
                    ]
                    if capture_indexer_updates
                    else [selected_indices, self.output_names["present.indexer"][layer_id]]
                ),
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
            if capture_indexer_updates:
                indexer_update_value = self.make_value(
                    f"state_update.{layer_id}.indexer_value",
                    self.io_dtype,
                    ["batch_size", "sequence_length", self.indexer_head_dim],
                )
                indexer_update_row = self.make_value(
                    f"state_update.{layer_id}.indexer_row",
                    ir.DataType.INT32,
                    ["batch_size", "sequence_length"],
                )
                self.model.graph.outputs.extend(
                    [indexer_update_value, indexer_update_row]
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
        hidden_states_output = (
            self.output_names["hidden_states"]
            if layer_id == self.num_layers - 1 and self.emit_pre_final_hidden_states
            and (self.include_hidden_states or self.exclude_lm_head)
            else None
        )
        hyper_states = self.make_hyper_connection_injection(
            layer_id, moe_output, residual, injection, "mlp", output_name=hidden_states_output
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
            if (self.include_hidden_states or self.exclude_lm_head) and not self.emit_pre_final_hidden_states:
                self.make_node(
                    "Identity",
                    inputs=[final_output],
                    outputs=[self.output_names["hidden_states"]],
                    name="/model/final_hidden_states/Identity",
                )
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



class Qwen4ExpEngramModel(_Qwen4ExpGraphModel):
    def __init__(self, ple, config, io_dtype, scale_dtype=None):
        super().__init__(io_dtype, "engram.onnx", "qwen4_exp_engram")
        self.graph.opset_imports["com.microsoft"] = 1
        embedding = ple.ple_embedding
        ngram_size = config.ngram_size
        heads_per_ngram = config.heads_per_ngram
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        ple_embed_dim = config.ple_embed_dim
        head_dim = ple_embed_dim // ngram_heads
        ple_layer_id = config.ple_layer_ids[0] - 1

        input_ids = self.make_value("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        past_tokens = self.make_value(
            f"past.{ple_layer_id}.ple_tokens", ir.DataType.INT64, ["batch_size", ngram_size - 1]
        )
        engram_embeddings = self.make_value(
            "engram_embeddings", self.io_dtype, ["batch_size", "sequence_length", ple_embed_dim]
        )
        present_tokens = self.make_value(
            f"present.{ple_layer_id}.ple_tokens", ir.DataType.INT64, ["batch_size", ngram_size - 1]
        )
        self.graph.inputs.extend([input_ids, past_tokens])
        self.graph.outputs.extend([engram_embeddings, present_tokens])

        multipliers = "model.ple.layer_multipliers"
        vocab_sizes = "model.ple.head_vocab_sizes"
        offsets = "model.ple.head_offsets"
        eos = "model.ple.eos_token_id"
        self.make_initializer(embedding.layer_multipliers, multipliers)
        self.make_initializer(embedding.ngram_heads_vocab_sizes, vocab_sizes)
        self.make_initializer(embedding.ngram_heads_offsets, offsets)
        self.make_initializer(torch.tensor(embedding.eos_token_id, dtype=torch.int64), eos)

        ngram_ids = "/model/ple/NGramHashMapping/output_0"
        self.make_node(
            "NGramHashMapping",
            inputs=[input_ids.name, multipliers, vocab_sizes, past_tokens.name, offsets, eos],
            outputs=[ngram_ids, present_tokens.name],
            name="/model/ple/NGramHashMapping",
            domain="com.microsoft",
            max_ngram_size=ngram_size,
            n_head_per_ngram=heads_per_ngram,
            pad_id=embedding.eos_token_id,
            reset_on_eos=1,
        )
        self.make_value(
            ngram_ids,
            ir.DataType.INT64,
            ["batch_size", "sequence_length", ngram_heads],
        )

        table_name = "model.ple.ngram_embedding.weight"
        quantized_weight, weight_scale = Qwen4ExpTextModel.prepare_engram_embedding(
            embedding.ngram_embedding
        )
        self.make_initializer(quantized_weight, table_name)
        self.make_initializer(
            weight_scale.reshape(1, 1),
            "model.ple.ngram_embedding.weight_scale",
            to=scale_dtype,
        )
        scale_dtype = self.graph.initializers["model.ple.ngram_embedding.weight_scale"].dtype
        gathered = "/model/ple/ngram_embedding/GatherBlockQuantized/output_0"
        self.make_node(
            "GatherBlockQuantized",
            inputs=[table_name, ngram_ids, "model.ple.ngram_embedding.weight_scale"],
            outputs=[gathered],
            name="/model/ple/ngram_embedding/GatherBlockQuantized",
            domain="com.microsoft",
            metadata_props={"layer_ann": Qwen4ExpTextModel.CPU_EMBEDDING_ANNOTATION},
            gather_axis=0,
            quantize_axis=1,
            block_size=0,
        )
        self.make_value(
            gathered,
            scale_dtype,
            ["batch_size", "sequence_length", ngram_heads, head_dim],
        )
        if scale_dtype != self.io_dtype:
            cast_name = "/model/ple/ngram_embedding/Cast"
            self.make_cast(
                cast_name, gathered, self.io_dtype,
                ["batch_size", "sequence_length", ngram_heads, head_dim],
            )
            gathered = f"{cast_name}/output_0"
        self.make_reshape(
            "/model/ple/ngram_embedding/Reshape",
            [gathered, f"/model/constants/INT64/[0, 0, {ple_embed_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", ple_embed_dim],
            output_name=engram_embeddings.name,
        )

    def save_model(self, output_dir):
        ir.save(
            self.model,
            os.path.join(output_dir, self.filename),
            external_data="engram.onnx.data",
            size_threshold_bytes=0,
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
        self.text_only = self.extra_options.get("text_only", False)
        decoder_options = self.make_mtp_init(config, self.extra_options)
        decoder_options["external_engram"] = True
        self.decoder = Qwen4ExpTextModel(
            copy.deepcopy(config), io_dtype, onnx_dtype, ep, cache_dir, decoder_options
        )
        self.decoder.model_type = "qwen4_exp_text" if self.text_only else "qwen3_5"
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
        self.model_type = self.decoder.model_type

    def make_mtp_init(self, config, extra_options):
        decoder_options = super().make_mtp_init(config, extra_options)
        num_mtp_layers = getattr(config.text_config, "mtp_num_hidden_layers", 0) or 0
        self.mtp_attrs["build"] = num_mtp_layers > 0 and not extra_options.get("exclude_mtp", False)
        self.mtp_attrs["shared_initializer_names"] = {"model.embed_tokens.weight"}
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
        if getattr(self.decoder, "quant_type", None) in {"modelopt", "compressed-tensors", "fp8"} and "mtp_quant_config" not in extra_options:
            mtp_quant_config = copy.deepcopy(self.decoder.quant_config)
            mtp_quant_config.weights.type = "none"
            mtp_quant_config.moe.type = "none"
            self.mtp_attrs["extra_options"]["_quant_config"] = mtp_quant_config
            self.mtp_attrs["onnx_dtype"] = io_dtype
        self.resolve_mtp_model_config(extra_options)
        mtp_options = self.mtp_attrs["extra_options"]
        for option in (
            "use_paged_attention", "paged_block_size", "enable_webgpu_graph",
            "gpu_utilization_factor", "max_batch_size", "state_update_capacity",
            "max_draft_tokens", "indexshare_mtp",
        ):
            if option in extra_options:
                mtp_options[option] = copy.deepcopy(extra_options[option])
        mtp_options["text_only"] = True
        mtp_options["filename"] = "mtp.onnx"
        mtp_options.pop("include_hidden_states", None)
        mtp_options.pop("exclude_lm_head", None)
        mtp_options.pop("external_engram", None)
        self.mtp = Qwen4ExpMTPTextModel(
            copy.deepcopy(config),
            self.mtp_attrs["io_dtype"],
            self.mtp_attrs["onnx_dtype"],
            ep,
            cache_dir,
            mtp_options,
        )

    def make_model(self, input_path):
        if self.decoder.quant_type == "fp8" and not os.path.isdir(input_path):
            input_path = snapshot_download(
                repo_id=self.decoder.model_name_or_path,
                cache_dir=self.decoder.cache_dir,
                token=self.decoder.hf_token,
                allow_patterns=["*.json", "*.safetensors"],
            )
        self.input_path = input_path
        self.decoder.make_model(input_path)
        if self.mtp is not None:
            print("Building Qwen4-Exp MTP (multi-token prediction) head -> mtp.onnx")
            self.mtp.make_model(input_path)

    def save_model(self, output_dir):
        table_name = "model.ple.ngram_embedding.weight"
        if self.input_path is None:
            raise RuntimeError("make_model must be called before save_model.")
        weights = self.decoder.load_weights(self.input_path)
        language_model = weights.model.language_model
        if len(self.decoder.ple_layer_ids) != 1:
            raise ValueError(
                f"Qwen4-Exp Engram export requires exactly one PLE layer, got {sorted(self.decoder.ple_layer_ids)}."
            )
        ple_layer_id = next(iter(self.decoder.ple_layer_ids))
        engram_model = Qwen4ExpEngramModel(
            language_model.layers[ple_layer_id].ple,
            self.config.text_config,
            self.decoder.io_dtype,
            scale_dtype=ir.DataType.FLOAT,
        )
        engram_model.save_model(output_dir)
        table = ir.load(os.path.join(output_dir, engram_model.filename)).graph.initializers[table_name].const_value
        for component in (self.decoder, self.mtp):
            if component is not None and table_name in getattr(component, "external_data_files", {}):
                component.external_data_tensors = {table_name: table}

        self.decoder.save_model(output_dir)
        if self.mtp is not None:
            self.mtp.save_model(output_dir)
            self.mtp_attrs["shared_initializers"] = self.share_initializers(
                output_dir, self.decoder.filename, self.mtp.filename
            )
            if self.extra_options.get("indexshare_mtp", self.mtp.use_paged_attention):
                self.mtp_attrs["index_share"] = self.export_indexshare_graphs(
                    output_dir, self.mtp.filename, int(self.extra_options.get("indexshare_max_draft_tokens", 7))
                )
        if self.text_only:
            return
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
        visual = weights.load_visual() if hasattr(weights, "load_visual") else weights.model.visual
        vision_model = Qwen4ExpVisionModel(vision_config, visual, self.decoder.io_dtype)
        vision_model.save_model(output_dir)
        del weights

    def configure_paged_sessions(self, genai_config):
        if self.decoder.ep not in {"cuda", "webgpu"} or not getattr(self.decoder, "use_paged_attention", False):
            return
        model_config = genai_config["model"]
        decoder_options = model_config["decoder"].get("session_options", {})
        ep = self.decoder.ep
        inherited_provider_options = next(
            (provider[ep] for provider in decoder_options.get("provider_options", []) if ep in provider),
            {},
        )
        for section in ("decoder", "mtp"):
            if section not in model_config:
                continue
            options = model_config[section].setdefault("session_options", {})
            providers = options.setdefault("provider_options", [])
            provider_options = next((provider[ep] for provider in providers if ep in provider), None)
            if provider_options is None:
                provider_options = {}
                providers.append({ep: provider_options})
            provider_options.update(copy.deepcopy(inherited_provider_options))
            options["session.use_device_allocator_for_initializers"] = "1"
            if ep != "cuda":
                continue
            options.update({
                "ep.cuda.fpa_intb_gemm": "1",
                "ep.cuda.qmoe_skip_nvfp4_gemv_profiling": "1",
                "ep.cuda.sparse_paged_attention_grouped": "1",
                "ep.cuda.sparse_paged_attention_grouped_decode": "0",
                "ep.cuda.sparse_paged_attention_grouped_decode_splits": "0",
                "ep.cuda.sparse_paged_attention_grouped_tile_size": "8",
                "ep.cuda.sparse_paged_attention_grouped_vectorized": "1",
                "ep.cuda.sparse_paged_attention_warp_reduction": "0",
                "session.use_device_allocator_for_initializers": "1",
            })

    def make_genai_config(self, config, extra_kwargs, out_dir):
        self.decoder.make_genai_config(config.text_config, extra_kwargs, out_dir)
        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as config_file:
            genai_config = json.load(config_file)

        model_config = genai_config["model"]
        if (
            self.decoder.ep == "cuda"
            and self.decoder.moe_attrs.get("quant_type") == "nvfp4"
            and self.decoder.moe_attrs.get("weights_prepacked", -1) != 1
        ):
            model_config["decoder"].setdefault("session_options", {})["session.disable_prepacking"] = "1"
        decoder_inputs = model_config["decoder"]["inputs"]
        decoder_inputs["engram_embeddings"] = "engram_embeddings"
        if not self.text_only:
            model_config["type"] = "qwen3_5"
            model_config["image_token_id"] = config.image_token_id
            model_config["vision_start_token_id"] = config.vision_start_token_id
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
        ple_layer_id = config.text_config.ple_layer_ids[0] - 1
        engram_session_options = {
            "provider_options": [{"cuda" if self.decoder.ep == "cuda" else "cpu": {}}],
        }
        if self.decoder.ep == "cuda":
            engram_session_options["session.layer_assignment_settings"] = (
                f"cpu(={Qwen4ExpTextModel.CPU_EMBEDDING_ANNOTATION})"
            )
        model_config["engram"] = {
            "filename": "engram.onnx",
            "cache_capacity": 4096,
            "session_options": engram_session_options,
            "inputs": {
                "input_ids": "input_ids",
                "past_ple_token_names": f"past.{ple_layer_id}.ple_tokens",
            },
            "outputs": {
                "embeddings": "engram_embeddings",
                "present_ple_token_names": f"present.{ple_layer_id}.ple_tokens",
            },
        }
        self.configure_paged_sessions(genai_config)
        with open(config_path, "w") as config_file:
            config_file.write(serialize_genai_config(genai_config))
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
                "past_indexer_names": "past.%d.indexer_key",
                "past_sequence_length": "past_sequence_length",
            },
            "outputs": {
                "logits": "logits",
                "hidden_states": "hidden_states_out",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
                "present_indexer_names": "present.%d.indexer_key",
            },
            "session_options": {
                name: value
                for name, value in genai_config["model"]["decoder"].get("session_options", {}).items()
                if name in ("ep.cuda.fpa_intb_gemm", "session.use_device_allocator_for_initializers")
            },
        }
        self.add_shared_initializers_to_genai_config(genai_config)
        if "index_share" in self.mtp_attrs:
            metadata = self.mtp_attrs["index_share"]
            mtp_config = genai_config["model"]["mtp"]
            mtp_config["base_capacity"] = metadata["base_capacity"]
            mtp_config["inputs"]["past_indices"] = "indexshare.past_indices"
            mtp_config["inputs"]["past_counts"] = "indexshare.past_counts"
            mtp_config["outputs"]["present_indices"] = metadata["indices_output"]
            mtp_config["outputs"]["present_counts"] = metadata["counts_output"]
            genai_config.setdefault("speculative", {}).setdefault("max_draft_tokens", metadata["max_draft_tokens"])
        self.configure_paged_sessions(genai_config)
        with open(config_path, "w") as config_file:
            config_file.write(serialize_genai_config(genai_config))

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.decoder.save_processing(model_name_or_path, extra_kwargs, out_dir)
        preprocessor_path = (
            os.path.join(model_name_or_path, "preprocessor_config.json")
            if os.path.isdir(model_name_or_path)
            else hf_hub_download(
                model_name_or_path,
                "preprocessor_config.json",
                cache_dir=extra_kwargs.get("cache_dir"),
                token=self.hf_token,
            )
        )
        with open(preprocessor_path) as preprocessor_file:
            image_processor = json.load(preprocessor_file)
        processor_config = {
            "processor": {
                "name": "qwen4_exp_image_processor",
                "transforms": [
                    {
                        "operation": {
                            "name": "decode_image",
                            "type": "DecodeImage",
                            "attrs": {"color_space": "RGB"},
                        }
                    },
                    {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                    {
                        "operation": {
                            "name": "resize",
                            "type": "Resize",
                            "attrs": {
                                "width": 540,
                                "height": 360,
                                "smart_resize": 1,
                                "min_pixels": image_processor["size"]["shortest_edge"],
                                "max_pixels": image_processor["size"]["longest_edge"],
                                "patch_size": image_processor["patch_size"],
                                "merge_size": image_processor["merge_size"],
                            },
                        }
                    },
                    {
                        "operation": {
                            "name": "rescale",
                            "type": "Rescale",
                            "attrs": {"rescale_factor": 1 / 255},
                        }
                    },
                    {
                        "operation": {
                            "name": "normalize",
                            "type": "Normalize",
                            "attrs": {
                                "mean": image_processor["image_mean"],
                                "std": image_processor["image_std"],
                                "qwen3_vl": 1,
                            },
                        }
                    },
                    {
                        "operation": {
                            "name": "patch_image",
                            "type": "PatchImage",
                            "attrs": {
                                "patch_size": image_processor["patch_size"],
                                "temporal_patch_size": image_processor["temporal_patch_size"],
                                "merge_size": image_processor["merge_size"],
                            },
                        }
                    },
                ],
            }
        }
        with open(os.path.join(out_dir, "processor_config.json"), "w") as processor_config_file:
            json.dump(processor_config, processor_config_file, indent=4)



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
