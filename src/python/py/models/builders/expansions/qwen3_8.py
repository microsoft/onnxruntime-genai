# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy

import onnx_ir as ir
import torch

from .engram import Qwen38Engram
from .standard import Standard


class Qwen38(Qwen38Engram):
    """Standard ONNX subgraphs retained around Qwen3.8 contrib-op replacements."""

    make_branchwise_rms_norm_expansion = Standard.make_branchwise_rms_norm_expansion
    make_scaled_silu_expansion = Standard.make_scaled_silu_expansion
    make_hyper_connection_pre_mix_expansion = Standard.make_hyper_connection_pre_mix_expansion
    make_hyper_connection_post_mix_expansion = Standard.make_hyper_connection_post_mix_expansion
    make_selected_counts = Standard.make_selected_counts

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
        return None

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
        packable = self.is_qkv_projection_packable(layer_id, attention)
        if indexer_proj is not None and self.attention_attrs["use_packed_matmul"] and packable:
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
            return None

        if not self.attention_attrs["use_packed_matmul"] or not packable:
            return super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)

        q_size = self.q_size
        self.q_size = 2 * q_size
        try:
            super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)
        finally:
            self.q_size = q_size
        return None

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
        return f"{qkv_name}/output_0", f"{z_name}/output_0"

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
        return f"{b_name}/output_0", f"{a_name}/output_0"

    def make_gated_rms_norm(self, name, root_input, scale, gate, shape, epsilon=1e-5):
        output = f"{name}/output_0"
        self.make_node(
            "GatedRMSNorm",
            inputs=[root_input, scale, gate],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            epsilon=epsilon,
            activation=self.attention_attrs["output_gate_type"],
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
            num_branches=self.hyper_connection_attrs["num_branches"],
        )
        self.make_value(output, self.io_dtype, [*token_shape, self.hyper_connection_attrs["num_branches"] * hidden_size])
        return output

    def make_scaled_silu(self, name, root_input, shape):
        output = f"{name}/output_0"
        self.make_node(
            "ScaledSiLU",
            inputs=[root_input],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            alpha=1.0 / self.hyper_connection_attrs["num_branches"],
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
            num_branches=self.hyper_connection_attrs["num_branches"],
            reduction_scale=1.0 / self.hyper_connection_attrs["num_branches"],
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
            num_branches=self.hyper_connection_attrs["num_branches"],
        )
        self.make_value(output, self.io_dtype, [*token_shape, self.hyper_connection_attrs["hidden_size"]])
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
        mix_sigmoid_shape = [*token_shape, self.hyper_connection_attrs["hidden_size"]]
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
        inject_shape = [*token_shape, self.hyper_connection_attrs["num_branches"]]
        self.make_div(
            inject_div,
            [f"{inject_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.hyper_connection_attrs['num_branches']}"],
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

    def make_hyper_connection_injection(
        self, layer_id, block_output, hyper_input, injection_weights, location, output_name=None
    ):
        basename = f"/model/layers.{layer_id}/{location}_hyper_connection/injection"
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        return self.make_hyper_connection_post_mix(
            basename, hyper_input, block_output, injection_weights, token_shape, output_name=output_name
        )

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
        multimodal = getattr(self, "sparse_rotary_attrs", {}).get("multimodal", False)
        if multimodal:
            self.make_sparse_mrope_projections(layer_id, attention, cos_cache, sin_cache)
            indexer_cos, indexer_sin = self.make_indexer_mrope_caches(cos_cache, sin_cache)
        else:
            indexer_cos, indexer_sin = self.make_indexer_rope_caches(cos_cache, sin_cache)
        past_k, past_v, present_k, present_v = self.make_key_value_cache_names(layer_id)
        capacity = self.sparse_indexer_attrs["token_budget"] + self.sparse_indexer_attrs["compress_ratio"] - 1
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
            if self.sparse_indexer_attrs["kv_heads"] != 1:
                raise ValueError("PackedSparseAttentionIndexer requires indexer_kv_heads=1 for QSA.")
            indexer_name = f"/model/layers.{layer_id}/attn/PackedSparseAttentionIndexer"
            index_query = f"{indexer_name}/query"
            index_key = f"{indexer_name}/key"
            query_size = self.sparse_indexer_attrs["num_heads"] * self.sparse_indexer_attrs["head_dim"]
            key_size = self.sparse_indexer_attrs["head_dim"]
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
                    indexer_cos,
                    indexer_sin,
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
                compress_ratio=self.sparse_indexer_attrs["compress_ratio"],
                state_capacity=self.sparse_indexer_attrs["state_capacity"],
                token_budget=self.sparse_indexer_attrs["token_budget"],
                epsilon=self.layernorm_attrs["epsilon"],
                scale=self.sparse_indexer_attrs["head_dim"]**-0.5,
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
            if self.sparse_indexer_attrs["fixed_cache"]:
                indexer_inputs = [
                    index_qk_path,
                    "",
                    index_q_scale,
                    index_k_scale,
                    indexer_cos,
                    indexer_sin,
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
                    indexer_cos,
                    indexer_sin,
                    self.input_names["attention_mask"],
                    self.input_names["past.indexer"][layer_id],
                ]
            capture_indexer_updates = getattr(self, "context_length_attrs", {}).get("state_window", 0) > 0
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
                compress_ratio=self.sparse_indexer_attrs["compress_ratio"],
                token_budget=self.sparse_indexer_attrs["token_budget"],
                epsilon=self.layernorm_attrs["epsilon"],
                scale=self.sparse_indexer_attrs["head_dim"]**-0.5,
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
                    ["batch_size", "sequence_length", self.sparse_indexer_attrs["head_dim"]],
                )
                indexer_update_row = self.make_value(
                    f"state_update.{layer_id}.indexer_row",
                    ir.DataType.INT32,
                    ["batch_size", "sequence_length"],
                )
                self.model.graph.outputs.extend([indexer_update_value, indexer_update_row])
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
                "" if multimodal else cos_cache,
                "" if multimodal else sin_cache,
                "",
                "" if multimodal else q_norm_weight,
                "" if multimodal else k_norm_weight,
                "",
                "",
                self.input_names["attention_metadata"],
            ]
        else:
            attention_position_ids = ""
            if not multimodal:
                position_name = f"{name}/position_ids/Gather"
                self.make_gather(
                    position_name,
                    [self.input_names["position_ids"], "/model/constants/INT64/0"],
                    ir.DataType.INT64,
                    ["batch_size", "sequence_length"],
                    axis=0,
                )
                attention_position_ids = f"{position_name}/output_0"
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
                "" if multimodal else cos_cache,
                "" if multimodal else sin_cache,
                attention_position_ids,
                "" if multimodal else q_norm_weight,
                "" if multimodal else k_norm_weight,
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
            do_rotary=int(not multimodal),
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

    def make_sparse_mrope_projections(self, layer_id, attention, cos_cache, sin_cache):
        """Normalize and rotate current Q/K using all three coordinate streams."""
        if not self.attention_attrs["k_path"]:
            name = f"/model/layers.{layer_id}/attn/sparse_qkv/Split"
            sizes = [self.q_size, self.kv_size, self.kv_size]
            outputs = [f"{name}/output_{index}" for index in range(3)]
            self.make_split(
                name,
                [self.attention_attrs["q_path"], f"/model/constants/INT64/{sizes}"],
                outputs,
                [self.io_dtype] * 3,
                [self.make_hidden_state_shape(last_dim=size) for size in sizes],
                axis=-1,
            )
            self.attention_attrs.update(zip(("q_path", "k_path", "v_path"), outputs, strict=True))
        self.make_qk_norm(layer_id, attention)
        for path, heads in (("q_path", self.num_attn_heads), ("k_path", self.num_kv_heads)):
            name = f"/model/layers.{layer_id}/attn/{path}/MRotaryEmbedding"
            output = f"{name}/output_0"
            self.make_mrotary_embedding(
                name,
                self.attention_attrs[path],
                output,
                num_heads=heads,
                dtype=self.io_dtype,
                position_ids=self.input_names["position_ids"],
                cos_cache_name=cos_cache,
                sin_cache_name=sin_cache,
            )
            self.attention_attrs[path] = output

    def make_indexer_rope_caches(self, cos_cache, sin_cache):
        """QSA uses a full split-half rotary table, unlike attention's half-width cache."""
        outputs = tuple(f"/model/indexer_rotary/{kind}/Concat/output_0" for kind in ("cos", "sin"))
        if outputs[0] not in self.values:
            length, width = self.values[cos_cache].shape
            if 2 * width > self.sparse_indexer_attrs["head_dim"]:
                raise ValueError("Qwen3.8 indexer head must cover the full rotary dimension.")
            for kind, cache in (("cos", cos_cache), ("sin", sin_cache)):
                self.make_concat(
                    f"/model/indexer_rotary/{kind}/Concat",
                    [cache, cache],
                    self.io_dtype,
                    [length, 2 * width],
                    axis=-1,
                )
        return outputs

    def make_indexer_mrope_caches(self, cos_cache, sin_cache):
        """Recompose request-specific tables indexed by chronological key position."""
        basename = "/model/indexer_rotary"
        outputs = tuple(f"{basename}/{kind}/Concat/output_0" for kind in ("cos", "sin"))
        if outputs[0] in self.values:
            return outputs
        rotary_dim = self.rope_attrs["rotary_embedding_dim"] or self.head_size
        if rotary_dim > self.sparse_indexer_attrs["head_dim"]:
            raise ValueError("Qwen3.8 indexer head must cover the full rotary dimension.")
        owners = Standard.get_mrope_owners(self, rotary_dim)
        shape = ["batch_size", "rotary_sequence_length", rotary_dim // 2]
        positions = []
        for dim in range(3):
            name = f"{basename}/position_{dim}/Gather"
            self.make_gather(
                name,
                [self.input_names["rotary_position_ids"], f"/model/constants/INT64/{dim}"],
                ir.DataType.INT64,
                shape[:-1],
                axis=0,
            )
            positions.append(f"{name}/output_0")
        for kind, cache in (("cos", cos_cache), ("sin", sin_cache)):
            streams = []
            for dim, position in enumerate(positions):
                name = f"{basename}/{kind}/stream_{dim}/Gather"
                self.make_gather(name, [cache, position], self.io_dtype, shape, axis=0)
                streams.append(f"{name}/output_0")
            selected = streams[0]
            for dim in (1, 2):
                mask = f"{basename}/mask_{dim}"
                if mask not in self.values:
                    self.make_initializer(torch.from_numpy(owners == dim), mask)
                name = f"{basename}/{kind}/select_{dim}/Where"
                self.make_where(name, [mask, streams[dim], selected], self.io_dtype, shape)
                selected = f"{name}/output_0"
            self.make_concat(
                f"{basename}/{kind}/Concat",
                [selected, selected],
                self.io_dtype,
                ["batch_size", "rotary_sequence_length", rotary_dim],
                axis=-1,
            )
        return outputs
