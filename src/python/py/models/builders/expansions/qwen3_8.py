# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import types

import onnx_ir as ir
import torch


class Qwen38:
    """Standard ONNX subgraphs retained around Qwen3.8 contrib-op replacements."""

    merge_shared_input_projections = False
    merged_shared_expert_layers = frozenset()

    def make_branchwise_rms_norm_expansion(self, name, root_input, norm, hidden_size):
        """Apply RMS normalization independently to each hyper-connection branch."""
        token_shape = ["num_tokens"] if self.use_paged_attention else ["batch_size", "sequence_length"]
        grouped_shape = [*token_shape, self.hc_count, hidden_size]
        grouped_dims = (
            [-1, self.hc_count, hidden_size] if self.use_paged_attention else [0, 0, self.hc_count, hidden_size]
        )
        reshape_name = f"{name}/Reshape"
        self.make_reshape(
            reshape_name,
            [root_input, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        norm_scale_name = f"{name[1:].replace('/', '.')}.norm_scale"
        self.make_initializer(norm.weight.new_ones(hidden_size), norm_scale_name, to=self.io_dtype)
        normalized_name = f"{name}/SimplifiedLayerNormalization"
        normalized = f"{normalized_name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[f"{reshape_name}/output_0", norm_scale_name],
            outputs=[normalized],
            name=normalized_name,
            axis=-1,
            epsilon=self.layernorm_attrs["epsilon"],
            stash_type=1,
        )
        self.make_value(normalized, self.io_dtype, grouped_shape)
        flatten_name = f"{name}/Flatten"
        flat_shape = [*token_shape, self.hc_count * hidden_size]
        flat_dims = (
            [-1, self.hc_count * hidden_size] if self.use_paged_attention else [0, 0, self.hc_count * hidden_size]
        )
        self.make_reshape(
            flatten_name,
            [normalized, f"/model/constants/INT64/{flat_dims}"],
            self.io_dtype,
            flat_shape,
        )
        scale_name = f"{name[1:].replace('/', '.')}.weight"
        self.make_initializer(norm.weight + 1.0, scale_name, to=self.io_dtype)
        scale_mul_name = f"{name}/Scale"
        self.make_mul(scale_mul_name, [f"{flatten_name}/output_0", scale_name], self.io_dtype, flat_shape)
        return f"{scale_mul_name}/output_0"

    def make_scaled_silu_expansion(self, name, root_input, shape, alpha):
        """Expand ScaledSiLU into standard ONNX operators."""
        divide_name = f"{name}/Div"
        self.make_div(
            divide_name,
            [root_input, f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{1 / alpha:g}"],
            self.io_dtype,
            shape,
        )
        sigmoid_name = f"{name}/Sigmoid"
        self.make_sigmoid(sigmoid_name, f"{divide_name}/output_0", self.io_dtype, shape)
        self.make_mul(
            name,
            [f"{divide_name}/output_0", f"{sigmoid_name}/output_0"],
            self.io_dtype,
            shape,
        )
        return f"{name}/output_0"

    def make_hyper_connection_pre_mix_expansion(self, name, streams, pre_mix, token_shape, hidden_size):
        """Expand HyperConnectionPreMix into reshape, multiply, and reduction nodes."""
        grouped_shape = [*token_shape, self.hc_count, hidden_size]
        grouped_dims = (
            [-1, self.hc_count, hidden_size] if self.use_paged_attention else [0, 0, self.hc_count, hidden_size]
        )
        stream_reshape = f"{name}/streams/Reshape"
        self.make_reshape(
            stream_reshape,
            [streams, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        gate_reshape = f"{name}/pre_mix/Reshape"
        self.make_reshape(
            gate_reshape,
            [pre_mix, f"/model/constants/INT64/{grouped_dims}"],
            self.io_dtype,
            grouped_shape,
        )
        weighted = f"{name}/Mul"
        self.make_mul(
            weighted,
            [f"{stream_reshape}/output_0", f"{gate_reshape}/output_0"],
            self.io_dtype,
            grouped_shape,
        )
        self.make_reduce_mean(
            name,
            [f"{weighted}/output_0", "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [*token_shape, hidden_size],
        )
        return f"{name}/output_0"

    def make_hyper_connection_post_mix_expansion(self, name, streams, block_output, post_mix, token_shape, hidden_size):
        """Expand identity-stream HyperConnectionPostMix into standard ONNX operators."""
        output_unsqueeze = f"{name}/output/Unsqueeze"
        self.make_unsqueeze(
            output_unsqueeze,
            [block_output, "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [*token_shape, 1, hidden_size],
        )
        weight_unsqueeze = f"{name}/weight/Unsqueeze"
        self.make_unsqueeze(
            weight_unsqueeze,
            [post_mix, "/model/constants/INT64/[-1]"],
            self.io_dtype,
            [*token_shape, self.hc_count, 1],
        )
        weighted = f"{name}/Mul"
        self.make_mul(
            weighted,
            [f"{output_unsqueeze}/output_0", f"{weight_unsqueeze}/output_0"],
            self.io_dtype,
            [*token_shape, self.hc_count, hidden_size],
        )
        flatten = f"{name}/Reshape"
        flat_dims = (
            [-1, self.hc_count * hidden_size] if self.use_paged_attention else [0, 0, self.hc_count * hidden_size]
        )
        self.make_reshape(
            flatten,
            [f"{weighted}/output_0", f"/model/constants/INT64/{flat_dims}"],
            self.io_dtype,
            [*token_shape, self.hc_count * hidden_size],
        )
        self.make_add(
            name,
            [streams, f"{flatten}/output_0"],
            self.io_dtype,
            [*token_shape, self.hc_count * hidden_size],
        )
        return f"{name}/output_0"

    def make_selected_counts(self, layer_id, selected_indices, capacity, packed):
        """Expand the indexer's -1 padding convention into executor counts."""
        basename = f"/model/layers.{layer_id}/attn/indexer/selected"
        valid = f"{basename}/GreaterOrEqual"
        selected_shape = ["num_tokens", capacity] if packed else ["batch_size", "sequence_length", capacity]
        self.make_greater_or_equal(
            valid,
            [selected_indices, "/model/constants/INT32/0"],
            selected_shape,
        )
        valid_int = f"{basename}/Where"
        self.make_where(
            valid_int,
            [f"{valid}/output_0", "/model/constants/INT32/1", "/model/constants/INT32/0"],
            ir.DataType.INT32,
            selected_shape,
        )
        counts = f"{basename}/ReduceSum"
        count_shape = ["num_tokens"] if packed else ["batch_size", "sequence_length"]
        self.make_reduce_sum(
            counts,
            [f"{valid_int}/output_0", "/model/constants/INT64/[-1]"],
            ir.DataType.INT32,
            count_shape,
        )
        return f"{counts}/output_0"

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
            if self.merge_shared_input_projections:
                basename = f"/model/layers.{layer_id}/attn"
                matmul_names = [
                    f"{basename}/qkv_proj/MatMul",
                    f"{basename}/gate_proj/MatMul",
                    f"{basename}/indexer/index_qk_proj/MatMul",
                ]
                if any(
                    getattr(projection, "bias", None) is not None and torch.count_nonzero(projection.bias) > 0
                    for projection in qkv_projections
                ):
                    raise ValueError("merge_shared_input_projections requires bias-free Q/K/V projections.")
                self.make_merged_projections(
                    f"{basename}/qkv_gate_indexer_proj",
                    root_input,
                    [packed_qkv, gate_proj, indexer_proj],
                    matmul_names,
                )
                self.attention_attrs["q_path"] = f"{matmul_names[0]}/output_0"
                self.attention_attrs["k_path"] = ""
                self.attention_attrs["v_path"] = ""
                self.attention_attrs["gate_path"] = f"{matmul_names[1]}/output_0"
                self.attention_attrs["indexer_qk_path"] = f"{matmul_names[2]}/output_0"
                return

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

    def make_linear_attention_projections(self, layer_id, attention, root_input):
        if not self.merge_shared_input_projections:
            return super().make_linear_attention_projections(layer_id, attention, root_input)

        basename = f"/model/layers.{layer_id}/linear_attn"
        qkv_name, z_name, a_name, b_name = (f"{basename}/{name}_proj/MatMul" for name in ("qkv", "z", "a", "b"))
        self.make_merged_projections(
            f"{basename}/qkv_z_a_b_proj",
            root_input,
            [attention.in_proj_qkv, attention.in_proj_z, attention.in_proj_a, attention.in_proj_b],
            [qkv_name, z_name, a_name, b_name],
        )
        return qkv_name, z_name, b_name, a_name

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
        if not self.merge_shared_input_projections:
            return super().make_moe_router(layer_id, moe, root_input)

        # The router and the shared expert's gate and up projections read the same activation. The
        # shared expert's one-wide output gate is left out: a width that is not a multiple of 8 would
        # make the merged GEMM's output unaligned.
        basename = f"/model/layers.{layer_id}/moe"
        shared_basename = f"/model/layers.{layer_id}/mlp"
        shared_expert = moe.shared_expert
        router_name = f"{basename}/router/MatMul"
        self.make_merged_projections(
            f"{basename}/router_shared_gate_up_proj",
            root_input,
            [moe.gate, shared_expert.gate_proj, shared_expert.up_proj],
            [router_name, f"{shared_basename}/gate_proj/MatMul", f"{shared_basename}/up_proj/MatMul"],
        )
        self.merged_shared_expert_layers.add(layer_id)
        self.make_reshape(
            f"{basename}/router/Reshape",
            [f"{router_name}/output_0", f"/model/constants/INT64/{[-1, self.moe_attrs['num_experts']]}"],
            dtype=self.io_dtype,
            shape=self.make_moe_router_shape(),
        )

    def make_shared_expert(self, layer_id, shared_expert, shared_expert_gate, root_input):
        if layer_id not in self.merged_shared_expert_layers:
            return super().make_shared_expert(layer_id, shared_expert, shared_expert_gate, root_input)

        # Gate and up were produced by the merged router projection; this is the rest of the unmerged
        # fused-gate/up shared expert.
        self.merged_shared_expert_layers.remove(layer_id)
        down_bias = getattr(shared_expert.down_proj, "bias", None)
        if down_bias is not None and torch.count_nonzero(down_bias) > 0:
            raise ValueError("merge_shared_input_projections requires a bias-free shared-expert down projection.")
        basename = f"/model/layers.{layer_id}/mlp"
        intermediate_size = self.intermediate_size
        try:
            self.intermediate_size = self.shared_expert_intermediate_size
            act_fn_name = self.make_activation(layer_id, root_input=f"{basename}/gate_proj/MatMul/output_0")
            mul_name = f"{basename}/Mul"
            self.make_mul(
                mul_name,
                [f"{act_fn_name}/output_0", f"{basename}/up_proj/MatMul/output_0"],
                dtype=self.io_dtype,
                shape=self.make_hidden_state_shape(last_dim=self.intermediate_size),
            )
            down_name = self.make_matmul(shared_expert.down_proj, f"{basename}/down_proj/MatMul", f"{mul_name}/output_0")
        finally:
            self.intermediate_size = intermediate_size

        gate_matmul_name = self.make_matmul(
            shared_expert_gate, f"/model/layers.{layer_id}/shared_expert_gate/MatMul", root_input
        )
        gate_sigmoid_name = f"/model/layers.{layer_id}/shared_expert_gate/Sigmoid"
        self.make_sigmoid(
            gate_sigmoid_name,
            f"{gate_matmul_name}/output_0",
            self.io_dtype,
            shape=self.make_hidden_state_shape(last_dim=1),
        )
        return f"{down_name}/output_0", f"{gate_sigmoid_name}/output_0"

    def make_merged_projections(self, basename, root_input, projections, matmul_names):
        """Emit one MatMul over the row-concatenated weights of ``projections``, then Split it.

        The projections read the same activation, so one GEMM reads it once and launches once. Each Split
        output takes the name that projection's own MatMul output would have had, so consumers are
        unchanged. The results are the same dot products as the separate MatMuls.
        """
        sizes = []
        exclusions = set()
        for matmul_name, projection in zip(matmul_names, projections, strict=True):
            weight = getattr(projection, "weight", None)
            bias = getattr(projection, "bias", None)
            if (
                weight is None
                or not weight.is_floating_point()
                or hasattr(projection, "qweight")
                or hasattr(projection, "base_layer")
                or getattr(projection, "quant_type", "none") != "none"
                or (bias is not None and torch.count_nonzero(bias) > 0)
            ):
                raise ValueError(
                    f"merge_shared_input_projections requires dense, bias-free projections, but '{matmul_name}' is not."
                )
            if weight.shape[0] % 8:
                raise ValueError(
                    f"merge_shared_input_projections requires every merged projection to have a multiple of 8 outputs, "
                    f"but '{matmul_name}' has {weight.shape[0]}."
                )
            sizes.append(weight.shape[0])
            exclusions.add(bool(getattr(projection, "exclude_from_quantization", False)))
        if len(exclusions) != 1:
            raise ValueError(
                "merge_shared_input_projections cannot preserve a quantization exclusion that applies to only some of "
                f"{matmul_names}."
            )

        merged = types.SimpleNamespace(
            weight=torch.cat([projection.weight for projection in projections]),
            exclude_from_quantization=exclusions.pop(),
        )
        merged_name = self.make_matmul(merged, f"{basename}/MatMul", root_input)
        self.make_split(
            f"{basename}/Split",
            [f"{merged_name}/output_0", f"/model/constants/INT64/{sizes}"],
            [f"{matmul_name}/output_0" for matmul_name in matmul_names],
            [self.io_dtype] * len(sizes),
            [self.make_hidden_state_shape(last_dim=size) for size in sizes],
            axis=-1,
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

    def make_hyper_connection_injection(
        self, layer_id, block_output, hyper_input, injection_weights, location, output_name=None
    ):
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
            getattr(self, "context_length_attrs", {}).get("state_update_capacity", 0) if self.use_paged_attention else 0
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
            ngram_inputs.extend([self.input_names["past.ple_tokens"][layer_id], offsets, eos])
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
                self.make_value(ngram_outputs[1], ir.DataType.INT64, ["batch_size", self.ngram_size - 1])
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
        self.make_initializer(
            (ple.norm_key.weight + 1).reshape(self.hc_count, self.hidden_size), key_scale, to=self.io_dtype
        )
        self.make_initializer(
            (ple.norm_query.weight + 1).reshape(self.hc_count, self.hidden_size), query_scale, to=self.io_dtype
        )
        self.make_initializer(
            (ple.norm_conv.weight + 1).reshape(self.hc_count, self.hidden_size), conv_scale, to=self.io_dtype
        )
        key_matmul = self.make_matmul(ple.key_proj, f"{basename}/key_proj/MatMul", engram_embeddings)
        value_matmul = self.make_matmul(ple.value_proj, f"{basename}/value_proj/MatMul", engram_embeddings)
        grouped_shape = [*token_shape, self.hc_count, self.hidden_size]
        key_reshape = f"{basename}/key_proj/Reshape"
        query_reshape = f"{basename}/query/Reshape"
        grouped_dims = (
            [-1, self.hc_count, self.hidden_size]
            if self.use_paged_attention
            else [0, 0, self.hc_count, self.hidden_size]
        )
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


class Qwen38Graph:
    """Shared graph primitives for Qwen3.8 auxiliary models."""

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


class Qwen38Vision:
    """Qwen3.8 vision position, rotary, and attention graph expansions."""

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
        frame_size = self.make_binary("Mul", "/vision/grid/frame_size/Mul", [height, width], ir.DataType.INT64, [])
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


class Qwen38MTP:
    """Qwen3.8 MTP normalization and projection expansions."""

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
        grouped_dims = (
            [-1, self.hc_count, self.hidden_size]
            if self.use_paged_attention
            else [0, 0, self.hc_count, self.hidden_size]
        )
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
