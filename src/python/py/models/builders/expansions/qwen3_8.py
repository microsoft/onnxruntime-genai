# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import onnx_ir as ir
import torch

from .engram import Qwen38Engram


class Qwen38(Qwen38Engram):
    """Standard ONNX subgraphs retained around Qwen3.8 contrib-op replacements."""

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


class Qwen38Graph:
    """Shared graph primitives for Qwen3.8 auxiliary models."""

    def make_cast(self, name, root_input, dtype, shape):
        super().make_cast(name, root_input, dtype, shape)
        return f"{name}/output_0"


class Qwen38Vision:
    """Qwen3.8 vision position, rotary, and attention graph expansions."""

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
        frame_mask = self.make_frame_attention_mask(layer_id)
        masked = f"/visual/blocks/{layer_id}/attn/scores/Where"
        mask_value = f"/visual/blocks/{layer_id}/attn/mask_value"
        self.make_initializer(np.asarray(-np.inf, dtype=np.float32), mask_value, to=self.io_dtype)
        self.make_where(
            masked,
            [frame_mask, scaled, mask_value],
            self.io_dtype,
            [self.num_heads, "num_patches", "num_patches"],
        )
        self.make_softmax(
            f"/visual/blocks/{layer_id}/attn/Softmax",
            f"{masked}/output_0",
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
