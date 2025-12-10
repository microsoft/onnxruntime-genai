# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnx_ir as ir
import torch

from .base import Model


class GPTOSSModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.is_local = lambda layer_id: bool(layer_id % 2 == 0)
        self.attention_attrs["sinks"] = True

        self.moe_attrs["activation_alpha"] = 1.702
        self.moe_attrs["activation_beta"] = 1.0
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["normalize_routing_weights"] = True
        self.moe_attrs["swiglu_fusion"] = 1

    def make_layer(self, layer_id, layer):
        # Each LLM decoder layer is typically defined as:
        # input_layernorm --> attention --> output_layernorm --> MoE
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
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
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        if "final_norm" in location:
            # Set cast for final LayerNorm since it is a special case and not covered in `make_layer`
            self.layernorm_attrs["cast"]["root_input"] = False
        super().make_layernorm(layer_id, layernorm, skip, simple, location)

    def make_rotary_embedding_caches_from_scratch(self):
        inv_freq = self.rope_attrs["theta"] ** (torch.arange(0, self.head_size, 2, dtype=torch.float) / self.head_size)
        inv_freq = self.make_inv_freq_rescaled(inv_freq)

        t = torch.arange(self.rope_attrs["cache_length"], dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos_cache, sin_cache = freqs.cos() * self.rope_attrs["mscale"], freqs.sin() * self.rope_attrs["mscale"]
        return cos_cache, sin_cache

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        original_window_size = self.window_size
        self.window_size = (
            original_window_size if self.is_local(layer_id) else -1
        )  # default is -1 in GroupQueryAttention kernel
        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.window_size = original_window_size

    def make_moe(self, layer_id, mlp, root_input):
        if self.ep in {"cpu", "cuda"}:
            self.make_moe_fused(layer_id, mlp, root_input)
        else:
            self.make_moe_decomposed(layer_id, mlp, root_input)

    def make_moe_decomposed(self, layer_id, mlp, root_input):
        # Make nodes for the MoE subgraph
        #
        #                                              root_input
        #                                                  |
        #          +---------------------------------------+
        #          |                                       |
        #      Unsqueeze                                 MatMul
        #       (dim=2)                                 (Router)
        #          |                                       |
        #       Expand                                    Add
        #          |                                       |
        #      Unsqueeze                                 Cast
        #      (dim=-1)                               (to = FP32)
        #          |                                       |
        #          |                                     TopK
        #          |                                       |
        #          |                                     Cast
        #          |                                 (to = io_dtype)
        #          |                                       |
        #          |           +-------------+-------------+-------------+-------------+
        #          |           |             |             |             |             |
        #          |         Gather        Gather        Gather        Gather       Softmax
        #          |      (MLP1 Weight)  (MLP1 Bias)  (MLP2 Weight)  (MLP2 Bias)   (Expert Weights)
        #          |           |             |             |             |             |
        #          +-----+-----+         Unsqueeze         |         Unsqueeze     Unsqueeze
        #                |               (dim=-1)          |         (dim=-1)      (dim=-1)
        #             MatMul                 |             |             |             |
        #          (gate_up_proj)            |             |             |         Unsqueeze
        #                |                   |             |             |         (dim=-1)
        #                +---------+---------+             |             |             |
        #                          |                       |             |           Cast
        #                         Add                      |             |        (to = FP32)
        #                          |                       |             |             |
        #                    +-----+-----+                 |             |             |
        #                    |           |                 |             |             |
        #                  Slice       Slice               |             |             |
        #                    |           |                 |             |             |
        #                  Clip        Clip                |             |             |
        #                    |  \        |                 |             |             |
        #                   Mul  \       |                 |             |             |
        #                    |    \      |                 |             |             |
        #                 Sigmoid  |     |                 |             |             |
        #                    |     |    Add                |             |             |
        #                    \    /      |                 |             |             |
        #                     \  /       |                 |             |             |
        #                      Mul       |                 |             |             |
        #                       |        |                 |             |             |
        #                       +----+---+                 |             |             |
        #                            |                     |             |             |
        #                           Mul                    |             |             |
        #                            |                     |             |             |
        #                            +----------+----------+             |             |
        #                                       |                        |             |
        #                                    MatMul                      |             |
        #                                  (down_proj)                   |             |
        #                                       |                        |             |
        #                                       +------------+-----------+             |
        #                                                    |                         |
        #                                                   Add                        |
        #                                                    |                         |
        #                                                   Cast                       |
        #                                               (to = FP32)                    |
        #                                                    |                         |
        #                                                    +------------+------------+
        #                                                                 |
        #                                                                Mul
        #                                                                 |
        #                                                             ReduceSum
        #                                                              (dim=2)
        #                                                                 |
        #                                                              Squeeze
        #                                                              (dim=-1)
        #                                                                 |
        #                                                                Cast
        #                                                           (to = io_dtype)
        basename = f"/model/layers.{layer_id}/moe"
        use_cast = self.io_dtype != ir.DataType.FLOAT

        # Make root_input expansion nodes (root_input --> Unsqueeze --> Expand --> Unsqueeze)
        expand_root_input_unsqueeze_1_name = f"{basename}/expand_root_input/Unsqueeze_1"
        expand_root_input_unsqueeze_1_inputs = [root_input, "/model/constants/INT64/[2]"]
        self.make_unsqueeze(
            expand_root_input_unsqueeze_1_name,
            expand_root_input_unsqueeze_1_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", 1, self.hidden_size],
        )
        expand_name = f"{basename}/expand_root_input/Expand"
        expand_inputs = [
            f"{expand_root_input_unsqueeze_1_name}/output_0",
            f"/model/constants/INT64/[1, 1, {self.moe_attrs['top_k']}, 1]",
        ]
        self.make_expand(
            expand_name,
            expand_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.hidden_size],
        )
        expand_root_input_unsqueeze_2_name = f"{basename}/expand_root_input/Unsqueeze_2"
        expand_root_input_unsqueeze_2_inputs = [f"{expand_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_unsqueeze(
            expand_root_input_unsqueeze_2_name,
            expand_root_input_unsqueeze_2_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.hidden_size, 1],
        )

        # Make router nodes
        #                                          +--> Gather --> (MLP1 weight)
        #                                          |
        #                                          +--> Gather --> Unsqueeze --> (MLP1 bias)
        #                                          |
        # root_input --> MatMul --> Add --> TopK --+--> Gather --> (MLP2 weight)
        #                                          |
        #                                          +--> Gather --> Unsqueeze --> (MLP2 bias)
        #                                          |
        #                                          +--> Softmax --> Unsqueeze --> Unsqueeze --> Cast
        #
        router_basename = f"{basename}/router/MatMul"
        router_matmul_name = self.make_matmul(mlp.router, router_basename, root_input)
        router_add_name = f"{basename}/router/Add"
        self.make_add_bias(mlp.router.bias, router_add_name, root_input=f"{router_matmul_name}/output_0")

        if use_cast:
            topk_fp32_name = f"{basename}/topk_fp32/Cast"
            self.make_cast(
                topk_fp32_name,
                f"{router_add_name}/output_0",
                ir.DataType.FLOAT,
                shape=["batch_size", "sequence_length", self.moe_attrs["num_experts"]],
            )
        topk_name = f"{basename}/TopK"
        topk_inputs = [
            f"{topk_fp32_name if use_cast else router_add_name}/output_0",
            f"/model/constants/INT64/[{self.moe_attrs['top_k']}]",
        ]
        topk_outputs = [f"{topk_name}/output_0", f"{topk_name}/output_1"]
        self.make_node(
            "TopK", inputs=topk_inputs, outputs=topk_outputs, name=topk_name, axis=-1, largest=True, sorted=True
        )
        self.make_value(
            topk_outputs[0], ir.DataType.FLOAT, shape=["batch_size", "sequence_length", self.moe_attrs["top_k"]]
        )
        self.make_value(
            topk_outputs[1], ir.DataType.INT64, shape=["batch_size", "sequence_length", self.moe_attrs["top_k"]]
        )
        if use_cast:
            topk_io_name = f"{basename}/topk_io/Cast"
            self.make_cast(
                topk_io_name,
                topk_outputs[0],
                self.io_dtype,
                shape=["batch_size", "sequence_length", self.moe_attrs["top_k"]],
            )

        # Save initializers to use with Gather nodes
        gate_up_proj_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.weight"
        self.make_initializer(mlp.experts.gate_up_proj, gate_up_proj_weight, to=self.io_dtype)
        gate_up_proj_bias = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        self.make_initializer(mlp.experts.gate_up_proj_bias, gate_up_proj_bias, to=self.io_dtype)
        down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.weight"
        self.make_initializer(mlp.experts.down_proj, down_proj_weight, to=self.io_dtype)
        down_proj_bias = f"model.layers.{layer_id}.moe.experts.down_proj.bias"
        self.make_initializer(mlp.experts.down_proj_bias, down_proj_bias, to=self.io_dtype)

        # Make Gather nodes + Unsqueeze nodes for biases
        mlp1_weight_gather_name = f"{basename}/mlp1/weight/Gather"
        mlp1_weight_gather_inputs = [gate_up_proj_weight, f"{topk_name}/output_1"]
        self.make_gather(
            mlp1_weight_gather_name,
            mlp1_weight_gather_inputs,
            dtype=self.io_dtype,
            shape=[
                "batch_size",
                "sequence_length",
                self.moe_attrs["top_k"],
                2 * self.intermediate_size,
                self.hidden_size,
            ],
            axis=0,
        )
        mlp1_bias_gather_name = f"{basename}/mlp1/bias/Gather"
        mlp1_bias_gather_inputs = [gate_up_proj_bias, f"{topk_name}/output_1"]
        self.make_gather(
            mlp1_bias_gather_name,
            mlp1_bias_gather_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], 2 * self.intermediate_size],
            axis=0,
        )
        mlp1_bias_unsqueeze_name = f"{basename}/mlp1/bias/Unsqueeze"
        mlp1_bias_unsqueeze_inputs = [f"{mlp1_bias_gather_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_unsqueeze(
            mlp1_bias_unsqueeze_name,
            mlp1_bias_unsqueeze_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], 2 * self.intermediate_size, 1],
        )
        mlp2_weight_gather_name = f"{basename}/mlp2/weight/Gather"
        mlp2_weight_gather_inputs = [down_proj_weight, f"{topk_name}/output_1"]
        self.make_gather(
            mlp2_weight_gather_name,
            mlp2_weight_gather_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.hidden_size, self.intermediate_size],
            axis=0,
        )
        mlp2_bias_gather_name = f"{basename}/mlp2/bias/Gather"
        mlp2_bias_gather_inputs = [down_proj_bias, f"{topk_name}/output_1"]
        self.make_gather(
            mlp2_bias_gather_name,
            mlp2_bias_gather_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.hidden_size],
            axis=0,
        )
        mlp2_bias_unsqueeze_name = f"{basename}/mlp2/bias/Unsqueeze"
        mlp2_bias_unsqueeze_inputs = [f"{mlp2_bias_gather_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_unsqueeze(
            mlp2_bias_unsqueeze_name,
            mlp2_bias_unsqueeze_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.hidden_size, 1],
        )

        # Make expert_weights path (Softmax --> Unsqueeze --> Unsqueeze --> Cast)
        softmax_name = f"{basename}/expert_weights/Softmax"
        self.make_softmax(
            softmax_name,
            f"{topk_io_name if use_cast else topk_name}/output_0",
            self.io_dtype,
            ["batch_size", "sequence_length", "num_experts_per_token"],
        )
        expert_weights_unsqueeze_1_name = f"{basename}/expert_weights/Unsqueeze_1"
        expert_weights_unsqueeze_1_inputs = [f"{softmax_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_unsqueeze(
            expert_weights_unsqueeze_1_name,
            expert_weights_unsqueeze_1_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", "num_experts_per_token", 1],
        )
        expert_weights_unsqueeze_2_name = f"{basename}/expert_weights/Unsqueeze_2"
        expert_weights_unsqueeze_2_inputs = [
            f"{expert_weights_unsqueeze_1_name}/output_0",
            "/model/constants/INT64/[-1]",
        ]
        self.make_unsqueeze(
            expert_weights_unsqueeze_2_name,
            expert_weights_unsqueeze_2_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", "num_experts_per_token", 1, 1],
        )
        if use_cast:
            expert_weights_cast_name = f"{basename}/expert_weights/Cast"
            self.make_cast(
                expert_weights_cast_name,
                f"{expert_weights_unsqueeze_2_name}/output_0",
                ir.DataType.FLOAT,
                shape=["batch_size", "sequence_length", "num_experts_per_token", 1, 1],
            )

        # Make Gate/Up proj nodes (MatMul --> Add)
        gate_up_proj_weight_name = f"{basename}/gate_up_proj/MatMul"
        gate_up_proj_weight_output = f"{gate_up_proj_weight_name}/output_0"
        self.make_node(
            "MatMul",
            inputs=[f"{mlp1_weight_gather_name}/output_0", f"{expand_root_input_unsqueeze_2_name}/output_0"],
            outputs=[gate_up_proj_weight_output],
            name=gate_up_proj_weight_name,
        )
        self.make_value(
            gate_up_proj_weight_output,
            self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], 2 * self.intermediate_size, 1],
        )
        gate_up_proj_bias_name = f"{basename}/gate_up_proj/Add"
        self.make_add(
            gate_up_proj_bias_name,
            [gate_up_proj_weight_output, f"{mlp1_bias_unsqueeze_name}/output_0"],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], 2 * self.intermediate_size, 1],
        )

        # Make activation nodes
        #
        #         +--> Slice --> Clamp --> Mul --> Sigmoid --+
        #         |      |                                   |
        #         |      |                                  Mul --+
        # Add ----+      |                                   |    |
        #         |      +-----------------------------------+    +--> Mul
        #         |                                               |
        #         +---> Slice --> Clamp --> Add ------------------+
        glu_slice_name = f"{basename}/act_fn/Slice_1"
        glu_slice_inputs = [
            f"{gate_up_proj_bias_name}/output_0",
            "/model/constants/INT64/[0]",
            f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]",
            "/model/constants/INT64/[3]",
            "/model/constants/INT64/[2]",
        ]
        self.make_slice(
            glu_slice_name,
            glu_slice_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        glu_clip_name = f"{basename}/act_fn/Clip_1"
        glu_clip_inputs = [
            f"{glu_slice_name}/output_0",
            "",
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.moe_attrs['swiglu_limit']}",
        ]
        self.make_clip(
            glu_clip_name,
            glu_clip_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        linear_slice_name = f"{basename}/act_fn/Slice_2"
        linear_slice_inputs = [
            f"{gate_up_proj_bias_name}/output_0",
            "/model/constants/INT64/[1]",
            f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]",
            "/model/constants/INT64/[3]",
            "/model/constants/INT64/[2]",
        ]
        self.make_slice(
            linear_slice_name,
            linear_slice_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        linear_clip_name = f"{basename}/act_fn/Clip_2"
        linear_clip_inputs = [
            f"{linear_slice_name}/output_0",
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{-self.moe_attrs['swiglu_limit']}",
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.moe_attrs['swiglu_limit']}",
        ]
        self.make_clip(
            linear_clip_name,
            linear_clip_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )

        # Make Mul node after activation
        act_fn_mul_1_name = f"{basename}/act_fn/Mul_1"
        act_fn_mul_1_inputs = [
            f"{glu_clip_name}/output_0",
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/1.703125",
        ]
        self.make_mul(
            act_fn_mul_1_name,
            act_fn_mul_1_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        sigmoid_name = f"{basename}/act_fn/Sigmoid"
        self.make_sigmoid(
            sigmoid_name,
            f"{act_fn_mul_1_name}/output_0",
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        act_fn_mul_2_name = f"{basename}/act_fn/Mul_2"
        act_fn_mul_2_inputs = [f"{glu_clip_name}/output_0", f"{sigmoid_name}/output_0"]
        self.make_mul(
            act_fn_mul_2_name,
            act_fn_mul_2_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        act_fn_add_name = f"{basename}/act_fn/Add"
        self.make_add(
            act_fn_add_name,
            [f"{linear_clip_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/1"],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        act_fn_mul_3_name = f"{basename}/act_fn/Mul_3"
        act_fn_mul_3_inputs = [f"{act_fn_mul_2_name}/output_0", f"{act_fn_add_name}/output_0"]
        self.make_mul(
            act_fn_mul_3_name,
            act_fn_mul_3_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )

        # Make Down proj nodes (MatMul --> Add --> Cast)
        down_proj_weight_name = f"{basename}/down_proj/MatMul"
        down_proj_weight_output = f"{down_proj_weight_name}/output_0"
        self.make_node(
            "MatMul",
            inputs=[f"{mlp2_weight_gather_name}/output_0", f"{act_fn_mul_3_name}/output_0"],
            outputs=[down_proj_weight_output],
            name=down_proj_weight_name,
        )
        self.make_value(
            down_proj_weight_output,
            self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        down_proj_bias_name = f"{basename}/down_proj/Add"
        self.make_add(
            down_proj_bias_name,
            [down_proj_weight_output, f"{mlp2_bias_unsqueeze_name}/output_0"],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        if use_cast:
            down_proj_cast_name = f"{basename}/down_proj/Cast"
            self.make_cast(
                down_proj_cast_name,
                f"{down_proj_bias_name}/output_0",
                ir.DataType.FLOAT,
                shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
            )

        # Make weighted sum nodes
        #
        # Cast (from Down proj) ------->
        #                               \
        #                                Mul --> ReduceSum --> Squeeze --> Cast (created in LayerNorm)
        #                               /
        # Cast (from expert weights) -->
        weighted_sum_mul_name = f"{basename}/weighted_sum/Mul"
        weighted_sum_mul_inputs = [
            f"{down_proj_cast_name if use_cast else down_proj_bias_name}/output_0",
            f"{expert_weights_cast_name if use_cast else expert_weights_unsqueeze_2_name}/output_0",
        ]
        self.make_mul(
            weighted_sum_mul_name,
            weighted_sum_mul_inputs,
            dtype=ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1],
        )
        reduce_sum_name = f"{basename}/weighted_sum/ReduceSum"
        reduce_sum_inputs = [f"{weighted_sum_mul_name}/output_0", "/model/constants/INT64/[2]"]
        self.make_reduce_sum(
            reduce_sum_name,
            reduce_sum_inputs,
            dtype=ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.intermediate_size, 1],
        )
        weighted_sum_squeeze_name = f"{basename}/weighted_sum/Squeeze"
        weighted_sum_squeeze_inputs = [f"{reduce_sum_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_squeeze(
            weighted_sum_squeeze_name,
            weighted_sum_squeeze_inputs,
            dtype=ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.intermediate_size],
        )

        # Assign output 0 of previous MoE as root input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{weighted_sum_squeeze_name}/output_0"

    def make_moe_fused(self, layer_id, mlp, root_input):
        # Make nodes for the fused MoE subgraph
        #
        #               root_input
        #               /        \
        #             MatMul      |
        #            (router)     |
        #               |         |
        #              Add        |
        #            (router)     |
        #               |         |
        #            Reshape      |
        #               |         |
        #               +----+----+
        #                    |
        #                 MoE/QMoE
        basename = f"/model/layers.{layer_id}/moe"
        op_type = self.moe_attrs["op_type"]
        moe_weight_type = f"{'q' if op_type == 'QMoE' else ''}weight"

        # Make router nodes
        router_basename = f"{basename}/router/MatMul"
        router_matmul_name = self.make_matmul(mlp.router, router_basename, root_input)
        router_add_name = f"{basename}/router/Add"
        self.make_add_bias(mlp.router.bias, router_add_name, root_input=f"{router_matmul_name}/output_0")
        router_reshape_name = f"{basename}/router/Reshape"
        router_reshape_inputs = [
            f"{router_add_name}/output_0",
            f"/model/constants/INT64/{[-1, self.moe_attrs['num_experts']]}",
        ]
        self.make_reshape(
            router_reshape_name,
            router_reshape_inputs,
            dtype=self.io_dtype,
            shape=["batch_size * sequence_length", self.moe_attrs["num_experts"]],
        )

        gate_up_proj_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.{moe_weight_type}"
        gate_up_proj_scales = f"model.layers.{layer_id}.moe.experts.gate_up_proj.scales"
        gate_up_proj_bias = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.{moe_weight_type}"
        down_proj_scales = f"model.layers.{layer_id}.moe.experts.down_proj.scales"
        down_proj_bias = f"model.layers.{layer_id}.moe.experts.down_proj.bias"

        # Apply transpose depending on EP/op requirements
        # For quantized QMoE on CUDA, kernels expect scales along the hidden_size axis,
        # so we keep original orientation (last axis = hidden_size) when quantizing.
        # For non-quantized MoE or non-CUDA EPs, transpose to align MatMul layout.
        if op_type == "QMoE" and self.ep == "cuda":
            gate_up_proj_layout = mlp.experts.gate_up_proj
            down_proj_layout = mlp.experts.down_proj
        else:
            gate_up_proj_layout = mlp.experts.gate_up_proj.transpose(-1, -2)
            down_proj_layout = mlp.experts.down_proj.transpose(-1, -2)

        if op_type == "MoE":
            # Save non-quantized MoE weights as initializers
            self.make_initializer(
                gate_up_proj_layout.view(self.moe_attrs["num_experts"], -1, self.hidden_size),
                gate_up_proj_weight,
                to=self.io_dtype,
            )
            self.make_initializer(
                down_proj_layout.view(self.moe_attrs["num_experts"], self.hidden_size, self.intermediate_size),
                down_proj_weight,
                to=self.io_dtype,
            )
        else:
            # Create and save quantized MoE weights as initializers
            gate_up_proj_qweight_list, gate_up_proj_scales_list = [], []
            down_proj_qweight_list, down_proj_scales_list = [], []

            for i in range(self.moe_attrs["num_experts"]):
                qweight1, scales1 = self.make_qmoe_weights(gate_up_proj_layout[i])
                gate_up_proj_qweight_list.append(qweight1)
                gate_up_proj_scales_list.append(scales1)
                qweight2, scales2 = self.make_qmoe_weights(down_proj_layout[i])
                down_proj_qweight_list.append(qweight2)
                down_proj_scales_list.append(scales2)

            gate_up_proj_qweight_tensor = torch.stack(gate_up_proj_qweight_list, dim=0).to(torch.uint8)
            gate_up_proj_scales_tensor = torch.stack(gate_up_proj_scales_list, dim=0)
            down_proj_qweight_tensor = torch.stack(down_proj_qweight_list, dim=0).to(torch.uint8)
            down_proj_scales_tensor = torch.stack(down_proj_scales_list, dim=0)

            # qweight tensors always use the same shape regardless of quantization method
            pack_size = 8 // self.moe_attrs["expert_weight_bits"]
            self.make_initializer(
                gate_up_proj_qweight_tensor.view(self.moe_attrs["num_experts"], -1, self.hidden_size // pack_size),
                gate_up_proj_weight,
            )
            self.make_initializer(
                down_proj_qweight_tensor.view(
                    self.moe_attrs["num_experts"], self.hidden_size, self.intermediate_size // pack_size
                ),
                down_proj_weight,
            )

            # scales tensors have different shapes depending on quantization method
            self.make_initializer(gate_up_proj_scales_tensor, gate_up_proj_scales, to=self.io_dtype)
            self.make_initializer(down_proj_scales_tensor, down_proj_scales, to=self.io_dtype)

        # Save MoE biases as initializers
        self.make_initializer(mlp.experts.gate_up_proj_bias, gate_up_proj_bias, to=self.io_dtype)
        self.make_initializer(mlp.experts.down_proj_bias, down_proj_bias, to=self.io_dtype)

        moe_name = f"{basename}/{op_type}"
        self.make_moe_op(
            moe_name,
            root_input=root_input,
            router_probs=f"{router_reshape_name}/output_0",
            weight1=gate_up_proj_weight,
            scales1=gate_up_proj_scales,
            bias1=gate_up_proj_bias,
            weight2=down_proj_weight,
            scales2=down_proj_scales,
            bias2=down_proj_bias,
        )

        # Assign output 0 of previous MoE as root input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{moe_name}/output_0"
