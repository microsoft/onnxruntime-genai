# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .mistral import MistralModel


class GraniteModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.embed_attrs["scale"] = config.embedding_multiplier
        self.attention_attrs["scale"] = config.attention_multiplier
        self.lm_head_attrs["scale"] = 1 / config.logits_scaling
        self.residual_scale = config.residual_multiplier

    def make_layer(self, layer_id, layer):
        # Each Granite decoder layer is defined as:
        # input_layernorm --> attention --> Mul --> output_layernorm --> MLP --> Mul
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        residual_mul_1_name = f"/model/layers.{layer_id}/residual_mul/Mul_1"
        residual_mul_1_inputs = [
            self.layernorm_attrs["skip_input"],
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.residual_scale}",
        ]
        self.make_mul(
            residual_mul_1_name,
            residual_mul_1_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )
        # Assign output 0 of previous output node as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{residual_mul_1_name}/output_0"

        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        residual_mul_2_name = f"/model/layers.{layer_id}/residual_mul/Mul_2"
        residual_mul_2_inputs = [
            self.layernorm_attrs["skip_input"],
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.residual_scale}",
        ]
        self.make_mul(
            residual_mul_2_name,
            residual_mul_2_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )
        # Assign output 0 of previous output node as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{residual_mul_2_name}/output_0"

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True


class GraniteMoeHybridModel(GraniteModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # GraniteMoeHybrid uses shared_intermediate_size for the always-on dense MLP
        # (when num_local_experts=0). Override so the base model reads the correct width.
        config.intermediate_size = config.shared_intermediate_size
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_layer(self, layer_id, layer):
        # GraniteMoeHybrid stores the MLP as `shared_mlp` with a fused input_linear
        # (gate+up concatenated, shape [2*intermediate_size, hidden_size]) and
        # output_linear (down projection). Adapt to the standard layout expected by
        # make_mlp_unpacked_regular (gate_up_proj) and make_mlp_proj (down_proj).
        mlp = layer.shared_mlp
        mlp.gate_up_proj = mlp.input_linear
        mlp.down_proj = mlp.output_linear
        layer.mlp = mlp
        super().make_layer(layer_id, layer)
