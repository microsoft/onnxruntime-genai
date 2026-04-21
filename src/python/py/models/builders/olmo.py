# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os

import torch

from .base import Model


class OLMoModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # OLMo v1 uses standard LayerNorm (with mean subtraction), not RMSNorm.
        self.layernorm_attrs["simple"] = False
        # OlmoConfig has no rms_norm_eps or layer_norm_eps attribute, so the base class
        # defaults to 1e-6; override here to match OlmoLayerNorm's hard-coded eps=1e-5.
        self.layernorm_attrs["epsilon"] = 1e-5

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        # OlmoLayerNorm has no learnable weight or bias; inject identity values so
        # the base class make_layernorm_op can create the initializer tensors it needs.
        layernorm.weight = torch.ones(self.hidden_size)
        layernorm.bias = torch.zeros(self.hidden_size)
        super().make_layernorm(layer_id, layernorm, skip, simple, location)


class OLMo2Model(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()

    def make_qk_norm(self, layer_id, attention):
        # OLMo2/3: q_norm and k_norm are applied to the *full* projected output
        # (BxSxD) before the per-head reshape, unlike Gemma3 which normalizes
        # per head.  The weight vectors therefore have shape D and KD rather
        # than head_size, so we apply SimplifiedLayerNormalization directly
        # without the Reshape round-trip used in the base implementation.
        layernorm_kwargs = {"epsilon": self.layernorm_attrs["epsilon"], "axis": -1, "stash_type": 1}

        # Q norm (BxSxD, weight shape D)
        q_ln_name = f"/model/layers.{layer_id}/attn/q_norm/SimplifiedLayerNormalization"
        q_weight_name = f"model.layers.{layer_id}.attn.q_norm.layernorm.weight"
        q_ln_output = f"{q_ln_name}/output_0"
        self.make_initializer(
            attention.q_norm.weight + self.layernorm_attrs["add_offset"], q_weight_name, to=self.io_dtype
        )
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[self.attention_attrs["q_path"], q_weight_name],
            outputs=[q_ln_output],
            name=q_ln_name,
            **layernorm_kwargs,
        )
        self.make_value(
            q_ln_output, self.io_dtype, shape=["batch_size", "sequence_length", self.num_attn_heads * self.head_size]
        )
        self.attention_attrs["q_path"] = q_ln_output

        # K norm (BxSxKD, weight shape KD)
        k_ln_name = f"/model/layers.{layer_id}/attn/k_norm/SimplifiedLayerNormalization"
        k_weight_name = f"model.layers.{layer_id}.attn.k_norm.layernorm.weight"
        k_ln_output = f"{k_ln_name}/output_0"
        self.make_initializer(
            attention.k_norm.weight + self.layernorm_attrs["add_offset"], k_weight_name, to=self.io_dtype
        )
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[self.attention_attrs["k_path"], k_weight_name],
            outputs=[k_ln_output],
            name=k_ln_name,
            **layernorm_kwargs,
        )
        self.make_value(
            k_ln_output, self.io_dtype, shape=["batch_size", "sequence_length", self.num_kv_heads * self.head_size]
        )
        self.attention_attrs["k_path"] = k_ln_output

    def make_layer(self, layer_id, layer):
        # OLMo2/3 uses a post-norm architecture with no input_layernorm:
        #   attn(x) -> post_attn_norm(attn_out) -> x2 = x + normed_attn
        #   mlp(x2) -> post_ff_norm(mlp_out)    -> result = x2 + normed_mlp
        x = self.layernorm_attrs["root_input"]

        # Attention takes raw residual directly (no pre-layernorm)
        self.make_attention(layer_id, layer.self_attn, root_input=x)
        attn_out = self.layernorm_attrs["skip_input"]

        # post_attention_layernorm: simple LayerNorm on the attention output only
        self.layernorm_attrs["root_input"] = attn_out
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        normed_attn = self.layernorm_attrs["output_0"]

        # Explicit residual add: x2 = x + normed_attn
        attn_add_name = f"/model/layers.{layer_id}/attn/residual_add"
        self.make_add(
            attn_add_name,
            [x, normed_attn],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )
        x2 = f"{attn_add_name}/output_0"

        # MLP takes x2 (post-attention residual, unnormalized)
        self.make_mlp(layer_id, layer.mlp, root_input=x2)
        mlp_out = self.layernorm_attrs["skip_input"]

        # post_feedforward_layernorm: simple LayerNorm on the MLP output only
        self.layernorm_attrs["root_input"] = mlp_out
        self.make_layernorm(
            layer_id,
            layer.post_feedforward_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_feedforward",
        )
        normed_mlp = self.layernorm_attrs["output_0"]

        if layer_id < self.num_layers - 1:
            # Intermediate layer: compute result explicitly for the next layer's attention
            mlp_add_name = f"/model/layers.{layer_id}/mlp/residual_add"
            self.make_add(
                mlp_add_name,
                [x2, normed_mlp],
                dtype=self.io_dtype,
                shape=["batch_size", "sequence_length", self.hidden_size],
            )
            result = f"{mlp_add_name}/output_0"
            self.layernorm_attrs["root_input"] = result
            self.layernorm_attrs["skip_input"] = result
        else:
            # Last layer: defer the MLP residual add to the model's final SkipLayerNorm.
            # The final norm will compute SkipLayerNorm(x2, normed_mlp) = normalize(x2 + normed_mlp).
            self.layernorm_attrs["root_input"] = x2
            self.layernorm_attrs["skip_input"] = normed_mlp

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """`olmo2` is not supported as an architecture, let's replace with `olmo`."""
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        genai_config["model"]["type"] = "olmo"

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)


class OLMo3Model(OLMo2Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layer_types = config.layer_types

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # OLMo3 uses per-layer sliding window attention controlled by layer_types.
        # We temporarily override window_size for full-attention layers, then restore it.
        original_window_size = self.window_size
        if self.layer_types[layer_id] == "full_attention":
            self.window_size = -1
        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.window_size = original_window_size

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """`olmo3` is not supported as an architecture, let's replace with `olmo`."""
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        genai_config["model"]["type"] = "olmo"

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)
