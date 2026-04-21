# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
from .llama import LlamaModel


class NemotronHModel(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # NemotronH uses `mlp_hidden_act` instead of `hidden_act`
        if not hasattr(config, "hidden_act"):
            config.hidden_act = getattr(config, "mlp_hidden_act", "relu2")
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # NemotronH attention does not use rotary position embeddings (NoPE)
        self.attention_attrs["rope"] = False
        self.attention_attrs["use_rope_in_attn"] = False
        # NemotronH uses RMSNorm (simplified, no bias)
        self.layernorm_attrs["simple"] = True
        self.layernorm_attrs["epsilon"] = config.layer_norm_epsilon

    def is_layer(self, module):
        return module.__class__.__name__ == "NemotronHBlock"

    def has_final_norm(self, module, orig_model):
        return (
            hasattr(orig_model, "model") and hasattr(orig_model.model, "norm_f") and module == orig_model.model.norm_f
        )

    def make_layer(self, layer_id, layer):
        # Each NemotronH decoder block is defined as:
        # pre_norm --> mixer (attention / mamba / moe) --> residual add
        #
        # Only attention blocks are supported for ONNX export.
        if layer.block_type != "attention":
            raise NotImplementedError(
                f"NemotronH block type '{layer.block_type}' is not supported for ONNX export. "
                "Only 'attention' layers are currently supported."
            )

        self.make_layernorm(
            layer_id, layer.norm, skip=not self.layernorm_attrs["first_layernorm"], simple=True, location="input"
        )
        self.make_attention(layer_id, layer.mixer, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """`nemotronh` is not supported as an architecture, let's replace with `llama`."""
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        genai_config["model"]["type"] = "llama"

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)


class NemotronModel(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        self.layernorm_attrs["add_offset"] = 1
        if hasattr(config, "norm_eps"):
            self.layernorm_attrs["epsilon"] = config.norm_eps

    def make_mlp_proj(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #          root_input
        #              |
        #         UpProjMatMul
        #              |
        #           ActFunc
        #              |
        #         DownProjMatMul

        up_basename = f"/model/layers.{layer_id}/mlp/up_proj/MatMul"
        up_name = self.make_matmul(mlp.up_proj, up_basename, root_input)

        act_fn_name = self.make_activation(layer_id, root_input=f"{up_name}/output_0")

        # Make output MatMul node
        down_basename = f"/model/layers.{layer_id}/mlp/down_proj/MatMul"
        down_name = self.make_matmul(mlp.down_proj, down_basename, f"{act_fn_name}/output_0")

        # Assign output 0 of previous MatMul as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{down_name}/output_0"
