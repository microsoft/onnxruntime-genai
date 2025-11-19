# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .base import Model


class ChatGLMModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.rope_attrs["partial_rotary_factor"] = (
            0.5  # Line 755 of modeling_chatglm.py check self.rotary_pos_emb declaration
        )
        self.rope_attrs["num_heads"] = self.num_attn_heads
        self.rope_attrs["rotary_embedding_dim"] = int(self.head_size * self.rope_attrs["partial_rotary_factor"])
        self.rope_attrs["interleaved"] = 1

    def make_mlp(self, layer_id, mlp, root_input):
        if not hasattr(mlp, "down_proj"):
            # Attribute does not exist for original PyTorch model only
            mlp.down_proj = mlp.dense_4h_to_h
        super().make_mlp(layer_id, mlp, root_input)

    def make_layer(self, layer_id, layer):
        layer.self_attn = layer.self_attn if hasattr(layer, "self_attn") else layer.self_attention
        super().make_layer(layer_id, layer)
