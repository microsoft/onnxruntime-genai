# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .llama import LlamaModel


class SmolLM3Model(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layer_types = config.layer_types
        self.no_rope_layers = config.no_rope_layers

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # SmolLM3 uses per-layer conditional RoPE and Sliding Window Attention.
        # So, we temporarily modify the model's attributes before calling the
        # base `make_attention` method, then restore them immediately after.
        original_use_rope = self.attention_attrs["use_rope_in_attn"]
        original_rope = self.attention_attrs["rope"]
        original_window_size = self.window_size

        # Enable/disable RoPE for the current layer.
        # no_rope_layers[i] = 1 means layer i uses RoPE; 0 means no RoPE.
        has_rope = bool(self.no_rope_layers[layer_id])
        self.attention_attrs["rope"] = has_rope
        self.attention_attrs["use_rope_in_attn"] = has_rope and original_use_rope

        # Set the sliding window size for the current layer.
        assert self.layer_types[layer_id] in {"sliding_attention", "full_attention"}
        if self.layer_types[layer_id] == "full_attention":
            self.window_size = -1

        # Call the original `make_attention` with the temporarily-modified settings.
        super().make_attention(layer_id, attention, root_input, **kwargs)

        # Restore original values
        self.attention_attrs["use_rope_in_attn"] = original_use_rope
        self.attention_attrs["rope"] = original_rope
        self.window_size = original_window_size
