# -------------------------------------------------------------------------
# Copyright (C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .base import Model


class HunyuanDenseV1Model(Model):
    """
    Builder for tencent/HY-MT (HunYuanDenseV1) models.

    Key architectural differences from standard Llama-style models:
      1. QK norm (query_layernorm / key_layernorm) applied AFTER RoPE, not before.
      2. Dynamic NTK-alpha RoPE scaling:
             effective_theta = rope_theta * alpha ^ (head_dim / (head_dim - 2))
         This is baked into a static theta so the ONNX export uses standard RoPE.
    All weight names (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
    down_proj, input_layernorm, post_attention_layernorm) are standard.
    """

    def make_rope_init(self, config):
        if config.rope_parameters["rope_type"] == "dynamic":
            # Hunyuan bakes dynamic NTK-alpha scaling into theta and then uses standard RoPE.
            # Compute effective rope_theta from the Dynamic NTK-alpha scaling used by Hunyuan.
            # From modular_hunyuan_v1_dense.py:
            #   base = rope_theta * alpha ^ (head_dim / (head_dim - 2))
            # With alpha=1000, head_dim=128:
            #   effective_theta ≈ 10000 * 1000^(128/126) ≈ 10,359,000
            alpha = config.rope_parameters["alpha"]
            self.rope_attrs["theta"] *= alpha ** (self.head_size / (self.head_size - 2))

    def is_fused_rope_supported(self):
        # GQA fuses RoPE inside the attention op which makes it impossible to
        # insert QK norms between RoPE output and the attention op.
        # Force explicit RotaryEmbedding nodes so QK norms can be placed after them.
        return False

    def make_attention_init(self, config):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init(config)

    def make_attention_qk_rope_and_norm(self, layer_id, attention, **kwargs):
        """
        Override to apply RoPE THEN QK norm (Hunyuan-specific ordering).

        Base order: [QK norm] -> RoPE
        Hunyuan order: RoPE -> [QK norm]

        query_layernorm / key_layernorm are aliased to q_norm / k_norm so
        the existing make_qk_norm() infrastructure can be reused.
        """
        # Alias Hunyuan weight names to what make_qk_norm expects.
        # Some Transformers versions expose q_norm/k_norm as None while the
        # real modules live under query_layernorm/key_layernorm.
        if getattr(getattr(attention, "q_norm", None), "weight", None) is None and hasattr(attention, "query_layernorm"):
            attention.q_norm = attention.query_layernorm
        if getattr(getattr(attention, "k_norm", None), "weight", None) is None and hasattr(attention, "key_layernorm"):
            attention.k_norm = attention.key_layernorm

        # RoPE first, then QK norm
        cos_cache_name, sin_cache_name = self.make_attention_qk_rope(layer_id, **kwargs)
        self.make_attention_qk_norm(layer_id, attention)
        return cos_cache_name, sin_cache_name
