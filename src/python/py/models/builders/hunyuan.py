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

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Compute effective rope_theta from the Dynamic NTK-alpha scaling used by Hunyuan.
        # From modular_hunyuan_v1_dense.py:
        #   base = rope_theta * alpha ^ (head_dim / (head_dim - 2))
        # With alpha=1000, head_dim=128:
        #   effective_theta ≈ 10000 * 1000^(128/126) ≈ 10,359,000
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            alpha = config.rope_scaling.get("alpha", 1.0)
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
            if alpha != 1.0 and head_dim is not None and head_dim > 2:
                config.rope_theta = config.rope_theta * (alpha ** (head_dim / (head_dim - 2)))

        # Disable rope_scaling: effective theta is now baked into config.rope_theta above.
        config.rope_scaling = None

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # GQA fuses RoPE inside the attention op (use_rope_in_attn=True) which makes
        # it impossible to insert QK norms between RoPE output and the attention op.
        # Force explicit RotaryEmbedding nodes so our override can place QK norms after them.
        if self.attention_attrs.get("use_rope_in_attn", False):
            self.attention_attrs["use_rope_in_attn"] = False
            # position_ids was removed from graph inputs when use_rope_in_attn was True; restore it.
            if "position_ids" not in self.input_names:
                self.input_names["position_ids"] = "position_ids"

        self.model_type = "hunyuandensev1"

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()

    def make_attention_qk_rope_and_norm(self, layer_id, attention, **kwargs):
        """
        Override to apply RoPE THEN QK norm (Hunyuan-specific ordering).

        Base order: [QK norm] -> RoPE
        Hunyuan order: RoPE -> [QK norm]

        query_layernorm / key_layernorm are aliased to q_norm / k_norm so
        the existing make_qk_norm() infrastructure can be reused.
        """
        # Alias Hunyuan weight names to what make_qk_norm expects
        if not hasattr(attention, "q_norm"):
            attention.q_norm = attention.query_layernorm
        if not hasattr(attention, "k_norm"):
            attention.k_norm = attention.key_layernorm

        # RoPE first, then QK norm
        cos_cache_name, sin_cache_name = self.make_attention_qk_rope(layer_id, **kwargs)
        self.make_attention_qk_norm(layer_id, attention)
        return cos_cache_name, sin_cache_name
