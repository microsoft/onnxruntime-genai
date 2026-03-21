# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
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
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            if alpha != 1.0:
                config.rope_theta = config.rope_theta * (alpha ** (head_dim / (head_dim - 2)))

        # Disable rope_scaling: effective theta is now baked into config.rope_theta above.
        config.rope_scaling = None

        # Disable QKV fusion so separate q_path/k_path/v_path are created in
        # make_attention_input_proj — required so our override can apply QK norms
        # on individual Q and K paths after RoPE.
        extra_options = {**extra_options, "disable_qkv_fusion": True}

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # GQA fuses RoPE inside the attention op (use_rope_in_attn=True) which makes
        # it impossible to insert QK norms between RoPE output and the attention op.
        # Force explicit RotaryEmbedding nodes so our override can place QK norms after them.
        if self.attention_attrs.get("use_rope_in_attn", False):
            self.attention_attrs["use_rope_in_attn"] = False
            # position_ids was removed from graph inputs when use_rope_in_attn was True; restore it.
            if "position_ids" not in self.input_names:
                self.input_names.append("position_ids")

        self.model_type = "hunyuan_v1_dense"

    def make_attention_qk_subgraph(self, layer_id, attention, root_input, **kwargs):
        """
        Override to apply QK norms AFTER RoPE (Hunyuan-specific ordering).

        Base class order: [optional QK norm] -> RoPE -> repeat_kv -> attn_op
        Hunyuan order:    RoPE -> [QK norm]  -> repeat_kv -> attn_op

        The query_layernorm / key_layernorm weight attributes are aliased to
        q_norm / k_norm so the existing make_qk_norm() infrastructure can be reused.
        """

        # Step 1: RoPE (no pre-RoPE QK norm for Hunyuan)
        cos_cache_name, sin_cache_name = "", ""
        if self.attention_attrs["use_rope_in_attn"]:
            cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches()
        else:
            q_rotary_name = f"/model/layers.{layer_id}/attn/q_rotary/RotaryEmbedding"
            self.make_rotary_embedding(
                q_rotary_name,
                root_input=self.attention_attrs["q_path"],
                position_ids=kwargs.get("position_ids", "position_ids"),
            )
            self.attention_attrs["q_path"] = f"{q_rotary_name}/output_0"
            k_rotary_name = f"/model/layers.{layer_id}/attn/k_rotary/RotaryEmbedding"
            self.make_rotary_embedding(
                k_rotary_name,
                root_input=self.attention_attrs["k_path"],
                position_ids=kwargs.get("position_ids", "position_ids"),
            )
            self.attention_attrs["k_path"] = f"{k_rotary_name}/output_0"

        # Step 2: QK norm AFTER RoPE — alias Hunyuan's attribute names to what make_qk_norm expects
        attention.q_norm = attention.query_layernorm
        attention.k_norm = attention.key_layernorm
        self.make_qk_norm(layer_id, attention)

        # Step 3: Repeat KV (for GQA with MultiHeadAttention op)
        past_k = f"past_key_values.{layer_id}.key"
        past_v = f"past_key_values.{layer_id}.value"
        present_k = f"present.{layer_id}.key"
        present_v = f"present.{layer_id}.value"
        if self.num_attn_heads != self.num_kv_heads and self.attention_attrs["op_type"] == "MultiHeadAttention":
            self.attention_attrs["k_path"] = self.make_repeat_kv(
                layer_id, root_input=self.attention_attrs["k_path"], past_kv=past_k, present_kv=present_k
            )
            self.attention_attrs["v_path"] = self.make_repeat_kv(
                layer_id, root_input=self.attention_attrs["v_path"], past_kv=past_v, present_kv=present_v
            )
            past_k, past_v, present_k, present_v = "", "", "", ""

        # Step 4: Sinks (attention sink tokens, rarely used)
        sinks_name = ""
        if self.attention_attrs["sinks"]:
            sinks_name = f"model.layers.{layer_id}.attn.sinks"
            self.make_initializer(attention.sinks, sinks_name, to=self.io_dtype)

        # Step 5: Attention op (GroupQueryAttention or MultiHeadAttention)
        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        self.make_attention_op(
            attn_name,
            q_path=self.attention_attrs["q_path"],
            k_path=self.attention_attrs["k_path"],
            v_path=self.attention_attrs["v_path"],
            past_k=past_k,
            past_v=past_v,
            present_k=present_k,
            present_v=present_v,
            cos_cache=cos_cache_name,
            sin_cache=sin_cache_name,
            sinks=sinks_name,
            **kwargs,
        )
