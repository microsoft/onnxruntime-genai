"""
SeedOssModel ONNX builder for ONNX Runtime GenAI.

SeedOss architecture (ByteDance Seed) is structurally similar to Llama with:
  - attention_bias=True (QKV bias)
  - attention_out_bias=False
  - attn_post_norm per decoder layer (extra vs Llama)
  - ffn_post_norm per decoder layer (extra vs Llama)
  - GQA (num_key_value_heads != num_attention_heads)
  - RoPE with rope_theta=1e7

This builder inherits from LlamaModel and extends make_layer() to
inject the extra post-norms in the residual path.
"""

from .llama import LlamaModel


class SeedOssModel(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # LlamaModel base does not always populate layer_attrs (varies by
        # ORT-GenAI version); ensure the dict exists before we set keys on it.
        if not hasattr(self, "layer_attrs") or self.layer_attrs is None:
            self.layer_attrs = {}

        # SeedOss has per-layer post-norms after attention and MLP
        # These bools inform the downstream graph builder to wire extra
        # LayerNorm/RMSNorm nodes into the decoder block.
        self.layer_attrs["has_attn_post_norm"] = getattr(config, "has_attn_post_norm", True)
        self.layer_attrs["has_ffn_post_norm"] = getattr(config, "has_ffn_post_norm", True)

        # attention_bias is auto-detected in base.py via hasattr on attn.q_proj.bias
        # attention_out_bias is auto-detected similarly on attn.o_proj.bias
        # We only need to set flags if the config explicitly overrides detection
        self.attention_bias = getattr(config, "attention_bias", True)
        self.attention_out_bias = getattr(config, "attention_out_bias", False)

    def make_layer(self, layer_id: int, layer) -> None:
        """
        SeedOss decoder block topology:

          input_layernorm  -->  self_attn  --> attn_post_norm
               |                                          |
               +------------------------------------------+   (residual)

          post_attention_layernorm  -->  MLP  --> ffn_post_norm
               |                                                     |
               +-----------------------------------------------------+   (residual)

        This is pre-norm + post-norm hybrid.  The extra post-norms sit
        inside the residual branches, which LlamaModel does not create.
        """

        # ---- input norm + attention ----
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )

        root_after_input_norm = self.layernorm_attrs["output_0"]

        self.make_attention(
            layer_id,
            layer.self_attn,
            root_input=root_after_input_norm,
        )

        # ---- attn_post_norm (SeedOss specific) ----
        if self.layer_attrs.get("has_attn_post_norm", False):
            # Some HF weights flatten this into self_attn.post_attn_layernorm,
            # others expose it as layer.attn_post_norm.  Try both.
            attn_post_norm = None
            if hasattr(layer.self_attn, "post_attn_layernorm"):
                attn_post_norm = layer.self_attn.post_attn_layernorm
            elif hasattr(layer, "attn_post_norm"):
                attn_post_norm = layer.attn_post_norm

            if attn_post_norm is not None:
                self.make_layernorm(
                    layer_id,
                    attn_post_norm,
                    skip=True,
                    simple=self.layernorm_attrs["simple"],
                    location="post_attention",
                )
                # base.py accumulates into layernorm_attrs["output_0"]
                # the attention output is still in the residual path

        # ---- post_attention_layernorm (standard) ----
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )

        # ---- MLP ----
        root_after_post_attn_norm = self.layernorm_attrs["output_0"]
        self.make_mlp(
            layer_id,
            layer.mlp,
            root_input=root_after_post_attn_norm,
        )

        # ---- ffn_post_norm (SeedOss specific) ----
        if self.layer_attrs.get("has_ffn_post_norm", False):
            ffn_post_norm = None
            if hasattr(layer.mlp, "post_mlp_layernorm"):
                ffn_post_norm = layer.mlp.post_mlp_layernorm
            elif hasattr(layer, "ffn_post_norm"):
                ffn_post_norm = layer.ffn_post_norm

            if ffn_post_norm is not None:
                self.make_layernorm(
                    layer_id,
                    ffn_post_norm,
                    skip=True,
                    simple=self.layernorm_attrs["simple"],
                    location="post_mlp",
                )

        # ---- bookkeeping for base.py ----
        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True
