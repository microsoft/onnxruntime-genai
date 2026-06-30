# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
DINOv2 vision encoder export.

Supports: facebook/dinov2-base, facebook/dinov2-large, facebook/dinov2-giant, etc.
Also supports DINOv2 with register tokens (facebook/dinov2-base-imagenet1k-1-layer).

Key differences from plain ViT:
  - Optional register tokens (extra learnable tokens inserted after CLS)
  - LayerScale: per-layer learnable scalar multiplied onto attn/MLP outputs
  - activation = "gelu" (same as ViT)
  - CLS token present (same as ViT)
  - Absolute positional embedding (same as ViT)
  - FC1/GELU/FC2 MLP (same as ViT)

LayerScale note:
  HF DINOv2 applies LayerScale as:
      output = layer_scale_weight * attn_output  (element-wise)
  This is a Mul node inserted after the attention output projection and
  after the MLP output, before the residual add.
  We handle this by overriding make_layer() to insert the Mul nodes.
"""
import torch

from .base import VisionModel


class DINOv2Model(VisionModel):
    """
    DINOv2 vision encoder (facebook/dinov2-*).

    Differences from plain ViT:
      - Optional register tokens (num_register_tokens > 0)
      - LayerScale (lambda1/lambda2 per layer)
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Register tokens (DINOv2 with registers)
        self.num_register_tokens = getattr(config, "num_register_tokens", 0)

        # LayerScale is present in all DINOv2 models
        self.has_layer_scale = True

    def make_layer(self, layer_id, layer):
        """
        Override to insert LayerScale Mul nodes after attention and MLP outputs.

        DINOv2 layer structure:
            norm1 → attention → LayerScale(lambda1) → residual add
            norm2 → mlp       → LayerScale(lambda2) → residual add

        The SkipLayerNorm op fuses residual add + norm, so we insert the
        LayerScale Mul before the skip_input is consumed by SkipLayerNorm.
        """
        # Detect sub-modules
        ln1 = getattr(layer, "norm1", None) or getattr(layer, "layernorm_before", None)
        attn = getattr(layer, "attention", None) or getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        ln2 = getattr(layer, "norm2", None) or getattr(layer, "layernorm_after", None)
        mlp = getattr(layer, "mlp", None) or getattr(layer, "intermediate", None)
        ls1 = getattr(layer, "layer_scale1", None) or getattr(layer, "ls1", None)
        ls2 = getattr(layer, "layer_scale2", None) or getattr(layer, "ls2", None)

        if ln1 is None or attn is None or ln2 is None or mlp is None:
            raise AttributeError(
                f"Cannot find expected sub-modules in DINOv2 layer {layer_id} ({type(layer).__name__}). "
                f"Found: {[n for n, _ in layer.named_children()]}"
            )

        # DINOv2 has the same nested attention structure as HF ViT:
        #   Dinov2Attention
        #     ├── attention (Dinov2SelfAttention)  ← Q/K/V projections
        #     └── output (Dinov2SelfOutput)         ← output projection (.dense)
        # We must set _attn_out_module before calling make_attention() so that
        # _make_attention_output() can find the output projection.
        if hasattr(attn, "attention") and hasattr(attn, "output"):
            self._attn_out_module = attn.output   # Dinov2SelfOutput: .dense
            attn_qkv = attn.attention             # Dinov2SelfAttention: query, key, value
        else:
            self._attn_out_module = attn
            attn_qkv = attn

        # DINOv2 MLP: layer.mlp has fc1/fc2 directly (no split intermediate/output)
        self._mlp_output_module = None

        # 1. Pre-attention LayerNorm
        self.make_layernorm(layer_id, ln1,
                            skip=not self.layernorm_attrs["first_layernorm"],
                            simple=self.layernorm_attrs["simple"],
                            location="input")

        # 2. Attention
        self.make_attention(layer_id, attn_qkv, root_input=self.layernorm_attrs["output_0"])

        # 3. LayerScale after attention (if present)
        if ls1 is not None:
            self._make_layer_scale(layer_id, ls1, location="attn")

        # 4. Post-attention LayerNorm (SkipLayerNorm fuses residual add)
        self.make_layernorm(layer_id, ln2,
                            skip=True,
                            simple=self.layernorm_attrs["simple"],
                            location="post_attention")

        # 5. MLP
        self.make_mlp(layer_id, mlp, root_input=self.layernorm_attrs["output_0"])

        # 6. LayerScale after MLP (if present)
        if ls2 is not None:
            self._make_layer_scale(layer_id, ls2, location="mlp")

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    def _make_layer_scale(self, layer_id, layer_scale_module, location: str):
        """
        Insert a LayerScale Mul node.

        LayerScale: output = gamma * x  where gamma is a learnable vector [D].
        This is applied element-wise to the attention or MLP output before
        the residual add (which is fused into the next SkipLayerNorm).
        """
        gamma_name = f"vision_model.encoder.layers.{layer_id}.layer_scale_{location}.lambda1"
        gamma = (
            layer_scale_module.lambda1
            if hasattr(layer_scale_module, "lambda1")
            else layer_scale_module.gamma
            if hasattr(layer_scale_module, "gamma")
            else layer_scale_module.weight
        )
        self.make_initializer(gamma, gamma_name, to=self.io_dtype)

        # The current skip_input is the attn/MLP output — apply LayerScale to it
        skip_input = self.layernorm_attrs["skip_input"]
        mul_name = f"/vision_model/encoder/layers.{layer_id}/layer_scale_{location}/Mul"
        mul_output = f"{mul_name}/output_0"
        self.make_node("Mul", inputs=[skip_input, gamma_name], outputs=[mul_output], name=mul_name)
        self.make_value(mul_output, self.io_dtype, shape=["batch_size", "num_image_tokens", self.hidden_size])

        # Update skip_input to point to the scaled output
        self.layernorm_attrs["skip_input"] = mul_output
