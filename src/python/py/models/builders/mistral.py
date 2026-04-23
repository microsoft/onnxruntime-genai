# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import json
import os

import numpy as np
import onnx_ir as ir
import torch

from .base import Model


class MistralModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class MistralNeMoModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Ministral3TextModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    @classmethod
    def _dequantize_fp8_weights(cls, model):
        """Dequantize float8_e4m3fn weights in place using per-tensor weight_scale_inv.

        The official Ministral-3B-Instruct-2512 model stores linear layer weights
        as float8_e4m3fn with a per-tensor inverse scale factor (weight_scale_inv).
        Standard PyTorch matmul cannot consume float8 parameters, so we eagerly
        cast them back to float32 before building the ONNX graph.

        Dequantization formula: weight_fp32 = weight_fp8.float() * weight_scale_inv
        """
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            return  # PyTorch version does not support FP8; nothing to do
        for module in model.modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            if module.weight.dtype != fp8_dtype:
                continue
            if not hasattr(module, "weight_scale_inv"):
                continue
            scale_inv = module.weight_scale_inv
            with torch.no_grad():
                dequantized = module.weight.float() * scale_inv.float()
                module.weight = torch.nn.Parameter(dequantized, requires_grad=False)

    def load_weights(self, input_path):
        # Mistral3ForConditionalGeneration (model_type="mistral3") is not
        # registered with AutoModelForCausalLM.  Load the full multimodal
        # model directly; make_model will find the embedded language-model
        # sub-modules (embed_tokens, MistralDecoderLayer, norm, lm_head)
        # via the standard module iteration already present in base.Model.
        if "ConditionalGeneration" in self.model_type:
            from transformers import Mistral3ForConditionalGeneration as HF_Mistral3VL

            model = HF_Mistral3VL.from_pretrained(
                self.model_name_or_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=self.hf_remote
            )
        else:
            model = super().load_weights(input_path)
        # Dequantize FP8 weights if present (official Ministral-3B model uses
        # float8_e4m3fn with per-tensor weight_scale_inv).
        self._dequantize_fp8_weights(model)
        return model

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """`minitral3` is not supported as an architecture, let's replace with `mistral`."""
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        genai_config["model"]["type"] = "mistral"

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)


class Ministral3VisionEncoderModel(Model):
    """Direct ``onnx_ir`` graph builder for the Pixtral vision encoder + multimodal projector.

    Builds the ONNX graph manually (analogous to other model builders in this
    codebase) rather than going through :func:`torch.onnx.export`.

    For a fixed image size (``vision_config.image_size`` x same):

    * Input:  ``pixel_values`` [1, num_channels, image_size, image_size]
    * Output: ``image_features`` [num_merged_patches, text_hidden_size]

    The 2-D RoPE (cos/sin) is pre-computed at graph-build time and stored as
    constant initialisers, removing the need for runtime position-id
    computation.

    The model is designed for batch_size = 1 (a single image per forward
    pass), matching how onnxruntime-genai processes multimodal inputs: each
    image is encoded independently before being concatenated into
    ``inputs_embeds``.
    """

    FILENAME = "vision_encoder.onnx"

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        vc = config.vision_config

        # Patch a copy of config with vision encoder attributes so that
        # Model.__init__ (which expects an LLM-style config) can initialise
        # the shared graph/values state without errors.
        vis_config = copy.deepcopy(config)
        vis_config.hidden_size = vc.hidden_size
        vis_config.intermediate_size = vc.intermediate_size
        vis_config.num_attention_heads = vc.num_attention_heads
        vis_config.num_key_value_heads = vc.num_attention_heads  # no GQA in vision encoder
        vis_config.num_hidden_layers = vc.num_hidden_layers
        vis_config.head_dim = getattr(vc, "head_dim", None) or vc.hidden_size // vc.num_attention_heads
        vis_config.hidden_act = getattr(vc, "hidden_act", "silu")
        vis_config.vocab_size = getattr(vc, "vocab_size", 1)
        vis_config.max_position_embeddings = (vc.image_size // vc.patch_size) ** 2
        vis_config.rms_norm_eps = 1e-5  # hardcoded in PixtralRMSNorm
        vis_config.rope_scaling = None  # prevent text-model rope init

        extra_options = {**extra_options, "filename": self.FILENAME, "exclude_lm_head": True, "exclude_embeds": True}

        super().__init__(vis_config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Re-initialise the graph with the vision-encoder name and reset
        # shared graph state so make_model() starts from a clean slate.
        self.graph.name = "pixtral_vision_encoder"

        # Store original (unpatched) config for callers that need top-level
        # Mistral3 attributes (e.g. spatial_merge_size, text_config, …).
        self.config = config
        self.vision_config = config.vision_config

        self.image_size = vc.image_size
        self.patch_size = vc.patch_size
        self.num_channels = vc.num_channels
        self.n_patches_per_side = self.image_size // self.patch_size
        self.n_patches = self.n_patches_per_side**2
        self.vis_hidden_size = vc.hidden_size
        self.vis_intermediate_size = vc.intermediate_size
        self.vis_num_heads = vc.num_attention_heads
        self.vis_head_dim = vc.head_dim if hasattr(vc, "head_dim") and vc.head_dim else vc.hidden_size // vc.num_attention_heads
        self.vis_num_layers = vc.num_hidden_layers
        self.vis_rms_norm_eps = 1e-5  # hardcoded in PixtralRMSNorm
        self.vis_attn_scale = float(self.vis_head_dim**-0.5)

        tc = config.text_config
        self.spatial_merge_size = config.spatial_merge_size
        self.n_merged_patches = self.n_patches // (self.spatial_merge_size**2)
        self.text_hidden_size = tc.hidden_size
        self.vision_feature_layer = config.vision_feature_layer
        self.projector_hidden_act = config.projector_hidden_act

    # ------------------------------------------------------------------ #
    #  Graph-construction helpers                                        #
    # ------------------------------------------------------------------ #

    def _rms_norm(self, name, root_input, weight_tensor, weight_name, shape):
        """SimplifiedLayerNormalization (PixtralRMSNorm)."""
        self.make_initializer(weight_tensor, weight_name, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, weight_name],
            outputs=[output],
            name=name,
            axis=-1,
            epsilon=self.vis_rms_norm_eps,
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape=shape)
        return output

    # ------------------------------------------------------------------ #
    #  2-D RoPE (pre-computed at graph-build time)                        #
    # ------------------------------------------------------------------ #

    def _precompute_rope_cos_sin(self):
        """Return cos/sin tensors shaped [n_patches, head_dim // 2].

        Pre-computes the Pixtral 2-D rotary embeddings for a fixed image
        grid.  The tensors are stored as constant initialisers (via
        ``make_rotary_embedding``) so they require no runtime computation.

        The ORT ``com.microsoft.RotaryEmbedding`` operator expects cos/sin
        caches of shape ``[max_sequence_length, head_dim // 2]`` and doubles
        them internally to produce the full-dimension rotation, so we return
        only the unique (non-duplicated) half-dimension slice.
        """
        vc = self.vision_config
        head_dim = self.vis_head_dim
        base = vc.rope_parameters["rope_theta"]
        n = self.n_patches_per_side

        h_idx = torch.arange(n)
        w_idx = torch.arange(n)
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs_h = torch.outer(h_idx, freqs[::2]).float()
        freqs_w = torch.outer(w_idx, freqs[1::2]).float()
        inv_freq = torch.cat([freqs_h[:, None, :].repeat(1, n, 1), freqs_w[None, :, :].repeat(n, 1, 1)], dim=-1).reshape(-1, head_dim // 2)
        # inv_freq: [n_patches, head_dim // 2]

        # Position IDs: row-major (h * max_width + w)
        h_grid, w_grid = torch.meshgrid(h_idx, w_idx, indexing="ij")
        position_ids = (h_grid * n + w_grid).reshape(-1)  # [n_patches]

        freqs_at_pos = inv_freq[position_ids]  # [n_patches, head_dim // 2]
        cos = freqs_at_pos.cos()
        sin = freqs_at_pos.sin()
        return cos, sin

    def make_rotary_embedding(self, name, root_input, **kwargs):
        """Vision-encoder override: pre-computed 2-D RoPE via com.microsoft.RotaryEmbedding.

        Unlike the text-model version, position IDs and cos/sin caches are
        pre-computed constants (one fixed image grid per model build).  The
        caches are created lazily on the first call and shared across all
        transformer layers.

        root_input: value name of shape [1, n_patches, n_heads * head_dim].
        Returns: output value name of the same shape.
        """
        if not hasattr(self, "_vis_rope_initialized"):
            cos, sin = self._precompute_rope_cos_sin()
            # cos/sin: [n_patches, head_dim // 2] — ORT RotaryEmbedding doubles internally
            self.make_initializer(cos, "vision.rope.cos_cache", to=self.io_dtype)
            self.make_initializer(sin, "vision.rope.sin_cache", to=self.io_dtype)
            pos_ids = torch.arange(self.n_patches, dtype=torch.int64).unsqueeze(0)
            self.make_initializer(pos_ids, "vision.rope.position_ids")
            self._vis_rope_initialized = True

        output = f"{name}/output_0"
        self.make_node(
            "RotaryEmbedding",
            inputs=[root_input, "vision.rope.position_ids", "vision.rope.cos_cache", "vision.rope.sin_cache"],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            interleaved=0,
            num_heads=self.vis_num_heads,
            rotary_embedding_dim=self.vis_head_dim,
        )
        self.make_value(output, self.io_dtype, shape=[1, self.n_patches, self.vis_hidden_size])
        return output

    # ------------------------------------------------------------------ #
    #  Attention layer                                                     #
    # ------------------------------------------------------------------ #

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Build one PixtralAttention layer (encoder-style, no KV cache).

        Overrides Model.make_attention for the vision encoder.

        root_input: [1, n_patches, vis_hidden_size]
        Sets self.layernorm_attrs["skip_input"] to the output name of the
        same shape, following the base-class convention.
        """
        b = f"/vision/layers.{layer_id}/attn"
        n_p = self.n_patches
        d = self.vis_hidden_size
        nh = self.vis_num_heads
        hd = self.vis_head_dim

        # Q / K / V projections (no bias in Pixtral attention)
        # -> [1, n_patches, n_heads * head_dim]
        q = f"{self.make_matmul(attention.q_proj, f'{b}/q_proj/MatMul', root_input)}/output_0"
        k = f"{self.make_matmul(attention.k_proj, f'{b}/k_proj/MatMul', root_input)}/output_0"
        v = f"{self.make_matmul(attention.v_proj, f'{b}/v_proj/MatMul', root_input)}/output_0"

        # Apply 2-D RoPE to Q and K in [1, n_patches, n_heads * head_dim] format
        q_rope = self.make_rotary_embedding(f"{b}/q_rotary/RotaryEmbedding", q)
        k_rope = self.make_rotary_embedding(f"{b}/k_rotary/RotaryEmbedding", k)

        # Reshape to [1, n_patches, n_heads, head_dim] and transpose to [1, n_heads, n_patches, head_dim]
        qkv_shape_4d = [1, n_p, nh, hd]
        q_4d = self.make_reshape(f"{b}/q_reshape", [q_rope, [1, n_p, nh, hd]], self.io_dtype, qkv_shape_4d)
        k_4d = self.make_reshape(f"{b}/k_reshape", [k_rope, [1, n_p, nh, hd]], self.io_dtype, qkv_shape_4d)
        v_4d = self.make_reshape(f"{b}/v_reshape", [v, [1, n_p, nh, hd]], self.io_dtype, qkv_shape_4d)

        qkv_t_shape = [1, nh, n_p, hd]
        q_t = self.make_transpose(f"{b}/q_t", q_4d, self.io_dtype, qkv_t_shape, perm=[0, 2, 1, 3])
        # k_t = self.make_transpose(f"{b}/k_t", k_4d, self.io_dtype, qkv_t_shape, perm=[0, 2, 1, 3])
        v_t = self.make_transpose(f"{b}/v_t", v_4d, self.io_dtype, qkv_t_shape, perm=[0, 2, 1, 3])

        # Scaled dot-product attention (encoder, no causal mask)
        # K^T: [1, nh, hd, n_p]
        # k_T = self.make_transpose(f"{b}/k_T", k_t, self.io_dtype, [1, nh, hd, n_p], perm=[0, 1, 3, 2])
        k_T = self.make_transpose(f"{b}/k_T", k_4d, self.io_dtype, [1, nh, hd, n_p], perm=[0, 2, 3, 1])
        attn_w = f"{b}/attn_w/MatMul/output_0"
        self.make_node("MatMul", inputs=[q_t, k_T], outputs=[attn_w], name=f"{b}/attn_w/MatMul")
        self.make_value(attn_w, self.io_dtype, shape=[1, nh, n_p, n_p])
        # Scale
        np_dtype = {ir.DataType.FLOAT: np.float32, ir.DataType.FLOAT16: np.float16}.get(self.io_dtype, np.float32)
        scale_name = f"{b}/attn_scale/scale"
        self.make_initializer(np.array(self.vis_attn_scale, dtype=np_dtype), scale_name)
        attn_ws = self.make_mul(f"{b}/attn_scale", [attn_w, scale_name], self.io_dtype, [1, nh, n_p, n_p])
        attn_probs = self.make_softmax(f"{b}/attn_softmax", attn_ws, self.io_dtype, [1, nh, n_p, n_p])
        attn_out_t = f"{b}/attn_out/MatMul/output_0"
        self.make_node("MatMul", inputs=[attn_probs, v_t], outputs=[attn_out_t], name=f"{b}/attn_out/MatMul")
        self.make_value(attn_out_t, self.io_dtype, shape=qkv_t_shape)

        # Transpose + Reshape back to [1, n_patches, hidden_size]
        attn_out = self.make_transpose(f"{b}/attn_out_t", attn_out_t, self.io_dtype, [1, n_p, nh, hd], perm=[0, 2, 1, 3])
        attn_out_2d = self.make_reshape(f"{b}/attn_out_reshape", [attn_out, [1, n_p, d]], self.io_dtype, [1, n_p, d])

        # O projection (no bias in Pixtral attention)
        o = f"{self.make_matmul(attention.o_proj, f'{b}/o_proj/MatMul', attn_out_2d)}/output_0"

        # Follow Model.make_attention convention: store output in layernorm_attrs["skip_input"]
        self.layernorm_attrs["skip_input"] = o

    # ------------------------------------------------------------------ #
    #  MLP (SiLU-gated)                                                   #
    # ------------------------------------------------------------------ #

    def _build_mlp(self, layer_id, mlp, root_input):
        """Build one PixtralMLP layer (SiLU(gate_proj) * up_proj, then down_proj).

        SiLU(x) is implemented as ``x * Sigmoid(x)`` using standard ONNX ops.

        root_input: [1, n_patches, vis_hidden_size]
        Returns: output name, same shape.
        """
        b = f"/vision/layers.{layer_id}/mlp"
        n_p = self.n_patches
        ff = self.vis_intermediate_size

        gate = f"{self.make_matmul(mlp.gate_proj, f'{b}/gate_proj/MatMul', root_input)}/output_0"
        up = f"{self.make_matmul(mlp.up_proj, f'{b}/up_proj/MatMul', root_input)}/output_0"

        # SiLU(gate) * up  (SiLU(x) = x * Sigmoid(x))
        sig_name = f"{b}/act/Sigmoid"
        self.make_sigmoid(sig_name, gate, self.io_dtype, [1, n_p, ff])
        sig_out = f"{sig_name}/output_0"

        silu_out = self.make_mul(f"{b}/act/Mul_silu", [gate, sig_out], self.io_dtype, [1, n_p, ff])
        gate_up = self.make_mul(f"{b}/gate_up/Mul", [silu_out, up], self.io_dtype, [1, n_p, ff])

        down = f"{self.make_matmul(mlp.down_proj, f'{b}/down_proj/MatMul', gate_up)}/output_0"
        return down

    # ------------------------------------------------------------------ #
    #  Single transformer layer                                           #
    # ------------------------------------------------------------------ #

    def make_layer(self, layer_id, layer):
        """Build one PixtralAttentionLayer.

        Pipeline:
          attention_norm -> attention -> residual ->
          ffn_norm -> feed_forward -> residual

        Reads the hidden-states tensor name from
        ``self.layernorm_attrs["root_input"]`` (set by the previous layer or
        by ``make_model`` before the first layer) and stores the output tensor
        name back there, following the ``Model.make_layer`` convention.
        """
        root_input = self.layernorm_attrs["root_input"]
        b = f"/vision/layers.{layer_id}"
        n_p = self.n_patches
        d = self.vis_hidden_size

        # attention_norm (RMSNorm, no skip)
        norm1_out = self._rms_norm(
            f"{b}/attention_norm/SimplifiedLayerNorm",
            root_input,
            layer.attention_norm.weight,
            f"{b}/attention_norm.weight",
            shape=[1, n_p, d],
        )

        # Attention
        self.make_attention(layer_id, layer.attention, norm1_out)
        attn_out = self.layernorm_attrs["skip_input"]

        # Residual 1
        res1 = self.make_add(f"{b}/residual1/Add", [root_input, attn_out], self.io_dtype, [1, n_p, d])

        # ffn_norm (RMSNorm, no skip)
        norm2_out = self._rms_norm(
            f"{b}/ffn_norm/SimplifiedLayerNorm", res1, layer.ffn_norm.weight, f"{b}/ffn_norm.weight", shape=[1, n_p, d]
        )

        # Feed-forward (SiLU-gated MLP)
        mlp_out = self._build_mlp(layer_id, layer.feed_forward, norm2_out)

        # Residual 2
        res2 = self.make_add(f"{b}/residual2/Add", [res1, mlp_out], self.io_dtype, [1, n_p, d])

        # Store output for next layer or post-processing
        self.layernorm_attrs["root_input"] = res2

    # ------------------------------------------------------------------ #
    #  Patch embedding (Conv2d + reshape + RMSNorm)                       #
    # ------------------------------------------------------------------ #

    def _build_patch_embedding(self, vt):
        """Build: pixel_values -> Conv2d -> flatten -> transpose -> ln_pre.

        Returns the value name of shape [1, n_patches, vis_hidden_size].
        """
        # Conv2d weights: [hidden_size, in_channels, patch_size, patch_size]
        conv_w = "vision.patch_conv.weight"
        self.make_initializer(vt.patch_conv.weight, conv_w, to=self.io_dtype)

        n_h = n_w = self.n_patches_per_side
        self.make_conv(
            "/vision/patch_conv/Conv",
            ["pixel_values", conv_w],
            self.io_dtype,
            [1, self.vis_hidden_size, n_h, n_w],
            dilations=[1, 1],
            group=1,
            kernel_shape=[self.patch_size, self.patch_size],
            pads=[0, 0, 0, 0],
            strides=[self.patch_size, self.patch_size],
        )
        conv_out = "/vision/patch_conv/Conv/output_0"

        # Transpose NCHW→NHWC: [1, hidden_size, n_h, n_w] → [1, n_h, n_w, hidden_size]
        # then Reshape to merge spatial dims: → [1, n_patches, hidden_size].
        # Transpose-before-Reshape avoids a rank-changing Reshape before a Transpose,
        # which can confuse graph optimisers.
        transposed = self.make_transpose(
            "/vision/patch_embed/Transpose", conv_out, self.io_dtype, [1, n_h, n_w, self.vis_hidden_size], perm=[0, 2, 3, 1]
        )
        patch_embed = self.make_reshape(
            "/vision/patch_embed/Reshape",
            [transposed, [1, self.n_patches, self.vis_hidden_size]],
            self.io_dtype,
            [1, self.n_patches, self.vis_hidden_size],
        )

        # ln_pre (SimplifiedLayerNormalization)
        ln_pre_out = self._rms_norm(
            "/vision/ln_pre/SimplifiedLayerNorm",
            patch_embed,
            vt.ln_pre.weight,
            "vision.ln_pre.weight",
            shape=[1, self.n_patches, self.vis_hidden_size],
        )
        return ln_pre_out

    # ------------------------------------------------------------------ #
    #  Multimodal projector                                               #
    # ------------------------------------------------------------------ #

    def _build_projector(self, proj, root_input):
        """Build the Mistral3MultiModalProjector.

        root_input: [1, n_patches, vis_hidden_size]

        Pipeline:
          norm -> patch_merger (unfold + linear) -> linear_1 -> gelu -> linear_2

        Returns: value name of shape [n_merged_patches, text_hidden_size].
        """
        n_p = self.n_patches
        d = self.vis_hidden_size
        s = self.spatial_merge_size
        nm = self.n_merged_patches
        n_h = n_w = self.n_patches_per_side
        mh, mw = n_h // s, n_w // s

        assert n_h % s == 0 and n_w % s == 0, f"image grid {n_h}x{n_w} not divisible by spatial_merge_size={s}"

        # --- Projector RMSNorm ---
        proj_norm_eps = float(self.config.text_config.rms_norm_eps)
        norm_w = "vision.projector.norm.weight"
        self.make_initializer(proj.norm.weight, norm_w, to=self.io_dtype)
        norm_out = "/vision/projector/norm/SimplifiedLayerNorm/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, norm_w],
            outputs=[norm_out],
            name="/vision/projector/norm/SimplifiedLayerNorm",
            axis=-1,
            epsilon=proj_norm_eps,
            stash_type=1,
        )
        self.make_value(norm_out, self.io_dtype, shape=[1, n_p, d])

        # Squeeze batch dimension: [1, n_patches, d] -> [n_patches, d]
        squeeze_out = self.make_reshape("/vision/projector/squeeze", [norm_out, [n_p, d]], self.io_dtype, [n_p, d])

        # --- Patch Merger (unfold equivalent for non-overlapping windows) ---
        #
        # PyTorch: image_tokens [n_patches, d]
        #   -> view(n_h, n_w, d)
        #   -> permute(2,0,1).unsqueeze(0) -> [1, d, n_h, n_w]
        #   -> unfold(kernel=s, stride=s)  -> [1, d*s*s, n_h//s * n_w//s]
        #   -> view(d*s*s, n_merged).t()   -> [n_merged, d*s*s]
        #
        # Equivalent single-reshape + transpose + reshape (no overlap, stride==kernel):
        #   [n_patches, d]
        #   -> [n_h//s, s, n_w//s, s, d]                   Reshape   (direct, skips intermediate [n_h, n_w, d])
        #   -> [n_h//s, n_w//s, d, s, s]  perm=[0,2,4,1,3] Transpose
        #   -> [n_merged, d*s*s]                            Reshape
        r = self.make_reshape("/vision/projector/merge/Reshape1", [squeeze_out, [mh, s, mw, s, d]], self.io_dtype, [mh, s, mw, s, d])
        tp = self.make_transpose("/vision/projector/merge/Transpose", r, self.io_dtype, [mh, mw, d, s, s], perm=[0, 2, 4, 1, 3])
        merged = self.make_reshape("/vision/projector/merge/Reshape2", [tp, [nm, d * s * s]], self.io_dtype, [nm, d * s * s])

        # Merging linear (no bias): [nm, d*s*s] -> [nm, d]
        merged_out = f"{self.make_matmul(proj.patch_merger.merging_layer, '/vision/projector/merging_layer/MatMul', merged)}/output_0"

        # --- linear_1 + gelu + linear_2 ---
        t_hid = self.text_hidden_size
        lin1_name = "/vision/projector/linear_1/MatMul"
        lin1_out = f"{self.make_matmul(proj.linear_1, lin1_name, merged_out)}/output_0"
        lin1_bias = getattr(proj.linear_1, "bias", None)
        if lin1_bias is not None and torch.count_nonzero(lin1_bias) > 0:
            lin1_bias_name = "vision.projector.linear_1.bias"
            self.make_initializer(lin1_bias, lin1_bias_name, to=self.io_dtype)
            lin1_add_name = f"{lin1_name}/BiasAdd"
            lin1_add_out = f"{lin1_add_name}/output_0"
            self.make_node("Add", inputs=[lin1_out, lin1_bias_name], outputs=[lin1_add_out], name=lin1_add_name)
            self.make_value(lin1_add_out, self.io_dtype, shape=[nm, t_hid])
            lin1_out = lin1_add_out

        # GELU activation (default projector_hidden_act is "gelu")
        gelu_out = "/vision/projector/gelu/output_0"
        self.make_node("Gelu", inputs=[lin1_out], outputs=[gelu_out], name="/vision/projector/gelu/Gelu", domain="com.microsoft")
        self.make_value(gelu_out, self.io_dtype, shape=[nm, t_hid])

        # linear_2: [nm, text_hidden_size] -> [nm, text_hidden_size]
        lin2_name = "/vision/projector/linear_2/MatMul"
        lin2_out = f"{self.make_matmul(proj.linear_2, lin2_name, gelu_out)}/output_0"
        lin2_bias = getattr(proj.linear_2, "bias", None)
        if lin2_bias is not None and torch.count_nonzero(lin2_bias) > 0:
            lin2_bias_name = "vision.projector.linear_2.bias"
            self.make_initializer(lin2_bias, lin2_bias_name, to=self.io_dtype)
            lin2_add_name = f"{lin2_name}/BiasAdd"
            lin2_add_out = f"{lin2_add_name}/output_0"
            self.make_node("Add", inputs=[lin2_out, lin2_bias_name], outputs=[lin2_add_out], name=lin2_add_name)
            self.make_value(lin2_add_out, self.io_dtype, shape=[nm, t_hid])
            lin2_out = lin2_add_out
        return lin2_out

    # ------------------------------------------------------------------ #
    #  Main entry points                                                  #
    # ------------------------------------------------------------------ #

    def _load_hf_model(self, input_path):
        from transformers import Mistral3ForConditionalGeneration

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Mistral3ForConditionalGeneration.from_pretrained(src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)

    def make_model(self, input_path):
        """Load HF weights and build the ONNX graph in-memory."""
        hf_model = self._load_hf_model(input_path)
        hf_model.eval()

        vt = hf_model.model.vision_tower  # PixtralVisionModel
        proj = hf_model.model.multi_modal_projector  # Mistral3MultiModalProjector

        # Graph input
        pixel_values_in = self.make_value("pixel_values", self.io_dtype, shape=[1, self.num_channels, self.image_size, self.image_size])
        self.graph.inputs.append(pixel_values_in)

        # Patch embedding
        x = self._build_patch_embedding(vt)

        # Transformer layers (2-D RoPE cos/sin caches are created lazily on the
        # first make_rotary_embedding call and shared across all layers)
        self.layernorm_attrs["root_input"] = x
        for layer_id, layer in enumerate(vt.transformer.layers):
            self.make_layer(layer_id, layer)
        x = self.layernorm_attrs["root_input"]

        # Projector
        image_features = self._build_projector(proj, x)

        # Graph output (rename via Identity so the output has the clean name)
        self.make_node("Identity", inputs=[image_features], outputs=["image_features"], name="/vision/output/Identity")
        out_val = self.make_value("image_features", self.io_dtype, shape=[self.n_merged_patches, self.text_hidden_size])
        self.graph.outputs.append(out_val)

        self.graph.sort()


class Ministral3EmbeddingModel(Model):
    """ONNX embedding model for the ``phi3v``-style multimodal pipeline.

    Inherits from :class:`Model` to fit the standard builder interface.

    ``input_ids`` must include ``image_token_id`` placeholder tokens at the
    positions where image features should be inserted.  The model:

    1. Embeds **all** tokens with the standard token-embedding table.
    2. Identifies positions where ``input_ids == image_token_id`` using
       ``Equal`` + ``NonZero``.
    3. Scatters the vision-encoder output ``image_features`` into those
       positions via ``ScatterND``.

    This keeps ``T_total = len(input_ids)`` unchanged, so ORT-GenAI's
    sequence-length tracking (KV cache, position IDs, attention mask) remains
    consistent.  During token generation ``input_ids`` contains a single new
    token (never an image placeholder), so ``NonZero`` returns an empty index
    tensor and ``ScatterND`` is a no-op.

    Graph (2-D ``input_ids [1, T]`` from ORT-GenAI's ``EmbeddingState``)::

        text_embeds   = Gather(embed_tokens_weight, input_ids)  # [1, T, H]
        text_2d       = Squeeze(text_embeds, [0])               # [T, H]
        flat_ids      = Squeeze(input_ids, [0])                 # [T]
        is_img        = Equal(flat_ids, image_token_id_const)   # [T] bool
        img_pos       = NonZero(is_img)                         # [1, N]
        img_pos_idx   = Transpose(img_pos, [1, 0])              # [N, 1]
        scattered_2d  = ScatterND(text_2d, img_pos_idx,
                                  image_features)               # [T, H]
        inputs_embeds = Unsqueeze(scattered_2d, [0])            # [1, T, H]
    """

    FILENAME = "embedding.onnx"
    filename = FILENAME

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.image_token_id = extra_options["image_token_id"]

    # ------------------------------------------------------------------

    def _load_hf_model(self, input_path):
        from transformers import Mistral3ForConditionalGeneration

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Mistral3ForConditionalGeneration.from_pretrained(src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)

    def make_model(self, input_path):
        """Load HF weights and build the embedding ONNX graph."""
        hf_model = self._load_hf_model(input_path)
        hf_model.eval()
        embed_weight = hf_model.model.language_model.embed_tokens.weight.detach().float().numpy()

        # Initialisers
        self.make_initializer(embed_weight, name="embed_tokens_weight")
        self.make_initializer(np.array(self.image_token_id, dtype=np.int64), name="image_token_id_const")
        # Use a Constant node (always inline) rather than an initializer so that
        # shape inference can read the axes value even when external data is used.
        _squeeze_axes = ir.Tensor(np.array([0], dtype=np.int64), name="squeeze_batch_axes")
        self.make_node(
            "Constant", inputs=[], outputs=["squeeze_batch_axes"], name="/embed/squeeze_batch_axes/Constant", value=_squeeze_axes
        )
        self.make_value("squeeze_batch_axes", ir.DataType.INT64, shape=[1])

        # Graph inputs (dynamic shapes).
        # ORT-GenAI passes input_ids as 2D [batch, seq_len].
        self.graph.inputs.append(self.make_value("input_ids", ir.DataType.INT64, shape=[None, None]))
        self.graph.inputs.append(self.make_value("image_features", ir.DataType.FLOAT, shape=[None, self.hidden_size]))

        # Nodes:
        # 1. Embed all tokens: input_ids [1, T] -> text_embeds [1, T, H]
        self.make_node("Gather", inputs=["embed_tokens_weight", "input_ids"], outputs=["text_embeds"], name="/embed/Gather", axis=0)
        # 2. Squeeze batch dim for easier indexing: [1, T, H] → [T, H]
        self.make_node("Squeeze", inputs=["text_embeds", "squeeze_batch_axes"], outputs=["text_2d"], name="/embed/Squeeze_3d")
        # 3. Flatten input_ids: [1, T] → [T]
        self.make_node("Squeeze", inputs=["input_ids", "squeeze_batch_axes"], outputs=["flat_ids"], name="/embed/Squeeze_ids")
        # 4. Boolean mask where tokens are image placeholders: [T] bool
        self.make_node("Equal", inputs=["flat_ids", "image_token_id_const"], outputs=["is_image"], name="/embed/Equal")
        # 5. Positions of image placeholders: [1, N] int64
        self.make_node("NonZero", inputs=["is_image"], outputs=["img_pos"], name="/embed/NonZero")
        # 6. Transpose to [N, 1] for ScatterND
        self.make_node("Transpose", inputs=["img_pos"], outputs=["img_pos_idx"], name="/embed/Transpose", perm=[1, 0])
        # 7. Scatter image_features into text embeddings at placeholder positions
        self.make_node("ScatterND", inputs=["text_2d", "img_pos_idx", "image_features"], outputs=["scattered_2d"], name="/embed/ScatterND")
        # 8. Re-add batch dimension: [T, H] → [1, T, H]
        self.make_node("Unsqueeze", inputs=["scattered_2d", "squeeze_batch_axes"], outputs=["inputs_embeds"], name="/embed/Unsqueeze")

        # Graph output
        self.graph.outputs.append(self.make_value("inputs_embeds", ir.DataType.FLOAT, shape=[1, None, self.hidden_size]))

        self.graph.sort()


class Ministral3ConditionalGenerationModel(Model):
    """Orchestrates exporting the vision encoder, embedding model, and text
    decoder for ``Mistral3ForConditionalGeneration`` (Ministral-3-3B-Instruct).

    The exported artifacts are:

    * ``vision_encoder.onnx`` - Pixtral vision tower + multimodal projector.
    * ``embedding.onnx`` - token-embedding table + image-feature prepend.
    * ``model.onnx`` - Mistral text decoder (``inputs_embeds`` → logits).
    * ``genai_config.json`` - ``phi3v``-type VLM config understood by
      ``onnxruntime-genai ≥ 0.12``.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # --- Vision encoder ---
        self.vision_encoder = Ministral3VisionEncoderModel(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # --- Embedding model ---
        # Flatten text_config attributes onto the top-level config so that
        # Model.__init__ (inside Ministral3EmbeddingModel) finds hidden_size etc.
        text_obj_config = copy.deepcopy(config)
        text_config = config.text_config
        # Ministral3VisionEncoderModel needs information from text_config and vision_config.
        # We should give Ministral3VisionEncoderModel vision_config and the couple of parameters from text_config.
        # Let's keep that for a later changer when there are more Vision models.
        for key in text_config:
            if not hasattr(text_obj_config, key):
                setattr(text_obj_config, key, getattr(text_config, key))

        embed_extra_options = dict(extra_options)
        embed_extra_options["image_token_id"] = config.image_token_id

        # The embedding table is always stored as float32.
        self.embedding_model = Ministral3EmbeddingModel(text_obj_config, io_dtype, ir.DataType.FLOAT, ep, cache_dir, embed_extra_options)

        # --- Text decoder (same flattened config, exclude_embeds=True) ---
        text_extra_options = dict(extra_options)
        text_extra_options["exclude_embeds"] = True

        self.text_model = Ministral3TextModel(text_obj_config, io_dtype, onnx_dtype, ep, cache_dir, text_extra_options)

    # ------------------------------------------------------------------
    # builder.py interface
    # ------------------------------------------------------------------

    def make_model(self, input_path):
        print("Building vision encoder (Pixtral + multimodal projector) for Mistral3ForConditionalGeneration...")
        self.vision_encoder.make_model(input_path)
        print("Building embedding model for Mistral3ForConditionalGeneration...")
        self.embedding_model.make_model(input_path)
        print("Building text decoder for Mistral3ForConditionalGeneration...")
        self.text_model.make_model(input_path)

    def save_model(self, out_dir):
        self.vision_encoder.save_model(out_dir)
        self.embedding_model.save_model(out_dir)
        self.text_model.save_model(out_dir)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        # Let the text model write genai_config.json first, then extend it
        # with vision + embedding sections for the phi3v VLM pipeline.
        self.text_model.make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        spatial_merge_size = self.vision_encoder.config.spatial_merge_size

        # onnxruntime-genai uses "phi3v" as the model type for the
        # Vision + Embedding + Decoder multimodal pipeline.
        genai_config["model"]["type"] = "phi3v"

        genai_config["model"]["vision"] = {
            "filename": self.vision_encoder.filename,
            "spatial_merge_size": spatial_merge_size,
            "inputs": {"pixel_values": "pixel_values"},
            "outputs": {"image_features": "image_features"},
        }

        genai_config["model"]["embedding"] = {
            "filename": self.embedding_model.filename,
            "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
        }

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.text_model.save_processing(model_name_or_path, extra_kwargs, out_dir)
