# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Qwen2.5-VL and Qwen3-VL vision encoder export.

Architecture (Qwen2VisionTransformerPretrainedModel):
  - Patch embedding: Conv2d(14x14) → reshape (NO CLS token, NO absolute pos embed)
  - Positional encoding: 2D RoPE applied inside each attention layer
  - Encoder: N × Qwen2VisionEncoderLayer
      layernorm1 → attention (with 2D RoPE) → layernorm2 → SwiGLU MLP
  - Patch merger: spatial downsampling + linear projection → image_features

Extra inputs beyond pixel_values:
  - image_grid_thw [num_images, 3]: (temporal, height, width) grid dims per image
    Used to compute 2D RoPE position IDs dynamically.

Key differences from plain ViT:
  - No CLS token
  - No absolute positional embedding
  - 2D RoPE (height + width coordinates) applied per attention layer
  - SwiGLU MLP (gate_proj + up_proj → SiLU → down_proj)
  - Variable resolution input (dynamic num_patches)
  - Patch merger at the end (spatial_merge_size=2 downsampling)

Qwen3-VL adds:
  - QK-norm (RMSNorm after Q and K projections)
"""
import math

import onnx_ir as ir
import torch

from ..base import VisionModel


class Qwen25VLVisionModel(VisionModel):
    """
    Qwen2.5-VL vision encoder (Qwen/Qwen2.5-VL-*).

    Inputs:
      - pixel_values [total_patches, num_channels, patch_h, patch_w]
        (flattened across all images in the batch)
      - image_grid_thw [num_images, 3]  (T, H, W grid dims per image)

    Output:
      - image_features [total_patches / merge_size^2, hidden_size]
        (after patch merger downsampling)
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # ── Qwen-VL specific config ────────────────────────────────────────
        # config here is config.vision_config from the VLM config
        self.spatial_merge_size = getattr(config, "spatial_merge_size", 2)
        self.temporal_patch_size = getattr(config, "temporal_patch_size", 2)
        self.in_channels = getattr(config, "in_channels", 3)

        # Qwen-VL uses a different patch embedding: Conv3d or Conv2d with
        # temporal_patch_size channels. The actual in_channels for the conv
        # is num_channels * temporal_patch_size.
        self.conv_in_channels = self.in_channels * self.temporal_patch_size

        # ── Patch embedding: no CLS, no absolute pos embed ─────────────────
        self.patch_embed_attrs["has_cls_token"] = False
        self.patch_embed_attrs["has_pos_embed"] = False
        self.patch_embed_attrs["pos_embed_type"] = "2d_rope"

        # ── MLP: FC1/GELU/FC2 (Qwen-VL ViT uses linear_fc1/linear_fc2) ───
        # Despite the "VL" name, the ViT MLP is standard FC1/GELU/FC2,
        # not SwiGLU. Weight names: visual.blocks.N.mlp.linear_fc1/linear_fc2
        self.mlp_attrs["use_fc"] = True
        self.mlp_attrs["use_proj"] = False

        # ── Attention: no packed matmul (2D RoPE applied per Q/K separately) ─
        self.attention_attrs["use_packed_matmul"] = False
        self.attention_attrs["use_rope_in_attn"] = False

        # ── Extra inputs ───────────────────────────────────────────────────
        # image_grid_thw: [num_images, 3] — (T, H, W) grid dims per image
        # Used to compute 2D RoPE position IDs.
        self.input_names["image_grid_thw"] = "image_grid_thw"
        self.input_types["image_grid_thw"] = ir.DataType.INT32
        self.input_shapes["image_grid_thw"] = ["num_images", 3]

        # ── Override input shape: Qwen-VL uses flattened patches ───────────
        # pixel_values: [total_patches, conv_in_channels, patch_size, patch_size]
        self.input_shapes["pixel_values"] = [
            "total_patches",
            self.conv_in_channels,
            self.patch_size,
            self.patch_size,
        ]
        self.input_types["pixel_values"] = self.io_dtype

        # ── Output shape: after patch merger ──────────────────────────────
        # The patch merger reduces spatial dims by spatial_merge_size^2
        self.output_shapes["image_features"] = ["num_image_tokens", self.hidden_size]

        # ── RoPE: 2D rotary embedding ──────────────────────────────────────
        # head_dim for RoPE = head_size // 2 (split between height and width)
        self.rope_head_dim = self.head_size // 2

    # ── Patch embedding override ───────────────────────────────────────────

    def make_patch_embedding(self, patch_embed):
        """
        Qwen-VL patch embedding:
            pixel_values [total_patches, conv_in_channels, patch_size, patch_size]
                ↓ Conv2d (no bias in Qwen-VL)
            [total_patches, hidden_size, 1, 1]
                ↓ Reshape
            [total_patches, hidden_size]
                ↓ (no CLS, no pos embed — 2D RoPE applied per layer)
            → layernorm_attrs["root_input"] = "patch_embed_output"
        """
        basename = "/vision_model/patch_embedding"

        proj = getattr(patch_embed, "proj", None) or getattr(patch_embed, "projection", None)
        if proj is None:
            raise AttributeError(f"Cannot find proj in Qwen-VL patch_embed: {type(patch_embed)}")

        # Conv2d weight: [hidden_size, conv_in_channels, patch_size, patch_size]
        conv_weight_name = "vision_model.patch_embedding.proj.weight"
        self.make_initializer(proj.weight, conv_weight_name, to=self.io_dtype)

        conv_inputs = ["pixel_values", conv_weight_name]
        if proj.bias is not None and torch.count_nonzero(proj.bias) > 0:
            conv_bias_name = "vision_model.patch_embedding.proj.bias"
            self.make_initializer(proj.bias, conv_bias_name, to=self.io_dtype)
            conv_inputs.append(conv_bias_name)

        # Conv output: [total_patches, hidden_size, 1, 1]
        self.make_conv(
            f"{basename}/Conv",
            conv_inputs,
            dtype=self.io_dtype,
            shape=["total_patches", self.hidden_size, 1, 1],
            kernel_shape=[self.patch_size, self.patch_size],
            strides=[self.patch_size, self.patch_size],
        )

        # Reshape to [total_patches, hidden_size]
        self.make_reshape(
            f"{basename}/Reshape",
            [f"{basename}/Conv/output_0",
             f"/model/constants/INT64/[0, {self.hidden_size}]"],
            dtype=self.io_dtype,
            shape=["total_patches", self.hidden_size],
        )

        # Update residual stream pointer
        self.layernorm_attrs["root_input"] = f"{basename}/Reshape/output_0"
        self.layernorm_attrs["skip_input"] = f"{basename}/Reshape/output_0"

    # ── 2D RoPE subgraph ───────────────────────────────────────────────────

    def make_2d_rope_caches(self, layer_id):
        """
        Build the 2D RoPE cos/sin caches from image_grid_thw.

        Qwen-VL 2D RoPE algorithm (from HF source):
          1. Extract H, W from image_grid_thw[0, 1] and image_grid_thw[0, 2]
          2. Build row positions [0..H-1] and col positions [0..W-1]
          3. Tile to get [total_patches] position vectors for each axis
          4. Compute freqs = outer(positions, inv_freq) → [total_patches, head_dim/2]
          5. cos_cache = cos(freqs), sin_cache = sin(freqs)

        The C++ runtime (QwenVisionState) runs the vision encoder one image at a
        time, so image_grid_thw has shape [1, 3] = [[T, H, W]] for a single image.

        Graph structure:
          image_grid_thw [1, 3]
              ↓ Gather(axis=1, indices=[1]) → H_scalar
              ↓ Gather(axis=1, indices=[2]) → W_scalar
              ↓ Range(0, H) → row_ids [H]
              ↓ Range(0, W) → col_ids [W]
              ↓ Tile/Repeat to [H*W] for each
              ↓ Cast to float
              ↓ MatMul with inv_freq [head_dim/2] → freqs [H*W, head_dim/2]
              ↓ Cos/Sin → cos_cache, sin_cache [H*W, head_dim/2]

        Returns: (cos_output_name, sin_output_name)
        """
        basename = f"/vision_model/rope"

        # ── inv_freq initializer (shared across all layers) ────────────────
        rope_theta = getattr(self, "rope_theta", 10000.0)
        dim = self.rope_head_dim  # head_size // 2
        inv_freq_vals = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        inv_freq_name = "vision_model.rope.inv_freq"
        if inv_freq_name not in self.node_names:
            self.make_initializer(inv_freq_vals, inv_freq_name, to=ir.DataType.FLOAT)
            # Register as a "node" so we don't re-create it
            self.node_names.add(inv_freq_name)

        # Only build the RoPE subgraph once (shared across all layers)
        cos_output = f"{basename}/cos_cache"
        sin_output = f"{basename}/sin_cache"

        if cos_output in self.values:
            # Already built — reuse
            return cos_output, sin_output

        # ── Step 1: Extract H and W from image_grid_thw[0] ────────────────
        # image_grid_thw: [1, 3] = [[T, H, W]]
        # Squeeze to [3]: Gather(image_grid_thw, 0, axis=0) → [3]
        gather_row_name = f"{basename}/gather_row"
        self.make_gather(gather_row_name,
                         ["image_grid_thw", "/model/constants/INT64/0"],
                         ir.DataType.INT32, [3], axis=0)
        row_tensor = f"{gather_row_name}/output_0"

        # H = row_tensor[1], W = row_tensor[2]
        gather_H_name = f"{basename}/gather_H"
        self.make_gather(gather_H_name,
                         [row_tensor, "/model/constants/INT64/1"],
                         ir.DataType.INT32, [1], axis=0)
        H_tensor = f"{gather_H_name}/output_0"

        gather_W_name = f"{basename}/gather_W"
        self.make_gather(gather_W_name,
                         [row_tensor, "/model/constants/INT64/2"],
                         ir.DataType.INT32, [1], axis=0)
        W_tensor = f"{gather_W_name}/output_0"

        # Cast H and W to INT64 for Range op
        cast_H_name = f"{basename}/cast_H"
        self.make_cast(cast_H_name, H_tensor, ir.DataType.INT64, [1])
        H_int64 = f"{cast_H_name}/output_0"

        cast_W_name = f"{basename}/cast_W"
        self.make_cast(cast_W_name, W_tensor, ir.DataType.INT64, [1])
        W_int64 = f"{cast_W_name}/output_0"

        # Squeeze scalars: [1] → scalar
        squeeze_H_name = f"{basename}/squeeze_H"
        self.make_squeeze(squeeze_H_name,
                          [H_int64, "/model/constants/INT64/[0]"],
                          ir.DataType.INT64, [])
        H_scalar = f"{squeeze_H_name}/output_0"

        squeeze_W_name = f"{basename}/squeeze_W"
        self.make_squeeze(squeeze_W_name,
                          [W_int64, "/model/constants/INT64/[0]"],
                          ir.DataType.INT64, [])
        W_scalar = f"{squeeze_W_name}/output_0"

        # ── Step 2: Build row position IDs [0..H-1] ───────────────────────
        # Range(start=0, limit=H, delta=1) → [H]
        range_H_name = f"{basename}/range_H"
        range_H_output = f"{range_H_name}/output_0"
        self.make_node("Range",
                       inputs=["/model/constants/INT64/0", H_scalar, "/model/constants/INT64/1"],
                       outputs=[range_H_output],
                       name=range_H_name)
        self.make_value(range_H_output, ir.DataType.INT64, ["H"])

        # ── Step 3: Build col position IDs [0..W-1] ───────────────────────
        range_W_name = f"{basename}/range_W"
        range_W_output = f"{range_W_name}/output_0"
        self.make_node("Range",
                       inputs=["/model/constants/INT64/0", W_scalar, "/model/constants/INT64/1"],
                       outputs=[range_W_output],
                       name=range_W_name)
        self.make_value(range_W_output, ir.DataType.INT64, ["W"])

        # ── Step 4: Tile row IDs to [H*W]: each row repeated W times ──────
        # row_ids = repeat_interleave(range_H, W) = [0,0,...,0, 1,1,...,1, ..., H-1,...]
        # Implemented as: Reshape([H,1]) → Expand([H,W]) → Reshape([H*W])
        reshape_H_name = f"{basename}/reshape_H_col"
        self.make_reshape(reshape_H_name,
                          [range_H_output, "/model/constants/INT64/[-1, 1]"],
                          ir.DataType.INT64, ["H", 1])

        # Expand shape: [H, W]
        expand_shape_HW_name = f"{basename}/expand_shape_HW"
        self.make_concat(expand_shape_HW_name,
                         [H_int64, W_int64],
                         ir.DataType.INT64, [2], axis=0)

        expand_H_name = f"{basename}/expand_H"
        self.make_expand(expand_H_name,
                         [f"{reshape_H_name}/output_0", f"{expand_shape_HW_name}/output_0"],
                         ir.DataType.INT64, ["H", "W"])

        # Compute H*W for reshape
        mul_HW_name = f"{basename}/mul_HW"
        mul_HW_output = f"{mul_HW_name}/output_0"
        self.make_node("Mul", inputs=[H_scalar, W_scalar], outputs=[mul_HW_output], name=mul_HW_name)
        self.make_value(mul_HW_output, ir.DataType.INT64, [])

        unsqueeze_HW_name = f"{basename}/unsqueeze_HW"
        self.make_unsqueeze(unsqueeze_HW_name,
                            [mul_HW_output, "/model/constants/INT64/[0]"],
                            ir.DataType.INT64, [1])

        reshape_row_ids_name = f"{basename}/reshape_row_ids"
        self.make_reshape(reshape_row_ids_name,
                          [f"{expand_H_name}/output_0", f"{unsqueeze_HW_name}/output_0"],
                          ir.DataType.INT64, ["total_patches"])
        row_ids = f"{reshape_row_ids_name}/output_0"

        # ── Step 5: Tile col IDs to [H*W]: [0,1,...,W-1, 0,1,...,W-1, ...] ──
        # Implemented as: Reshape([1,W]) → Expand([H,W]) → Reshape([H*W])
        reshape_W_name = f"{basename}/reshape_W_row"
        self.make_reshape(reshape_W_name,
                          [range_W_output, "/model/constants/INT64/[1, -1]"],
                          ir.DataType.INT64, [1, "W"])

        expand_W_name = f"{basename}/expand_W"
        self.make_expand(expand_W_name,
                         [f"{reshape_W_name}/output_0", f"{expand_shape_HW_name}/output_0"],
                         ir.DataType.INT64, ["H", "W"])

        reshape_col_ids_name = f"{basename}/reshape_col_ids"
        self.make_reshape(reshape_col_ids_name,
                          [f"{expand_W_name}/output_0", f"{unsqueeze_HW_name}/output_0"],
                          ir.DataType.INT64, ["total_patches"])
        col_ids = f"{reshape_col_ids_name}/output_0"

        # ── Step 6: Cast position IDs to float ────────────────────────────
        cast_row_name = f"{basename}/cast_row_ids"
        self.make_cast(cast_row_name, row_ids, ir.DataType.FLOAT, ["total_patches"])
        row_ids_f = f"{cast_row_name}/output_0"

        cast_col_name = f"{basename}/cast_col_ids"
        self.make_cast(cast_col_name, col_ids, ir.DataType.FLOAT, ["total_patches"])
        col_ids_f = f"{cast_col_name}/output_0"

        # ── Step 7: Unsqueeze to [total_patches, 1] for MatMul ────────────
        unsqueeze_row_name = f"{basename}/unsqueeze_row"
        self.make_unsqueeze(unsqueeze_row_name,
                            [row_ids_f, "/model/constants/INT64/[1]"],
                            ir.DataType.FLOAT, ["total_patches", 1])

        unsqueeze_col_name = f"{basename}/unsqueeze_col"
        self.make_unsqueeze(unsqueeze_col_name,
                            [col_ids_f, "/model/constants/INT64/[1]"],
                            ir.DataType.FLOAT, ["total_patches", 1])

        # ── Step 8: Unsqueeze inv_freq to [1, head_dim/2] for MatMul ──────
        unsqueeze_inv_name = f"{basename}/unsqueeze_inv_freq"
        self.make_unsqueeze(unsqueeze_inv_name,
                            [inv_freq_name, "/model/constants/INT64/[0]"],
                            ir.DataType.FLOAT, [1, dim // 2])

        # ── Step 9: freqs_row = row_ids_f @ inv_freq → [total_patches, head_dim/2] ──
        matmul_row_name = f"{basename}/matmul_row_freqs"
        matmul_row_output = f"{matmul_row_name}/output_0"
        self.make_node("MatMul",
                       inputs=[f"{unsqueeze_row_name}/output_0", f"{unsqueeze_inv_name}/output_0"],
                       outputs=[matmul_row_output],
                       name=matmul_row_name)
        self.make_value(matmul_row_output, ir.DataType.FLOAT, ["total_patches", dim // 2])

        # ── Step 10: freqs_col = col_ids_f @ inv_freq → [total_patches, head_dim/2] ──
        matmul_col_name = f"{basename}/matmul_col_freqs"
        matmul_col_output = f"{matmul_col_name}/output_0"
        self.make_node("MatMul",
                       inputs=[f"{unsqueeze_col_name}/output_0", f"{unsqueeze_inv_name}/output_0"],
                       outputs=[matmul_col_output],
                       name=matmul_col_name)
        self.make_value(matmul_col_output, ir.DataType.FLOAT, ["total_patches", dim // 2])

        # ── Step 11: Concat [freqs_row, freqs_col] → [total_patches, head_dim] ──
        # Qwen-VL interleaves row and col freqs: [row_half, col_half]
        concat_freqs_name = f"{basename}/concat_freqs"
        concat_freqs_output = f"{concat_freqs_name}/output_0"
        self.make_concat(concat_freqs_name,
                         [matmul_row_output, matmul_col_output],
                         ir.DataType.FLOAT, ["total_patches", dim], axis=-1)

        # ── Step 12: cos and sin caches ────────────────────────────────────
        self.make_cos(f"{basename}/cos", concat_freqs_output, ir.DataType.FLOAT, ["total_patches", dim])
        self.make_sin(f"{basename}/sin", concat_freqs_output, ir.DataType.FLOAT, ["total_patches", dim])

        cos_output_raw = f"{basename}/cos/output_0"
        sin_output_raw = f"{basename}/sin/output_0"

        # Cast to io_dtype for use in attention
        if self.io_dtype != ir.DataType.FLOAT:
            cast_cos_name = f"{basename}/cast_cos"
            self.make_cast(cast_cos_name, cos_output_raw, self.io_dtype, ["total_patches", dim])
            cast_sin_name = f"{basename}/cast_sin"
            self.make_cast(cast_sin_name, sin_output_raw, self.io_dtype, ["total_patches", dim])
            cos_final = f"{cast_cos_name}/output_0"
            sin_final = f"{cast_sin_name}/output_0"
        else:
            cos_final = cos_output_raw
            sin_final = sin_output_raw

        # Register final output names in values dict so callers can reference them
        self.make_value(cos_output, self.io_dtype, ["total_patches", dim])
        self.make_value(sin_output, self.io_dtype, ["total_patches", dim])

        # Add Identity nodes to give them the canonical names
        self.make_node("Identity", inputs=[cos_final], outputs=[cos_output], name=f"{basename}/cos_identity")
        self.make_node("Identity", inputs=[sin_final], outputs=[sin_output], name=f"{basename}/sin_identity")

        return cos_output, sin_output

    # ── Layer override ─────────────────────────────────────────────────────

    def make_layer(self, layer_id, layer):
        """
        Qwen-VL encoder layer:
            layernorm1 → attention (with 2D RoPE on Q/K) → layernorm2 → SwiGLU MLP

        The 2D RoPE is applied to Q and K after the projection MatMuls.
        """
        # Detect sub-modules (Qwen-VL uses norm1/norm2 naming)
        ln1 = getattr(layer, "norm1", None) or getattr(layer, "layernorm_before", None)
        attn = getattr(layer, "attn", None) or getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        ln2 = getattr(layer, "norm2", None) or getattr(layer, "layernorm_after", None)
        mlp = getattr(layer, "mlp", None) or getattr(layer, "intermediate", None)

        if ln1 is None or attn is None or ln2 is None or mlp is None:
            raise AttributeError(
                f"Cannot find expected sub-modules in Qwen-VL layer {layer_id} ({type(layer).__name__}). "
                f"Found: {[n for n, _ in layer.named_children()]}"
            )

        # 1. Pre-attention LayerNorm
        self.make_layernorm(layer_id, ln1,
                            skip=not self.layernorm_attrs["first_layernorm"],
                            simple=self.layernorm_attrs["simple"],
                            location="input")

        # 2. Build 2D RoPE cos/sin caches from image_grid_thw
        cos_cache, sin_cache = self.make_2d_rope_caches(layer_id)

        # 3. Attention (with 2D RoPE applied to Q and K)
        self.make_attention(layer_id, attn, root_input=self.layernorm_attrs["output_0"],
                            cos_cache=cos_cache, sin_cache=sin_cache)

        # 3. Post-attention LayerNorm
        self.make_layernorm(layer_id, ln2,
                            skip=True,
                            simple=self.layernorm_attrs["simple"],
                            location="post_attention")

        # 4. SwiGLU MLP
        self.make_mlp(layer_id, mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    # ── Attention override: Qwen-VL uses qkv_proj (packed) ────────────────

    def _make_attention_qkv(self, layer_id, attention, root_input):
        """
        Qwen-VL attention uses a single qkv_proj Linear layer.
        Override to handle this packed projection.
        """
        qkv_proj = getattr(attention, "qkv", None) or getattr(attention, "qkv_proj", None)

        if qkv_proj is not None:
            # Single packed QKV projection
            qkv_name = f"/vision_model/encoder/layers.{layer_id}/attn/qkv_proj/MatMul"
            q_size = self.num_attn_heads * self.head_size
            kv_size = self.num_kv_heads * self.head_size

            weight_name = qkv_name[1:].replace("/", ".") + ".weight"
            self.make_initializer(qkv_proj.weight.T, weight_name, to=self.io_dtype)

            output = f"{qkv_name}/output_0"
            self.make_node("MatMul", inputs=[root_input, weight_name], outputs=[output], name=qkv_name)
            self.make_value(output, self.io_dtype,
                            shape=["total_patches", q_size + kv_size + kv_size])

            if qkv_proj.bias is not None and torch.count_nonzero(qkv_proj.bias) > 0:
                bias_name = qkv_name[1:].replace("/", ".") + ".bias"
                self.make_initializer(qkv_proj.bias, bias_name, to=self.io_dtype)
                add_name = f"{qkv_name}/Add"
                add_output = f"{add_name}/output_0"
                self.make_node("Add", inputs=[output, bias_name], outputs=[add_output], name=add_name)
                self.make_value(add_output, self.io_dtype,
                                shape=["total_patches", q_size + kv_size + kv_size])
                qkv_output = add_output
            else:
                qkv_output = output

            # Split QKV into separate Q, K, V tensors for 2D RoPE application
            split_name = f"/vision_model/encoder/layers.{layer_id}/attn/qkv_split/Split"
            q_out = f"{split_name}/output_0"
            k_out = f"{split_name}/output_1"
            v_out = f"{split_name}/output_2"
            self.make_node(
                "Split",
                inputs=[qkv_output, f"/model/constants/INT64/[{q_size}, {kv_size}, {kv_size}]"],
                outputs=[q_out, k_out, v_out],
                name=split_name,
                axis=-1,
            )
            self.make_value(q_out, self.io_dtype, shape=["total_patches", q_size])
            self.make_value(k_out, self.io_dtype, shape=["total_patches", kv_size])
            self.make_value(v_out, self.io_dtype, shape=["total_patches", kv_size])

            self.attention_attrs["q_path"] = q_out
            self.attention_attrs["k_path"] = k_out
            self.attention_attrs["v_path"] = v_out
        else:
            # Fall back to separate Q, K, V projections
            super()._make_attention_qkv(layer_id, attention, root_input)

    def _apply_rope_to_qk(self, layer_id, q_path, k_path, cos_cache, sin_cache):
        """
        Apply 2D RoPE to Q and K tensors.

        Qwen-VL RoPE formula (standard rotary embedding):
            q_rot = q * cos + rotate_half(q) * sin

        where rotate_half(x) = [-x[..., head_dim/2:], x[..., :head_dim/2]]

        We implement this using the com.microsoft.RotaryEmbedding op which
        takes (input, position_ids, cos_cache, sin_cache) and applies RoPE.
        Since we have pre-computed cos/sin per patch, we pass them directly.

        The RotaryEmbedding op expects:
          - input: [batch, seq, num_heads * head_dim] or [batch, num_heads, seq, head_dim]
          - cos_cache: [max_seq, head_dim/2] or [seq, head_dim]
          - sin_cache: [max_seq, head_dim/2] or [seq, head_dim]

        For Qwen-VL vision, Q/K are [total_patches, q_size/kv_size] (2D, no batch).
        We need to reshape to [1, total_patches, q_size] for the RotaryEmbedding op.
        """
        q_size = self.num_attn_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        basename = f"/vision_model/encoder/layers.{layer_id}/attn/rope"

        def apply_rope(path, size, num_heads, name_suffix):
            # Reshape [total_patches, size] → [1, total_patches, size]
            unsq_name = f"{basename}/unsqueeze_{name_suffix}"
            self.make_unsqueeze(unsq_name,
                                [path, "/model/constants/INT64/[0]"],
                                self.io_dtype, [1, "total_patches", size])
            unsq_out = f"{unsq_name}/output_0"

            # Apply RotaryEmbedding
            rope_name = f"{basename}/RotaryEmbedding_{name_suffix}"
            rope_output = f"{rope_name}/output_0"
            self.make_node(
                "RotaryEmbedding",
                inputs=[unsq_out, "", cos_cache, sin_cache],
                outputs=[rope_output],
                name=rope_name,
                domain="com.microsoft",
                interleaved=0,
                num_heads=num_heads,
            )
            self.make_value(rope_output, self.io_dtype, [1, "total_patches", size])

            # Squeeze back: [1, total_patches, size] → [total_patches, size]
            sq_name = f"{basename}/squeeze_{name_suffix}"
            self.make_squeeze(sq_name,
                              [rope_output, "/model/constants/INT64/[0]"],
                              self.io_dtype, ["total_patches", size])
            return f"{sq_name}/output_0"

        q_rope = apply_rope(q_path, q_size, self.num_attn_heads, "Q")
        k_rope = apply_rope(k_path, kv_size, self.num_kv_heads, "K")
        return q_rope, k_rope

    def _make_attention_op(self, layer_id, attention, cos_cache=None, sin_cache=None):
        """
        Qwen-VL uses full self-attention over all patches.
        Sequence dimension is "total_patches" (not "num_image_tokens").
        If cos_cache and sin_cache are provided, applies 2D RoPE to Q and K.
        """
        q_path = self.attention_attrs["q_path"]
        k_path = self.attention_attrs["k_path"]

        # Apply 2D RoPE to Q and K if caches are available
        if cos_cache is not None and sin_cache is not None:
            q_path, k_path = self._apply_rope_to_qk(
                layer_id, q_path, k_path, cos_cache, sin_cache
            )

        attn_name = f"/vision_model/encoder/layers.{layer_id}/attn/MultiHeadAttention"
        inputs = [
            q_path,
            k_path,
            self.attention_attrs["v_path"],
            "",   # bias
            "",   # attn_mask
            "",   # add_qk
            "",   # past_k
            "",   # past_v
        ]
        output = f"{attn_name}/output_0"
        outputs = [output, "", ""]

        self.make_node(
            "MultiHeadAttention",
            inputs=inputs,
            outputs=outputs,
            name=attn_name,
            domain="com.microsoft",
            num_heads=self.num_attn_heads,
            scale=self.attention_attrs["scale"],
            unidirectional=0,
        )
        self.make_value(output, self.io_dtype,
                        shape=["total_patches", self.num_attn_heads * self.head_size])
        self.attention_attrs["attn_output"] = output

    def _make_attention_output(self, layer_id, attention):
        """Qwen-VL output projection."""
        out_proj = (
            getattr(attention, "proj", None)
            or getattr(attention, "out_proj", None)
            or getattr(attention, "o_proj", None)
        )
        if out_proj is None:
            raise AttributeError(f"Cannot find output projection in Qwen-VL attention layer {layer_id}")

        o_name = f"/vision_model/encoder/layers.{layer_id}/attn/out_proj/MatMul"
        weight_name = o_name[1:].replace("/", ".") + ".weight"
        self.make_initializer(out_proj.weight.T, weight_name, to=self.io_dtype)

        output = f"{o_name}/output_0"
        self.make_node("MatMul",
                       inputs=[self.attention_attrs["attn_output"], weight_name],
                       outputs=[output], name=o_name)
        self.make_value(output, self.io_dtype, shape=["total_patches", self.hidden_size])

        if out_proj.bias is not None and torch.count_nonzero(out_proj.bias) > 0:
            bias_name = o_name[1:].replace("/", ".") + ".bias"
            self.make_initializer(out_proj.bias, bias_name, to=self.io_dtype)
            add_name = f"{o_name}/Add"
            add_output = f"{add_name}/output_0"
            self.make_node("Add", inputs=[output, bias_name], outputs=[add_output], name=add_name)
            self.make_value(add_output, self.io_dtype, shape=["total_patches", self.hidden_size])
            self.layernorm_attrs["skip_input"] = add_output
        else:
            self.layernorm_attrs["skip_input"] = output

    # ── LayerNorm override: Qwen-VL uses 1D sequence (total_patches) ──────

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        """
        Override to use "total_patches" as the sequence dimension
        instead of "num_image_tokens".
        """
        root_input = self.layernorm_attrs["root_input"]
        skip_input = self.layernorm_attrs["skip_input"]

        weight_name = f"vision_model.encoder.layers.{layer_id}.{location}_layernorm.weight"
        bias_name = f"vision_model.encoder.layers.{layer_id}.{location}_layernorm.bias"
        self.make_initializer(layernorm.weight, weight_name, to=self.io_dtype)
        if not simple and hasattr(layernorm, "bias") and layernorm.bias is not None:
            self.make_initializer(layernorm.bias, bias_name, to=self.io_dtype)

        if skip:
            inputs = [root_input, skip_input, weight_name]
        else:
            inputs = [root_input, weight_name]
        if not simple and hasattr(layernorm, "bias") and layernorm.bias is not None:
            inputs.append(bias_name)

        op_type = f"{'Skip' if skip else ''}{'Simplified' if simple else ''}LayerNormalization"
        name = f"/vision_model/encoder/layers.{layer_id}/{location}_layernorm/{'Skip' if skip else ''}LayerNorm"
        kwargs = {"epsilon": self.layernorm_attrs["epsilon"]}
        if not skip:
            kwargs.update({"axis": -1, "stash_type": 1})

        output_0 = f"/vision_model/encoder/layers.{layer_id}/{location}_layernorm/output_0"
        output_3 = f"/vision_model/encoder/layers.{layer_id}/{location}_layernorm/output_3"
        outputs = [output_0, "", "", output_3] if skip and not self.layernorm_attrs["last_layernorm"] else [output_0]

        self.make_node(op_type, inputs=inputs, outputs=outputs, name=name,
                       domain=("com.microsoft" if skip else None), **kwargs)
        # Use "total_patches" as sequence dim for Qwen-VL
        self.make_value(output_0, self.io_dtype, shape=["total_patches", self.hidden_size])
        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.make_value(output_3, self.io_dtype, shape=["total_patches", self.hidden_size])

        self.layernorm_attrs["output_0"] = output_0
        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.layernorm_attrs["output_3"] = output_3
            self.layernorm_attrs["root_input"] = output_3

    # ── MLP override: Qwen-VL uses "total_patches" sequence dim ───────────

    def _make_mlp_proj(self, layer_id, mlp, root_input):
        """SwiGLU MLP with total_patches sequence dimension."""
        gate_proj = getattr(mlp, "gate_proj", None) or getattr(mlp, "w1", None)
        up_proj = getattr(mlp, "up_proj", None) or getattr(mlp, "w3", None)
        down_proj = getattr(mlp, "down_proj", None) or getattr(mlp, "w2", None)

        if gate_proj is None or up_proj is None or down_proj is None:
            raise AttributeError(f"Cannot find gate/up/down proj in Qwen-VL MLP layer {layer_id}")

        def _matmul(proj, basename):
            weight_name = basename[1:].replace("/", ".") + ".weight"
            self.make_initializer(proj.weight.T, weight_name, to=self.io_dtype)
            output = f"{basename}/output_0"
            self.make_node("MatMul", inputs=[root_input, weight_name], outputs=[output], name=basename)
            self.make_value(output, self.io_dtype, shape=["total_patches", self.intermediate_size])
            if proj.bias is not None and torch.count_nonzero(proj.bias) > 0:
                bias_name = basename[1:].replace("/", ".") + ".bias"
                self.make_initializer(proj.bias, bias_name, to=self.io_dtype)
                add_name = f"{basename}/Add"
                add_output = f"{add_name}/output_0"
                self.make_node("Add", inputs=[output, bias_name], outputs=[add_output], name=add_name)
                self.make_value(add_output, self.io_dtype, shape=["total_patches", self.intermediate_size])
                return add_name
            return basename

        gate_name = _matmul(gate_proj, f"/vision_model/encoder/layers.{layer_id}/mlp/gate_proj/MatMul")
        up_name = _matmul(up_proj, f"/vision_model/encoder/layers.{layer_id}/mlp/up_proj/MatMul")

        # SiLU activation on gate
        act_name = f"/vision_model/encoder/layers.{layer_id}/mlp/act_fn/Silu"
        act_output = f"{act_name}/output_0"
        self.make_node("Silu", inputs=[f"{gate_name}/output_0"], outputs=[act_output],
                       name=act_name, domain="com.microsoft")
        self.make_value(act_output, self.io_dtype, shape=["total_patches", self.intermediate_size])

        # Mul(act, up)
        mul_name = f"/vision_model/encoder/layers.{layer_id}/mlp/Mul"
        mul_output = f"{mul_name}/output_0"
        self.make_node("Mul", inputs=[act_output, f"{up_name}/output_0"], outputs=[mul_output], name=mul_name)
        self.make_value(mul_output, self.io_dtype, shape=["total_patches", self.intermediate_size])

        # Down projection
        down_weight_name = f"vision_model.encoder.layers.{layer_id}.mlp.down_proj.weight"
        self.make_initializer(down_proj.weight.T, down_weight_name, to=self.io_dtype)
        down_out_name = f"/vision_model/encoder/layers.{layer_id}/mlp/down_proj/MatMul"
        down_output = f"{down_out_name}/output_0"
        self.make_node("MatMul", inputs=[mul_output, down_weight_name], outputs=[down_output], name=down_out_name)
        self.make_value(down_output, self.io_dtype, shape=["total_patches", self.hidden_size])

        if down_proj.bias is not None and torch.count_nonzero(down_proj.bias) > 0:
            bias_name = f"vision_model.encoder.layers.{layer_id}.mlp.down_proj.bias"
            self.make_initializer(down_proj.bias, bias_name, to=self.io_dtype)
            add_name = f"{down_out_name}/Add"
            add_output = f"{add_name}/output_0"
            self.make_node("Add", inputs=[down_output, bias_name], outputs=[add_output], name=add_name)
            self.make_value(add_output, self.io_dtype, shape=["total_patches", self.hidden_size])
            self.layernorm_attrs["skip_input"] = add_output
        else:
            self.layernorm_attrs["skip_input"] = down_output

    # ── Final norm override ────────────────────────────────────────────────

    def make_final_norm(self, norm_module):
        """Final LayerNorm with total_patches sequence dimension."""
        weight_name = "vision_model.post_layernorm.weight"
        bias_name = "vision_model.post_layernorm.bias"
        self.make_initializer(norm_module.weight, weight_name, to=self.io_dtype)

        root_input = self.layernorm_attrs["root_input"]
        skip_input = self.layernorm_attrs["skip_input"]

        inputs = [root_input, skip_input, weight_name]
        if hasattr(norm_module, "bias") and norm_module.bias is not None:
            self.make_initializer(norm_module.bias, bias_name, to=self.io_dtype)
            inputs.append(bias_name)

        name = "/vision_model/post_layernorm/SkipLayerNorm"
        output_0 = "/vision_model/post_layernorm/output_0"
        self.make_node(
            "SkipLayerNormalization",
            inputs=inputs,
            outputs=[output_0],
            name=name,
            domain="com.microsoft",
            epsilon=self.layernorm_attrs["epsilon"],
        )
        self.make_value(output_0, self.io_dtype, shape=["total_patches", self.hidden_size])
        self.layernorm_attrs["output_0"] = output_0
        self.layernorm_attrs["root_input"] = output_0

    # ── Patch merger ───────────────────────────────────────────────────────

    def make_patch_merger(self, merger_module):
        """
        Qwen-VL patch merger: spatial downsampling + linear projection.

        The merger reduces the number of patches by spatial_merge_size^2
        (default: 4x reduction) and projects to hidden_size.

        Structure:
            [total_patches, hidden_size]
                ↓ Reshape to [total_patches / merge_size^2, hidden_size * merge_size^2]
                ↓ LayerNorm
                ↓ Linear (mlp[0])
                ↓ GELU
                ↓ Linear (mlp[2])
            [num_image_tokens, hidden_size]  → "image_features"
        """
        basename = "/vision_model/patch_merger"
        root_input = self.layernorm_attrs["root_input"]

        # The merger MLP is typically: [Linear, GELU, Linear]
        # or accessed as merger.mlp
        mlp = getattr(merger_module, "mlp", merger_module)

        # Get the linear layers
        if hasattr(mlp, "__getitem__"):
            # Sequential: mlp[0], mlp[2]
            fc1 = mlp[0]
            fc2 = mlp[2] if len(mlp) > 2 else None
        else:
            fc1 = getattr(mlp, "fc1", None) or getattr(mlp, "linear_1", None)
            fc2 = getattr(mlp, "fc2", None) or getattr(mlp, "linear_2", None)

        # LayerNorm before merger (if present)
        ln = getattr(merger_module, "ln_q", None) or getattr(merger_module, "norm", None)
        if ln is not None:
            ln_weight_name = "vision_model.patch_merger.ln_q.weight"
            self.make_initializer(ln.weight, ln_weight_name, to=self.io_dtype)
            ln_inputs = [root_input, ln_weight_name]
            if hasattr(ln, "bias") and ln.bias is not None:
                ln_bias_name = "vision_model.patch_merger.ln_q.bias"
                self.make_initializer(ln.bias, ln_bias_name, to=self.io_dtype)
                ln_inputs.append(ln_bias_name)
            ln_name = f"{basename}/LayerNorm"
            ln_output = f"{ln_name}/output_0"
            self.make_node(
                "LayerNormalization",
                inputs=ln_inputs,
                outputs=[ln_output],
                name=ln_name,
                axis=-1,
                epsilon=1e-6,
                stash_type=1,
            )
            self.make_value(ln_output, self.io_dtype, shape=["total_patches", self.hidden_size])
            root_input = ln_output

        # FC1
        if fc1 is not None:
            fc1_weight_name = "vision_model.patch_merger.mlp.0.weight"
            self.make_initializer(fc1.weight.T, fc1_weight_name, to=self.io_dtype)
            fc1_out_dim = fc1.weight.shape[0]
            fc1_name = f"{basename}/fc1/MatMul"
            fc1_output = f"{fc1_name}/output_0"
            self.make_node("MatMul", inputs=[root_input, fc1_weight_name], outputs=[fc1_output], name=fc1_name)
            self.make_value(fc1_output, self.io_dtype, shape=["num_image_tokens", fc1_out_dim])

            if fc1.bias is not None and torch.count_nonzero(fc1.bias) > 0:
                fc1_bias_name = "vision_model.patch_merger.mlp.0.bias"
                self.make_initializer(fc1.bias, fc1_bias_name, to=self.io_dtype)
                add_name = f"{basename}/fc1/Add"
                add_output = f"{add_name}/output_0"
                self.make_node("Add", inputs=[fc1_output, fc1_bias_name], outputs=[add_output], name=add_name)
                self.make_value(add_output, self.io_dtype, shape=["num_image_tokens", fc1_out_dim])
                fc1_output = add_output

            # GELU activation
            gelu_name = f"{basename}/gelu"
            gelu_output = f"{gelu_name}/output_0"
            self.make_node("Gelu", inputs=[fc1_output], outputs=[gelu_output], name=gelu_name, approximate="none")
            self.make_value(gelu_output, self.io_dtype, shape=["num_image_tokens", fc1_out_dim])
            root_input = gelu_output

        # FC2
        if fc2 is not None:
            fc2_weight_name = "vision_model.patch_merger.mlp.2.weight"
            self.make_initializer(fc2.weight.T, fc2_weight_name, to=self.io_dtype)
            fc2_name = f"{basename}/fc2/MatMul"
            self.make_node("MatMul", inputs=[root_input, fc2_weight_name], outputs=["image_features"], name=fc2_name)
            self.make_value("image_features", self.io_dtype, shape=["num_image_tokens", self.hidden_size])

            if fc2.bias is not None and torch.count_nonzero(fc2.bias) > 0:
                # Need intermediate output then add bias
                fc2_output = f"{fc2_name}/output_0"
                self.make_node("MatMul", inputs=[root_input, fc2_weight_name], outputs=[fc2_output], name=fc2_name)
                self.make_value(fc2_output, self.io_dtype, shape=["num_image_tokens", self.hidden_size])
                fc2_bias_name = "vision_model.patch_merger.mlp.2.bias"
                self.make_initializer(fc2.bias, fc2_bias_name, to=self.io_dtype)
                self.make_node("Add", inputs=[fc2_output, fc2_bias_name], outputs=["image_features"],
                               name=f"{basename}/fc2/Add")
                self.make_value("image_features", self.io_dtype, shape=["num_image_tokens", self.hidden_size])
        else:
            # No FC2 — connect directly to image_features
            identity_name = f"{basename}/Identity"
            self.make_node("Identity", inputs=[root_input], outputs=["image_features"], name=identity_name)
            self.make_value("image_features", self.io_dtype, shape=["num_image_tokens", self.hidden_size])

    # ── Module detection overrides ─────────────────────────────────────────

    def is_patch_embedding(self, module) -> bool:
        name = module.__class__.__name__
        return name in {
            "Qwen2VisionPatchEmbed",
            "Qwen2_5_VLPatchEmbed",
            "Qwen3VLPatchEmbed",
            "VisionPatchEmbed",
            "PatchEmbed",
        }

    def is_layer(self, module) -> bool:
        name = module.__class__.__name__
        return name in {
            "Qwen2VisionBlock",
            "Qwen2_5_VLVisionBlock",
            "Qwen3VLVisionBlock",
            "VisionBlock",
        }

    def has_final_norm(self, module, orig_model) -> bool:
        # Qwen-VL: visual.blocks[-1] is the last block, then merger
        # The final norm is inside the merger (ln_q)
        return False  # Handled in make_patch_merger

    def has_patch_merger(self, module, orig_model) -> bool:
        """Detect the patch merger module."""
        name = module.__class__.__name__
        return name in {
            "Qwen2VisionPatchMerger",
            "Qwen2_5_VLPatchMerger",
            "Qwen3VLPatchMerger",
            "PatchMerger",
        }

    # ── make_model override ────────────────────────────────────────────────

    def make_model(self, input_path):
        """
        Override to handle Qwen-VL specific module structure:
          - patch_embed → N × encoder blocks → patch_merger

        Qwen-VL visual module children order:
          patch_embed, pos_embed, rotary_pos_emb, blocks, merger, deepstack_merger_list

        We process patch_embed explicitly first (by direct attribute access) to
        guarantee it runs before the encoder blocks, regardless of modules() order.
        """
        self.make_inputs_and_outputs()
        self.weights = self.load_vision_weights(input_path)

        # Extract the visual sub-model
        visual = (
            getattr(self.weights, "visual", None)
            or getattr(self.weights, "vision_model", None)
            or self.weights
        )

        # ── Step 1: Process patch embedding explicitly ─────────────────────
        # Access patch_embed directly rather than relying on modules() order,
        # because blocks may appear before patch_embed in depth-first traversal.
        patch_embed = getattr(visual, "patch_embed", None)
        if patch_embed is not None:
            print("Reading Qwen-VL patch embedding")
            self.make_patch_embedding(patch_embed)
        else:
            raise AttributeError(
                f"Cannot find patch_embed in Qwen-VL visual model: {type(visual).__name__}. "
                f"Children: {[n for n, _ in visual.named_children()]}"
            )

        # ── Step 2: Process encoder blocks ────────────────────────────────
        self.layer_id = 0
        for module in visual.modules():
            if self.is_layer(module) and self.layer_id < self.num_layers:
                print(f"Reading Qwen-VL vision encoder layer {self.layer_id}")
                self.make_layer(self.layer_id, module)
                self.layer_id += 1

            elif self.has_patch_merger(module, visual):
                print("Reading Qwen-VL patch merger")
                self.make_patch_merger(module)

        # If patch merger was not found, connect final output to image_features
        if "image_features" not in self.node_names:
            final_output = self.layernorm_attrs["root_input"]
            identity_name = "/vision_model/output/Identity"
            self.make_node("Identity", inputs=[final_output], outputs=["image_features"],
                           name=identity_name)
            self.make_value("image_features", self.io_dtype,
                            shape=["total_patches", self.hidden_size])

        del self.weights

    # ── genai_config override ──────────────────────────────────────────────

    def update_genai_config(self, vision_section: dict):
        """Add Qwen-VL specific fields to the vision config section."""
        vision_section["spatial_merge_size"] = self.spatial_merge_size
        vision_section["temporal_patch_size"] = self.temporal_patch_size


class Qwen3VLVisionModel(Qwen25VLVisionModel):
    """
    Qwen3-VL vision encoder (Qwen/Qwen3-VL-*).

    Inherits Qwen25VLVisionModel and adds:
      - QK-norm (RMSNorm after Q and K projections)
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Qwen3-VL adds QK-norm
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True

    def is_patch_embedding(self, module) -> bool:
        name = module.__class__.__name__
        return name in {
            "Qwen3VLPatchEmbed",
            "Qwen2VisionPatchEmbed",
            "Qwen2_5_VLPatchEmbed",
            "VisionPatchEmbed",
            "PatchEmbed",
        }

    def is_layer(self, module) -> bool:
        name = module.__class__.__name__
        return name in {
            "Qwen3VLVisionBlock",
            "Qwen2VisionBlock",
            "Qwen2_5_VLVisionBlock",
            "VisionBlock",
        }

    def has_patch_merger(self, module, orig_model) -> bool:
        name = module.__class__.__name__
        return name in {
            "Qwen3VLPatchMerger",
            "Qwen2VisionPatchMerger",
            "Qwen2_5_VLPatchMerger",
            "PatchMerger",
        }
