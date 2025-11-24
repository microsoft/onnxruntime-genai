# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


import onnx_ir as ir
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from .base import Model


class QwenModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Qwen3Model(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()


class Qwen25VLTextModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # The HF model (Qwen2RMSNorm) *always* computes LayerNorm in float32.
        # By inheriting from `base.Model`, all `layernorm_attrs["cast"]` flags
        # are `False`. This causes parity loss and type mismatch error.
        #
        # SOLUTION: Manually set all `cast` flags to `True`. This forces the
        # builder to cast bf16 inputs -> fp32, compute LN, and cast fp32
        # outputs -> bf16, matching the HF model and fixing both errors.
        #
        print("Forcing LayerNorm computation to float32 (and enabling all casts) for Qwen2.5-VL parity.")
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = True
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = True

        # Qwen2's RoPE *always* computes in float32.
        # We must replicate this behavior.
        print("Forcing RoPE computation to float32 for Qwen2.5-VL parity.")
        if "rope_cast" not in self.attention_attrs:
            self.attention_attrs["rope_cast"] = {}
        self.attention_attrs["rope_cast"]["use_fp32"] = True

        # Check rope type since huggingface model supports yarn but that is not recommended as mentioned in model card. Example:
        #    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24,24]}
        if config.rope_scaling and "type" in config.rope_scaling:
            assert config.rope_scaling["type"] == "mrope"

        # Qwen 2.5 VL applies RoPE manually before attention, not fused in the op
        self.attention_attrs["use_rope_in_attn"] = False

        if "position_ids" not in self.input_names:
            print("Re-adding 'position_ids' to self.input_names.")
            if "attention_mask" in self.input_names:
                idx = self.input_names.index("attention_mask")
                self.input_names.insert(idx + 1, "position_ids")
            else:
                self.input_names.append("position_ids")

        self.mrope_sections = self.rope_attrs.get("mrope", {}).get("sections", [])
        if not self.mrope_sections:
            raise ValueError("MRoPE sections not found in config.text_config.rope_scaling.mrope_section")

        # The HF logic is `mrope_section * 2`, not `[s * 2 for s in mrope_section]`.
        # This results in [16, 24, 24, 16, 24, 24]
        self.mrope_splits = self.mrope_sections * 2

        if sum(self.mrope_splits) != self.head_size:
            # The sum (128) should now correctly match self.head_size (128)
            raise ValueError(
                f"MRoPE splits {self.mrope_splits} sum ({sum(self.mrope_splits)}) does not match head size ({self.head_size})"
            )

        # Force GroupQueryAttention since make_attention() below only implements GQA.
        self.attention_attrs["op_type"] = "GroupQueryAttention"

        if not self.is_gqa_supported():
            print(f"Warning: {self.ep} does not support GQA for {self.io_dtype}, so GQA might fallback to CPU!")

        # Create and save the inv_freq tensor
        self.make_inv_freq_tensor()

    def make_inv_freq_tensor(self):
        """
        Calculates and saves the `inv_freq` tensor as an initializer.
        This is copied from base.py:make_rotary_embedding_caches_from_scratch
        """
        dim = int(self.rope_attrs["partial_rotary_factor"] * self.head_size)
        inv_freq = 1.0 / (
            self.rope_attrs["rescale_factors"]
            * (self.rope_attrs["theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        )

        # The HF model expects H/2, not R/2
        if dim != self.head_size:
            print(
                f"Warning: partial_rotary_factor ({self.rope_attrs['partial_rotary_factor']}) is not 1. This might be unsupported."
            )
            inv_freq = inv_freq[: (self.head_size // 2)]

        self.make_initializer(inv_freq, "model.inv_freq", to=ir.DataType.FLOAT)
        print("Created and saved 'model.inv_freq' initializer.")

    def make_inputs_and_outputs(self):
        # Qwen2.5-VL uses 3D position_ids
        self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]

        # Call the base Model's make_inputs_and_outputs (skipping MistralModel's)
        super().make_inputs_and_outputs()

    def make_dynamic_rope_caches(self, layer_id, basename):
        # Make nodes for the Dynamic RoPE Cache subgraph
        #
        # Re-implements Qwen2_5_VLRotaryEmbedding.forward using ONNX ops.
        # Takes 3D position_ids and inv_freq and dynamically creates
        # the cos/sin caches.
        #
        #         inv_freq (H/2)                                     position_ids (3, B, S)
        #             |                                                      |
        #         Unsqueeze                                              Unsqueeze
        #             |                                                      |
        #           Expand                                                  Cast
        #      (3, B, H/2, 1)                                           (3, B, 1, S)
        #             |                                                      |
        #             +--------------------------+---------------------------+
        #                                        |
        #                                      MatMul
        #                                   (3, B, H/2, S)
        #                                        |
        #                                    Transpose
        #                                   (3, B, S, H/2)
        #                                        |
        #                                     Concat
        #                                  (3, B, S, H)
        #                                        |
        #                          +-------------+-------------+
        #                          |                           |
        #                         Cos                         Sin
        #                          |                           |
        #                         Mul                         Mul
        #                   (apply scaling)             (apply scaling)
        #
        pos_ids_name = "position_ids"
        inv_freq_name = "model.inv_freq"
        head_dim_half = self.head_size // 2

        # Get Batch Size from position_ids.shape[1]
        shape_pos_ids_name = f"{basename}/pos_ids/Shape"
        shape_pos_ids_output = f"{shape_pos_ids_name}/output_0"
        self.make_shape(shape_pos_ids_name, pos_ids_name, [3])

        gather_batch_size_name = f"{basename}/pos_ids/Gather"
        gather_batch_size_output = f"{gather_batch_size_name}/output_0"
        self.make_gather(
            gather_batch_size_name,
            [shape_pos_ids_output, "/model/constants/INT64/[1]"],
            ir.DataType.INT64,
            [1],
            axis=0,
        )

        # Expand inv_freq: [H/2] -> [1, 1, H/2, 1]
        unsqueeze_1_name = f"{basename}/inv_freq/Unsqueeze"
        unsqueeze_1_output = f"{unsqueeze_1_name}/output_0"
        self.make_unsqueeze(
            unsqueeze_1_name,
            [inv_freq_name, "/model/constants/INT64/[0, 1, 3]"],
            ir.DataType.FLOAT,
            [1, 1, head_dim_half, 1],
        )

        # Create target shape for Expand: [3, B, H/2, 1]
        concat_expand_shape_name = f"{basename}/expand_shape/Concat"
        concat_expand_shape_output = f"{concat_expand_shape_name}/output_0"
        self.make_concat(
            concat_expand_shape_name,
            [
                "/model/constants/INT64/[3]",
                gather_batch_size_output,
                f"/model/constants/INT64/[{head_dim_half}, 1]",
            ],
            ir.DataType.INT64,
            [4],
            axis=0,
        )

        expand_name = f"{basename}/inv_freq/Expand"
        expand_output = f"{expand_name}/output_0"
        self.make_expand(
            expand_name,
            [unsqueeze_1_output, concat_expand_shape_output],
            ir.DataType.FLOAT,
            [3, "batch_size", head_dim_half, 1],
        )

        # Expand position_ids: [3, B, S] -> [3, B, 1, S]
        unsqueeze_2_name = f"{basename}/pos_ids/Unsqueeze"
        unsqueeze_2_output = f"{unsqueeze_2_name}/output_0"
        self.make_unsqueeze(
            unsqueeze_2_name,
            [pos_ids_name, "/model/constants/INT64/[2]"],
            ir.DataType.INT64,
            [3, "batch_size", 1, "sequence_length"],
        )

        # Cast position_ids to float
        cast_name = f"{basename}/pos_ids/Cast"
        cast_output = f"{cast_name}/output_0"
        self.make_cast(
            cast_name,
            unsqueeze_2_output,
            ir.DataType.FLOAT,
            [3, "batch_size", 1, "sequence_length"],
        )

        # MatMul: [3, B, H/2, 1] @ [3, B, 1, S] -> [3, B, H/2, S]
        matmul_name = f"{basename}/freqs/MatMul"
        matmul_output = f"{matmul_name}/output_0"
        self.make_node("MatMul", [expand_output, cast_output], [matmul_output], name=matmul_name)
        self.make_value(
            matmul_output,
            ir.DataType.FLOAT,
            [3, "batch_size", head_dim_half, "sequence_length"],
        )

        # Transpose: [3, B, H/2, S] -> [3, B, S, H/2]
        transpose_name = f"{basename}/freqs/Transpose"
        transpose_output = f"{transpose_name}/output_0"
        self.make_transpose(
            transpose_name,
            matmul_output,
            ir.DataType.FLOAT,
            [3, "batch_size", "sequence_length", head_dim_half],
            perm=[0, 1, 3, 2],
        )

        # Concat (freqs, freqs): [3, B, S, H/2] -> [3, B, S, H]
        concat_name = f"{basename}/Concat"
        concat_output = f"{concat_name}/output_0"
        self.make_concat(
            concat_name,
            [transpose_output, transpose_output],
            ir.DataType.FLOAT,
            [3, "batch_size", "sequence_length", self.head_size],
            axis=-1,
        )

        # Cos(emb) and Sin(emb)
        cos_name = f"{basename}/Cos"
        cos_output = f"{cos_name}/output_0"
        self.make_node("Cos", [concat_output], [cos_output], name=cos_name)
        self.make_value(
            cos_output,
            ir.DataType.FLOAT,
            [3, "batch_size", "sequence_length", self.head_size],
        )

        sin_name = f"{basename}/Sin"
        sin_output = f"{sin_name}/output_0"
        self.make_node("Sin", [concat_output], [sin_output], name=sin_name)
        self.make_value(
            sin_output,
            ir.DataType.FLOAT,
            [3, "batch_size", "sequence_length", self.head_size],
        )

        return cos_output, sin_output

    def rotate_half(self, x_name, x_shape, basename, compute_dtype):
        # Make nodes for rotate_half subgraph
        #
        #       x (B, N, S, H)
        #             |
        #           Split
        #          /     \
        #         /       \
        #    x1 (..., H/2)  x2 (..., H/2)
        #        |          |
        #        |         Neg
        #        |          |
        #        |         -x2
        #        \         /
        #         \       /
        #          Concat
        #            |
        #      output (..., H)

        # Split: [B, N, S, H] -> [B, N, S, H/2], [B, N, S, H/2]
        split_name = f"{basename}/rotate_half/Split"
        split_output_0 = f"{split_name}/output_0"
        split_output_1 = f"{split_name}/output_1"
        self.make_node(
            "Split",
            [x_name],
            [split_output_0, split_output_1],
            name=split_name,
            axis=-1,
            num_outputs=2,
        )
        half_shape = [*x_shape[:-1], x_shape[-1] // 2]
        self.make_value(split_output_0, compute_dtype, half_shape)
        self.make_value(split_output_1, compute_dtype, half_shape)

        # Negate x2
        neg_name = f"{basename}/rotate_half/Neg"
        neg_output = f"{neg_name}/output_0"
        self.make_node("Neg", [split_output_1], [neg_output], name=neg_name)
        self.make_value(neg_output, compute_dtype, half_shape)

        # Concat (-x2, x1)
        concat_name = f"{basename}/rotate_half/Concat"
        concat_output = f"{concat_name}/output_0"
        self.make_concat(concat_name, [neg_output, split_output_0], compute_dtype, x_shape, axis=-1)

        return concat_output

    def apply_mrope_rotation(self, layer_id, q_or_k_path, q_or_k_shape, dyn_cos, dyn_sin, num_heads, basename):
        # Make nodes for the MRoPE rotation subgraph
        #
        # Re-implements apply_multimodal_rotary_pos_emb using ONNX ops.
        # Takes Q/K tensor and the dynamically generated 3D caches
        # and applies the rotation.
        #
        #      dyn_cos (3, B, S, H)   dyn_sin (3, B, S, H)     q_or_k (B, S, N*H)
        #              |                      |                        |
        #            Split                  Split                   Reshape
        #        (into 6 parts)         (into 6 parts)                 |
        #              |                      |                    Transpose
        #      +-------+-------+      +-------+-------+          (B, N, S, H)
        #      |    ...loop... |      |    ...loop... |                |
        #    Gather(dim_idx)   |    Gather(dim_idx)   |                |
        #      |               |      |               |                |
        #   Unsqueeze          |    Unsqueeze         |                |
        #      |               |      |               |                |
        #      +-------+-------+      +-------+-------+                |
        #              |                      |                        |
        #            Concat                 Concat                     |
        #         (B, 1, S, H)           (B, 1, S, H)                  |
        #              |                      |                        |
        #              +-----------+----------+                        |
        #                          |                                   |
        #                  (Mixed Precision Casts)                     |
        #                          |                                   |
        #                          +-----------------------+-----------+
        #                                                  |
        #                                       (q * cos) + (rotate_half(q) * sin)
        #                                                  |
        #                                              Transpose
        #                                                  |
        #                                               Reshape
        #

        # --- Handle precision for RoPE ---
        # Check if we need to force float32 computation
        force_fp32 = self.attention_attrs.get("rope_cast", {}).get("use_fp32", False)

        # Set compute_dtype (precision for math) and output_dtype (final precision)
        compute_dtype = ir.DataType.FLOAT if force_fp32 else self.io_dtype
        output_dtype = self.io_dtype
        # --------------------------------

        # Create a Constant node for mrope_splits
        # This holds the correct splits, e.g., [16, 24, 24, 16, 24, 24]
        mrope_splits_node_name = f"{basename}/mrope_splits/Constant"
        mrope_splits_output_name = f"{basename}/mrope_splits"
        mrope_splits_tensor = ir.tensor(
            torch.tensor(self.mrope_splits, dtype=torch.int64),
            name=mrope_splits_output_name,
        )
        self.make_node(
            "Constant",
            inputs=[],
            outputs=[mrope_splits_output_name],
            name=mrope_splits_node_name,
            value=mrope_splits_tensor,
        )
        self.make_value(mrope_splits_output_name, ir.DataType.INT64, [len(self.mrope_splits)])

        # Split the dynamic caches [3, B, S, H] into 6 chunks on axis -1
        # Caches (dyn_cos, dyn_sin) are already in float32
        num_splits = len(self.mrope_splits)

        cos_split_name = f"{basename}/cos/Split"
        cos_split_outputs = [f"{cos_split_name}/output_{i}" for i in range(num_splits)]
        self.make_node(
            "Split",
            [dyn_cos, mrope_splits_output_name],
            cos_split_outputs,
            name=cos_split_name,
            axis=-1,
        )

        sin_split_name = f"{basename}/sin/Split"
        sin_split_outputs = [f"{sin_split_name}/output_{i}" for i in range(num_splits)]
        self.make_node(
            "Split",
            [dyn_sin, mrope_splits_output_name],
            sin_split_outputs,
            name=sin_split_name,
            axis=-1,
        )

        # Re-order the caches: [T, H, W, T, H, W]
        cos_reordered = []
        sin_reordered = []
        for i in range(num_splits):
            dim_chunk = self.mrope_splits[i]
            cache_dim_to_use = i % 3  # 0 for T, 1 for H, 2 for W

            # Gather from dim 0 of the split cache chunk
            # input is [3, B, S, H_chunk], indices is [0, 1, or 2]
            gather_cos_name = f"{basename}/cos_{i}/Gather"
            gather_cos_output = f"{gather_cos_name}/output_0"
            self.make_node(
                "Gather",
                [cos_split_outputs[i], f"/model/constants/INT64/[{cache_dim_to_use}]"],
                [gather_cos_output],
                name=gather_cos_name,
                axis=0,
            )
            self.make_value(
                gather_cos_output,
                ir.DataType.FLOAT,
                [1, "batch_size", "sequence_length", dim_chunk],
            )  # Shape [1, B, S, H_chunk]

            gather_sin_name = f"{basename}/sin_{i}/Gather"
            gather_sin_output = f"{gather_sin_name}/output_0"
            self.make_node(
                "Gather",
                [sin_split_outputs[i], f"/model/constants/INT64/[{cache_dim_to_use}]"],
                [gather_sin_output],
                name=gather_sin_name,
                axis=0,
            )
            self.make_value(
                gather_sin_output,
                ir.DataType.FLOAT,
                [1, "batch_size", "sequence_length", dim_chunk],
            )  # Shape [1, B, S, H_chunk]

            # FIX: Squeeze the gathered cache to [B, S, H_chunk]
            squeeze_cos_name = f"{basename}/cos_{i}/Squeeze"
            squeeze_cos_output = f"{squeeze_cos_name}/output_0"
            self.make_squeeze(
                squeeze_cos_name,
                [gather_cos_output, "/model/constants/INT64/[0]"],
                ir.DataType.FLOAT,
                ["batch_size", "sequence_length", dim_chunk],
            )

            squeeze_sin_name = f"{basename}/sin_{i}/Squeeze"
            squeeze_sin_output = f"{squeeze_sin_name}/output_0"
            self.make_squeeze(
                squeeze_sin_name,
                [gather_sin_output, "/model/constants/INT64/[0]"],
                ir.DataType.FLOAT,
                ["batch_size", "sequence_length", dim_chunk],
            )

            # Unsqueeze to add the NumHeads dim: [B, 1, S, H_chunk]
            unsqueeze_cos_name = f"{basename}/cos_{i}/Unsqueeze"
            unsqueeze_cos_output = f"{unsqueeze_cos_name}/output_0"
            self.make_unsqueeze(
                unsqueeze_cos_name,
                [squeeze_cos_output, "/model/constants/INT64/[1]"],
                ir.DataType.FLOAT,
                ["batch_size", 1, "sequence_length", dim_chunk],
            )
            cos_reordered.append(unsqueeze_cos_output)

            unsqueeze_sin_name = f"{basename}/sin_{i}/Unsqueeze"
            unsqueeze_sin_output = f"{unsqueeze_sin_name}/output_0"
            self.make_unsqueeze(
                unsqueeze_sin_name,
                [squeeze_sin_output, "/model/constants/INT64/[1]"],
                ir.DataType.FLOAT,
                ["batch_size", 1, "sequence_length", dim_chunk],
            )
            sin_reordered.append(unsqueeze_sin_output)

        # Concat re-ordered chunks back to [B, 1, S, H]
        final_cos_concat_name = f"{basename}/cos_final/Concat"
        final_cos_concat_output = f"{final_cos_concat_name}/output_0"
        self.make_concat(
            final_cos_concat_name,
            cos_reordered,
            ir.DataType.FLOAT,
            ["batch_size", 1, "sequence_length", self.head_size],
            axis=-1,
        )

        final_sin_concat_name = f"{basename}/sin_final/Concat"
        final_sin_concat_output = f"{final_sin_concat_name}/output_0"
        self.make_concat(
            final_sin_concat_name,
            sin_reordered,
            ir.DataType.FLOAT,
            ["batch_size", 1, "sequence_length", self.head_size],
            axis=-1,
        )

        # Caches (final_cos_concat_output, final_sin_concat_output) are now in float32

        # Reshape input Q/K: [B, S, N*H] -> [B, S, N, H]
        reshape_1_name = f"{basename}/q_or_k_bsd_to_bsnh/Reshape"
        reshape_1_output = f"{reshape_1_name}/output_0"
        reshape_1_target_shape_onnx = f"/model/constants/INT64/[0, 0, {num_heads}, {self.head_size}]"
        reshape_1_target_shape_ort = [
            "batch_size",
            "sequence_length",
            num_heads,
            self.head_size,
        ]
        self.make_reshape(
            reshape_1_name,
            [q_or_k_path, reshape_1_target_shape_onnx],
            self.io_dtype,
            reshape_1_target_shape_ort,
        )

        # Transpose Q/K: [B, S, N, H] -> [B, N, S, H]
        transpose_1_name = f"{basename}/q_or_k_bsnh_to_bnsh/Transpose"
        transpose_1_output = f"{transpose_1_name}/output_0"
        transpose_1_target_shape = [
            "batch_size",
            num_heads,
            "sequence_length",
            self.head_size,
        ]
        self.make_transpose(
            transpose_1_name,
            reshape_1_output,
            self.io_dtype,
            transpose_1_target_shape,
            perm=[0, 2, 1, 3],
        )

        # --- Start RoPE computation ---
        q_or_k_compute_input = transpose_1_output
        cos_cache_compute_input = final_cos_concat_output
        sin_cache_compute_input = final_sin_concat_output

        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            # Cast Q/K (self.io_dtype) up to float32
            q_or_k_cast_name = f"{basename}/q_or_k/Cast"
            q_or_k_cast_output = f"{q_or_k_cast_name}/output_0"
            self.make_cast(
                q_or_k_cast_name,
                transpose_1_output,
                compute_dtype,
                transpose_1_target_shape,
            )
            q_or_k_compute_input = q_or_k_cast_output
        elif not force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            # Cast Caches (float32) down to self.io_dtype
            cos_cache_cast_name = f"{basename}/cos_final/Cast"
            cos_cache_cast_output = f"{cos_cache_cast_name}/output_0"
            self.make_cast(
                cos_cache_cast_name,
                final_cos_concat_output,
                compute_dtype,
                ["batch_size", 1, "sequence_length", self.head_size],
            )
            cos_cache_compute_input = cos_cache_cast_output

            sin_cache_cast_name = f"{basename}/sin_final/Cast"
            sin_cache_cast_output = f"{sin_cache_cast_name}/output_0"
            self.make_cast(
                sin_cache_cast_name,
                final_sin_concat_output,
                compute_dtype,
                ["batch_size", 1, "sequence_length", self.head_size],
            )
            sin_cache_compute_input = sin_cache_cast_output

        # Apply rotation: (q * cos) + (rotate_half(q) * sin)

        # 1. (q * cos)
        mul_1_name = f"{basename}/Mul_1"
        mul_1_output = f"{mul_1_name}/output_0"
        self.make_mul(
            mul_1_name,
            [q_or_k_compute_input, cos_cache_compute_input],
            compute_dtype,
            transpose_1_target_shape,
        )

        # 2. rotate_half(q)
        rotated_half_q_name = self.rotate_half(q_or_k_compute_input, transpose_1_target_shape, basename, compute_dtype)

        # 3. (rotate_half(q) * sin)
        mul_2_name = f"{basename}/Mul_2"
        mul_2_output = f"{mul_2_name}/output_0"
        self.make_mul(
            mul_2_name,
            [rotated_half_q_name, sin_cache_compute_input],
            compute_dtype,
            transpose_1_target_shape,
        )

        # 4. (q * cos) + (rotate_half(q) * sin)
        add_name = f"{basename}/add/Add"
        add_output = f"{add_name}/output_0"
        self.make_add(
            add_name,
            [mul_1_output, mul_2_output],
            compute_dtype,
            transpose_1_target_shape,
        )

        # --- End RoPE computation ---

        add_output_final = add_output
        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            # Cast result back down to self.io_dtype
            add_cast_name = f"{basename}/add/Cast"
            add_cast_output = f"{add_cast_name}/output_0"
            self.make_cast(add_cast_name, add_output, output_dtype, transpose_1_target_shape)
            add_output_final = add_cast_output

        # Transpose back: [B, N, S, H] -> [B, S, N, H]
        transpose_2_name = f"{basename}/q_or_k_bnsh_to_bsnh/Transpose"
        transpose_2_output = f"{transpose_2_name}/output_0"
        self.make_transpose(
            transpose_2_name,
            add_output_final,
            output_dtype,
            reshape_1_target_shape_ort,
            perm=[0, 2, 1, 3],
        )

        # Reshape back: [B, S, N, H] -> [B, S, N*H]
        reshape_2_name = f"{basename}/q_or_k_bsnh_to_bsd/Reshape"
        reshape_2_output = f"{reshape_2_name}/output_0"
        self.make_reshape(
            reshape_2_name,
            [
                transpose_2_output,
                f"/model/constants/INT64/[0, 0, {num_heads * self.head_size}]",
            ],
            output_dtype,
            q_or_k_shape,
        )

        return reshape_2_output

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # Make nodes for the Attention subgraph (with MRoPE)
        #
        #               root_input
        #              /    |     \
        #             /     |      \
        #       Q_MatMul K_MatMul V_MatMul
        #          |        |        |
        #        Q_Add    K_Add    V_Add
        #          |        |        |
        #          |        |        +-----------------+
        #          |        |                          |
        #   (make_dynamic_rope_caches)                 |
        #          |                                   |
        #    +-----+-----+                             |
        #    |           |                             |
        # dyn_cos     dyn_sin                          |
        #    |           |                             |
        #    v           v                             |
        # (apply_mrope_rotation for Q)                 |
        #          |                                   |
        #        Q_Rot                                 |
        #          |     (apply_mrope_rotation for K)  |
        #          |                 |                 |
        #          |               K_Rot               |
        #          |                 |                 |
        #          +--------+--------+                 |
        #                   |                          |
        #           GroupQueryAttention <--------------+
        #                   |
        #                O_MatMul
        #                   |
        #                 O_Add

        # 1. Unpack QKV if necessary (e.g. qkv_proj)
        super().make_attention_unpacked(layer_id, attention, root_input, **kwargs)

        # 2. Build Q/K/V MatMul and Add nodes
        q_matmul_basename = f"/model/layers.{layer_id}/attn/q_proj/MatMul"
        q_matmul_name = self.make_matmul(attention.q_proj, q_matmul_basename, root_input)
        self.attention_attrs["q_path"] = f"{q_matmul_name}/output_0"
        q_shape = [
            "batch_size",
            "sequence_length",
            self.num_attn_heads * self.head_size,
        ]

        k_matmul_basename = f"/model/layers.{layer_id}/attn/k_proj/MatMul"
        k_matmul_name = self.make_matmul(attention.k_proj, k_matmul_basename, root_input)
        self.attention_attrs["k_path"] = f"{k_matmul_name}/output_0"
        k_shape = ["batch_size", "sequence_length", self.num_kv_heads * self.head_size]

        v_matmul_basename = f"/model/layers.{layer_id}/attn/v_proj/MatMul"
        v_matmul_name = self.make_matmul(attention.v_proj, v_matmul_basename, root_input)
        self.attention_attrs["v_path"] = f"{v_matmul_name}/output_0"

        # Handle biases
        q_bias_exists = attention.q_proj.bias is not None and torch.count_nonzero(attention.q_proj.bias) > 0
        k_bias_exists = attention.k_proj.bias is not None and torch.count_nonzero(attention.k_proj.bias) > 0
        v_bias_exists = attention.v_proj.bias is not None and torch.count_nonzero(attention.v_proj.bias) > 0

        if q_bias_exists:
            q_add_name = f"/model/layers.{layer_id}/attn/q_proj/Add"
            self.make_add_bias(
                attention.q_proj.bias,
                q_add_name,
                root_input=self.attention_attrs["q_path"],
            )
            self.attention_attrs["q_path"] = f"{q_add_name}/output_0"
        if k_bias_exists:
            k_add_name = f"/model/layers.{layer_id}/attn/k_proj/Add"
            self.make_add_bias(
                attention.k_proj.bias,
                k_add_name,
                root_input=self.attention_attrs["k_path"],
            )
            self.attention_attrs["k_path"] = f"{k_add_name}/output_0"
        if v_bias_exists:
            v_add_name = f"/model/layers.{layer_id}/attn/v_proj/Add"
            self.make_add_bias(
                attention.v_proj.bias,
                v_add_name,
                root_input=self.attention_attrs["v_path"],
            )
            self.attention_attrs["v_path"] = f"{v_add_name}/output_0"

        # 3. Apply 3D RoPE (MRoPE)
        cos_dynamic, sin_dynamic = self.make_dynamic_rope_caches(
            layer_id, basename=f"/model/layers.{layer_id}/attn/mrope_dynamic_cache"
        )

        # Apply rotation to Q
        self.attention_attrs["q_path"] = self.apply_mrope_rotation(
            layer_id,
            self.attention_attrs["q_path"],
            q_shape,
            cos_dynamic,
            sin_dynamic,
            self.num_attn_heads,
            basename=f"/model/layers.{layer_id}/attn/q_mrope",
        )

        # Apply rotation to K
        self.attention_attrs["k_path"] = self.apply_mrope_rotation(
            layer_id,
            self.attention_attrs["k_path"],
            k_shape,
            cos_dynamic,
            sin_dynamic,
            self.num_kv_heads,
            basename=f"/model/layers.{layer_id}/attn/k_mrope",
        )

        # 4. Call GroupQueryAttention op
        past_k = f"past_key_values.{layer_id}.key"
        past_v = f"past_key_values.{layer_id}.value"
        present_k = f"present.{layer_id}.key"
        present_v = f"present.{layer_id}.value"

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
            # Pass empty strings for fused caches since we applied RoPE manually
            cos_cache="",
            sin_cache="",
            **kwargs,
        )

        # 5. Build O-proj
        o_proj = "o_proj" if hasattr(attention, "o_proj") else "dense"
        o_matmul_basename = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        o_weight = getattr(attention, o_proj)
        o_matmul_name = self.make_matmul(o_weight, o_matmul_basename, f"{attn_name}/output_0")

        o_bias_exists = getattr(attention, o_proj).bias is not None
        if o_bias_exists:
            o_add_name = f"/model/layers.{layer_id}/attn/o_proj/Add"
            o_bias = getattr(attention, o_proj).bias
            self.make_add_bias(o_bias, o_add_name, root_input=f"{o_matmul_name}/output_0")
            self.layernorm_attrs["skip_input"] = f"{o_add_name}/output_0"
        else:
            self.layernorm_attrs["skip_input"] = f"{o_matmul_name}/output_0"

    def make_model(self, input_path):
        # Make inputs and outputs to ONNX model
        self.make_inputs_and_outputs()

        # Make pre-processing nodes
        self.make_preprocessing_nodes()

        # Load the Hugging Face model
        print("Loading Qwen2_5_VLForConditionalGeneration model...")
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            config=self.config,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
        )

        # We only want to export the text model
        model = hf_model.language_model
        print(f"Isolated language_model ({model.__class__.__name__}) for ONNX export.")

        # Loop through model and map each module to ONNX/ORT ops
        self.layer_id = 0

        # The base.Model.make_model() loop expects modules from a standard causal LM,
        # so we replicate its logic here but point to the correct modules in the hf_model

        # Handle Embeddings
        if not self.exclude_embeds:
            print("Reading embedding layer")
            # The text model's embeddings are at model.embed_tokens
            self.make_embedding(model.embed_tokens.weight)
        else:
            # When excluding embeds, the input is `inputs_embeds`
            print("Skipping embedding layer, model will expect 'inputs_embeds'.")
            self.layernorm_attrs["root_input"] = "inputs_embeds"
            self.layernorm_attrs["skip_input"] = "inputs_embeds"

        # Handle Decoder Layers
        for layer in model.layers:
            if self.layer_id < self.num_layers:
                print(f"Reading decoder layer {self.layer_id}")
                self.make_layer(self.layer_id, layer)
                self.layer_id += 1

        # Handle Final Norm
        if self.layer_id == self.num_layers and hasattr(model, "norm"):
            print("Reading final norm")
            self.make_layernorm(
                self.layer_id,
                model.norm,
                skip=True,
                simple=self.layernorm_attrs["simple"],
                location="final_norm",
            )

        # Handle LM Head
        if not self.exclude_lm_head:
            # The LM head is part of the parent Qwen2_5_VLForConditionalGeneration model
            print("Reading LM head")
            self.make_lm_head(hf_model.lm_head)
