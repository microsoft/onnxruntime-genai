# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------



import numpy as np
import onnx_ir as ir
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

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
        self.rope_attrs["cast_to_fp32"] = True

        # Check rope type since huggingface model supports yarn but that is not recommended as mentioned in model card. Example:
        #    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24,24]}
        if config.rope_scaling and "type" in config.rope_scaling:
            assert config.rope_scaling["type"] in ["mrope", "default"]

        # Qwen 2.5 VL applies RoPE manually before attention, not fused in the op
        self.attention_attrs["use_rope_in_attn"] = False

        # We need separate Q, K, V tensors to apply MRoPE manually.
        # Packed MatMul provides a single output which would require splitting.
        self.attention_attrs["use_packed_matmul"] = False

        self.input_names["position_ids"] = "position_ids"

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
        pos_ids_name = self.input_names["position_ids"]
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

    def make_mrope_flattened_caches(self, layer_id, dyn_cos, dyn_sin):
        # Converts the 3D MRoPE caches [3, B, S, H] into flattened, interleaved caches [B*S, H/2]
        # suitable for the RotaryEmbedding operator.
        # The logic is:
        #   1. Slice dynamic caches to H/2.
        #   2. Split into 3 chunks based on mrope_sections (e.g. 16, 24, 24).
        #   3. Gather Temporal(0), Height(1), Width(2) specific slices for each chunk.
        #   4. Concat back to H/2.
        #   5. Flatten to [B*S, H/2].
        # The subgraph looks like:
        #      dyn_cos (3, B, S, H)
        #             |
        #           Slice
        #      (3, B, S, H/2)
        #             |
        #           Split
        #   (3, B, S, sections[i])
        #       /     |     \
        #  Gather  Gather  Gather
        #   idx=0   idx=1   idx=2
        #    /        |       \
        # Squeeze  Squeeze  Squeeze
        #    \        |       /
        #     \       |      /
        #      \      |     /
        #          Concat
        #       (B, S, H/2)
        #             |
        #          Reshape
        #        (B*S, H/2)

        basename = f"/model/layers.{layer_id}/attn/mrope_flattened_cache"

        def process_cache(input_name, name_suffix):
            # 1. Slice to H/2: [3, B, S, H] -> [3, B, S, H/2]
            slice_name = f"{basename}/{name_suffix}/half/Slice"
            slice_output = f"{slice_name}/output_0"
            self.make_slice(
                slice_name,
                [
                    input_name,
                    "/model/constants/INT64/[0]",
                    f"/model/constants/INT64/[{self.head_size // 2}]",
                    "/model/constants/INT64/[-1]",
                ],
                ir.DataType.FLOAT,
                [3, "batch_size", "sequence_length", self.head_size // 2],
            )

            # Create a Constant node for mrope_sections: [16, 24, 24]
            sections_name = f"{basename}/mrope_sections/Constant"
            sections_output = f"{basename}/mrope_sections"
            self.make_node(
                "Constant",
                [],
                [sections_output],
                name=sections_name,
                value=ir.tensor(torch.tensor(self.mrope_sections, dtype=torch.int64), name=sections_output),
            )
            self.make_value(sections_output, ir.DataType.INT64, [3])

            # 2. Split: [3, B, S, H/2] -> 3 * [3, B, S, section_dim]
            split_name = f"{basename}/{name_suffix}/Split"
            split_outputs = [f"{split_name}/output_{i}" for i in range(3)]
            self.make_node(
                "Split",
                [slice_output, sections_output],
                split_outputs,
                name=split_name,
                axis=-1,
            )

            # 3. Gather + Squeeze: Reorder T, H, W
            gathered_chunks = []
            for i in range(3):
                # Chunk 0->T(0), Chunk 1->H(1), Chunk 2->W(2)
                gather_name = f"{basename}/{name_suffix}/chunk_{i}/Gather"
                gather_output = f"{gather_name}/output_0"
                self.make_node(
                    "Gather",
                    [split_outputs[i], f"/model/constants/INT64/[{i}]"],
                    [gather_output],
                    name=gather_name,
                    axis=0,
                )
                # Gather output is [1, B, S, dim]

                squeeze_name = f"{basename}/{name_suffix}/chunk_{i}/Squeeze"
                squeeze_output = f"{squeeze_name}/output_0"
                self.make_squeeze(
                    squeeze_name,
                    [gather_output, "/model/constants/INT64/[0]"],
                    ir.DataType.FLOAT,
                    ["batch_size", "sequence_length", self.mrope_sections[i]],
                )
                gathered_chunks.append(squeeze_output)

            # 4. Concat: -> [B, S, H/2]
            concat_name = f"{basename}/{name_suffix}/Concat"
            concat_output = f"{concat_name}/output_0"
            self.make_concat(
                concat_name,
                gathered_chunks,
                ir.DataType.FLOAT,
                ["batch_size", "sequence_length", self.head_size // 2],
                axis=-1,
            )

            # 5. Flatten: -> [B*S, H/2]
            reshape_name = f"{basename}/{name_suffix}_flat/Reshape"
            reshape_output = f"{reshape_name}/output_0"
            self.make_reshape(
                reshape_name,
                [concat_output, f"/model/constants/INT64/[-1, {self.head_size // 2}]"],
                ir.DataType.FLOAT,
                ["total_token_count", self.head_size // 2],
            )
            return reshape_output

        flat_cos = process_cache(dyn_cos, "cos")
        flat_sin = process_cache(dyn_sin, "sin")

        return flat_cos, flat_sin

    def apply_mrope_rotation(self, layer_id, q_or_k_path, q_or_k_shape, dyn_cos, dyn_sin, num_heads, basename):
        # Make nodes for the MRoPE rotation subgraph using RotaryEmbedding op
        #
        # 1. Flatten 3D caches [3, B, S, H] -> [B*S, H/2] (via make_mrope_flattened_caches)
        # 2. Generate linear position IDs [B, S] (0 .. B*S-1)
        # 3. Apply RotaryEmbedding
        #
        #      dyn_cos (3, B, S, H)   dyn_sin (3, B, S, H)
        #              |                      |
        #    make_mrope_flattened_caches (slice, split, gather, concat, flatten)
        #              |                      |
        #        flat_cos               flat_sin
        #      (B*S, H/2)             (B*S, H/2)
        #              |                      |
        #              +-----------+----------+
        #                          |
        #      q_or_k              |              position_ids
        #    (B, S, N*H)           |            (0 .. B*S-1)
        #        |                 |                 |
        #     Reshape              |              Reshape
        #        |                 |                 |
        #    Transpose             |                 |
        #   (B, N, S, H)           |               (B, S)
        #        |                 |                 |
        #        +--------+--------+--------+--------+
        #                 |                 |
        #          RotaryEmbedding (com.microsoft)
        #                 |
        #            output (B, N, S, H)
        #                 |
        #             Transpose
        #                 |
        #              Reshape
        #            (B, S, N*H)

        # 1. Prepare flattened MRoPE caches [B*S, H/2]
        #    This slices, splits, and re-assembles the 3D dynamic caches into the correct per-token layout.
        flat_cos, flat_sin = self.make_mrope_flattened_caches(layer_id, dyn_cos, dyn_sin)

        # 2. Prepare position_ids [B, S] (values 0 to B*S - 1)
        #    RotaryEmbedding will use these indices to access the flattened cache.
        #    Get B*S from q_or_k shape. q_or_k is [B, S, N*H].
        shape_node = f"{basename}/Shape"
        self.make_shape(shape_node, q_or_k_path, [3])

        # Extract B and S
        batch_size_node = f"{basename}/BatchSize/Gather"
        batch_size_out = f"{batch_size_node}/output_0"
        self.make_gather(
            batch_size_node, [f"{shape_node}/output_0", "/model/constants/INT64/[0]"], ir.DataType.INT64, [], 0
        )

        seq_len_node = f"{basename}/SeqLen/Gather"
        seq_len_out = f"{seq_len_node}/output_0"
        self.make_gather(
            seq_len_node, [f"{shape_node}/output_0", "/model/constants/INT64/[1]"], ir.DataType.INT64, [], 0
        )

        # Calculate Total Tokens = B * S
        mul_len_node = f"{basename}/TotalLen/Mul"
        mul_len_out = f"{mul_len_node}/output_0"
        self.make_node("Mul", [batch_size_out, seq_len_out], [mul_len_out], name=mul_len_node)
        self.make_value(mul_len_out, ir.DataType.INT64, [])

        # Range(0, TotalTokens)
        range_node = f"{basename}/Range"
        range_out = f"{range_node}/output_0"
        self.make_node(
            "Range", ["/model/constants/INT64/0", mul_len_out, "/model/constants/INT64/1"], [range_out], name=range_node
        )
        self.make_value(range_out, ir.DataType.INT64, ["total_token_count"])

        # Slice Position IDs shape from input shape (take first 2 dims)
        slice_shape_node = f"{basename}/SliceShape"
        slice_shape_out = f"{slice_shape_node}/output_0"
        self.make_slice(
            slice_shape_node,
            [
                f"{shape_node}/output_0",
                "/model/constants/INT64/[0]",
                "/model/constants/INT64/[2]",
                "/model/constants/INT64/[0]",
            ],
            ir.DataType.INT64,
            [2],
        )

        # Reshape Range output to [B, S]
        pos_ids_reshape_node = f"{basename}/PosIds/Reshape"
        pos_ids_out = f"{pos_ids_reshape_node}/output_0"
        self.make_reshape(
            pos_ids_reshape_node, [range_out, slice_shape_out], ir.DataType.INT64, ["batch_size", "sequence_length"]
        )

        # 3. Prepare Q/K input [B, N, S, H]
        #    Input is [B, S, N*H]. Reshape -> [B, S, N, H] -> Transpose -> [B, N, S, H]
        reshape_in_node = f"{basename}/Input/Reshape"
        reshape_in_out = f"{reshape_in_node}/output_0"
        self.make_reshape(
            reshape_in_node,
            [q_or_k_path, f"/model/constants/INT64/[0, 0, {num_heads}, {self.head_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", num_heads, self.head_size],
        )

        transpose_in_node = f"{basename}/Input/Transpose"
        transpose_in_out = f"{transpose_in_node}/output_0"
        target_shape_bnsh = ["batch_size", num_heads, "sequence_length", self.head_size]
        self.make_transpose(transpose_in_node, reshape_in_out, self.io_dtype, target_shape_bnsh, [0, 2, 1, 3])

        # 4. Handle Type Casting
        #    RotaryEmbedding requires input, cos, sin to be same type.
        #    Qwen2.5-VL forces float32 computation.
        force_fp32 = self.rope_attrs.get("cast_to_fp32", False)
        compute_dtype = ir.DataType.FLOAT if force_fp32 else self.io_dtype

        rope_input = transpose_in_out
        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            cast_in_node = f"{basename}/Input/Cast"
            rope_input = f"{cast_in_node}/output_0"
            self.make_cast(cast_in_node, transpose_in_out, compute_dtype, target_shape_bnsh)

        rope_cos = flat_cos
        rope_sin = flat_sin
        # Note: dyn_cos is Float. flat_cos is Float. If compute_dtype is not Float (e.g. fp16), we must cast cache.
        if compute_dtype != ir.DataType.FLOAT:
            # Cache is Float, we need FP16
            cast_cos_node = f"{basename}/Cos/Cast"
            rope_cos = f"{cast_cos_node}/output_0"
            self.make_cast(cast_cos_node, flat_cos, compute_dtype, ["total_token_count", self.head_size // 2])

            cast_sin_node = f"{basename}/Sin/Cast"
            rope_sin = f"{cast_sin_node}/output_0"
            self.make_cast(cast_sin_node, flat_sin, compute_dtype, ["total_token_count", self.head_size // 2])

        # 5. RotaryEmbedding Node
        rope_node = f"{basename}/RotaryEmbedding"
        rope_output = f"{rope_node}/output_0"
        self.make_node(
            "RotaryEmbedding",
            [rope_input, pos_ids_out, rope_cos, rope_sin],
            [rope_output],
            name=rope_node,
            domain="com.microsoft",
            rotary_embedding_dim=self.head_size,
            num_heads=num_heads,
            interleaved=0,  # False, matches rotate_half logic
        )
        self.make_value(rope_output, compute_dtype, target_shape_bnsh)

        # 6. Post-process Output
        #    Cast back if needed -> Transpose -> Reshape
        final_rope_output = rope_output
        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            cast_out_node = f"{basename}/Output/Cast"
            final_rope_output = f"{cast_out_node}/output_0"
            self.make_cast(cast_out_node, rope_output, self.io_dtype, target_shape_bnsh)

        transpose_out_node = f"{basename}/Output/Transpose"
        transpose_out_out = f"{transpose_out_node}/output_0"
        self.make_transpose(
            transpose_out_node,
            final_rope_output,
            self.io_dtype,
            ["batch_size", "sequence_length", num_heads, self.head_size],
            [0, 2, 1, 3],
        )

        reshape_out_node = f"{basename}/Output/Reshape"
        reshape_out_out = f"{reshape_out_node}/output_0"
        self.make_reshape(
            reshape_out_node,
            [transpose_out_out, f"/model/constants/INT64/[0, 0, {num_heads * self.head_size}]"],
            self.io_dtype,
            q_or_k_shape,
        )

        return reshape_out_out

    def make_attention_qk_subgraph(self, layer_id, attention, root_input, **kwargs):
        # Make nodes for the Attention subgraph (with MRoPE)
        #
        #        q_path    k_path    v_path
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

        # 1. Calculate shapes for MRoPE rotation
        q_shape = [
            "batch_size",
            "sequence_length",
            self.num_attn_heads * self.head_size,
        ]
        k_shape = [
            "batch_size",
            "sequence_length",
            self.num_kv_heads * self.head_size,
        ]

        # 2. Apply 3D RoPE (MRoPE)
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

        # 3. Call GroupQueryAttention op
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

    def load_weights(self, input_path):
        # For quantized models (e.g., Quark, AWQ, GPTQ) or GGUF, use base class logic
        # which loads weights directly via QuantModel
        if self.quant_type is not None or input_path.endswith(".gguf"):
            return super().load_weights(input_path)

        # For non-quantized models, load the Hugging Face model
        print("Loading Qwen2_5_VLForConditionalGeneration model...")
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
        )


class Qwen3VLTextModel(Qwen25VLTextModel):
    """
    Qwen3-VL text model builder. Inherits from Qwen25VLTextModel.

    Key differences from Qwen2.5-VL:
    - Uses interleaved MRoPE layout [THWTHWTHW...TT] instead of chunked [TTT...HHH...WWW]
    - Adds QK normalization (q_norm, k_norm) from Qwen3 base architecture
    - Default mrope_section is [24, 20, 20] (vs [16, 24, 24] in Qwen2.5-VL)
    - Vision encoder uses DeepStack for multi-layer feature injection (handled by vision ONNX model)
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Fix model_type: HF architecture "Qwen3VLForConditionalGeneration" would produce "qwen3vl"
        # but the C++ runtime expects "qwen3_vl" (with underscore).
        # Intentional override of the superclass attribute (used in genai_config.json).
        self.model_type = "Qwen3_VLForConditionalGeneration" # noqa: overrides Model.model_type on purpose

        # Qwen3 attention uses QK normalization
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True

    def make_attention_qk_subgraph(self, layer_id, attention, root_input, **kwargs):
        # Qwen3-VL adds QK normalization before MRoPE rotation
        # The parent class (Qwen25VLTextModel) skips make_qk_norm since Qwen2.5-VL doesn't use it.
        # We must call it here before proceeding with MRoPE.
        if self.attention_attrs["q_norm"] and self.attention_attrs["k_norm"]:
            self.make_qk_norm(layer_id, attention)

        # Delegate to parent for MRoPE rotation + GQA
        super().make_attention_qk_subgraph(layer_id, attention, root_input, **kwargs)

    def make_mrope_flattened_caches(self, layer_id, dyn_cos, dyn_sin):
        """
        Converts the 3D MRoPE caches [3, B, S, H] into flattened, interleaved caches [B*S, H/2]
        suitable for the RotaryEmbedding operator.

        Qwen3-VL uses interleaved MRoPE layout: [THWTHWTHW...TT]
        This differs from Qwen2.5-VL's chunked layout: [TTT...HHH...WWW]

        The interleaving logic (from HuggingFace Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope):
          freqs_t = freqs[0]  # start with temporal
          for dim, offset in enumerate((1, 2), start=1):  # H=1, W=2
              length = mrope_section[dim] * 3
              idx = slice(offset, length, 3)
              freqs_t[..., idx] = freqs[dim, ..., idx]

        For mrope_section = [24, 20, 20], head_dim/2 = 64:
          - All 64 positions start with Temporal values
          - Height overwrites positions [1, 4, 7, ..., 58] (20 values)
          - Width overwrites positions [2, 5, 8, ..., 59] (20 values)
          - Result pattern: [T,H,W, T,H,W, ..., T,H,W, T,T,T,T] (20 THW groups + 4 T-only)
        """
        basename = f"/model/layers.{layer_id}/attn/mrope_interleaved_cache"
        shared_base = "/model/attn/mrope_interleaved_cache"

        half_head = self.head_size // 2

        # Cache the deterministic index mappings on self so we compute them once
        # and emit shared ONNX Constant nodes that all layers reference.
        if not hasattr(self, "_mrope_cache"):
            # Pre-compute the interleaved index mapping: for each position in H/2,
            # which dimension (0=T, 1=H, 2=W)?
            dim_assignments = [0] * half_head  # Start all positions as Temporal
            for dim_idx, offset in enumerate((1, 2), start=1):  # H=1, W=2
                length = self.mrope_sections[dim_idx] * 3
                for i in range(offset, length, 3):
                    if i < half_head:
                        dim_assignments[i] = dim_idx

            dim_to_positions = {0: [], 1: [], 2: []}
            for pos, dim in enumerate(dim_assignments):
                dim_to_positions[dim].append(pos)

            # Build reorder indices (same for cos and sin, all layers)
            concat_order = []
            for dim_idx in range(3):
                concat_order.extend(dim_to_positions[dim_idx])
            reorder_indices = [0] * half_head
            for concat_idx, orig_pos in enumerate(concat_order):
                reorder_indices[orig_pos] = concat_idx

            # Emit shared position constants (one per dimension, reused across all layers)
            positions_outputs = {}
            for dim_idx in range(3):
                positions = dim_to_positions[dim_idx]
                if not positions:
                    continue
                pname = f"{shared_base}/dim{dim_idx}/Positions/Constant"
                pout = f"{shared_base}/dim{dim_idx}/positions"
                self.make_node(
                    "Constant",
                    [],
                    [pout],
                    name=pname,
                    value=ir.tensor(torch.tensor(positions, dtype=torch.int64), name=pout),
                )
                self.make_value(pout, ir.DataType.INT64, [len(positions)])
                positions_outputs[dim_idx] = pout

            # Emit shared reorder constant
            rname = f"{shared_base}/Reorder/Constant"
            rout = f"{shared_base}/reorder"
            self.make_node(
                "Constant",
                [],
                [rout],
                name=rname,
                value=ir.tensor(torch.tensor(reorder_indices, dtype=torch.int64), name=rout),
            )
            self.make_value(rout, ir.DataType.INT64, [half_head])

            self._mrope_cache = {
                "dim_to_positions": dim_to_positions,
                "positions_outputs": positions_outputs,
                "reorder_output": rout,
            }

        dim_to_positions = self._mrope_cache["dim_to_positions"]
        positions_outputs = self._mrope_cache["positions_outputs"]
        reorder_const_output = self._mrope_cache["reorder_output"]

        def process_cache(input_name, name_suffix):
            # 1. Slice to H/2: [3, B, S, H] -> [3, B, S, H/2]
            slice_name = f"{basename}/{name_suffix}/half/Slice"
            slice_output = f"{slice_name}/output_0"
            self.make_slice(
                slice_name,
                [
                    input_name,
                    "/model/constants/INT64/[0]",
                    f"/model/constants/INT64/[{half_head}]",
                    "/model/constants/INT64/[-1]",
                ],
                ir.DataType.FLOAT,
                [3, "batch_size", "sequence_length", half_head],
            )

            # 2. Build interleaved output by gathering individual positions from appropriate dimensions
            gathered_pieces = []
            for dim_idx in range(3):
                positions = dim_to_positions[dim_idx]
                if not positions:
                    continue

                # Gather this dimension: [3, B, S, H/2] -> [1, B, S, H/2] via index dim_idx on axis 0
                gather_dim_name = f"{basename}/{name_suffix}/dim{dim_idx}/Gather"
                gather_dim_output = f"{gather_dim_name}/output_0"
                self.make_node(
                    "Gather",
                    [slice_output, f"/model/constants/INT64/[{dim_idx}]"],
                    [gather_dim_output],
                    name=gather_dim_name,
                    axis=0,
                )

                squeeze_dim_name = f"{basename}/{name_suffix}/dim{dim_idx}/Squeeze"
                squeeze_dim_output = f"{squeeze_dim_name}/output_0"
                self.make_squeeze(
                    squeeze_dim_name,
                    [gather_dim_output, "/model/constants/INT64/[0]"],
                    ir.DataType.FLOAT,
                    ["batch_size", "sequence_length", half_head],
                )

                # Gather specific positions (reuse shared constant node)
                gather_pos_name = f"{basename}/{name_suffix}/dim{dim_idx}/Positions/Gather"
                gather_pos_output = f"{gather_pos_name}/output_0"
                self.make_node(
                    "Gather",
                    [squeeze_dim_output, positions_outputs[dim_idx]],
                    [gather_pos_output],
                    name=gather_pos_name,
                    axis=-1,
                )
                self.make_value(gather_pos_output, ir.DataType.FLOAT, ["batch_size", "sequence_length", len(positions)])

                gathered_pieces.append((positions, gather_pos_output))

            # 3. Concatenate all pieces and reorder to interleaved layout
            all_outputs = [(positions, output) for positions, output in gathered_pieces]

            if len(all_outputs) == 1:
                concat_output = all_outputs[0][1]
            else:
                concat_name = f"{basename}/{name_suffix}/AllPieces/Concat"
                concat_output = f"{concat_name}/output_0"
                self.make_concat(
                    concat_name,
                    [out for _, out in all_outputs],
                    ir.DataType.FLOAT,
                    ["batch_size", "sequence_length", half_head],
                    axis=-1,
                )

            # Reorder using shared constant
            gather_reorder_name = f"{basename}/{name_suffix}/Reorder/Gather"
            gather_reorder_output = f"{gather_reorder_name}/output_0"
            self.make_node(
                "Gather",
                [concat_output, reorder_const_output],
                [gather_reorder_output],
                name=gather_reorder_name,
                axis=-1,
            )
            self.make_value(gather_reorder_output, ir.DataType.FLOAT, ["batch_size", "sequence_length", half_head])

            # 4. Flatten: -> [B*S, H/2]
            reshape_name = f"{basename}/{name_suffix}_flat/Reshape"
            reshape_output = f"{reshape_name}/output_0"
            self.make_reshape(
                reshape_name,
                [gather_reorder_output, f"/model/constants/INT64/[-1, {half_head}]"],
                ir.DataType.FLOAT,
                ["total_token_count", half_head],
            )
            return reshape_output

        flat_cos = process_cache(dyn_cos, "cos")
        flat_sin = process_cache(dyn_sin, "sin")

        return flat_cos, flat_sin

    def load_weights(self, input_path):
        # Load the Hugging Face model
        print("Loading Qwen3VLForConditionalGeneration model...")
        return Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
        )


class Qwen35TextModel(Model):
    """Qwen3.5 hybrid model builder.

    Qwen3.5 uses a hybrid architecture with two layer types:
    - ``full_attention``: Attention with doubled Q projection (Q + output gate),
      per-head QK RMSNorm, partial rotary embeddings, and output gating
    - ``linear_attention``: GatedDeltaNet recurrent layer with depthwise
      causal conv1d, L2-normalised Q/K, and linear attention recurrence

    The layer type pattern is controlled by ``config.layer_types`` (or
    derived from ``config.full_attention_interval``).

    Both layer types use OffsetRMSNorm (the ``1 + weight`` variant).
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Qwen3.5 is a VL model. The decoder takes inputs_embeds.
        if "exclude_embeds" not in extra_options:
            extra_options["exclude_embeds"] = True
            print("Setting exclude_embeds=True for Qwen3.5 VL decoder.")

        # Qwen3.5 is a multimodal model whose HF config nests text config
        # under text_config. Flatten text_config attributes onto config so
        # the base Model init finds them where it expects.
        if hasattr(config, "text_config"):
            text_config = config.text_config
            for key in text_config:
                if not hasattr(config, key) or getattr(config, key) is None:
                    setattr(config, key, getattr(text_config, key))

        # rope_scaling contains the actual rope_theta for Qwen3.5
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            if "rope_theta" in config.rope_scaling:
                config.rope_theta = config.rope_scaling["rope_theta"]
            if "partial_rotary_factor" in config.rope_scaling:
                config.partial_rotary_factor = config.rope_scaling["partial_rotary_factor"]

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # OffsetRMSNorm: Qwen3.5 uses (1 + weight) * RMSNorm(x).
        # Pre-bake the +1 into the weight initializer so the base class's
        # SkipSimplifiedLayerNormalization can be used directly.
        self.layernorm_attrs["add_offset"] = 1

        # HF Qwen3_5RMSNorm always computes in float32 regardless of model
        # dtype.  Force the builder to cast inputs to fp32 before LayerNorm
        # and cast back after, matching HF behaviour and preventing precision
        # loss that compounds across 36+ layers in fp16/bf16 builds.
        self.layernorm_attrs["cast"]["use_fp32"] = True

        # 3D position_ids for mRoPE: [3, batch_size, sequence_length]
        self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]
        self.input_names["position_ids"] = "position_ids"

        # mRoPE config
        self.mrope_sections = self.rope_attrs.get("mrope", {}).get("sections", [])
        if not self.mrope_sections:
            raise ValueError("MRoPE sections not found in config.text_config.rope_scaling.mrope_section")
        if len(self.mrope_sections) != 3:
            raise ValueError(
                f"Expected 3 MRoPE sections [T, H, W], got {len(self.mrope_sections)}: {self.mrope_sections}"
            )
        self.mrope_rotary_dim = int(self.rope_attrs["partial_rotary_factor"] * self.head_size)

        # Force RoPE computation in float32 for numerical stability
        self.rope_attrs["cast_to_fp32"] = True

        # Pre-compute cos/sin cache tables and interleaving masks for mRoPE
        self._make_rotary_caches()

        # Parse layer types
        if hasattr(config, "layer_types") and config.layer_types is not None:
            self.layer_types = list(config.layer_types)
        elif hasattr(config, "full_attention_interval") and config.full_attention_interval is not None:
            interval = config.full_attention_interval
            self.layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(self.num_layers)
            ]
        else:
            self.layer_types = ["full_attention"] * self.num_layers

        # Store linear attention config
        self.linear_key_head_dim = getattr(config, "linear_key_head_dim", 128)
        self.linear_value_head_dim = getattr(config, "linear_value_head_dim", 128)
        self.linear_num_key_heads = getattr(config, "linear_num_key_heads", 16)
        self.linear_num_value_heads = getattr(config, "linear_num_value_heads", 16)
        self.linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)

        # Derived dimensions for GatedDeltaNet
        self.linear_key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        self.linear_conv_dim = self.linear_key_dim * 2 + self.linear_value_dim

        # Full attention uses QK norm and output gating
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        # Disable fused RoPE in attention op - we apply mRoPE manually
        self.attention_attrs["use_rope_in_attn"] = False

        # Auto-exclude linear attention gate projections from int4 quantization.
        # in_proj_a and in_proj_b project to just n_kv dims (~16) and control
        # recurrence gates: exp(decay) errors compound across the sequence,
        # causing output collapse when quantized to int4.
        gate_exclusions = []
        for i, lt in enumerate(self.layer_types):
            if lt == "linear_attention":
                gate_exclusions.append(f"/model/layers.{i}/linear_attn/in_proj_b/MatMul")
                gate_exclusions.append(f"/model/layers.{i}/linear_attn/in_proj_a/MatMul")
        if gate_exclusions:
            existing = list(self.quant_attrs["int4"]["nodes_to_exclude"])
            existing.extend(n for n in gate_exclusions if n not in existing)
            self.quant_attrs["int4"]["nodes_to_exclude"] = existing

        # Replace standard KV cache I/O with hybrid cache I/O
        self._setup_hybrid_cache_io()

    def _setup_hybrid_cache_io(self):
        """Set up hybrid cache I/O: KV cache for attention layers,
        conv_state + recurrent_state for linear attention layers."""

        # The base class creates KV cache entries for all num_layers.
        # We rebuild the lists: keep KV entries only for full-attention layers,
        # and add conv/recurrent state entries for linear-attention layers.
        kv_key_inputs = self.input_names["past_key_values.key"]
        kv_value_inputs = self.input_names["past_key_values.value"]
        kv_key_outputs = self.output_names["present.key"]
        kv_value_outputs = self.output_names["present.value"]

        filtered_key_inputs = []
        filtered_value_inputs = []
        filtered_key_outputs = []
        filtered_value_outputs = []

        for i, lt in enumerate(self.layer_types):
            if lt == "full_attention":
                filtered_key_inputs.append(kv_key_inputs[i])
                filtered_value_inputs.append(kv_value_inputs[i])
                filtered_key_outputs.append(kv_key_outputs[i])
                filtered_value_outputs.append(kv_value_outputs[i])
            else:
                # linear_attention: add conv_state + recurrent_state
                self.input_names[f"past_state.{i}.conv"] = f"past_key_values.{i}.conv_state"
                self.input_types[f"past_state.{i}.conv"] = self.io_dtype
                self.input_shapes[f"past_state.{i}.conv"] = [
                    "batch_size",
                    self.linear_conv_dim,
                    self.linear_conv_kernel_dim - 1,
                ]

                self.input_names[f"past_state.{i}.recurrent"] = f"past_key_values.{i}.recurrent_state"
                self.input_types[f"past_state.{i}.recurrent"] = self.io_dtype
                self.input_shapes[f"past_state.{i}.recurrent"] = [
                    "batch_size",
                    self.linear_num_value_heads,
                    self.linear_key_head_dim,
                    self.linear_value_head_dim,
                ]

                self.output_names[f"present_state.{i}.conv"] = f"present.{i}.conv_state"
                self.output_types[f"present_state.{i}.conv"] = self.io_dtype
                self.output_shapes[f"present_state.{i}.conv"] = [
                    "batch_size",
                    self.linear_conv_dim,
                    self.linear_conv_kernel_dim - 1,
                ]

                self.output_names[f"present_state.{i}.recurrent"] = f"present.{i}.recurrent_state"
                self.output_types[f"present_state.{i}.recurrent"] = self.io_dtype
                self.output_shapes[f"present_state.{i}.recurrent"] = [
                    "batch_size",
                    self.linear_num_value_heads,
                    self.linear_key_head_dim,
                    self.linear_value_head_dim,
                ]

        self.input_names["past_key_values.key"] = filtered_key_inputs
        self.input_names["past_key_values.value"] = filtered_value_inputs
        self.output_names["present.key"] = filtered_key_outputs
        self.output_names["present.value"] = filtered_value_outputs

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Dispatch to full attention or GatedDeltaNet based on layer type."""
        if self.layer_types[layer_id] == "linear_attention":
            self._make_linear_attention(layer_id, attention, root_input)
        else:
            self._make_full_attention(layer_id, attention, root_input)

    def make_layer(self, layer_id, layer):
        """Override to pass ``linear_attn`` instead of ``self_attn`` for
        linear-attention layers (the base class assumes ``self_attn``)."""
        attn_module = (
            layer.linear_attn
            if self.layer_types[layer_id] == "linear_attention"
            else layer.self_attn
        )
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, attn_module, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    def _make_full_attention(self, layer_id, attn, root_input):
        """Build full attention with output gating.

        Qwen3.5 full attention has a doubled Q projection that produces both
        Q and a gating signal. After attention, the output is multiplied by
        sigmoid(gate) before the output projection.
        """
        # 1. Q projection (doubled: outputs Q and gate)
        q_matmul_name = f"/model/layers.{layer_id}/attn/q_proj/MatMul"
        self.make_matmul(attn.q_proj, q_matmul_name, root_input)
        q_gate_path = f"{q_matmul_name}/output_0"

        # Split Q and gate PER-HEAD: reshape [B,S,N*2H] -> [B,S,N,2H] -> split -> [B,S,N,H] each -> reshape back
        q_size = self.num_attn_heads * self.head_size

        # Reshape to [B, S, N, 2*H]
        rs_qg_name = f"/model/layers.{layer_id}/attn/q_gate_reshape/Reshape"
        rs_qg_output = f"{rs_qg_name}/output_0"
        self.make_reshape(
            rs_qg_name,
            [q_gate_path, f"/model/constants/INT64/[0, 0, {self.num_attn_heads}, {self.head_size * 2}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", self.num_attn_heads, self.head_size * 2],
        )

        # Split per-head: [B, S, N, 2H] -> [B, S, N, H] + [B, S, N, H]
        split_name = f"/model/layers.{layer_id}/attn/q_gate_split/Split"
        q_4d_output = f"{split_name}/output_0"
        gate_4d_output = f"{split_name}/output_1"
        self.make_node(
            "Split",
            [rs_qg_output, f"/model/constants/INT64/[{self.head_size}, {self.head_size}]"],
            [q_4d_output, gate_4d_output],
            name=split_name,
            axis=-1,
        )
        self.make_value(
            q_4d_output, self.io_dtype, ["batch_size", "sequence_length", self.num_attn_heads, self.head_size]
        )
        self.make_value(
            gate_4d_output, self.io_dtype, ["batch_size", "sequence_length", self.num_attn_heads, self.head_size]
        )

        # Reshape Q back to [B, S, N*H]
        rs_q_name = f"/model/layers.{layer_id}/attn/q_reshape/Reshape"
        q_output = f"{rs_q_name}/output_0"
        self.make_reshape(
            rs_q_name,
            [q_4d_output, f"/model/constants/INT64/[0, 0, {q_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", q_size],
        )

        # Reshape gate back to [B, S, N*H]
        rs_g_name = f"/model/layers.{layer_id}/attn/gate_reshape/Reshape"
        gate_output = f"{rs_g_name}/output_0"
        self.make_reshape(
            rs_g_name,
            [gate_4d_output, f"/model/constants/INT64/[0, 0, {q_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", q_size],
        )

        self.attention_attrs["q_path"] = q_output

        # 3. K and V projections
        k_matmul_name = f"/model/layers.{layer_id}/attn/k_proj/MatMul"
        self.make_matmul(attn.k_proj, k_matmul_name, root_input)
        self.attention_attrs["k_path"] = f"{k_matmul_name}/output_0"

        v_matmul_name = f"/model/layers.{layer_id}/attn/v_proj/MatMul"
        self.make_matmul(attn.v_proj, v_matmul_name, root_input)
        self.attention_attrs["v_path"] = f"{v_matmul_name}/output_0"

        # 4. Per-head QK RMSNorm (on Q and K separately)
        #    The base class's make_qk_norm uses SimplifiedLayerNormalization
        #    and pre-bakes the +1 offset via layernorm_attrs["add_offset"].
        self.make_qk_norm(layer_id, attn)

        # 5. Apply interleaved mRoPE to Q and K
        if self.attention_attrs["rope"]:
            q_shape = ["batch_size", "sequence_length", self.num_attn_heads * self.head_size]
            k_shape = ["batch_size", "sequence_length", self.num_kv_heads * self.head_size]

            # Build interleaved cos/sin from pre-computed cache + 3D position_ids
            cos_dyn, sin_dyn = self._make_mrope_cos_sin("/model/rotary_emb")

            # Apply mRoPE rotation to Q
            self.attention_attrs["q_path"] = self._apply_mrope_rotation(
                layer_id,
                self.attention_attrs["q_path"],
                q_shape,
                cos_dyn,
                sin_dyn,
                self.num_attn_heads,
                f"/model/layers.{layer_id}/attn/q_mrope",
            )

            # Apply mRoPE rotation to K
            self.attention_attrs["k_path"] = self._apply_mrope_rotation(
                layer_id,
                self.attention_attrs["k_path"],
                k_shape,
                cos_dyn,
                sin_dyn,
                self.num_kv_heads,
                f"/model/layers.{layer_id}/attn/k_mrope",
            )

        # 6. GroupQueryAttention with per-layer KV cache
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
            cos_cache="",
            sin_cache="",
        )
        attn_output = f"{attn_name}/output_0"

        # 7. Output gating: attn_output * sigmoid(gate)
        sigmoid_name = f"/model/layers.{layer_id}/attn/gate/Sigmoid"
        self.make_sigmoid(sigmoid_name, gate_output, self.io_dtype, ["batch_size", "sequence_length", q_size])
        sigmoid_output = f"{sigmoid_name}/output_0"

        gated_name = f"/model/layers.{layer_id}/attn/gate/Mul"
        self.make_mul(gated_name, [attn_output, sigmoid_output], self.io_dtype, ["batch_size", "sequence_length", q_size])
        gated_output = f"{gated_name}/output_0"

        # 8. Output projection
        o_matmul_name = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        self.make_matmul(attn.o_proj, o_matmul_name, gated_output)
        self.layernorm_attrs["skip_input"] = f"{o_matmul_name}/output_0"

    def _make_rotary_caches(self):
        """Pre-compute cos/sin cache table and h/w interleaving masks.

        Matches the reference model's approach:
        - cos_cache [max_len, rdim_half]: pre-computed cos(pos * inv_freq)
        - sin_cache [max_len, rdim_half]: pre-computed sin(pos * inv_freq)
        - h_mask [rdim_half]: bool mask for height positions
        - w_mask [rdim_half]: bool mask for width positions
        """
        rdim = self.mrope_rotary_dim
        rdim_half = rdim // 2
        max_len = self.context_length

        inv_freq = 1.0 / (
            self.rope_attrs["rescale_factors"]
            * (self.rope_attrs["theta"] ** (torch.arange(0, rdim, 2, dtype=torch.int64).float() / rdim))
        )

        positions = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_len, rdim_half]
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        self.make_initializer(cos_cache, "model.rotary_emb.cos_cache", to=ir.DataType.FLOAT)
        self.make_initializer(sin_cache, "model.rotary_emb.sin_cache", to=ir.DataType.FLOAT)

        # Build interleaving masks
        dim_assignments = [0] * rdim_half
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.mrope_sections[dim_idx] * 3
            for i in range(offset, length, 3):
                if i < rdim_half:
                    dim_assignments[i] = dim_idx

        h_mask = torch.tensor([d == 1 for d in dim_assignments], dtype=torch.bool)
        w_mask = torch.tensor([d == 2 for d in dim_assignments], dtype=torch.bool)

        self.make_initializer(h_mask, "model.rotary_emb.h_mask", to=ir.DataType.BOOL)
        self.make_initializer(w_mask, "model.rotary_emb.w_mask", to=ir.DataType.BOOL)
        print(f"Created rotary caches [{max_len}, {rdim_half}] + h/w masks [{rdim_half}].")

    def _get_shared_q_scale(self, head_dim):
        """Return the name of a shared 1/sqrt(head_dim) constant (created once)."""
        name = "model.constants.q_scale"
        scale_val = float(1.0 / np.sqrt(head_dim))
        self.make_initializer(
            torch.tensor([scale_val], dtype=torch.float32),
            name,
            to=self.io_dtype,
        )
        return name

    def _get_shared_l2_eps(self):
        """Return the name of a shared L2 epsilon constant (created once)."""
        name = "model.constants.l2_eps"
        self.make_initializer(
            torch.tensor([1e-6], dtype=torch.float32),
            name,
            to=self.io_dtype,
        )
        return name

    def _make_mrope_cos_sin(self, basename):
        """Build interleaved mRoPE cos/sin from pre-computed cache + position_ids.

        Input: position_ids [3, B, S]
        Output: cos [B, S, rdim_half], sin [B, S, rdim_half]
        """
        pos_ids = self.input_names["position_ids"]
        cos_cache = "model.rotary_emb.cos_cache"
        sin_cache = "model.rotary_emb.sin_cache"
        h_mask = "model.rotary_emb.h_mask"
        w_mask = "model.rotary_emb.w_mask"
        rdim_half = self.mrope_rotary_dim // 2

        def gather_dim(dim_idx, cache_name, suffix):
            g_name = f"{basename}/{suffix}/dim{dim_idx}/pos/Gather"
            self.make_gather(
                g_name, [pos_ids, f"/model/constants/INT64/[{dim_idx}]"], ir.DataType.INT64,
                [1, "batch_size", "sequence_length"], axis=0,
            )
            sq_name = f"{basename}/{suffix}/dim{dim_idx}/Squeeze"
            self.make_squeeze(
                sq_name, [f"{g_name}/output_0", "/model/constants/INT64/[0]"],
                ir.DataType.INT64, ["batch_size", "sequence_length"],
            )
            gc_name = f"{basename}/{suffix}/dim{dim_idx}/cache/Gather"
            self.make_gather(
                gc_name, [cache_name, f"{sq_name}/output_0"], ir.DataType.FLOAT,
                ["batch_size", "sequence_length", rdim_half], axis=0,
            )
            return f"{gc_name}/output_0"

        def interleave(suffix, cache_name):
            t = gather_dim(0, cache_name, suffix)
            h = gather_dim(1, cache_name, suffix)
            w = gather_dim(2, cache_name, suffix)
            ww_name = f"{basename}/{suffix}/w/Where"
            ww_out = f"{ww_name}/output_0"
            self.make_where(ww_name, [w_mask, w, t], ir.DataType.FLOAT, ["batch_size", "sequence_length", rdim_half])
            ww_out = f"{ww_name}/output_0"
            hh_name = f"{basename}/{suffix}/h/Where"
            self.make_where(hh_name, [h_mask, h, ww_out], ir.DataType.FLOAT, ["batch_size", "sequence_length", rdim_half])
            hh_out = f"{hh_name}/output_0"
            return hh_out

        return interleave("cos", cos_cache), interleave("sin", sin_cache)

    def _apply_mrope_rotation(self, layer_id, qk_path, qk_shape, dyn_cos, dyn_sin, num_heads, basename):
        """Apply mRoPE via RotaryEmbedding (opset 23).

        cos/sin caches are always float32. When io_dtype differs (fp16/bf16),
        cast Q/K to float32 before rotation, then cast back — preserving
        numerical precision in the RoPE computation.
        """
        force_fp32 = self.rope_attrs.get("cast_to_fp32", False)
        compute_dtype = ir.DataType.FLOAT if force_fp32 else self.io_dtype

        # Cast Q/K to compute dtype if needed
        rope_input = qk_path
        if compute_dtype != self.io_dtype:
            cast_in_name = f"{basename}/input/Cast"
            self.make_cast(cast_in_name, qk_path, compute_dtype, qk_shape)
            rope_input = f"{cast_in_name}/output_0"

        # Cast cos/sin to compute dtype if they differ
        rope_cos = dyn_cos
        rope_sin = dyn_sin
        rdim_half = self.mrope_rotary_dim // 2
        cos_sin_shape = ["batch_size", "sequence_length", rdim_half]
        if compute_dtype != ir.DataType.FLOAT:
            cos_cast_name = f"{basename}/cos/Cast"
            self.make_cast(cos_cast_name, dyn_cos, compute_dtype, cos_sin_shape)
            rope_cos = f"{cos_cast_name}/output_0"

            sin_cast_name = f"{basename}/sin/Cast"
            self.make_cast(sin_cast_name, dyn_sin, compute_dtype, cos_sin_shape)
            rope_sin = f"{sin_cast_name}/output_0"

        rope_name = f"{basename}/RotaryEmbedding"
        rope_out = f"{rope_name}/output_0"
        self.make_node(
            "RotaryEmbedding",
            [rope_input, rope_cos, rope_sin],
            [rope_out],
            name=rope_name,
            num_heads=num_heads,
            rotary_embedding_dim=self.mrope_rotary_dim,
            interleaved=0,
        )
        self.make_value(rope_out, compute_dtype, qk_shape)

        # Cast back to io_dtype if needed
        final_output = rope_out
        if compute_dtype != self.io_dtype:
            cast_out_name = f"{basename}/output/Cast"
            self.make_cast(cast_out_name, rope_out, self.io_dtype, qk_shape)
            final_output = f"{cast_out_name}/output_0"

        return final_output

    def _make_linear_attention(self, layer_id, linear_attn, root_input):
        """Build a GatedDeltaNet linear attention layer.

        Implements the full GatedDeltaNet forward pass:
        1. Linear projections (QKV fused, z gate, beta, alpha)
        2. Depthwise causal conv1d with carry state
        3. Split into Q, K, V + L2 normalize Q and K
        4. Compute decay and forget gates
        5. Linear attention recurrence
        6. Gated RMSNorm + output projection
        """
        basename = f"/model/layers.{layer_id}/linear_attn"

        k_dim = self.linear_key_dim  # e.g. 2048
        v_dim = self.linear_value_dim  # e.g. 2048
        conv_dim = self.linear_conv_dim  # e.g. 6144
        n_kv = self.linear_num_value_heads  # e.g. 16
        hk = self.linear_key_head_dim  # e.g. 128
        hv = self.linear_value_head_dim  # e.g. 128
        kernel_size = self.linear_conv_kernel_dim  # e.g. 4

        # 2. Linear projections
        # QKV fused: [B, S, hidden] -> [B, S, conv_dim]
        qkv_name = f"{basename}/in_proj_qkv/MatMul"
        self.make_matmul(linear_attn.in_proj_qkv, qkv_name, root_input)

        # z (output gate): [B, S, hidden] -> [B, S, v_dim]
        z_name = f"{basename}/in_proj_z/MatMul"
        self.make_matmul(linear_attn.in_proj_z, z_name, root_input)

        # b (forget/beta): [B, S, hidden] -> [B, S, n_kv]
        b_name = f"{basename}/in_proj_b/MatMul"
        self.make_matmul(linear_attn.in_proj_b, b_name, root_input)

        # a (decay/alpha): [B, S, hidden] -> [B, S, n_kv]
        a_name = f"{basename}/in_proj_a/MatMul"
        self.make_matmul(linear_attn.in_proj_a, a_name, root_input)

        # 3. Depthwise causal conv1d
        # Transpose QKV: [B, S, D] -> [B, D, S]
        qkv_t_name = f"{basename}/qkv_transpose/Transpose"
        qkv_t_output = f"{qkv_t_name}/output_0"
        self.make_transpose(
            qkv_t_name,
            f"{qkv_name}/output_0",
            self.io_dtype,
            ["batch_size", conv_dim, "sequence_length"],
            [0, 2, 1],
        )

        # Conv1d weight: [conv_dim, 1, kernel_size]
        conv_weight_name = f"model.layers.{layer_id}.linear_attn.conv1d.weight"
        self.make_initializer(linear_attn.conv1d.weight, conv_weight_name, to=self.io_dtype)

        # Past conv state: [B, D, kernel_size-1]
        past_conv = f"past_key_values.{layer_id}.conv_state"

        # Concatenate past conv state with current input: [B, D, K-1+S]
        conv_cat_name = f"{basename}/conv/Concat"
        conv_cat_output = f"{conv_cat_name}/output_0"
        self.make_concat(
            conv_cat_name,
            [past_conv, qkv_t_output],
            self.io_dtype,
            ["batch_size", conv_dim, "conv_total_length"],
            axis=-1,
        )

        # Depthwise Conv1d (groups=conv_dim)
        conv_name = f"{basename}/conv/Conv"
        conv_output = f"{conv_name}/output_0"
        self.make_node(
            "Conv",
            [conv_cat_output, conv_weight_name],
            [conv_output],
            name=conv_name,
            group=conv_dim,
            pads=[0, 0],
        )
        self.make_value(conv_output, self.io_dtype, ["batch_size", conv_dim, "sequence_length"])

        # SiLU activation on conv output
        silu_name = f"{basename}/conv/Sigmoid"
        self.make_sigmoid(silu_name, conv_output, self.io_dtype, ["batch_size", conv_dim, "sequence_length"])
        silu_sig_output = f"{silu_name}/output_0"

        silu_mul_name = f"{basename}/conv/SiLU/Mul"
        self.make_mul(silu_mul_name, [conv_output, silu_sig_output], self.io_dtype, ["batch_size", conv_dim, "sequence_length"])
        silu_output = f"{silu_mul_name}/output_0"

        # Save new conv state: last (kernel_size-1) timesteps of the padded input
        present_conv = f"present.{layer_id}.conv_state"
        conv_state_slice_name = f"{basename}/conv_state/Slice"
        self.make_node(
            "Slice",
            [
                conv_cat_output,
                f"/model/constants/INT64/[{-(kernel_size - 1)}]",
                f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]",
                "/model/constants/INT64/[-1]",
            ],
            [present_conv],
            name=conv_state_slice_name,
        )
        self.make_value(present_conv, self.io_dtype, ["batch_size", conv_dim, kernel_size - 1])

        # Transpose conv output back: [B, D, S] -> [B, S, D]
        conv_out_t_name = f"{basename}/conv_out/Transpose"
        conv_out_t_output = f"{conv_out_t_name}/output_0"
        self.make_transpose(
            conv_out_t_name,
            silu_output,
            self.io_dtype,
            ["batch_size", "sequence_length", conv_dim],
            [0, 2, 1],
        )

        # 4. Split into Q, K, V: [B, S, D] -> Q:[B,S,k_dim], K:[B,S,k_dim], V:[B,S,v_dim]
        split_qkv_name = f"{basename}/split_qkv/Split"
        q_out = f"{split_qkv_name}/output_0"
        k_out = f"{split_qkv_name}/output_1"
        v_out = f"{split_qkv_name}/output_2"
        self.make_node(
            "Split",
            [conv_out_t_output, f"/model/constants/INT64/[{k_dim}, {k_dim}, {v_dim}]"],
            [q_out, k_out, v_out],
            name=split_qkv_name,
            axis=-1,
        )
        self.make_value(q_out, self.io_dtype, ["batch_size", "sequence_length", k_dim])
        self.make_value(k_out, self.io_dtype, ["batch_size", "sequence_length", k_dim])
        self.make_value(v_out, self.io_dtype, ["batch_size", "sequence_length", v_dim])

        # 5. Per-head L2 normalize Q and K
        # Must reshape [B, S, N*hk] -> [B*S*N, hk], normalize, reshape back
        n_k = self.linear_num_key_heads
        q_norm_out = self._make_per_head_l2_normalize(f"{basename}/q_l2norm", q_out, n_k, hk)
        k_norm_out = self._make_per_head_l2_normalize(f"{basename}/k_l2norm", k_out, n_k, hk)

        # Scale Q by 1/sqrt(head_k_dim) — shared constant created once
        scale_name = self._get_shared_q_scale(hk)

        q_scaled_name = f"{basename}/q_scaled/Mul"
        self.make_mul(q_scaled_name, [q_norm_out, scale_name], self.io_dtype, ["batch_size", "sequence_length", k_dim])
        q_scaled_output = f"{q_scaled_name}/output_0"

        # 6. Compute decay (g) and forget (beta) gates
        # beta = sigmoid(b): [B, S, n_kv]
        beta_name = f"{basename}/beta/Sigmoid"
        self.make_sigmoid(beta_name, f"{b_name}/output_0", self.io_dtype, ["batch_size", "sequence_length", n_kv])
        beta_output = f"{beta_name}/output_0"

        # g = -exp(A_log) * softplus(a + dt_bias)
        # Pre-compute -exp(A_log) as an initializer (constant per layer)
        dt_bias_init = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(linear_attn.dt_bias, dt_bias_init, to=self.io_dtype)

        neg_exp_a_name = f"model.layers.{layer_id}.linear_attn.neg_exp_A"
        neg_exp_a = (-linear_attn.A_log.data.exp()).detach()
        self.make_initializer(neg_exp_a, neg_exp_a_name, to=self.io_dtype)

        # a + dt_bias
        a_plus_dt_name = f"{basename}/decay/Add"
        self.make_add(a_plus_dt_name, [f"{a_name}/output_0", dt_bias_init], self.io_dtype, ["batch_size", "sequence_length", n_kv])
        a_plus_dt_output = f"{a_plus_dt_name}/output_0"

        # softplus(a + dt_bias) = log(1 + exp(x))
        softplus_name = f"{basename}/decay/Softplus"
        softplus_output = f"{softplus_name}/output_0"
        self.make_node("Softplus", [a_plus_dt_output], [softplus_output], name=softplus_name)
        self.make_value(softplus_output, self.io_dtype, ["batch_size", "sequence_length", n_kv])

        # g = neg_exp_a * softplus
        g_name = f"{basename}/decay/Mul"
        self.make_mul(g_name, [neg_exp_a_name, softplus_output], self.io_dtype, ["batch_size", "sequence_length", n_kv])
        g_output = f"{g_name}/output_0"

        # 7. Reshape Q, K, V to per-head: [B, S, D] -> [B, S, N, H] -> [B, N, S, H]
        q_4d = self._reshape_to_bhsd(f"{basename}/q", q_scaled_output, self.linear_num_key_heads, hk)
        k_4d = self._reshape_to_bhsd(f"{basename}/k", k_norm_out, self.linear_num_key_heads, hk)
        v_4d = self._reshape_to_bhsd(f"{basename}/v", v_out, n_kv, hv)

        # If num_value_heads > num_key_heads, repeat Q and K to match (GQA-style).
        # HF does: query.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        kv_groups = n_kv // self.linear_num_key_heads
        if kv_groups > 1:
            q_4d = self._repeat_kv_heads(f"{basename}/q_repeat", q_4d, self.linear_num_key_heads, kv_groups, hk)
            k_4d = self._repeat_kv_heads(f"{basename}/k_repeat", k_4d, self.linear_num_key_heads, kv_groups, hk)

        # Transpose gates: [B, S, N] -> [B, N, S]
        beta_t_name = f"{basename}/beta/Transpose"
        beta_t_output = f"{beta_t_name}/output_0"
        self.make_transpose(beta_t_name, beta_output, self.io_dtype, ["batch_size", n_kv, "sequence_length"], [0, 2, 1])

        g_t_name = f"{basename}/g/Transpose"
        g_t_output = f"{g_t_name}/output_0"
        self.make_transpose(g_t_name, g_output, self.io_dtype, ["batch_size", n_kv, "sequence_length"], [0, 2, 1])

        # 8. Linear attention recurrence via Scan op
        #
        # The GatedDeltaNet recurrence for each head is:
        #   S_t = exp(g_t) * S_{t-1} + beta_t * outer(k_t, v_t)
        #   o_t = q_t @ S_t
        #
        # We use the ONNX Scan op to iterate over the sequence dimension.
        # First, merge batch and heads: [B, N, S, D] -> [B*N, S, D]
        past_recurrent = f"past_key_values.{layer_id}.recurrent_state"
        present_recurrent = f"present.{layer_id}.recurrent_state"

        la_output, la_state_output = self._make_scan_recurrence(
            basename,
            q_4d,
            k_4d,
            v_4d,
            past_recurrent,
            g_t_output,
            beta_t_output,
            n_kv,
            hk,
            hv,
            present_state_name=present_recurrent,
        )

        # 9. Transpose back: [B, N, S, H] -> [B, S, N*H]
        la_t_name = f"{basename}/la/Transpose"
        la_t_output = f"{la_t_name}/output_0"
        self.make_transpose(
            la_t_name,
            la_output,
            self.io_dtype,
            ["batch_size", "sequence_length", n_kv, hv],
            [0, 2, 1, 3],
        )

        la_flat_name = f"{basename}/la/Reshape"
        la_flat_output = f"{la_flat_name}/output_0"
        self.make_reshape(
            la_flat_name,
            [la_t_output, f"/model/constants/INT64/[0, 0, {v_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", v_dim],
        )

        # 10. Gated RMSNorm: norm(output) * silu(z)
        z_output = f"{z_name}/output_0"
        gated_norm_output = self._make_gated_rms_norm(
            f"{basename}/gated_norm",
            la_flat_output,
            z_output,
            linear_attn.norm,
            layer_id,
        )

        # 11. Output projection
        o_name = f"{basename}/out_proj/MatMul"
        self.make_matmul(linear_attn.out_proj, o_name, gated_norm_output)
        self.layernorm_attrs["skip_input"] = f"{o_name}/output_0"

    def _make_per_head_l2_normalize(self, basename, input_name, n_heads, head_dim):
        """Per-head L2 normalize: reshape [B, S, N*H] -> [B*S*N, H], norm, reshape back."""
        total_dim = n_heads * head_dim

        # Reshape to [B*S*N, H] for per-head normalization
        flat_name = f"{basename}/flat/Reshape"
        flat_out = f"{flat_name}/output_0"
        self.make_reshape(
            flat_name,
            [input_name, f"/model/constants/INT64/[-1, {head_dim}]"],
            self.io_dtype,
            ["flat_tokens", head_dim],
        )

        # L2 normalize along last dim (head_dim) — input is 2D [flat_tokens, head_dim]
        norm_out = self._make_l2_normalize(basename, flat_out, head_dim, leading_dims=["flat_tokens"])

        # Reshape back to [B, S, N*H] using input shape
        in_shape_name = f"{basename}/in_shape/Shape"
        self.make_shape(in_shape_name, input_name, [3])

        unflat_name = f"{basename}/unflat/Reshape"
        unflat_out = f"{unflat_name}/output_0"
        self.make_reshape(
            unflat_name,
            [norm_out, f"{in_shape_name}/output_0"],
            self.io_dtype,
            ["batch_size", "sequence_length", total_dim],
        )
        return unflat_out

    def _make_l2_normalize(self, basename, input_name, last_dim, leading_dims=None):
        """L2-normalize along last dimension: x / max(||x||_2, eps)

        Args:
            leading_dims: Shape prefix for intermediate values (e.g., ["flat_tokens"] for 2D,
                          ["batch_size", "sequence_length"] for 3D). Defaults to 3D if not specified.
        """
        if leading_dims is None:
            leading_dims = ["batch_size", "sequence_length"]
        full_shape = [*leading_dims, last_dim]
        reduced_shape = [*leading_dims, 1]

        # ||x||_2 along last dim
        l2_name = f"{basename}/ReduceL2"
        l2_output = f"{l2_name}/output_0"
        self.make_node(
            "ReduceL2",
            [input_name, "/model/constants/INT64/[-1]"],
            [l2_output],
            name=l2_name,
            keepdims=1,
        )
        self.make_value(l2_output, self.io_dtype, reduced_shape)

        # Clamp to avoid division by zero
        eps_name = self._get_shared_l2_eps()
        clip_name = f"{basename}/Clip"
        clip_output = f"{clip_name}/output_0"
        self.make_node("Clip", [l2_output, eps_name], [clip_output], name=clip_name)
        self.make_value(clip_output, self.io_dtype, reduced_shape)

        # x / ||x||_2
        norm_name = f"{basename}/Normalize/Div"
        self.make_div(norm_name, [input_name, clip_output], self.io_dtype, full_shape)
        norm_output = f"{norm_name}/output_0"

        return norm_output

    def _reshape_to_bhsd(self, basename, input_name, num_heads, head_dim):
        """Reshape [B, S, N*H] -> [B, S, N, H] -> Transpose to [B, N, S, H]"""
        rs_name = f"{basename}/reshape_4d/Reshape"
        rs_output = f"{rs_name}/output_0"
        self.make_reshape(
            rs_name,
            [input_name, f"/model/constants/INT64/[0, 0, {num_heads}, {head_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", num_heads, head_dim],
        )

        tp_name = f"{basename}/transpose_bhsd/Transpose"
        tp_output = f"{tp_name}/output_0"
        self.make_transpose(
            tp_name,
            rs_output,
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", head_dim],
            [0, 2, 1, 3],
        )
        return tp_output

    def _repeat_kv_heads(self, basename, input_4d, n_heads, n_rep, head_dim):
        """Repeat [B, N, S, H] -> [B, N*n_rep, S, H] (GQA-style head expansion).

        Equivalent to HF's repeat_interleave(n_rep, dim=1) on BNSH layout.
        Implemented as: [B, N, S, H] -> [B, N, 1, S, H] -> expand [B, N, n_rep, S, H] -> reshape [B, N*n_rep, S, H].
        """
        n_out = n_heads * n_rep

        # Unsqueeze: [B, N, S, H] -> [B, N, 1, S, H]
        unsq_name = f"{basename}/Unsqueeze"
        unsq_out = f"{unsq_name}/output_0"
        self.make_unsqueeze(
            unsq_name,
            [input_4d, "/model/constants/INT64/[2]"],
            self.io_dtype,
            ["batch_size", n_heads, 1, "sequence_length", head_dim],
        )

        # Expand: [B, N, 1, S, H] -> [B, N, n_rep, S, H]
        expand_name = f"{basename}/Expand"
        expand_out = f"{expand_name}/output_0"
        self.make_node(
            "Expand",
            [unsq_out, f"/model/constants/INT64/[1, 1, {n_rep}, 1, 1]"],
            [expand_out],
            name=expand_name,
        )
        self.make_value(expand_out, self.io_dtype, ["batch_size", n_heads, n_rep, "sequence_length", head_dim])

        # Reshape: [B, N, n_rep, S, H] -> [B, N*n_rep, S, H]
        # Use dynamic reshape to handle variable S
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, input_4d, [4])

        s_name = f"{basename}/S/Gather"
        s_out = f"{s_name}/output_0"
        self.make_gather(s_name, [f"{shape_name}/output_0", "/model/constants/INT64/[2]"], ir.DataType.INT64, [1], 0)

        target_name = f"{basename}/target_shape/Concat"
        target_out = f"{target_name}/output_0"
        self.make_concat(
            target_name,
            ["/model/constants/INT64/[-1]", f"/model/constants/INT64/[{n_out}]", s_out, f"/model/constants/INT64/[{head_dim}]"],
            ir.DataType.INT64,
            [4],
            axis=0,
        )

        rs_name = f"{basename}/Reshape"
        rs_out = f"{rs_name}/output_0"
        self.make_reshape(rs_name, [expand_out, target_out], self.io_dtype, ["batch_size", n_out, "sequence_length", head_dim])
        return rs_out

    def _make_scan_recurrence(
        self, basename, q_4d, k_4d, v_4d, past_state, g_output, beta_output, n_heads, hk, hv, present_state_name=None
    ):
        """Implement GatedDeltaNet recurrence using ONNX Scan op.

        Inputs (all in BNSH layout):
            q_4d:        [B, N, S, hk]   - queries (L2-normed and scaled)
            k_4d:        [B, N, S, hk]   - keys (L2-normed)
            v_4d:        [B, N, S, hv]   - values
            past_state:  [B, N, hk, hv]  - previous recurrent state
            g_output:    [B, N, S]        - decay (log-space, negative)
            beta_output: [B, N, S]        - update rate (sigmoid output)

        Returns:
            (output_name, state_name) where:
                output: [B, N, S, hv]
                state:  [B, N, hk, hv]
        """
        scan_basename = f"{basename}/scan"

        # Merge B and N: [B, N, S, D] -> [B*N, S, D]
        q_bn = self._merge_batch_heads(f"{scan_basename}/q", q_4d, n_heads, hk)
        k_bn = self._merge_batch_heads(f"{scan_basename}/k", k_4d, n_heads, hk)
        v_bn = self._merge_batch_heads(f"{scan_basename}/v", v_4d, n_heads, hv)

        # Merge state: [B, N, hk, hv] -> [B*N, hk, hv]
        # hk and hv are constants, so [-1, hk, hv] correctly infers B*N
        state_rs_name = f"{scan_basename}/state_merge/Reshape"
        state_rs_output = f"{state_rs_name}/output_0"
        self.make_reshape(
            state_rs_name,
            [past_state, f"/model/constants/INT64/[-1, {hk}, {hv}]"],
            self.io_dtype,
            ["batch_heads", hk, hv],
        )

        # Merge gates: [B, N, S] -> [B*N, S]
        g_bn_output = self._merge_batch_heads_2d(f"{scan_basename}/g", g_output, n_heads)
        beta_bn_output = self._merge_batch_heads_2d(f"{scan_basename}/beta", beta_output, n_heads)

        # Build the Scan body subgraph (names prefixed by layer for uniqueness)
        layer_id = int(basename.split("layers.")[1].split("/")[0])
        body = self._build_scan_body(layer_id, hk, hv)

        # Create Scan node
        # Scan carries: state [B*N, hk, hv]
        # Scan inputs: q [B*N, S, hk], k [B*N, S, hk], v [B*N, S, hv],
        #              g [B*N, S], beta [B*N, S]
        # Scan outputs: output [B*N, S, hv]
        scan_name = f"{scan_basename}/Scan"
        scan_output = f"{scan_name}/output_0"  # carry out: state
        scan_output_seq = f"{scan_name}/output_1"  # scan out: per-step outputs

        scan_inputs = [
            self.make_value(state_rs_output),  # carry input
            self.make_value(q_bn),  # scan input 0
            self.make_value(k_bn),  # scan input 1
            self.make_value(v_bn),  # scan input 2
            self.make_value(g_bn_output),  # scan input 3
            self.make_value(beta_bn_output),  # scan input 4
        ]
        scan_out_state = ir.Value(name=scan_output)
        scan_out_state.dtype = ir.DataType(self.io_dtype)
        scan_out_state.shape = ir.Shape(["batch_heads", hk, hv])
        self.values[scan_output] = scan_out_state

        scan_out_seq = ir.Value(name=scan_output_seq)
        scan_out_seq.dtype = ir.DataType(self.io_dtype)
        scan_out_seq.shape = ir.Shape(["batch_heads", "sequence_length", hv])
        self.values[scan_output_seq] = scan_out_seq

        scan_node = ir.node(
            "Scan",
            inputs=scan_inputs,
            outputs=[scan_out_state, scan_out_seq],
            attributes={
                "body": ir.AttrGraph("body", body),
                "num_scan_inputs": ir.AttrInt64("num_scan_inputs", 5),
                "scan_input_axes": ir.AttrInt64s("scan_input_axes", [1, 1, 1, 1, 1]),
                "scan_output_axes": ir.AttrInt64s("scan_output_axes", [1]),
            },
            name=scan_name,
        )
        self.model.graph.append(scan_node)
        self.node_names.add(scan_name)

        # Unmerge batch and heads for output: [B*N, S, hv] -> [B, N, S, hv]
        # Need dynamic reshape since S is variable
        out_shape_name = f"{scan_basename}/out_unmerge/Shape"
        self.make_shape(out_shape_name, scan_output_seq, [3])

        out_s_name = f"{scan_basename}/out_unmerge/S/Gather"
        out_s_out = f"{out_s_name}/output_0"
        self.make_gather(
            out_s_name, [f"{out_shape_name}/output_0", "/model/constants/INT64/1"], ir.DataType.INT64, [], 0
        )

        out_s_unsq_name = f"{scan_basename}/out_unmerge/S/Unsqueeze"
        out_s_unsq_out = f"{out_s_unsq_name}/output_0"
        self.make_unsqueeze(out_s_unsq_name, [out_s_out, "/model/constants/INT64/[0]"], ir.DataType.INT64, [1])

        out_target_name = f"{scan_basename}/out_unmerge/target_shape/Concat"
        out_target_out = f"{out_target_name}/output_0"
        self.make_concat(
            out_target_name,
            [
                "/model/constants/INT64/[-1]",
                f"/model/constants/INT64/[{n_heads}]",
                out_s_unsq_out,
                f"/model/constants/INT64/[{hv}]",
            ],
            ir.DataType.INT64,
            [4],
            axis=0,
        )

        out_rs_name = f"{scan_basename}/out_unmerge/Reshape"
        out_rs_output = f"{out_rs_name}/output_0"
        self.make_reshape(
            out_rs_name,
            [scan_output_seq, out_target_out],
            self.io_dtype,
            ["batch_size", n_heads, "sequence_length", hv],
        )

        # Unmerge state: [B*N, hk, hv] -> [B, N, hk, hv]
        state_out_name = present_state_name if present_state_name else f"{scan_basename}/state_out_unmerge/Reshape/output_0"
        state_out_rs_name = f"{scan_basename}/state_out_unmerge/Reshape"
        self.make_node(
            "Reshape",
            [scan_output, f"/model/constants/INT64/[-1, {n_heads}, {hk}, {hv}]"],
            [state_out_name],
            name=state_out_rs_name,
        )
        self.make_value(state_out_name, self.io_dtype, ["batch_size", n_heads, hk, hv])

        return out_rs_output, state_out_name

    def _merge_batch_heads(self, basename, input_4d, n_heads, head_dim):
        """[B, N, S, D] -> [B*N, S, D] via static reshape.

        Since N and D are compile-time constants, we reshape with
        ``[-1, S, D]`` which lets ONNX infer ``B*N`` from the total
        element count.  This replaces a 7-op dynamic Shape/Gather/Mul
        chain with a single Reshape.
        """
        rs_name = f"{basename}/merge_bn/Reshape"
        rs_output = f"{rs_name}/output_0"
        # Reshape [B, N, S, D] → [B*N, S, D]:  N and D are constants,
        # so [0, 0, -1, D] preserves B and N while merging them is
        # impossible with two unknowns.  Instead, transpose to
        # [B*N, S, D] in one step by using the 3-D target shape.
        # Because the total number of elements is fixed, ONNX can
        # infer B*N from [-1, (original S), D].  We use a Shape op
        # to extract S dynamically.
        shape_name = f"{basename}/merge_bn/Shape"
        self.make_shape(shape_name, input_4d, [4])

        s_name = f"{basename}/merge_bn/S/Gather"
        s_out = f"{s_name}/output_0"
        self.make_gather(s_name, [f"{shape_name}/output_0", "/model/constants/INT64/[2]"], ir.DataType.INT64, [1], 0)

        target_name = f"{basename}/merge_bn/target_shape/Concat"
        target_out = f"{target_name}/output_0"
        self.make_concat(
            target_name,
            ["/model/constants/INT64/[-1]", s_out, f"/model/constants/INT64/[{head_dim}]"],
            ir.DataType.INT64,
            [3],
            axis=0,
        )

        self.make_reshape(rs_name, [input_4d, target_out], self.io_dtype, ["batch_heads", "sequence_length", head_dim])
        return rs_output

    def _merge_batch_heads_2d(self, basename, input_3d, n_heads):
        """[B, N, S] -> [B*N, S] via static reshape."""
        shape_name = f"{basename}/merge_bn/Shape"
        self.make_shape(shape_name, input_3d, [3])

        s_name = f"{basename}/merge_bn/S/Gather"
        s_out = f"{s_name}/output_0"
        self.make_gather(s_name, [f"{shape_name}/output_0", "/model/constants/INT64/[2]"], ir.DataType.INT64, [1], 0)

        target_name = f"{basename}/merge_bn/target_shape/Concat"
        target_out = f"{target_name}/output_0"
        self.make_concat(
            target_name,
            ["/model/constants/INT64/[-1]", s_out],
            ir.DataType.INT64,
            [2],
            axis=0,
        )

        rs_name = f"{basename}/merge_bn/Reshape"
        rs_output = f"{rs_name}/output_0"
        self.make_reshape(rs_name, [input_3d, target_out], self.io_dtype, ["batch_heads", "sequence_length"])
        return rs_output

    def _build_scan_body(self, layer_id, hk, hv):
        """Build the Scan body graph for one timestep of GatedDeltaNet.

        Names are prefixed with the layer index so that each Scan body
        has globally unique value and node names (required by Olive's
        OnnxDAG which processes all subgraphs together).

        Carry input:  state [B*N, hk, hv]
        Scan inputs:  q_t [B*N, hk], k_t [B*N, hk], v_t [B*N, hv],
                      g_t [B*N], beta_t [B*N]
        Carry output: new_state [B*N, hk, hv]
        Scan output:  out_t [B*N, hv]
        """
        p = f"L{layer_id}"
        body = ir.Graph(inputs=(), outputs=(), nodes=(), name=f"gated_deltanet_body_{layer_id}")
        io_dt = ir.DataType(self.io_dtype)

        def body_val(name, shape):
            v = ir.Value(name=f"{p}/{name}")
            v.dtype = io_dt
            v.shape = ir.Shape(shape)
            return v

        axes_neg1 = self._make_body_const(body, f"{p}/axes_neg1", [-1])
        axes_neg2 = self._make_body_const(body, f"{p}/axes_neg2", [-2])
        axes_1 = self._make_body_const(body, f"{p}/axes_1", [1])

        # Define body inputs
        state_in = body_val("state_in", ["batch_heads", hk, hv])
        q_t = body_val("q_t", ["batch_heads", hk])
        k_t = body_val("k_t", ["batch_heads", hk])
        v_t = body_val("v_t", ["batch_heads", hv])
        g_t = body_val("g_t", ["batch_heads"])
        beta_t = body_val("beta_t", ["batch_heads"])

        body.inputs.extend([state_in, q_t, k_t, v_t, g_t, beta_t])

        # 1. decay = exp(unsqueeze(g_t, -1)): [B*N] -> [B*N, 1] -> broadcast
        g_unsq = body_val("g_unsq", ["batch_heads", 1])
        body.append(ir.node("Unsqueeze", inputs=[g_t, axes_neg1], outputs=[g_unsq], name=f"{p}/g_unsq"))

        decay = body_val("decay", ["batch_heads", 1])
        body.append(ir.node("Exp", inputs=[g_unsq], outputs=[decay], name=f"{p}/Exp"))

        # 2. decayed_state = state * decay: [B*N, hk, hv] * [B*N, 1, 1]
        decay_2d = body_val("decay_2d", ["batch_heads", 1, 1])
        body.append(ir.node("Unsqueeze", inputs=[decay, axes_neg1], outputs=[decay_2d], name=f"{p}/decay_2d"))

        decayed_state = body_val("decayed_state", ["batch_heads", hk, hv])
        body.append(ir.node("Mul", inputs=[state_in, decay_2d], outputs=[decayed_state], name=f"{p}/decay_mul"))

        # 3. k_state = k_t @ decayed_state
        k_unsq = body_val("k_unsq", ["batch_heads", 1, hk])
        body.append(ir.node("Unsqueeze", inputs=[k_t, axes_neg2], outputs=[k_unsq], name=f"{p}/k_unsq"))

        k_state_3d = body_val("k_state_3d", ["batch_heads", 1, hv])
        body.append(ir.node("MatMul", inputs=[k_unsq, decayed_state], outputs=[k_state_3d], name=f"{p}/k_matmul"))

        k_state = body_val("k_state", ["batch_heads", hv])
        body.append(ir.node("Squeeze", inputs=[k_state_3d, axes_neg2], outputs=[k_state], name=f"{p}/k_squeeze"))

        # 4. delta_v = v_t - k_state
        delta_v = body_val("delta_v", ["batch_heads", hv])
        body.append(ir.node("Sub", inputs=[v_t, k_state], outputs=[delta_v], name=f"{p}/delta_v"))

        # 5. update = delta_v * beta_t
        beta_unsq = body_val("beta_unsq", ["batch_heads", 1])
        body.append(ir.node("Unsqueeze", inputs=[beta_t, axes_neg1], outputs=[beta_unsq], name=f"{p}/beta_unsq"))

        update = body_val("update", ["batch_heads", hv])
        body.append(ir.node("Mul", inputs=[delta_v, beta_unsq], outputs=[update], name=f"{p}/update"))

        # 6. new_state = decayed_state + outer(k_t, update)
        k_col = body_val("k_col", ["batch_heads", hk, 1])
        body.append(ir.node("Unsqueeze", inputs=[k_t, axes_neg1], outputs=[k_col], name=f"{p}/k_col"))

        update_row = body_val("update_row", ["batch_heads", 1, hv])
        body.append(ir.node("Unsqueeze", inputs=[update, axes_neg2], outputs=[update_row], name=f"{p}/update_row"))

        kv_update = body_val("kv_update", ["batch_heads", hk, hv])
        body.append(ir.node("MatMul", inputs=[k_col, update_row], outputs=[kv_update], name=f"{p}/kv_matmul"))

        new_state = body_val("new_state", ["batch_heads", hk, hv])
        body.append(ir.node("Add", inputs=[decayed_state, kv_update], outputs=[new_state], name=f"{p}/state_add"))

        # 7. out_t = q_t @ new_state
        q_unsq = body_val("q_unsq", ["batch_heads", 1, hk])
        body.append(ir.node("Unsqueeze", inputs=[q_t, axes_neg2], outputs=[q_unsq], name=f"{p}/q_unsq"))

        out_3d = body_val("out_3d", ["batch_heads", 1, hv])
        body.append(ir.node("MatMul", inputs=[q_unsq, new_state], outputs=[out_3d], name=f"{p}/matmul"))

        out_t = body_val("out_t", ["batch_heads", hv])
        body.append(ir.node("Squeeze", inputs=[out_3d, axes_1], outputs=[out_t], name=f"{p}/squeeze"))

        body.outputs.extend([new_state, out_t])
        return body

    @staticmethod
    def _make_body_const(body, name, values):
        """Create a constant value inside a Scan body graph."""
        val = ir.Value(name=name)
        val.dtype = ir.DataType.INT64
        val.shape = ir.Shape([len(values)])
        tensor = ir.tensor(torch.tensor(values, dtype=torch.int64), name=name)
        body.append(
            ir.node(
                "Constant",
                inputs=[],
                outputs=[val],
                name=f"{name}/Constant",
                attributes={"value": tensor},
            )
        )
        return val

    def _make_gated_rms_norm(self, basename, input_name, gate_name, norm_module, layer_id):
        """Gated RMSNorm: RMSNorm(x) * SiLU(z).

        The norm weight is per-head (shape [head_v_dim]).
        Input and gate are [B, S, v_dim]. We reshape to per-head,
        apply per-head norm, gate, and reshape back.
        """
        v_dim = self.linear_value_dim
        hv = self.linear_value_head_dim

        # Reshape input to [B*S*N, H] for per-head norm
        flat_name = f"{basename}/input_flat/Reshape"
        flat_output = f"{flat_name}/output_0"
        self.make_reshape(
            flat_name,
            [input_name, f"/model/constants/INT64/[-1, {hv}]"],
            self.io_dtype,
            ["batch_seq_heads", hv],
        )

        # Norm weight (NO offset — Qwen3_5RMSNormGated uses raw weight, not 1+w)
        norm_weight = f"model.layers.{layer_id}.linear_attn.norm.weight"
        self.make_initializer(norm_module.weight, norm_weight, to=self.io_dtype)

        # SimplifiedLayerNormalization (com.microsoft, no offset for gated norm)
        norm_name = f"{basename}/SimplifiedLayerNormalization"
        norm_output = f"{norm_name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            [flat_output, norm_weight],
            [norm_output],
            name=norm_name,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(norm_output, self.io_dtype, ["batch_seq_heads", hv])

        # Reshape back to [B, S, v_dim] using input shape
        # Get B and S from input_name shape [B, S, v_dim]
        in_shape_name = f"{basename}/in_shape/Shape"
        self.make_shape(in_shape_name, input_name, [3])

        in_bs_name = f"{basename}/in_shape/BS/Slice"
        in_bs_out = f"{in_bs_name}/output_0"
        self.make_slice(
            in_bs_name,
            [
                f"{in_shape_name}/output_0",
                "/model/constants/INT64/[0]",
                "/model/constants/INT64/[2]",
                "/model/constants/INT64/[0]",
            ],
            ir.DataType.INT64,
            [2],
        )

        unflat_target_name = f"{basename}/norm_unflat/target_shape/Concat"
        unflat_target_out = f"{unflat_target_name}/output_0"
        self.make_concat(
            unflat_target_name,
            [in_bs_out, f"/model/constants/INT64/[{v_dim}]"],
            ir.DataType.INT64,
            [3],
            axis=0,
        )

        unflat_name = f"{basename}/norm_unflat/Reshape"
        unflat_output = f"{unflat_name}/output_0"
        self.make_reshape(
            unflat_name,
            [norm_output, unflat_target_out],
            self.io_dtype,
            ["batch_size", "sequence_length", v_dim],
        )

        # SiLU(z)
        z_sigmoid_name = f"{basename}/z_sigmoid/Sigmoid"
        self.make_sigmoid(z_sigmoid_name, gate_name, self.io_dtype, ["batch_size", "sequence_length", v_dim])
        z_sigmoid_output = f"{z_sigmoid_name}/output_0"

        z_silu_name = f"{basename}/z_silu/Mul"
        self.make_mul(z_silu_name, [gate_name, z_sigmoid_output], self.io_dtype, ["batch_size", "sequence_length", v_dim])
        z_silu_output = f"{z_silu_name}/output_0"

        # output = norm * silu(z)
        gated_name = f"{basename}/gated/Mul"
        self.make_mul(gated_name, [unflat_output, z_silu_output], self.io_dtype, ["batch_size", "sequence_length", v_dim])
        gated_output = f"{gated_name}/output_0"

        return gated_output

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """Generate genai_config.json for the decoder (text-only) model.

        Temporarily adjusts attributes so the base class produces the correct
        config for Qwen3.5's hybrid architecture (sparse KV cache, nested
        token IDs in ``text_config``).
        """
        # Flatten text_config token IDs onto the HF config so the base class
        # can access them.  Save to out_dir so AutoConfig.from_pretrained
        # picks up the patched version.
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs
        )
        text_cfg = getattr(hf_config, "text_config", hf_config)
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
            val = getattr(text_cfg, attr, None)
            if val is not None:
                setattr(hf_config, attr, val)
        hf_config.save_pretrained(out_dir)

        # Temporarily restore the KV cache template keys and adjust attributes
        # so the base class generates the right entries.
        saved = {
            "num_layers": self.num_layers,
            "model_type": self.model_type,
            "past_present_share_buffer": self.past_present_share_buffer,
        }
        self.num_layers = len(self.layer_types)
        self.model_type = "Qwen3_5ForConditionalGeneration"
        self.past_present_share_buffer = False
        self.input_names["past_key_values.key"] = "past_key_values.%d.key"
        self.input_names["past_key_values.value"] = "past_key_values.%d.value"
        self.output_names["present.key"] = "present.%d.key"
        self.output_names["present.value"] = "present.%d.value"

        super().make_genai_config(out_dir, {}, out_dir)

        # Restore
        self.num_layers = saved["num_layers"]
        self.model_type = saved["model_type"]
        self.past_present_share_buffer = saved["past_present_share_buffer"]
        del self.input_names["past_key_values.key"]
        del self.input_names["past_key_values.value"]
        del self.output_names["present.key"]
        del self.output_names["present.value"]
