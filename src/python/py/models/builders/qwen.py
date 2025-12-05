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
            assert config.rope_scaling["type"] in ["mrope", "default"]

        # Qwen 2.5 VL applies RoPE manually before attention, not fused in the op
        self.attention_attrs["use_rope_in_attn"] = False

        # We need separate Q, K, V tensors to apply MRoPE manually.
        # Packed MatMul provides a single output which would require splitting.
        self.attention_attrs["use_packed_matmul"] = False

        if "position_ids" not in self.input_names:
            print("Re-adding 'position_ids' to self.input_names.")
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
        force_fp32 = self.attention_attrs.get("rope_cast", {}).get("use_fp32", False)
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
        # Load the Hugging Face model
        print("Loading Qwen2_5_VLForConditionalGeneration model...")
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
        )
