# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnx_ir as ir


class WebGPU:
    """
    WebGPU specific subgraph expansions
    """
    def make_attention_mask_graph_capture_reformatting_for_gqa(self):
        # Make nodes for the attention mask subgraph that calculates
        # attributes about the 2D attention mask to use in GroupQueryAttention
        #
        # Key difference vs make_attention_mask_standard_reformatting_for_gqa:
        # - Standard mode: total_seq_len is calculated from Shape op (always runs on CPU)
        # - Graph capture mode: No Shape ops inserted to ensure all ops run on GPU (no CPU ops)
        #
        #          attention_mask
        #               |
        #         Cast to int32
        #               |
        #           ReduceSum (keepdims=0)
        #              /    \
        #             /      \
        #           Sub    ReduceMax
        #            |        |
        #       seqlens_k  total_seq_len
        #         (1D)       (int)
        basename = "/model/attn_mask_reformat"
        attn_mask_basename = f"{basename}/attn_mask_subgraph"

        # Calculate ReduceSum from attention_mask
        cast_1_name = f"{attn_mask_basename}/Cast"
        self.make_cast(
            cast_1_name, self.input_names["attention_mask"], dtype=ir.DataType.INT32, shape=["batch_size", "total_sequence_length"]
        )
        reduce_sum_name = f"{attn_mask_basename}/ReduceSum"
        reduce_sum_inputs = [f"{cast_1_name}/output_0", "/model/constants/INT64/[1]"]
        self.make_reduce_sum(reduce_sum_name, reduce_sum_inputs, dtype=ir.DataType.INT32, shape=["batch_size"])

        # Left branch: Calculate seqlens_k = ReduceSum - 1
        sub_name = f"{attn_mask_basename}/Sub"
        sub_inputs = [f"{reduce_sum_name}/output_0", "/model/constants/INT32/[1]"]
        self.make_sub(sub_name, sub_inputs, dtype=ir.DataType.INT32, shape=["batch_size"])

        # Right branch: ReduceMax to get maximum int value for total_seq_len
        reduce_max_name = f"{attn_mask_basename}/ReduceMax"
        reduce_max_inputs = [f"{reduce_sum_name}/output_0"]
        self.make_reduce_max(reduce_max_name, reduce_max_inputs, dtype=ir.DataType.INT32, shape=[])

        self.mask_attrs["seqlens_k"] = sub_name
        self.mask_attrs["total_seq_len"] = reduce_max_name

    def make_synthetic_position_ids_graph_capture(self):
        # Shape-free construction of the [B, S] INT64 tensor of linear indices
        # r*S + c, replacing Qwen35TextModel._make_synthetic_position_ids for
        # WebGPU graph capture.
        #
        # Standard version emits two Shape ops (Shape -> Slice for [B, S], and
        # Shape -> Gather for scalar B*S) whose INT64 outputs land on CPU and
        # force Memcpy nodes that break graph capture.
        #
        # Graph-capture version derives everything from the existing 3D
        # position_ids input using only GPU-native ops (Gather, Squeeze, Mul,
        # Add, CumSum, ReduceSum). All intermediates keep symbolic [B, S]
        # shape from data-flow, so no scalar length is ever materialized.
        #
        # Layout:
        #   position_ids [3, B, S]
        #        |
        #     Gather(idx=[0], axis=0)  -> [1, B, S]
        #        |
        #     Squeeze(axis=[0])        -> [B, S]                (t_positions)
        #        |
        #     Mul(*0) then Add(+1)     -> [B, S] all ones       (ones_bs)
        #        |----------------\
        #        |                 \
        #     CumSum(axis=1,        ReduceSum(axes=[1],
        #            exclusive=1)          keepdims=1)          -> [B, 1] = S per row
        #        |                        |
        #     row_idx [B, S]         CumSum(axis=0,
        #     (values 0..S-1)               exclusive=1)         -> [B, 1] = 0, S, 2S, ...
        #        |                        |
        #        +------- Add ------------+                     -> [B, S] = r*S + c
        basename = "/model/attn/synthetic_pos_ids"
        pos_ids_input = self.position_ids_reformatted

        # Gather axis-0 slice [0] -> [1, B, S]
        gather_name = f"{basename}/t/Gather"
        self.make_gather(
            gather_name,
            inputs=[pos_ids_input, "/model/constants/INT64/[0]"],
            dtype=ir.DataType.INT64,
            shape=[1, "batch_size", "sequence_length"],
            axis=0,
        )

        # Squeeze axis 0 -> [B, S]
        squeeze_name = f"{basename}/t/Squeeze"
        self.make_squeeze(
            squeeze_name,
            inputs=[f"{gather_name}/output_0", "/model/constants/INT64/[0]"],
            dtype=ir.DataType.INT64,
            shape=["batch_size", "sequence_length"],
        )

        # Zero out and add 1 to get [B, S] of ones. Two-step (Mul 0, Add 1)
        # avoids needing broadcasted constants of the right shape.
        zero_name = f"{basename}/zeros/Mul"
        self.make_mul(
            zero_name,
            inputs=[f"{squeeze_name}/output_0", "/model/constants/INT64/0"],
            dtype=ir.DataType.INT64,
            shape=["batch_size", "sequence_length"],
        )
        ones_name = f"{basename}/ones/Add"
        self.make_add(
            ones_name,
            inputs=[f"{zero_name}/output_0", "/model/constants/INT64/1"],
            dtype=ir.DataType.INT64,
            shape=["batch_size", "sequence_length"],
        )
        ones_bs = f"{ones_name}/output_0"

        # Per-row indices 0..S-1 via CumSum(axis=1, exclusive=1)
        row_idx_name = f"{basename}/row_idx/CumSum"
        self.make_cum_sum(
            row_idx_name,
            inputs=[ones_bs, "/model/constants/INT64/1"],
            dtype=ir.DataType.INT64,
            shape=["batch_size", "sequence_length"],
            exclusive=1,
        )

        # Per-row totals [B, 1] via ReduceSum(axes=[1], keepdims=1)
        row_sum_name = f"{basename}/row_sum/ReduceSum"
        self.make_reduce_sum(
            row_sum_name,
            inputs=[ones_bs, "/model/constants/INT64/[1]"],
            dtype=ir.DataType.INT64,
            shape=["batch_size", 1],
            keepdims=True,
        )

        # Batch offsets 0, S, 2S, ... via CumSum(axis=0, exclusive=1) on [B, 1]
        batch_off_name = f"{basename}/batch_off/CumSum"
        self.make_cum_sum(
            batch_off_name,
            inputs=[f"{row_sum_name}/output_0", "/model/constants/INT64/0"],
            dtype=ir.DataType.INT64,
            shape=["batch_size", 1],
            exclusive=1,
        )

        # linear = row_idx + batch_offset (broadcast [B, 1] over [B, S])
        linear_name = f"{basename}/linear/Add"
        self.make_add(
            linear_name,
            inputs=[f"{row_idx_name}/output_0", f"{batch_off_name}/output_0"],
            dtype=ir.DataType.INT64,
            shape=["batch_size", "sequence_length"],
        )

        return f"{linear_name}/output_0"
