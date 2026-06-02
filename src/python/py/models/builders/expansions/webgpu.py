# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnx_ir as ir
import torch


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
