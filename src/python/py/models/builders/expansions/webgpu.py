# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import onnx_ir as ir


class WebGPU:
    def make_attention_mask_reformatting_for_gqa(self):
        # Graph capture derives both lengths on the GPU without inserting Shape operations.
        basename = "/model/attn_mask_reformat"
        attn_mask_basename = f"{basename}/attn_mask_subgraph"

        cast_1_name = f"{attn_mask_basename}/Cast"
        self.make_cast(
            cast_1_name,
            self.input_names["attention_mask"],
            dtype=ir.DataType.INT32,
            shape=["batch_size", "total_sequence_length"],
        )
        reduce_sum_name = f"{attn_mask_basename}/ReduceSum"
        reduce_sum_inputs = [f"{cast_1_name}/output_0", "/model/constants/INT64/[1]"]
        self.make_reduce_sum(reduce_sum_name, reduce_sum_inputs, dtype=ir.DataType.INT32, shape=["batch_size"])

        sub_name = f"{attn_mask_basename}/Sub"
        sub_inputs = [f"{reduce_sum_name}/output_0", "/model/constants/INT32/[1]"]
        self.make_sub(sub_name, sub_inputs, dtype=ir.DataType.INT32, shape=["batch_size"])

        reduce_max_name = f"{attn_mask_basename}/ReduceMax"
        reduce_max_inputs = [f"{reduce_sum_name}/output_0"]
        self.make_reduce_max(reduce_max_name, reduce_max_inputs, dtype=ir.DataType.INT32, shape=[])

        self.mask_attrs["seqlens_k"] = sub_name
        self.mask_attrs["total_seq_len"] = reduce_max_name
