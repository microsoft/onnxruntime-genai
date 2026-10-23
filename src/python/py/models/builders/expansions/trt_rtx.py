# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from .standard import Standard


class TRT_RTX:  # noqa: N801
    make_gated_add = Standard.make_gated_add
    make_linear_attention_gate = Standard.make_linear_attention_gate
    make_gated_rms_norm = Standard.make_gated_rms_norm
    make_mrotary_embedding = Standard.make_mrotary_embedding
    make_layernorm_subgraph = Standard.make_layernorm_subgraph
