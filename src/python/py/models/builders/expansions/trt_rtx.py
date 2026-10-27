# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from .standard import Standard


class TRT_RTX:  # noqa: N801
    make_gated_add = Standard.make_gated_add
    make_linear_attention_gate = Standard.make_linear_attention_gate
    make_gated_rms_norm = Standard.make_gated_rms_norm
    make_mrotary_embedding = Standard.make_mrotary_embedding
    make_layernorm_subgraph = Standard.make_layernorm_subgraph
    make_skip_simplified_layer_norm = Standard.make_skip_simplified_layer_norm
    make_skip_layer_norm = Standard.make_skip_layer_norm
    make_simplified_layer_norm = Standard.make_simplified_layer_norm
    make_mrope_positions = Standard.make_mrope_positions
    make_mrope_cache = Standard.make_mrope_cache
    make_mrope_rotation = Standard.make_mrope_rotation
    make_mrope_output = Standard.make_mrope_output
    make_branchwise_rms_norm = Standard.make_branchwise_rms_norm
    make_scaled_silu = Standard.make_scaled_silu
    make_hyper_connection_pre_mix = Standard.make_hyper_connection_pre_mix
    make_hyper_connection_post_mix = Standard.make_hyper_connection_post_mix
