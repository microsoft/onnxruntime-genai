# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from .standard import Standard


class DML:
    make_gated_add = Standard.make_gated_add
    make_simplified_layer_norm = Standard.make_simplified_layer_norm
    make_branchwise_rms_norm = Standard.make_branchwise_rms_norm
    make_scaled_silu = Standard.make_scaled_silu
    make_hyper_connection_pre_mix = Standard.make_hyper_connection_pre_mix
    make_hyper_connection_post_mix = Standard.make_hyper_connection_post_mix
