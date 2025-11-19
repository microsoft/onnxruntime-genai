# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .mistral import MistralModel


class ErnieModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Ernie uses interleaved rotary position embeddings.
        self.rotemb_attrs["interleaved"] = 1

        # Ernie uses a `compression_ratio` for its RoPE scaling.
        # The original RoPE logic in ernie is: position_ids / compression_ratio,
        # which is equivalent to scaling the frequencies (inv_freq) by 1 / compression_ratio.
        if hasattr(config, "compression_ratio") and config.compression_ratio != 1.0:
            self.rotemb_attrs["rescale_factors"] = 1.0 / config.compression_ratio
