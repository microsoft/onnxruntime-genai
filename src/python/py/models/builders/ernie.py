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
        self.rope_attrs["interleaved"] = 1

        # Ernie4_5Config stores rope_theta inside a `rope_parameters` dict rather
        # than as a top-level attribute.  The base Model.__init__ falls back to
        # 10000 when `config.rope_theta` is absent, so we patch the value here.
        if not hasattr(config, "rope_theta") and hasattr(config, "rope_parameters"):
            rope_theta = config.rope_parameters.get("rope_theta")
            if rope_theta is not None:
                self.rope_attrs["theta"] = rope_theta

        # Ernie uses a `compression_ratio` for its RoPE scaling.
        # The original RoPE logic in ernie is: position_ids / compression_ratio,
        # which is equivalent to scaling the frequencies (inv_freq) by 1 / compression_ratio.
        if hasattr(config, "compression_ratio") and config.compression_ratio != 1.0:
            self.rope_attrs["rescale_factors"] = 1.0 / config.compression_ratio
